# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# MLA async_copy layouts require D=512 for legal lowering on gfx950 (head_dim=128
# kernels pad KV slot-linear caches to 512 and mask in compute).
_HEAD_DIM_ASYNC_LOAD: int = 512


def _kv_to_slot_linear_pad512(k_cache, v_cache):
    """Convert split-K / blocked-V paged caches to slot-major [slots, heads, 512]."""
    block_size = k_cache.shape[3]
    num_kv_heads = k_cache.shape[1]
    head_dim = k_cache.shape[2] * k_cache.shape[4]
    k_lin = (
        k_cache.permute(0, 3, 1, 2, 4)
        .reshape(k_cache.shape[0] * block_size, num_kv_heads, head_dim)
        .contiguous()
    )
    v_lin = (
        v_cache.permute(0, 3, 1, 2)
        .reshape(v_cache.shape[0] * block_size, num_kv_heads, head_dim)
        .contiguous()
    )
    if head_dim < _HEAD_DIM_ASYNC_LOAD:
        k_pad = torch.zeros(
            k_lin.shape[0],
            num_kv_heads,
            _HEAD_DIM_ASYNC_LOAD,
            dtype=k_lin.dtype,
            device=k_lin.device,
        )
        v_pad = torch.zeros_like(k_pad)
        k_pad[:, :, :head_dim] = k_lin
        v_pad[:, :, :head_dim] = v_lin
        return k_pad, v_pad
    return k_lin, v_lin


@gluon.jit
def _pa_prefill_async_load_kv_slices(
    bufs_k,
    bufs_v,
    buf_page,
    buf_idx,
    tile_start,
    K_gluon,
    V_gluon,
    cur_kv_head,
    cur_batch_ctx_len,
    block_size,
    stride_slot,
    stride_h,
    blocked_kv_slice: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_N_HALF: gl.constexpr,
    HEAD_DIM_LOAD: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
):
    bufs_k0 = bufs_k.index(buf_idx).slice(0, BLOCK_N_HALF, 1)
    bufs_k1 = bufs_k.index(buf_idx).slice(BLOCK_N_HALF, BLOCK_N_HALF, 1)
    bufs_v0 = bufs_v.index(buf_idx).slice(0, BLOCK_N_HALF, 1)
    bufs_v1 = bufs_v.index(buf_idx).slice(BLOCK_N_HALF, BLOCK_N_HALF, 1)
    buf_page_0 = buf_page.slice(0, BLOCK_N_HALF, 0)
    buf_page_1 = buf_page.slice(BLOCK_N_HALF, BLOCK_N_HALF, 0)

    page_bn_0 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        buf_page_0, gl.SliceLayout(0, blocked_kv_slice)
    )
    slot_0 = page_bn_0 * block_size + (
        tile_start + gl.arange(0, BLOCK_N_HALF, layout=gl.SliceLayout(0, blocked_kv_slice))
    ) % block_size
    offs_d = gl.arange(0, HEAD_DIM_LOAD, layout=gl.SliceLayout(1, blocked_kv_slice))
    offs_n0 = tile_start + gl.arange(0, BLOCK_N_HALF, layout=gl.SliceLayout(0, blocked_kv_slice))
    offs_k0 = slot_0[None, :] * stride_slot + cur_kv_head * stride_h + offs_d[:, None]
    offs_v0 = offs_k0
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_k0, K_gluon, offs_k0, mask=offs_n0[None, :] < cur_batch_ctx_len
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_v0, V_gluon, offs_v0, mask=offs_n0[None, :] < cur_batch_ctx_len
        )
        gl.amd.cdna4.async_copy.commit_group()
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_k0, K_gluon + offs_k0)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_v0, V_gluon + offs_v0)
        gl.amd.cdna4.async_copy.commit_group()

    page_bn_1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
        buf_page_1, gl.SliceLayout(0, blocked_kv_slice)
    )
    slot_1 = page_bn_1 * block_size + (
        tile_start + BLOCK_N_HALF
        + gl.arange(0, BLOCK_N_HALF, layout=gl.SliceLayout(0, blocked_kv_slice))
    ) % block_size
    offs_n1 = offs_n0 + BLOCK_N_HALF
    offs_k1 = slot_1[None, :] * stride_slot + cur_kv_head * stride_h + offs_d[:, None]
    offs_v1 = offs_k1
    if WITHIN_2GB:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_k1, K_gluon, offs_k1, mask=offs_n1[None, :] < cur_batch_ctx_len
        )
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_v1, V_gluon, offs_v1, mask=offs_n1[None, :] < cur_batch_ctx_len
        )
        gl.amd.cdna4.async_copy.commit_group()
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_k1, K_gluon + offs_k1)
        gl.amd.cdna4.async_copy.commit_group()
        gl.amd.cdna4.async_copy.global_load_to_shared(bufs_v1, V_gluon + offs_v1)
        gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _pa_prefill_ctx_tile(
    q,
    buf_k,
    buf_v,
    m_i,
    l_i,
    acc,
    mfma_layout,
    dot_a,
    dot_b,
    linear_v,
    sm_scale,
    k_scale,
    v_scale,
    tile_start,
    cur_batch_ctx_len,
    cur_batch_seq_len,
    cur_batch_ctx_len_total,
    offs_n_qk,
    offs_m_qk,
    alibi_slope,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_FP8_KV: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    USE_ALIBI: gl.constexpr,
    USE_ASYNC_LOAD: gl.constexpr,
    HEAD_DIM_LOAD: gl.constexpr,
):
    LOG2E: gl.constexpr = 1.4426950408889634
    buf_k_m = buf_k.slice(0, BLOCK_DMODEL, 0)
    buf_v_m = buf_v.slice(0, BLOCK_DMODEL, 0)
    if USE_ASYNC_LOAD:
        k = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_k_m, dot_b)
    else:
        k = buf_k_m.load(layout=dot_b)
    if IS_FP8_KV:
        k = (k.to(gl.float32) * k_scale).to(q.dtype)
    else:
        k = k.to(q.dtype)
    zeros = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
    qk = gl.amd.cdna4.mfma(q, k, zeros)
    qk = gl.where((tile_start + offs_n_qk[None, :]) < cur_batch_ctx_len, qk, float("-inf"))
    qk *= sm_scale
    if SLIDING_WINDOW > 0:
        qk = gl.where(
            (cur_batch_ctx_len_total + offs_m_qk[:, None]) - (tile_start + offs_n_qk[None, :])
            < SLIDING_WINDOW,
            qk,
            -10000.0,
        )
    if USE_ALIBI:
        alibi_start_q = offs_m_qk + cur_batch_ctx_len_total
        alibi = (offs_n_qk[None, :] + tile_start - alibi_start_q[:, None]) * alibi_slope
        alibi = gl.where(
            (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
            alibi,
            float("-inf"),
        )
        qk += alibi
    m_ij = gl.max(qk, 1)
    m_i_new = gl.maximum(m_i, m_ij)
    alpha = gl.exp2((m_i - m_i_new) * LOG2E)
    p = gl.exp2((qk - m_i_new[:, None]) * LOG2E)
    l_i_new = l_i * alpha + gl.sum(p, 1)
    acc_new = acc * alpha[:, None]
    if USE_ASYNC_LOAD:
        v_c = gl.amd.cdna4.async_copy.load_shared_relaxed(buf_v_m, linear_v)
    else:
        v_c = buf_v_m.load(layout=linear_v)
    if IS_FP8_KV:
        v_c = (v_c.to(gl.float32) * v_scale).to(q.dtype)
    else:
        v_c = v_c.to(q.dtype)
    v_c = gl.permute(v_c, [1, 0])
    v_c = gl.convert_layout(v_c, dot_b)
    p_dot = gl.convert_layout(p.to(q.dtype), dot_a)
    acc_new = gl.amd.cdna4.mfma(p_dot, v_c, acc_new)
    return m_i_new, l_i_new, acc_new


@gluon.jit
def _fwd_kernel_gluon(
    Q,
    K,
    V,
    K_cache,
    V_cache,
    K_gluon,
    V_gluon,
    B_Loc,
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc,
    B_Seqlen,
    Alibi_slopes,
    block_size,
    x,
    Out,
    stride_b_loc_b,
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    stride_kv_slot,
    stride_kv_h,
    num_queries_per_kv: int,
    IS_FP8_KV: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_PADDED: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    SKIP_DECODE: gl.constexpr,
    USE_ALIBI: gl.constexpr,
    WARP_M: gl.constexpr,
    WITHIN_2GB: gl.constexpr,
    HEAD_DIM_LOAD: gl.constexpr,
    USE_FULL_ASYNC_CTX: gl.constexpr,
):
    cur_batch = gl.program_id(0)
    cur_head = gl.program_id(1)
    start_m = gl.program_id(2)

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_seq_len = gl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = gl.load(B_Start_Loc + cur_batch)
    cur_batch_in_all_stop_index = gl.load(B_Start_Loc + cur_batch + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len

    if SKIP_DECODE and cur_batch_query_len == 1:
        return

    block_start_loc = BLOCK_M * start_m
    LOG2E: gl.constexpr = 1.4426950408889634

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[WARP_M, 1]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=8)

    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[16, 4],
        warps_per_cta=[WARP_M, 1],
        order=[1, 0],
    )
    blocked_kt: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, WARP_M],
        order=[0, 1],
    )
    blocked_v: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4],
        threads_per_warp=[16, 4],
        warps_per_cta=[WARP_M, 1],
        order=[1, 0],
    )

    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_q))
    offs_d_q = gl.arange(0, BLOCK_DMODEL_PADDED, layout=gl.SliceLayout(0, blocked_q))
    dim_mask_q = offs_d_q < BLOCK_DMODEL

    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d_q[None, :] * stride_qd
    )
    q = gl.load(
        Q + off_q,
        mask=dim_mask_q[None, :] & (offs_m[:, None] < cur_batch_query_len),
        other=0.0,
    )
    q = gl.convert_layout(q, dot_a)

    m_i = gl.zeros([BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout)) - float("inf")
    l_i = gl.zeros([BLOCK_M], dtype=gl.float32, layout=gl.SliceLayout(1, mfma_layout))
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=gl.float32, layout=mfma_layout)

    offs_n_kt = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kt))
    offs_d_kt = gl.arange(0, BLOCK_DMODEL_PADDED, layout=gl.SliceLayout(1, blocked_kt))
    dim_mask_kt = offs_d_kt < BLOCK_DMODEL

    offs_n_v = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, blocked_v))
    offs_d_v = gl.arange(0, BLOCK_DMODEL_PADDED, layout=gl.SliceLayout(0, blocked_v))
    dim_mask_v = offs_d_v < BLOCK_DMODEL

    offs_n_qk = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    offs_m_qk = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))

    if USE_ALIBI:
        alibi_slope = gl.load(Alibi_slopes + cur_head)
    else:
        alibi_slope = 0.0

    USE_ASYNC_CTX: gl.constexpr = (
        BLOCK_M == 64 and BLOCK_N == 64 and WARP_M == 4 and USE_FULL_ASYNC_CTX
    )
    BLOCK_N_HALF: gl.constexpr = BLOCK_N // 2

    # ===== context phase (no causal mask) =====
    if USE_ASYNC_CTX and cur_batch_ctx_len >= 3 * BLOCK_N:
        num_ctx_iter = gl.cdiv(cur_batch_ctx_len, BLOCK_N)
        gl.assume(num_ctx_iter >= 3)

        blocked_kv: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16), (0, 32)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[HEAD_DIM_LOAD, BLOCK_N],
        )
        blocked_kv_slice: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((1, 0), (2, 0), (4, 0), (0, 8), (0, 4), (0, 16)),
            lane_bases=((8, 0), (16, 0), (32, 0), (64, 0), (128, 0), (256, 0)),
            warp_bases=((0, 1), (0, 2)),
            block_bases=[],
            shape=[HEAD_DIM_LOAD, BLOCK_N_HALF],
        )
        shared_kv: gl.constexpr = gl.PaddedSharedLayout(
            interval_padding_pairs=[[HEAD_DIM_LOAD, 16]],
            offset_bases=[
                [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0],
                [0, 1], [0, 2], [0, 8], [0, 4], [0, 16], [0, 32],
            ],
            cga_layout=[],
            shape=[HEAD_DIM_LOAD, BLOCK_N],
        )
        linear_v: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0, 1), (0, 2), (0, 4), (0, 32), (64, 0), (128, 0), (256, 0)),
            lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 8), (0, 16)),
            warp_bases=((0, 0), (0, 0)),
            block_bases=[],
            shape=[HEAD_DIM_LOAD, BLOCK_N],
        )
        blocked_page: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=((0,),),
            lane_bases=((1,), (2,), (4,), (8,), (16,), (32,)),
            warp_bases=((0,), (0,)),
            block_bases=[],
            shape=[BLOCK_N],
        )
        shared_page: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[0]
        )

        kvtype = K_gluon.type.element_ty
        bufs_k = gl.allocate_shared_memory(
            kvtype, shape=[2, HEAD_DIM_LOAD, BLOCK_N], layout=shared_kv
        )
        bufs_v = gl.allocate_shared_memory(
            kvtype, shape=[2, HEAD_DIM_LOAD, BLOCK_N], layout=shared_kv
        )
        bufs_page = gl.allocate_shared_memory(
            gl.int32, shape=[2, BLOCK_N], layout=shared_page
        )

        batch_page_base = cur_batch * stride_b_loc_b
        offs_page_raw = gl.arange(0, BLOCK_N, layout=blocked_page)
        kv_scale_k = 1.0
        kv_scale_v = 1.0

        ################ prologue
        start_n = 0
        offs_n_page = start_n + offs_page_raw
        offs_page = batch_page_base + (offs_n_page // block_size) * stride_b_loc_s
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_page.index(0), B_Loc, offs_page, offs_n_page < cur_batch_ctx_len
        )
        gl.amd.cdna4.async_copy.commit_group()

        start_n += BLOCK_N
        offs_n_page = start_n + offs_page_raw
        offs_page = batch_page_base + (offs_n_page // block_size) * stride_b_loc_s
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            bufs_page.index(1), B_Loc, offs_page, offs_n_page < cur_batch_ctx_len
        )
        gl.amd.cdna4.async_copy.commit_group()

        gl.amd.cdna4.async_copy.wait_group(1)
        _pa_prefill_async_load_kv_slices(
            bufs_k, bufs_v, bufs_page.index(0), 0, 0,
            K_gluon, V_gluon, cur_kv_head, cur_batch_ctx_len, block_size,
            stride_kv_slot, stride_kv_h,
            blocked_kv_slice, BLOCK_N, BLOCK_N_HALF, HEAD_DIM_LOAD, WITHIN_2GB,
        )

        buf_idx = 0
        tile_compute = 0
        start_n_load = BLOCK_N
        ################ loop
        for i in range(num_ctx_iter - 2):
            async_idx = (buf_idx + 1) % 2

            gl.amd.cdna4.async_copy.wait_group(0)
            offs_n_page = start_n + BLOCK_N + offs_page_raw
            offs_page = batch_page_base + (offs_n_page // block_size) * stride_b_loc_s
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                bufs_page.index(buf_idx), B_Loc, offs_page, offs_n_page < cur_batch_ctx_len
            )
            gl.amd.cdna4.async_copy.commit_group()

            _pa_prefill_async_load_kv_slices(
                bufs_k, bufs_v, bufs_page.index(async_idx), async_idx, start_n_load,
                K_gluon, V_gluon, cur_kv_head, cur_batch_ctx_len, block_size,
                stride_kv_slot, stride_kv_h, stride_kv_h,
                blocked_kv_slice, BLOCK_N, BLOCK_N_HALF, HEAD_DIM_LOAD, WITHIN_2GB,
            )

            m_i, l_i, acc = _pa_prefill_ctx_tile(
                q, bufs_k.index(buf_idx), bufs_v.index(buf_idx),
                m_i, l_i, acc, mfma_layout, dot_a, dot_b, linear_v,
                sm_scale, kv_scale_k, kv_scale_v,
                tile_compute, cur_batch_ctx_len, cur_batch_seq_len, cur_batch_ctx_len,
                offs_n_qk, offs_m_qk, alibi_slope,
                BLOCK_M, BLOCK_N, IS_FP8_KV, SLIDING_WINDOW, USE_ALIBI,
                True, HEAD_DIM_LOAD,
            )

            start_n += BLOCK_N
            start_n_load += BLOCK_N
            tile_compute += BLOCK_N
            buf_idx = async_idx

        ################ epilogue 1
        if num_ctx_iter >= 2:
            async_idx = (buf_idx + 1) % 2
            gl.amd.cdna4.async_copy.wait_group(3)
            if start_n_load < num_ctx_iter * BLOCK_N:
                _pa_prefill_async_load_kv_slices(
                    bufs_k, bufs_v, bufs_page.index(async_idx), async_idx, start_n_load,
                    K_gluon, V_gluon, cur_kv_head, cur_batch_ctx_len, block_size,
                    stride_kv_slot, stride_kv_h, stride_kv_h,
                    blocked_kv_slice, BLOCK_N, BLOCK_N_HALF, HEAD_DIM_LOAD, WITHIN_2GB,
                )

            m_i, l_i, acc = _pa_prefill_ctx_tile(
                q, bufs_k.index(buf_idx), bufs_v.index(buf_idx),
                m_i, l_i, acc, mfma_layout, dot_a, dot_b, linear_v,
                sm_scale, kv_scale_k, kv_scale_v,
                tile_compute, cur_batch_ctx_len, cur_batch_seq_len, cur_batch_ctx_len,
                offs_n_qk, offs_m_qk, alibi_slope,
                BLOCK_M, BLOCK_N, IS_FP8_KV, SLIDING_WINDOW, USE_ALIBI,
                True, HEAD_DIM_LOAD,
            )
            tile_compute += BLOCK_N
            buf_idx = async_idx

        ################ epilogue 2
        gl.amd.cdna4.async_copy.wait_group(0)
        m_i, l_i, acc = _pa_prefill_ctx_tile(
            q, bufs_k.index(buf_idx), bufs_v.index(buf_idx),
            m_i, l_i, acc, mfma_layout, dot_a, dot_b, linear_v,
            sm_scale, kv_scale_k, kv_scale_v,
            tile_compute, cur_batch_ctx_len, cur_batch_seq_len, cur_batch_ctx_len,
            offs_n_qk, offs_m_qk, alibi_slope,
            BLOCK_M, BLOCK_N, IS_FP8_KV, SLIDING_WINDOW, USE_ALIBI,
            True, HEAD_DIM_LOAD,
        )

    elif cur_batch_ctx_len > 0:
        bn0 = gl.load(
            B_Loc + cur_batch * stride_b_loc_b
            + ((0 + offs_n_kt) // block_size) * stride_b_loc_s,
            mask=(0 + offs_n_kt) < cur_batch_ctx_len,
            other=0,
        )
        off_k0 = (
            bn0[None, :] * stride_k_cache_bs
            + cur_kv_head * stride_k_cache_h
            + (offs_d_kt[:, None] // x) * stride_k_cache_d
            + ((0 + offs_n_kt[None, :]) % block_size) * stride_k_cache_bl
            + (offs_d_kt[:, None] % x) * stride_k_cache_x
        )
        k_raw = gl.load(
            K_cache + off_k0,
            mask=dim_mask_kt[:, None] & ((0 + offs_n_kt[None, :]) < cur_batch_ctx_len),
            other=0.0,
        )
        bn0v = gl.load(
            B_Loc + cur_batch * stride_b_loc_b
            + ((0 + offs_n_v) // block_size) * stride_b_loc_s,
            mask=(0 + offs_n_v) < cur_batch_ctx_len,
            other=0,
        )
        off_v0 = (
            bn0v[:, None] * stride_v_cache_bs
            + cur_kv_head * stride_v_cache_h
            + offs_d_v[None, :] * stride_v_cache_d
            + ((0 + offs_n_v[:, None]) % block_size) * stride_v_cache_bl
        )
        v_raw = gl.load(
            V_cache + off_v0,
            mask=dim_mask_v[None, :] & ((0 + offs_n_v[:, None]) < cur_batch_ctx_len),
            other=0.0,
        )

        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = gl.multiple_of(start_n, BLOCK_N)
            if IS_FP8_KV:
                k = (k_raw.to(gl.float32) * gl.load(k_scale)).to(q.dtype)
            else:
                k = k_raw.to(q.dtype)
            k = gl.convert_layout(k, dot_b)

            zeros = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
            qk = gl.amd.cdna4.mfma(q, k, zeros)

            sn = start_n + BLOCK_N
            bn_n = gl.load(
                B_Loc + cur_batch * stride_b_loc_b
                + ((sn + offs_n_kt) // block_size) * stride_b_loc_s,
                mask=(sn + offs_n_kt) < cur_batch_ctx_len,
                other=0,
            )
            off_kn = (
                bn_n[None, :] * stride_k_cache_bs
                + cur_kv_head * stride_k_cache_h
                + (offs_d_kt[:, None] // x) * stride_k_cache_d
                + ((sn + offs_n_kt[None, :]) % block_size) * stride_k_cache_bl
                + (offs_d_kt[:, None] % x) * stride_k_cache_x
            )
            k_next = gl.load(
                K_cache + off_kn,
                mask=dim_mask_kt[:, None] & ((sn + offs_n_kt[None, :]) < cur_batch_ctx_len),
                other=0.0,
            )
            bn_nv = gl.load(
                B_Loc + cur_batch * stride_b_loc_b
                + ((sn + offs_n_v) // block_size) * stride_b_loc_s,
                mask=(sn + offs_n_v) < cur_batch_ctx_len,
                other=0,
            )
            off_vn = (
                bn_nv[:, None] * stride_v_cache_bs
                + cur_kv_head * stride_v_cache_h
                + offs_d_v[None, :] * stride_v_cache_d
                + ((sn + offs_n_v[:, None]) % block_size) * stride_v_cache_bl
            )
            v_next = gl.load(
                V_cache + off_vn,
                mask=dim_mask_v[None, :] & ((sn + offs_n_v[:, None]) < cur_batch_ctx_len),
                other=0.0,
            )

            qk = gl.where(
                (start_n + offs_n_qk[None, :]) < cur_batch_ctx_len, qk, float("-inf")
            )
            qk *= sm_scale
            if SLIDING_WINDOW > 0:
                qk = gl.where(
                    (cur_batch_ctx_len + offs_m_qk[:, None]) - (start_n + offs_n_qk[None, :])
                    < SLIDING_WINDOW,
                    qk,
                    -10000.0,
                )
            if USE_ALIBI:
                alibi_start_q = offs_m_qk + cur_batch_ctx_len
                alibi = (
                    offs_n_qk[None, :] + start_n - alibi_start_q[:, None]
                ) * alibi_slope
                alibi = gl.where(
                    (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                    alibi,
                    float("-inf"),
                )
                qk += alibi

            m_ij = gl.max(qk, 1)
            m_i_new = gl.maximum(m_i, m_ij)
            alpha = gl.exp2((m_i - m_i_new) * LOG2E)
            p = gl.exp2((qk - m_i_new[:, None]) * LOG2E)
            l_ij = gl.sum(p, 1)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            if IS_FP8_KV:
                v = (v_raw.to(gl.float32) * gl.load(v_scale)).to(q.dtype)
            else:
                v = v_raw.to(q.dtype)
            p_dot = gl.convert_layout(p.to(q.dtype), dot_a)
            v = gl.convert_layout(v, dot_b)
            acc = gl.amd.cdna4.mfma(p_dot, v, acc)
            m_i = m_i_new
            k_raw = k_next
            v_raw = v_next

    # ===== self phase (causal mask over query) =====
    block_mask = 1
    if block_start_loc >= cur_batch_query_len:
        block_mask = 0

    offs_n_kself = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kt))
    offs_n_vself = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, blocked_v))

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = gl.multiple_of(start_n, BLOCK_N)
        off_k = (
            (cur_batch_in_all_start_index + start_n + offs_n_kself[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d_kt[:, None] * stride_kd
        )
        k = gl.load(
            K + off_k,
            mask=dim_mask_kt[:, None]
            & ((start_n + offs_n_kself[None, :]) < cur_batch_query_len),
            other=0.0,
        )
        k = gl.convert_layout(k.to(q.dtype), dot_b)

        zeros = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=mfma_layout)
        qk = gl.amd.cdna4.mfma(q, k, zeros)
        qk *= sm_scale
        qk = gl.where(
            offs_m_qk[:, None] >= (start_n + offs_n_qk[None, :]), qk, float("-inf")
        )
        if SLIDING_WINDOW > 0:
            qk = gl.where(
                offs_m_qk[:, None] - (start_n + offs_n_qk[None, :]) < SLIDING_WINDOW,
                qk,
                -10000.0,
            )
        if USE_ALIBI:
            alibi_start_q = offs_m_qk + cur_batch_ctx_len
            alibi_start_k = cur_batch_ctx_len + start_n
            alibi = (
                offs_n_qk[None, :] + alibi_start_k - alibi_start_q[:, None]
            ) * alibi_slope
            alibi = gl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi,
                float("-inf"),
            )
            qk += alibi

        m_ij = gl.max(qk, 1)
        m_i_new = gl.maximum(m_i, m_ij)
        alpha = gl.exp2((m_i - m_i_new) * LOG2E)
        p = gl.exp2((qk - m_i_new[:, None]) * LOG2E)
        l_ij = gl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        off_v = (
            (cur_batch_in_all_start_index + start_n + offs_n_vself[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d_v[None, :] * stride_vd
        )
        v = gl.load(
            V + off_v,
            mask=dim_mask_v[None, :]
            & ((start_n + offs_n_vself[:, None]) < cur_batch_query_len),
            other=0.0,
        )
        p_dot = gl.convert_layout(p.to(q.dtype), dot_a)
        v = gl.convert_layout(v.to(q.dtype), dot_b)
        acc = gl.amd.cdna4.mfma(p_dot, v, acc)
        m_i = m_i_new

    l_recip = 1.0 / l_i
    acc = acc * l_recip[:, None]

    offs_o_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout))
    offs_o_d = gl.arange(0, BLOCK_DMODEL_PADDED, layout=gl.SliceLayout(0, mfma_layout))
    dim_mask_o = offs_o_d < BLOCK_DMODEL
    off_o = (
        (cur_batch_in_all_start_index + offs_o_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_o_d[None, :] * stride_od
    )
    gl.store(
        Out + off_o,
        acc.to(Out.dtype.element_ty),
        mask=dim_mask_o[None, :] & (offs_o_m[:, None] < cur_batch_query_len),
    )


@torch.inference_mode()
def context_attention_fwd_gluon(
    q,
    k,
    v,
    o,
    kv_cache_dtype: str,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_input_len,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode=False,
):
    _LOGGER.info(
        f"PA_PREFILL_GLUON: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
    )
    q_dtype_is_f32 = q.dtype is torch.float32

    is_fp8_kv = (
        torch.finfo(k_cache.dtype).bits == 8 or torch.finfo(v_cache.dtype).bits == 8
    )
    if is_fp8_kv and kv_cache_dtype == "auto":
        raise ValueError(
            "kv_cache_dtype='auto' unsupported for FP8 KV Cache prefill kernel"
        )

    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    Lk_padded = triton.next_power_of_2(Lk)

    if sm_scale is None:
        sm_scale = 1.0 / (Lq**0.5)

    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]
    assert batch + 1 == len(b_start_loc)

    if q_dtype_is_f32:
        BLOCK = 64
        BLOCK_N_cfg = 32
        WARP_M = 4
    elif Lk_padded <= 64:
        BLOCK = 256
        BLOCK_N_cfg = 64
        WARP_M = 8
    else:
        BLOCK = 128
        BLOCK_N_cfg = 64
        WARP_M = 4

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    use_alibi = alibi_slopes is not None
    if use_alibi:
        sliding_window = 0
        alibi_arg = alibi_slopes
    else:
        alibi_arg = b_seq_len

    if b_loc.dtype != torch.int32:
        b_loc = b_loc.contiguous().to(torch.int32)

    max_kv_bytes = k_cache.shape[0] * k_cache.stride(0) * k_cache.element_size()
    within_2gb = max_kv_bytes <= 0x80000000

    use_full_async = (
        not is_fp8_kv
        and not q_dtype_is_f32
        and Lk == 128
        and BLOCK == 64
        and BLOCK_N_cfg == 64
        and WARP_M == 4
    )
    if use_full_async:
        k_gluon, v_gluon = _kv_to_slot_linear_pad512(k_cache, v_cache)
        num_kv_heads = k_cache.shape[1]
        stride_kv_slot = num_kv_heads * _HEAD_DIM_ASYNC_LOAD
        stride_kv_h = _HEAD_DIM_ASYNC_LOAD
        head_dim_load = _HEAD_DIM_ASYNC_LOAD
    else:
        k_gluon = k_cache
        v_gluon = v_cache
        stride_kv_slot = 0
        stride_kv_h = 0
        head_dim_load = Lk_padded

    _fwd_kernel_gluon[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        k_gluon,
        v_gluon,
        b_loc,
        sm_scale,
        k_scale,
        v_scale,
        b_start_loc,
        b_seq_len,
        alibi_arg,
        v_cache.shape[3],
        k_cache.shape[4],
        o,
        b_loc.stride(0),
        b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(4),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        stride_kv_slot,
        stride_kv_h,
        num_queries_per_kv=num_queries_per_kv,
        IS_FP8_KV=is_fp8_kv,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,
        BLOCK_DMODEL_PADDED=Lk_padded,
        BLOCK_N=BLOCK_N_cfg,
        SLIDING_WINDOW=sliding_window,
        SKIP_DECODE=skip_decode,
        USE_ALIBI=use_alibi,
        WARP_M=WARP_M,
        WITHIN_2GB=within_2gb,
        HEAD_DIM_LOAD=head_dim_load,
        USE_FULL_ASYNC_CTX=use_full_async,
        num_warps=WARP_M,
        num_stages=1,
    )
    return
