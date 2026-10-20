# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Direct BF16 expert-route decode GEMMs for gfx950 (at most 16 tokens).

Both stages launch one program per (route slot, output tile) and read the
expert straight from ``topk_ids``: no sort and no routing metadata. The first
slot that selects an expert leads it: its program gathers every token routed
to that expert (at most one route per token, so at most 16 rows) and streams
the expert's weight tile once through a 16-row MFMA. Programs of later slots
with the same expert exit, so each selected weight tile is read once per
launch. In stage 2 the program of a slot whose expert is not local (or
invalid) writes that route's zero output instead.

The waves split every K iteration (a batched MFMA whose batch dimension is
the wave) and reduce through LDS once, like the dense decode GEMV
(``gemm/fp16/gemv.py``). The K loop is unrolled with the next iteration's
loads issued before the current MFMAs; weights stream with ``.cg``.

  Stage 1: inter[route] = silu(x @ gate.T) * (x @ up.T), gate/up columns
           interleaved inside the tile so SwiGLU pairs live in one lane.
  Stage 2: out[route] = inter[route] @ w2.T, unweighted.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

cdna4 = gl.amd.cdna4

__all__ = [
    "MAX_TOKENS",
    "gluon_bf16_moe_routes_stage1_gfx950",
    "gluon_bf16_moe_routes_stage2_gfx950",
    "launch_gluon_bf16_moe_routes_decode_gfx950",
]

MAX_TOKENS = 16
NUM_WARPS = 4
_WAVE = 64  # wavefront lanes (host-side layout helpers)


def _decode_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (tokens={args['M']})"}


@gluon.constexpr_function
def _mfma_layout(num_warps: int):
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1, 1],
    )


@gluon.constexpr_function
def _weight_layout(block_k: int, num_warps: int):
    # [waves, BLOCK_K, BLOCK_N]: 8 lanes x 16 B read 128 contiguous bytes of
    # one weight row per instruction.
    tk = min(block_k // 8, _WAVE)
    return gl.BlockedLayout(
        [1, 8, 1], [1, tk, _WAVE // tk], [num_warps, 1, 1], [1, 2, 0]
    )


@gluon.constexpr_function
def _ids_layout(num_warps: int):
    return gl.BlockedLayout([1, 1], [16, 4], [1, num_warps], [1, 0])


@gluon.constexpr_function
def _swiglu_layout(block_n: int, num_warps: int):
    # Gate/up column pairs stay in one lane for the split.
    tn = min(block_n // 2, _WAVE)
    return gl.BlockedLayout([1, 2], [_WAVE // tn, tn], [num_warps, 1], [1, 0])


@gluon.jit
def _expert_routes(ids_ptr, M, slot, TOPK: gl.constexpr, BLOCK_M: gl.constexpr):
    """Expert of ``slot``, whether ``slot`` is its first route, per token the
    last route row (``token * TOPK + k``) that selects it or -1, and the
    ``[BLOCK_M, TOPK]`` route rows that select it (-1 elsewhere)."""
    L: gl.constexpr = _ids_layout(gl.num_warps())
    t = gl.arange(0, BLOCK_M, gl.SliceLayout(1, L))
    j = gl.arange(0, TOPK, gl.SliceLayout(0, L))
    flat = t[:, None] * TOPK + j[None, :]
    ids = gl.load(ids_ptr + flat, mask=flat < M * TOPK, other=-1)
    expert = gl.load(ids_ptr + slot)
    selected = gl.where(ids == expert, flat, -1)
    first = gl.min(
        gl.min(gl.where(selected >= 0, flat, BLOCK_M * TOPK), axis=1), axis=0
    )
    return expert, first == slot, gl.max(selected, axis=1), selected


@gluon.jit
def _store_routes(value, out_ptr, selected, col, WIDTH: gl.constexpr):
    """Store each token's ``value`` row to every route row that selected the
    expert (a token may select one expert in several slots)."""
    TOPK: gl.constexpr = selected.shape[1]
    j = gl.arange(0, TOPK, gl.SliceLayout(0, selected.type.layout))
    for k in gl.static_range(TOPK):
        rows = gl.max(gl.where(j[None, :] == k, selected, -1), axis=1)
        rows = gl.convert_layout(rows, gl.SliceLayout(1, value.type.layout))
        cdna4.buffer_store(
            value,
            out_ptr,
            rows[:, None] * WIDTH + col[None, :],
            mask=(rows >= 0)[:, None] & (col < WIDTH)[None, :],
        )


@gluon.jit
def _route_gemm(
    a_ptr,
    a_rows,
    w_ptr,
    w_rows,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """``a[a_rows] @ w[w_rows].T`` over K; rows < 0 of ``a`` read zeros.

    ``a_rows`` is a ``[BLOCK_M]`` vector of any layout, ``w_rows`` a
    ``[BLOCK_N]`` vector in the weight-load column layout.
    """
    KS: gl.constexpr = gl.num_warps()
    KITER: gl.constexpr = KS * BLOCK_K
    gl.static_assert(K % KITER == 0, "K must be a multiple of waves * BLOCK_K")
    ITERS: gl.constexpr = K // KITER
    mfma: gl.constexpr = _mfma_layout(KS)
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    b_l: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    WL: gl.constexpr = _weight_layout(BLOCK_K, KS)
    s = gl.arange(0, KS, gl.SliceLayout(1, gl.SliceLayout(2, WL)))
    kk = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gl.SliceLayout(2, WL)))
    w_offs = (s[:, None, None] * BLOCK_K + kk[None, :, None]) + w_rows[
        None, None, :
    ] * K
    w = cdna4.buffer_load(w_ptr, w_offs, cache=".cg")
    a_s = gl.arange(0, KS, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    a_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    rows = gl.convert_layout(a_rows, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    ak = a_s[:, None, None] * BLOCK_K + a_k[None, None, :]
    a_mask = (rows >= 0)[None, :, None] & (ak < K)
    a_offs = rows[None, :, None] * K + ak
    a = cdna4.buffer_load(a_ptr, a_offs, mask=a_mask, other=0.0)
    acc = gl.zeros((KS, BLOCK_M, BLOCK_N), gl.float32, mfma)
    # Unrolled: a loop-carried copy of a pending load would wait for it.
    for it in gl.static_range(ITERS):
        if it + 1 < ITERS:
            next_w = cdna4.buffer_load(w_ptr, w_offs + (it + 1) * KITER, cache=".cg")
            next_a = cdna4.buffer_load(
                a_ptr, a_offs + (it + 1) * KITER, mask=a_mask, other=0.0
            )
        acc = cdna4.mfma(gl.convert_layout(a, a_l), gl.convert_layout(w, b_l), acc)
        if it + 1 < ITERS:
            w = next_w
            a = next_a
    # The waves' K partials, summed through LDS.
    return gl.sum(acc, axis=0)


@gluon.jit(launch_metadata=_decode_launch_metadata, do_not_specialize=("M",))
def gluon_bf16_moe_routes_stage1_gfx950(
    x_ptr,
    w13_ptr,
    ids_ptr,
    inter_ptr,
    M,
    expert_start,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Gate/up GEMM + SwiGLU of output columns ``[tile * BLOCK_N / 2, ...)``.

    ``x`` is ``[M, HIDDEN]``, ``w13`` ``[NUM_EXPERTS, 2 * INTERMEDIATE,
    HIDDEN]`` (gate rows first), ``ids`` ``[M, TOPK]`` int32 global expert
    ids and ``inter`` ``[M * TOPK, INTERMEDIATE]``. Rows of routes whose
    expert is not local are left unwritten.
    """
    BLOCK_M: gl.constexpr = 16
    # Tiles vary fastest: consecutive workgroups go to different XCDs, so the
    # few local routes of an expert-parallel rank still spread over all of them.
    tile = gl.program_id(0)
    slot = gl.program_id(1)
    expert, leader, routes, selected = _expert_routes(ids_ptr, M, slot, TOPK, BLOCK_M)
    local = expert - expert_start
    if leader & (local >= 0) & (local < NUM_EXPERTS):
        WL: gl.constexpr = _weight_layout(BLOCK_K, gl.num_warps())
        c = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, WL)))
        # Even columns are gate rows, odd columns the matching up rows.
        w_rows = (c % 2) * INTERMEDIATE + tile * (BLOCK_N // 2) + c // 2
        w_base = w13_ptr + local.to(gl.int64) * (2 * INTERMEDIATE * HIDDEN)
        tokens = gl.arange(0, BLOCK_M, gl.SliceLayout(1, _ids_layout(gl.num_warps())))
        a_rows = gl.where(routes >= 0, tokens, -1)
        acc = _route_gemm(
            x_ptr, a_rows, w_base, w_rows, HIDDEN, BLOCK_M, BLOCK_N, BLOCK_K
        )
        SL: gl.constexpr = _swiglu_layout(BLOCK_N, gl.num_warps())
        acc = gl.convert_layout(acc, SL)
        gate, up = gl.split(acc.reshape((BLOCK_M, BLOCK_N // 2, 2)))
        h = gate * (1.0 / (1.0 + gl.exp(-gate))) * up
        col = tile * (BLOCK_N // 2) + gl.arange(
            0, BLOCK_N // 2, gl.SliceLayout(0, h.type.layout)
        )
        _store_routes(
            h.to(inter_ptr.dtype.element_ty), inter_ptr, selected, col, INTERMEDIATE
        )


@gluon.jit(launch_metadata=_decode_launch_metadata, do_not_specialize=("M",))
def gluon_bf16_moe_routes_stage2_gfx950(
    inter_ptr,
    w2_ptr,
    ids_ptr,
    out_ptr,
    M,
    expert_start,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Down GEMM of output columns ``[tile * BLOCK_N, ...)`` of every route.

    ``w2`` is ``[NUM_EXPERTS, HIDDEN, INTERMEDIATE]`` and ``out`` ``[M *
    TOPK, HIDDEN]``; routes whose expert is not local get zeros.
    """
    BLOCK_M: gl.constexpr = 16
    # Tiles vary fastest: consecutive workgroups go to different XCDs, so the
    # few local routes of an expert-parallel rank still spread over all of them.
    tile = gl.program_id(0)
    slot = gl.program_id(1)
    expert, leader, routes, selected = _expert_routes(ids_ptr, M, slot, TOPK, BLOCK_M)
    local = expert - expert_start
    owned = (local >= 0) & (local < NUM_EXPERTS)
    if leader & owned:
        WL: gl.constexpr = _weight_layout(BLOCK_K, gl.num_warps())
        n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, WL)))
        w_base = w2_ptr + local.to(gl.int64) * (HIDDEN * INTERMEDIATE)
        acc = _route_gemm(
            inter_ptr,
            routes,
            w_base,
            tile * BLOCK_N + n,
            INTERMEDIATE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        col = tile * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, acc.type.layout))
        _store_routes(acc.to(out_ptr.dtype.element_ty), out_ptr, selected, col, HIDDEN)
    elif not owned:
        ZL: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
        col = tile * BLOCK_N + gl.arange(0, BLOCK_N, ZL)
        cdna4.buffer_store(
            gl.zeros((BLOCK_N,), out_ptr.dtype.element_ty, ZL),
            out_ptr,
            slot * HIDDEN + col,
        )


def launch_gluon_bf16_moe_routes_decode_gfx950(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
    out: torch.Tensor,
    config: tuple[int, int, int, int],
) -> torch.Tensor:
    """Write unweighted BF16 expert routes of at most 16 tokens into ``out``.

    Args:
        x: Contiguous BF16 ``[M, H]`` activations, ``1 <= M <= 16``.
        w13: Contiguous ``[E, 2I, H]`` gate/up weights (gate rows first).
        w2: Contiguous ``[E, H, I]`` down weights.
        topk_ids: Contiguous int32 ``[M, top_k]`` global expert ids; a token
            selects an expert at most once.
        expert_start: Global id of local expert 0.
        out: Contiguous ``[M * top_k, H]`` BF16 destination. Routes of
            non-local or invalid experts are written as zeros.
        config: ``(stage1 BLOCK_N, stage1 BLOCK_K, stage2 BLOCK_N,
            stage2 BLOCK_K)``.

    Returns:
        ``out``.
    """
    m, hidden = x.shape
    experts, two_i, _ = w13.shape
    inter_size = two_i // 2
    top_k = topk_ids.shape[1]
    s1_n, s1_k, s2_n, s2_k = config
    inter = torch.empty((m * top_k, inter_size), dtype=x.dtype, device=x.device)
    gluon_bf16_moe_routes_stage1_gfx950[(2 * inter_size // s1_n, m * top_k)](
        x,
        w13,
        topk_ids,
        inter,
        m,
        expert_start,
        HIDDEN=hidden,
        INTERMEDIATE=inter_size,
        TOPK=top_k,
        NUM_EXPERTS=experts,
        BLOCK_N=s1_n,
        BLOCK_K=s1_k,
        num_warps=NUM_WARPS,
    )
    gluon_bf16_moe_routes_stage2_gfx950[(hidden // s2_n, m * top_k)](
        inter,
        w2,
        topk_ids,
        out,
        m,
        expert_start,
        HIDDEN=hidden,
        INTERMEDIATE=inter_size,
        TOPK=top_k,
        NUM_EXPERTS=experts,
        BLOCK_N=s2_n,
        BLOCK_K=s2_k,
        num_warps=NUM_WARPS,
    )
    return out
