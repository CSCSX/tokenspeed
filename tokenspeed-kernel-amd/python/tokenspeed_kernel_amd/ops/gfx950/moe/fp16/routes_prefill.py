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

"""Grouped BF16 expert-route GEMMs for gfx950 prefill.

``routes_align.py`` packs each local expert's routes contiguously and cuts
them into 256-row tiles plus at most one 128-row remainder tile; rows past a
tile's route count are masked. Each workgroup computes
one ``BLOCK_M x 256`` output tile of one expert with the eight-wave,
two-buffer, warp-pipelined MFMA/LDS schedule of the dense prefill GEMM
(``gemm/fp16/largem.py``), whose activation rows are gathered through the
sorted route ids instead of read contiguously.

  Stage 1: the tile's left half reads gate rows ``[n, n + 128)`` and its right
           half the matching up rows ``I + [n, n + 128)``, so SwiGLU pairs
           share a lane; it stores ``inter[route, n:n + 128]``.
  Stage 2: ``out[route, n:n + 256] = inter[route] @ w2[e, n:n + 256].T``,
           unweighted.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, tl
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.routes_align import (
    align_bf16_moe_routes_gfx950,
)

cdna4 = gl.amd.cdna4
async_copy = gl.amd.cdna4.async_copy

__all__ = [
    "BIG_TILE",
    "BLOCK_N",
    "gluon_bf16_moe_routes_prefill_stage1_gfx950",
    "gluon_bf16_moe_routes_prefill_stage2_gfx950",
    "gluon_bf16_moe_routes_zero_gfx950",
    "launch_gluon_bf16_moe_routes_prefill_gfx950",
]

BLOCK_N = 256
# Route rows of a full tile; an expert's remainder of at most half this many
# rows takes one half-height tile.
BIG_TILE = 256
NUM_WARPS = 8
NUM_XCDS = gl.constexpr(8)
GROUP_SIZE_M = gl.constexpr(4)


def _prefill_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (tokens={args['num_tokens']})"}


@gluon.constexpr_function
def _load_layout_a(half_m: int):
    # [half_m, 64]: 16 bytes per lane; eight waves stack along M.
    if half_m == 128:
        return gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0]],
            lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
            warp_bases=[[1, 0], [2, 0], [4, 0]],
            block_bases=[],
            shape=[half_m, 64],
        )
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [8, 0], [16, 0], [32, 0]],
        warp_bases=[[1, 0], [2, 0], [4, 0]],
        block_bases=[],
        shape=[half_m, 64],
    )


@gluon.constexpr_function
def _shared_layout_a(half_m: int):
    # Offsets follow the load layout's lane order (a warp's copy fills one
    # contiguous range), then its register and warp rows; padded every 512.
    if half_m == 128:
        rows = [[16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]]
    else:
        rows = [[8, 0], [16, 0], [32, 0], [1, 0], [2, 0], [4, 0]]
    return gl.PaddedSharedLayout(
        [[512, 16]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]] + rows,
        [],
        [half_m, 64],
    )


@gluon.constexpr_function
def _store_layout(half_m: int):
    # Covers a [half_m, 128] quadrant once with eight waves.
    return gl.BlockedLayout([half_m // 32, 8], [4, 16], [8, 1], [1, 0])


@gluon.jit
def _grouped_tile(
    a_ptr,
    b_ptr,
    a_top_offsets,
    a_bot_offsets,
    right_delta,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Accumulate ``a[rows] @ b[cols].T`` for one ``BLOCK_M x 256`` tile.

    ``a_top_offsets`` / ``a_bot_offsets`` are the gathered element offsets of
    K step 0 of the top and bottom half rows in the A load layout. ``b_ptr``
    points at the tile's first weight row; the right half starts
    ``right_delta`` rows further. Returns the four FP32 quadrants (top-left,
    bottom-left, top-right, bottom-right).
    """
    HALF_M: gl.constexpr = BLOCK_M // 2
    HALF_N: gl.constexpr = 128
    BK: gl.constexpr = 64
    gl.static_assert(K % (2 * BK) == 0 and K >= 4 * BK, "K must pair 64-wide steps")
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2], [0, 4]],
        block_bases=[],
        shape=[BK, HALF_N],
    )
    sharedLayoutA: gl.constexpr = _shared_layout_a(HALF_M)
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        [],
        [BK, HALF_N],
    )
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 4]
    )
    dotA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )

    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, HALF_M, BK], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, HALF_M, BK], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, BK, HALF_N], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, BK, HALF_N], sharedLayoutB
    )

    offs_bn = gl.arange(0, HALF_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BK, gl.SliceLayout(1, gLoadLayoutB))
    b_left_offsets = offs_bk[:, None] + offs_bn[None, :] * K
    b_right_offsets = offs_bk[:, None] + (offs_bn + right_delta)[None, :] * K
    a_top_next = a_top_offsets + BK
    a_bot_next = a_bot_offsets + BK
    b_left_next = b_left_offsets + BK
    b_right_next = b_right_offsets + BK
    a_base = a_ptr
    b_base = b_ptr

    acc_tl = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)
    acc_br = gl.zeros((HALF_M, HALF_N), gl.float32, mfmaLayout)

    ITERS: gl.constexpr = K // BK

    async_copy.buffer_load_to_shared(smemB_left.index(0), b_base, b_left_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemB_right.index(0), b_base, b_right_offsets)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_next)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_top_next)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemA_bot.index(1), a_base, a_bot_next)
    async_copy.commit_group()
    async_copy.buffer_load_to_shared(smemB_right.index(1), b_base, b_right_next)
    async_copy.commit_group()

    a_base += 2 * BK
    b_base += 2 * BK

    async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotB)
    a_top = smemA_top.index(0).load(dotA)

    for _ in tl.range(0, ITERS - 2, 2):
        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(0).load(dotA)
            async_copy.buffer_load_to_shared(
                smemB_left.index(0), b_base, b_left_offsets
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(0).load(dotB)
            async_copy.buffer_load_to_shared(smemA_top.index(0), a_base, a_top_offsets)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(1).load(dotB)
            async_copy.buffer_load_to_shared(smemA_bot.index(0), a_base, a_bot_offsets)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(1).load(dotA)
            async_copy.buffer_load_to_shared(
                smemB_right.index(0), b_base, b_right_offsets
            )
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_bot = smemA_bot.index(1).load(dotA)
            async_copy.buffer_load_to_shared(smemB_left.index(1), b_base, b_left_next)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_right = smemB_right.index(1).load(dotB)
            async_copy.buffer_load_to_shared(smemA_top.index(1), a_base, a_top_next)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            b_left = smemB_left.index(0).load(dotB)
            async_copy.buffer_load_to_shared(smemA_bot.index(1), a_base, a_bot_next)
            async_copy.commit_group()

        async_copy.wait_group(5)
        with gl.amd.warp_pipeline_stage("mfma", priority=0):
            acc_br = cdna4.mfma(a_bot, b_right, acc_br)
        with gl.amd.warp_pipeline_stage("mem", priority=1):
            a_top = smemA_top.index(0).load(dotA)
            async_copy.buffer_load_to_shared(smemB_right.index(1), b_base, b_right_next)
            async_copy.commit_group()
            a_base += 2 * BK
            b_base += 2 * BK

    acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
    async_copy.wait_group(5)
    l_idx: gl.constexpr = (ITERS - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotA)
    acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
    async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotB)
    acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
    async_copy.wait_group(3)
    g_idx: gl.constexpr = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotB)
    acc_br = cdna4.mfma(a_bot, b_right, acc_br)
    async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotA)
    acc_tl = cdna4.mfma(a_top, b_left, acc_tl)
    async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotA)
    acc_bl = cdna4.mfma(a_bot, b_left, acc_bl)
    async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotB)
    acc_tr = cdna4.mfma(a_top, b_right, acc_tr)
    acc_br = cdna4.mfma(a_bot, b_right, acc_br)
    return acc_tl, acc_bl, acc_tr, acc_br


@gluon.jit
def _tile_ids(
    tile_expert_ptr,
    tile_row0_ptr,
    tile_rows_ptr,
    num_tiles_ptr,
    N_TILES: gl.constexpr,
):
    """``(pid_n, expert, row0, rows)`` of this program; expert is -1 past the
    live route tiles.

    The grid covers the worst case, but only the first ``num_tiles`` route
    tiles hold routes. Programs are numbered over those live tiles only, so
    the XCD remap below spreads them over all eight XCDs instead of handing
    the live prefix to the first few.
    """
    pid = gl.program_id(0)
    num_m = gl.load(num_tiles_ptr)
    grid_mn = num_m * N_TILES
    live = pid < grid_mn
    # Adjacent programs land on different XCDs; give each XCD a contiguous
    # range of tiles, then walk GROUP_SIZE_M route tiles per weight column.
    pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = grid_mn % NUM_XCDS
    tall_xcds = gl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    pid = gl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )
    group = GROUP_SIZE_M * N_TILES
    first_m = (pid // group) * GROUP_SIZE_M
    group_m = gl.maximum(gl.minimum(num_m - first_m, GROUP_SIZE_M), 1)
    pid_m = first_m + (pid % group) % group_m
    pid_n = (pid % group) // group_m
    expert = gl.load(tile_expert_ptr + pid_m, mask=live, other=-1)
    row0 = gl.load(tile_row0_ptr + pid_m, mask=live, other=0)
    rows = gl.load(tile_rows_ptr + pid_m, mask=live, other=0)
    return pid_n, expert, row0, rows


@gluon.jit(launch_metadata=_prefill_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_prefill_stage1_gfx950(
    x_ptr,
    w13_ptr,
    inter_ptr,
    sorted_routes_ptr,
    tile_expert_ptr,
    tile_row0_ptr,
    tile_rows_ptr,
    num_tiles_ptr,
    num_tokens,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Gate/up GEMM + SwiGLU of one ``BLOCK_M`` route block x 128 columns."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    N_TILES: gl.constexpr = INTERMEDIATE // 128
    pid_n, expert, row0, rows = _tile_ids(
        tile_expert_ptr, tile_row0_ptr, tile_rows_ptr, num_tiles_ptr, N_TILES
    )
    if expert >= 0:
        LA: gl.constexpr = _load_layout_a(HALF_M)
        r = gl.arange(0, HALF_M, gl.SliceLayout(1, LA))
        k = gl.arange(0, 64, gl.SliceLayout(0, LA))
        block = sorted_routes_ptr + row0
        # Rows past the block's routes read token 0 and are not stored.
        top = gl.load(block + r, mask=r < rows, other=0) // TOPK
        bot = gl.load(block + HALF_M + r, mask=r + HALF_M < rows, other=0) // TOPK
        acc_tl, acc_bl, acc_tr, acc_br = _grouped_tile(
            x_ptr,
            w13_ptr
            + expert.to(gl.int64) * (2 * INTERMEDIATE * HIDDEN)
            + (pid_n * 128) * HIDDEN,
            top[:, None] * HIDDEN + k[None, :],
            bot[:, None] * HIDDEN + k[None, :],
            INTERMEDIATE,
            HIDDEN,
            BLOCK_M,
        )
        SL: gl.constexpr = _store_layout(HALF_M)
        sr = gl.arange(0, HALF_M, gl.SliceLayout(1, SL))
        sc = pid_n * 128 + gl.arange(0, 128, gl.SliceLayout(0, SL))
        for half in gl.static_range(2):
            if half == 0:
                gate = acc_tl
                up = acc_tr
            else:
                gate = acc_bl
                up = acc_br
            h = gate * (1.0 / (1.0 + gl.exp(-gate))) * up
            h = gl.convert_layout(h.to(inter_ptr.dtype.element_ty), SL)
            live = sr + half * HALF_M < rows
            route = gl.load(block + half * HALF_M + sr, mask=live, other=0)
            cdna4.buffer_store(
                h,
                inter_ptr,
                route[:, None] * INTERMEDIATE + sc[None, :],
                mask=live[:, None] & (sc < INTERMEDIATE)[None, :],
            )


@gluon.jit(launch_metadata=_prefill_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_prefill_stage2_gfx950(
    inter_ptr,
    w2_ptr,
    out_ptr,
    sorted_routes_ptr,
    tile_expert_ptr,
    tile_row0_ptr,
    tile_rows_ptr,
    num_tiles_ptr,
    num_tokens,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    BLOCK_M: gl.constexpr,
):
    """Down GEMM of one ``BLOCK_M`` route block x 256 output columns."""
    HALF_M: gl.constexpr = BLOCK_M // 2
    N_TILES: gl.constexpr = HIDDEN // 256
    pid_n, expert, row0, rows = _tile_ids(
        tile_expert_ptr, tile_row0_ptr, tile_rows_ptr, num_tiles_ptr, N_TILES
    )
    if expert >= 0:
        LA: gl.constexpr = _load_layout_a(HALF_M)
        r = gl.arange(0, HALF_M, gl.SliceLayout(1, LA))
        k = gl.arange(0, 64, gl.SliceLayout(0, LA))
        block = sorted_routes_ptr + row0
        top = gl.load(block + r, mask=r < rows, other=0)
        bot = gl.load(block + HALF_M + r, mask=r + HALF_M < rows, other=0)
        acc_tl, acc_bl, acc_tr, acc_br = _grouped_tile(
            inter_ptr,
            w2_ptr
            + expert.to(gl.int64) * (HIDDEN * INTERMEDIATE)
            + (pid_n * 256) * INTERMEDIATE,
            top[:, None] * INTERMEDIATE + k[None, :],
            bot[:, None] * INTERMEDIATE + k[None, :],
            128,
            INTERMEDIATE,
            BLOCK_M,
        )
        SL: gl.constexpr = _store_layout(HALF_M)
        sr = gl.arange(0, HALF_M, gl.SliceLayout(1, SL))
        sc = gl.arange(0, 128, gl.SliceLayout(0, SL))
        for quad in gl.static_range(4):
            if quad == 0:
                acc = acc_tl
            elif quad == 1:
                acc = acc_bl
            elif quad == 2:
                acc = acc_tr
            else:
                acc = acc_br
            value = gl.convert_layout(acc.to(out_ptr.dtype.element_ty), SL)
            live = sr + (quad % 2) * HALF_M < rows
            route = gl.load(block + (quad % 2) * HALF_M + sr, mask=live, other=0)
            col = pid_n * 256 + (quad // 2) * 128 + sc
            cdna4.buffer_store(
                value,
                out_ptr,
                route[:, None] * HIDDEN + col[None, :],
                mask=live[:, None] & (col < HIDDEN)[None, :],
            )


@gluon.jit(launch_metadata=_prefill_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_zero_gfx950(
    ids_ptr,
    out_ptr,
    num_tokens,
    expert_start,
    HIDDEN: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_R: gl.constexpr,
):
    """Zero the output rows of routes whose expert is not local.

    One program checks ``BLOCK_R`` routes and writes only when one of them is
    absent, so a rank that owns every selected expert reads just the ids.
    """
    L: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [gl.num_warps(), 1], [1, 0])
    r = gl.program_id(0) * BLOCK_R + gl.arange(0, BLOCK_R, gl.SliceLayout(1, L))
    valid = r < num_tokens * TOPK
    local = gl.load(ids_ptr + r, mask=valid, other=expert_start) - expert_start
    absent = valid & ((local < 0) | (local >= NUM_EXPERTS))
    if gl.max(absent.to(gl.int32), axis=0) > 0:
        c = gl.arange(0, 512, gl.SliceLayout(0, L))
        zeros = gl.zeros((BLOCK_R, 512), out_ptr.dtype.element_ty, L)
        for chunk in gl.static_range(HIDDEN // 512):
            cdna4.buffer_store(
                zeros,
                out_ptr,
                r[:, None] * HIDDEN + chunk * 512 + c[None, :],
                mask=absent[:, None] & (c < 512)[None, :],
            )


def launch_gluon_bf16_moe_routes_prefill_gfx950(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write unweighted BF16 expert routes into ``out`` with grouped GEMMs.

    Args mirror ``launch_gluon_bf16_moe_routes_decode_gfx950``; ``out`` rows
    of routes whose expert is not local are left untouched. Each stage runs
    once over the full 256-row tiles and once over the 128-row remainder
    tiles.
    """
    m, hidden = x.shape
    experts, two_i, _ = w13.shape
    inter_size = two_i // 2
    top_k = topk_ids.shape[1]
    tiles = align_bf16_moe_routes_gfx950(topk_ids, experts, expert_start, BIG_TILE)
    lists = (
        (BIG_TILE, tiles.big_expert, tiles.big_row0, tiles.big_rows, 0),
        (BIG_TILE // 2, tiles.small_expert, tiles.small_row0, tiles.small_rows, 1),
    )
    inter = torch.empty((m * top_k, inter_size), dtype=x.dtype, device=x.device)
    # Accumulators stay in VGPRs, as in the dense prefill GEMM.
    attrs = (("amdgpu-agpr-alloc", "0,0"),)
    for block_m, expert, row0, rows, index in lists:
        gluon_bf16_moe_routes_prefill_stage1_gfx950[
            (expert.shape[0] * (inter_size // 128),)
        ](
            x,
            w13,
            inter,
            tiles.sorted_routes,
            expert,
            row0,
            rows,
            tiles.num_tiles[index:],
            m,
            HIDDEN=hidden,
            INTERMEDIATE=inter_size,
            TOPK=top_k,
            BLOCK_M=block_m,
            num_warps=NUM_WARPS,
            llvm_fn_attrs=attrs,
        )
    for block_m, expert, row0, rows, index in lists:
        gluon_bf16_moe_routes_prefill_stage2_gfx950[
            (expert.shape[0] * (hidden // BLOCK_N),)
        ](
            inter,
            w2,
            out,
            tiles.sorted_routes,
            expert,
            row0,
            rows,
            tiles.num_tiles[index:],
            m,
            HIDDEN=hidden,
            INTERMEDIATE=inter_size,
            TOPK=top_k,
            BLOCK_M=block_m,
            num_warps=NUM_WARPS,
            llvm_fn_attrs=attrs,
        )
    return out
