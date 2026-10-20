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

"""Stable, atomic-free grouping of local expert routes into GEMM tiles (gfx950).

Packs the route indices ``token * top_k + slot`` of every local expert
contiguously, in route order, and cuts each expert's run into row tiles of
``BIG`` rows, with one ``SMALL`` (``BIG / 2``) tile for a remainder that fits
it, so remainders waste at most half as much MFMA work. Three launches:

  count:   one program per ``CHUNK`` routes stores its per-expert histogram.
  offsets: one program prefix-sums the histograms over chunks (each chunk's
           first slot within its expert) and over experts (each expert's
           first sorted row), and writes both tile lists: each tile's expert,
           first sorted row and valid row count.
  scatter: one program per chunk ranks each route among the chunk's earlier
           routes of the same expert and stores it at its sorted position.

All integers that vary per batch are runtime values.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

__all__ = ["CHUNK", "RouteTiles", "align_bf16_moe_routes_gfx950"]

CHUNK = 256
_RANK_STEP = 64


def _align_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (grid={grid[0]})"}


# Route and chunk counts follow the batch; one binary serves every count.
@gluon.jit(launch_metadata=_align_launch_metadata, do_not_specialize=("num_routes",))
def gluon_bf16_moe_routes_count_gfx950(
    ids_ptr,
    counts_ptr,
    num_routes,
    expert_start,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_E: gl.constexpr,
    CHUNK: gl.constexpr,
):
    """Per-expert route counts of chunk ``pid`` into ``counts[pid, :]``."""
    L: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    pid = gl.program_id(0)
    r = pid * CHUNK + gl.arange(0, CHUNK, L)
    valid = r < num_routes
    local = gl.load(ids_ptr + r, mask=valid, other=-1) - expert_start
    owned = valid & (local >= 0) & (local < NUM_EXPERTS)
    counts = gl.histogram(gl.where(owned, local, 0), BLOCK_E, mask=owned, layout=L)
    gl.store(counts_ptr + pid * BLOCK_E + gl.arange(0, BLOCK_E, L), counts)


@gluon.jit
def _exclusive_cumsum(values, layout: gl.constexpr):
    # Scan in a plain 1-D layout (the scan lowering rejects slice layouts).
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    flat = gl.convert_layout(values, L1)
    return gl.convert_layout(gl.cumsum(flat, axis=0) - flat, layout)


@gluon.jit(launch_metadata=_align_launch_metadata, do_not_specialize=("num_chunks",))
def gluon_bf16_moe_routes_offsets_gfx950(
    counts_ptr,
    chunk_base_ptr,
    expert_offset_ptr,
    big_expert_ptr,
    big_row0_ptr,
    big_rows_ptr,
    small_expert_ptr,
    small_row0_ptr,
    small_rows_ptr,
    num_tiles_ptr,
    num_chunks,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BIG: gl.constexpr,
):
    """Chunk and expert prefix sums, and both tile lists."""
    SMALL: gl.constexpr = BIG // 2
    TILE: gl.constexpr = 16
    # A thread holds a whole TILE column, so the chunk scan stays in registers.
    LT: gl.constexpr = gl.BlockedLayout([TILE, 1], [1, 64], [1, gl.num_warps()], [0, 1])
    LE: gl.constexpr = gl.SliceLayout(0, LT)
    LR: gl.constexpr = gl.SliceLayout(1, LT)
    e = gl.arange(0, BLOCK_E, LE)
    own = e < NUM_EXPERTS
    totals = gl.zeros((BLOCK_E,), gl.int32, LE)
    for c0 in range(0, num_chunks, TILE):
        c = c0 + gl.arange(0, TILE, LR)
        offs = c[:, None] * BLOCK_E + e[None, :]
        mask = (c < num_chunks)[:, None] & own[None, :]
        tile = gl.load(counts_ptr + offs, mask=mask, other=0)
        gl.store(
            chunk_base_ptr + offs,
            gl.cumsum(tile, axis=0) - tile + totals[None, :],
            mask=mask,
        )
        totals += gl.sum(tile, axis=0)
    first_row = _exclusive_cumsum(totals, LE)
    gl.store(expert_offset_ptr + e, first_row, mask=own)
    full = totals // BIG
    rem = totals - full * BIG
    big = full + (rem > SMALL).to(gl.int32)
    small = ((rem > 0) & (rem <= SMALL)).to(gl.int32)
    first_big = _exclusive_cumsum(big, LE)
    first_small = _exclusive_cumsum(small, LE)
    gl.store(num_tiles_ptr, gl.sum(big, axis=0))
    gl.store(num_tiles_ptr + 1, gl.sum(small, axis=0))
    # Each expert labels its own tiles, one per expert per step.
    for k in range(0, gl.max(big, axis=0)):
        live = own & (k < big)
        gl.store(big_expert_ptr + first_big + k, e, mask=live)
        gl.store(big_row0_ptr + first_big + k, first_row + k * BIG, mask=live)
        gl.store(
            big_rows_ptr + first_big + k,
            gl.minimum(totals - k * BIG, BIG),
            mask=live,
        )
    live = own & (small > 0)
    gl.store(small_expert_ptr + first_small, e, mask=live)
    gl.store(small_row0_ptr + first_small, first_row + full * BIG, mask=live)
    gl.store(small_rows_ptr + first_small, rem, mask=live)


@gluon.jit(launch_metadata=_align_launch_metadata, do_not_specialize=("num_routes",))
def gluon_bf16_moe_routes_scatter_gfx950(
    ids_ptr,
    chunk_base_ptr,
    expert_offset_ptr,
    sorted_routes_ptr,
    num_routes,
    expert_start,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_E: gl.constexpr,
    CHUNK: gl.constexpr,
    RANK_STEP: gl.constexpr,
):
    """Store each owned route of chunk ``pid`` at its stable sorted position."""
    LS: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [gl.num_warps(), 1], [0, 1])
    LI: gl.constexpr = gl.SliceLayout(1, LS)
    LJ: gl.constexpr = gl.SliceLayout(0, LS)
    pid = gl.program_id(0)
    i = pid * CHUNK + gl.arange(0, CHUNK, LI)
    valid = i < num_routes
    local = gl.load(ids_ptr + i, mask=valid, other=-1) - expert_start
    owned = valid & (local >= 0) & (local < NUM_EXPERTS)
    rank = gl.zeros((CHUNK,), gl.int32, LI)
    for j0 in range(0, CHUNK, RANK_STEP):
        j = pid * CHUNK + j0 + gl.arange(0, RANK_STEP, LJ)
        other = gl.load(ids_ptr + j, mask=j < num_routes, other=-1) - expert_start
        earlier = (other[None, :] == local[:, None]) & (j[None, :] < i[:, None])
        rank += gl.sum(earlier.to(gl.int32), axis=1)
    safe = gl.where(owned, local, 0)
    base = gl.load(chunk_base_ptr + pid * BLOCK_E + safe, mask=owned, other=0)
    first = gl.load(expert_offset_ptr + safe, mask=owned, other=0)
    gl.store(sorted_routes_ptr + first + base + rank, i, mask=owned)


class RouteTiles(NamedTuple):
    """Local routes packed per expert and cut into ``BIG``/``SMALL`` tiles.

    ``sorted_routes`` holds route indices grouped by expert in route order.
    Tile list ``i`` (0 = ``BIG`` rows, 1 = ``SMALL`` rows) has
    ``num_tiles[i]`` live tiles; tile ``t`` covers ``rows[t]`` sorted rows
    from ``row0[t]``, all routed to local expert ``expert[t]``.
    """

    sorted_routes: torch.Tensor
    big_expert: torch.Tensor
    big_row0: torch.Tensor
    big_rows: torch.Tensor
    small_expert: torch.Tensor
    small_row0: torch.Tensor
    small_rows: torch.Tensor
    num_tiles: torch.Tensor


def align_bf16_moe_routes_gfx950(
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_start: int,
    big: int,
) -> RouteTiles:
    """Group the local routes of ``topk_ids`` into ``big`` and ``big / 2`` tiles.

    Args:
        topk_ids: Contiguous int32 ``[tokens, top_k]`` global expert ids.
        num_experts: Local expert count.
        expert_start: Global id of local expert 0.
        big: Rows of a full tile.

    Returns:
        The tile lists; see :class:`RouteTiles`.
    """
    device = topk_ids.device
    routes = topk_ids.numel()
    chunks = triton.cdiv(routes, CHUNK)
    max_big = triton.cdiv(routes, big) + num_experts
    block_e = triton.next_power_of_2(num_experts)

    def ints(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    counts = ints(chunks, block_e)
    chunk_base = ints(chunks, block_e)
    expert_offset = ints(block_e)
    tiles = RouteTiles(
        ints(routes),
        ints(max_big),
        ints(max_big),
        ints(max_big),
        ints(num_experts),
        ints(num_experts),
        ints(num_experts),
        ints(2),
    )
    gluon_bf16_moe_routes_count_gfx950[(chunks,)](
        topk_ids,
        counts,
        routes,
        expert_start,
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        CHUNK=CHUNK,
        num_warps=4,
    )
    gluon_bf16_moe_routes_offsets_gfx950[(1,)](
        counts,
        chunk_base,
        expert_offset,
        tiles.big_expert,
        tiles.big_row0,
        tiles.big_rows,
        tiles.small_expert,
        tiles.small_row0,
        tiles.small_rows,
        tiles.num_tiles,
        chunks,
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        BIG=big,
        num_warps=4,
    )
    gluon_bf16_moe_routes_scatter_gfx950[(chunks,)](
        topk_ids,
        chunk_base,
        expert_offset,
        tiles.sorted_routes,
        routes,
        expert_start,
        NUM_EXPERTS=num_experts,
        BLOCK_E=block_e,
        CHUNK=CHUNK,
        RANK_STEP=_RANK_STEP,
        num_warps=4,
    )
    return tiles
