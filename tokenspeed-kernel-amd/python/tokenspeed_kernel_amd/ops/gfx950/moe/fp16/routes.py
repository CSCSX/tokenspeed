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

"""Unweighted BF16 expert routes on gfx950: decode or grouped prefill GEMMs."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.routes_decode import (
    MAX_TOKENS,
    launch_gluon_bf16_moe_routes_decode_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.routes_prefill import (
    gluon_bf16_moe_routes_zero_gfx950,
    launch_gluon_bf16_moe_routes_prefill_gfx950,
)

__all__ = [
    "launch_gluon_bf16_moe_routes_gfx950",
    "supports_gluon_bf16_moe_routes_gfx950",
]

_ZERO_BLOCK_ROUTES = 16


def supports_gluon_bf16_moe_routes_gfx950(
    hidden: int, intermediate: int, top_k: int
) -> bool:
    """Whether the decode and prefill tiles divide this expert geometry."""
    return hidden % 512 == 0 and intermediate % 256 == 0 and top_k in (1, 2, 4, 8, 16)


def _decode_config(hidden: int, intermediate: int) -> tuple[int, int, int, int]:
    """``(stage1 BLOCK_N, BLOCK_K, stage2 BLOCK_N, BLOCK_K)``.

    Cold-weight sweeps on MI355X at Muse's 256 x I512 (TP8/EP1) and 32 x I4096
    (TP8/EP8) expert shards: stage 1 is flat across 16/32-column tiles; stage 2
    wants 32 output columns for a 512-wide reduction and 16 for a long one,
    where more programs keep more weight streams in flight.
    """
    del hidden
    s2_n = 16 if intermediate >= 2048 else 32
    s2_k = 128 if intermediate % 512 == 0 else 64
    return 32, 64, s2_n, s2_k


def launch_gluon_bf16_moe_routes_gfx950(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write unweighted BF16 SwiGLU expert routes into ``out``.

    Args:
        x: Contiguous BF16 ``[tokens, H]``.
        w13: Contiguous ``[E, 2I, H]`` gate/up weights, gate rows first.
        w2: Contiguous ``[E, H, I]`` down weights.
        topk_ids: Contiguous int32 ``[tokens, top_k]`` global expert ids.
        expert_start: Global id of local expert 0.
        out: Contiguous BF16 ``[tokens * top_k, H]``; every route row is
            written, with zeros for experts outside the local range.

    Returns:
        ``out``.
    """
    tokens, hidden = x.shape
    experts, two_i, _ = w13.shape
    if tokens <= MAX_TOKENS:
        return launch_gluon_bf16_moe_routes_decode_gfx950(
            x,
            w13,
            w2,
            topk_ids,
            expert_start,
            out,
            _decode_config(hidden, two_i // 2),
        )
    top_k = topk_ids.shape[1]
    routes = tokens * top_k
    gluon_bf16_moe_routes_zero_gfx950[(triton.cdiv(routes, _ZERO_BLOCK_ROUTES),)](
        topk_ids,
        out,
        tokens,
        expert_start,
        HIDDEN=hidden,
        TOPK=top_k,
        NUM_EXPERTS=experts,
        BLOCK_R=_ZERO_BLOCK_ROUTES,
        num_warps=4,
    )
    return launch_gluon_bf16_moe_routes_prefill_gfx950(
        x,
        w13,
        w2,
        topk_ids,
        expert_start,
        out,
    )
