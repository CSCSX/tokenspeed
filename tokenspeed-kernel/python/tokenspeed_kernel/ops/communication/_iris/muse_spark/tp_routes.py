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

"""Complete expert TP routes on their token owners before route normalization."""

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.reduce_scatter import (
    iris_pull_reduce_scatter,
)
from tokenspeed_kernel.ops.moe import gather_moe_routes

_TOPK = 8
_WIDTH = 4096
_PROGRAMS = 24


class MuseTPRouteReduceScatter:
    """Own route staging and borrow complete local routes until the next call.

    Staging is independent of the smaller attention/EP producer buffer. The
    exit barrier protects peer reads; the local norm must consume the borrowed
    output on the same stream before another route reduction overwrites it.
    """

    @staticmethod
    def heap_bytes(max_rows: int, world_size: int) -> int:
        return allocation_bytes((max_rows, _TOPK, _WIDTH), torch.bfloat16) + (
            allocation_bytes((_PROGRAMS, world_size), torch.uint32)
        )

    def __init__(self, peers: IrisGroup, max_rows: int):
        self.peers = peers
        self.max_rows = max_rows
        self.input = peers.ctx.empty((max_rows, _TOPK, _WIDTH), dtype=torch.bfloat16)
        self.output = torch.empty(
            (triton.cdiv(max_rows, peers.world_size), _TOPK, _WIDTH),
            dtype=torch.bfloat16,
            device=peers.device,
        )
        self.flags = peers.ctx.zeros((_PROGRAMS, peers.world_size), dtype=torch.uint32)

    def run(
        self, routes: torch.Tensor, route_map: torch.Tensor | None, rows: int
    ) -> torch.Tensor:
        """Return BF16 ``[ceil(rows/P),8,4096]`` routes, with zero padded rows.

        ``routes`` holds this TP rank's partial expert outputs. A supplied
        ``route_map`` maps canonical token/route order into its physical rows;
        without a map the input must already be canonical and contiguous.
        """
        assert 0 < rows <= self.max_rows
        assert routes.dtype == torch.bfloat16 and routes.device == self.peers.device
        staged = self.input[:rows]
        if route_map is None:
            assert routes.is_contiguous() and routes.numel() == staged.numel()
            staged.copy_(routes.view_as(staged))
        else:
            gather_moe_routes(routes, route_map, rows, _TOPK, _WIDTH, output=staged)
        output = self.output[: triton.cdiv(rows, self.peers.world_size)]
        # Partition whole tokens: all eight routes must reach the same owner.
        iris_pull_reduce_scatter[(_PROGRAMS,)](
            staged,
            output,
            self.flags,
            self.peers.addresses[: self.peers.world_size],
            ROWS=rows,
            RANK=self.peers.rank,
            WIDTH=_TOPK * _WIDTH,
            BLOCK=2048,
            PROGRAMS=_PROGRAMS,
            NUM_WARPS=4,
            EXIT_BARRIER=True,
            PRODUCER_WRITES_THROUGH=False,
            num_warps=4,
        )
        return output
