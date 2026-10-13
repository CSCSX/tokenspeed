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

"""One process-wide Iris heap; operations retain their own group and buffers."""

import math
import os

import torch
import torch.distributed as dist
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

with redirect_triton_to_tokenspeed_triton():
    import iris

_context = None
_heap_size: int | None = None
_allocator_type: str | None = None


def configure_iris(*, heap_size: int | None, allocator_type: str | None) -> None:
    """Set heap options before preparation; None retains Iris's allocator default.

    The allocator name is passed unchanged to Iris. Its IRIS_ALLOCATOR environment
    override retains upstream precedence. Neither option may change a live heap.
    """
    global _heap_size, _allocator_type
    if _context is not None and (heap_size, allocator_type) != (
        _heap_size,
        _allocator_type,
    ):
        raise RuntimeError("Configure Iris before preparing communication buffers")
    _heap_size = heap_size
    _allocator_type = allocator_type


def allocator_name() -> str:
    return os.environ.get("IRIS_ALLOCATOR", _allocator_type or "torch")


def allocation_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    """Account for each allocation's alignment in the selected Iris allocator."""
    alignment = 1024
    if allocator_name() == "vmem":
        from iris.host.platform.hip import get_allocation_granularity

        alignment = max(
            alignment, get_allocation_granularity(torch.cuda.current_device())
        )
    size = max(1, math.prod(shape)) * dtype.itemsize
    return (size + alignment - 1) // alignment * alignment


def get_or_create_context(*, heap_size: int):
    """Create the shared context at its initial capacity, or return it unchanged.

    Configured heap capacity takes precedence over the caller's initial size.
    Iris enforces capacity as states allocate their buffers.
    """
    global _context
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare Iris communication buffers before graph capture")
    if _context is None:
        size = _heap_size if _heap_size is not None else heap_size
        options = {} if _allocator_type is None else {"allocator_type": _allocator_type}
        _context = iris.iris(heap_size=size, **options)
        # Iris allocates its shared barrier state on the first barrier.
        _context.device_barrier()
    return _context


class IrisGroup:
    """Retain the context and the ordered heap addresses for one process group."""

    def __init__(self, ctx, group: dist.ProcessGroup, rank: int, device: torch.device):
        self.ctx = ctx
        self.group = group
        self.rank = rank
        self.device = device
        self.world_size = group.size()
        ranks = dist.get_process_group_ranks(group)
        assert ranks[rank] == dist.get_rank()
        self.bases = ctx.get_heap_bases()[ranks].contiguous()
        addresses = [int(address) for address in self.bases.tolist()]
        self.addresses = tuple(addresses + [addresses[-1]] * (8 - len(addresses)))

    def finish_prepare(self) -> None:
        """Publish initialized buffers before any peer can use them."""
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
