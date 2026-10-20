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

"""Host checks for caller-owned canonical route destinations."""

import torch


def validate_deferred_output(
    output: torch.Tensor,
    *,
    tokens: int,
    topk: int,
    width: int,
    device: torch.device,
    do_finalize: bool,
    inputs: tuple[torch.Tensor | None, ...],
) -> None:
    if do_finalize:
        raise ValueError("deferred output requires do_finalize=False")
    if (
        output.shape != (tokens * topk, width)
        or output.dtype != torch.bfloat16
        or output.device != device
        or not output.is_contiguous()
    ):
        raise ValueError("deferred output must be contiguous colocated BF16 routes")
    if not output.numel():
        return
    start = output.data_ptr()
    end = start + output.numel() * output.element_size()
    for tensor in inputs:
        if tensor is None or not tensor.numel() or tensor.device != device:
            continue
        # Disjoint views of one symmetric heap are valid. Include the full
        # addressed span of strided inputs, including gaps between rows.
        offsets = [
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        ]
        first = (
            tensor.data_ptr()
            + sum(min(0, offset) for offset in offsets) * tensor.element_size()
        )
        last = (
            tensor.data_ptr()
            + (1 + sum(max(0, offset) for offset in offsets)) * tensor.element_size()
        )
        if first < end and start < last:
            raise ValueError("deferred output must not overlap an input")
