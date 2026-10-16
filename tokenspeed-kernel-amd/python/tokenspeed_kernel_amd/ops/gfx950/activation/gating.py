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

"""One-launch FP32 bias, sigmoid and attention-output product for gfx950."""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


def _launch_metadata(grid, kernel, args):
    count = args["N"]
    return {
        "name": kernel.name,
        "bytes": count
        * (2 * args["x_ptr"].element_size() + args["gate_ptr"].element_size()),
        "flops32": count * 6,
    }


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("N", "stride_xr", "stride_xh", "stride_gr", "stride_gh"),
)
def gluon_attention_output_gate_gfx950(
    x_ptr,
    gate_ptr,
    bias_ptr,
    out_ptr,
    N,
    stride_xr,
    stride_xh,
    stride_gr,
    stride_gh,
    HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    index = gl.program_id(0).to(gl.int64) * BLOCK + gl.arange(0, BLOCK, layout)
    row = index // (HEADS * WIDTH)
    head = index // WIDTH % HEADS
    column = index % WIDTH
    valid = index < N
    x = gl.load(x_ptr + row * stride_xr + head * stride_xh + column, valid, other=0).to(
        gl.float32
    )
    gate = gl.load(
        gate_ptr + row * stride_gr + head * stride_gh + column, valid, other=0
    ).to(gl.float32)
    bias = gl.load(bias_ptr + head, valid, other=0).to(gl.float32)
    logits = gate + bias
    # Use the eager FP32 sigmoid, not exp2 or an approximate reciprocal.
    sigmoid = gl.div_rn(1.0, 1.0 + gl.extra.libdevice.exp(-logits))
    gl.store(out_ptr + index, x * sigmoid, valid)


def launch_gluon_attention_output_gate_gfx950(
    output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
) -> torch.Tensor:
    """Return output * sigmoid(gate + bias), rounding only on the final store.

    Inputs are [rows, heads, width] with unit width stride and a contiguous
    per-head bias. Row and head strides may differ between the two inputs.
    BF16, FP16 and FP32 inputs may be mixed; output keeps the first input's
    dtype. Neither inputs nor their views are mutated.
    """
    assert output.ndim == 3 and gate.shape == output.shape
    assert output.stride(-1) == gate.stride(-1) == 1
    assert gate_bias.shape == (output.shape[1],) and gate_bias.is_contiguous()
    assert output.is_cuda and output.device == gate.device == gate_bias.device
    assert all(
        x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        for x in (output, gate, gate_bias)
    )
    result = torch.empty(output.shape, device=output.device, dtype=output.dtype)
    if output.numel():
        gluon_attention_output_gate_gfx950[(triton.cdiv(output.numel(), 256),)](
            output,
            gate,
            gate_bias,
            result,
            output.numel(),
            output.stride(0),
            output.stride(1),
            gate.stride(0),
            gate.stride(1),
            output.shape[1],
            output.shape[2],
            256,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return result
