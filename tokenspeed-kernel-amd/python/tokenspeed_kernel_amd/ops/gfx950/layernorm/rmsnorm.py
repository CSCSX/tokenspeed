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

"""BF16 RMSNorm with rounded residual products and optional token-FP8 output."""

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton


def _launch_metadata(grid, kernel, args):
    rows, width = args["x_ptr"].shape[0], args["WIDTH"]
    elements = (4 if args["HAS_RESIDUAL"] else 2) * rows * width + width
    return {
        "name": kernel.name,
        "bytes": 2 * elements + (rows * (width + 4) if args["QUANTIZE"] else 0),
        "flops32": rows * width * (9 if args["HAS_RESIDUAL"] else 5),
    }


@gluon.jit
def _mul_f32(value, scale):
    # Keep the FP32 product's rounding before the eager-equivalent BF16 cast.
    return gl.inline_asm_elementwise(
        "v_mul_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[value, scale],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("stride_x", "stride_residual", "x_scale", "residual_scale"),
)
def gluon_rmsnorm_bf16_gfx950(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    sum_ptr,
    codes_ptr,
    scales_ptr,
    stride_x,
    stride_residual,
    x_scale,
    residual_scale,
    WIDTH: gl.constexpr,
    EPS: gl.constexpr,
    BLOCK: gl.constexpr,
    HAS_RESIDUAL: gl.constexpr,
    QUANTIZE: gl.constexpr,
):
    # Match the existing portable norm's FP32 reduction order, including when
    # token-FP8 stores could otherwise encourage a different vector layout.
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [4], [0])
    row = gl.program_id(0).to(gl.int64)
    column = gl.arange(0, BLOCK, layout)
    valid = column < WIDTH
    stride_x = gl.multiple_of(stride_x, 8)
    stride_residual = gl.multiple_of(stride_residual, 8)
    value = gl.load(x_ptr + row * stride_x + column, valid, other=0).to(gl.float32)
    if HAS_RESIDUAL:
        residual = gl.load(
            residual_ptr + row * stride_residual + column, valid, other=0
        ).to(gl.float32)
        value = _mul_f32(value, x_scale).to(gl.bfloat16).to(gl.float32)
        residual = _mul_f32(residual, residual_scale).to(gl.bfloat16).to(gl.float32)
        value = (value + residual).to(gl.bfloat16).to(gl.float32)
        gl.store(sum_ptr + row * WIDTH + column, value, valid)
    variance = gl.sum(value * value, axis=0) / WIDTH
    normalized = value * gl.rsqrt(variance + EPS)
    weight = gl.load(weight_ptr + column, valid, other=0).to(gl.float32)
    output = (normalized * weight).to(gl.bfloat16)
    gl.store(output_ptr + row * WIDTH + column, output, valid)
    if QUANTIZE:
        # Quantize the stored BF16 row, not its unrounded FP32 precursor.
        stored = output.to(gl.float32)
        amax = gl.max(gl.abs(stored), axis=0)
        multiplier = gl.minimum(gl.div_rn(448.0, amax), 448.0 * 512.0)
        scale = gl.maximum(gl.div_rn(amax, 448.0), 1.0 / (448.0 * 512.0))
        codes = gl.clamp(stored * multiplier, -448.0, 448.0).to(
            gl.float8e4nv, fp_downcast_rounding="rtne"
        )
        gl.store(codes_ptr + row * WIDTH + column, codes, valid)
        gl.store(scales_ptr + row, scale)


def launch_gluon_rmsnorm_bf16_gfx950(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    residual: torch.Tensor | None,
    output: torch.Tensor,
    residual_output: torch.Tensor | None,
    x_scale: float,
    residual_scale: float,
    token_fp8_out: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    """Write BF16 norm/residual rows and optional per-token E4M3 codes/scales.

    ``x`` and optional ``residual`` are BF16 [M,5120] with contiguous columns,
    disjoint rows and eight-element-aligned row strides. ``weight`` is dense
    BF16 [5120]. Outputs are caller-owned, contiguous, on the same device;
    ``residual_output`` is present exactly when ``residual`` is. Both products
    and the sum round to BF16 before normalization. Optional token-FP8 output
    holds E4M3 [M,5120] codes and contiguous FP32 scales with M elements.
    Return None.
    """
    rows, width = x.shape
    assert width == 5120 and x.dtype == weight.dtype == torch.bfloat16
    assert x.is_cuda and x.stride(1) == 1 and x.stride(0) >= width
    assert x.stride(0) % 8 == 0
    assert weight.shape == (width,) and weight.is_contiguous()
    assert (
        output.shape == x.shape and output.dtype == x.dtype and output.is_contiguous()
    )
    assert weight.device == output.device == x.device
    assert (residual is None) == (residual_output is None)
    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype
        assert residual.stride(1) == 1 and residual.stride(0) >= width
        assert residual.stride(0) % 8 == 0
        assert residual_output.shape == x.shape and residual_output.dtype == x.dtype
        assert residual_output.is_contiguous()
        assert residual.device == residual_output.device == x.device
    codes, scales = (None, None) if token_fp8_out is None else token_fp8_out
    if token_fp8_out is not None:
        assert codes.shape == x.shape and codes.dtype == torch.float8_e4m3fn
        assert scales.numel() == rows and scales.dtype == torch.float32
        assert codes.is_contiguous() and scales.is_contiguous()
        assert codes.device == scales.device == x.device
    if rows:
        gluon_rmsnorm_bf16_gfx950[(rows,)](
            x,
            x if residual is None else residual,
            weight,
            output,
            output if residual_output is None else residual_output,
            output if codes is None else codes,
            output if scales is None else scales,
            x.stride(0),
            0 if residual is None else residual.stride(0),
            x_scale,
            residual_scale,
            width,
            eps,
            triton.next_power_of_2(width),
            residual is not None,
            token_fp8_out is not None,
            num_warps=4,
            enable_fp_fusion=True,
        )
