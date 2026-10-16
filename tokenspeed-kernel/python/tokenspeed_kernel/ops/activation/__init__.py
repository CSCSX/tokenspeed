# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Activation kernel entry points."""

from __future__ import annotations

import tokenspeed_kernel.ops.activation.gluon  # noqa: F401
import torch as _torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.activation.torch import torch_attention_output_gate
from tokenspeed_kernel.ops.activation.triton import (
    add3,
    relu2,
)
from tokenspeed_kernel.ops.activation.triton import silu_and_mul as triton_silu_and_mul
from tokenspeed_kernel.ops.activation.triton import situ_and_mul as triton_situ_and_mul
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import error_fn
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def silu_and_mul(
    x: _torch.Tensor,
    out: _torch.Tensor | None = None,
    limit: float | None = None,
) -> _torch.Tensor:
    """Apply SwiGLU through the platform implementation.

    Positive ``limit`` values use the portable Triton implementation because
    the CUDA implementation does not expose the checkpoint's clamp semantics.

    """
    if current_platform().is_npu and limit is None:
        from tokenspeed_kernel.ops.activation.ascend import silu_and_mul as npu_swiglu

        return npu_swiglu(x, out)
    if (
        limit is not None
        or current_platform().is_amd
        or flashinfer_silu_and_mul is error_fn
    ):
        return triton_silu_and_mul(x, out, limit=limit)
    return flashinfer_silu_and_mul(x, out, enable_pdl=pdl_enabled())


def situ_and_mul(
    x: _torch.Tensor,
    out: _torch.Tensor | None = None,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
) -> _torch.Tensor:
    """Apply SiTU through the portable Triton implementation."""

    return triton_situ_and_mul(
        x,
        out,
        beta=beta,
        linear_beta=linear_beta,
    )


_FLOAT_DTYPES = (_torch.bfloat16, _torch.float16, _torch.float32)


def attention_output_gate(
    output: _torch.Tensor,
    gate: _torch.Tensor,
    gate_bias: _torch.Tensor,
    floor: float,
    temperature: float,
) -> _torch.Tensor:
    """Gate attention outputs, evaluating the expression in FP32.

    Args:
        output: Attention output shaped [..., heads, head_dim].
        gate: Projected gate logits with the same shape as output.
        gate_bias: One learned scalar per head, broadcast over head_dim.
        floor: Lower bound in floor + (1 - floor) * sigmoid(logits).
        temperature: Positive divisor after adding the bias to the logits.

    Returns:
        A new tensor with output's shape and dtype. Inputs are not modified.
    """
    if gate.ndim < 2 or output.shape != gate.shape:
        raise ValueError("attention output and gate must share head and width axes")
    if gate_bias.ndim != 1 or gate_bias.shape[0] != gate.shape[-2]:
        raise ValueError("gate_bias must contain one scalar per attention head")
    if temperature <= 0:
        raise ValueError(
            f"attention gate temperature must be positive, got {temperature}"
        )
    if output.dtype not in _FLOAT_DTYPES:
        return torch_attention_output_gate(output, gate, gate_bias, floor, temperature)
    layout_supported = (
        output.ndim == 3
        and output.is_cuda
        and gate.device == output.device == gate_bias.device
        and output.stride(-1) == gate.stride(-1) == 1
        and gate_bias.is_contiguous()
        and gate.dtype in _FLOAT_DTYPES
        and gate_bias.dtype in _FLOAT_DTYPES
    )
    kernel = select_kernel(
        "activation",
        "attention_output_gate",
        format_signature(output=dense_tensor_format(output.dtype)),
        traits={
            "layout_supported": layout_supported,
            "zero_floor": floor == 0.0,
            "unit_temperature": temperature == 1.0,
        },
    )
    return kernel(output, gate, gate_bias, floor, temperature)


__all__ = [
    "add3",
    "attention_output_gate",
    "relu2",
    "silu_and_mul",
    "situ_and_mul",
]
