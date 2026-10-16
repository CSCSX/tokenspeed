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

"""Portable FP32 activation reference implementations."""

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "activation",
    "attention_output_gate",
    name="torch_attention_output_gate",
    solution="torch",
    signatures=format_signatures(
        "output", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    traits={},
    priority=Priority.PORTABLE,
)
def torch_attention_output_gate(
    output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
    floor: float,
    temperature: float,
) -> torch.Tensor:
    """Evaluate the biased sigmoid, floor, temperature and product in FP32."""
    bias_shape = (1,) * (gate.ndim - 2) + (gate.shape[-2], 1)
    logits = (gate.float() + gate_bias.float().reshape(bias_shape)) / float(temperature)
    multiplier = float(floor) + (1.0 - float(floor)) * torch.sigmoid(logits)
    return (output.float() * multiplier).to(output.dtype)
