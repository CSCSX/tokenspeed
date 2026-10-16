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

"""Portable attention gating preserves its FP32 equation and validation."""

import pytest
import torch
from tokenspeed_kernel.ops.activation import attention_output_gate


@pytest.mark.parametrize("shape", ((3, 5), (2, 3, 5), (2, 4, 3, 5)))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32, torch.float64))
@pytest.mark.parametrize("floor,temperature", ((0.0, 1.0), (0.125, 0.7)))
def test_cpu_gate_matches_the_expression(shape, dtype, floor, temperature):
    output = torch.randn(shape, dtype=dtype)
    gate = torch.randn_like(output)
    bias = torch.randn(shape[-2], dtype=dtype)
    shape_bias = (1,) * (len(shape) - 2) + (shape[-2], 1)
    multiplier = floor + (1.0 - floor) * torch.sigmoid(
        (gate.float() + bias.float().view(shape_bias)) / temperature
    )
    expected = (output.float() * multiplier).to(dtype)
    actual = attention_output_gate(output, gate, bias, floor, temperature)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.data_ptr() != output.data_ptr()


@pytest.mark.parametrize("temperature", (0.0, -0.5))
def test_gate_rejects_nonpositive_temperature(temperature):
    output = torch.ones(2, 3, 5)
    with pytest.raises(ValueError, match="temperature must be positive"):
        attention_output_gate(output, output, torch.zeros(3), 0.0, temperature)


def test_gate_rejects_incompatible_shapes():
    output = torch.ones(2, 3, 5)
    with pytest.raises(ValueError, match="share head and width axes"):
        attention_output_gate(output, output[0], torch.zeros(3), 0.0, 1.0)
    with pytest.raises(ValueError, match="one scalar per attention head"):
        attention_output_gate(output, output, torch.zeros(4), 0.0, 1.0)
