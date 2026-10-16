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

"""Exact FP32 gating, non-mutating views, and graph replay on gfx950."""

import pytest
import torch
from tokenspeed_kernel.ops.activation import attention_output_gate
from tokenspeed_kernel_amd.ops.gfx950.activation.gating import (
    gluon_attention_output_gate_gfx950,
    launch_gluon_attention_output_gate_gfx950,
)
from utils import assert_no_triton_compile, is_cdna4

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="requires AMD gfx950")


def _reference(output, gate, bias):
    return (
        output.float() * torch.sigmoid(gate.float() + bias.float()[None, :, None])
    ).to(output.dtype)


def _assert_bits(actual, expected):
    assert torch.equal(actual.isnan(), expected.isnan())
    keep = ~expected.isnan()
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(integer)[keep], expected.view(integer)[keep])


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@pytest.mark.parametrize("rows", (0, 1, 16, 256, 8192))
def test_fused_gate_matches_fp32_chain_without_changing_inputs(dtype, rows):
    torch.manual_seed(23)
    output = torch.randn(rows, 6, 256, device="cuda", dtype=dtype)[..., :128]
    gate = torch.randn(rows, 6, 256, device="cuda", dtype=dtype)[..., 128:]
    bias = torch.randn(6, device="cuda", dtype=dtype)
    saved = [x.clone() for x in (output, gate, bias)]
    actual = attention_output_gate(output, gate, bias, 0.0, 1.0)
    _assert_bits(actual, _reference(output, gate, bias))
    for tensor, original in zip((output, gate, bias), saved):
        _assert_bits(tensor, original)
    assert actual.is_contiguous()
    if rows:
        assert actual.data_ptr() != output.data_ptr()


@pytest.mark.parametrize("bias_value", (0.0, 0.25, -0.75))
def test_every_bf16_gate_code_matches_torch(bias_value):
    codes = torch.arange(65536, device="cuda", dtype=torch.int32)
    gate = codes.to(torch.int16).view(torch.bfloat16).reshape(512, 1, 128)
    output = torch.randn(gate.shape, device="cuda", dtype=gate.dtype)
    bias = torch.tensor([bias_value], device="cuda", dtype=gate.dtype)
    _assert_bits(
        attention_output_gate(output, gate, bias, 0.0, 1.0),
        _reference(output, gate, bias),
    )


def test_mixed_input_dtypes_preserve_the_fp32_expression():
    output = torch.randn(17, 3, 64, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(output.shape, device="cuda", dtype=torch.float32)
    bias = torch.randn(3, device="cuda", dtype=torch.float16)
    _assert_bits(
        attention_output_gate(output, gate, bias, 0.0, 1.0),
        _reference(output, gate, bias),
    )


def test_changed_values_and_runtime_row_strides_replay_without_compiling():
    torch.manual_seed(24)
    output = torch.randn(64, 6, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(64, 6, 256, device="cuda", dtype=output.dtype)[..., 128:]
    bias = torch.randn(6, device="cuda", dtype=output.dtype)

    def run(rows):
        return launch_gluon_attention_output_gate_gfx950(
            output[:rows], gate[:rows], bias
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(17)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run(17)
    output.neg_()
    gate.add_(0.25)
    bias.neg_()
    graph.replay()
    _assert_bits(captured, _reference(output[:17], gate[:17], bias))
    with assert_no_triton_compile(gluon_attention_output_gate_gfx950):
        for rows in (1, 3, 16, 24, 32, 63, 64):
            _assert_bits(run(rows), _reference(output[:rows], gate[:rows], bias))
        # Different runtime row strides keep the already compiled kernel.
        _assert_bits(
            launch_gluon_attention_output_gate_gfx950(output[::2], gate[::2], bias),
            _reference(output[::2], gate[::2], bias),
        )
    graph.reset()


@pytest.mark.parametrize("floor,temperature", ((0.125, 1.0), (0.0, 0.7), (0.125, 0.7)))
def test_unsupported_fusion_parameters_keep_the_portable_expression(floor, temperature):
    output = torch.randn(16, 6, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(output)
    bias = torch.randn(6, device="cuda", dtype=output.dtype)
    multiplier = floor + (1.0 - floor) * torch.sigmoid(
        (gate.float() + bias.float()[None, :, None]) / temperature
    )
    expected = (output.float() * multiplier).to(output.dtype)
    _assert_bits(
        attention_output_gate(output, gate, bias, floor, temperature), expected
    )
