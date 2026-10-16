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

"""The shared norm APIs preserve BF16 bytes and the token-FP8 recipe."""

import math

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import _rmsnorm_kernel, rmsnorm
from tokenspeed_kernel.ops.quantization import rmsnorm_quantize_fp8
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize_per_token
from tokenspeed_kernel_amd.ops.gfx950.layernorm.rmsnorm import gluon_rmsnorm_bf16_gfx950
from utils import assert_no_triton_compile, is_cdna4


@pytest.fixture(autouse=True)
def require_gfx950():
    if not is_cdna4():
        pytest.skip("AMD gfx950 is required for Gluon BF16 RMSNorm tests")


def assert_bytes(actual, expected):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    mask = torch.isnan(expected.float())
    assert torch.equal(torch.isnan(actual.float()), mask)
    dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[actual.element_size()]
    a = actual.contiguous().view(dtype).to(torch.int64)
    b = expected.contiguous().view(dtype).to(torch.int64)
    assert torch.equal(a[~mask], b[~mask])


def reference(x, residual, weight, x_scale, residual_scale, quantize):
    if residual is None:
        summed = None
    else:
        scaled_x = x if x_scale is None else x * x_scale
        scaled_residual = (
            residual if residual_scale is None else residual * residual_scale
        )
        summed = scaled_x + scaled_residual
    source = x if summed is None else summed
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.shape[0]:
        # The unchanged portable norm, bypassing the new backend dispatcher.
        _rmsnorm_kernel[(x.shape[0],)](
            source,
            None,
            weight,
            output,
            None,
            1.0,
            1.0,
            source.stride(0),
            5120,
            1e-6,
            BLOCK=8192,
            HAS_RESIDUAL=False,
            ROUND_RESIDUAL_SUM_BF16=False,
            SCALE_INPUTS_BF16=False,
            ENABLE_PDL=False,
            fp8_out_ptr=None,
            fp8_scale_ptr=None,
            TOKEN_FP8_OUT=False,
            num_warps=4,
            enable_fp_fusion=True,
        )
    codes, scales = (
        fp8_quantize_per_token(output, enable_pdl=False) if quantize else (None, None)
    )
    return output, summed, codes, scales


def invoke(x, residual, weight, x_scale, residual_scale, quantize):
    options = dict(
        residual=residual,
        round_residual_sum_bf16=residual is not None,
        x_scale=x_scale,
        residual_scale=residual_scale,
        enable_pdl=False,
    )
    if quantize:
        return rmsnorm_quantize_fp8(x, weight, 1e-6, **options)
    result = rmsnorm(x, weight, 1e-6, out=None, token_fp8_out=None, **options)
    if residual is None:
        return result, None, None, None
    output, summed = result
    return output, summed, None, None


@pytest.mark.parametrize("rows", [0, 1, 2, 16, 256, 8192])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("quantize", [False, True])
def test_shared_apis_match_existing_operations(rows, has_residual, quantize):
    torch.manual_seed(1861433681)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if has_residual else None
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
    xs, rs = (0.5, 0.875) if has_residual and rows > 1 else (None, None)
    for actual, expected in zip(
        invoke(x, residual, weight, xs, rs, quantize),
        reference(x, residual, weight, xs, rs, quantize),
        strict=True,
    ):
        assert_bytes(actual, expected)


def test_depth_scales_extremes_and_all_bf16_codes():
    torch.manual_seed(1861433681)
    x = torch.randn(16, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
    cases = [
        (x, residual, 1 / math.sqrt(depth), math.sqrt(depth - 1) / math.sqrt(depth))
        for depth in range(2, 106)
    ]
    cases += [
        (x * magnitude, residual * magnitude, -0.375, 1.125)
        for magnitude in (0.0, 2.0**-120, 2.0**120)
    ]
    codes = torch.arange(13 * 5120, device="cuda", dtype=torch.int32)
    cases.append(
        (
            (codes % 65536).to(torch.uint16).view(torch.bfloat16).view(13, 5120),
            ((codes * 17 + 11) % 65536)
            .to(torch.uint16)
            .view(torch.bfloat16)
            .view(13, 5120),
            1 / math.sqrt(3),
            math.sqrt(2) / math.sqrt(3),
        )
    )
    for values, summed, xs, rs in cases:
        for actual, expected in zip(
            invoke(values, summed, weight, xs, rs, True),
            reference(values, summed, weight, xs, rs, True),
            strict=True,
        ):
            assert_bytes(actual, expected)


def test_runtime_scalars_and_changed_input_graph_replay():
    weight = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(3, 5120, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    invoke(x, residual, weight, 0.5, 0.875, True)
    with assert_no_triton_compile(gluon_rmsnorm_bf16_gfx950):
        for rows, xs in ((15, 0.25), (16, 1.0), (17, -1.0)):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            for actual, expected in zip(
                invoke(x, residual, weight, xs, 0.75, True),
                reference(x, residual, weight, xs, 0.75, True),
                strict=True,
            ):
                assert_bytes(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke(x, residual, weight, xs, 0.75, True)
    x.neg_()
    residual.mul_(1.125)
    graph.replay()
    torch.cuda.synchronize()
    for actual, expected in zip(
        output, reference(x, residual, weight, xs, 0.75, True), strict=True
    ):
        assert_bytes(actual, expected)
    graph.reset()


def test_strided_rows_and_flat_scale_destination():
    x = torch.randn(17, 5128, device="cuda", dtype=torch.bfloat16)[:, :5120]
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
    codes = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(17, device="cuda", dtype=torch.float32)
    output = rmsnorm(
        x,
        weight,
        1e-6,
        residual=None,
        out=None,
        enable_pdl=False,
        round_residual_sum_bf16=False,
        x_scale=None,
        residual_scale=None,
        token_fp8_out=(codes, scales),
    )
    expected, _, expected_codes, expected_scales = reference(
        x, None, weight, None, None, True
    )
    assert_bytes(output, expected)
    assert_bytes(codes, expected_codes)
    assert_bytes(scales[:, None], expected_scales)
