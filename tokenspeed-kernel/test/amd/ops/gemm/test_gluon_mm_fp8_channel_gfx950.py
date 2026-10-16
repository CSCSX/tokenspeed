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

import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp8.channel import (
    gluon_mm_fp8_channel_gfx950,
    gluon_mm_fp8_channel_reduce_gfx950,
    launch_gluon_mm_fp8_channel_gfx950,
)
from utils import assert_no_triton_compile, is_cdna4

pytestmark = pytest.mark.skipif(not is_cdna4(), reason="requires AMD gfx950")


def test_all_finite_e4m3_products_are_exact():
    bits = torch.arange(256, device="cuda", dtype=torch.uint8)
    bits = bits[torch.isfinite(bits.view(torch.float8_e4m3fn).float())]
    codes = torch.zeros(bits.numel(), 32, device="cuda", dtype=torch.float8_e4m3fn)
    codes.view(torch.uint8)[:, 0].copy_(bits)
    scales = torch.ones(bits.numel(), 1, device="cuda")
    result = launch_gluon_mm_fp8_channel_gfx950(
        codes,
        codes.T,
        scales,
        scales,
        torch.float32,
        out=None,
    )
    torch.testing.assert_close(
        result, (codes.double() @ codes.double().T).float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("position", [1, 7, 8, 31, 63, 127])
@pytest.mark.parametrize("sign", [-1, 1])
def test_small_product_survives_a_large_neighbor(position, sign):
    a = torch.zeros(1, 128, device="cuda")
    b = torch.zeros(16, 128, device="cuda")
    a[:, 0], b[:, 0] = 256, 256
    a[:, position], b[:, position] = 0.25, sign * 0.25
    result = launch_gluon_mm_fp8_channel_gfx950(
        a.to(torch.float8_e4m3fn),
        b.to(torch.float8_e4m3fn).T,
        torch.ones(1, device="cuda"),
        torch.ones(16, device="cuda"),
        torch.float32,
        out=None,
    )
    assert torch.all(result == 65536 + sign * 0.0625)


@pytest.mark.parametrize("rows", [1, 16, 65, 256])
@pytest.mark.parametrize("n,k", [(37, 333), (1792, 5120), (5120, 768)])
def test_fp32_accumulator_respects_componentwise_error_bound(rows, n, k):
    torch.manual_seed(196)
    a = (
        (torch.randn(rows, k, device="cuda") * 128)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    b = (
        (torch.randn(n, k, device="cuda") * 128)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    result = launch_gluon_mm_fp8_channel_gfx950(
        a,
        b.T,
        torch.ones(rows, 1, device="cuda"),
        torch.ones(n, 1, device="cuda"),
        torch.float32,
        out=None,
    )
    reference = a.double() @ b.double().T
    magnitude = a.double().abs() @ b.double().abs().T
    u = 2**-24
    gamma = (k + 8) * u / (1 - (k + 8) * u)
    assert torch.all(
        (result.double() - reference).abs() <= gamma * magnitude + u * reference.abs()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "scalar_a,scalar_b", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("n,k", [(37, 333), (1792, 5120)])
def test_scales_strides_and_caller_output(dtype, scalar_a, scalar_b, n, k):
    torch.manual_seed(197)
    rows = 33
    a = torch.randint(-8, 9, (rows, k + 7), device="cuda").to(torch.float8_e4m3fn)[
        :, :k
    ]
    b = torch.randint(-8, 9, (n, k + 9), device="cuda").to(torch.float8_e4m3fn)[:, :k]
    sa = torch.linspace(0.001, 0.02, (1 if scalar_a else rows) * 3, device="cuda")[::3]
    sb = torch.linspace(0.001, 0.01, (1 if scalar_b else n) * 3, device="cuda")[::3]
    storage = torch.full((rows, n + 16), 17, dtype=dtype, device="cuda")
    output = storage[:, :n]
    actual = launch_gluon_mm_fp8_channel_gfx950(a, b.T, sa, sb, dtype, out=output)
    # These integer products/sums are exact, so this independently tests the
    # FP32 token-then-channel scaling and final rounding without tolerances.
    reference = ((a.double() @ b.double().T).float() * sa[:, None]) * sb[None, :]
    assert actual is output
    torch.testing.assert_close(actual, reference.to(dtype), rtol=0, atol=0)
    assert torch.all(storage[:, n:] == 17)


def test_overlapping_output_is_rejected_before_writing():
    a = torch.zeros(2, 128, device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.zeros(16, 128, device="cuda", dtype=a.dtype)
    storage = torch.full((1, 16), 17, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="nonoverlapping"):
        launch_gluon_mm_fp8_channel_gfx950(
            a,
            b.T,
            torch.ones(2, 1, device="cuda"),
            torch.ones(16, 1, device="cuda"),
            torch.bfloat16,
            out=storage.expand(2, -1),
        )
    assert torch.all(storage == 17)


def test_graph_refresh_and_row_count_do_not_recompile():
    torch.manual_seed(198)
    a = torch.randn(32, 5120, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(1792, 5120, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(32, 1, device="cuda")
    sb = torch.ones(1792, 1, device="cuda")
    output = torch.empty(32, 1792, device="cuda", dtype=torch.bfloat16)

    def run(rows):
        return launch_gluon_mm_fp8_channel_gfx950(
            a[:rows], b.T, sa[:rows], sb, torch.bfloat16, out=output[:rows]
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(17)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run(17)
    for scale in (0.5, 2.0, 1.0):
        sa.fill_(scale)
        graph.replay()
        torch.cuda.synchronize()
        captured = output[:17].clone()
        with assert_no_triton_compile(
            gluon_mm_fp8_channel_gfx950, gluon_mm_fp8_channel_reduce_gfx950
        ):
            torch.testing.assert_close(run(17), captured, rtol=0, atol=0)
            run(24)
            run(32)
    graph.reset()
