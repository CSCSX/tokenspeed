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


"""Check the compact FC1 pipeline against the established channel-FP8 body."""

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform


@pytest.fixture(scope="module")
def expert_weights():
    platform = current_platform()
    if not (platform.is_amd and platform.arch_version == ArchVersion(9, 5)):
        pytest.skip("requires gfx950")
    generator = torch.Generator(device="cuda").manual_seed(1861433681)
    weight = (
        (
            torch.randn(
                256, 640, 5120, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 100
        )
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.rand(256, 640, device="cuda", generator=generator) * 0.0002
    return weight, scale


def make_case(tokens, clustered):
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.exact_mfma import (
        _select_route_block_size,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )

    generator = torch.Generator(device="cuda").manual_seed(75 + tokens)
    codes = (
        (torch.randn(tokens, 5120, device="cuda", generator=generator) * 100)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.rand(tokens, 1, device="cuda", generator=generator) * 0.01
    ids = (
        torch.argsort(
            torch.rand(tokens, 256, device="cuda", generator=generator), dim=-1
        )[:, :8]
        .int()
        .contiguous()
    )
    if clustered:
        ids.copy_(torch.arange(8, device="cuda", dtype=torch.int32).expand(tokens, 8))
    route_weights = torch.rand(tokens, 8, device="cuda", generator=generator)
    block_m = _select_route_block_size(tokens, 8, 256)
    routing = moe_align_block_size_device(
        ids, route_weights, 256, block_m, expert_start=0
    )
    return codes, scales, routing, block_m


def legacy(codes, weights, row_scales, column_scales, routing, block_m):
    from tokenspeed_kernel_amd._triton import triton
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage1_kernel import (
        gluon_bf16_moe_stage1_kernel,
    )

    sorted_ids, sorted_experts, _, valid = routing
    output = torch.empty(codes.shape[0] * 8, 320, device="cuda", dtype=torch.bfloat16)
    em = sorted_ids.numel()
    grid = triton.cdiv(em, block_m) * 5
    gluon_bf16_moe_stage1_kernel[(grid,)](
        codes,
        weights,
        column_scales,
        row_scales,
        output,
        sorted_ids,
        sorted_experts,
        valid,
        5120,
        em,
        codes.shape[0],
        8,
        *codes.stride(),
        *weights.stride(),
        0,
        0,
        0,
        *output.stride(),
        grid,
        I_r=320,
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_K=64,
        NUM_WARPS=4,
        NUM_XCDS=1 if block_m == 16 else 8,
        GROUP_SIZE_M=4 if block_m == 16 else 8,
        WEIGHT_FP8=False,
        FP8_ASYNC=False,
        CHANNEL_FP8=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def candidate(codes, weights, row_scales, column_scales, routing, block_m):
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.channel_stage1 import (
        launch_gluon_fp8_channel_moe_stage1_gfx950,
    )

    sorted_ids, sorted_experts, _, valid = routing
    output = torch.empty(codes.shape[0] * 8, 320, device="cuda", dtype=torch.bfloat16)
    launch_gluon_fp8_channel_moe_stage1_gfx950(
        codes,
        weights,
        row_scales,
        column_scales,
        sorted_ids,
        sorted_experts,
        valid,
        output,
        8,
        block_m=block_m,
    )
    return output


@pytest.mark.parametrize(
    "tokens", [1, 2, 4, 8, 16, 32, 63, 64, 65, 128, 255, 256, 257, 896, 8192]
)
@pytest.mark.parametrize("clustered", [False, True])
def test_async_channel_fc1_matches_established_bytes(expert_weights, tokens, clustered):
    weight, scale = expert_weights
    codes, row_scales, routing, block_m = make_case(tokens, clustered)
    expected = legacy(codes, weight, row_scales, scale, routing, block_m)
    actual = candidate(codes, weight, row_scales, scale, routing, block_m)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_async_channel_fc1_changed_input_graph_replay(expert_weights):
    weight, scale = expert_weights
    codes, row_scales, routing, block_m = make_case(32, False)
    candidate(codes, weight, row_scales, scale, routing, block_m)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = candidate(codes, weight, row_scales, scale, routing, block_m)
    codes.copy_((-codes.float()).to(codes.dtype))
    row_scales.mul_(1.125)
    _, _, clustered_routing, _ = make_case(32, True)
    for destination, source in zip(routing, clustered_routing, strict=True):
        destination.copy_(source)
    graph.replay()
    torch.testing.assert_close(
        output,
        legacy(codes, weight, row_scales, scale, routing, block_m),
        rtol=0,
        atol=0,
    )


def test_async_channel_fc1_neighboring_rows_do_not_compile(expert_weights):
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.channel_stage1 import (
        gluon_fp8_channel_moe_stage1_gfx950,
    )
    from utils import assert_no_triton_compile

    weight, scale = expert_weights
    cases = [make_case(tokens, False) for tokens in (17, 24, 32)]
    codes, row_scales, routing, block_m = cases[0]
    candidate(codes, weight, row_scales, scale, routing, block_m)
    for codes, row_scales, routing, block_m in cases[1:]:
        expected = legacy(codes, weight, row_scales, scale, routing, block_m)
        with assert_no_triton_compile(gluon_fp8_channel_moe_stage1_gfx950):
            actual = candidate(codes, weight, row_scales, scale, routing, block_m)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
