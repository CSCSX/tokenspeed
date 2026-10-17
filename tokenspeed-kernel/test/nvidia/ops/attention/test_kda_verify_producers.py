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

"""Numerical checks for the combined KDA convolution and gate producer."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)

from tokenspeed_kernel.ops.attention.kda import (
    kda_verify_conv_gate,
    kda_verify_conv_update,
)


@pytest.mark.parametrize(
    "batch,width,channels,rank",
    [
        (1, 1, 1536, 128),
        (4, 4, 1536, 128),
        (8, 1, 1536, 128),
        (16, 4, 1536, 128),
        (32, 1, 1536, 128),
        (32, 4, 1536, 128),
        (3, 1, 387, 137),
        (3, 4, 387, 96),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "alias"])
@pytest.mark.parametrize("exact_inputs", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_verify_conv_gate(
    batch: int,
    width: int,
    channels: int,
    rank: int,
    layout: str,
    exact_inputs: bool,
    captured: bool,
) -> None:
    torch.manual_seed(717)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    rows = batch * width
    pad = 0 if layout == "contiguous" else 13
    raw_storage = torch.randn(rows, 3 * channels + pad, **options)
    raw = raw_storage[:, : 3 * channels]
    fa = torch.randn(rows, rank + pad, **options)[:, :rank]
    fb = torch.randn(channels, rank + pad, **options)[:, :rank]
    if exact_inputs:
        # Every product and partial sum is exactly representable in FP32.
        fa.copy_(torch.randint(-8, 9, fa.shape, device="cuda").float() / 8)
        fb.copy_(torch.randint(-8, 9, fb.shape, device="cuda").float() / 8)
    weights = torch.randn(3 * channels, 4, **options) * 0.3
    pool = torch.randn(batch + 1, 3 * channels + 17, 3, **options)
    reads_storage = torch.empty(batch * 2, device="cuda", dtype=torch.int64)
    reads = reads_storage[::2]
    reads.copy_(torch.arange(batch, device="cuda"))
    reads[0] = -1
    if batch > 2:
        reads[1:3] = 0
    replay_storage = torch.full((rows + 3, 3 * channels + 23), -17.0, **options)
    replay = raw if layout == "alias" else replay_storage[:rows, : 3 * channels]
    input_tensors = (raw_storage, fa, fb, weights, pool, reads)
    originals = tuple(t.clone() for t in input_tensors)

    def produce() -> tuple[torch.Tensor, torch.Tensor]:
        return kda_verify_conv_gate(
            raw,
            weights,
            pool,
            reads,
            fa,
            fb,
            draft_token_num=width,
            replay_mixed_qkv=replay,
        )

    if captured:
        produce()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            convolved, gate = produce()
        graph.replay()
    else:
        convolved, gate = produce()
    reference_conv = kda_verify_conv_update(
        raw,
        weights,
        pool,
        reads.contiguous(),
        num_heads=1,
        head_dim=channels,
        draft_token_num=width,
        recurrent_layout="v_major",
    )
    torch.testing.assert_close(convolved, reference_conv, atol=0, rtol=0)
    torch.testing.assert_close(replay, raw, atol=0, rtol=0)
    torch.testing.assert_close(
        replay_storage[rows:],
        torch.full_like(replay_storage[rows:], -17),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        replay_storage[:rows, 3 * channels :],
        torch.full_like(replay_storage[:rows, 3 * channels :], -17),
        atol=0,
        rtol=0,
    )
    for tensor, original in zip(input_tensors, originals, strict=True):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)

    oracle = fa.double() @ fb.double().T
    reference_gate = torch.nn.functional.linear(fa, fb)
    if exact_inputs:
        torch.testing.assert_close(gate, oracle.bfloat16(), atol=0, rtol=0)
        torch.testing.assert_close(gate, reference_gate, atol=0, rtol=0)
    else:
        # Bound the FP32 dot error before testing the final BF16 rounding bin.
        # Different valid reduction orders can straddle a BF16 midpoint.
        magnitude = fa.double().abs() @ fb.double().abs().T
        eps32, eps64 = 2.0**-23, 2.0**-53
        bound = (
            rank * eps32 / (1 - rank * eps32) + rank * eps64 / (1 - rank * eps64)
        ) * magnitude
        for value in (gate, reference_gate):
            assert bool(torch.isfinite(value).all())
            prev = torch.nextafter(
                value, torch.full_like(value, -float("inf"))
            ).double()
            next_value = torch.nextafter(
                value, torch.full_like(value, float("inf"))
            ).double()
            low = (prev + value.double()) * 0.5
            high = (next_value + value.double()) * 0.5
            assert bool(((low <= oracle + bound) & (high >= oracle - bound)).all())


def test_verify_conv_gate_without_capture_destination() -> None:
    torch.manual_seed(719)
    options = {"device": "cuda", "dtype": torch.bfloat16}
    raw = torch.randn(4, 384, **options)
    weights = torch.randn(384, 4, **options)
    pool = torch.randn(2, 384, 3, **options)
    reads = torch.tensor([1], device="cuda", dtype=torch.int32)
    fa = torch.randint(-8, 9, (4, 128), device="cuda").to(torch.bfloat16) / 8
    fb = torch.randint(-8, 9, (128, 128), device="cuda").to(torch.bfloat16) / 8
    conv, gate = kda_verify_conv_gate(
        raw,
        weights,
        pool,
        reads,
        fa,
        fb,
        draft_token_num=4,
    )
    reference = kda_verify_conv_update(
        raw,
        weights,
        pool,
        reads,
        num_heads=1,
        head_dim=128,
        draft_token_num=4,
        recurrent_layout="v_major",
    )
    torch.testing.assert_close(conv, reference, atol=0, rtol=0)
    torch.testing.assert_close(
        gate, (fa.double() @ fb.double().T).bfloat16(), atol=0, rtol=0
    )


def _gate_for_exact_inputs(fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    rows, channels = fa.shape[0], fb.shape[0]
    raw = torch.zeros((rows, 3 * channels), device=fa.device, dtype=fa.dtype)
    weights = torch.zeros((3 * channels, 4), device=fa.device, dtype=fa.dtype)
    state = torch.zeros((1, 3 * channels, 3), device=fa.device, dtype=fa.dtype)
    reads = torch.zeros(rows, device=fa.device, dtype=torch.int32)
    convolved, gate = kda_verify_conv_gate(
        raw,
        weights,
        state,
        reads,
        fa,
        fb,
        draft_token_num=1,
    )
    torch.testing.assert_close(convolved, raw, atol=0, rtol=0)
    return gate


@pytest.mark.parametrize("rank", [31, 32, 33, 127, 128, 129])
def test_verify_gate_reads_every_reduction_position(rank: int) -> None:
    # A one-hot row selects a different coefficient at each reduction index.
    # Scaling by a power of two introduces no BF16 or FP32 rounding error.
    fa = torch.eye(rank, device="cuda", dtype=torch.bfloat16) / 128
    coefficients = torch.arange(1, rank + 1, device="cuda", dtype=torch.float32)
    channels = torch.arange(33, device="cuda")
    fb = (
        coefficients[None, :] * torch.where(channels[:, None] % 2 == 0, 1, -1)
    ).bfloat16()
    gate = _gate_for_exact_inputs(fa, fb)
    torch.testing.assert_close(gate, fb.T / 128, atol=0, rtol=0)


def test_verify_gate_rounds_exact_midpoints_to_even() -> None:
    # Every intermediate sum is representable in FP32, including the sticky bit.
    terms = [
        [1.0, 2.0**-8, 0.0],
        [1.0, 2.0**-7, 2.0**-8],
        [1.0, 2.0**-8, 2.0**-23],
        [1.0, 2.0**-8, -(2.0**-23)],
        [-1.0, -(2.0**-8), 0.0],
        [-1.0, -(2.0**-7), -(2.0**-8)],
    ]
    expected = [1.0, 1.0 + 2.0**-6, 1.0 + 2.0**-7, 1.0, -1.0, -1.0 - 2.0**-6]
    fa = torch.zeros((len(terms), 128), device="cuda", dtype=torch.bfloat16)
    fa[:, :3] = torch.tensor(terms, device="cuda", dtype=torch.bfloat16)
    fb = torch.ones((33, 128), device="cuda", dtype=torch.bfloat16)
    gate = _gate_for_exact_inputs(fa, fb)
    reference = torch.tensor(expected, device="cuda", dtype=torch.bfloat16)[
        :, None
    ].expand_as(gate)
    torch.testing.assert_close(gate, reference, atol=0, rtol=0)


def test_verify_gate_retains_small_accumulation_terms() -> None:
    fa = torch.full((1, 128), 2.0**-8, device="cuda", dtype=torch.bfloat16)
    fa[0, 0] = 1.0
    fb = torch.ones((33, 128), device="cuda", dtype=torch.bfloat16)
    gate = _gate_for_exact_inputs(fa, fb)
    # The exact FP32 sum is 1 + 127/256, a midpoint that rounds to 1.5.
    torch.testing.assert_close(gate, torch.full_like(gate, 1.5), atol=0, rtol=0)
