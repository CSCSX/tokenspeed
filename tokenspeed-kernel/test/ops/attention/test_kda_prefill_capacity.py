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

"""Admission contract for capacity-planned KDA execution."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda import KdaPrefillCapacity


@pytest.mark.parametrize("lengths", ([1, 511], [511, 1], [1, 1], [256, 256]))
def test_capacity_admits_unequal_live_partitions(lengths):
    capacity = KdaPrefillCapacity(512, 2)
    bounds = torch.tensor([0, lengths[0], sum(lengths)])
    capacity.validate(bounds, 512)
    torch.testing.assert_close(capacity.boundaries_cpu(), torch.tensor([0, 512, 1024]))
    torch.testing.assert_close(bounds, torch.tensor([0, lengths[0], sum(lengths)]))


@pytest.mark.parametrize(
    "bounds,tokens",
    (
        ([0, 1, 513], 512),
        ([0, 0, 1], 512),
        ([1, 2, 3], 512),
        ([0, 2], 512),
        ([0, 1, 2], 256),
        ([0, 2, 1], 512),
    ),
)
def test_capacity_rejects_invalid_admission(bounds, tokens):
    with pytest.raises(ValueError):
        KdaPrefillCapacity(512, 2).validate(torch.tensor(bounds), tokens)


@pytest.mark.parametrize("tokens,sequences", ((0, 1), (1, 0), (-1, 2)))
def test_capacity_rejects_invalid_geometry(tokens, sequences):
    with pytest.raises(ValueError):
        KdaPrefillCapacity(tokens, sequences)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_conv_capacity_refresh_masks_history_loads():
    from tokenspeed_kernel.ops.attention.gdn.triton import (
        CausalConv1dPrefillMetadata,
        refresh_causal_conv1d_capacity_metadata,
    )

    maps = CausalConv1dPrefillMetadata(
        batch_indices=torch.empty(19, dtype=torch.int32, device="cuda"),
        chunk_offsets=torch.empty(19, dtype=torch.int32, device="cuda"),
        block_m=8,
    )
    # Zero length tests the conv map builder only, not KDA/native scan admission.
    for lengths in ([1, 1, 1, 125], [125, 1, 1, 1], [0, 8, 9, 16]):
        bounds = torch.tensor(
            [0] + torch.tensor(lengths).cumsum(0).tolist(), device="cuda"
        )
        refresh_causal_conv1d_capacity_metadata(bounds, maps, 128)
        expected = torch.full((19,), -1, dtype=torch.int32)
        offsets = torch.zeros(19, dtype=torch.int32)
        cursor = 0
        for row, length in enumerate(lengths):
            count = (length + 7) // 8
            expected[cursor : cursor + count] = row
            offsets[cursor : cursor + count] = torch.arange(count)
            cursor += count
        torch.testing.assert_close(maps.batch_indices.cpu(), expected)
        torch.testing.assert_close(maps.chunk_offsets.cpu(), offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scan_preparation_across_scalar_boundaries(dtype):
    from tokenspeed_kernel.ops.attention.kda._triton.prefill_scan_inputs import (
        _prepare_capacity_scan_kernel,
        prepare_capacity_scan,
    )
    from utils import assert_no_triton_compile

    def run(tokens, sequences):
        live = tokens if tokens <= sequences + 7 else tokens - 7
        lengths = [1 + row % 17 for row in range(sequences)]
        if sum(lengths) > live:
            lengths = [1] * sequences
        lengths[-1] += live - sum(lengths)
        boundaries = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        shape = (1, tokens, 2, 128)
        q, k, v = [torch.randn(shape, dtype=dtype, device="cuda") for _ in range(3)]
        gate = torch.randn(shape, dtype=torch.float32, device="cuda")
        beta = torch.randn((1, tokens, 2), dtype=dtype, device="cuda")
        inputs = (q, k, v, gate, beta)
        for tensor in inputs:
            tensor[:, live::2] = float("nan")
            tensor[:, live + 1 :: 2] = float("inf")
        *outputs, chunks, chunk_rows = prepare_capacity_scan(
            q, k, v, gate, beta, boundaries
        )
        for source, output in zip(inputs, outputs, strict=True):
            expected = source.clone()
            expected[:, live:] = 0
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        counts = [(length + 15) // 16 for length in lengths]
        expected_chunks = torch.tensor(
            [0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32
        )
        expected_rows = [row for row, count in enumerate(counts) for _ in range(count)]
        capacity = (tokens + 15) // 16 + sequences - 1
        expected_rows += [sequences - 1] * (capacity - len(expected_rows))
        torch.testing.assert_close(chunks.cpu(), expected_chunks, rtol=0, atol=0)
        torch.testing.assert_close(
            chunk_rows.cpu(),
            torch.tensor(expected_rows, dtype=torch.int32),
            rtol=0,
            atol=0,
        )

    # A single startup shape covers small batches with different live counts.
    # Every call still checks the scrubbed inputs and complete device plan.
    run(1, 1)
    with assert_no_triton_compile(_prepare_capacity_scan_kernel):
        for shape in (
            (1, 1),
            (15, 1),
            (16, 1),
            (17, 1),
            (32, 1),
            (96, 1),
            (33, 2),
            (97, 3),
            (96, 4),
            (480, 5),
            (576, 6),
            (672, 7),
            (768, 8),
            (832, 9),
            (1440, 15),
            (1536, 16),
            (1632, 17),
            (3072, 32),
            # MAX_CHUNKS crosses 127/128/129 and 256/257 loop boundaries.
            (2032, 1),
            (2048, 1),
            (2064, 1),
            (4096, 1),
            (4112, 1),
        ):
            run(*shape)

    # Larger sequence counts keep bounded tiles. Exercise their padded rows
    # and chunk plans without introducing variants inside an existing tile.
    for warm, shapes in (
        ((3168, 33), ((6048, 63), (6144, 64))),
        ((8192, 65), ((8192, 96), (8192, 127), (8192, 128))),
    ):
        run(*warm)
        with assert_no_triton_compile(_prepare_capacity_scan_kernel):
            for shape in shapes:
                run(*shape)
