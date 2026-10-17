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

import json

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda import (
    KdaReplayHistory,
    KdaReplayHistoryBatch,
    KdaReplayRecords,
    kda_materialize_history,
    try_kda_fused_paged_decode,
    try_kda_fused_paged_verify,
)
from tokenspeed_kernel.ops.attention.kda._triton.buffered import (
    append_kda_history_kernel,
)
from tokenspeed_kernel.ops.attention.kda._triton.nvidia_history_decode import (
    _fused_history_decode_kernel,
)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    batched_recoverssm_commit,
)
from utils import assert_no_triton_compile


def _verify(
    inputs,
    weights,
    conv,
    fa,
    fb,
    beta,
    a,
    bias,
    state,
    reads,
    writes,
    qsl,
    history,
    raw,
    out,
    lower_bound,
    u,
    kd,
    metadata,
    stamps,
):
    batch = reads.numel()
    heads, dim, _ = state.shape[1:]
    records = KdaReplayRecords(
        correction=u.view(batch, heads, 1, dim),
        key_gate=kd.view(batch, heads, 1, 2 * dim),
        key_gate_format="normalized_key_decay",
        metadata=metadata,
        stamps=stamps,
        query_start_loc=qsl,
        history=history,
    )
    result = try_kda_fused_paged_verify(
        inputs,
        weights,
        conv,
        conv,
        fa,
        fb,
        beta,
        a,
        bias,
        state_pool=state,
        state_scratch=None,
        read_indices=reads,
        write_indices=writes,
        num_heads=heads,
        head_dim=dim,
        draft_token_num=1,
        lower_bound=lower_bound,
        store_states=False,
        recurrent_layout="v_major",
        replay_records=records,
        replay_mixed_qkv=raw,
    )
    assert result is not None
    out.copy_(result.reshape_as(out))


def _commit(
    descriptors,
    groups,
    reads,
    writes,
    accepted,
    history,
    heads,
    dim,
    raw_stride,
    conv_stride,
    metadata,
    stamps,
    lower_bound,
):
    batched_recoverssm_commit(
        descriptors,
        groups,
        reads,
        writes,
        accepted,
        draft_token_num=1,
        num_heads=heads,
        head_dim=dim,
        qkv_stride=raw_stride,
        conv_stride=conv_stride,
        correction_stride=heads * dim,
        key_gate_stride=2 * heads * dim,
        key_gate_format="normalized_key_decay",
        state_stride=heads * dim * dim,
        conv_width=4,
        record_metadata=metadata,
        record_stamps=stamps,
        lower_bound=lower_bound,
        history=history,
    )


@pytest.mark.parametrize("capacity", [8, 16, 32])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("batch", [8, 32])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_fused_t1_history_lifecycle(
    capacity: int, captured: bool, batch: int, lower_bound: float | None
) -> None:
    """Compare real fused decode across acceptance, page reuse and graph replay."""
    torch.manual_seed(70031)
    heads, dim = 12, 128
    channels = heads * dim
    pages = 4 * batch + 1
    initial = torch.randn(pages, heads, dim, dim, device="cuda") * 0.02
    initial[0].zero_()
    initial_conv = (
        torch.randn(pages, 3 * channels, 3, device="cuda", dtype=torch.bfloat16) * 0.3
    )
    initial_conv[0].zero_()
    state = initial.clone()
    reference = initial.clone()
    conv = initial_conv.clone()
    reference_conv = initial_conv.clone()
    inputs = torch.randn(batch, 3 * channels, device="cuda", dtype=torch.bfloat16) * 0.3
    raw = torch.empty_like(inputs)
    fa = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16) * 0.3
    fb = torch.randn(channels, dim, device="cuda", dtype=torch.bfloat16) * 0.05
    beta = torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16) * 0.3
    weights = torch.randn(3 * channels, 4, device="cuda", dtype=torch.bfloat16) * 0.3
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    reads[1] = 1
    writes = torch.arange(batch + 1, 2 * batch + 1, device="cuda", dtype=torch.int32)
    groups = torch.zeros(1, device="cuda", dtype=torch.int32)
    qsl = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 1
    stamps = torch.full((1, batch, 1, 2), -1, device="cuda", dtype=torch.int64)
    u = torch.empty(batch, channels, device="cuda")
    kd = torch.empty(batch, 2 * channels, device="cuda")
    output = torch.empty(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    factors = [
        torch.full((pages, heads, capacity, dim), float("nan"), device="cuda")
        for _ in range(3)
    ]
    lengths = torch.zeros(pages, device="cuda", dtype=torch.int32)
    offsets = torch.full((1, batch), -1, device="cuda", dtype=torch.int32)
    history = KdaReplayHistory(*factors, lengths, writes, offsets[0], False)
    descriptors = torch.zeros(1, 10, device="cuda", dtype=torch.int64)
    for col, tensor in (
        (0, raw),
        (2, conv),
        (3, u),
        (5, kd),
        (6, a),
        (7, bias),
        (8, state),
    ):
        descriptors[0, col] = tensor.data_ptr()
    hdesc = torch.tensor(
        [[t.data_ptr() for t in [*factors, lengths]]], device="cuda", dtype=torch.int64
    )
    hb = KdaReplayHistoryBatch(
        hdesc, offsets, capacity, *[f.stride(0) for f in factors], lengths.stride(0)
    )
    recovered = torch.empty_like(state[:batch])

    def step():
        _verify(
            inputs,
            weights,
            conv,
            fa,
            fb,
            beta,
            a,
            bias,
            state,
            reads,
            writes,
            qsl,
            history,
            raw,
            output,
            lower_bound,
            u,
            kd,
            metadata,
            stamps[0],
        )
        _commit(
            descriptors,
            groups,
            reads[None],
            writes[None],
            accepted,
            hb,
            heads,
            dim,
            raw.stride(0),
            conv.stride(0),
            metadata,
            stamps,
            lower_bound,
        )

    step()
    torch.cuda.synchronize()
    graph = None
    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
    state.copy_(initial)
    conv.copy_(initial_conv)
    lengths.zero_()
    stamps.fill_(-1)
    offsets.fill_(-1)
    for f in factors:
        f.fill_(float("nan"))
    current = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    current[1] = 1
    bank_a = torch.arange(batch + 1, 2 * batch + 1, device="cuda", dtype=torch.int32)
    bank_b = bank_a + batch
    temporary = bank_b + batch
    positions = torch.arange(batch, device="cuda", dtype=torch.int64) * 19
    zero_accepts = 0
    rotations = 0
    padding = 0
    full_zero_accepts = 0
    for rnd in range(96):
        inputs.normal_(std=0.3)
        fa.normal_(std=0.3)
        beta.normal_(std=0.3)
        active = torch.ones(batch, device="cuda", dtype=torch.bool)
        if rnd >= capacity + 2 and rnd % 4 == 0:
            active[1::3] = False
        counts = active.to(torch.int32)
        if rnd == capacity:
            counts[::2] = 0
        elif rnd >= capacity + 2 and rnd % 3 == 0:
            counts[::3] = 0
        if rnd == 60:
            current[7] = -1
        if rnd == 70:
            current[0] = 0
        destination = current.clone()
        if rnd == 0 or (rnd > capacity and rnd % 11 == 0):
            destination = torch.where(current == bank_a, bank_b, bank_a)
        if rnd == 60:
            destination[7] = bank_a[7]
        if rnd == 70:
            destination[0] = bank_a[0]
        destination = torch.where(current <= 0, bank_a, destination)
        destination = torch.where(counts > 0, destination, current)
        if rnd == capacity:
            destination[2::4] = bank_b[2::4]
        destination = torch.where(active, destination, 0).to(torch.int32)
        reads.copy_(current)
        writes.copy_(destination)
        accepted.copy_(counts)
        qsl[0] = 0
        qsl[1:].copy_(active.to(torch.int32).cumsum(0))
        slots = torch.roll(
            torch.arange(batch, device="cuda", dtype=torch.int64), rnd % batch
        )
        metadata[:, 0] = torch.where(active, slots, -1)
        metadata[:, 1] = rnd + 2
        metadata[:, 2] = positions
        zero_accepts += int((active & (counts == 0)).sum())
        padding += int((~active).sum())
        rotations += int(((destination != current) & (counts > 0)).sum())
        full_zero_accepts += int(
            (active & (counts == 0) & (lengths[current.long()] == capacity)).sum()
        )
        zero_move = (
            active & (counts == 0) & (destination != current) & (destination > 0)
        )
        zero_sources = current.clamp_min(0).long()
        preserved_state = torch.where(
            (current > 0)[:, None, None, None], reference[zero_sources], 0.0
        )
        preserved_conv = torch.where(
            (current > 0)[:, None, None], reference_conv[zero_sources], 0.0
        )
        expected = try_kda_fused_paged_decode(
            inputs,
            weights,
            reference_conv,
            fa,
            fb,
            beta,
            a,
            bias,
            state_pool=reference,
            read_indices=current,
            write_indices=torch.where(counts > 0, destination, temporary),
            num_heads=heads,
            head_dim=dim,
            cu_seqlens=qsl,
            lower_bound=lower_bound,
            recurrent_layout="v_major",
        )
        assert expected is not None
        reference[destination[zero_move].long()] = preserved_state[zero_move]
        reference_conv[destination[zero_move].long()] = preserved_conv[zero_move]
        unchanged = torch.ones(pages, device="cuda", dtype=torch.bool)
        unchanged[destination[active].long()] = False
        before = [t[unchanged].clone() for t in [state, conv, *factors, lengths]]
        if graph is None:
            with assert_no_triton_compile(
                _fused_history_decode_kernel.fn, append_kda_history_kernel
            ):
                step()
        else:
            graph.replay()
        new_current = torch.where(active, destination, current)
        kda_materialize_history(
            state, *factors, lengths, new_current, out=recovered, use_tensor_cores=False
        )
        live_tokens = int(active.sum())
        torch.testing.assert_close(
            output[:live_tokens],
            expected.reshape(batch, heads, dim)[:live_tokens],
            atol=0,
            rtol=0,
        )
        valid_pages = new_current > 0
        safe_pages = new_current.clamp_min(0).long()
        expected_state = torch.where(
            valid_pages[:, None, None, None], reference[safe_pages], 0.0
        )
        actual_conv = torch.where(valid_pages[:, None, None], conv[safe_pages], 0.0)
        expected_conv = torch.where(
            valid_pages[:, None, None], reference_conv[safe_pages], 0.0
        )
        try:
            torch.testing.assert_close(recovered, expected_state, atol=0, rtol=0)
        except AssertionError:
            print(
                json.dumps(
                    {
                        "failed_round": rnd,
                        "field": "endpoint",
                        "captured": captured,
                        "capacity": capacity,
                        "read_pages": reads.tolist(),
                        "write_pages": writes.tolist(),
                        "accepted": counts.tolist(),
                        "new_pages": new_current.tolist(),
                    }
                ),
                flush=True,
            )
            raise
        torch.testing.assert_close(actual_conv, expected_conv, atol=0, rtol=0)
        for field, actual, old in zip(
            ["state", "conv", "correction", "key", "decay", "length"],
            [state, conv, *factors, lengths],
            before,
        ):
            try:
                torch.testing.assert_close(
                    actual[unchanged], old, atol=0, rtol=0, equal_nan=True
                )
            except AssertionError:
                print(
                    json.dumps(
                        {
                            "failed_round": rnd,
                            "field": field,
                            "captured": captured,
                            "capacity": capacity,
                            "read_pages": reads.tolist(),
                            "write_pages": writes.tolist(),
                            "accepted": counts.tolist(),
                            "unwritten_pages": unchanged.nonzero().flatten().tolist(),
                        }
                    ),
                    flush=True,
                )
                raise
        actual_stamps = stamps[0, slots[active], 0]
        expected_stamps = torch.stack(
            [positions[active], torch.full_like(positions[active], rnd + 2)], dim=-1
        )
        torch.testing.assert_close(actual_stamps, expected_stamps, atol=0, rtol=0)
        assert bool(
            (
                (lengths[new_current.long()] >= 0)
                & (lengths[new_current.long()] <= capacity)
            ).all()
        )
        current = new_current
        positions += counts
    assert full_zero_accepts > 0 and rotations > 0 and padding > 0


def test_fused_t1_batch_changes_preserve_output_without_compilation() -> None:
    """Changing live request counts reuses the warmed model specialization."""
    batch, capacity = 32, 8
    torch.manual_seed(70031)
    heads, dim = 12, 128
    channels = heads * dim
    pages = 4 * batch + 1
    initial = torch.randn(pages, heads, dim, dim, device="cuda") * 0.02
    initial[0].zero_()
    initial_conv = (
        torch.randn(pages, 3 * channels, 3, device="cuda", dtype=torch.bfloat16) * 0.3
    )
    initial_conv[0].zero_()
    state = initial.clone()
    reference = initial.clone()
    conv = initial_conv.clone()
    reference_conv = initial_conv.clone()
    inputs = torch.randn(batch, 3 * channels, device="cuda", dtype=torch.bfloat16) * 0.3
    raw = torch.empty_like(inputs)
    fa = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16) * 0.3
    fb = torch.randn(channels, dim, device="cuda", dtype=torch.bfloat16) * 0.05
    beta = torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16) * 0.3
    weights = torch.randn(3 * channels, 4, device="cuda", dtype=torch.bfloat16) * 0.3
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    reads[1] = 1
    writes = torch.arange(batch + 1, 2 * batch + 1, device="cuda", dtype=torch.int32)
    qsl = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 1
    stamps = torch.full((1, batch, 1, 2), -1, device="cuda", dtype=torch.int64)
    u = torch.empty(batch, channels, device="cuda")
    kd = torch.empty(batch, 2 * channels, device="cuda")
    output = torch.empty(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    factors = [
        torch.full((pages, heads, capacity, dim), float("nan"), device="cuda")
        for _ in range(3)
    ]
    lengths = torch.zeros(pages, device="cuda", dtype=torch.int32)
    offsets = torch.full((1, batch), -1, device="cuda", dtype=torch.int32)

    def verify_prefix(live_batch):
        live_history = KdaReplayHistory(
            *factors, lengths, writes[:live_batch], offsets[0, :live_batch], False
        )
        _verify(
            inputs[:live_batch],
            weights,
            conv,
            fa[:live_batch],
            fb,
            beta[:live_batch],
            a,
            bias,
            state,
            reads[:live_batch],
            writes[:live_batch],
            qsl[: live_batch + 1],
            live_history,
            raw[:live_batch],
            output[:live_batch],
            -5.0,
            u[:live_batch],
            kd[:live_batch],
            metadata[:live_batch],
            stamps[0, :live_batch],
        )

    verify_prefix(batch)
    for live_batch in (1, 3, 4, 7, 8, 16, 23, 32):
        metadata[:, 1].add_(1)
        with assert_no_triton_compile(_fused_history_decode_kernel.fn):
            verify_prefix(live_batch)
        expected = try_kda_fused_paged_decode(
            inputs[:live_batch],
            weights,
            reference_conv,
            fa[:live_batch],
            fb,
            beta[:live_batch],
            a,
            bias,
            state_pool=reference,
            read_indices=reads[:live_batch],
            write_indices=writes[:live_batch],
            num_heads=heads,
            head_dim=dim,
            cu_seqlens=qsl[: live_batch + 1],
            lower_bound=-5.0,
            recurrent_layout="v_major",
        )
        assert expected is not None
        torch.testing.assert_close(
            output[:live_batch],
            expected.reshape(live_batch, heads, dim),
            atol=0,
            rtol=0,
        )
