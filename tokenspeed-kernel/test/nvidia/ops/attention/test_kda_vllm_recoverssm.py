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

"""Accuracy tests for the vendored vLLM KDA RecoverSSM kernels."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
    vllm_triton_kda_recoverssm_commit,
    vllm_triton_kda_recoverssm_verify,
)


def _device_addresses(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        device=tensors[0].device,
        dtype=torch.int64,
    )


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    q32 = q.float()
    k32 = k.float()
    q32 *= torch.rsqrt(q32.square().sum(dim=-1, keepdim=True) + 1e-06)
    q32 *= q.shape[-1] ** (-0.5)
    k32 *= torch.rsqrt(k32.square().sum(dim=-1, keepdim=True) + 1e-06)
    gate_input = raw_g.float() + dt_bias[None, None, :, :]
    if lower_bound is None:
        gate = -A_log.exp()[None, None, :, None] * torch.nn.functional.softplus(
            gate_input
        )
    else:
        gate = lower_bound * torch.sigmoid(
            A_log.exp()[None, None, :, None] * gate_input
        )
    beta = raw_beta.float().sigmoid()
    current = state.float().clone()
    output = torch.empty_like(v, dtype=torch.float32)
    states = []
    for token in range(q.shape[1]):
        current *= gate[:, token].exp()[:, :, None, :]
        correction = v[:, token].float() - torch.einsum(
            "bhvk,bhk->bhv", current, k32[:, token]
        )
        correction *= beta[:, token, :, None]
        current += torch.einsum("bhv,bhk->bhvk", correction, k32[:, token])
        output[:, token] = torch.einsum("bhvk,bhk->bhv", current, q32[:, token])
        states.append(current.clone())
    return (output, states)


@pytest.mark.parametrize(
    "width,accepted",
    [
        (width, accepted)
        for width in (1, 2, 3, 4, 8, 16)
        for accepted in range(width + 1)
    ],
)
@pytest.mark.parametrize("dim", [64, 128])
def test_vllm_recoverssm_verify_and_commit(width: int, accepted: int, dim: int) -> None:
    torch.manual_seed(7331)
    device = "cuda"
    batch, heads = (8 if width == 16 else 4, 12)
    q = torch.randn(batch, width, heads, dim, device=device).bfloat16()
    k = torch.randn_like(q)
    v = torch.randn_like(q).mul_(0.2)
    raw_g = torch.randn_like(q)
    raw_beta = torch.randn(batch, width, heads, device=device).bfloat16()
    A_log = torch.randn(heads, device=device).mul_(0.1)
    dt_bias = torch.randn(heads, dim, device=device).mul_(0.1)
    initial_state = torch.randn(batch, heads, dim, dim, device=device).mul_(0.02)
    expected_out, expected_states = _reference(
        q, k, v, raw_g, raw_beta, A_log, dt_bias, initial_state, -5.0
    )
    state_pool = torch.zeros(
        batch + 1, heads, dim, dim, device=device, dtype=torch.float32
    )
    state_pool[1:].copy_(initial_state)
    correction_cache = torch.empty(
        batch + 1, heads, width, dim, device=device, dtype=torch.float32
    )
    kg_cache = torch.empty(
        batch + 1, heads, width, 2 * dim, device=device, dtype=torch.bfloat16
    )
    state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    query_start_loc = torch.arange(
        0, (batch + 1) * width, width, device=device, dtype=torch.int32
    )
    packed_q = q.reshape(1, batch * width, heads, dim)
    packed_k = k.reshape_as(packed_q)
    packed_v = v.reshape_as(packed_q)
    packed_g = raw_g.reshape_as(packed_q)
    packed_beta = raw_beta.reshape(1, batch * width, heads)
    out = torch.empty_like(packed_v)
    vllm_triton_kda_recoverssm_verify(
        packed_q,
        packed_k,
        packed_v,
        packed_g,
        packed_beta,
        A_log,
        dt_bias,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        record_metadata=None,
        record_stamps=None,
    )
    num_accepted = torch.full((batch,), accepted, device=device, dtype=torch.int32)
    plan = tuple(
        (torch.empty(batch, device=device, dtype=torch.int32) for _ in range(4))
    )
    vllm_triton_kda_recoverssm_commit(
        packed_q,
        packed_k,
        packed_v,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kg_cache=kg_cache,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state_pool,)),
        correction_cache_base_addrs=_device_addresses((correction_cache,)),
        kg_cache_base_addrs=_device_addresses((kg_cache,)),
        spec_query_len=width,
        record_metadata=None,
        record_stamps=None,
        gate_parameters=(A_log, dt_bias, -5.0),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out.float().reshape_as(expected_out), expected_out, atol=0.02, rtol=0.02
    )
    expected_state = initial_state if accepted == 0 else expected_states[accepted - 1]
    torch.testing.assert_close(state_pool[1:], expected_state, atol=0.03, rtol=0.02)
    torch.testing.assert_close(
        kg_cache[:batch, ..., :dim], k.transpose(1, 2), atol=0, rtol=0
    )
    torch.testing.assert_close(
        kg_cache[:batch, ..., dim:], raw_g.transpose(1, 2), atol=0, rtol=0
    )
    torch.testing.assert_close(state_pool[1:], expected_state, atol=2e-06, rtol=2e-05)


def _recover_window(
    qkv: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    accepted: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return output and actual producer records after current-window recovery."""
    batch, heads, dim, _ = state[1:].shape
    q, k, v = (
        value.reshape(1, batch * width, heads, dim) for value in qkv.chunk(3, dim=-1)
    )
    gates = gate.reshape_as(q)
    betas = beta.reshape(1, batch * width, heads)
    u = torch.empty(batch + 1, heads, width, dim, device=q.device)
    kd = torch.empty(batch + 1, heads, width, 2 * dim, device=q.device, dtype=q.dtype)
    starts = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * width
    out = torch.empty_like(v)
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        gates,
        betas,
        A_log,
        dt_bias,
        checkpoint_state=state,
        correction_cache=u,
        kg_cache=kd,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        record_metadata=None,
        record_stamps=None,
    )
    plan = tuple((torch.empty_like(indices) for _ in range(4)))
    vllm_triton_kda_recoverssm_commit(
        q,
        k,
        v,
        checkpoint_state=state,
        correction_cache=u,
        kg_cache=kd,
        state_indices=indices,
        query_start_loc=starts,
        num_accepted_tokens=accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state,)),
        correction_cache_base_addrs=_device_addresses((u,)),
        kg_cache_base_addrs=_device_addresses((kd,)),
        spec_query_len=width,
        record_metadata=None,
        record_stamps=None,
        gate_parameters=(A_log, dt_bias, -5.0),
    )
    return (out, u, kd)


@pytest.mark.parametrize(
    "batch,width,capacity",
    [(4, width, capacity) for width in (1, 4) for capacity in (8, 16, 32)]
    + [(batch, 4, 8) for batch in (1, 2, 3, 8, 16, 32)],
)
def test_recoverssm_raw_records_over_rounds(
    batch: int, width: int, capacity: int
) -> None:
    """Check raw records across wraps, slot reuse and retained prefixes."""
    _check_raw_ring_rounds(batch, width, capacity, capacity - 1)


@pytest.mark.parametrize("batch,width", [(4, 2), (8, 3), (16, 4), (32, 4)])
@pytest.mark.parametrize("start_position", [2**31 - 2, 2**32 - 2])
def test_recoverssm_records_with_large_positions(
    batch: int, width: int, start_position: int
) -> None:
    """Recover accepted prefixes across INT32 boundaries and ring wraps."""
    _check_raw_ring_rounds(batch, width, 7, start_position)


def test_record_chain_uses_verify_inputs() -> None:
    """Perturb verify inputs and check the shared recovery records change too."""
    torch.manual_seed(173)
    batch, width, heads, dim = (4, 4, 12, 128)
    shape = (1, batch * width, heads, dim)
    q, k, v, raw_gate = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, batch * width, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.01
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    results = []
    for perturb in (False, True):
        u = torch.empty(batch, heads, width, dim, device="cuda")
        kd = torch.empty(
            batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16
        )
        out = torch.empty_like(v)
        vllm_triton_kda_recoverssm_verify(
            q + 1 if perturb else q,
            k * 2 if perturb else k,
            v - 1 if perturb else v,
            raw_gate + 1 if perturb else raw_gate,
            beta,
            A,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kd,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            record_metadata=None,
            record_stamps=None,
        )
        results.append((out, u, kd))
    assert not torch.equal(results[0][0], results[1][0])
    assert not torch.equal(results[0][1], results[1][1])
    assert not torch.equal(results[0][2], results[1][2])


@pytest.mark.parametrize("batch", [4, 6, 8, 9, 16, 17, 32])
@pytest.mark.parametrize("captured", [False, True])
def test_recoverssm_ragged_prefix_and_null_rows(batch: int, captured: bool) -> None:
    """Verify valid prefixes and preserve unwritten records in every value tile."""
    torch.manual_seed(1209)
    lengths = ([4, 2, 0, 1, 3, 4] * ((batch + 5) // 6))[:batch]
    lengths[-1] = 4
    heads, dim, width = (12, 128, 4)
    starts_cpu = [0]
    for length in lengths:
        starts_cpu.append(starts_cpu[-1] + length)
    rows = starts_cpu[-1]
    shape = (1, rows, heads, dim)
    q, k, v, g = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, rows, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    initial = state.clone()
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    indices[-1] = 0
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    u = torch.full((batch, heads, width, dim), float("nan"), device="cuda")
    kd = torch.full(
        (batch, heads, width, 2 * dim),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = torch.full_like(v, float("nan"))
    metadata = torch.zeros(batch, 3, dtype=torch.int64, device="cuda")
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 1
    stamps = torch.full((batch, width, 2), -1, dtype=torch.int64, device="cuda")
    record_k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-06)
    record_v = v.float()
    record_d = torch.exp(
        -5.0 * torch.sigmoid(A.exp()[None, None, :, None] * (g.float() + bias))
    )

    def verify() -> None:
        vllm_triton_kda_recoverssm_verify(
            q,
            k,
            v,
            g,
            beta,
            A,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kd,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            record_metadata=metadata,
            record_stamps=stamps,
        )

    verify()
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            verify()
        # Only replay writes may satisfy the checks below.
        out.fill_(float("nan"))
        u.fill_(float("nan"))
        kd.fill_(float("nan"))
        stamps.fill_(-1)
        graph.replay()
    torch.testing.assert_close(state, initial, atol=0, rtol=0)
    for request, length in enumerate(lengths):
        begin, end = starts_cpu[request : request + 2]
        if request == batch - 1:
            torch.testing.assert_close(
                out[:, begin:end], torch.zeros_like(out[:, begin:end]), atol=0, rtol=0
            )
            valid = 0
        else:
            valid = length
            if length:
                expected_out, _ = _reference(
                    q[:, begin:end],
                    k[:, begin:end],
                    v[:, begin:end],
                    g[:, begin:end],
                    beta[:, begin:end],
                    A,
                    bias,
                    initial[request + 1 : request + 2],
                    -5.0,
                )
                torch.testing.assert_close(
                    out[:, begin:end].float(), expected_out, atol=2e-3, rtol=8e-3
                )
                current = initial[request + 1].clone()
                for offset in range(length):
                    token = begin + offset
                    key = record_k[0, token]
                    decay = record_d[0, token]
                    current = current * decay[:, None, :]
                    correction = (
                        record_v[0, token] - (current * key[:, None, :]).sum(-1)
                    ) * beta[0, token].float().sigmoid()[:, None]
                    torch.testing.assert_close(
                        u[request, :, offset], correction, atol=1e-05, rtol=0.001
                    )
                    torch.testing.assert_close(
                        kd[request, :, offset, :dim], k[0, token], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        kd[request, :, offset, dim:], g[0, token], atol=0, rtol=0
                    )
                    current = current + correction[:, :, None] * key[:, None, :]
        assert torch.isnan(u[request, :, valid:]).all()
        assert torch.isnan(kd[request, :, valid:]).all()
        assert (stamps[request, valid:] == -1).all()
        if valid:
            torch.testing.assert_close(
                stamps[request, :valid, 0],
                torch.arange(valid, device="cuda", dtype=torch.int64),
                atol=0,
                rtol=0,
            )
            assert (stamps[request, :valid, 1] == 1).all()


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize(
    "width,capacity", [(4, 4), (4, 32), (16, 16), (16, 19), (16, 32), (32, 32)]
)
@pytest.mark.parametrize("heads", [2, 8])
@pytest.mark.parametrize("buffered", [False, True])
def test_batched_recoverssm_commit_groups_alias_and_padding(
    captured: bool, width: int, capacity: int, heads: int, buffered: bool
) -> None:
    """Replay both recovery variants with changing lengths and wrapped records."""
    from tokenspeed_kernel.ops.attention.kda import KdaReplayHistoryBatch
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    torch.manual_seed(982)
    layers, batch, dim, pages = (3, 6, 128, 12)
    channels = heads * dim
    groups_cpu = [1, 0, 1]
    read_cpu = [[1, 2, 3, 4, 5, -1], [5, 4, 3, 2, 1, -1]]
    write_cpu = [[6, 2, 7, 4, 8, -1], [6, 4, 7, 2, 8, -1]]
    accepted_cpu = [0, 1, 2, width - 1, width, 0]
    groups = torch.tensor(groups_cpu, device="cuda", dtype=torch.int32)
    reads = torch.tensor(read_cpu, device="cuda", dtype=torch.int32)
    writes = torch.tensor(write_cpu, device="cuda", dtype=torch.int32)
    accepted = torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32)
    states = torch.randn(layers, pages, heads, dim, dim, device="cuda") * 0.02
    conv = torch.randn(
        layers, pages, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    raw = torch.randn(
        layers, batch * width, 3 * channels + 16, device="cuda", dtype=torch.bfloat16
    )
    corrections = (
        torch.randn(layers, batch * capacity, channels + 16, device="cuda") * 0.01
    )
    kd = torch.randn(layers, batch * capacity, 2 * channels + 16, device="cuda") * 0.02
    kd[:, :, : 2 * channels].view(layers, batch * capacity, heads, 2, dim)[
        :, :, :, 1
    ].sigmoid_()
    a_log = torch.randn(layers, heads, device="cuda") * 0.1
    dt_bias = torch.randn(layers, heads, dim, device="cuda") * 0.1
    kd = kd.bfloat16()
    record_start = 2**32 + capacity - 2
    metadata = torch.tensor(
        [[request, 7, record_start] for request in range(batch)],
        device="cuda",
        dtype=torch.int64,
    )
    stamps = torch.empty(layers, batch, capacity, 2, device="cuda", dtype=torch.int64)
    positions = record_start + torch.arange(capacity, device="cuda", dtype=torch.int64)
    stamps[:, :, positions % capacity, 0] = positions
    stamps[..., 1] = 7
    stamps[:, -1].fill_(-1)
    seed_state, seed_conv = (states.clone(), conv.clone())
    descriptors = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    for column, storage in (
        (0, raw),
        (2, conv),
        (3, corrections),
        (5, kd),
        (8, states),
        (6, a_log),
        (7, dt_bias),
    ):
        descriptors[:, column] = _device_addresses(tuple(storage.unbind()))

    retained_capacity = max(8, 2 * capacity)
    retained_u = torch.randn(
        layers, pages, heads * retained_capacity * dim + 16, device="cuda"
    )[:, :, : heads * retained_capacity * dim].view(
        layers, pages, heads, retained_capacity, dim
    )
    retained_key = torch.randn_like(retained_u)
    retained_decay = torch.rand(
        layers, pages, heads * retained_capacity * dim + 16, device="cuda"
    )[:, :, : heads * retained_capacity * dim].view(
        layers, pages, heads, retained_capacity, dim
    )
    retained_lengths = torch.zeros(layers, pages, 5, device="cuda", dtype=torch.int32)[
        :, :, 0
    ]
    append_offsets = (
        torch.arange(layers * (batch + 3), device="cuda", dtype=torch.int32).view(
            layers, batch + 3
        )
        % 3
    )
    for layer, group in enumerate(groups_cpu):
        for row, (read_page, write_page) in enumerate(
            zip(read_cpu[group], write_cpu[group])
        ):
            if read_page != write_page:
                append_offsets[layer, row] = 0
    history_descriptors = torch.stack(
        [
            _device_addresses(tuple(tensor.unbind()))
            for tensor in (retained_u, retained_key, retained_decay, retained_lengths)
        ],
        dim=1,
    )
    history_batch = (
        KdaReplayHistoryBatch(
            descriptors=history_descriptors,
            append_offsets=append_offsets,
            capacity=retained_capacity,
            correction_page_stride=retained_u.stride(1),
            key_page_stride=retained_key.stride(1),
            decay_page_stride=retained_decay.stride(1),
            length_page_stride=retained_lengths.stride(1),
        )
        if buffered
        else None
    )
    seed_u, seed_key, seed_decay, seed_lengths = (
        retained_u.clone(),
        retained_key.clone(),
        retained_decay.clone(),
        retained_lengths.clone(),
    )

    def launch() -> None:
        batched_recoverssm_commit(
            descriptors,
            groups,
            reads,
            writes,
            accepted,
            draft_token_num=width,
            num_heads=heads,
            head_dim=dim,
            qkv_stride=raw.stride(1),
            conv_stride=conv.stride(1),
            correction_stride=corrections.stride(1),
            key_gate_stride=kd.stride(1),
            key_gate_format="raw_key_gate",
            state_stride=states.stride(1),
            conv_width=4,
            record_metadata=metadata,
            record_stamps=stamps,
            lower_bound=-5.0,
            history=history_batch,
        )

    launch()
    graph = None
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
    for accepted_cpu in (
        accepted_cpu,
        [width, 0, width - 1, 1, 2, 0],
        [1, 3, 0, width, 4, 0],
    ):
        accepted.copy_(torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32))
        expected_state, expected_conv = (seed_state.clone(), seed_conv.clone())
        expected_u, expected_key, expected_decay, expected_lengths = (
            seed_u.clone(),
            seed_key.clone(),
            seed_decay.clone(),
            seed_lengths.clone(),
        )
        for layer, group in enumerate(groups_cpu):
            for request, count in enumerate(accepted_cpu[:-1]):
                source, dest = (read_cpu[group][request], write_cpu[group][request])
                state = seed_state[layer, source].clone()
                history = seed_conv[layer, source].clone()
                for token in range(count):
                    row = request * capacity + (record_start + token) % capacity
                    key, decay = (
                        kd[layer, row, : 2 * channels].view(heads, 2, dim).unbind(1)
                    )
                    correction = corrections[layer, row, :channels].view(heads, dim)
                    key = key.float()
                    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-06)
                    decay = torch.exp(
                        -5.0
                        * torch.sigmoid(
                            a_log[layer].exp()[:, None]
                            * (decay.float() + dt_bias[layer])
                        )
                    )
                    state = state * decay[:, None, :]
                    state = state + correction[:, :, None] * key[:, None, :]
                    history = torch.cat(
                        (
                            history[:, 1:],
                            raw[layer, request * width + token, : 3 * channels, None],
                        ),
                        dim=1,
                    )
                if buffered:
                    offset = int(append_offsets[layer, request].item())
                    for token in range(count):
                        row = request * capacity + (record_start + token) % capacity
                        raw_key, raw_gate = (
                            kd[layer, row, : 2 * channels]
                            .float()
                            .view(heads, 2, dim)
                            .unbind(1)
                        )
                        key = raw_key * torch.rsqrt(
                            raw_key.square().sum(-1, keepdim=True) + 1e-6
                        )
                        decay = torch.exp(
                            -5.0
                            * torch.sigmoid(
                                a_log[layer].exp()[:, None]
                                * (raw_gate + dt_bias[layer])
                            )
                        )
                        expected_decay[layer, dest, :, offset + token] = decay
                        expected_u[layer, dest, :, offset + token] = corrections[
                            layer, row, :channels
                        ].view(heads, dim)
                        expected_key[layer, dest, :, offset + token] = key
                    expected_lengths[layer, dest] = offset + count
                else:
                    expected_state[layer, dest] = state
                expected_conv[layer, dest] = history
        states.copy_(seed_state)
        conv.copy_(seed_conv)
        retained_u.copy_(seed_u)
        retained_key.copy_(seed_key)
        retained_decay.copy_(seed_decay)
        retained_lengths.copy_(seed_lengths)
        if graph is None:
            launch()
        else:
            graph.replay()
        torch.testing.assert_close(states, expected_state, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        if buffered:
            torch.testing.assert_close(retained_u, expected_u, atol=0, rtol=0)
            torch.testing.assert_close(retained_key, expected_key, atol=2e-7, rtol=2e-5)
            torch.testing.assert_close(
                retained_decay, expected_decay, atol=2e-7, rtol=2e-5
            )
            torch.testing.assert_close(
                retained_lengths, expected_lengths, atol=0, rtol=0
            )
        states = torch.empty_like(states)
        conv = torch.empty_like(conv)
        descriptors[:, 8] = _device_addresses(tuple(states.unbind()))
        descriptors[:, 2] = _device_addresses(tuple(conv.unbind()))


@pytest.mark.parametrize("batch", [1, 7, 33])
def test_commit_state_groups_live_slots_strides_and_graph(batch: int) -> None:
    from tokenspeed_kernel.ops.attention.kda.triton import commit_state_groups

    tables = [
        torch.arange(batch * (columns + 3), device="cuda", dtype=torch.int32).view(
            batch, columns + 3
        )[:, :columns]
        + group * 1000
        for group, columns in enumerate((2, 3, 5))
    ]
    tables = [torch.nn.functional.pad(table, (0, 2))[:, :-2] for table in tables]
    sources = [
        torch.arange(batch, device="cuda", dtype=torch.int32) + group * 100
        for group in range(3)
    ]
    descriptors = torch.tensor(
        [
            (
                (table.data_ptr() - tables[0].data_ptr()) // tables[0].element_size(),
                table.stride(0),
                table.shape[1],
                source.data_ptr(),
            )
            for table, source in zip(tables, sources, strict=True)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    accepted = torch.arange(batch, device="cuda", dtype=torch.int32) % 6
    pages = torch.empty((3, batch), device="cuda", dtype=torch.int32)
    reads = torch.empty_like(pages)
    steps = torch.empty(batch, device="cuda", dtype=torch.int32)

    def launch() -> None:
        commit_state_groups(
            accepted,
            tables[0],
            descriptors,
            batch_size=batch,
            draft_tokens=4,
            pages_out=pages,
            reads_out=reads,
            steps_out=steps,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for iteration in range(2):
        if iteration:
            accepted.add_(1)
            for source in sources:
                source.add_(7)
            tables[1].add_(23)
        graph.replay()
        torch.testing.assert_close(reads, torch.stack(sources), atol=0, rtol=0)
        expected_steps = accepted.clamp(1, 4)
        torch.testing.assert_close(steps, expected_steps, atol=0, rtol=0)
        for group, table in enumerate(tables):
            expected = table[:, 1]
            torch.testing.assert_close(pages[group], expected, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [4, 8, 12, 16, 32, 64, 128])
@pytest.mark.parametrize("layers", [1, 69])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("seed", [0, 1573])
def test_recoverssm_gate_recovery_across_layers(
    rows: int, layers: int, lower_bound: float | None, seed: int
) -> None:
    """Check raw gate recovery across batch, layer and gate configurations."""
    _check_gate_recovery(rows, layers, lower_bound, seed)


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("combined", [False, True])
def test_verify_derived_records_with_serving_producers(
    batch: int, combined: bool
) -> None:
    """Commit each accepted prefix against an independent FP32 recurrence."""
    torch.manual_seed(814)
    heads, dim, width = (12, 128, 4)
    rows, channels = (batch * width, heads * dim)
    kwargs = {"device": "cuda", "dtype": torch.bfloat16}
    raw = torch.randn(rows, 3 * channels, **kwargs)
    conv_weights = torch.randn(3 * channels, 4, **kwargs) * 0.3
    conv_state = torch.randn(batch + 1, 3 * channels, 3, **kwargs)
    fa = torch.randn(rows, dim, **kwargs)
    fb = torch.randn(channels, dim, **kwargs) * 0.05
    beta = torch.randn(rows, heads, **kwargs)
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    captured = torch.empty_like(raw)
    from tokenspeed_kernel.ops.attention.kda import (
        kda_verify_conv_gate,
        kda_verify_conv_update,
    )

    if combined:
        qkv, gate = kda_verify_conv_gate(
            raw,
            conv_weights,
            conv_state,
            indices,
            fa,
            fb,
            draft_token_num=width,
            replay_mixed_qkv=captured,
        )
    else:
        captured.copy_(raw)
        qkv = kda_verify_conv_update(
            raw,
            conv_weights,
            conv_state,
            indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            recurrent_layout="v_major",
        )
        gate = torch.nn.functional.linear(fa, fb)
    torch.testing.assert_close(captured, raw, atol=0, rtol=0)
    q, k, _ = (x.float().reshape(batch, width, heads, dim) for x in qkv.chunk(3, -1))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-06) * dim ** (-0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-06)
    d = torch.exp(
        -5.0
        * torch.sigmoid(
            a.exp()[None, None, :, None]
            * (gate.float().reshape(batch, width, heads, dim) + bias)
        )
    )
    v = qkv.chunk(3, -1)[2].float().reshape(batch, width, heads, dim)
    b = beta.float().sigmoid().reshape(batch, width, heads)
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    for accepted_count in (1, 2, 3, 4):
        reference = state[1:].clone()
        outputs, states = ([], [])
        for token in range(width):
            reference = reference * d[:, token, :, None, :]
            u = v[:, token] - (reference * k[:, token, :, None, :]).sum(-1)
            u = u * b[:, token, :, None]
            reference = reference + u[..., None] * k[:, token, :, None, :]
            outputs.append((reference * q[:, token, :, None, :]).sum(-1))
            states.append(reference.clone())
        accepted = torch.full(
            (batch,), accepted_count, device="cuda", dtype=torch.int32
        )
        out, _, _ = _recover_window(
            qkv, gate, beta, a, bias, state, indices, accepted, width
        )
        torch.testing.assert_close(
            state[1:], states[accepted_count - 1], atol=2e-06, rtol=2e-05
        )
        expected = torch.stack(outputs, dim=1).bfloat16()
        torch.testing.assert_close(
            out.reshape_as(expected), expected, atol=0.002, rtol=0.008
        )


def _check_raw_ring_rounds(
    batch: int, width: int, capacity: int, start_position: int
) -> None:
    torch.manual_seed(781)
    heads, dim = 2, 128
    device = "cuda"
    initial = torch.randn(batch + 1, heads, dim, dim, device=device) * 0.02
    state, checkpoint = initial.clone(), initial.clone()
    reference = initial[1:].clone()
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    starts = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    u = torch.full((batch, heads, capacity, dim), float("nan"), device=device)
    kg = torch.full(
        (batch, heads, capacity, 2 * dim),
        float("nan"),
        device=device,
        dtype=torch.bfloat16,
    )
    stamps = torch.full((batch, capacity, 2), -1, device=device, dtype=torch.int64)
    slots = [(i + 1) % batch for i in range(batch)]
    positions = [start_position for _ in range(batch)]
    checkpoint_positions = positions.copy()
    generations = [1] * batch
    a = torch.randn(heads, device=device) * 0.1
    bias = torch.randn(heads, dim, device=device) * 0.1
    for step in range(128):
        if step and step % 43 == 0:
            slots = [(slot + 1) % batch for slot in slots]
            state.normal_(std=0.02)
            reference = state[1:].clone()
            checkpoint.copy_(state)
            checkpoint_positions = positions.copy()
            generations = [g + 1 for g in generations]
        for request in range(batch):
            if positions[request] - checkpoint_positions[request] + width > capacity:
                checkpoint[request + 1].copy_(state[request + 1])
                checkpoint_positions[request] = positions[request]
                generations[request] += 1
        q, k, v, g = [
            torch.randn(batch, width, heads, dim, device=device, dtype=torch.bfloat16)
            * 0.2
            for _ in range(4)
        ]
        beta = torch.randn(batch, width, heads, device=device, dtype=torch.bfloat16)
        expected_out, expected_states = _reference(
            q, k, v, g, beta, a, bias, reference, -5.0
        )
        counts = [(step + i) % (width + 1) for i in range(batch)]
        meta = torch.tensor(
            list(zip(slots, generations, positions)), device=device, dtype=torch.int64
        )
        packed = [x.reshape(1, batch * width, heads, dim) for x in (q, k, v, g)]
        out = torch.empty_like(packed[2])
        vllm_triton_kda_recoverssm_verify(
            *packed,
            beta.reshape(1, batch * width, heads),
            a,
            bias,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kg,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            record_metadata=meta,
            record_stamps=stamps,
        )
        torch.testing.assert_close(
            out.reshape_as(expected_out).float(), expected_out, atol=2e-3, rtol=8e-3
        )
        for i, count in enumerate(counts):
            if count:
                reference[i].copy_(expected_states[count - 1][i])
            for token in range(count, width):
                row = (positions[i] + token) % capacity
                u[slots[i], :, row].fill_(float("nan"))
                kg[slots[i], :, row].fill_(float("nan"))
            positions[i] += count
        retained = torch.tensor(
            [p - c for p, c in zip(positions, checkpoint_positions)],
            device=device,
            dtype=torch.int32,
        )
        recovery_meta = torch.tensor(
            list(zip(slots, generations, checkpoint_positions)),
            device=device,
            dtype=torch.int64,
        )
        state.copy_(checkpoint)
        plans = [torch.empty_like(indices) for _ in range(4)]
        # Recovery spans retained accepted tokens, not merely the latest window.
        recovery_starts = (
            torch.arange(batch + 1, device=device, dtype=torch.int32) * capacity
        )
        vllm_triton_kda_recoverssm_commit(
            *packed[:3],
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kg,
            state_indices=indices,
            query_start_loc=recovery_starts,
            num_accepted_tokens=retained,
            commit_lens=plans[0],
            final_state_indices=plans[1],
            boundary_state_indices=plans[2],
            boundary_recovery_lens=plans[3],
            state_base_addrs=_device_addresses((state,)),
            correction_cache_base_addrs=_device_addresses((u,)),
            kg_cache_base_addrs=_device_addresses((kg,)),
            spec_query_len=capacity,
            record_metadata=recovery_meta,
            record_stamps=stamps,
            gate_parameters=(a, bias, -5.0),
        )
        torch.testing.assert_close(state[1:], reference, atol=2e-6, rtol=2e-5)


def _check_gate_recovery(
    rows: int, layers: int, lower_bound: float | None, seed: int
) -> None:
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    torch.manual_seed(seed)
    batch, width, heads, dim = rows // 4, 4, 2, 128
    channels = heads * dim
    state = torch.randn(layers, batch + 1, heads, dim, dim, device="cuda") * 0.02
    expected = state.clone()
    conv = torch.zeros(
        layers, batch + 1, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    raw = torch.zeros(layers, rows, 3 * channels, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(layers, rows, channels, device="cuda") * 0.02
    kg = (
        torch.randn(layers, rows, heads, 2, dim, device="cuda", dtype=torch.bfloat16)
        * 0.2
    )
    a = torch.randn(layers, heads, device="cuda") * 0.1
    bias = torch.randn(layers, heads, dim, device="cuda") * 0.1
    descriptors = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    for col, tensor in (
        (0, raw),
        (2, conv),
        (3, u),
        (5, kg),
        (6, a),
        (7, bias),
        (8, state),
    ):
        descriptors[:, col] = _device_addresses(tuple(tensor.unbind()))
    groups = torch.zeros(layers, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)[None]
    counts = torch.arange(batch, device="cuda", dtype=torch.int32) % width + 1
    for token in range(width):
        key = kg[:, token::width, :, 0].float()
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        gate_input = kg[:, token::width, :, 1].float() + bias[:, None]
        log_gate = (
            lower_bound * torch.sigmoid(a.exp()[:, None, :, None] * gate_input)
            if lower_bound is not None
            else -a.exp()[:, None, :, None] * torch.nn.functional.softplus(gate_input)
        )
        correction = u[:, token::width].reshape(layers, batch, heads, dim)
        updated = (
            expected[:, 1:] * log_gate.exp()[..., None, :]
            + correction[..., None] * key[..., None, :]
        )
        expected[:, 1:] = torch.where(
            (counts > token)[None, :, None, None, None], updated, expected[:, 1:]
        )
    batched_recoverssm_commit(
        descriptors,
        groups,
        indices,
        indices,
        counts,
        draft_token_num=width,
        num_heads=heads,
        head_dim=dim,
        qkv_stride=raw.stride(1),
        conv_stride=conv.stride(1),
        correction_stride=u.stride(1),
        key_gate_stride=kg.stride(1),
        key_gate_format="raw_key_gate",
        state_stride=state.stride(1),
        conv_width=4,
        record_metadata=None,
        record_stamps=None,
        lower_bound=lower_bound,
    )
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("width", [4, 16, 32])
@pytest.mark.parametrize("raw_gate", [-20.0, 20.0])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("captured", [False, True])
def test_recoverssm_parallel_recovery_extreme_decays(
    width: int, raw_gate: float, lower_bound: float | None, captured: bool
) -> None:
    """Recover every prefix with extreme decays, eagerly or in a CUDA graph."""
    torch.manual_seed(1893)
    batch, heads, dim = width + 1, 8, 128
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.02
    initial_state = state.clone()
    expected = initial_state.clone()
    u = torch.randn(batch, heads, width, dim, device="cuda") * 0.2
    kg = torch.empty(batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16)
    kg[..., :dim].normal_()
    kg[..., dim:].fill_(raw_gate)
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    counts = torch.arange(batch, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    gate_input = torch.tensor(raw_gate, device="cuda")
    gate = (
        -torch.nn.functional.softplus(gate_input)
        if lower_bound is None
        else lower_bound * torch.sigmoid(gate_input)
    )
    decay = gate.exp()
    for token in range(width):
        key = kg[:, :, token, :dim].float()
        key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        update = u[:, :, token, :, None] * key[:, :, None, :]
        expected[1:] = torch.where(
            (counts > token)[:, None, None, None],
            expected[1:] * decay + update,
            expected[1:],
        )
    # Unaccepted records may contain stale or undefined values.
    invalid = (
        torch.arange(width, device="cuda")[None, None, :, None]
        >= counts[:, None, None, None]
    )
    u.masked_fill_(invalid, float("nan"))
    kg.masked_fill_(invalid, float("nan"))
    placeholder = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    plan = [torch.empty_like(indices) for _ in range(4)]
    state_addresses = _device_addresses((state,))
    correction_addresses = _device_addresses((u,))
    key_gate_addresses = _device_addresses((kg,))

    def commit():
        state.copy_(initial_state)
        vllm_triton_kda_recoverssm_commit(
            placeholder,
            placeholder,
            placeholder,
            checkpoint_state=state,
            correction_cache=u,
            kg_cache=kg,
            state_indices=indices,
            query_start_loc=starts,
            num_accepted_tokens=counts,
            commit_lens=plan[0],
            final_state_indices=plan[1],
            boundary_state_indices=plan[2],
            boundary_recovery_lens=plan[3],
            state_base_addrs=state_addresses,
            correction_cache_base_addrs=correction_addresses,
            kg_cache_base_addrs=key_gate_addresses,
            spec_query_len=width,
            record_metadata=None,
            record_stamps=None,
            gate_parameters=(a, bias, lower_bound),
        )

    commit()
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit()
        graph.replay()
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("width", [16, 32])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_recoverssm_aligned_checkpoint_states(
    width: int, use_tensor_cores: bool, captured: bool
) -> None:
    """Aligned recovery preserves boundary, final, and masked states."""
    from tokenspeed_kernel._triton import triton
    from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
        _commit_kda_state_kernel,
    )

    torch.manual_seed(2423)
    batch, heads, dim = 5, 2, 128
    initial = torch.randn(1 + 3 * batch, heads, dim, dim, device="cuda") * 0.02
    state, expected = initial.clone(), initial.clone()
    corrections = torch.randn(batch, heads, width, dim, device="cuda") * 0.2
    key_gate = torch.randn(
        batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16
    )
    a_log = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    sources = [1, 2, 3, 4, 0]
    destinations = [6, 7, 3, 0, 10]
    boundaries = [11, 12, -1, 14, 15]
    counts = [width, width // 2, width, 0, width]
    boundary_lengths = [width // 2, width // 4, width // 2, 0, width // 2]
    for row, count in enumerate(counts):
        if sources[row] <= 0 or destinations[row] <= 0:
            corrections[row].fill_(float("nan"))
            key_gate[row].fill_(float("nan"))
            continue
        current = initial[sources[row]].clone()
        for token in range(count):
            key = key_gate[row, :, token, :dim].float()
            key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
            raw_gate = key_gate[row, :, token, dim:].float()
            decay = (
                -5.0 * torch.sigmoid(a_log.exp()[:, None] * (raw_gate + bias))
            ).exp()
            current = current * decay[:, None, :] + (
                corrections[row, :, token, :, None] * key[:, None, :]
            )
            if token + 1 == boundary_lengths[row] and boundaries[row] > 0:
                expected[boundaries[row]].copy_(current)
        expected[destinations[row]].copy_(current)
        corrections[row, :, count:].fill_(float("nan"))
        key_gate[row, :, count:].fill_(float("nan"))
    indices, final_indices, boundary_indices, commit_lengths, boundary_steps = (
        torch.tensor(values, device="cuda", dtype=torch.int32)
        for values in (sources, destinations, boundaries, counts, boundary_lengths)
    )
    state_addresses = _device_addresses((state,))
    correction_addresses = _device_addresses((corrections,))
    key_gate_addresses = _device_addresses((key_gate,))
    a_addresses = _device_addresses((a_log,))
    bias_addresses = _device_addresses((bias,))
    grid = lambda meta: (triton.cdiv(dim, meta["BV"]), batch, heads)

    def commit():
        state.copy_(initial)
        _commit_kda_state_kernel[grid](
            state_ref_ptr=state,
            state_base_addrs_ptr=state_addresses,
            state_block_stride=state.stride(0),
            correction_cache_ref_ptr=corrections,
            correction_cache_base_addrs_ptr=correction_addresses,
            correction_cache_block_stride=corrections.stride(0),
            kg_cache_ref_ptr=key_gate,
            kg_cache_base_addrs_ptr=key_gate_addresses,
            kg_cache_block_stride=key_gate.stride(0),
            state_indices_ptr=indices,
            commit_lens_ptr=commit_lengths,
            final_state_indices_ptr=final_indices,
            boundary_state_indices_ptr=boundary_indices,
            boundary_recovery_lens_ptr=boundary_steps,
            null_block_id=0,
            stride_state_head=state.stride(1),
            stride_state_v=state.stride(2),
            stride_state_k=state.stride(3),
            stride_correction_cache_head=corrections.stride(1),
            stride_correction_cache_pos=corrections.stride(2),
            stride_correction_cache_dim=corrections.stride(3),
            stride_kg_cache_head=key_gate.stride(1),
            stride_kg_cache_pos=key_gate.stride(2),
            stride_kg_cache_dim=key_gate.stride(3),
            stride_state_indices=indices.stride(0),
            group_indices_ptr=None,
            stride_state_group=0,
            stride_pointer_layer=1,
            record_metadata_ptr=None,
            record_stamps_ptr=None,
            stamp_layer_stride=0,
            RECORD_CAPACITY=width,
            RECORD_SLOTS=batch,
            K=dim,
            V=dim,
            BK=dim,
            BT=width,
            USE_TENSOR_CORES=use_tensor_cores,
            NUM_HEADS=heads,
            ALIGN_MODE=True,
            BATCH=batch,
            LAUNCH_DEPENDENT_KERNELS=False,
            a_ptr=a_addresses,
            bias_ptr=bias_addresses,
            USE_GATE_BASE_ADDRESSES=True,
            lower_bound=-5.0,
            USE_LOWER_BOUND=True,
            debug=True,
            sanitize_overflow=False,
            num_stages=2,
        )

    commit()
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            commit()
        graph.replay()
    torch.testing.assert_close(state, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("capacity", [4, 8, 16, 32])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize(
    "batch,lower_bound", [(4, -0.1), (8, -5.0), (32, -5.0), (32, None)]
)
def test_buffered_recoverssm_multiple_rounds(
    width: int,
    capacity: int,
    use_tensor_cores: bool,
    captured: bool,
    batch: int,
    lower_bound: float | None,
) -> None:
    from tokenspeed_kernel.ops.attention.kda import KdaReplayHistory, KdaReplayRecords
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        commit_recoverssm_history,
    )

    torch.manual_seed(9127)
    heads, dim = 4, 128
    pages = 2 * batch + 1
    q = torch.randn(batch, width, heads, dim, device="cuda").bfloat16()
    k, v, gate = (torch.empty_like(q) for _ in range(3))
    beta = torch.empty(batch, width, heads, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    initial = torch.randn(batch, heads, dim, dim, device="cuda") * 0.02
    state = torch.zeros(pages, heads, dim, dim, device="cuda")
    state[1 : batch + 1].copy_(initial)
    history_u = torch.full((pages, heads, capacity, dim), float("nan"), device="cuda")
    history_key = torch.full((pages, heads, capacity, dim), float("nan"), device="cuda")
    history_decay = torch.zeros(pages, heads, capacity, dim, device="cuda")
    lengths = torch.zeros(pages, dtype=torch.int32, device="cuda")
    read = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    write = read + batch
    read[-1] = 0
    write[-1] = 0
    offsets = torch.full((batch,), -1, dtype=torch.int32, device="cuda")
    history = KdaReplayHistory(
        correction=history_u,
        key=history_key,
        decay=history_decay,
        lengths=lengths,
        write_indices=write,
        append_offsets=offsets,
        use_tensor_cores=use_tensor_cores,
    )
    positions = torch.full((batch,), 137, dtype=torch.int64, device="cuda")
    metadata = torch.stack(
        (torch.arange(batch, device="cuda"), positions * 0, positions), dim=1
    )
    records = KdaReplayRecords(
        key_gate_format="raw_key_gate",
        correction=torch.empty(batch, heads, width, dim, device="cuda"),
        key_gate=torch.empty(
            batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16
        ),
        metadata=metadata,
        stamps=torch.full((batch, width, 2), -1, dtype=torch.int64, device="cuda"),
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32, device="cuda")
        * width,
        history=history,
    )
    accepted = torch.zeros(batch, dtype=torch.int32, device="cuda")
    output = torch.empty_like(v)

    def verify():
        vllm_triton_kda_recoverssm_verify(
            q.view(1, -1, heads, dim),
            k.view(1, -1, heads, dim),
            v.view(1, -1, heads, dim),
            gate.view(1, -1, heads, dim),
            beta.view(1, -1, heads),
            a,
            bias,
            checkpoint_state=state,
            correction_cache=records.correction,
            kg_cache=records.key_gate,
            query_start_loc=records.query_start_loc,
            state_indices=read,
            spec_query_len=width,
            lower_bound=lower_bound,
            out=output.view(1, -1, heads, dim),
            record_metadata=metadata,
            record_stamps=records.stamps,
            history=history,
        )

    def commit():
        commit_recoverssm_history(
            records, accepted, A_log=a, dt_bias=bias, lower_bound=lower_bound
        )

    for tensor in (k, v, gate, beta):
        tensor.normal_()
    verify()
    commit()
    if captured:
        forward_graph, commit_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(forward_graph):
            verify()
        with torch.cuda.graph(commit_graph):
            commit()
        forward, finish = forward_graph.replay, commit_graph.replay
    else:
        forward, finish = verify, commit

    reference = initial.clone()
    for round_index in range(128):
        for tensor in (q, k, v, gate, beta):
            tensor.normal_()
        v.mul_(0.2)
        if 40 <= round_index < 56:
            gate.fill_(20)
        elif 56 <= round_index < 72:
            gate.fill_(-20)
        metadata[:, 1].fill_(round_index + 1)
        metadata[:, 2].copy_(positions)
        counts = [(round_index + row) % (width + 1) for row in range(batch)]
        counts[-1] = 0
        accepted.copy_(torch.tensor(counts, dtype=torch.int32, device="cuda"))
        # A published composite is read-only. Resume into the other page,
        # as after a prefix hit or a retraction restore.
        rotating = round_index in (0, 70, 111)
        if rotating:
            write.copy_(torch.where(read <= batch, read + batch, read - batch))
            write[-1] = 0
            old_state = state[read[:-1]].clone()
            old_u = history_u[read[:-1]].clone()
            old_key = history_key[read[:-1]].clone()
            old_decay = history_decay[read[:-1]].clone()
            old_lengths = lengths[read[:-1]].clone()
        else:
            write.copy_(read)
        expected_out, expected_states = _reference(
            q, k, v, gate, beta, a, bias, reference, lower_bound
        )
        forward()
        finish()
        torch.testing.assert_close(
            output[:-1].float(), expected_out[:-1], atol=2e-3, rtol=8e-3
        )
        torch.testing.assert_close(
            output[-1], torch.zeros_like(output[-1]), atol=0, rtol=0
        )
        for row, count in enumerate(counts[:-1]):
            if count:
                reference[row].copy_(expected_states[count - 1][row])
        if rotating:
            torch.testing.assert_close(state[read[:-1]], old_state, atol=0, rtol=0)
            torch.testing.assert_close(
                history_u[read[:-1]], old_u, atol=0, rtol=0, equal_nan=True
            )
            torch.testing.assert_close(
                history_key[read[:-1]], old_key, atol=0, rtol=0, equal_nan=True
            )
            torch.testing.assert_close(
                history_decay[read[:-1]], old_decay, atol=0, rtol=0
            )
            torch.testing.assert_close(lengths[read[:-1]], old_lengths, atol=0, rtol=0)
        read.copy_(write)
        positions.add_(accepted)

    # Materialize the composite without accepting a candidate, then compare
    # the actual FP32 state to the independent sequential recurrence oracle.
    write.copy_(torch.where(read <= batch, read + batch, read - batch))
    write[-1] = 0
    accepted.zero_()
    metadata[:, 1].add_(1)
    metadata[:, 2].copy_(positions)
    forward()
    finish()
    torch.testing.assert_close(state[write[:-1]], reference[:-1], atol=3e-6, rtol=5e-5)
    torch.testing.assert_close(state[0], torch.zeros_like(state[0]), atol=0, rtol=0)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("capacity", [16, 32])
@pytest.mark.parametrize("rotate", [False, True])
@pytest.mark.parametrize("lower_bound", [-0.1, None])
def test_buffered_state_cancellation_against_fp64(
    width: int, capacity: int, rotate: bool, lower_bound: float | None
) -> None:
    from tokenspeed_kernel.ops.attention.kda import KdaReplayHistory

    torch.manual_seed(577)
    heads, dim, retained = 2, 128, capacity // 2
    state = torch.zeros(3, heads, dim, dim, device="cuda")
    u = torch.randn(3, heads, capacity, dim, device="cuda") * 0.2
    w = torch.randn_like(u)
    w *= torch.rsqrt(w.square().sum(-1, keepdim=True) + 1e-6)
    decay = torch.rand_like(u) * 0.1 + 0.9
    # A nearly cancelled checkpoint exposes errors hidden by BF16 outputs or
    # by checking only the final state after strongly contracting updates.
    product = torch.zeros_like(state[1], dtype=torch.float64)
    total_decay = torch.ones_like(decay[1, :, 0], dtype=torch.float64)
    for token in range(retained):
        token_decay = decay[1, :, token].double()
        product *= token_decay[:, None, :]
        product += u[1, :, token, :, None].double() * w[1, :, token, None, :].double()
        total_decay *= token_decay
    state[1].copy_((-product / total_decay[:, None, :]).float())
    state[1].add_(torch.randn_like(state[1]) * 1e-4)
    current = state[1].double()
    for token in range(retained):
        current *= decay[1, :, token, None, :].double()
        current += u[1, :, token, :, None].double() * w[1, :, token, None, :].double()
    lengths = torch.tensor([0, retained, 0], device="cuda", dtype=torch.int32)
    read = torch.tensor([1], device="cuda", dtype=torch.int32)
    write = torch.tensor([2 if rotate else 1], device="cuda", dtype=torch.int32)
    offsets = torch.empty(1, device="cuda", dtype=torch.int32)
    history = KdaReplayHistory(u, w, decay, lengths, write, offsets, True)
    q = torch.randn(1, width, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, heads, dim, device="cuda", dtype=torch.bfloat16).repeat(
        1, width, 1, 1
    )
    v = (torch.randn_like(q).float() * 1e-4).bfloat16()
    raw_gate = torch.full_like(q, -20)
    raw_beta = torch.full((1, width, heads), 10, device="cuda", dtype=torch.bfloat16)
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    out = torch.empty_like(q, dtype=torch.float32)
    correction = torch.empty(1, heads, width, dim, device="cuda")
    kg = torch.empty(1, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16)
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        a,
        bias,
        checkpoint_state=state,
        correction_cache=correction,
        kg_cache=kg,
        query_start_loc=torch.tensor([0, width], device="cuda", dtype=torch.int32),
        state_indices=read,
        spec_query_len=width,
        lower_bound=lower_bound,
        out=out,
        record_metadata=None,
        record_stamps=None,
        history=history,
    )
    expected_out, expected_u = [], []
    for token in range(width):
        key = k[0, token].double()
        key *= torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        query = q[0, token].double()
        query *= torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
        raw = raw_gate[0, token].double() + bias.double()
        gate = (
            lower_bound * torch.sigmoid(a.double().exp()[:, None] * raw)
            if lower_bound is not None
            else -a.double().exp()[:, None] * torch.nn.functional.softplus(raw)
        )
        current *= gate.exp()[:, None, :]
        delta = v[0, token].double() - torch.einsum("hvk,hk->hv", current, key)
        delta *= raw_beta[0, token].double().sigmoid()[:, None]
        current += delta[:, :, None] * key[:, None, :]
        expected_u.append(delta)
        expected_out.append(torch.einsum("hvk,hk->hv", current, query))
    torch.testing.assert_close(
        correction[0].double(), torch.stack(expected_u, 1), atol=2e-7, rtol=2e-4
    )
    torch.testing.assert_close(
        out[0].double(), torch.stack(expected_out), atol=3e-8, rtol=2e-4
    )


@pytest.mark.parametrize("batch,width", [(1, 1), (1, 4), (17, 1), (17, 4)])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "alias"])
@pytest.mark.parametrize("captured_graph", [False, True])
def test_verify_convolution_captures_original_rows(
    batch: int, width: int, layout: str, captured_graph: bool
) -> None:
    from tokenspeed_kernel.ops.attention.kda import kda_verify_conv_update

    torch.manual_seed(5821)
    heads, dim = 2, 96
    rows, channels = batch * width, 3 * heads * dim
    options = {"device": "cuda", "dtype": torch.bfloat16}
    raw_storage = torch.randn(
        rows, channels + (0 if layout == "contiguous" else 11), **options
    )
    raw = raw_storage[:, :channels]
    replay_storage = torch.full((rows + 3, channels + 23), -17.0, **options)
    replay = raw if layout == "alias" else replay_storage[:rows, :channels]
    weights = torch.randn(channels, 4, **options) * 0.3
    states = torch.randn(batch + 1, channels + 13, 3, **options)
    original_states = states.clone()
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    indices[0] = -1
    if batch > 1:
        indices[1] = 0
        indices[2:4] = 2

    def convolve(destination: torch.Tensor | None) -> torch.Tensor:
        return kda_verify_conv_update(
            raw,
            weights,
            states,
            indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            recurrent_layout="v_major",
            replay_mixed_qkv=destination,
        )

    result = convolve(replay)
    if captured_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = convolve(replay)
    for _ in range(2):
        raw.copy_(torch.randn_like(raw))
        raw.view(torch.int16)[0, :7] = torch.tensor(
            [0, -32768, 32640, -128, 32705, 1, 127],
            device="cuda",
            dtype=torch.int16,
        )
        expected = convolve(None)
        if captured_graph:
            graph.replay()
        else:
            result = convolve(replay)
        torch.testing.assert_close(
            result.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
        )
        torch.testing.assert_close(
            replay.view(torch.int16), raw.view(torch.int16), atol=0, rtol=0
        )
        torch.testing.assert_close(states, original_states, atol=0, rtol=0)
        if layout != "alias":
            torch.testing.assert_close(
                replay_storage[:, channels:],
                torch.full_like(replay_storage[:, channels:], -17.0),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                replay_storage[rows:],
                torch.full_like(replay_storage[rows:], -17.0),
                atol=0,
                rtol=0,
            )


@pytest.mark.parametrize("batch", [4, 16])
@pytest.mark.parametrize("partition", [(1,), (3,), (3, 3, 2)])
def test_retained_scalar_state_matches_dense_recovery(
    batch: int, partition: tuple[int, ...]
) -> None:
    """Retain identical accepted records without changing FP32 recovery order."""
    from tokenspeed_kernel.ops.attention.kda import (
        KdaReplayHistory,
        KdaReplayRecords,
        kda_materialize_history,
    )
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        commit_recoverssm_history,
    )

    torch.manual_seed(59000)
    width, heads, dim = 8, 12, 128
    pages = batch + 1
    q = torch.randn(1, batch * width, heads, dim, device="cuda", dtype=torch.bfloat16)
    k, v, gate = (torch.randn_like(q) for _ in range(3))
    beta = torch.randn(1, batch * width, heads, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads, dim, device="cuda") * 0.1
    base = torch.randn(pages, heads, dim, dim, device="cuda") * 0.02
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    query_start = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    u = torch.empty(batch, heads, width, dim, device="cuda")
    raw = torch.empty(batch, heads, width, 2 * dim, device="cuda", dtype=torch.bfloat16)
    vllm_triton_kda_recoverssm_verify(
        q,
        k,
        v,
        gate,
        beta,
        a,
        bias,
        checkpoint_state=base,
        correction_cache=u,
        kg_cache=raw,
        query_start_loc=query_start,
        state_indices=reads,
        spec_query_len=width,
        lower_bound=-5.0,
        out=torch.empty_like(q),
        record_metadata=None,
        record_stamps=None,
    )
    dense = base.clone()
    accepted = torch.full((batch,), sum(partition), device="cuda", dtype=torch.int32)
    plan = [torch.empty_like(reads) for _ in range(4)]
    vllm_triton_kda_recoverssm_commit(
        q,
        k,
        v,
        checkpoint_state=dense,
        correction_cache=u,
        kg_cache=raw,
        state_indices=reads,
        query_start_loc=query_start,
        num_accepted_tokens=accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((dense,)),
        correction_cache_base_addrs=_device_addresses((u,)),
        kg_cache_base_addrs=_device_addresses((raw,)),
        spec_query_len=width,
        record_metadata=None,
        record_stamps=None,
        gate_parameters=(a, bias, -5.0),
    )
    factors = [torch.empty(pages, heads, width, dim, device="cuda") for _ in range(3)]
    lengths = torch.zeros(pages, device="cuda", dtype=torch.int32)
    offsets = torch.zeros_like(reads)
    history = KdaReplayHistory(*factors, lengths, reads, offsets, False)
    metadata = torch.zeros(batch, 3, device="cuda", dtype=torch.int64)
    metadata[:, 0] = torch.arange(batch, device="cuda")
    metadata[:, 1] = 1
    stamps = torch.ones(batch, width, 2, device="cuda", dtype=torch.int64)
    stamps[:, :, 0] = torch.arange(width, device="cuda")
    records = KdaReplayRecords(
        key_gate_format="raw_key_gate",
        correction=u,
        key_gate=raw,
        metadata=metadata,
        stamps=stamps,
        query_start_loc=query_start,
        history=history,
    )
    position = 0
    for count in partition:
        metadata[:, 2] = position
        offsets.fill_(position)
        accepted.fill_(count)
        commit_recoverssm_history(
            records, accepted, A_log=a, dt_bias=bias, lower_bound=-5.0
        )
        position += count
    recovered = torch.empty_like(dense[1:])
    kda_materialize_history(
        base,
        *factors,
        lengths,
        reads,
        out=recovered,
        use_tensor_cores=False,
    )
    torch.testing.assert_close(recovered, dense[1:], atol=0, rtol=0)
