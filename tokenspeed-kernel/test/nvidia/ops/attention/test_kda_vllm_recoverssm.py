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

from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (  # noqa: E402
    prepare_recoverssm_conv,
    prepare_recoverssm_gate,
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
    lower_bound: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    q32 = q.float()
    k32 = k.float()
    q32 *= torch.rsqrt(q32.square().sum(dim=-1, keepdim=True) + 1e-6)
    q32 *= q.shape[-1] ** -0.5
    k32 *= torch.rsqrt(k32.square().sum(dim=-1, keepdim=True) + 1e-6)
    gate = lower_bound * torch.sigmoid(
        A_log.exp()[None, None, :, None] * (raw_g.float() + dt_bias[None, None, :, :])
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
    return output, states


@pytest.mark.parametrize(
    "width,accepted", [(1, 0), (1, 1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
)
def test_vllm_recoverssm_verify_and_commit(width: int, accepted: int) -> None:
    torch.manual_seed(7331)
    device = "cuda"
    batch, heads, dim = 4, 12, 128
    q = torch.randn(batch, width, heads, dim, device=device).bfloat16()
    k = torch.randn_like(q)
    v = torch.randn_like(q).mul_(0.2)
    raw_g = torch.randn_like(q)
    raw_beta = torch.randn(batch, width, heads, device=device).bfloat16()
    A_log = torch.randn(heads, device=device).mul_(0.1)
    dt_bias = torch.randn(heads, dim, device=device).mul_(0.1)
    initial_state = torch.randn(batch, heads, dim, dim, device=device).mul_(0.02)
    expected_out, expected_states = _reference(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        initial_state,
        -5.0,
    )

    state_pool = torch.zeros(
        batch + 1, heads, dim, dim, device=device, dtype=torch.float32
    )
    state_pool[1:].copy_(initial_state)
    correction_cache = torch.empty(
        batch + 1, heads, width, dim, device=device, dtype=torch.float32
    )
    kd_cache = torch.empty(
        batch + 1, heads, width, 2 * dim, device=device, dtype=torch.float32
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
        kd_cache=kd_cache,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        replay_inputs=None,
        verify_decay=None,
        verify_qk=None,
    )

    num_accepted = torch.full((batch,), accepted, device=device, dtype=torch.int32)
    plan = tuple(torch.empty(batch, device=device, dtype=torch.int32) for _ in range(4))
    vllm_triton_kda_recoverssm_commit(
        packed_q,
        packed_k,
        packed_v,
        checkpoint_state=state_pool,
        correction_cache=correction_cache,
        kd_cache=kd_cache,
        state_indices=state_indices,
        query_start_loc=query_start_loc,
        num_accepted_tokens=num_accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state_pool,)),
        correction_cache_base_addrs=_device_addresses((correction_cache,)),
        kd_cache_base_addrs=_device_addresses((kd_cache,)),
        spec_query_len=width,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out.float().reshape_as(expected_out), expected_out, atol=2e-2, rtol=2e-2
    )
    expected_state = initial_state if accepted == 0 else expected_states[accepted - 1]
    torch.testing.assert_close(state_pool[1:], expected_state, atol=3e-2, rtol=2e-2)


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
    replay_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    verify_decay: torch.Tensor,
    verify_qk: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run the record producer and recovery on the same prepared inputs."""
    batch, heads, dim, _ = state[1:].shape
    q, k, v = (
        value.reshape(1, batch * width, heads, dim) for value in qkv.chunk(3, dim=-1)
    )
    gates = gate.reshape_as(q)
    betas = beta.reshape(1, batch * width, heads)
    u = torch.empty(batch + 1, heads, width, dim, device=q.device)
    kd = torch.empty(batch + 1, heads, width, 2 * dim, device=q.device)
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
        kd_cache=kd,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        replay_inputs=replay_inputs,
        verify_decay=verify_decay,
        verify_qk=verify_qk,
    )
    plan = tuple(torch.empty_like(indices) for _ in range(4))
    vllm_triton_kda_recoverssm_commit(
        q,
        k,
        v,
        checkpoint_state=state,
        correction_cache=u,
        kd_cache=kd,
        state_indices=indices,
        query_start_loc=starts,
        num_accepted_tokens=accepted,
        commit_lens=plan[0],
        final_state_indices=plan[1],
        boundary_state_indices=plan[2],
        boundary_recovery_lens=plan[3],
        state_base_addrs=_device_addresses((state,)),
        correction_cache_base_addrs=_device_addresses((u,)),
        kd_cache_base_addrs=_device_addresses((kd,)),
        spec_query_len=width,
    )
    return out


@pytest.mark.parametrize("batch,width", [(4, 1), (4, 4), (16, 4)])
def test_recoverssm_matches_main_producers_over_rounds(batch: int, width: int) -> None:
    """Compare against main's full-precision replay, not just rounded verify."""
    from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
        batched_recurrent_kda_replay_commit,
        fused_kda_verify_conv_update,
        fused_recurrent_kda_verify_megafuse,
    )

    torch.manual_seed(781)
    heads, dim = 12, 128
    channels, rows = heads * dim, batch * width
    device = "cuda"
    raw = torch.randn(rows, 3 * channels, device=device, dtype=torch.bfloat16)
    conv_w = torch.randn(3 * channels, 4, device=device, dtype=torch.bfloat16) * 0.3
    conv = torch.randn(batch + 1, 3 * channels, 3, device=device, dtype=torch.bfloat16)
    fa = torch.randn(rows, dim, device=device, dtype=torch.bfloat16)
    fb = torch.randn(channels, dim, device=device, dtype=torch.bfloat16) * 0.05
    beta = torch.randn(rows, heads, device=device, dtype=torch.bfloat16)
    A = torch.randn(heads, device=device) * 0.1
    bias = torch.randn(heads, dim, device=device) * 0.1
    main_state = torch.randn(batch + 1, heads, dim, dim, device=device) * 0.02
    candidate_state = main_state.clone()
    gate_scratch = torch.empty(rows, channels, device=device)
    indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
    groups = torch.zeros(1, device=device, dtype=torch.int32)
    descriptors = _device_addresses(
        (raw, conv_w, conv, fa, fb, beta, A, bias, main_state, gate_scratch)
    ).reshape(1, 10)
    scratch_indices = torch.arange(rows, device=device, dtype=torch.int32).view(
        batch, width
    )
    largest_output_error = 0.0
    largest_state_error = 0.0
    squared_state_error = 0.0
    state_elements = 0
    for step in range(128):
        raw.normal_()
        fa.normal_()
        beta.normal_()
        accepted = (
            torch.arange(batch, device=device, dtype=torch.int32) + step
        ) % width + 1
        conv_qkv = fused_kda_verify_conv_update(
            raw,
            conv_w,
            conv,
            indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            out=None,
            block_c=256,
            num_warps=4,
        )
        raw_gate = torch.nn.functional.linear(fa, fb)
        expected_out = fused_recurrent_kda_verify_megafuse(
            raw,
            conv_w,
            conv,
            conv,
            fa,
            fb,
            beta,
            A,
            bias,
            main_state,
            main_state,
            indices,
            scratch_indices,
            num_heads=heads,
            head_dim=dim,
            draft_token_num=width,
            scale=dim**-0.5,
            lower_bound=-5.0,
            store_states=False,
            bv=None,
            g_raw=raw_gate,
            conv_qkv=conv_qkv,
            num_warps=None,
            num_stages=None,
            enable_pdl=False,
        )
        candidate_conv, replay_key, replay_value, verify_qk = prepare_recoverssm_conv(
            raw,
            torch.empty_like(raw),
            conv_w,
            conv,
            indices,
            num_heads=heads,
            head_dim=dim,
            width=width,
        )
        candidate_gate, replay_gate, verify_decay = prepare_recoverssm_gate(
            fa,
            fb,
            A,
            bias,
            num_heads=heads,
            head_dim=dim,
            lower_bound=-5.0,
        )
        record_inputs = (replay_key, replay_value, replay_gate)
        torch.testing.assert_close(candidate_conv, conv_qkv, atol=0, rtol=0)
        out = _recover_window(
            candidate_conv,
            candidate_gate,
            beta,
            A,
            bias,
            candidate_state,
            indices,
            accepted,
            width,
            record_inputs,
            verify_decay,
            verify_qk,
        )
        batched_recurrent_kda_replay_commit(
            descriptors,
            groups,
            indices[None],
            indices[None],
            accepted,
            draft_token_num=width,
            num_heads=heads,
            head_dim=dim,
            f_a_dim=dim,
            qkv_stride=raw.stride(0),
            conv_stride=conv.stride(0),
            f_a_stride=fa.stride(0),
            beta_stride=beta.stride(0),
            state_stride=main_state.stride(0),
            gate_stride=gate_scratch.stride(0),
            conv_width=4,
            lower_bound=-5.0,
        )
        largest_output_error = max(
            largest_output_error,
            (out.reshape_as(expected_out).float() - expected_out.float())
            .abs()
            .max()
            .item(),
        )
        largest_state_error = max(
            largest_state_error, (candidate_state - main_state).abs().max().item()
        )
        torch.testing.assert_close(
            out.reshape_as(expected_out), expected_out, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(candidate_state, main_state, atol=1e-5, rtol=1e-3)
        squared_state_error += (candidate_state - main_state).square().sum().item()
        state_elements += main_state.numel()
    print(
        f"B={batch} T={width}: max output error={largest_output_error:.8g}, max state error={largest_state_error:.8g}, state RMS={(squared_state_error / state_elements)**0.5:.8g}"
    )


def test_record_chain_does_not_read_rounded_verify_inputs() -> None:
    """Perturb verify inputs without changing the recovery record producers."""
    torch.manual_seed(173)
    batch, width, heads, dim = 4, 4, 12, 128
    shape = (1, batch * width, heads, dim)
    q, k, v, raw_gate = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    )
    beta = torch.randn(1, batch * width, heads, device="cuda", dtype=torch.bfloat16)
    A = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    normalized_k = k.float() / torch.sqrt(
        k.float().square().sum(-1, keepdim=True) + 1e-6
    )
    replay_inputs = (normalized_k, v.float(), torch.rand(shape, device="cuda"))
    state = torch.randn(batch + 1, heads, dim, dim, device="cuda") * 0.01
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
    results = []
    for perturb in (False, True):
        u = torch.empty(batch, heads, width, dim, device="cuda")
        kd = torch.empty(batch, heads, width, 2 * dim, device="cuda")
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
            kd_cache=kd,
            query_start_loc=starts,
            state_indices=indices,
            spec_query_len=width,
            lower_bound=-5.0,
            out=out,
            replay_inputs=replay_inputs,
            verify_decay=None,
            verify_qk=None,
        )
        results.append((out, u, kd))
    assert not torch.equal(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1], atol=0, rtol=0)
    torch.testing.assert_close(results[0][2], results[1][2], atol=0, rtol=0)


def test_recoverssm_reuses_compilation_within_launch_buckets() -> None:
    """Request counts change metadata, not one binary per exact batch size."""
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        _recoverssm_gate_inputs_kernel,
        _replay_conv_inputs_kernel,
    )
    from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
        _commit_kda_state_kernel,
        _kda_recoverssm_verify_kernel,
        _prepare_commit_plan_kernel,
    )
    from utils import assert_no_triton_compile

    torch.manual_seed(974)
    heads, dim, width = 12, 128, 4
    channels = heads * dim
    weights = torch.randn(3 * channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
    fb = torch.randn(channels, dim, device="cuda", dtype=torch.bfloat16) * 0.05
    A = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")

    def gate(rows: int):
        fa = torch.randn(rows, dim, device="cuda", dtype=torch.bfloat16)
        return prepare_recoverssm_gate(
            fa, fb, A, bias, num_heads=heads, head_dim=dim, lower_bound=-5.0
        )

    def run(batch: int) -> None:
        rows = batch * width
        raw = torch.randn(rows, 3 * channels, device="cuda", dtype=torch.bfloat16)
        conv = torch.zeros(
            batch + 1, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
        )
        state = torch.zeros(batch + 1, heads, dim, dim, device="cuda")
        indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
        qkv, replay_k, replay_v, qk = prepare_recoverssm_conv(
            raw,
            torch.empty_like(raw),
            weights,
            conv,
            indices,
            num_heads=heads,
            head_dim=dim,
            width=width,
        )
        raw_gate, replay_decay, verify_decay = gate(rows)
        beta = torch.zeros(rows, heads, device="cuda", dtype=torch.bfloat16)
        accepted = torch.full((batch,), 2, device="cuda", dtype=torch.int32)
        _recover_window(
            qkv,
            raw_gate,
            beta,
            A,
            bias,
            state,
            indices,
            accepted,
            width,
            (replay_k, replay_v, replay_decay),
            verify_decay,
            qk,
        )

    # These warm the same verify batch bucket and both gate tile widths.
    run(5)
    run(8)
    with assert_no_triton_compile(
        _replay_conv_inputs_kernel,
        _recoverssm_gate_inputs_kernel,
        _kda_recoverssm_verify_kernel.fn,
        _prepare_commit_plan_kernel.fn,
        _commit_kda_state_kernel.fn,
    ):
        for batch in (6, 7):
            run(batch)
        for rows in (40, 48, 60):
            gate(rows)


@pytest.mark.parametrize("separate_records", [False, True])
def test_recoverssm_ragged_prefix_and_null_rows(separate_records: bool) -> None:
    """Padded suffix state is unobservable; only valid prefix records change."""
    torch.manual_seed(1209)
    lengths = [4, 2, 0, 1, 3, 4]
    batch, heads, dim, width = len(lengths), 12, 128, 4
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
    indices = torch.tensor([1, 2, 3, 4, 5, 0], device="cuda", dtype=torch.int32)
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    u = torch.full((batch, heads, width, dim), float("nan"), device="cuda")
    kd = torch.full((batch, heads, width, 2 * dim), float("nan"), device="cuda")
    out = torch.full_like(v, float("nan"))
    if separate_records:
        record_k = k.float() / torch.sqrt(
            k.float().square().sum(-1, keepdim=True) + 1e-6
        )
        record_v = v.float() * 1.25
        record_d = torch.rand(shape, device="cuda")
        replay_inputs = (record_k, record_v, record_d)
    else:
        record_k = k.float() * torch.rsqrt(
            k.float().square().sum(-1, keepdim=True) + 1e-6
        )
        record_v = v.float()
        record_d = torch.exp(
            -5.0 * torch.sigmoid(A.exp()[None, None, :, None] * (g.float() + bias))
        )
        replay_inputs = None
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
        kd_cache=kd,
        query_start_loc=starts,
        state_indices=indices,
        spec_query_len=width,
        lower_bound=-5.0,
        out=out,
        replay_inputs=replay_inputs,
        verify_decay=None,
        verify_qk=None,
    )
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
                    out[:, begin:end].float(), expected_out, atol=2e-2, rtol=2e-2
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
                        u[request, :, offset], correction, atol=1e-5, rtol=1e-3
                    )
                    torch.testing.assert_close(
                        kd[request, :, offset, :dim], key, atol=1e-6, rtol=1e-5
                    )
                    torch.testing.assert_close(
                        kd[request, :, offset, dim:], decay, atol=1e-6, rtol=1e-5
                    )
                    current = current + correction[:, :, None] * key[:, None, :]
        assert torch.isnan(u[request, :, valid:]).all()
        assert torch.isnan(kd[request, :, valid:]).all()


@pytest.mark.parametrize("captured", [False, True])
def test_batched_recoverssm_commit_groups_alias_and_padding(captured: bool) -> None:
    """Exercise the production two-kernel commit, not the single-layer adapter."""
    from tokenspeed_kernel.ops.attention.kda.vllm_recoverssm import (
        batched_recoverssm_commit,
    )

    torch.manual_seed(982)
    layers, batch, width, heads, dim, pages = 3, 6, 4, 2, 128, 12
    channels = heads * dim
    groups_cpu = [1, 0, 1]
    read_cpu = [[1, 2, 3, 4, 5, -1], [5, 4, 3, 2, 1, -1]]
    write_cpu = [[6, 2, 7, 4, 8, -1], [6, 4, 7, 2, 8, -1]]
    accepted_cpu = [0, 1, 2, 3, 4, 0]
    groups = torch.tensor(groups_cpu, device="cuda", dtype=torch.int32)
    reads = torch.tensor(read_cpu, device="cuda", dtype=torch.int32)
    writes = torch.tensor(write_cpu, device="cuda", dtype=torch.int32)
    accepted = torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32)
    states = torch.randn(layers, pages, heads, dim, dim, device="cuda") * 0.02
    conv = torch.randn(
        layers, pages, 3 * channels, 3, device="cuda", dtype=torch.bfloat16
    )
    # Padded token rows catch accidental use of packed strides.
    raw = torch.randn(
        layers,
        batch * width,
        3 * channels + 16,
        device="cuda",
        dtype=torch.bfloat16,
    )
    corrections = (
        torch.randn(layers, batch * width, channels + 16, device="cuda") * 0.01
    )
    kd = torch.randn(layers, batch * width, 2 * channels + 16, device="cuda") * 0.02
    kd[:, :, : 2 * channels].view(layers, batch * width, heads, 2, dim)[
        :, :, :, 1
    ].sigmoid_()
    seed_state, seed_conv = states.clone(), conv.clone()
    expected_state, expected_conv = states.clone(), conv.clone()
    descriptors = torch.zeros(layers, 10, device="cuda", dtype=torch.int64)
    for column, storage in (
        (0, raw),
        (2, conv),
        (3, corrections),
        (5, kd),
        (8, states),
    ):
        descriptors[:, column] = _device_addresses(tuple(storage.unbind()))
    for layer, group in enumerate(groups_cpu):
        for request, count in enumerate(accepted_cpu[:-1]):
            source, dest = read_cpu[group][request], write_cpu[group][request]
            state = seed_state[layer, source].clone()
            history = seed_conv[layer, source].clone()
            for token in range(count):
                row = request * width + token
                key, decay = (
                    kd[layer, row, : 2 * channels].view(heads, 2, dim).unbind(1)
                )
                correction = corrections[layer, row, :channels].view(heads, dim)
                state = state * decay[:, None, :]
                state = state + correction[:, :, None] * key[:, None, :]
                history = torch.cat(
                    (history[:, 1:], raw[layer, row, : 3 * channels, None]), dim=1
                )
            expected_state[layer, dest] = state
            expected_conv[layer, dest] = history

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
            key_decay_stride=kd.stride(1),
            state_stride=states.stride(1),
            conv_width=4,
        )

    launch()
    graph = None
    if captured:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
    # Replay twice from the same seed. Refresh descriptor addresses between
    # runs to catch pointers cached outside the persistent table.
    for _ in range(2):
        states.copy_(seed_state)
        conv.copy_(seed_conv)
        if graph is None:
            launch()
        else:
            graph.replay()
        torch.testing.assert_close(states, expected_state, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
        states = torch.empty_like(states)
        conv = torch.empty_like(conv)
        descriptors[:, 8] = _device_addresses(tuple(states.unbind()))
        descriptors[:, 2] = _device_addresses(tuple(conv.unbind()))
