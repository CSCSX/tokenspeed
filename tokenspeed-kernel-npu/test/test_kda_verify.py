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

"""Precision oracle for Lite featurewise-beta KDA target verification."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel_npu.ops.kda import (
    torch_kda_fused_paged_verify,
    torch_kda_paged_decode,
    torch_kda_replay_commit,
)


def _case(device: str, *, batch: int, width: int, heads: int = 4):
    generator = torch.Generator().manual_seed(20260917)
    dim = 128
    channels = 3 * heads * dim
    gate_rank = 256
    pages = batch * width + batch + 4

    def randn(*shape, dtype=torch.bfloat16):
        value = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
        return value.to(dtype=dtype, device=device)

    return {
        "mixed_qkv": randn(batch * width, channels),
        "conv_weights": randn(channels, 4),
        "conv_states": randn(pages, channels, 3),
        "conv_scratch": torch.zeros(
            pages, channels, 3, dtype=torch.bfloat16, device=device
        ),
        "f_a_out": randn(batch * width, gate_rank),
        "f_b_weight": randn(heads * dim, gate_rank),
        "beta_logits": randn(batch * width, heads * dim),
        "A_log": randn(heads, dtype=torch.float32),
        "dt_bias": randn(heads * dim, dtype=torch.float32),
        "state_pool": randn(pages, heads, dim, dim, dtype=torch.float32),
        "state_scratch": torch.zeros(
            pages, heads, dim, dim, dtype=torch.float32, device=device
        ),
        "read_indices": torch.arange(batch, dtype=torch.int64, device=device),
        "write_indices": torch.arange(
            batch, batch + batch * width, dtype=torch.int64, device=device
        ).view(batch, width),
        "num_heads": heads,
        "head_dim": dim,
        "draft_token_num": width,
        "lower_bound": -8.0,
    }


def _sequential_reference(case):
    batch, width = case["write_indices"].shape
    heads, dim = case["num_heads"], case["head_dim"]
    channels = 3 * heads * dim
    history = case["conv_states"].index_select(0, case["read_indices"]).clone()
    state = case["state_pool"].index_select(0, case["read_indices"]).clone()
    rows = case["mixed_qkv"].view(batch, width, channels)
    gate_input = case["f_a_out"].view(batch, width, -1)
    beta = case["beta_logits"].view(batch, width, heads, dim)
    expected_output = []
    expected_conv = []
    expected_state = []
    local_pages = torch.arange(batch, dtype=torch.int64, device=rows.device)
    boundaries = torch.arange(batch + 1, dtype=torch.int32, device=rows.device)

    for token in range(width):
        window = torch.cat((history, rows[:, token].unsqueeze(-1)), dim=-1)
        conv = (window.float() * case["conv_weights"].float()).sum(dim=-1)
        conv = F.silu(conv).to(rows.dtype)
        history = window[:, :, 1:]
        expected_conv.append(history)
        q, k, v = conv.view(batch, 3, heads, dim).unbind(1)
        gate = F.linear(gate_input[:, token], case["f_b_weight"])
        output = torch_kda_paged_decode(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            gate.view(1, batch, heads, dim),
            beta[:, token].unsqueeze(0),
            case["A_log"],
            case["dt_bias"],
            state_pool=state,
            read_indices=local_pages,
            write_indices=local_pages,
            cu_seqlens=boundaries,
            lower_bound=case["lower_bound"],
        )
        expected_output.append(output[0])
        expected_state.append(state.clone())

    return (
        torch.stack(expected_output, dim=1).reshape(1, batch * width, heads, dim),
        torch.stack(expected_conv, dim=1).reshape(batch * width, channels, 3),
        torch.stack(expected_state, dim=1).reshape(batch * width, heads, dim, dim),
    )


@pytest.mark.parametrize("width", [2, 4, 8])
def test_featurewise_verify_matches_sequential_decode(width):
    case = _case("cpu", batch=2, width=width)
    committed_conv = case["conv_states"].clone()
    committed_state = case["state_pool"].clone()
    expected, expected_conv, expected_state = _sequential_reference(case)

    actual = torch_kda_fused_paged_verify(**case)
    writes = case["write_indices"].reshape(-1)

    # Gate projection is intentionally issued once for the whole packed
    # verify window. Its FP32 reduction order differs from N separate GEMVs,
    # but the final BF16 output remains well below one meaningful ULP.
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=4e-6)
    torch.testing.assert_close(
        case["conv_scratch"].index_select(0, writes),
        expected_conv,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        case["state_scratch"].index_select(0, writes),
        expected_state,
        rtol=2e-4,
        atol=1e-4,
    )
    assert torch.equal(case["conv_states"], committed_conv)
    assert torch.equal(case["state_pool"], committed_state)


@pytest.mark.skipif(not hasattr(torch, "npu"), reason="requires torch_npu")
@pytest.mark.parametrize("width", [2, 4, 8])
def test_featurewise_verify_npu_matches_cpu_oracle(width):
    torch.npu.set_device(0)
    cpu_case = _case("cpu", batch=2, width=width)
    expected, expected_conv, expected_state = _sequential_reference(cpu_case)
    npu_case = {
        name: value.npu() if isinstance(value, torch.Tensor) else value
        for name, value in cpu_case.items()
    }

    actual = torch_kda_fused_paged_verify(**npu_case)
    torch.npu.synchronize()
    writes = cpu_case["write_indices"].reshape(-1)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        npu_case["conv_scratch"].cpu().index_select(0, writes),
        expected_conv,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        npu_case["state_scratch"].cpu().index_select(0, writes),
        expected_state,
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("width", [2, 4, 8])
def test_replay_commit_publishes_only_the_accepted_prefix(device, width):
    if device == "npu":
        if not hasattr(torch, "npu"):
            pytest.skip("requires torch_npu")
        torch.npu.set_device(0)
        device = "npu:0"
    case = _case(device, batch=3, width=width)
    # Exercise fresh history, partial acceptance, and the complete window.
    case["read_indices"][0] = -1
    accepted = torch.tensor(
        [0, max(1, width // 2), width], dtype=torch.int32, device=device
    )
    # The sequential oracle uses zero state/history for a fresh request.
    if case["read_indices"][0] < 0:
        case["read_indices"][0] = 0
        case["conv_states"][0].zero_()
        case["state_pool"][0].zero_()
    _, expected_conv, expected_state = _sequential_reference(case)
    case["read_indices"][0] = -1
    source_conv = case["conv_states"].clone()
    source_state = case["state_pool"].clone()
    destination_conv = case["conv_states"].clone()
    destination_state = case["state_pool"].clone()

    torch_kda_replay_commit(
        case["mixed_qkv"],
        case["conv_weights"],
        case["conv_states"],
        destination_conv,
        case["f_a_out"],
        case["f_b_weight"],
        case["beta_logits"],
        case["A_log"],
        case["dt_bias"],
        state_pool=case["state_pool"],
        state_out=destination_state,
        read_indices=case["read_indices"],
        write_indices=case["write_indices"][:, 0],
        accepted_length=accepted,
        num_heads=case["num_heads"],
        head_dim=case["head_dim"],
        draft_token_num=width,
        lower_bound=case["lower_bound"],
    )
    if str(device).startswith("npu"):
        torch.npu.synchronize()

    for request, steps in enumerate(accepted.cpu().tolist()):
        destination = int(case["write_indices"][request, 0].cpu())
        if steps == 0:
            assert torch.equal(destination_conv[destination], source_conv[destination])
            assert torch.equal(
                destination_state[destination], source_state[destination]
            )
            continue
        row = request * width + steps - 1
        torch.testing.assert_close(
            destination_conv[destination], expected_conv[row], rtol=0, atol=0
        )
        torch.testing.assert_close(
            destination_state[destination],
            expected_state[row],
            rtol=2e-3 if str(device).startswith("npu") else 2e-4,
            atol=2e-3 if str(device).startswith("npu") else 1e-4,
        )
    assert torch.equal(case["conv_states"], source_conv)
    assert torch.equal(case["state_pool"], source_state)
