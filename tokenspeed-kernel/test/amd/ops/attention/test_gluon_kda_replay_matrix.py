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

"""AMD KDA correctness and paired original-versus-records graph timings."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from amd.ops.attention.kda_replay_baseline import (
    OriginalKda,
    RecordsKda,
    _original_module,
    check_original_gate_projection,
    original_gate_projection,
    prepare_records_verify,
)
from ops.attention.test_kda_replay_commit import (
    LOWER_BOUND,
    K,
    _batched_descriptor,
    _prepare_records,
    _window,
)
from tokenspeed_kernel.benchmark.graph import (
    GraphBenchmarkConfig,
    GraphTimer,
    PreparedInvocation,
)
from tokenspeed_kernel.numerics.torch.attention.kda import kda_gate
from tokenspeed_kernel.ops.attention.kda import resolve_kda_batched_replay_commit
from tokenspeed_kernel.ops.attention.kda._triton.recurrent import (
    fused_kda_verify_conv_update,
)
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
_HEADS = 12
_STATE_PAGE_BYTES = 884736


def _reference(
    x, t, accepted, *, rounded_conv: bool, lower_bound: float | None, raw_gate=None
):
    """Check recurrence independently of saved corrections and keys.

    The candidate uses rounded BF16 convolution outputs. The original keeps
    FP32 convolution outputs; compute those with the torch reference below.
    """
    batch = x["read_indices"].numel()
    heads = x["beta"].shape[-1]
    width = heads * K
    if rounded_conv:
        convolved = (
            fused_kda_verify_conv_update(
                x["qkv_raw"],
                x["conv_w"],
                x["conv_pool"],
                x["read_indices"],
                num_heads=heads,
                head_dim=K,
                draft_token_num=t,
            )
            .float()
            .view(batch, t, 3, heads, K)
        )
    else:
        # The original AMD kernel keeps convolution outputs in FP32.
        history = x["conv_pool"][x["read_indices"].long()].float()
        weights = x["conv_w"].float()
        rows = x["qkv_raw"].float().view(batch, t, 3 * width)
        values = []
        for token in range(t):
            value = (
                history[:, :, 0] * weights[:, 0]
                + history[:, :, 1] * weights[:, 1]
                + history[:, :, 2] * weights[:, 2]
                + rows[:, token] * weights[:, 3]
            )
            values.append(torch.nn.functional.silu(value))
            history = torch.cat((history[:, :, 1:], rows[:, token, :, None]), dim=-1)
        convolved = torch.stack(values, dim=1).view(batch, t, 3, heads, K)
    if raw_gate is None:
        raw_gate = torch.mm(x["f_a"], x["w_fb"].t())
    raw_gate = raw_gate.view(batch, t, heads, K)
    decay = (
        kda_gate(
            raw_gate.flatten(0, 1),
            x["A_log"],
            x["dt_bias"].view(heads, K),
            lower_bound=lower_bound,
        )
        .exp()
        .view(batch, t, heads, K)
    )
    beta = x["beta"].float().sigmoid().view(batch, t, heads)
    reads = x["read_indices"].long()
    running = x["h_pool"][reads].clone()
    selected = running.clone()
    history = x["conv_pool"][reads].clone()
    selected_history = history.clone()
    raw = x["qkv_raw"].view(batch, t, 3 * width)
    outputs, corrections = [], []
    for token in range(t):
        query, key, value = convolved[:, token].unbind(1)
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
        query = query * K**-0.5
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        running = running * decay[:, token, :, None, :]
        correction = value - (running * key[:, :, None, :]).sum(-1)
        correction = correction * beta[:, token, :, None]
        running = running + correction[:, :, :, None] * key[:, :, None, :]
        outputs.append((running * query[:, :, None, :]).sum(-1))
        corrections.append(correction)
        selected = torch.where(
            (accepted > token)[:, None, None, None], running, selected
        )
        history = torch.cat((history[:, :, 1:], raw[:, token, :, None]), dim=-1)
        selected_history = torch.where(
            (accepted > token)[:, None, None], history, selected_history
        )
    return (
        torch.stack(outputs, dim=1).to(torch.bfloat16),
        torch.stack(corrections, dim=1),
        torch.cat((convolved[:, :, 1].to(torch.bfloat16), raw_gate), dim=-1),
        selected,
        selected_history,
    )


class ReplayCase:
    """Own the tensors behind all-layer descriptors and repeatable calls."""

    def __init__(self, batch: int, t: int, layers: int, *, in_place: bool, seed: int):
        self.batch = batch
        self.t = t
        self.layers = [
            _window(batch, t, pages=2 * batch + 2, seed=seed + layer, num_heads=_HEADS)
            for layer in range(layers)
        ]
        group_count = min(3, layers)
        self.groups = (
            torch.arange(layers, device="cuda", dtype=torch.int32) % group_count
        )
        self.reads = torch.stack(
            [self.layers[0]["read_indices"].roll(group) for group in range(group_count)]
        )
        self.writes = self.reads.clone() if in_place else self.reads + batch
        self.accepted = torch.full((batch,), t, device="cuda", dtype=torch.int32)
        self.originals = []
        self.verifies = []
        self.gate_reference_mismatches = 0
        self.retained_qkv = []
        for layer, x in enumerate(self.layers):
            # Use the serving arena's page stride for both pool components.
            pages = x["h_pool"].shape[0]
            arena = torch.empty(
                (pages, _STATE_PAGE_BYTES), device="cuda", dtype=torch.uint8
            )
            state_bytes = _HEADS * K * K * 4
            conv_bytes = 3 * _HEADS * K * 3 * 2
            state = arena[:, :state_bytes].view(torch.float32).view(pages, _HEADS, K, K)
            conv = (
                arena[:, state_bytes : state_bytes + conv_bytes]
                .view(torch.bfloat16)
                .view(pages, 3 * _HEADS * K, 3)
            )
            state.copy_(x["h_pool"])
            conv.copy_(x["conv_pool"])
            x["h_pool"], x["conv_pool"] = state, conv
            x["read_indices"] = self.reads[layer % group_count]
            x["write_indices"] = self.writes[layer % group_count]
            self.originals.append((conv.clone(), state.clone()))
            self.retained_qkv.append(torch.empty_like(x["qkv_raw"]))
            self.verifies.append(
                _prepare_records(
                    x, t, raw_gate=None, original_qkv=self.retained_qkv[layer]
                )
            )
        self.descriptors = _batched_descriptor(self.layers)
        for layer, x in enumerate(self.layers):
            self.descriptors[layer, 0] = self.retained_qkv[layer].data_ptr()
            self.descriptors[layer, 3] = x["correction"].data_ptr()
            self.descriptors[layer, 5] = x["key_gate"].data_ptr()
        self.stamps = torch.stack([x["stamps"] for x in self.layers])
        for layer, x in enumerate(self.layers):
            # Verify and commit share the same stamp storage, as in serving.
            x["stamps"].set_(self.stamps[layer])
        self.kernel = resolve_kda_batched_replay_commit(
            torch.bfloat16, num_heads=_HEADS, head_dim=K
        )
        if self.kernel is None:
            raise RuntimeError("AMD KDA replay commit is unavailable")

    def reset(self):
        for x, (conv, state) in zip(self.layers, self.originals, strict=True):
            x["conv_pool"].copy_(conv)
            x["h_pool"].copy_(state)

    def commit(self):
        x = self.layers[0]
        self.kernel(
            descriptors=self.descriptors,
            group_indices=self.groups,
            read_indices=self.reads,
            write_indices=self.writes,
            accepted_length=self.accepted,
            activation_dtype=torch.bfloat16,
            conv_state_dtype=x["conv_pool"].dtype,
            draft_token_num=self.t,
            num_heads=_HEADS,
            head_dim=K,
            qkv_stride=self.retained_qkv[0].stride(0),
            conv_stride=x["conv_pool"].stride(0),
            correction_stride=x["correction"].stride(1),
            key_gate_stride=x["key_gate"].stride(1),
            state_stride=x["h_pool"].stride(0),
            conv_width=4,
            lower_bound=LOWER_BOUND,
            record_metadata=x["metadata"],
            record_stamps=self.stamps,
        )

    def verify_all(self):
        for verify in self.verifies:
            verify()

    def lifecycle(self):
        self.verify_all()
        self.commit()

    def check(self, *, graph: bool, runner):
        """Check output, records, accepted state and every untouched pool page."""
        self.reset()
        runner.verify_all()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                runner.verify_all()
            capture.replay()
        references = []
        self.gate_reference_mismatches = 0
        for x, (conv, state), retained in zip(
            self.layers, self.originals, self.retained_qkv, strict=True
        ):
            torch.testing.assert_close(retained, x["qkv_raw"], atol=0, rtol=0)
            torch.testing.assert_close(x["conv_pool"], conv, atol=0, rtol=0)
            torch.testing.assert_close(x["h_pool"], state, atol=0, rtol=0)
            positions = x["metadata"][:, 2, None] + torch.arange(self.t, device="cuda")
            ring = positions % self.t
            rows = torch.arange(self.batch, device="cuda")[:, None]
            raw_gate = (
                x["key_gate"][rows, ring][..., K:]
                .reshape(self.batch * self.t, _HEADS * K)
                .contiguous()
            )
            # Validate the producer independently before using its rounded gate.
            check_original_gate_projection(raw_gate, x["f_a"], x["w_fb"])
            if current_platform().is_cdna4:
                helper_gate = original_gate_projection(x["f_a"], x["w_fb"])
                check_original_gate_projection(helper_gate, x["f_a"], x["w_fb"])
                self.gate_reference_mismatches += int((raw_gate != helper_gate).sum())
            output, correction, kg, final_state, final_conv = _reference(
                x,
                self.t,
                self.accepted,
                lower_bound=LOWER_BOUND,
                rounded_conv=True,
                raw_gate=raw_gate,
            )
            torch.testing.assert_close(
                x["verify_output"].view_as(output), output, atol=1.6e-2, rtol=1e-2
            )
            torch.testing.assert_close(
                x["correction"][rows, ring], correction, atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(x["key_gate"][rows, ring], kg, atol=0, rtol=0)
            expected_stamps = torch.stack(
                (positions, x["metadata"][:, 1, None].expand_as(positions)), dim=-1
            )
            torch.testing.assert_close(
                x["stamps"][rows, ring], expected_stamps, atol=0, rtol=0
            )
            references.append((final_state, final_conv))
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                runner.commit()
            # Capture may execute the operation on some backends.
            self.reset()
            capture.replay()
        else:
            runner.commit()
        for layer, (x, original, reference) in enumerate(
            zip(self.layers, self.originals, references, strict=True)
        ):
            expected_conv, expected_state = (value.clone() for value in original)
            writes = self.writes[layer % self.reads.shape[0]].long()
            expected_state[writes], expected_conv[writes] = reference
            torch.testing.assert_close(
                x["h_pool"], expected_state, atol=2e-6, rtol=2e-5
            )
            torch.testing.assert_close(x["conv_pool"], expected_conv, atol=0, rtol=0)
            untouched = torch.ones(
                expected_state.shape[0], device="cuda", dtype=torch.bool
            )
            untouched[writes] = False
            torch.testing.assert_close(
                x["h_pool"][untouched], original[1][untouched], atol=0, rtol=0
            )


@pytest.mark.skipif(
    not (current_platform().is_cdna4 or current_platform().is_cdna5),
    reason="requires AMD gfx950 or gfx1250",
)
def test_commit_ignores_rejected_records():
    case = ReplayCase(3, 4, 2, in_place=True, seed=8900)
    case.accepted.copy_(torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32))
    case.verify_all()
    case.commit()
    expected = [(x["h_pool"].clone(), x["conv_pool"].clone()) for x in case.layers]
    case.reset()
    offsets = torch.arange(case.t, device="cuda")
    rejected = offsets[None, :] >= case.accepted[:, None]
    for x in case.layers:
        positions = x["metadata"][:, 2, None] + offsets
        ring = positions % case.t
        slots = x["metadata"][:, 0, None].expand_as(positions)
        x["correction"][slots[rejected], ring[rejected]] = float("nan")
        x["key_gate"][slots[rejected], ring[rejected]] = float("nan")
    case.commit()
    for x, (state, history) in zip(case.layers, expected, strict=True):
        torch.testing.assert_close(x["h_pool"], state, atol=0, rtol=0)
        torch.testing.assert_close(x["conv_pool"], history, atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform().is_cdna4 or current_platform().is_cdna5),
    reason="requires AMD gfx950 or gfx1250",
)
def test_commit_wide_record_ring():
    capacity = 129
    case = ReplayCase(3, 1, 2, in_place=True, seed=8901)
    case.accepted.copy_(torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32))
    case.verify_all()
    case.commit()
    expected = [(x["h_pool"].clone(), x["conv_pool"].clone()) for x in case.layers]

    # Move the saved records into a wider ring; inactive entries stay poisoned.
    stamps = case.stamps.new_full((len(case.layers), case.batch, capacity, 2), -1)
    for layer, x in enumerate(case.layers):
        slots = x["metadata"][:, 0]
        positions = x["metadata"][:, 2] % capacity
        correction, key_gate = x["correction"], x["key_gate"]
        x["correction"] = correction.new_full(
            (case.batch, capacity, *correction.shape[2:]), float("nan")
        )
        x["key_gate"] = key_gate.new_full(
            (case.batch, capacity, *key_gate.shape[2:]), float("nan")
        )
        x["correction"][slots, positions] = correction[slots, 0]
        x["key_gate"][slots, positions] = key_gate[slots, 0]
        stamps[layer, slots, positions] = x["stamps"][slots, 0]
        x["stamps"] = stamps[layer]
        case.descriptors[layer, 3] = x["correction"].data_ptr()
        case.descriptors[layer, 5] = x["key_gate"].data_ptr()
    case.stamps = stamps

    # Eager recovery warms the wide-ring specialization before graph capture.
    for graph in (False, True):
        case.reset()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                case.commit()
            capture.replay()
        else:
            case.commit()
        for x, (state, history) in zip(case.layers, expected, strict=True):
            torch.testing.assert_close(x["h_pool"], state, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(x["conv_pool"], history, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform().is_cdna4,
    reason="requires the gfx950 checkpoint prefetch",
)
@pytest.mark.parametrize("batch", [4, 16])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("graph", [False, True])
def test_verify_checkpoint_prefetch_preserves_page_guards(batch, lower_bound, graph):
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_verify_gfx950,
    )

    t = 1
    pages = 2 * batch + 2
    live = batch - 3
    x = _window(batch, t, pages=pages, seed=8931, num_heads=_HEADS)
    raw_gate = torch.mm(x["f_a"], x["w_fb"].t())
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    reference = _reference(
        x,
        t,
        accepted,
        rounded_conv=True,
        lower_bound=lower_bound,
        raw_gate=raw_gate,
    )
    original_state = x["h_pool"].clone()
    original_history = x["conv_pool"].clone()
    retained = torch.empty_like(x["qkv_raw"])
    # Initialize the existing records contract while every source page is valid.
    _prepare_records(x, t, raw_gate=raw_gate, original_qkv=retained)
    x["read_indices"][live:] = torch.tensor(
        [0, -1, pages], device="cuda", dtype=torch.int32
    )
    snapshots = {
        name: x[name].clone()
        for name in (
            "qkv_raw",
            "conv_w",
            "f_a",
            "w_fb",
            "beta",
            "A_log",
            "dt_bias",
            "read_indices",
            "metadata",
        )
    }
    original_gate = raw_gate.clone()
    verify = prepare_records_verify(
        gluon_kda_fused_verify_gfx950,
        x,
        t,
        raw_gate=raw_gate,
        original_qkv=retained,
        lower_bound=lower_bound,
    )
    verify()
    sentinels = {"correction": -19, "key_gate": -23, "stamps": -29}

    def poison():
        for name, value in sentinels.items():
            x[name].fill_(value)
        retained.fill_(-31)

    poison()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            output = verify()
        poison()
        output.fill_(float("nan"))
        capture.replay()
    else:
        output = verify()

    expected_output, expected_u, expected_kg = reference[:3]
    output = output.view_as(expected_output)
    torch.testing.assert_close(
        output[:live], expected_output[:live], atol=1.6e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        x["correction"][:live], expected_u[:live], atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(x["key_gate"][:live], expected_kg[:live], atol=0, rtol=0)
    expected_stamps = torch.stack(
        (x["metadata"][:live, 2], x["metadata"][:live, 1]), dim=-1
    )[:, None]
    torch.testing.assert_close(x["stamps"][:live], expected_stamps, atol=0, rtol=0)
    torch.testing.assert_close(retained[:live], x["qkv_raw"][:live], atol=0, rtol=0)
    torch.testing.assert_close(
        output[live:], torch.zeros_like(output[live:]), atol=0, rtol=0
    )
    for name, value in sentinels.items():
        torch.testing.assert_close(
            x[name][live:], torch.full_like(x[name][live:], value), atol=0, rtol=0
        )
    torch.testing.assert_close(
        retained[live:], torch.full_like(retained[live:], -31), atol=0, rtol=0
    )
    torch.testing.assert_close(x["h_pool"], original_state, atol=0, rtol=0)
    torch.testing.assert_close(x["conv_pool"], original_history, atol=0, rtol=0)
    torch.testing.assert_close(raw_gate, original_gate, atol=0, rtol=0)
    for name, original in snapshots.items():
        torch.testing.assert_close(x[name], original, atol=0, rtol=0)


@pytest.mark.skipif(
    not (current_platform().is_cdna4 or current_platform().is_cdna5),
    reason="requires AMD gfx950 or gfx1250",
)
@pytest.mark.parametrize("graph", [False, True])
def test_verify_retains_original_qkv(graph):
    batch, t, heads = 3, 4, _HEADS
    x = _window(batch, t, pages=2 * batch + 2, seed=8910, num_heads=heads)
    rows, width = x["qkv_raw"].shape
    source_storage = torch.empty(rows + 1, width, device="cuda", dtype=torch.bfloat16)
    source_storage[:rows].copy_(x["qkv_raw"])
    x["qkv_raw"] = source_storage[:rows]
    raw_gate = torch.mm(x["f_a"], x["w_fb"].t())
    with pytest.raises(ValueError, match="must not overlap producers"):
        _prepare_records(x, t, raw_gate=raw_gate, original_qkv=source_storage[1:])
    _prepare_records(x, t, raw_gate=raw_gate, original_qkv=None)
    expected_output = x["verify_output"].clone()
    original = x["qkv_raw"].clone()
    accepted = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    reference = _reference(x, t, accepted, lower_bound=LOWER_BOUND, rounded_conv=True)
    writes = x["read_indices"] + batch
    expected_state, expected_history = x["h_pool"].clone(), x["conv_pool"].clone()
    expected_state[writes.long()], expected_history[writes.long()] = reference[-2:]
    destinations = [
        torch.full((rows, width + padding), -9, device="cuda", dtype=torch.bfloat16)
        for padding in (7, 16)
    ]
    _prepare_records(x, t, raw_gate=raw_gate, original_qkv=destinations[0][:, :width])
    arch = "gfx950" if current_platform().is_cdna4 else "gfx1250"
    module = importlib.import_module(
        f"tokenspeed_kernel_amd.ops.{arch}.attention.kda.decode"
    )
    verify_kernel = getattr(
        module, f"gluon_kda_fused_paged_verify_nostore_vmajor_{arch}"
    )
    for destination in destinations:
        retained = destination[:, :width]
        retained.fill_(-9)
        # Row strides remain runtime values across alignment classes.
        with assert_no_triton_compile(verify_kernel):
            verify = _prepare_records(x, t, raw_gate=raw_gate, original_qkv=retained)
            if graph:
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    verify()
                retained.fill_(-9)
                capture.replay()
        torch.testing.assert_close(retained, original, atol=0, rtol=0)
        torch.testing.assert_close(
            destination[:, width:],
            torch.full_like(destination[:, width:], -9),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(x["verify_output"], expected_output, atol=0, rtol=0)

    descriptors = _batched_descriptor([x])
    descriptors[0, 0] = retained.data_ptr()
    descriptors[0, 3] = x["correction"].data_ptr()
    descriptors[0, 5] = x["key_gate"].data_ptr()
    groups = torch.zeros(1, device="cuda", dtype=torch.int32)
    kernel = resolve_kda_batched_replay_commit(
        torch.bfloat16, num_heads=heads, head_dim=K
    )
    assert kernel is not None

    def commit():
        kernel(
            descriptors=descriptors,
            group_indices=groups,
            read_indices=x["read_indices"][None],
            write_indices=writes[None],
            accepted_length=accepted,
            activation_dtype=x["qkv_raw"].dtype,
            conv_state_dtype=x["conv_pool"].dtype,
            draft_token_num=t,
            num_heads=heads,
            head_dim=K,
            qkv_stride=retained.stride(0),
            conv_stride=x["conv_pool"].stride(0),
            correction_stride=x["correction"].stride(1),
            key_gate_stride=x["key_gate"].stride(1),
            state_stride=x["h_pool"].stride(0),
            conv_width=4,
            lower_bound=LOWER_BOUND,
            record_metadata=x["metadata"],
            record_stamps=x["stamps"][None],
        )

    # Projection storage may be reused before acceptance reaches commit.
    x["qkv_raw"].zero_()
    commit()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            commit()
        capture.replay()
    torch.testing.assert_close(x["h_pool"], expected_state, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(x["conv_pool"], expected_history, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform().is_cdna4,
    reason="requires the gfx950 inline gate producer",
)
@pytest.mark.parametrize("graph", [False, True])
def test_inline_gate_matches_supplied_gate(graph):
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_paged_verify_nostore_vmajor_gfx950,
    )

    batch, t, heads = 3, 8, _HEADS
    x = _window(batch, t, pages=2 * batch + 2, seed=8920, num_heads=heads)
    activation = x["f_a"].clone()
    inputs = [
        torch.empty(batch * t, K + padding, device="cuda", dtype=torch.bfloat16)
        for padding in (7, 16)
    ]
    for values in inputs:
        values[:, :K].copy_(activation)
    x["f_a"] = inputs[0][:, :K]
    _prepare_records(x, t, raw_gate=None)
    for values in inputs:
        x["f_a"] = values[:, :K]
        with assert_no_triton_compile(
            gluon_kda_fused_paged_verify_nostore_vmajor_gfx950
        ):
            verify = _prepare_records(x, t, raw_gate=None)
            if graph:
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    verify()
                capture.replay()
        output = x["verify_output"].clone()
        correction = x["correction"].clone()
        key_gate = x["key_gate"].clone()
        positions = x["metadata"][:, 2, None] + torch.arange(t, device="cuda")
        rows = torch.arange(batch, device="cuda")[:, None]
        raw_gate = (
            key_gate[rows, positions % t][..., K:]
            .reshape(batch * t, heads * K)
            .contiguous()
        )
        check_original_gate_projection(raw_gate, x["f_a"], x["w_fb"])
        verify = _prepare_records(x, t, raw_gate=raw_gate)
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                verify()
            capture.replay()
        torch.testing.assert_close(x["verify_output"], output, atol=1.6e-2, rtol=1e-2)
        torch.testing.assert_close(x["correction"], correction, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(x["key_gate"], key_gate, atol=0, rtol=0)


@pytest.mark.skipif(not current_platform().is_cdna4, reason="requires AMD gfx950")
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_one_token_recovery_preserves_decay(lower_bound):
    source_dir = os.environ.get("TOKENSPEED_KDA_BASELINE_DIR")
    if source_dir is None:
        pytest.skip("set TOKENSPEED_KDA_BASELINE_DIR to the accepted-parent source")
    parent = _original_module(
        Path(source_dir) / "accepted-parent-kda-gfx950.py", "gfx950"
    )
    current = importlib.import_module(
        "tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode"
    )
    case = ReplayCase(3, 1, 2, in_place=True, seed=8940)
    case.accepted.copy_(torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32))
    sweep = torch.linspace(-80, 80, _HEADS * K, device="cuda")
    sweep[:7] = torch.tensor([-1e-2, -1e-4, -1e-6, 0, 1e-6, 1e-4, 1e-2], device="cuda")
    raw_gate = sweep.repeat(case.batch, 1).to(torch.bfloat16)
    for layer, x in enumerate(case.layers):
        # K = U = 0 and checkpoint = 1 make one accepted update equal decay.
        x["conv_w"].zero_()
        x["h_pool"].fill_(1)
        case.originals[layer][1].fill_(1)
        x["correction"].fill_(float("nan"))
        x["key_gate"].fill_(float("nan"))
        x["stamps"].fill_(-1)
        retained = case.retained_qkv[layer]
        retained.fill_(float("nan"))
        output = prepare_records_verify(
            current.gluon_kda_fused_verify_gfx950,
            x,
            1,
            raw_gate=raw_gate,
            original_qkv=retained,
            lower_bound=lower_bound,
        )()
        for value in (output, x["correction"], x["key_gate"][..., :K]):
            torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
        torch.testing.assert_close(
            x["key_gate"][..., K:],
            raw_gate.view(case.batch, 1, _HEADS, K),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(retained, x["qkv_raw"], atol=0, rtol=0)
        torch.testing.assert_close(
            x["stamps"], x["metadata"][:, [2, 1]][:, None], atol=0, rtol=0
        )

    def commit(launcher):
        x = case.layers[0]
        launcher(
            descriptors=case.descriptors,
            group_indices=case.groups,
            read_indices=case.reads,
            write_indices=case.writes,
            accepted_length=case.accepted,
            draft_token_num=1,
            num_heads=_HEADS,
            head_dim=K,
            qkv_stride=case.retained_qkv[0].stride(0),
            conv_stride=x["conv_pool"].stride(0),
            correction_stride=x["correction"].stride(1),
            key_gate_stride=x["key_gate"].stride(1),
            state_stride=x["h_pool"].stride(0),
            conv_width=4,
            activation_dtype=torch.bfloat16,
            conv_state_dtype=x["conv_pool"].dtype,
            record_metadata=x["metadata"],
            record_stamps=case.stamps,
            lower_bound=lower_bound,
        )

    case.reset()
    commit(parent.launch_gluon_kda_fused_replay_gfx950)
    expected = [(x["h_pool"].clone(), x["conv_pool"].clone()) for x in case.layers]
    for layer, x in enumerate(case.layers):
        decay = kda_gate(
            raw_gate.view(case.batch, _HEADS, K),
            x["A_log"],
            x["dt_bias"].view(_HEADS, K),
            lower_bound=lower_bound,
        ).exp()
        writes = x["write_indices"].long()
        reference = case.originals[layer][1].clone()
        reference[writes] = torch.where(
            (case.accepted > 0)[:, None, None, None], decay[:, :, None, :], 1.0
        )
        assert bool((x["h_pool"][writes[case.accepted > 0]] != 1).any())
        torch.testing.assert_close(x["h_pool"], reference, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(
            x["h_pool"][writes[case.accepted == 0]],
            reference[writes[case.accepted == 0]],
            atol=0,
            rtol=0,
        )
    for graph in (False, True):
        case.reset()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                commit(current.launch_gluon_kda_fused_replay_gfx950)
            case.reset()
            capture.replay()
        else:
            commit(current.launch_gluon_kda_fused_replay_gfx950)
        for x, (state, conv) in zip(case.layers, expected, strict=True):
            torch.testing.assert_close(x["h_pool"], state, atol=0, rtol=0)
            torch.testing.assert_close(x["conv_pool"], conv, atol=0, rtol=0)


def _check_original(
    case: ReplayCase,
    original: OriginalKda,
    *,
    graph: bool,
    decode: bool,
    reference_gates,
):
    case.reset()
    references = [
        _reference(
            x,
            case.t,
            case.accepted,
            lower_bound=LOWER_BOUND,
            rounded_conv=False,
            raw_gate=reference_gates[layer] if decode and reference_gates else None,
        )
        for layer, x in enumerate(case.layers)
    ]
    invoke = original.decode_lifecycle if decode else original.replay_lifecycle
    if graph:
        invoke()
        case.reset()
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            invoke()
        case.reset()
        capture.replay()
    else:
        invoke()
    for layer, (x, initial, reference) in enumerate(
        zip(case.layers, case.originals, references, strict=True)
    ):
        output, _, _, selected_state, selected_conv = reference
        expected_conv, expected_state = (value.clone() for value in initial)
        writes = x["write_indices"].long()
        # Original replay leaves zero-acceptance destinations untouched.
        live = case.accepted > 0
        expected_state[writes[live]] = selected_state[live]
        expected_conv[writes[live]] = selected_conv[live]
        # Preserve the existing original-decode test's state tolerance.
        # A failing original reference excludes its case from speedup claims.
        state_atol = state_rtol = 2e-4 if decode else 2e-5
        checks = (
            ("output", original.outputs[layer].view_as(output), output, 1.6e-2, 1e-2),
            ("state", x["h_pool"], expected_state, state_atol, state_rtol),
            ("conv", x["conv_pool"], expected_conv, 0, 0),
        )
        for name, actual, expected, atol, rtol in checks:
            try:
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            except AssertionError as error:
                difference = (actual.float() - expected.float()).abs()
                absolute = difference.max().item()
                relative = (
                    (difference / expected.float().abs().clamp_min(1e-12)).max().item()
                )
                return {
                    "status": "failed",
                    "graph": graph,
                    "decode": decode,
                    "layer": layer,
                    "tensor": name,
                    "atol": atol,
                    "rtol": rtol,
                    "max_absolute_error": absolute if math.isfinite(absolute) else None,
                    "max_relative_error": relative if math.isfinite(relative) else None,
                    "relative_denominator_floor": 1e-12,
                    "error": str(error),
                }
    return {"status": "passed_eager_and_graph" if graph else "passed_eager"}


def _summary_without_samples(value):
    if isinstance(value, dict):
        return {
            key: _summary_without_samples(item)
            for key, item in value.items()
            if key != "rounds"
        }
    if isinstance(value, list):
        return [_summary_without_samples(item) for item in value]
    return value


def _paired_measurements(
    timer,
    candidate,
    original,
    *,
    reset,
    cold_cache,
    measurement_blocks,
    rounds,
    equivalent,
):
    measurements = {"candidate": [], "original": []}
    invocations = {"candidate": candidate, "original": original}
    for round_idx in range(rounds):
        order = (
            ("original", "candidate")
            if round_idx % 2 == 0
            else ("candidate", "original")
        )
        for side in order:
            invoke = invocations[side]
            if invoke is not None:
                measured = timer.measure(
                    PreparedInvocation(invoke=invoke, reset=reset),
                    cold_cache=cold_cache,
                    measurement_blocks=measurement_blocks,
                )
                measurements[side].append(asdict(measured))
    result = {"equivalent": equivalent, "speedup": None}
    for side, values in measurements.items():
        if not values:
            result[side] = None
            continue
        samples = sorted(sample for value in values for sample in value["samples_us"])
        median = statistics.median(samples)
        result[side] = {
            "median_us": median,
            "p90_us": samples[math.ceil(0.9 * len(samples)) - 1],
            "relative_mad": statistics.median(
                abs(sample - median) for sample in samples
            )
            / median,
            "rounds": values,
        }
    if equivalent and result["original"] is not None:
        result["speedup"] = (
            result["original"]["median_us"] / result["candidate"]["median_us"]
        )
    return result


def _measure_explicit_gate_verify(timer, case, parent_module, measure_options):
    """Compare one-layer verify using the same supplied gate on both sides."""
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.decode import (
        gluon_kda_fused_verify_gfx950,
    )

    case.reset()
    x, t = case.layers[0], case.t
    raw_gate = torch.mm(x["f_a"], x["w_fb"].t())
    original_input = x["qkv_raw"].clone()
    original_metadata = x["metadata"].clone()
    original_gate = raw_gate.clone()
    retained = case.retained_qkv[0]
    reference = _reference(
        x,
        t,
        case.accepted,
        lower_bound=LOWER_BOUND,
        rounded_conv=True,
        raw_gate=raw_gate,
    )
    positions = original_metadata[:, 2, None] + torch.arange(t, device="cuda")
    rows = torch.arange(case.batch, device="cuda")[:, None]
    ring = positions % t
    expected_stamps = torch.stack(
        (positions, original_metadata[:, 1, None].expand_as(positions)), dim=-1
    )
    invocations = [
        prepare_records_verify(
            launcher,
            x,
            t,
            raw_gate=raw_gate,
            original_qkv=retained,
            lower_bound=LOWER_BOUND,
        )
        for launcher in (
            gluon_kda_fused_verify_gfx950,
            parent_module.gluon_kda_fused_verify_gfx950,
        )
    ]

    def poison():
        x["correction"].fill_(float("nan"))
        x["key_gate"].fill_(float("nan"))
        retained.fill_(float("nan"))
        x["stamps"].fill_(-1)

    def check(output):
        expected_output, correction, kg = reference[:3]
        torch.testing.assert_close(
            output.view_as(expected_output), expected_output, atol=1.6e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            x["correction"][rows, ring], correction, atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(x["key_gate"][rows, ring], kg, atol=0, rtol=0)
        torch.testing.assert_close(
            x["stamps"][rows, ring], expected_stamps, atol=0, rtol=0
        )
        for actual, expected in (
            (retained, original_input),
            (x["qkv_raw"], original_input),
            (x["metadata"], original_metadata),
            (raw_gate, original_gate),
            (x["conv_pool"], case.originals[0][0]),
            (x["h_pool"], case.originals[0][1]),
        ):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    for invoke in invocations:
        case.reset()
        poison()
        check(invoke())
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            output = invoke()
        case.reset()
        poison()
        output.fill_(float("nan"))
        capture.replay()
        check(output)
    measured = _paired_measurements(
        timer, *invocations, equivalent=True, **measure_options
    )
    measured["accepted_parent"] = measured.pop("original")
    case.reset()
    return {
        "concurrency": case.batch,
        "window_length": t,
        "correctness": "both_passed_eager_and_graph",
        "measurement": measured,
    }


def _check_full_lifecycle_graph(case, runner):
    """Match a full graph to independently checked eager verify and commit."""
    inputs = [(x["qkv_raw"].clone(), x["metadata"].clone()) for x in case.layers]
    case.check(graph=False, runner=runner)
    fields = (
        "verify_output",
        "correction",
        "key_gate",
        "stamps",
        "h_pool",
        "conv_pool",
    )
    expected = [{name: x[name].clone() for name in fields} for x in case.layers]
    capture = torch.cuda.CUDAGraph()
    case.reset()
    with torch.cuda.graph(capture):
        runner.lifecycle()
    # Capture can execute kernels. Poison after capture to require fresh writes.
    case.reset()
    for x, retained in zip(case.layers, case.retained_qkv, strict=True):
        for name in ("verify_output", "correction", "key_gate"):
            x[name].fill_(float("nan"))
        x["stamps"].fill_(-1)
        retained.fill_(float("nan"))
    capture.replay()
    for x, retained, values, (qkv, metadata) in zip(
        case.layers, case.retained_qkv, expected, inputs, strict=True
    ):
        for name, value in values.items():
            torch.testing.assert_close(x[name], value, atol=0, rtol=0)
        torch.testing.assert_close(retained, qkv, atol=0, rtol=0)
        torch.testing.assert_close(x["qkv_raw"], qkv, atol=0, rtol=0)
        torch.testing.assert_close(x["metadata"], metadata, atol=0, rtol=0)
    case.reset()


def _measure_parent_lifecycle(timer, case, parent, measure_options):
    """Pair complete records lifecycles with equal work for all accepted counts."""
    case.check(graph=True, runner=parent)
    for runner in (case, parent):
        _check_full_lifecycle_graph(case, runner)
    measured = _paired_measurements(
        timer, case.lifecycle, parent.lifecycle, equivalent=True, **measure_options
    )
    measured["accepted_parent"] = measured.pop("original")
    measured["round_speedups"] = [
        accepted["median_us"] / candidate["median_us"]
        for accepted, candidate in zip(
            measured["accepted_parent"]["rounds"],
            measured["candidate"]["rounds"],
            strict=True,
        )
    ]
    case.reset()
    return {
        "concurrency": case.batch,
        "window_length": case.t,
        "accepted_counts": case.accepted.cpu().tolist(),
        "layers": len(case.layers),
        "correctness": "both_passed_eager_and_graph",
        "measurement": measured,
    }


def _acceptance_patterns(batch: int, t: int):
    for length in range(t + 1):
        yield str(length), torch.full(
            (batch,), length, device="cuda", dtype=torch.int32
        )
    if batch > 1:
        yield "mixed", torch.arange(batch, device="cuda", dtype=torch.int32) % (t + 1)
        if t > 1:
            yield "mixed-positive", 1 + torch.arange(
                batch, device="cuda", dtype=torch.int32
            ) % t


@pytest.mark.skipif(
    not (current_platform().is_cdna4 or current_platform().is_cdna5),
    reason="requires AMD gfx950 or gfx1250",
)
@pytest.mark.parametrize("batch", _CONCURRENCIES)
@pytest.mark.parametrize("t", [1, 4])
@pytest.mark.parametrize("in_place", [False, True], ids=["distinct", "in-place"])
def test_gluon_kda_verify_commit_matrix(batch, t, in_place):
    case = ReplayCase(batch, t, layers=3, in_place=in_place, seed=7100)
    for _, accepted in _acceptance_patterns(batch, t):
        case.accepted.copy_(accepted)
        case.check(graph=False, runner=case)
        case.check(graph=True, runner=case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, nargs="+", required=True)
    parser.add_argument(
        "--window-lengths", type=int, nargs="+", choices=[1, 4], required=True
    )
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument(
        "--page-relation", choices=["distinct", "in-place"], required=True
    )
    parser.add_argument("--cache", choices=["cold", "hot"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--accepted-parent-source", type=Path, required=True)
    parser.add_argument("--accepted-parent-revision", required=True)
    parser.add_argument("--measurement-blocks", type=int, default=30)
    parser.add_argument("--paired-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7100)
    args = parser.parse_args()
    started = time.monotonic()
    if (
        min(*args.concurrency, args.layers, args.measurement_blocks, args.paired_rounds)
        < 1
    ):
        parser.error("concurrency, layers and measurement counts must be positive")
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        parser.error("this matrix requires AMD gfx950 or gfx1250")
    arch = "gfx950" if platform.is_cdna4 else "gfx1250"
    timer = GraphTimer(
        GraphBenchmarkConfig(eager_warmup_iterations=5, replay_warmup_iterations=3)
    )
    root = Path(__file__).resolve().parents[5]
    report = {
        "schema_version": 2,
        "status": "running",
        "latency_unit": "us",
        "requested_concurrency": args.concurrency,
        "requested_window_lengths": args.window_lengths,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
        ),
        "baseline_revision": args.baseline_revision,
        "baseline_source_sha256": hashlib.sha256(
            args.baseline_source.read_bytes()
        ).hexdigest(),
        "device": platform.device_name,
        "dtype": "bfloat16",
        "heads": _HEADS,
        "head_dim": K,
        "seed": args.seed,
        "page_relation": args.page_relation,
        "cache": args.cache,
        "state_page_bytes": _STATE_PAGE_BYTES,
        "paired_rounds": args.paired_rounds,
        "verify_scope": "one layer; both retain payloads; candidate includes gate projection, original excludes gate GEMM; unequal work",
        "candidate_gate_producer": "launcher selects inline projection or GEMM from total input rows",
        "commit_scope": "all layers; includes conv history; excludes verify",
        "lifecycle_scope": {
            "candidate": "all layers: records verify with gate production and QKV retention, then one batched commit",
            "original_t1": "all layers: original fused decode; gfx1250 also includes gate GEMM",
            "original_t4": "all layers: gate GEMM, capture verify, then one batched raw-g replay",
        },
        "comparison_note": "Ratios apply only to equal work. Original replay does not copy zero-acceptance pages. Original standard decode accepts one token.",
        "accepted_parent_verify": {
            "status": "running" if arch == "gfx950" else "not_run",
            "reason": None if arch == "gfx950" else "gfx950-only comparison",
            "revision": args.accepted_parent_revision,
            "source_sha256": hashlib.sha256(
                args.accepted_parent_source.read_bytes()
            ).hexdigest(),
            "scope": "one layer; same supplied raw gate and retention; excludes gate projection and commit",
            "results": [],
        },
        "accepted_parent_lifecycle": {
            "status": "running" if arch == "gfx950" else "not_run",
            "reason": None if arch == "gfx950" else "gfx950-only comparison",
            "revision": args.accepted_parent_revision,
            "source_sha256": hashlib.sha256(
                args.accepted_parent_source.read_bytes()
            ).hexdigest(),
            "scope": "all layers on both sides: automatic gate production, records verify and QKV retention, then one batched commit; equal work for every acceptance pattern",
            "results": [],
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    publish()
    print(
        "C T accepted candidate_lifecycle_us original_lifecycle_us speedup", flush=True
    )
    try:
        for batch in args.concurrency:
            for t in args.window_lengths:
                report["active_case"] = {"concurrency": batch, "window_length": t}
                publish()
                print(
                    f"SETUP C{batch} T{t} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                case = ReplayCase(
                    batch,
                    t,
                    args.layers,
                    in_place=args.page_relation == "in-place",
                    seed=args.seed,
                )
                original = OriginalKda(
                    args.baseline_source,
                    arch,
                    case.layers,
                    case.groups,
                    case.reads,
                    case.writes,
                    case.accepted,
                    t,
                )
                parent = (
                    RecordsKda(
                        _original_module(args.accepted_parent_source.resolve(), arch),
                        case.layers,
                        case.retained_qkv,
                        case.descriptors,
                        case.groups,
                        case.reads,
                        case.writes,
                        case.accepted,
                        case.stamps,
                        t,
                        lower_bound=LOWER_BOUND,
                    )
                    if arch == "gfx950"
                    else None
                )
                reference_gates = []
                if arch == "gfx950" and t == 1:
                    for x in case.layers:
                        gate = original_gate_projection(x["f_a"], x["w_fb"])
                        check_original_gate_projection(gate, x["f_a"], x["w_fb"])
                        reference_gates.append(gate)
                measure_options = dict(
                    reset=case.reset,
                    cold_cache=args.cache == "cold",
                    measurement_blocks=args.measurement_blocks,
                    rounds=args.paired_rounds,
                )
                if arch == "gfx950" and (
                    (t == 1 and batch in (1, 2, 4, 8, 16, 128))
                    or (t == 4 and batch == 1)
                ):
                    report["accepted_parent_verify"]["results"].append(
                        _measure_explicit_gate_verify(
                            timer,
                            case,
                            _original_module(
                                args.accepted_parent_source.resolve(), arch
                            ),
                            measure_options,
                        )
                    )
                    publish()
                verify_measurements = {}
                for name, accepted in _acceptance_patterns(batch, t):
                    report["active_case"]["accepted_length"] = name
                    publish()
                    print(
                        f"CHECK C{batch} T{t} accepted={name} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    case.accepted.copy_(accepted)
                    case.check(graph=False, runner=case)
                    case.check(graph=True, runner=case)
                    candidate_gate_reference_mismatches = case.gate_reference_mismatches
                    all_positive = bool(torch.all(accepted > 0))
                    decode = t == 1 and all_positive
                    original_correctness = _check_original(
                        case,
                        original,
                        graph=False,
                        decode=decode,
                        reference_gates=reference_gates,
                    )
                    if original_correctness["status"] == "passed_eager":
                        original_correctness = _check_original(
                            case,
                            original,
                            graph=True,
                            decode=decode,
                            reference_gates=reference_gates,
                        )
                    baseline_valid = (
                        original_correctness["status"] == "passed_eager_and_graph"
                    )
                    if baseline_valid not in verify_measurements:
                        verify_measurements[baseline_valid] = _paired_measurements(
                            timer,
                            case.verifies[0],
                            (lambda: original.verify(0)) if baseline_valid else None,
                            equivalent=False,
                            **measure_options,
                        )
                    verify_measurement = verify_measurements[baseline_valid]
                    # Prepare original payload even when its standard decode was checked.
                    case.reset()
                    if baseline_valid:
                        for layer in range(args.layers):
                            original.verify(layer)
                    commit_measurement = _paired_measurements(
                        timer,
                        case.commit,
                        original.commit if baseline_valid else None,
                        equivalent=all_positive and baseline_valid,
                        **measure_options,
                    )
                    baseline_lifecycle = (
                        (
                            original.decode_lifecycle
                            if decode
                            else original.replay_lifecycle
                        )
                        if all_positive and baseline_valid
                        else None
                    )
                    lifecycle = _paired_measurements(
                        timer,
                        case.lifecycle,
                        baseline_lifecycle,
                        equivalent=all_positive and baseline_valid,
                        **measure_options,
                    )
                    if parent is not None:
                        parent_result = _measure_parent_lifecycle(
                            timer, case, parent, measure_options
                        )
                        parent_result["accepted_length"] = name
                        report["accepted_parent_lifecycle"]["results"].append(
                            parent_result
                        )
                        measured = parent_result["measurement"]
                        print(
                            f"PARENT C{batch} T{t} accepted={name} "
                            f"candidate_us={measured['candidate']['median_us']:.3f} "
                            f"parent_us={measured['accepted_parent']['median_us']:.3f} "
                            f"speedup={measured['speedup']:.3f}x "
                            f"round_speedups={measured['round_speedups']}",
                            flush=True,
                        )
                    report["results"].append(
                        {
                            "concurrency": batch,
                            "window_length": t,
                            "accepted_length": name,
                            "accepted_counts": accepted.cpu().tolist(),
                            "layers": args.layers,
                            "pool_pages": case.layers[0]["h_pool"].shape[0],
                            "correctness": "passed_eager_and_graph",
                            "original_correctness": original_correctness,
                            "reference_gate_producer": (
                                "gluon_projection_layout"
                                if decode and reference_gates
                                else "torch_mm"
                            ),
                            "candidate_gate_reference": "record gate checked against FP64 before recurrence",
                            "candidate_gate_helper_mismatches": candidate_gate_reference_mismatches,
                            "elapsed_seconds": time.monotonic() - started,
                            "verify": verify_measurement,
                            "commit": commit_measurement,
                            "lifecycle": lifecycle,
                        }
                    )
                    publish()
                    before = (
                        lifecycle["original"]["median_us"]
                        if lifecycle["original"]
                        else None
                    )
                    speedup = lifecycle["speedup"]
                    before_text = "-" if before is None else f"{before:.3f}"
                    speedup_text = "-" if speedup is None else f"{speedup:.3f}x"
                    print(
                        f"{batch} {t} {name} {lifecycle['candidate']['median_us']:.3f} "
                        f"{before_text} {speedup_text} elapsed={time.monotonic() - started:.1f}s "
                        f"baseline={original_correctness['status']}",
                        flush=True,
                    )
                del original, parent, case
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        publish()
        raise
    report["status"] = "complete"
    if arch == "gfx950":
        report["accepted_parent_verify"]["status"] = "complete"
        report["accepted_parent_lifecycle"]["status"] = "complete"
    report["baseline_validation_failures"] = sum(
        row["original_correctness"]["status"] == "failed" for row in report["results"]
    )
    report["comparison_complete"] = report["baseline_validation_failures"] == 0
    report["lifecycle_regressions"] = [
        {
            "concurrency": row["concurrency"],
            "window_length": row["window_length"],
            "accepted_length": row["accepted_length"],
            "speedup": row["lifecycle"]["speedup"],
        }
        for row in report["results"]
        if row["lifecycle"]["speedup"] is not None and row["lifecycle"]["speedup"] < 1
    ]
    report.pop("active_case")
    publish()
    # Keep raw samples in the artifact; log only aggregates below the summary limit.
    print(
        "KDA_COMPARISON_REPORT=" + json.dumps(_summary_without_samples(report)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
