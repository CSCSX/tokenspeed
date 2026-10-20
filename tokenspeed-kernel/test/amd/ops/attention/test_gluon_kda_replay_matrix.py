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
import json
import math
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from amd.ops.attention.kda_replay_baseline import (
    OriginalKda,
    check_original_gate_projection,
    original_gate_projection,
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

_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
_HEADS = 12
_STATE_PAGE_BYTES = 884736


def _reference(x, t, accepted, *, rounded_conv: bool, raw_gate=None):
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
            lower_bound=LOWER_BOUND,
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
        self.raw_gates = []
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
            raw_gate = torch.mm(x["f_a"], x["w_fb"].t())
            self.raw_gates.append(raw_gate)
            self.retained_qkv.append(x["qkv_raw"].clone())
            self.verifies.append(_prepare_records(x, t, raw_gate=raw_gate))
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
            qkv_stride=x["qkv_raw"].stride(0),
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
        for layer, x in enumerate(self.layers):
            self.retained_qkv[layer].copy_(x["qkv_raw"])
            torch.mm(x["f_a"], x["w_fb"].t(), out=self.raw_gates[layer])
            self.verifies[layer]()

    def lifecycle(self):
        self.verify_all()
        self.commit()

    def check(self, *, graph: bool):
        """Check output, records, accepted state and every untouched pool page."""
        self.reset()
        self.verify_all()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                self.verify_all()
            capture.replay()
        references = []
        for x, (conv, state) in zip(self.layers, self.originals, strict=True):
            torch.testing.assert_close(x["conv_pool"], conv, atol=0, rtol=0)
            torch.testing.assert_close(x["h_pool"], state, atol=0, rtol=0)
            output, correction, kg, final_state, final_conv = _reference(
                x, self.t, self.accepted, rounded_conv=True
            )
            torch.testing.assert_close(
                x["verify_output"].view_as(output), output, atol=1.6e-2, rtol=1e-2
            )
            positions = x["metadata"][:, 2, None] + torch.arange(self.t, device="cuda")
            ring = positions % self.t
            rows = torch.arange(self.batch, device="cuda")[:, None]
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
                self.commit()
            # Capture may execute the operation on some backends.
            self.reset()
            capture.replay()
        else:
            self.commit()
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
        case.check(graph=False)
        case.check(graph=True)


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
        "verify_scope": "one layer; excludes gate GEMM and candidate QKV retention copy",
        "commit_scope": "all layers; includes conv history; excludes verify",
        "lifecycle_scope": {
            "candidate": "all layers: QKV retention copy, gate GEMM, records verify, then one batched commit",
            "original_t1": "all layers: original fused decode; gfx1250 also includes gate GEMM",
            "original_t4": "all layers: gate GEMM, capture verify, then one batched raw-g replay",
        },
        "comparison_note": "Ratios apply only to equal work. Original replay does not copy zero-acceptance pages. Original standard decode accepts one token.",
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
                verify_measurements = {}
                for name, accepted in _acceptance_patterns(batch, t):
                    report["active_case"]["accepted_length"] = name
                    publish()
                    print(
                        f"CHECK C{batch} T{t} accepted={name} elapsed={time.monotonic() - started:.1f}s",
                        flush=True,
                    )
                    case.accepted.copy_(accepted)
                    case.check(graph=False)
                    case.check(graph=True)
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
                            equivalent=baseline_valid,
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
                del original, case
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        publish()
        raise
    report["status"] = "complete"
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
