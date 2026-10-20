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

"""AMD KDA correctness matrix and separate verify/commit graph timings."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
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


def _reference(x, t, accepted):
    """Use the existing conv producer and independent FP32 torch recurrence.

    The conv producer supplies the rounded BF16 inputs, as in the existing
    replay correctness tests. This checks the fused AMD producer as well as
    recurrence without using its saved U or K as reference inputs.
    """
    batch = x["read_indices"].numel()
    heads = x["beta"].shape[-1]
    width = heads * K
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
    raw_gate = torch.mm(x["f_a"], x["w_fb"].t()).view(batch, t, heads, K)
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
        for layer, x in enumerate(self.layers):
            x["read_indices"] = self.reads[layer % group_count]
            self.originals.append((x["conv_pool"].clone(), x["h_pool"].clone()))
            self.verifies.append(_prepare_records(x, t))
        self.descriptors = _batched_descriptor(self.layers)
        for layer, x in enumerate(self.layers):
            self.descriptors[layer, 3] = x["correction"].data_ptr()
            self.descriptors[layer, 5] = x["key_gate"].data_ptr()
        self.stamps = torch.stack([x["stamps"] for x in self.layers])
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

    def check(self, *, graph: bool):
        """Check output, records, accepted state and every untouched pool page."""
        self.reset()
        for verify in self.verifies:
            verify()
        if graph:
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                for verify in self.verifies:
                    verify()
            capture.replay()
        references = []
        for x, (conv, state) in zip(self.layers, self.originals, strict=True):
            torch.testing.assert_close(x["conv_pool"], conv, atol=0, rtol=0)
            torch.testing.assert_close(x["h_pool"], state, atol=0, rtol=0)
            output, correction, kg, final_state, final_conv = _reference(
                x, self.t, self.accepted
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
        self.stamps.copy_(torch.stack([x["stamps"] for x in self.layers]))
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


def _acceptance_patterns(batch: int, t: int):
    for length in range(t + 1):
        yield str(length), torch.full(
            (batch,), length, device="cuda", dtype=torch.int32
        )
    if batch > 1:
        yield "mixed", torch.arange(batch, device="cuda", dtype=torch.int32) % (t + 1)


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
    parser.add_argument("--measurement-blocks", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7100)
    args = parser.parse_args()
    if min(*args.concurrency, args.layers, args.measurement_blocks) < 1:
        parser.error("concurrency, layers and measurement blocks must be positive")
    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        parser.error("this matrix requires AMD gfx950 or gfx1250")
    timer = GraphTimer(
        GraphBenchmarkConfig(eager_warmup_iterations=5, replay_warmup_iterations=3)
    )
    root = Path(__file__).resolve().parents[5]
    report = {
        "schema_version": 1,
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
        "device": platform.device_name,
        "dtype": "bfloat16",
        "heads": _HEADS,
        "head_dim": K,
        "seed": args.seed,
        "page_relation": args.page_relation,
        "cache": args.cache,
        "verify_scope": "one fused AMD conv/recurrence kernel, gate projection excluded",
        "commit_scope": "all requested layers, includes conv history; verify excluded",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("C T accepted verify_1layer_us commit_all_layers_us", flush=True)
    try:
        for batch in args.concurrency:
            for t in args.window_lengths:
                report["active_case"] = {"concurrency": batch, "window_length": t}
                case = ReplayCase(
                    batch,
                    t,
                    args.layers,
                    in_place=args.page_relation == "in-place",
                    seed=args.seed,
                )
                verify_measurement = None
                for name, accepted in _acceptance_patterns(batch, t):
                    report["active_case"]["accepted_length"] = name
                    case.accepted.copy_(accepted)
                    # Validation must pass before either timing is published.
                    case.check(graph=False)
                    case.check(graph=True)
                    case.reset()
                    if verify_measurement is None:
                        verify_measurement = timer.measure(
                            PreparedInvocation(invoke=case.verifies[0]),
                            cold_cache=args.cache == "cold",
                            measurement_blocks=args.measurement_blocks,
                        )
                    commit_measurement = timer.measure(
                        PreparedInvocation(invoke=case.commit, reset=case.reset),
                        cold_cache=args.cache == "cold",
                        measurement_blocks=args.measurement_blocks,
                    )
                    report["results"].append(
                        {
                            "concurrency": batch,
                            "window_length": t,
                            "accepted_length": name,
                            "accepted_counts": accepted.cpu().tolist(),
                            "commit_layers": args.layers,
                            "correctness": "passed_eager_and_graph",
                            "verify": asdict(verify_measurement),
                            "commit": asdict(commit_measurement),
                        }
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        f"{batch} {t} {name} {verify_measurement.median_us:.3f} "
                        f"{commit_measurement.median_us:.3f}",
                        flush=True,
                    )
                del case
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    report["status"] = "complete"
    report.pop("active_case")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
