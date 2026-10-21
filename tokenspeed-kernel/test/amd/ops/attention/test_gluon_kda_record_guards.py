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

"""Check record guards in isolated processes on hardware or a fresh emulator.

The exact assertion prefix is from the compiled device-print payload. Both
architectures need runtime validation that this payload flushes before a trap.
The suite controller runs outside rocJITsu so a failed daemon cannot block it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

_CASES = (
    "invalid-verify",
    "invalid-stamped-interval",
    "invalid-unstamped-interval",
    "stale-stamp",
    "valid-stamped",
    "valid-unstamped",
)
_MESSAGES = {
    "invalid-verify": "invalid replay metadata",
    "invalid-stamped-interval": "invalid replay interval",
    "invalid-unstamped-interval": "invalid replay interval",
    "stale-stamp": "stale replay record",
}
_EVENT = "KDA_GUARD_EVENT="


class _GuardCase:
    def __init__(self, arch: str, device: str):
        import torch
        from amd.ops.attention.test_gluon_kda_replay_compile import _inputs

        self.module = importlib.import_module(
            f"tokenspeed_kernel_amd.ops.{arch}.attention.kda.decode"
        )
        self.verify_launcher = getattr(self.module, f"gluon_kda_fused_verify_{arch}")
        self.commit_launcher = getattr(
            self.module, f"launch_gluon_kda_fused_replay_{arch}"
        )
        self.projection = (
            {"f_a_out": None, "f_b_weight": None} if arch == "gfx950" else {}
        )
        self.x = _inputs(1, device, batch=1)
        generator = torch.Generator().manual_seed(8952)
        for name in (
            "qkv_raw",
            "conv_w",
            "conv_pool",
            "beta",
            "A_log",
            "dt_bias",
            "h_pool",
        ):
            value = self.x[name]
            value.copy_((torch.randn(value.shape, generator=generator) * 0.1).to(value))
        self.raw_gate = torch.randn(1, 12 * 128, generator=generator).to(
            device=device, dtype=torch.bfloat16
        )
        self.retained = torch.empty_like(self.x["qkv_raw"])
        self.correction = torch.full(
            (1, 129, 12, 128), float("nan"), device=device, dtype=torch.float32
        )
        self.key_gate = torch.full(
            (1, 129, 12, 256), float("nan"), device=device, dtype=torch.bfloat16
        )
        self.metadata = torch.tensor([[0, 7, 0]], device=device, dtype=torch.int64)
        self.stamps = torch.full((1, 1, 129, 2), -1, device=device, dtype=torch.int64)
        self.boundaries = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.groups = torch.zeros(1, device=device, dtype=torch.int32)
        self.accepted = torch.ones(1, device=device, dtype=torch.int32)
        self.reads = self.x["read_indices"][None]
        self.writes = self.x["write_indices"][None]
        self.descriptors = torch.tensor(
            [
                [
                    self.retained.data_ptr(),
                    0,
                    self.x["conv_pool"].data_ptr(),
                    self.correction.data_ptr(),
                    0,
                    self.key_gate.data_ptr(),
                    self.x["A_log"].data_ptr(),
                    self.x["dt_bias"].data_ptr(),
                    self.x["h_pool"].data_ptr(),
                    0,
                ]
            ],
            device=device,
            dtype=torch.uint64,
        )
        self.initial_state = self.x["h_pool"].clone()
        self.initial_history = self.x["conv_pool"].clone()

    def verify(self):
        x = self.x
        return self.verify_launcher(
            mixed_qkv=x["qkv_raw"],
            original_qkv=self.retained,
            conv_weights=x["conv_w"],
            conv_pool=x["conv_pool"],
            raw_g=self.raw_gate,
            **self.projection,
            beta_logits=x["beta"],
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            state_pool=x["h_pool"],
            read_indices=x["read_indices"],
            correction=self.correction.transpose(1, 2),
            key_gate=self.key_gate.transpose(1, 2),
            record_metadata=self.metadata,
            record_stamps=self.stamps[0],
            query_start_loc=self.boundaries,
            num_heads=12,
            head_dim=128,
            draft_token_num=1,
            lower_bound=-5.0,
        )

    def commit(self, *, stamped: bool):
        x = self.x
        self.commit_launcher(
            self.descriptors,
            self.groups,
            self.reads,
            self.writes,
            self.accepted,
            draft_token_num=1,
            num_heads=12,
            head_dim=128,
            qkv_stride=self.retained.stride(0),
            conv_stride=x["conv_pool"].stride(0),
            correction_stride=self.correction.stride(1),
            key_gate_stride=self.key_gate.stride(1),
            state_stride=x["h_pool"].stride(0),
            conv_width=4,
            activation_dtype=x["qkv_raw"].dtype,
            conv_state_dtype=x["conv_pool"].dtype,
            record_metadata=self.metadata if stamped else None,
            record_stamps=self.stamps if stamped else None,
            lower_bound=-5.0,
        )

    def validate(self, *, verify: bool, stamped: bool):
        from tokenspeed_kernel_amd.ops._kda_records import (
            validate_kda_records_commit,
            validate_kda_records_verify,
        )

        x = self.x
        if verify:
            validate_kda_records_verify(
                x["qkv_raw"],
                self.retained,
                self.raw_gate,
                x["beta"],
                x["h_pool"],
                x["read_indices"],
                self.correction.transpose(1, 2),
                self.key_gate.transpose(1, 2),
                self.metadata,
                self.stamps[0],
                self.boundaries,
                x["A_log"],
                x["dt_bias"],
                12,
                128,
                1,
                f_a_out=None,
                f_b_weight=None,
            )
        else:
            validate_kda_records_commit(
                self.descriptors,
                self.groups,
                self.reads,
                self.writes,
                self.accepted,
                self.metadata if stamped else None,
                self.stamps if stamped else None,
                1,
            )


def _emit(tag: str, arch: str, case: str):
    print(
        _EVENT
        + json.dumps(
            {"tag": tag, "arch": arch, "case": case, "time_ns": time.monotonic_ns()}
        ),
        flush=True,
    )


def _device_case(arch: str, scenario: str):
    import torch
    from amd.ops.attention.test_gluon_kda_replay_matrix import _reference

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    observed = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    _emit("architecture", observed, scenario)
    assert observed == arch, f"Expected {arch}, got {observed}"
    case = _GuardCase(arch, "cuda")
    stamped = scenario not in ("invalid-unstamped-interval", "valid-unstamped")
    output = case.verify()
    reference, _, _, state, history = _reference(
        case.x,
        1,
        case.accepted,
        rounded_conv=True,
        lower_bound=-5.0,
        raw_gate=case.raw_gate,
    )
    torch.testing.assert_close(
        output.view_as(reference), reference, atol=1.6e-2, rtol=1e-2
    )
    case.commit(stamped=stamped)
    expected_state = case.initial_state.clone()
    expected_history = case.initial_history.clone()
    writes = case.x["write_indices"].long()
    expected_state[writes] = state
    expected_history[writes] = history
    torch.testing.assert_close(case.x["h_pool"], expected_state, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(case.x["conv_pool"], expected_history, atol=0, rtol=0)
    torch.cuda.synchronize()
    _emit("ready", observed, scenario)
    if scenario.startswith("valid-"):
        _emit("completed", observed, scenario)
        return

    case.x["h_pool"].copy_(case.initial_state)
    case.x["conv_pool"].copy_(case.initial_history)
    if scenario == "invalid-verify":
        case.metadata[0, 0] = -1
    elif scenario in ("invalid-stamped-interval", "invalid-unstamped-interval"):
        case.accepted.fill_(2)
    else:
        case.stamps[0, 0, 0, 1] += 1
    verify = scenario == "invalid-verify"
    case.validate(verify=verify, stamped=stamped)
    torch.cuda.synchronize()
    _emit("host_validated", observed, scenario)
    if verify:
        case.verify()
    else:
        case.commit(stamped=stamped)
    torch.cuda.synchronize()
    _emit("completed", observed, scenario)


def _processes():
    """Read identities without collecting command lines or environments."""
    result = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text()
            fields = stat.rsplit(") ", 1)[1].split()
            item = {
                "pid": int(directory.name),
                "ppid": int(fields[1]),
                "group": int(fields[2]),
                "session": int(fields[3]),
                "start": int(fields[19]),
                "state": fields[0],
                "executable": stat.split("(", 1)[1].rsplit(") ", 1)[0],
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        try:
            item["executable"] = Path(os.readlink(directory / "exe")).name
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            # Keep zombies and inaccessible processes visible in the identity
            # scan; dropping them could hide a failed daemon cleanup.
            pass
        result[(item["pid"], item["start"])] = item
    return result


def _owned(snapshot, tracked, session):
    tracked = set(tracked)
    while True:
        parents = {pid for pid, start in tracked if (pid, start) in snapshot}
        added = {
            identity
            for identity, row in snapshot.items()
            if row["session"] == session or row["ppid"] in parents
        }
        if added <= tracked:
            return tracked
        tracked.update(added)


def _text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def _check_device_case(
    scenario: str, arch: str, launch_prefix: list[str], output: Path
):
    """Run one case with bounded cleanup; any timeout or survivor is a failure."""
    root = Path(__file__).resolve().parents[5]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(root / name)
        for name in (
            "tokenspeed-kernel/test",
            "tokenspeed-kernel/python",
            "tokenspeed-kernel-amd/python",
        )
    ) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [
        *launch_prefix,
        sys.executable,
        "-m",
        "amd.ops.attention.test_gluon_kda_record_guards",
        "--mode",
        "child",
        "--arch",
        arch,
        "--case",
        scenario,
    ]
    before = _processes()
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    tracked = set()
    timed_out = False
    simulators_seen = {}
    stdout = stderr = ""
    while True:
        snapshot = _processes()
        tracked = _owned(snapshot, tracked, proc.pid)
        simulators_seen.update(
            {
                identity: row
                for identity, row in snapshot.items()
                if identity not in before and "rocjitsu" in row["executable"].lower()
            }
        )
        try:
            stdout, stderr = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired as error:
            stdout, stderr = _text(error.output), _text(error.stderr)
            if time.monotonic() - started >= 180:
                timed_out = True
                break

    # Give a successful launcher's daemon time to exit, then clean only processes
    # whose ownership was observed. Never kill an unattributed new simulator.
    grace = time.monotonic() + (0 if timed_out else 2)
    while True:
        snapshot = _processes()
        tracked = _owned(snapshot, tracked, proc.pid)
        survivors = [row for identity, row in snapshot.items() if identity in tracked]
        if not survivors or time.monotonic() >= grace:
            break
        time.sleep(0.05)
    cleanup = survivors
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for row in cleanup:
        current = _processes()
        if (row["pid"], row["start"]) in current:
            try:
                os.kill(row["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = _text(error.output), _text(error.stderr)
        timed_out = True
        proc.stdout.close()
        proc.stderr.close()
    cleanup_deadline = time.monotonic() + 2
    while True:
        remaining = _processes()
        owned_survivors = [
            row for identity, row in remaining.items() if identity in tracked
        ]
        if not owned_survivors or time.monotonic() >= cleanup_deadline:
            break
        time.sleep(0.05)
    new_simulators = [
        row
        for identity, row in remaining.items()
        if identity not in before and "rocjitsu" in row["executable"].lower()
    ]
    events = [
        json.loads(line[len(_EVENT) :])
        for line in stdout.splitlines()
        if line.startswith(_EVENT)
    ]
    tags = [event["tag"] for event in events]
    expected = _MESSAGES.get(scenario)
    prefix = None if expected is None else f"device assertion failed: '{expected}'"
    report = {
        "case": scenario,
        "arch": arch,
        "scope": (
            "gfx1250 emulator validation" if launch_prefix else "hardware validation"
        ),
        "stdout": stdout,
        "stderr": stderr,
        "events": events,
        "returncode": proc.poll(),
        "wall_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "cleanup": cleanup,
        "owned_survivors": owned_survivors,
        "new_simulators": new_simulators,
        "simulators_seen": list(simulators_seen.values()),
        "expected_device_prefix": prefix,
        "prefix_source": "literal in compiled assertion payload",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{scenario}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    # Preserve the first invalid verify observation before judging other cases.
    print(f"KDA_GUARD_CASE={json.dumps(report)}", flush=True)
    assert not timed_out and not owned_survivors and not new_simulators, path
    if launch_prefix:
        assert simulators_seen, path
    assert all(
        event["arch"] == arch and event["case"] == scenario for event in events
    ), path
    assert "ready" in tags, path
    if expected is not None:
        assert "host_validated" in tags and "completed" not in tags, path
        assert proc.returncode not in (None, 0) and prefix in stdout + stderr, path
    else:
        assert proc.returncode == 0 and "completed" in tags, path
        assert "device assertion failed:" not in stdout + stderr, path


@pytest.mark.parametrize("scenario", _CASES)
def test_kda_record_guards(scenario, tmp_path):
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()
    if not (platform.is_cdna4 or platform.is_cdna5):
        pytest.skip("requires AMD gfx950 or gfx1250")
    arch = "gfx950" if platform.is_cdna4 else "gfx1250"
    _check_device_case(scenario, arch, [], tmp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("child", "suite"), required=True)
    parser.add_argument("--arch", choices=("gfx950", "gfx1250"), required=True)
    parser.add_argument("--case", choices=_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch-prefix", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode == "child":
        if args.case is None:
            parser.error("child mode requires --case")
        _device_case(args.arch, args.case)
    else:
        if args.output is None or args.launch_prefix is None:
            parser.error("suite mode requires --output and an explicit --launch-prefix")
        for scenario in _CASES:
            _check_device_case(scenario, args.arch, args.launch_prefix, args.output)


if __name__ == "__main__":
    main()
