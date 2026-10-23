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

"""Compare complete Kimi-K3 source revisions on one AMD TP8 allocation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import signal
import site
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import aiohttp


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def execute(command, *, cwd, env, log):
    with log.open("w") as output:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prepare_arm(root, work, output, name, revision):
    source = work / f"source-{name}"
    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", revision],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), revision],
        cwd=root,
        check=True,
    )
    venv = work / f"venv-{name}"
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True
    )
    python = venv / "bin/python"
    # A nested venv inherits the base interpreter, not its parent's packages.
    # Add the prepared CI environment after this venv's own site-packages so
    # each arm still imports its separately installed scheduler first.
    site_path = Path(
        subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    parent_sites = [
        p for p in site.getsitepackages() if Path(p).is_relative_to(sys.prefix)
    ]
    (site_path / "kda_ci_dependencies.pth").write_text(
        "import site; "
        + "; ".join(f"site.addsitedir({p!r})" for p in parent_sites)
        + "\n"
    )
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env['PATH']}"
    env["PYTHONPATH"] = os.pathsep.join(
        str(source / rel)
        for rel in (
            "python",
            "tokenspeed-kernel/python",
            "tokenspeed-kernel-amd/python",
        )
    )
    # The branches have different native scheduler sources and versions.
    execute(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--ignore-installed",
            str(source / "tokenspeed-scheduler"),
            f"--config-settings=build-dir={work / f'build-scheduler-{name}'}",
        ],
        cwd=source,
        env=env,
        log=output / f"install-{name}.log",
    )
    audit = """
import hashlib, importlib.metadata, json
from pathlib import Path
import tokenspeed, tokenspeed_kernel, tokenspeed_kernel_amd, tokenspeed_scheduler
modules = [tokenspeed, tokenspeed_kernel, tokenspeed_kernel_amd, tokenspeed_scheduler]
print(json.dumps({'modules': {m.__name__: str(Path(m.__file__).resolve()) for m in modules},
 'scheduler_files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
 for p in Path(tokenspeed_scheduler.__file__).parent.rglob('*.so')},
 'versions': {n: importlib.metadata.version(n) for n in ['torch', 'tokenspeed-triton', 'tokenspeed-scheduler']}}))
"""
    info = json.loads(
        subprocess.check_output(
            [str(python), "-c", audit], cwd=source, env=env, text=True
        )
    )
    for name in ("tokenspeed", "tokenspeed_kernel", "tokenspeed_kernel_amd"):
        if not Path(info["modules"][name]).is_relative_to(source):
            raise RuntimeError(f"Wrong source loaded for {name}: {info}")
    if not Path(info["modules"]["tokenspeed_scheduler"]).is_relative_to(venv):
        raise RuntimeError(f"Wrong scheduler loaded: {info}")
    return {"source": source, "python": python, "env": env, "audit": info}


def make_payloads(tokenizer, *, num_requests, input_tokens, output_tokens, seed):
    # The OpenAI gateway accepts text prompts, not arrays of token IDs.
    # ASCII text avoids partial UTF-8 characters when truncating token prefixes.
    rng = random.Random(seed)
    payloads = []
    for _ in range(num_requests):
        text = " ".join(
            f"word{rng.randrange(1_000_000):06d}" for _ in range(input_tokens)
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        budget = input_tokens - tokenizer.num_special_tokens_to_add(pair=False)
        for _ in range(8):
            prompt = tokenizer.decode(
                tokens[:budget], clean_up_tokenization_spaces=False
            )
            actual = len(tokenizer.encode(prompt, add_special_tokens=True))
            if actual == input_tokens:
                break
            budget += input_tokens - actual
        else:
            raise RuntimeError(f"Cannot construct an exact {input_tokens}-token prompt")
        payloads.append(
            json.dumps(
                {
                    "model": "kimi-k3",
                    "prompt": prompt,
                    "max_tokens": output_tokens,
                    "temperature": 0,
                    "seed": seed,
                    "ignore_eos": True,
                    "skip_special_tokens": False,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                separators=(",", ":"),
            )
        )
    return payloads


async def stream_request(session, url, payload, gate, index, expected_tokens):
    await gate.wait()
    started = time.perf_counter()
    first = None
    usage = None
    text_hash = hashlib.sha256()
    async with session.post(
        url, data=payload, headers={"Content-Type": "application/json"}
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {(await response.text())[:1000]}"
            )
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            item = json.loads(line[5:].strip())
            if "error" in item:
                raise RuntimeError(str(item["error"]))
            for choice in item.get("choices", []):
                text = (
                    choice.get("text", choice.get("delta", {}).get("content", "")) or ""
                )
                if text:
                    first = first or time.perf_counter()
                    text_hash.update(text.encode())
            if item.get("usage"):
                usage = item["usage"]
    ended = time.perf_counter()
    if usage is None or usage["completion_tokens"] != expected_tokens or first is None:
        raise RuntimeError(f"Incomplete request {index}: {usage}")
    return {
        "index": index,
        "started": started,
        "ended": ended,
        "latency_s": ended - started,
        "ttft_s": first - started,
        "tpot_s": (ended - first) / (expected_tokens - 1),
        "usage": usage,
        "output_sha256": text_hash.hexdigest(),
    }


def counters(text):
    names = (
        "spec_decode_num_drafts",
        "spec_decode_num_draft_tokens",
        "spec_decode_num_accepted_tokens",
    )
    result = dict.fromkeys(names, 0.0)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for name in names:
            if line.startswith(f"tokenspeed:{name}_total"):
                result[name] += float(line.split()[-1])
    return result


async def wave(url, metrics_url, payloads, output_tokens):
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=len(payloads))
    ) as session:
        async with session.get(metrics_url) as response:
            response.raise_for_status()
            before = await response.text()
        gate = asyncio.Event()
        tasks = [
            asyncio.create_task(
                stream_request(
                    session, f"{url}/v1/completions", p, gate, i, output_tokens
                )
            )
            for i, p in enumerate(payloads)
        ]
        started = time.perf_counter()
        gate.set()
        requests = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started
        async with session.get(metrics_url) as response:
            response.raise_for_status()
            after = await response.text()
    initial, final = counters(before), counters(after)
    delta = {k: final[k] - initial[k] for k in final}
    rounds = delta["spec_decode_num_drafts"]
    proposed = delta["spec_decode_num_draft_tokens"]
    accepted = delta["spec_decode_num_accepted_tokens"]
    return {
        "elapsed_s": elapsed,
        "throughput_tokens_s": len(payloads) * output_tokens / elapsed,
        "request_latency_mean_s": statistics.mean(r["latency_s"] for r in requests),
        "ttft_mean_s": statistics.mean(r["ttft_s"] for r in requests),
        "tpot_mean_s": statistics.mean(r["tpot_s"] for r in requests),
        "accepted_length": 1 + accepted / rounds if rounds else None,
        "acceptance_rate": accepted / proposed if proposed else None,
        "spec_counters": delta,
        "requests": requests,
        "metrics_before": before,
        "metrics_after": after,
    }


def stop_server(process):
    # Only this invocation's process group; never kill other CI jobs.
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def wait_ready(process, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}")
        try:
            with urlopen(f"{url}/readiness", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise TimeoutError("Server readiness timeout")


def serve_command(arm, args, model_revision, draft_path, mode):
    command = [
        str(arm["python"]),
        "-m",
        "tokenspeed.cli",
        "serve",
        "--model",
        args.model,
        "--revision",
        model_revision,
        "--served-model-name",
        "kimi-k3",
        "--tp",
        "8",
        "--ep-size",
        "1",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "gluon",
        "--moe-backend",
        "auto",
        "--kv-cache-dtype",
        "fp8",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "128",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--max-cudagraph-capture-size",
        "128",
        "--cudagraph-capture-sizes",
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
        "--disable-prefill-graph",
        "--gpu-memory-utilization",
        "0.92",
        "--trust-remote-code",
        "--sampling-backend",
        "greedy",
        "--seed",
        "1",
        "--disable-kvstore",
        "--kvstore-ratio",
        "0.0",
        "--disable-prefix-caching",
        "--mm-encoder-tp-mode",
        "data",
        "--host",
        "127.0.0.1",
        "--port",
        "21000",
        "--policy",
        "round_robin",
        "--engine-startup-timeout",
        "3000",
        "--gateway-startup-timeout",
        "600",
        "--enable-metrics",
        "--metrics-port",
        "21001",
        "--metrics-host",
        "127.0.0.1",
    ]
    if mode != "t1":
        command.extend(
            [
                "--speculative-algorithm",
                "EAGLE3" if mode == "eagle3" else "MTP",
                "--speculative-num-steps",
                "3",
                "--speculative-num-draft-tokens",
                "4",
                "--speculative-eagle-topk",
                "1",
                "--synthetic-acceptance-length",
                "3.7",
                "--drafter-attention-backend",
                "gluon",
            ]
        )
    if mode == "eagle3":
        command.extend(
            [
                "--speculative-draft-model-path",
                str(draft_path),
                "--speculative-draft-model-quantization",
                "unquant",
                "--eagle3-layers-to-capture",
                "2,46,90",
            ]
        )
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--candidates", nargs="+", required=True, metavar="NAME=REVISION"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--draft-revision")
    parser.add_argument(
        "--modes", nargs="+", choices=("t1", "mtp3", "eagle3"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--warmup-waves", type=int, default=2)
    parser.add_argument("--measured-waves", type=int, default=3)
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="Repeat the mirrored server order; each pass starts every arm twice",
    )
    args = parser.parse_args()
    candidates = dict(item.split("=", 1) for item in args.candidates)
    if "main" in candidates or len(candidates) != len(args.candidates):
        parser.error("Candidate names must be unique and must not be 'main'")
    if not all(0 < c <= 128 for c in args.concurrency):
        parser.error("Concurrency must be between 1 and the 128 prepared requests")
    if "eagle3" in args.modes:
        if not args.draft_model or not args.draft_revision:
            parser.error("EAGLE3 requires --draft-model and --draft-revision")
    elif args.draft_model or args.draft_revision:
        parser.error("Draft checkpoint arguments require --modes eagle3")
    root = Path.cwd()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    work = root / ".ci-artifacts/kda-e2e-work"
    work.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "preparing",
        "baseline": args.baseline,
        "candidates": candidates,
        "input_tokens": 4096,
        "output_tokens": 371,
        "concurrency": args.concurrency,
        "tp": 8,
        "ep": 1,
        "modes": args.modes,
        "synthetic_acceptance_length": 3.7,
        "warmup_waves": args.warmup_waves,
        "measured_waves_per_start": args.measured_waves,
        "server_order": ["main", *candidates, *reversed(candidates), "main"]
        * args.passes,
        "scope": "HTTP request E2E including prefill; prefix cache disabled; full source revisions with separately built schedulers; no profiler",
        "submission": "identical token requests released by one client barrier; scheduler batch identity is not guaranteed",
        "runs": [],
        "errors": [],
    }
    write_json(args.output / "report.json", report)
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer

    if torch.version.hip is None or torch.cuda.device_count() != 8:
        raise RuntimeError("Requires an exclusive eight-GPU AMD allocation")
    report["hardware"] = [str(torch.cuda.get_device_properties(i)) for i in range(8)]
    report["torch_version"] = torch.__version__
    report["hip_version"] = torch.version.hip
    write_json(args.output / "report.json", report)
    config = Path(
        hf_hub_download(args.model, "config.json", revision=args.model_revision)
    )
    model_revision = config.parent.name
    report["model"] = args.model
    report["model_revision"] = model_revision
    report["model_config"] = json.loads(config.read_text())
    text_config = report["model_config"].get("text_config", report["model_config"])
    draft_path = None
    report["draft"] = None
    if "eagle3" in args.modes:
        # The serving CLI has one shared revision argument. Pass a pinned local
        # draft snapshot so it cannot resolve against the target model's revision.
        draft_path = Path(
            snapshot_download(args.draft_model, revision=args.draft_revision)
        )
        report["draft"] = {
            "model": args.draft_model,
            "revision": draft_path.name,
            "config": json.loads((draft_path / "config.json").read_text()),
        }
    print(
        "KDA_E2E_ENV="
        + json.dumps(
            {
                "hardware": report["hardware"],
                "model_revision": model_revision,
                "draft": report["draft"],
                "num_nextn_predict_layers": text_config.get("num_nextn_predict_layers"),
                "architectures": report["model_config"].get("architectures"),
                "num_hidden_layers": text_config.get("num_hidden_layers"),
                "linear_attn_config": text_config.get("linear_attn_config"),
            }
        ),
        flush=True,
    )
    write_json(args.output / "report.json", report)
    if "mtp3" in args.modes and not text_config.get("num_nextn_predict_layers", 0):
        raise ValueError("The model checkpoint declares no native MTP layers")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=model_revision, trust_remote_code=True
    )
    payloads = make_payloads(
        tokenizer, num_requests=128, input_tokens=4096, output_tokens=371, seed=1
    )
    report["prompt_format"] = "text; tokenizer-verified 4096 input tokens"
    print("KDA_E2E_INPUTS: 128 text prompts, each verified as 4096 tokens", flush=True)
    report["payload_sha256"] = [
        hashlib.sha256(p.encode()).hexdigest() for p in payloads
    ]
    arms = {
        name: prepare_arm(root, work, args.output, name, revision)
        for name, revision in {"main": args.baseline, **candidates}.items()
    }
    report["source_audit"] = {name: arm["audit"] for name, arm in arms.items()}
    report["status"] = "running"
    write_json(args.output / "report.json", report)
    url = "http://127.0.0.1:21000"
    metrics_url = "http://127.0.0.1:21001/metrics"
    report["metrics_url"] = metrics_url
    failed_arms = set()
    for mode in report["modes"]:
        for start_index, name in enumerate(report["server_order"]):
            if (mode, name) in failed_arms:
                continue
            arm = arms[name]
            label = f"{mode}-{start_index}-{name}"
            command = serve_command(arm, args, model_revision, draft_path, mode)
            run = {
                "label": label,
                "mode": mode,
                "arm": name,
                "command": command,
                "waves": [],
                "status": "starting",
            }
            report["runs"].append(run)
            write_json(args.output / "report.json", report)
            print(f"KDA_E2E_START {label}", flush=True)
            with (args.output / f"{label}-server.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=arm["source"],
                    env=arm["env"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_ready(process, url, 3600)
                    run["status"] = "measuring"
                    for concurrency in report["concurrency"]:
                        for repetition in range(
                            args.warmup_waves + args.measured_waves
                        ):
                            result = asyncio.run(
                                wave(url, metrics_url, payloads[:concurrency], 371)
                            )
                            result.update(
                                concurrency=concurrency,
                                repetition=repetition,
                                warmup=repetition < args.warmup_waves,
                            )
                            for r in result["requests"]:
                                if r["usage"]["prompt_tokens"] != 4096:
                                    raise RuntimeError(
                                        f"Unexpected prompt length: {r['usage']}"
                                    )
                            if mode != "t1" and result["accepted_length"] is None:
                                raise RuntimeError(
                                    "Speculative acceptance counters were not exported"
                                )
                            filename = f"{label}-c{concurrency}-r{repetition}.json"
                            write_json(args.output / filename, result)
                            run["waves"].append(
                                {
                                    k: v
                                    for k, v in result.items()
                                    if k
                                    not in (
                                        "requests",
                                        "metrics_before",
                                        "metrics_after",
                                    )
                                }
                            )
                            write_json(args.output / "report.json", report)
                            print(
                                f"KDA_E2E_WAVE {label} C{concurrency} r{repetition} seconds={result['elapsed_s']:.6f} acceptance={result['accepted_length']}",
                                flush=True,
                            )
                    run["status"] = "complete"
                except Exception as error:
                    failed_arms.add((mode, name))
                    run["status"] = "failed"
                    run["error"] = f"{type(error).__name__}: {error}"
                    report["errors"].append({"run": label, "error": run["error"]})
                    print(f"KDA_E2E_ERROR {label} {run['error']}", flush=True)
                    log.flush()
                    tail = (
                        (args.output / f"{label}-server.log")
                        .read_text(errors="replace")
                        .splitlines()[-100:]
                    )
                    print("\n".join(tail), flush=True)
                finally:
                    stop_server(process)
                    write_json(args.output / "report.json", report)
    expected_waves = 2 * args.passes * args.measured_waves
    comparisons = []
    for mode in report["modes"]:
        for concurrency in report["concurrency"]:
            row = {"mode": mode, "concurrency": concurrency}
            for name in arms:
                samples = [
                    w
                    for r in report["runs"]
                    if r["mode"] == mode
                    and r["arm"] == name
                    and r["status"] == "complete"
                    for w in r["waves"]
                    if w["concurrency"] == concurrency and not w["warmup"]
                ]
                row[name] = {"waves": len(samples)}
                if len(samples) == expected_waves:
                    for key in (
                        "elapsed_s",
                        "throughput_tokens_s",
                        "request_latency_mean_s",
                        "ttft_mean_s",
                        "tpot_mean_s",
                    ):
                        row[name][key] = statistics.median(w[key] for w in samples)
                    if mode != "t1":
                        row[name]["accepted_length"] = statistics.mean(
                            w["accepted_length"] for w in samples
                        )
                        row[name]["acceptance_rate"] = statistics.mean(
                            w["acceptance_rate"] for w in samples
                        )
            if row["main"]["waves"] == expected_waves:
                row["speedup"] = {
                    name: row["main"]["elapsed_s"] / row[name]["elapsed_s"]
                    for name in candidates
                    if row[name]["waves"] == expected_waves
                }
            comparisons.append(row)
    report["comparisons"] = comparisons
    report["status"] = "failed" if report["errors"] else "complete"
    write_json(args.output / "report.json", report)
    print("KDA_E2E_COMPARISON=" + json.dumps(comparisons), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
