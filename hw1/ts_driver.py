#!/usr/bin/env python3
"""tokenspeed NPU driver for hw1 verification and perf runs.

Wraps run_inference.build_engine (same engine assembly) but drives a list of
cases from a JSON spec, each with `repeat` sequential requests inside one
process (fresh request id per repeat; previous request is Finish'ed before
the next is submitted). Records machine-readable results: exact generated
token ids, TTFT per repeat, per-decode-step times, HBM allocated/peak.

Case spec file:
    [{"name": "corr_len32", "prompt": "corr_len32", "max_new_tokens": 32,
      "repeat": 1}, ...]
`prompt` keys into ~/hw1/prompts.json.

Usage:
    python ts_driver.py --cases cases.json --out ts_out.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

for mode, kernel in (
    ("MHA_PREFILL", "triton_mha_prefill"),
    ("MHA_EXTEND_WITH_KVCACHE", "triton_mha_extend_with_kvcache"),
    ("MHA_DECODE_WITH_KVCACHE", "triton_mha_decode_with_kvcache"),
):
    os.environ.setdefault(f"TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_{mode}", kernel)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_inference import (  # noqa: E402
    UNBOUNDED_CACHED_PREFIX_TOKENS,
    _advance,
    _extend_result_event,
    build_engine,
)
from tokenspeed_scheduler import ForwardEvent, RequestSpec  # noqa: E402
from tokenspeed.runtime.execution.types import PlannedForward, StopCriteria  # noqa: E402
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402


def run_request(handle, scheduler, tokenizer, eos_token_ids, request_id,
                prompt_ids, max_new_tokens):
    sampling_params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    sampling_params.resolve_seed(request_id)
    sampling_params.normalize(tokenizer)

    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = prompt_ids
    spec.max_new_tokens = max_new_tokens
    spec.max_cached_prefix_tokens = UNBOUNDED_CACHED_PREFIX_TOKENS

    generated: list[int] = []
    decode_steps: list[float] = []
    ttft = None
    finished = False
    scheduler.submit_requests([spec])
    t_submit = time.perf_counter()
    while not finished:
        plan = scheduler.next_execution_plan()
        forward_ops = [op for op in plan.forward if op.request_ids]
        if not forward_ops:
            raise RuntimeError("scheduler emitted an idle plan")
        forward_op = forward_ops[0]
        request_ids = list(forward_op.request_ids)
        planned = PlannedForward(
            forward_op=forward_op,
            sampling_params_list=[sampling_params for _ in request_ids],
            dp_metadata=None,
            grammar_inputs=None,
            multimodal_context=None,
            ngram_inputs=None,
            request_history_seeds=None,
            input_logprob_plan=None,
            sampled_top_k=0,
            stop_criteria=tuple(
                StopCriteria(
                    stop_token_ids=tuple(sorted(eos_token_ids)),
                    token_limit=len(prompt_ids) + max_new_tokens,
                )
                for _ in request_ids
            ),
        )
        from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan
        t0 = time.perf_counter()
        results = handle.execute(plan, planned,
                                 cache_ops=cache_ops_from_plan(plan).device)
        results.sync()
        step = time.perf_counter() - t0
        new_tokens = list(results.output_tokens.tolist())
        generated.extend(new_tokens)
        if ttft is None:
            ttft = time.perf_counter() - t_submit
        else:
            decode_steps.extend([step] * len(new_tokens))
        finished = (new_tokens[-1] in eos_token_ids
                    or len(generated) >= max_new_tokens)
        events = [_extend_result_event(request_id, new_tokens)]
        if finished:
            fin = ForwardEvent.Finish()
            fin.request_id = request_id
            events.append(fin)
        _advance(scheduler, events)
    return {"generated_ids": generated, "ttft_ms": ttft * 1e3,
            "decode_steps_ms": [s * 1e3 for s in decode_steps]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path",
                    default=os.path.expanduser("~/models/Qwen3.5-0.8B"))
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--enable-prefix-caching", action="store_true",
                    help="default off here: every request must do a real prefill")
    args = ap.parse_args()

    with open(args.cases) as f:
        cases = json.load(f)
    with open(os.path.expanduser("~/hw1/prompts.json")) as f:
        prompts = json.load(f)

    class NS:
        max_num_seqs = 16
        gpu_memory_utilization = args.gpu_memory_utilization
        enable_prefix_caching = args.enable_prefix_caching

    t_build = time.perf_counter()
    server_args, model_config, device, scheduler, tokenizer = build_engine(
        args.model_path, NS)
    build_s = time.perf_counter() - t_build
    print(f"[load] engine built in {build_s:.1f}s", flush=True)

    eos_token_ids = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id else set()
    hf_eos = model_config.hf_eos_token_id
    if hf_eos is not None:
        if isinstance(hf_eos, (list, tuple, set, frozenset)):
            eos_token_ids.update(int(t) for t in hf_eos)
        else:
            eos_token_ids.add(int(hf_eos))
    print(f"[load] eos ids: {sorted(eos_token_ids)}", flush=True)

    handle = device.handle
    handle.bind_forward_thread()

    out = {"build_seconds": round(build_s, 2), "cases": {}}
    for case in cases:
        name = case["name"]
        prompt_ids = prompts[case["prompt"]]["ids"]
        max_new = case["max_new_tokens"]
        reps = case.get("repeat", 1)
        out["cases"][name] = {"prompt_len": len(prompt_ids),
                              "max_new_tokens": max_new, "reps": []}
        for rep in range(reps):
            rid = f"{name}-rep{rep}"
            r = run_request(handle, scheduler, tokenizer, eos_token_ids,
                            rid, prompt_ids, max_new)
            r["text"] = tokenizer.decode(r["generated_ids"],
                                         skip_special_tokens=True)
            out["cases"][name]["reps"].append(r)
            steady = r["decode_steps_ms"][1:]
            print(f"[case {name} rep{rep}] prompt={len(prompt_ids)} "
                  f"gen={len(r['generated_ids'])} ttft={r['ttft_ms']:.1f}ms "
                  f"decode_first={r['decode_steps_ms'][0] if r['decode_steps_ms'] else float('nan'):.1f}ms "
                  f"decode_avg={sum(steady)/len(steady) if steady else float('nan'):.2f}ms "
                  f"text={r['text'][:60]!r}", flush=True)
    out["hbm_allocated_gib"] = round(torch.npu.memory_allocated() / 2**30, 2)
    out["hbm_peak_gib"] = round(torch.npu.max_memory_allocated() / 2**30, 2)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[done] HBM alloc={out['hbm_allocated_gib']}GiB "
          f"peak={out['hbm_peak_gib']}GiB -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
