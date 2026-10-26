#!/usr/bin/env python3
"""Minimal pure-Python inference driver for tokenspeed on Ascend NPU.

Loads Qwen3.5-0.8B through the tokenspeed data plane (no ts-host Rust
binary, no gRPC SMG gateway) and runs one request end to end:
prefill + multi-step decode, driven by the C++ ``tokenspeed_scheduler``
through its Python binding.

Assembly follows ``engine/rust_host.py``'s ``HostBootstrap``:
ServerArgs -> ModelConfig -> init_distributed -> build_device_side ->
scheduler_config_from_args -> Scheduler. Each round turns the scheduler's
plan into a ``PlannedForward`` the way ``HostDataPlane.execute`` does for a
single text-only request (no speculation, LoRA, grammar, logprobs or
multimodal), runs it through ``DeviceHandle.execute`` and feeds the sampled
tokens back to the scheduler as ``ForwardEvent.ExtendResult``.

Usage (on the NPU host):

    set +u; source /usr/local/Ascend/cann-9.0.0/set_env.sh
    export ASCEND_RT_VISIBLE_DEVICES=1
    cd ~/tokenspeed
    export PYTHONPATH=$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python
    ~/venvs/aics-hw1/bin/python ~/hw1/run_inference.py \
        --model-path ~/models/Qwen3.5-0.8B --prompt "Hello" --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch
import torch_npu  # noqa: F401  -- registers the "npu" device with torch
from tokenspeed_scheduler import ExecutionEvent, ForwardEvent, RequestSpec, Scheduler

from tokenspeed.runtime.engine.rank_setup import (
    init_distributed,
    load_model_config,
    scheduler_config_from_args,
)
from tokenspeed.runtime.engine.scheduler_utils import cache_ops_from_plan
from tokenspeed.runtime.execution.device import build_device_side
from tokenspeed.runtime.execution.types import PlannedForward, StopCriteria
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import configure_logger
from tokenspeed.runtime.utils.hf_transformers_utils import get_tokenizer
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs

# "No bound" for RequestSpec.max_cached_prefix_tokens (the C++ default).
UNBOUNDED_CACHED_PREFIX_TOKENS = 2**31 - 1

REQUEST_ID = "req-0"


class _WeightMappingAudit(logging.Handler):
    """Counts the model loader's "not found in params_dict" warnings: every
    checkpoint tensor that no model parameter claimed."""

    def __init__(self) -> None:
        super().__init__()
        self.unmapped: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "not found in params_dict" in message:
            self.unmapped.append(message)


def _extend_result_event(request_id: str, tokens: list[int]) -> object:
    event = ForwardEvent.ExtendResult()
    event.request_id = request_id
    event.tokens = list(tokens)
    return event


def _advance(scheduler: Scheduler, events: list) -> None:
    batch = ExecutionEvent()
    for event in events:
        batch.add_event(event)
    scheduler.advance(batch)


def _checkpoint_tensor_count(model_path: str) -> int:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return len(json.load(f)["weight_map"])
    # Single-file checkpoint.
    from safetensors import safe_open

    count = 0
    for entry in os.listdir(model_path):
        if entry.endswith(".safetensors"):
            with safe_open(os.path.join(model_path, entry), framework="pt") as f:
                count += len(f.keys())
    return count


def build_engine(model_path: str, args: argparse.Namespace):
    server_args = ServerArgs(
        model=model_path,
        device="npu",
        dtype="bfloat16",
        attention_backend="mha",
        # The vision tower is built but never run on the text-only path;
        # Ascend has no default encoder backend, so pick the portable one.
        mm_attention_backend="triton_attn",
        # NPU execution refuses both (ServerArgs.validate).
        disable_prefill_graph=True,
        disable_pdl=True,
        # Synchronous loop: the next plan is emitted only after the previous
        # forward's result landed.
        disable_overlap_schedule=True,
        # Verification/perf runs want a genuine prefill on every request;
        # callers may opt out via the namespace. KVStore requires prefix
        # caching, so it must be disabled together (ServerArgs.validate).
        enable_prefix_caching=getattr(args, "enable_prefix_caching", True),
        disable_kvstore=not getattr(args, "enable_prefix_caching", True),
        enforce_eager=True,
        enable_nan_detection=True,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # ascend_full (the NPU default) needs flash_ops' npu_sample operators,
        # which are not built here; greedy is argmax-only and matches the
        # temperature=0 request this driver serves.
        sampling_backend="greedy",
        seed=0,
    )
    # The launcher stamps each rank's id onto its copy of the mapping
    # (entrypoints/engine.py `_start_ranks`); we are rank 0 of 1.
    server_args.mapping.rank = 0
    configure_logger(server_args)
    port_args = PortArgs.init_new(server_args)
    model_config = load_model_config(server_args, server_args.model, draft_of=None)
    gpu_id = 0  # ASCEND_RT_VISIBLE_DEVICES picks the physical chip.
    min_per_gpu_mem = init_distributed(
        server_args,
        port_args,
        gpu_id=gpu_id,
        global_rank=0,
        hidden_size=model_config.hidden_size,
    )
    max_batch_size = server_args.max_num_seqs
    device = build_device_side(
        server_args=server_args,
        model_config=model_config,
        draft_model_config=None,
        gpu_id=gpu_id,
        global_rank=0,
        attn_tp_rank=0,
        min_per_gpu_mem=min_per_gpu_mem,
        overlap_schedule_depth=0,
        decode_input_tokens=1,
        max_batch_size=max_batch_size,
        # No PD disaggregation: no transfer peer.
        kv_transfer_builder=lambda *_, **__: None,
    )
    scheduler_config = scheduler_config_from_args(
        server_args,
        device.specs,
        storage_config=None,
        max_scheduled_tokens=server_args.chunked_prefill_size,
        max_batch_size=max_batch_size,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
        enable_kv_cache_events=False,
        prefix_replay_tokens=0,
    )
    scheduler = Scheduler(scheduler_config)
    tokenizer = get_tokenizer(
        server_args.tokenizer,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        architectures=model_config.hf_config.architectures,
        **model_config.tokenizer_kwargs,
    )
    return server_args, model_config, device, scheduler, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--mha-solution",
        choices=["torch_npu", "triton"],
        default="triton",
        help=(
            "Kernel solution for the full-attention MHA ops. torch_npu's fused "
            "attention rejects this model's head_dim 256 (TND layout supports "
            "64/128/192), so the portable Triton kernels are the default."
        ),
    )
    args = parser.parse_args()

    if args.mha_solution == "triton":
        for mode, kernel in (
            ("MHA_PREFILL", "triton_mha_prefill"),
            ("MHA_EXTEND_WITH_KVCACHE", "triton_mha_extend_with_kvcache"),
            ("MHA_DECODE_WITH_KVCACHE", "triton_mha_decode_with_kvcache"),
        ):
            os.environ.setdefault(f"TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_{mode}", kernel)

    audit = _WeightMappingAudit()
    logging.getLogger().addHandler(audit)
    checkpoint_tensors = _checkpoint_tensor_count(args.model_path)

    t_build = time.perf_counter()
    server_args, model_config, device, scheduler, tokenizer = build_engine(
        args.model_path, args
    )
    build_seconds = time.perf_counter() - t_build
    print(f"[load] engine built in {build_seconds:.1f}s")
    print(f"[load] checkpoint tensors: {checkpoint_tensors}")
    print(f"[load] unmapped checkpoint tensors (missing keys): {len(audit.unmapped)}")
    for message in audit.unmapped:
        print(f"[load]   {message}")

    eos_token_ids = set()
    hf_eos = model_config.hf_eos_token_id
    if hf_eos is not None:
        if isinstance(hf_eos, (list, tuple, set, frozenset)):
            eos_token_ids.update(hf_eos)
        else:
            eos_token_ids.add(hf_eos)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    eos_token_ids = {int(t) for t in eos_token_ids}
    print(f"[load] eos token ids: {sorted(eos_token_ids)}")

    sampling_params = SamplingParams(
        temperature=0.0, max_new_tokens=args.max_new_tokens
    )
    sampling_params.resolve_seed(REQUEST_ID)
    sampling_params.normalize(tokenizer)

    prompt_ids = tokenizer.encode(args.prompt)
    print(f"[run] prompt: {args.prompt!r} -> {len(prompt_ids)} tokens")

    spec = RequestSpec()
    spec.request_id = REQUEST_ID
    spec.tokens = prompt_ids
    spec.max_new_tokens = args.max_new_tokens
    spec.max_cached_prefix_tokens = UNBOUNDED_CACHED_PREFIX_TOKENS

    handle = device.handle
    handle.bind_forward_thread()

    generated: list[int] = []
    ttft_seconds: float | None = None
    decode_step_seconds: list[float] = []
    finished = False

    scheduler.submit_requests([spec])
    t_submit = time.perf_counter()
    while not finished:
        plan = scheduler.next_execution_plan()
        forward_ops = [op for op in plan.forward if op.request_ids]
        if not forward_ops:
            raise RuntimeError(
                "scheduler emitted an idle plan before the request finished"
            )
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
                    token_limit=len(prompt_ids) + args.max_new_tokens,
                )
                for _ in request_ids
            ),
        )
        t0 = time.perf_counter()
        results = handle.execute(
            plan, planned, cache_ops=cache_ops_from_plan(plan).device
        )
        results.sync()
        step_seconds = time.perf_counter() - t0

        new_tokens = list(results.output_tokens.tolist())
        nan_flags = (
            list(results.output_nan_flags.tolist())
            if results.output_nan_flags is not None
            else None
        )
        generated.extend(new_tokens)
        if ttft_seconds is None:
            ttft_seconds = time.perf_counter() - t_submit
        else:
            decode_step_seconds.extend([step_seconds] * len(new_tokens))

        last = new_tokens[-1]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        nan_note = f" NaN-logits!" if nan_flags and any(nan_flags) else ""
        print(
            f"[token {len(generated):3d}] id={last} {text!r} "
            f"({step_seconds * 1e3:.1f} ms){nan_note}"
        )
        finished = last in eos_token_ids or len(generated) >= args.max_new_tokens
        events = [_extend_result_event(REQUEST_ID, new_tokens)]
        if finished:
            finish = ForwardEvent.Finish()
            finish.request_id = REQUEST_ID
            events.append(finish)
        _advance(scheduler, events)

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"[done] generated {len(generated)} tokens: {output_text!r}")
    print(f"[stat] TTFT: {ttft_seconds * 1e3:.1f} ms")
    if decode_step_seconds:
        first, steady = decode_step_seconds[0], decode_step_seconds[1:]
        print(
            f"[stat] decode: first step {first * 1e3:.1f} ms (includes one-time "
            f"decode-path JIT compile)"
        )
        if steady:
            avg = sum(steady) / len(steady)
            print(
                f"[stat] decode steady-state: {len(steady)} tokens, "
                f"avg {avg * 1e3:.1f} ms/token"
            )
    print(
        f"[stat] NPU HBM allocated: {torch.npu.memory_allocated() / 2**30:.2f} GiB, "
        f"peak: {torch.npu.max_memory_allocated() / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
