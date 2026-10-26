#!/usr/bin/env python3
"""HF CPU greedy reference for Qwen3.5-0.8B (transformers 5.12.0).

Loads the full Qwen3_5ForConditionalGeneration checkpoint in fp32 on CPU and
greedy-generates for each case. Records generated token ids plus the per-step
top-1/top-2 logit values (argmax margin) so a divergence can be attributed to
a genuinely near-tied distribution vs a real numeric error.

Usage:
    python hf_reference.py --cases corr_len32:32,corr_len256:32,... --out ref.json
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.expanduser("~/models/Qwen3.5-0.8B")
PROMPTS = os.path.expanduser("~/hw1/prompts.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True,
                    help="comma list of name:max_new_tokens")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    print(f"[ref] loaded {model.config.model_type} on CPU, fp32")

    with open(PROMPTS) as f:
        prompts = json.load(f)

    result = {"model": MODEL, "dtype": "fp32", "device": "cpu",
              "transformers": __import__("transformers").__version__,
              "cases": {}}
    for item in args.cases.split(","):
        name, n = item.rsplit(":", 1)
        max_new = int(n)
        ids = prompts[name]["ids"]
        input_ids = torch.tensor([ids], dtype=torch.long)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        dt = time.perf_counter() - t0
        gen_ids = out.sequences[0, len(ids):].tolist()
        margins = []
        for score in out.scores:
            top2 = torch.topk(score[0].float(), 2)
            margins.append({"top1_id": top2.indices[0].item(),
                            "top1": round(top2.values[0].item(), 4),
                            "top2_id": top2.indices[1].item(),
                            "top2": round(top2.values[1].item(), 4)})
        result["cases"][name] = {
            "prompt_len": len(ids),
            "max_new_tokens": max_new,
            "generated_ids": gen_ids,
            "generated_text": tok.decode(gen_ids, skip_special_tokens=True),
            "margins": margins,
            "wall_seconds": round(dt, 2),
        }
        print(f"[ref] {name}: prompt={len(ids)} gen={len(gen_ids)} "
              f"in {dt:.1f}s -> {result['cases'][name]['generated_text']!r}")

    with open(args.out, "w") as f:
        json.dump(result, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
