#!/usr/bin/env python3
"""Build the prompt set for hw1 verification/perf runs.

Every prompt is stored as exact token ids so the HF reference and the
tokenspeed driver consume byte-identical inputs. Long prompts are a repeated
paragraph truncated at the token level to the exact target length.

Output: ~/hw1/prompts.json  {name: {"ids": [...], "note": str}}
"""
import json
import os

from transformers import AutoTokenizer

MODEL = os.path.expanduser("~/models/Qwen3.5-0.8B")

PARA = (
    "The history of computing spans mechanical calculators, vacuum tube "
    "machines, transistorized mainframes, and modern silicon accelerators. "
    "Each generation changed not only the speed of arithmetic but the way "
    "people formulate problems, organize data, and reason about machines. "
)

CASES = {
    # correctness: 4 input-length tiers, greedy, >=16 generated tokens
    "corr_len32": 32,
    "corr_len256": 256,
    "corr_len1024": 1024,
    "corr_len4096": 4096,
    # multi-step state update check
    "corr_multistep": 64,
    # isolation pair: two semantically different prompts
    "iso_A": None,
    "iso_B": None,
    # perf tiers
    "perf_short": 128,
    "perf_long": 2048,
}

FIXED = {
    "iso_A": "The capital of France is",
    "iso_B": "Write a haiku about autumn leaves falling on a quiet pond.",
}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = {}
    for name, target in CASES.items():
        if name in FIXED:
            ids = tok.encode(FIXED[name])
            note = f"fixed text: {FIXED[name]!r}"
        else:
            text = PARA
            while len(tok.encode(text)) < target:
                text += PARA
            ids = tok.encode(text)[:target]
            note = f"repeated paragraph truncated to {target} tokens"
        prompts[name] = {"ids": ids, "note": note}
        print(f"{name}: {len(ids)} tokens  ({note})")
    out = os.path.expanduser("~/hw1/prompts.json")
    with open(out, "w") as f:
        json.dump(prompts, f)
    print("wrote", out)


if __name__ == "__main__":
    main()
