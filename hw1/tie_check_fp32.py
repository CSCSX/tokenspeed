#!/usr/bin/env python3
"""Teacher-forced fp32 check at the two bf16 divergence contexts.

Feeds prompt + bf16-reference prefix (up to the divergence step) into the
fp32 model and compares the logits of the two tied candidates.
"""
import json
import os

import torch
from transformers import AutoModelForCausalLM

MODEL = os.path.expanduser("~/models/Qwen3.5-0.8B")

prompts = json.load(open(os.path.expanduser("~/hw1/prompts.json")))
ref = json.load(open(os.path.expanduser("~/hw1/ref.json")))["cases"]
ts = json.load(open(os.path.expanduser("~/hw1/ts_corr.json")))["cases"]

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

for case, step in (("corr_len32", 13), ("iso_B", 6)):
    prefix = prompts[case]["ids"] + ref[case]["generated_ids"][:step]
    ids = torch.tensor([prefix])
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    cand = [ref[case]["generated_ids"][step], ts[case]["reps"][0]["generated_ids"][step]]
    vals = {t: logits[t].item() for t in cand}
    top = logits.argmax().item()
    print(f"{case}@{step}: candidates {vals} fp32_argmax={top} "
          f"(logit={logits[top].item():.4f}) gap={abs(vals[cand[0]]-vals[cand[1]]):.5f}")
