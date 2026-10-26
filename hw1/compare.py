#!/usr/bin/env python3
"""Compare tokenspeed output vs HF reference, token by token.

Usage: compare.py --ref ref.json --ts ts_out.json [--map case:case ...]
Reports per case: match length, first divergence index, HF argmax margin at
divergence (near-tie detection), and overall verdict.
"""
import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ts", required=True)
    ap.add_argument("--map", nargs="*", default=[],
                    help="ts_case=ref_case overrides")
    args = ap.parse_args()

    with open(args.ref) as f:
        ref = json.load(f)["cases"]
    with open(args.ts) as f:
        ts = json.load(f)["cases"]
    mapping = dict(m.split("=") for m in args.map)

    print(f"{'case':<16} {'prompt':>6} {'ref_n':>5} {'ts_n':>5} "
          f"{'match':>6} {'first_div':>9}  verdict")
    summary = []
    for ts_name, ts_case in ts.items():
        ref_name = mapping.get(ts_name, ts_name)
        if ref_name not in ref:
            continue
        r = ref[ref_name]
        ref_ids = r["generated_ids"]
        for rep_i, rep in enumerate(ts_case["reps"]):
            ts_ids = rep["generated_ids"]
            n = min(len(ref_ids), len(ts_ids))
            match = next((i for i in range(n) if ref_ids[i] != ts_ids[i]), n)
            div = match if match < n else -1
            note = ""
            if div >= 0:
                m = r["margins"][div]
                gap = m["top1"] - m["top2"]
                note = (f" | div@{div}: ref {ref_ids[div]} vs ts {ts_ids[div]}, "
                        f"HF margin {gap:.4f} (top1={m['top1']} top2={m['top2']})"
                        + ("  NEAR-TIE" if gap < 0.1 else ""))
            full = div < 0 and len(ref_ids) == len(ts_ids)
            prefix = f"{ts_name}[r{rep_i}]"
            print(f"{prefix:<16} {ts_case['prompt_len']:>6} {len(ref_ids):>5} "
                  f"{len(ts_ids):>5} {match:>6} {div:>9}  "
                  f"{'EXACT' if full else ('PREFIX-MATCH(len differs)' if div < 0 else 'DIVERGE')}{note}")
            summary.append({"case": ts_name, "rep": rep_i, "match": match,
                            "first_div": div, "exact": full})
    n_exact = sum(1 for s in summary if s["exact"])
    print(f"\n{n_exact}/{len(summary)} runs exactly match the HF reference")


if __name__ == "__main__":
    main()
