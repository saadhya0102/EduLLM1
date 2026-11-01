"""Human spot-check sampler (PRD §5.3 — the ONLY manual step).

Samples ~5-10% of verified ill-posed items and writes a human-readable review
file. You skim it, mark each 'y' (good) or 'n' (bad), and we compute the
generator's precision. If precision is high (say >=0.9), trust the full set; if
low, fix the generator/verifier prompts before scaling up.

This is a few hours, once — NOT verifying all 2k (which you said you don't want).

Run:
    python -m src.data.spotcheck --in data/generated_verified.jsonl --frac 0.1
    # -> edit data/spotcheck_review.txt, put y/n after each VERDICT:
    python -m src.data.spotcheck --score data/spotcheck_review.txt
"""
from __future__ import annotations

import argparse
import random

from src.schema import read_problems, DECLINE_KINDS


def make_review(inp, out, frac, seed):
    items = [p for p in read_problems(inp) if p.kind in DECLINE_KINDS]
    rng = random.Random(seed)
    rng.shuffle(items)
    n = max(1, int(round(len(items) * frac)))
    sample = items[:n]
    with open(out, "w", encoding="utf-8") as f:
        f.write("# SPOT-CHECK REVIEW\n")
        f.write("# For each item, put y (correct) or n (wrong) after 'VERDICT:'.\n")
        f.write("# 'correct' = genuinely ill-posed AND the missing hypothesis/"
                "counterexample is right.\n\n")
        for p in sample:
            f.write(f"ID: {p.id}   KIND: {p.kind}\n")
            f.write(f"STATEMENT: {p.statement}\n")
            f.write(f"MISSING HYP: {p.missing_hypothesis}\n")
            f.write(f"COUNTEREXAMPLE: {p.counterexample}\n")
            f.write("VERDICT: \n")
            f.write("-" * 70 + "\n")
    print(f"Wrote {n} items to {out}. Edit it, then run with --score {out}")


def score_review(path):
    y = n = blank = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("VERDICT:"):
                v = line.split(":", 1)[1].strip().lower()
                if v.startswith("y"):
                    y += 1
                elif v.startswith("n"):
                    n += 1
                else:
                    blank += 1
    total = y + n
    if total == 0:
        print("No verdicts filled in yet.")
        return
    prec = y / total
    print(f"Spot-check precision: {prec:.2%}  (good={y}, bad={n}, unmarked={blank})")
    if prec >= 0.9:
        print(">> Generator looks trustworthy. Scale up dataset generation.")
    elif prec >= 0.75:
        print(">> Acceptable but improve the generate/verify prompts.")
    else:
        print(">> Too many bad items. Fix the ablation prompt before scaling (risk #2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/generated_verified.jsonl")
    ap.add_argument("--out", default="data/spotcheck_review.txt")
    ap.add_argument("--frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--score", default=None, help="score a filled-in review file")
    args = ap.parse_args()
    if args.score:
        score_review(args.score)
    else:
        make_review(args.inp, args.out, args.frac, args.seed)


if __name__ == "__main__":
    main()
