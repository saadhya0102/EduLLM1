"""Dataset orchestrator (PRD §5.3 + M1).

Combines the answerable seed problems with the verified generated ill-posed /
out-of-scope items, balances the mix (guards over-refusal risk #4/PRD #1 by not
letting decline items dominate), does a stratified train/test split, normalizes
surface form a little, and emits:
  - data/dataset_train.jsonl  (Problem records)
  - data/dataset_test.jsonl
  - data/train_chat.jsonl      (chat-format for SFTTrainer)

Deterministic (seeded) so runs are reproducible.

Run:
    python -m src.data.build_dataset \
        --answerable data/seed_answerable.jsonl \
        --generated  data/generated_verified.jsonl \
        --test-frac 0.2
"""
from __future__ import annotations

import argparse
import json
import random

from src.schema import (
    Problem, read_problems, write_problems,
    KIND_ANSWERABLE, gold_decision,
)
from src.prompts import build_chat_example


def stratified_split(problems, test_frac, seed):
    rng = random.Random(seed)
    by_kind = {}
    for p in problems:
        by_kind.setdefault(p.kind, []).append(p)
    train, test = [], []
    for kind, items in by_kind.items():
        items = items[:]
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * test_frac))) if len(items) > 1 else 0
        for p in items[:n_test]:
            p.split = "test"
            test.append(p)
        for p in items[n_test:]:
            p.split = "train"
            train.append(p)
    return train, test


def assign_confidence(answerable, error_mining_path):
    """Set calibrated confidence on answerable items (PRD §5.6).

    If an error-mining file is given, use each item's empirical base-model
    accuracy as its target confidence (true calibration). Otherwise assign
    difficulty tiers by solution-length tertiles (easy/medium/hard) — a
    transparent stand-in that still gives the SFT targets graded confidence.
    """
    if error_mining_path:
        with open(error_mining_path, encoding="utf-8") as f:
            acc = json.load(f)  # {id: accuracy in [0,1]}
        n = 0
        for p in answerable:
            if p.id in acc:
                # clamp away from 0/1 so declines still dominate the extremes
                p.confidence = max(0.15, min(0.97, float(acc[p.id])))
                n += 1
        print(f"[build] error-mined confidence set on {n}/{len(answerable)} answerable items")
        return

    # difficulty tiers by solution length (proxy: longer proof ~ harder ~ lower conf)
    scored = [(len(p.solution or ""), p) for p in answerable if p.confidence is None]
    if not scored:
        return
    scored.sort(key=lambda t: t[0])
    n = len(scored)
    for i, (_, p) in enumerate(scored):
        if p.difficulty:
            continue
        frac = i / max(1, n - 1)
        p.difficulty = "easy" if frac < 1/3 else ("medium" if frac < 2/3 else "hard")
    from collections import Counter
    tiers = Counter(p.difficulty for _, p in scored)
    print(f"[build] difficulty tiers (by solution length): {dict(tiers)}")


def report_balance(name, problems):
    from collections import Counter
    kinds = Counter(p.kind for p in problems)
    dec = sum(1 for p in problems if gold_decision(p.kind) == "decline")
    ans = len(problems) - dec
    print(f"  {name}: {len(problems)} items | answer={ans} decline={dec} | {dict(kinds)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable", default="data/seed_*.jsonl",
                    help="path or glob; all matching files are pooled as answerable seeds")
    ap.add_argument("--generated", default="data/generated_verified.jsonl")
    ap.add_argument("--out-train", default="data/dataset_train.jsonl")
    ap.add_argument("--out-test", default="data/dataset_test.jsonl")
    ap.add_argument("--out-chat", default="data/train_chat.jsonl")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-decline-ratio", type=float, default=0.6,
                    help="cap decline fraction to avoid over-refusal (risk #4)")
    ap.add_argument("--error-mining", default=None,
                    help="optional data/error_mining.json {id: accuracy}; sets "
                         "calibrated confidence on answerable items (PRD §5.6)")
    args = ap.parse_args()

    import glob as _glob
    seed_files = sorted(_glob.glob(args.answerable)) or [args.answerable]
    answerable = []
    seen_ids = set()
    for sf in seed_files:
        for p in read_problems(sf):
            if p.kind == KIND_ANSWERABLE and p.id not in seen_ids:
                seen_ids.add(p.id)
                answerable.append(p)
    print(f"[build] pooled {len(answerable)} answerable seeds from {len(seed_files)} file(s): "
          f"{[s.split('/')[-1].split(chr(92))[-1] for s in seed_files]}")
    generated = read_problems(args.generated)

    decline = [p for p in generated if gold_decision(p.kind) == "decline"]
    # also fold any answerable that snuck into 'generated'
    answerable += [p for p in generated if p.kind == KIND_ANSWERABLE]

    # Balance: cap decline share.
    rng = random.Random(args.seed)
    rng.shuffle(decline)
    max_decline = int(len(answerable) * args.max_decline_ratio / (1 - args.max_decline_ratio))
    if len(decline) > max_decline and max_decline > 0:
        print(f"[build] capping decline items {len(decline)} -> {max_decline} "
              f"(max-decline-ratio={args.max_decline_ratio})")
        decline = decline[:max_decline]

    # Confidence calibration (PRD §5.6). Prefer error-mined empirical accuracy;
    # otherwise assign difficulty tiers by solution-length tertiles so the SFT
    # targets carry a graded (not constant) confidence.
    assign_confidence(answerable, args.error_mining)

    combined = answerable + decline
    print(f"[build] combined pool: {len(combined)} items")
    report_balance("pool", combined)

    train, test = stratified_split(combined, args.test_frac, args.seed)
    print("[build] split:")
    report_balance("train", train)
    report_balance("test", test)

    write_problems(args.out_train, train)
    write_problems(args.out_test, test)

    with open(args.out_chat, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(build_chat_example(p), ensure_ascii=False) + "\n")

    print(f"\n[build] wrote:\n  {args.out_train}\n  {args.out_test}\n  {args.out_chat}")
    if len(train) < 200:
        print("\n[build] NOTE: small dataset. Increase --n-per in generate.py and add "
              "more answerable seeds to reach the ~2k target (PRD).")


if __name__ == "__main__":
    main()
