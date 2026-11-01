"""Generate a synthetic MCQ answer-distribution dataset for developing/testing
the pipeline before the real (gated) data is available.

Each item: a templated arithmetic MCQ with a correct answer, a plausible "sticky"
distractor (a common-mistake value) that draws extra mass, and two weak
distractors. The distribution roughly mimics real student behavior (most pick
the correct answer, a chunk fall for the sticky distractor).

Run:
    python -m src.make_synthetic --n 400 --out-prefix data/synth
"""
from __future__ import annotations

import argparse
import random

from src.schema import Item, write_items


def make_item(i, rng):
    a = rng.randint(2, 40)
    b = rng.randint(2, 40)
    op = rng.choice(["+", "-", "x"])
    if op == "+":
        correct = a + b; sticky = a * b            # confuses add/multiply
    elif op == "-":
        correct = a - b; sticky = b - a            # sign/order error
    else:
        correct = a * b; sticky = a + b            # confuses multiply/add
    # option pool: correct, sticky, and two distractors
    pool = {correct, sticky, correct + rng.choice([-1, 1]) * rng.randint(1, 5),
            correct + rng.choice([-1, 1]) * rng.randint(6, 15)}
    while len(pool) < 4:
        pool.add(correct + rng.randint(-20, 20))
    opts = list(pool)[:4]
    rng.shuffle(opts)
    ci = opts.index(correct)
    si = opts.index(sticky) if sticky in opts else None

    # build a plausible student distribution
    mass = [rng.uniform(0.02, 0.08) for _ in range(4)]
    mass[ci] += rng.uniform(0.4, 0.75)                 # most get it right
    if si is not None and si != ci:
        mass[si] += rng.uniform(0.1, 0.35)             # sticky distractor draws mass
    s = sum(mass)
    dist = [m / s for m in mass]

    return Item(
        id=f"synth-{i:04d}",
        question=f"What is {a} {op} {b}?",
        options=[str(o) for o in opts],
        dist=dist,
        correct_index=ci,
        n_responses=rng.randint(50, 300),
        skill=f"arith_{op}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out-prefix", default="data/synth")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    items = [make_item(i, rng) for i in range(args.n)]
    rng.shuffle(items)
    n_test = int(round(args.n * args.test_frac))
    for it in items[:n_test]:
        it.split = "test"
    for it in items[n_test:]:
        it.split = "train"
    write_items(f"{args.out_prefix}_train.jsonl", [x for x in items if x.split == "train"])
    write_items(f"{args.out_prefix}_test.jsonl", [x for x in items if x.split == "test"])
    print(f"wrote {args.n - n_test} train / {n_test} test to {args.out_prefix}_*.jsonl")


if __name__ == "__main__":
    main()
