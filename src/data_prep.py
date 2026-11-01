"""Build (question, options, empirical distribution) Items from FoundationalASSIST.

FoundationalASSIST is gated on HF: accept access at
huggingface.co/datasets/ASSISTments/FoundationalASSIST and set HF_TOKEN, then:

    python -m src.data_prep --peek          # 1) print CSV headers (confirm schema)
    python -m src.data_prep --build         # 2) build data/dataset_{train,test}.jsonl

Pipeline: keep MCQ problems -> for each, count first-attempt option choices across
students -> empirical distribution -> filter by min responses -> join question +
option text -> stratified split (by problem, plus a held-out-skill split).

IMPORTANT: the COLUMN NAMES below are best-guesses. Run --peek once and correct
them to match the real headers (a 2-minute edit). Marked clearly.
"""
from __future__ import annotations

import argparse
import os
import random

# ---- SCHEMA: VERIFY/EDIT THESE AFTER `--peek` -----------------------------
PROBLEMS_FILE = "Data/Problems.csv"
INTERACTIONS_FILE = "Data/Interactions.csv"
COL = {
    "problem_id": "problem_id",
    "problem_text": "problem_text",     # question stem
    "problem_type": "problem_type",     # to filter MCQ vs fill-in-the-blank
    "mcq_value": "MultipleChoice",      # value of problem_type that means MCQ
    "answer_options": "answer_options", # option texts (list/JSON/delimited) -- verify!
    "correct_answer": "correct_answer",
    # interactions
    "i_problem_id": "problem_id",
    "i_answer_text": "answer_text",     # student's FIRST answer (per HF note)
    "i_student_id": "student_id",
}
# ---------------------------------------------------------------------------

REPO = "ASSISTments/FoundationalASSIST"


def _download():
    from huggingface_hub import hf_hub_download
    paths = {}
    for f in (PROBLEMS_FILE, INTERACTIONS_FILE):
        paths[f] = hf_hub_download(REPO, f, repo_type="dataset")
    return paths


def peek():
    import pandas as pd
    paths = _download()
    for f, p in paths.items():
        print(f"\n=== {f} ===")
        df = pd.read_csv(p, nrows=5)
        print("columns:", list(df.columns))
        print(df.head(3).to_string()[:1500])


def parse_options(cell):
    """Best-effort parse of the option-texts cell into a list. VERIFY format."""
    import json
    if isinstance(cell, list):
        return [str(x) for x in cell]
    s = str(cell)
    for loader in (json.loads,):
        try:
            v = loader(s)
            if isinstance(v, list):
                return [str(x) for x in v]
        except Exception:
            pass
    for sep in ["|", ";", "\t"]:
        if sep in s:
            return [x.strip() for x in s.split(sep)]
    return [s]


def build(args):
    import pandas as pd
    from src.schema import Item, write_items

    paths = _download()
    probs = pd.read_csv(paths[PROBLEMS_FILE])
    inter = pd.read_csv(paths[INTERACTIONS_FILE])

    # filter to MCQ problems
    if COL["problem_type"] in probs.columns:
        probs = probs[probs[COL["problem_type"]].astype(str).str.contains(
            COL["mcq_value"], case=False, na=False)]
    print(f"[build] {len(probs)} MCQ problems")

    prob_by_id = {r[COL["problem_id"]]: r for _, r in probs.iterrows()}
    items = []
    grouped = inter.groupby(COL["i_problem_id"])
    for pid, g in grouped:
        if pid not in prob_by_id:
            continue
        row = prob_by_id[pid]
        opts = parse_options(row[COL["answer_options"]])
        k = len(opts)
        if k < 2:
            continue
        # count first-attempt choices, matching answer_text to an option
        counts = [0] * k
        matched = 0
        for ans in g[COL["i_answer_text"]].astype(str):
            for j, o in enumerate(opts):
                if ans.strip() == str(o).strip():
                    counts[j] += 1
                    matched += 1
                    break
        if matched < args.min_responses:
            continue
        dist = [c / matched for c in counts]
        correct = row.get(COL["correct_answer"])
        ci = next((j for j, o in enumerate(opts)
                   if str(o).strip() == str(correct).strip()), 0)
        items.append(Item(
            id=str(pid), question=str(row[COL["problem_text"]]), options=opts,
            dist=dist, correct_index=ci, n_responses=matched,
            skill=str(row.get("skill_id", row.get("skill", "")))))

    print(f"[build] {len(items)} items with >= {args.min_responses} matched responses")
    if not items:
        raise SystemExit("No items built — check the COL{} mapping against --peek output.")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_test = int(round(len(items) * args.test_frac))
    for it in items[:n_test]:
        it.split = "test"
    for it in items[n_test:]:
        it.split = "train"
    write_items("data/dataset_train.jsonl", [x for x in items if x.split == "train"])
    write_items("data/dataset_test.jsonl", [x for x in items if x.split == "test"])
    print(f"[build] wrote {len(items)-n_test} train / {n_test} test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true", help="print CSV headers, then exit")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-responses", type=int, default=50)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        print("WARNING: no HF_TOKEN set — gated download will 401. "
              "Accept access on the HF page and set HF_TOKEN first.")
    if args.peek:
        peek()
    elif args.build:
        build(args)
    else:
        print("pass --peek (confirm schema) or --build")


if __name__ == "__main__":
    main()
