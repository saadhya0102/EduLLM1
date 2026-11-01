"""Build (question, options, empirical distribution) Items from DBE-KT22.

DBE-KT22 is a public knowledge-tracing dataset (ANU Relational Databases course).
Unlike FoundationalASSIST it is NOT HF-gated, but the ADA Dataverse sits behind an
anti-bot challenge, so it must be downloaded by hand:

    1. Open https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH
    2. Access Dataset -> Download ZIP  (public, no login/approval needed)
    3. Unzip the CSVs into  data/dbe_raw/

Then:

    python -m src.data_prep_dbe --peek     # 1) print every CSV's headers + sample rows
    python -m src.data_prep_dbe --build    # 2) build data/dbe_{train,test}.jsonl

Pipeline: questions table (id + text) x choices table (id + text + is_correct)
x transactions table (student picked which choice) -> per-question choice counts
-> empirical distribution -> filter by min responses -> stratified split.

IMPORTANT: DBE-KT22 stores choices in a SEPARATE table (questions have a variable
number of options). The COL / FILE guesses below are best-effort; run --peek once
and correct any that don't match the real headers (a 2-minute edit). Marked clearly.
"""
from __future__ import annotations

import argparse
import glob
import os
import random

RAW_DIR = "data/dbe_raw"

# ---- FILES / COLUMNS: VERIFY/EDIT AFTER `--peek` --------------------------
# Filenames are matched case-insensitively by substring, so "Questions.csv",
# "questions.csv" etc. all work. Column names likewise (see _col()).
FILES = {
    "questions":    "question",       # question stems         (e.g. Questions.csv)
    "choices":      "choice",         # per-option choice rows  (e.g. Question_Choices.csv)
    "transactions": "transaction",    # student attempts        (e.g. Transaction.csv)
}
COL = {
    # questions table
    "q_id":        ["id", "question_id"],
    "q_text":      ["question_text", "question_rich_text", "text", "question"],
    # choices table (one row per option)
    "c_id":        ["id", "choice_id"],
    "c_qid":       ["question_id", "q_id"],
    "c_text":      ["choice_text", "text", "choice"],
    "c_correct":   ["is_correct", "correct", "iscorrect"],
    # transactions table (one row per student answer)
    "t_qid":       ["question_id", "q_id"],
    "t_student":   ["student_id", "user_id", "learner_id"],
    "t_choice":    ["answer_choice_id", "selected_choice_id", "choice_id",
                    "answer_id", "answer_state", "answer"],
}
# ---------------------------------------------------------------------------


def _find_file(kind: str):
    key = FILES[kind].lower()
    hits = [p for p in glob.glob(os.path.join(RAW_DIR, "*.csv"))
            if key in os.path.basename(p).lower()]
    if not hits:
        raise SystemExit(f"no CSV matching '{key}' in {RAW_DIR}/ (run --peek to see files)")
    # prefer the shortest name (avoids e.g. question_kc_relationships for 'question')
    return min(hits, key=lambda p: len(os.path.basename(p)))


def _col(df, candidates, where):
    """Return the first df column matching any candidate (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    raise SystemExit(f"none of {candidates} found in {where}; columns={list(df.columns)}")


def peek():
    import pandas as pd
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"no CSVs in {RAW_DIR}/ — download DBE-KT22 there first (see module docstring)")
    for p in files:
        print(f"\n=== {os.path.basename(p)} ===")
        df = pd.read_csv(p, nrows=5)
        print("columns:", list(df.columns))
        print(df.head(3).to_string()[:1500])


def build(args):
    import pandas as pd
    from src.schema import Item, write_items

    q = pd.read_csv(_find_file("questions"))
    c = pd.read_csv(_find_file("choices"))
    t = pd.read_csv(_find_file("transactions"))

    q_id, q_text = _col(q, COL["q_id"], "questions"), _col(q, COL["q_text"], "questions")
    q_title = _col(q, ["question_title", "title"], "questions")
    c_id, c_qid = _col(c, COL["c_id"], "choices"), _col(c, COL["c_qid"], "choices")
    c_text, c_ok = _col(c, COL["c_text"], "choices"), _col(c, COL["c_correct"], "choices")
    t_qid = _col(t, COL["t_qid"], "transactions")
    t_student = _col(t, COL["t_student"], "transactions")
    t_choice = _col(t, COL["t_choice"], "transactions")
    t_time = _col(t, ["start_time", "end_time"], "transactions")

    # question text, falling back to the title when the stem is blank
    def _qtext(row):
        for col in (q_text, q_title):
            v = row[col]
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    q_text_by_id = {row[q_id]: _qtext(row) for _, row in q.iterrows()}

    # choices grouped per question, in stable order -> option list + correct index
    choices_by_q: dict = {}
    for _, row in c.iterrows():
        choices_by_q.setdefault(row[c_qid], []).append(
            (row[c_id], str(row[c_text]), bool(row[c_ok])))

    # one row per (student, question): their FIRST attempt (earliest start_time),
    # so repeat practice doesn't double-count in the distribution.
    tt = t[[t_qid, t_student, t_choice, t_time]].dropna(subset=[t_choice])
    tt = tt.sort_values(t_time).drop_duplicates(subset=[t_student, t_qid], keep="first")
    tt[t_choice] = tt[t_choice].astype(int)
    print(f"[build] {len(tt)} first-attempt responses over {tt[t_qid].nunique()} questions")

    items = []
    picks = tt.groupby(t_qid)[t_choice]
    for qid, chosen in picks:
        if qid not in choices_by_q or qid not in q_text_by_id:
            continue
        opts = choices_by_q[qid]                       # [(choice_id, text, is_correct), ...]
        k = len(opts)
        if k < 2:
            continue
        idx_of = {cid: j for j, (cid, _, _) in enumerate(opts)}
        counts = [0] * k
        matched = 0
        for ch in chosen:
            if ch in idx_of:
                counts[idx_of[ch]] += 1
                matched += 1
        if matched < args.min_responses:
            continue
        dist = [n / matched for n in counts]
        ci = next((j for j, (_, _, ok) in enumerate(opts) if ok), 0)
        items.append(Item(
            id=str(qid), question=str(q_text_by_id[qid]),
            options=[txt for _, txt, _ in opts], dist=dist,
            correct_index=ci, n_responses=matched, skill="dbe"))

    print(f"[build] {len(items)} items with >= {args.min_responses} matched responses")
    if args.only_k:
        items = [it for it in items if it.k() == args.only_k]
        print(f"[build] {len(items)} items after keeping only k={args.only_k} options")
    if not items:
        raise SystemExit("No items built — check FILES/COL against --peek output.")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    n_test = int(round(len(items) * args.test_frac))
    for it in items[:n_test]:
        it.split = "test"
    for it in items[n_test:]:
        it.split = "train"
    write_items("data/dbe_train.jsonl", [x for x in items if x.split == "train"])
    write_items("data/dbe_test.jsonl", [x for x in items if x.split == "test"])
    print(f"[build] wrote {len(items)-n_test} train / {n_test} test to data/dbe_*.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true", help="print CSV headers, then exit")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--min-responses", type=int, default=30)
    ap.add_argument("--only-k", type=int, default=4,
                    help="keep only items with exactly this many options (0 = keep all)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    if args.peek:
        peek()
    elif args.build:
        build(args)
    else:
        print("pass --peek (confirm schema) or --build")


if __name__ == "__main__":
    main()
