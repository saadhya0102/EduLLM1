"""Paraphrase answerable TEST statements to reduce contamination (PRD risk #3).

Textbook problems and their solutions are almost certainly in Qwen's pretraining,
so a verbatim test item may be memorized — inflating measured accuracy. This
rewrites each answerable test statement into an equivalent but novel phrasing
(same problem, same reference solution), lowering the chance the model has seen
the exact string.

Note: contamination inflates the baseline and the fine-tuned model roughly
equally, so it mostly affects absolute accuracy, not the selective-prediction
COMPARISON (our actual win condition). The ill-posed test items are already
novel (ablation-generated). Treat this as a rigor pass, not a blocker.

Run (after build_dataset):
    python -m src.data.paraphrase --in data/dataset_test.jsonl \
        --out data/dataset_test.jsonl --provider anthropic
"""
from __future__ import annotations

import argparse

from src.llm import chat
from src.schema import read_problems, write_problems, KIND_ANSWERABLE

PARA_SYSTEM = (
    "You rephrase point-set topology problems into fresh, equivalent wording for "
    "a held-out test set. You preserve the exact mathematical content, hypotheses, "
    "and the correct answer; you only change surface phrasing and notation."
)

PARA_PROMPT = """Rewrite this problem statement in different words while keeping the
mathematics exactly the same (same hypotheses, same conclusion, same difficulty).
Do NOT solve it. Do NOT weaken or strengthen any hypothesis. Return ONLY the
rewritten statement, nothing else.

STATEMENT: {statement}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/dataset_test.jsonl")
    ap.add_argument("--out", default="data/dataset_test.jsonl")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    items = read_problems(args.inp)
    n = 0
    for p in items:
        # only answerable items risk solution-memorization; declines are generated
        if p.kind != KIND_ANSWERABLE:
            continue
        try:
            new_stmt = chat(PARA_PROMPT.format(statement=p.statement),
                            provider=args.provider, model=args.model,
                            system=PARA_SYSTEM, temperature=0.7).strip()
            if new_stmt and len(new_stmt) > 15:
                p.statement = new_stmt
                n += 1
        except Exception as e:  # noqa: BLE001
            print(f"[paraphrase] {p.id} failed: {e}")

    write_problems(args.out, items)
    print(f"[paraphrase] rewrote {n} answerable test statements -> {args.out}")
    print("[paraphrase] reference solutions unchanged; the judge still grades "
          "the model's answer against them.")


if __name__ == "__main__":
    main()
