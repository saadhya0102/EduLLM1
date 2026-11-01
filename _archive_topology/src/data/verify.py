"""Auto-verifier (PRD §5.3, guards risk #2).

For each generated ill-posed item, ask a frontier model (ideally a DIFFERENT
provider than the generator) to independently confirm:
  - the statement really is ill-posed (false or underspecified), AND
  - the named missing hypothesis is genuinely load-bearing, AND
  - there is a valid witnessing counterexample.

Items that fail verification are dropped. This is where "not actually
load-bearing" ablations (the poison in risk #2) get filtered out — WITHOUT you
hand-checking all of them. You then spot-check ~5-10% (see spotcheck.py).

Run:
    python -m src.data.verify --in data/generated_raw.jsonl \
        --out data/generated_verified.jsonl --provider openai
"""
from __future__ import annotations

import argparse

from src.llm import chat, extract_json
from src.schema import (
    Problem, read_problems, write_problems,
    KIND_OUT_OF_SCOPE, DECLINE_KINDS,
)

VERIFY_SYSTEM = (
    "You are a rigorous point-set topology referee. You independently check "
    "whether a claimed ill-posed problem is genuinely ill-posed. You are skeptical: "
    "if the weakened statement is actually still TRUE (the removed hypothesis was "
    "not load-bearing), you must reject it."
)

VERIFY_PROMPT = """A dataset item claims to be ILL-POSED. Independently verify it.

STATEMENT: {statement}
CLAIMED KIND: {kind}
CLAIMED MISSING/FALSE HYPOTHESIS: {missing_hypothesis}
CLAIMED COUNTEREXAMPLE: {counterexample}

Check carefully:
1. Is the statement, as written, genuinely FALSE or UNDERSPECIFIED (not still true)?
2. Is the named hypothesis truly load-bearing (its removal actually breaks the result)?
3. For 'false': is the counterexample valid and does it really witness the failure?
   For 'underspecified': is it genuinely impossible to determine as stated?

Return ONLY JSON:
{{
  "verified": true | false,
  "actually_wellposed": true | false,   // true if the weakened statement is STILL a valid theorem
  "corrected_missing_hypothesis": "<the correct one, if the claim was off>",
  "corrected_counterexample": "<a valid witness, if you can give a better one>",
  "notes": "<brief reasoning>"
}}"""


def verify_items(items, provider, model):
    kept, dropped = [], []
    for p in items:
        if p.kind == KIND_OUT_OF_SCOPE:
            p.verified = True
            kept.append(p)
            continue
        if p.kind not in DECLINE_KINDS:
            # answerable passthrough (already have textbook ground truth)
            p.verified = True
            kept.append(p)
            continue
        prompt = VERIFY_PROMPT.format(
            statement=p.statement, kind=p.kind,
            missing_hypothesis=p.missing_hypothesis or "",
            counterexample=p.counterexample or "",
        )
        try:
            v = extract_json(chat(prompt, provider=provider, model=model,
                                  system=VERIFY_SYSTEM, temperature=0.0))
        except Exception as e:  # noqa: BLE001
            print(f"[verify] {p.id} error: {e}; dropping")
            dropped.append(p)
            continue

        if v.get("actually_wellposed") or not v.get("verified"):
            print(f"[verify] DROP {p.id}: {v.get('notes','')[:80]}")
            dropped.append(p)
            continue

        # Accept, absorbing corrections if provided.
        if v.get("corrected_missing_hypothesis"):
            p.missing_hypothesis = v["corrected_missing_hypothesis"]
        if v.get("corrected_counterexample"):
            p.counterexample = v["corrected_counterexample"]
        # require a witnessing counterexample for 'false' items
        if p.kind == "false" and not (p.counterexample and p.counterexample.strip()):
            print(f"[verify] DROP {p.id}: no counterexample for a 'false' item")
            dropped.append(p)
            continue
        p.verified = True
        kept.append(p)
        print(f"[verify] KEEP {p.id}")
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/generated_raw.jsonl")
    ap.add_argument("--out", default="data/generated_verified.jsonl")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"],
                    help="second grader. OpenAI 404s on this gateway, so default is "
                         "anthropic; use --model to pick a DIFFERENT Claude model than "
                         "generation to keep some independence (risk #7).")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    items = read_problems(args.inp)
    kept, dropped = verify_items(items, args.provider, args.model)
    write_problems(args.out, kept)
    print(f"\nVerified {len(kept)} / {len(items)} items "
          f"({len(dropped)} dropped) -> {args.out}")


if __name__ == "__main__":
    main()
