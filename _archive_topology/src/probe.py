"""M0.5 Tier-1 probe (PRD risk #1).

Runs a frontier model over hand-made point-set-topology items — a mix of
ill-posed (false/underspecified/out-of-scope) and answerable — under the exact
eval protocol, and measures the model's False Confidence Rate (FCR): how often
it confidently *answers* an item it should have *declined*.

Go/no-go decision BEFORE building the full dataset:
  - High frontier FCR (say > ~0.3): good, there's headroom to win. Proceed.
  - Near-zero frontier FCR: point-set topology may be too easy for this thesis;
    reconsider difficulty / sub-topics before investing in the dataset.

Run:
    python -m src.probe --provider anthropic --model claude-opus-4-8
    python -m src.probe --provider openai --model gpt-4o --tau 0.5
"""
from __future__ import annotations

import argparse

from src.llm import chat
from src.prompts import SYSTEM_PROMPT, build_inference_prompt
from src.schema import (
    ModelResponse,
    parse_model_output,
    gold_decision,
    read_problems,
    DECLINE_KINDS,
)


def run_probe(path: str, provider: str, model: str | None, tau: float) -> dict:
    items = read_problems(path)
    rows = []
    for p in items:
        prompt = build_inference_prompt(p.statement)
        try:
            raw = chat(prompt, provider=provider, model=model, system=SYSTEM_PROMPT,
                       temperature=0.0)
            resp = parse_model_output(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] {p.id} failed: {e}")
            resp = ModelResponse(decision="answer", confidence=1.0, raw=str(e))
        rows.append((p, resp))
        gold = gold_decision(p.kind)
        mark = "OK " if resp.decision == gold else "XX "
        print(f"{mark}{p.id:10s} gold={gold:7s} got={resp.decision:7s} "
              f"conf={resp.confidence:.2f} kind={p.kind}")

    # Metrics
    decline_items = [(p, r) for p, r in rows if p.kind in DECLINE_KINDS]
    answerable = [(p, r) for p, r in rows if gold_decision(p.kind) == "answer"]

    # False confidence: should decline, but confidently answered.
    false_conf = [
        (p, r) for p, r in decline_items
        if r.decision == "answer" and r.confidence >= tau
    ]
    fcr = len(false_conf) / max(1, len(decline_items))

    # Over-refusal: answerable items it declined.
    over_ref = [(p, r) for p, r in answerable if r.decision == "decline"]
    over_refusal = len(over_ref) / max(1, len(answerable))

    # Refusal recall on decline items.
    correctly_declined = [(p, r) for p, r in decline_items if r.decision == "decline"]
    refusal_recall = len(correctly_declined) / max(1, len(decline_items))

    print("\n" + "=" * 60)
    print(f"Provider/model : {provider} / {model or '(default)'}")
    print(f"Items          : {len(rows)}  "
          f"(decline={len(decline_items)}, answerable={len(answerable)})")
    print(f"tau (confidence threshold) : {tau}")
    print("-" * 60)
    print(f"FALSE CONFIDENCE RATE (FCR) : {fcr:.3f}   <-- the number that matters")
    print(f"Refusal recall              : {refusal_recall:.3f}")
    print(f"Over-refusal (answerable)   : {over_refusal:.3f}")
    print("=" * 60)
    if fcr >= 0.3:
        print(">> GO: frontier FCR is high enough that there is clear headroom.")
    elif fcr >= 0.15:
        print(">> MARGINAL: some headroom; consider harder sub-topics to widen it.")
    else:
        print(">> CAUTION: frontier FCR is low. Reconsider difficulty/sub-topics")
        print("   before building the full dataset (PRD risk #1).")

    if false_conf:
        print("\nExamples of false-confident answers (target failure mode):")
        for p, r in false_conf[:5]:
            print(f"  - {p.id} ({p.kind}) conf={r.confidence:.2f}: "
                  f"{p.statement[:70]}...")

    return {"fcr": fcr, "refusal_recall": refusal_recall,
            "over_refusal": over_refusal, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/probe_items.jsonl")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--tau", type=float, default=0.5,
                    help="confidence threshold for 'confident'")
    args = ap.parse_args()
    run_probe(args.path, args.provider, args.model, args.tau)


if __name__ == "__main__":
    main()
