"""Compare two eval runs (baseline vs fine-tuned) and plot risk-coverage curves.

Run:
    python -m src.eval.compare --baseline data/eval_baseline.json \
        --model data/eval_finetuned.json --out data/risk_coverage.png
"""
from __future__ import annotations

import argparse
import json

from src.eval.metrics import EvalRow, risk_coverage_curve


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["label"], d["metrics"], [EvalRow(**r) for r in d["rows"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="data/risk_coverage.png")
    args = ap.parse_args()

    bl, blm, blrows = load_rows(args.baseline)
    ft, ftm, ftrows = load_rows(args.model)

    print(f"{'metric':20s} {'baseline':>12s} {'fine-tuned':>12s} {'delta':>10s}")
    for k in ["FCR", "answer_accuracy", "coverage", "risk", "AURC", "ECE",
              "over_refusal", "refusal_recall"]:
        b, f = blm.get(k, float('nan')), ftm.get(k, float('nan'))
        print(f"{k:20s} {b:12.3f} {f:12.3f} {f-b:+10.3f}")

    win = (ftm["FCR"] < blm["FCR"]
           and ftm["answer_accuracy"] >= blm["answer_accuracy"] - 0.02
           and ftm["coverage"] >= 0.5 * blm["coverage"])
    print("\n>> " + ("WIN: lower FCR at matched accuracy + acceptable coverage."
                     if win else
                     "NOT YET a clean win — check the joint criterion (PRD §5.5)."))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for label, rows in [(bl, blrows), (ft, ftrows)]:
            covs, risks = risk_coverage_curve(rows)
            plt.plot(covs, risks, label=label)
        plt.xlabel("Coverage")
        plt.ylabel("Risk (error rate among answered)")
        plt.title("Risk–Coverage: baseline vs fine-tuned")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"[compare] wrote {args.out}")
    except ImportError:
        print("[compare] matplotlib not installed; skipped plot.")


if __name__ == "__main__":
    main()
