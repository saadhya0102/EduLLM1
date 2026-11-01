"""Selective-prediction metrics (PRD §5.5).

Given per-item (gold_decision, model_decision, confidence, answer_correct) rows,
compute the full metric table + risk-coverage curve + bootstrap confidence
intervals (small-sample noise, risk #10). No sklearn required; pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import random


@dataclass
class EvalRow:
    id: str
    gold_decision: str        # "answer" | "decline"
    model_decision: str       # "answer" | "decline"
    confidence: float
    answer_correct: Optional[bool] = None  # only meaningful for answerable+answered
    kind: str = ""


def _rate(num, den):
    return num / den if den else float("nan")


def compute_metrics(rows: list[EvalRow], tau: float = 0.5) -> dict:
    should_decline = [r for r in rows if r.gold_decision == "decline"]
    answerable = [r for r in rows if r.gold_decision == "answer"]

    # False Confidence Rate: should decline but confidently answered.
    false_conf = [r for r in should_decline
                  if r.model_decision == "answer" and r.confidence >= tau]
    fcr = _rate(len(false_conf), len(should_decline))

    # Over-refusal: answerable but declined.
    over = [r for r in answerable if r.model_decision == "decline"]
    over_refusal = _rate(len(over), len(answerable))

    # Answer accuracy on the answerable bin, among ATTEMPTED (answered).
    attempted = [r for r in answerable if r.model_decision == "answer"]
    correct = [r for r in attempted if r.answer_correct]
    answer_acc = _rate(len(correct), len(attempted))

    # Refusal precision / recall (positive class = "decline").
    tp = len([r for r in should_decline if r.model_decision == "decline"])
    fp = len([r for r in answerable if r.model_decision == "decline"])
    fn = len([r for r in should_decline if r.model_decision == "answer"])
    refusal_precision = _rate(tp, tp + fp)
    refusal_recall = _rate(tp, tp + fn)

    # Coverage = fraction answered.
    answered = [r for r in rows if r.model_decision == "answer"]
    coverage = _rate(len(answered), len(rows))

    # Risk = error rate among answered. An answered item is "wrong" if it was a
    # decline-gold item (should not have answered) OR an answerable it got wrong.
    def answered_wrong(r):
        if r.gold_decision == "decline":
            return True
        return r.answer_correct is False
    risk = _rate(sum(1 for r in answered if answered_wrong(r)), len(answered))

    aurc = area_under_risk_coverage(rows)
    ece = expected_calibration_error(rows)

    return {
        "n": len(rows),
        "tau": tau,
        "FCR": fcr,
        "answer_accuracy": answer_acc,
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "coverage": coverage,
        "risk": risk,
        "AURC": aurc,
        "ECE": ece,
        "over_refusal": over_refusal,
    }


def risk_coverage_curve(rows: list[EvalRow]):
    """Sort by confidence desc; sweep coverage, compute risk at each level.

    An item counts as answered when we 'accept' it (confidence above the sweep
    threshold). Wrong = decline-gold accepted, or answerable accepted-but-wrong.
    Returns (coverages, risks).
    """
    ranked = sorted(rows, key=lambda r: r.confidence, reverse=True)

    def wrong(r):
        if r.gold_decision == "decline":
            return True
        return r.answer_correct is False

    covs, risks = [], []
    errs = 0
    for i, r in enumerate(ranked, start=1):
        if wrong(r):
            errs += 1
        covs.append(i / len(ranked))
        risks.append(errs / i)
    return covs, risks


def area_under_risk_coverage(rows: list[EvalRow]) -> float:
    if not rows:
        return float("nan")
    covs, risks = risk_coverage_curve(rows)
    # trapezoidal integral of risk over coverage
    area = 0.0
    for i in range(1, len(covs)):
        area += (covs[i] - covs[i - 1]) * (risks[i] + risks[i - 1]) / 2
    return area


def expected_calibration_error(rows: list[EvalRow], n_bins: int = 10) -> float:
    """ECE over items the model chose to ANSWER: is stated confidence calibrated
    against actual correctness? For answered decline-gold items, correctness=0."""
    answered = [r for r in rows if r.model_decision == "answer"]
    if not answered:
        return float("nan")

    def correct(r):
        if r.gold_decision == "decline":
            return 0.0  # answering a decline item is wrong
        return 1.0 if r.answer_correct else 0.0

    bins = [[] for _ in range(n_bins)]
    for r in answered:
        b = min(n_bins - 1, int(r.confidence * n_bins))
        bins[b].append(r)
    ece = 0.0
    total = len(answered)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(r.confidence for r in b) / len(b)
        acc = sum(correct(r) for r in b) / len(b)
        ece += (len(b) / total) * abs(avg_conf - acc)
    return ece


def bootstrap_ci(rows: list[EvalRow], metric_key: str, tau: float = 0.5,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap CI for a single metric (risk #10: small-sample noise)."""
    rng = random.Random(seed)
    vals = []
    n = len(rows)
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        v = compute_metrics(sample, tau=tau).get(metric_key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int((alpha / 2) * len(vals))]
    hi = vals[int((1 - alpha / 2) * len(vals)) - 1]
    return (lo, hi)


def format_report(name: str, metrics: dict, cis: dict | None = None) -> str:
    lines = [f"\n=== {name} (n={metrics['n']}, tau={metrics['tau']}) ==="]
    order = ["FCR", "answer_accuracy", "refusal_precision", "refusal_recall",
             "coverage", "risk", "AURC", "ECE", "over_refusal"]
    for k in order:
        v = metrics.get(k, float("nan"))
        ci = ""
        if cis and k in cis:
            ci = f"  [95% CI {cis[k][0]:.3f}, {cis[k][1]:.3f}]"
        lines.append(f"  {k:20s}: {v:.3f}{ci}")
    return "\n".join(lines)
