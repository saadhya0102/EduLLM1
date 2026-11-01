"""Distribution-distance metrics for predicted vs. empirical answer distributions.

All operate on two same-length probability vectors (predicted p̂, true p). Pure
Python (no numpy needed) so it runs anywhere. Aggregate = mean over items.
"""
from __future__ import annotations

import math

EPS = 1e-9


def _norm(v):
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1 / len(v)] * len(v)


def mae(p_hat, p):
    """Mean absolute error across options (interpretable: avg pct-points off)."""
    assert len(p_hat) == len(p) and p, "probability vectors must have equal nonzero length"
    return sum(abs(a - b) for a, b in zip(p_hat, p)) / len(p)


def brier(p_hat, p):
    """Mean squared error across options (the user's squared-difference)."""
    assert len(p_hat) == len(p) and p, "probability vectors must have equal nonzero length"
    return sum((a - b) ** 2 for a, b in zip(p_hat, p)) / len(p)


def tv(p_hat, p):
    """Total variation distance = 1/2 sum|p̂-p| (max prob mass misplaced)."""
    assert len(p_hat) == len(p) and p, "probability vectors must have equal nonzero length"
    return 0.5 * sum(abs(a - b) for a, b in zip(p_hat, p))


def kl(p_hat, p):
    """KL(p || p̂): how well p̂ explains the true p. Smoothed to avoid log(0)."""
    assert len(p_hat) == len(p) and p, "probability vectors must have equal nonzero length"
    ph = _norm([x + EPS for x in p_hat])
    pt = _norm([x + EPS for x in p])
    return sum(t * math.log(t / h) for t, h in zip(pt, ph))


def js(p_hat, p):
    """Jensen-Shannon divergence (symmetric, bounded)."""
    ph = _norm([x + EPS for x in p_hat])
    pt = _norm([x + EPS for x in p])
    m = [(a + b) / 2 for a, b in zip(ph, pt)]
    return 0.5 * kl(m, pt) + 0.5 * kl(m, ph)


ALL = {"MAE": mae, "Brier": brier, "TV": tv, "KL": kl, "JS": js}


def aggregate(preds, trues):
    """preds/trues: lists of prob-vectors. Returns dict of mean metric values."""
    assert len(preds) == len(trues) and preds, "need matching non-empty lists"
    out = {}
    for name, fn in ALL.items():
        out[name] = sum(fn(ph, pt) for ph, pt in zip(preds, trues)) / len(preds)
    out["n"] = len(preds)
    return out


def format_report(label, m):
    order = ["MAE", "Brier", "TV", "KL", "JS"]
    body = "  ".join(f"{k}={m[k]:.4f}" for k in order)
    return f"[{label}] n={m['n']}  {body}"
