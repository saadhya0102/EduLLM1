"""Greedy ensemble selection over the trained adapters.

Averaging several models' predicted distributions often beats any single one.
We use Caruana-style greedy selection: start empty, repeatedly add the model
(with replacement) that most improves VALIDATION MAE of the averaged prediction,
stop when nothing helps. Membership is chosen on val only, then reported on test.

    python -m src.ensemble --val data/dbe_val.jsonl --test data/dbe_test.jsonl \
        --sweep-dir out/sweep --frontier-json results/results.json

Writes results/report_ensemble.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from src.schema import read_items
from src import metrics


def _base_of(ad):
    cfg = os.path.join(ad, "adapter_config.json")
    return json.load(open(cfg))["base_model_name_or_path"] if os.path.isfile(cfg) else None


def _avg(preds, keys, n_items):
    """Average selected models' predictions (keys may repeat = weighting)."""
    out = []
    for i in range(n_items):
        k_opts = len(preds[keys[0]][i])
        out.append([sum(preds[k][i][j] for k in keys) / len(keys) for j in range(k_opts)])
    return out


def _mae(P, T):
    return sum(metrics.mae(p, t) for p, t in zip(P, T)) / len(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/dbe_val.jsonl")
    ap.add_argument("--test", default="data/dbe_test.jsonl")
    ap.add_argument("--sweep-dir", default="out/sweep")
    ap.add_argument("--frontier-json", default="results/results.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-steps", type=int, default=12)
    args = ap.parse_args()

    val = read_items(args.val); test = read_items(args.test)
    vt = [it.normalized() for it in val]; tt = [it.normalized() for it in test]

    from src.predict_hf import HFPredictor
    adapters = sorted(d for d in glob.glob(os.path.join(args.sweep_dir, "*"))
                      if os.path.isfile(os.path.join(d, "adapter_model.safetensors")))

    pv, pt = {}, {}   # label -> per-item predictions on val / test
    for ad in adapters:
        label = os.path.basename(ad); base = _base_of(ad)
        print(f"[{label}] predicting val/test ...", flush=True)
        pred = HFPredictor(base=base, adapter=ad, load_4bit=("7b" in (base or "").lower()))
        pv[label] = [pred(it) for it in val]
        pt[label] = [pred(it) for it in test]
        del pred

    labels = list(pv)
    single = {k: _mae(pv[k], vt) for k in labels}
    print("\nval MAE per model:", {k: round(v, 4) for k, v in single.items()})

    # greedy selection with replacement
    chosen, best = [], float("inf")
    for _ in range(args.max_steps):
        cand_k, cand_m = None, float("inf")
        for k in labels:
            m = _mae(_avg(pv, chosen + [k], len(val)), vt)
            if m < cand_m:
                cand_m, cand_k = m, k
        if chosen and cand_m >= best - 1e-6:
            break
        chosen.append(cand_k); best = cand_m
    from collections import Counter
    weights = Counter(chosen)
    print(f"\nensemble (by val MAE {best:.4f}): {dict(weights)}")

    ens_test = metrics.aggregate(_avg(pt, chosen, len(test)), tt)

    # combine with frontier for a final board
    board = []
    if os.path.isfile(args.frontier_json):
        fj = json.load(open(args.frontier_json))
        for name in ("uniform", "mean", "sonnet", "haiku", "opus"):
            if name in fj["results"]:
                board.append((name, fj["results"][name]))
    for k in labels:
        board.append((k, metrics.aggregate(pt[k], tt)))
    board.append(("ENSEMBLE", ens_test))
    board.sort(key=lambda x: x[1]["MAE"])

    order = ["MAE", "Brier", "TV", "KL", "JS"]
    lines = ["# Ensemble results — test set (DBE-KT22)",
             f"\nGreedy ensemble (chosen on val): **{dict(weights)}**, val MAE {best:.4f}\n",
             "| Rank | Predictor | " + " | ".join(order) + " |",
             "|---|" + "---|" * len(order)]
    for rank, (name, m) in enumerate(board, 1):
        lines.append(f"| {rank} | {name} | " + " | ".join(f"{m[k]:.4f}" for k in order) + " |")
    report = "\n".join(lines) + "\n"

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "report_ensemble.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"saved -> {args.out_dir}/report_ensemble.md")


if __name__ == "__main__":
    main()
