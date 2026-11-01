"""Tune post-hoc temperature on a validation set for each trained adapter, pick
the best config by validation MAE, then report all configs on the held-out test
set alongside the frontier baselines.

Temperature scaling: predicted dist = softmax(letter_logits / T). A single scalar
T (fit on val, never on test) corrects systematic over/under-confidence — exactly
the calibration the thesis is about. Grid-searched, so no leakage into test.

    python -m src.tune --val data/dbe_val.jsonl --test data/dbe_test.jsonl \
        --sweep-dir out/sweep --frontier-json results/results.json

Writes results/results_tuned.json and results/report_tuned.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

from src.schema import read_items
from src import metrics
from src.progress import Progress


def _softmax(logits, T):
    m = max(logits)
    ex = [math.exp((x - m) / T) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]


def _mae_at_T(logits_list, trues, T):
    preds = [_softmax(lg, T) for lg in logits_list]
    return sum(metrics.mae(p, t) for p, t in zip(preds, trues)) / len(preds)


def _base_of(adapter_dir):
    cfg = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            return json.load(f).get("base_model_name_or_path")
    return None


def _grid_best_T(logits_list, trues, grid):
    best_T, best_mae = 1.0, float("inf")
    for T in grid:
        m = _mae_at_T(logits_list, trues, T)
        if m < best_mae:
            best_T, best_mae = T, m
    return best_T, best_mae


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/dbe_val.jsonl")
    ap.add_argument("--test", default="data/dbe_test.jsonl")
    ap.add_argument("--sweep-dir", default="out/sweep")
    ap.add_argument("--frontier-json", default="results/results.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dataset-name", default="DBE-KT22")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-len", type=int, default=0)
    args = ap.parse_args()

    val = read_items(args.val)
    test = read_items(args.test)
    val_true = [it.normalized() for it in val]
    test_true = [it.normalized() for it in test]
    grid = [round(0.5 + 0.05 * i, 2) for i in range(0, 91)]   # 0.50 .. 5.00

    from src.predict_hf import HFPredictor

    adapters = sorted(d for d in glob.glob(os.path.join(args.sweep_dir, "*"))
                      if os.path.isfile(os.path.join(d, "adapter_model.safetensors")))
    print(f"found {len(adapters)} trained adapters\n")
    score_progress = Progress(
        "local model scoring", len(adapters) * (len(val) + len(test)),
        unit="items", min_interval=10)
    scored = 0

    rows = {}   # label -> {val_mae_T1, val_mae_bestT, bestT, test metrics @T1 and @bestT}
    for ad in adapters:
        label = os.path.basename(ad)
        base = _base_of(ad)
        print(f"[{label}] base={base} — loading & scoring val/test ...", flush=True)
        pred = HFPredictor(base=base, adapter=ad,
                           load_4bit=(args.load_4bit or
                                      "7b" in (base or "").lower()),
                           max_len=(args.max_len or None))
        val_logits = []
        for item in val:
            val_logits.append(pred.raw_letter_logits(item))
            scored += 1
            score_progress.update(scored, part=f"{label}: validation {item.id}")
        test_logits = []
        for item in test:
            test_logits.append(pred.raw_letter_logits(item))
            scored += 1
            score_progress.update(scored, part=f"{label}: test {item.id}")
        del pred  # free GPU before next adapter

        val_mae_T1 = _mae_at_T(val_logits, val_true, 1.0)
        bestT, val_mae_bestT = _grid_best_T(val_logits, val_true, grid)

        def _metrics_at(T):
            preds = [_softmax(lg, T) for lg in test_logits]
            return metrics.aggregate(preds, test_true)

        rows[label] = {
            "base": base, "bestT": bestT,
            "val_mae_T1": val_mae_T1, "val_mae_bestT": val_mae_bestT,
            "test_T1": _metrics_at(1.0), "test_bestT": _metrics_at(bestT),
        }
        m1, mb = rows[label]["test_T1"], rows[label]["test_bestT"]
        print(f"    valMAE {val_mae_T1:.4f}->{val_mae_bestT:.4f} (T={bestT})  "
              f"testMAE {m1['MAE']:.4f}->{mb['MAE']:.4f}\n", flush=True)

    if not rows:
        raise SystemExit("no adapters found — did the sweep finish?")

    # pick best config by VALIDATION MAE at its best T (no test leakage)
    winner = min(rows, key=lambda k: rows[k]["val_mae_bestT"])

    # pull frontier + baseline test metrics (already computed on this test set)
    frontier = {}
    if os.path.isfile(args.frontier_json):
        with open(args.frontier_json) as f:
            fj = json.load(f)
        for name, result in fj.get("results", {}).items():
            frontier[name] = (result, fj.get("models", {}).get(name, "—"))

    # combined test leaderboard (tuned models use their best-T metrics)
    board = []
    for name, (m, model) in frontier.items():
        board.append((name, model, m))
    for label, r in rows.items():
        tag = f"{label} (T={r['bestT']})"
        board.append((tag, r["base"], r["test_bestT"]))
    board.sort(key=lambda x: x[2]["MAE"])

    order = ["MAE", "Brier", "TV", "KL", "JS"]
    base_mae = frontier["mean"][0]["MAE"] if "mean" in frontier else None
    lines = [f"# Tuned results — test set ({args.dataset_name})",
             f"\nWinner config (by validation MAE): **{winner}**  "
             f"(base {rows[winner]['base']}, T={rows[winner]['bestT']})",
             "\nTuned small models scored at their val-selected temperature. "
             "Lower is better.\n",
             "| Rank | Predictor | Model | " + " | ".join(order) + " | ΔMAE vs mean |",
             "|---|---|---|" + "---|" * len(order) + "---|"]
    for rank, (name, model, m) in enumerate(board, 1):
        d = "" if base_mae is None else f"{m['MAE'] - base_mae:+.4f}"
        lines.append(f"| {rank} | {name} | {model} | "
                     + " | ".join(f"{m[k]:.4f}" for k in order) + f" | {d} |")

    # temperature effect table
    lines += ["\n## Temperature effect (test MAE: T=1 → best T)", "",
              "| Config | base | T | val MAE (T1→best) | test MAE (T1→best) |",
              "|---|---|---|---|---|"]
    for label, r in sorted(rows.items()):
        lines.append(f"| {label} | {r['base']} | {r['bestT']} | "
                     f"{r['val_mae_T1']:.4f}→{r['val_mae_bestT']:.4f} | "
                     f"{r['test_T1']['MAE']:.4f}→{r['test_bestT']['MAE']:.4f} |")

    report = "\n".join(lines) + "\n"
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results_tuned.json"), "w") as f:
        json.dump({"winner": winner, "configs": rows,
                   "frontier": {k: v[0] for k, v in frontier.items()}}, f, indent=2)
    with open(os.path.join(args.out_dir, "report_tuned.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"saved -> {args.out_dir}/results_tuned.json and report_tuned.md")


if __name__ == "__main__":
    main()
