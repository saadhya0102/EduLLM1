"""Score every predictor on the test set and write a comparative report.

Predictors:
  - uniform            : 1/K floor
  - mean               : mean training distribution (the real bar to beat)
  - <frontier tags>    : loaded from results/preds_<tag>.json (Sonnet/Haiku/Opus,
                         produced by src.predict_frontier)
  - trained            : the fine-tuned Qwen adapter (HFPredictor), if present

Metrics (src.metrics): MAE (avg pct-pts off, primary), Brier (squared error -
penalizes larger misses), TV, KL, JS. Lower is better for all.

Outputs:
  results/results.json  - full metrics for every predictor
  results/report.md     - ranked table + deltas vs the mean baseline

    python -m src.run_experiment --test data/synth_test.jsonl \
        --train data/synth_train.jsonl --frontier sonnet,haiku,opus \
        --adapter out/qwen1.5b-adapter
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.schema import read_items
from src import baselines, metrics


def _predict_all(predictor, test_items):
    return [predictor(it) for it in test_items]


def _load_frontier(tag, out_dir, test_items):
    path = os.path.join(out_dir, f"preds_{tag}.json")
    if not os.path.isfile(path):
        print(f"[warn] {path} missing — skipping frontier '{tag}'")
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ids, preds = d["ids"], d["preds"]
    # align to test order by id
    by_id = {i: p for i, p in zip(ids, preds)}
    aligned, ok = [], True
    for it in test_items:
        if it.id not in by_id:
            ok = False
            break
        aligned.append(by_id[it.id])
    if not ok or len(aligned) != len(test_items):
        print(f"[warn] '{tag}' predictions don't cover all test items — skipping")
        return None, d.get("model", tag)
    return aligned, d.get("model", tag)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/synth_test.jsonl")
    ap.add_argument("--train", default="data/synth_train.jsonl")
    ap.add_argument("--frontier", default="sonnet,haiku,opus")
    ap.add_argument("--adapter", default="out/qwen1.5b-adapter")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dataset-name", default="synthetic")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-len", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_items = read_items(args.train)
    test_items = read_items(args.test)
    trues = [it.normalized() for it in test_items]
    print(f"train={len(train_items)}  test={len(test_items)}  dataset={args.dataset_name}\n")

    results = {}          # label -> metrics dict
    model_of = {}         # label -> underlying model id (for the report)

    # --- simple baselines (computed inline) ---
    results["uniform"] = metrics.aggregate(
        _predict_all(baselines.uniform_predictor(), test_items), trues)
    results["mean"] = metrics.aggregate(
        _predict_all(baselines.mean_distribution(train_items), test_items), trues)
    for lbl in ("uniform", "mean"):
        print(metrics.format_report(lbl, results[lbl]))

    # --- frontier baselines (loaded from subagent-produced JSON) ---
    for tag in [t.strip() for t in args.frontier.split(",") if t.strip()]:
        preds, model = _load_frontier(tag, args.out_dir, test_items)
        model_of[tag] = model
        if preds is not None:
            results[tag] = metrics.aggregate(preds, trues)
            print(metrics.format_report(tag, results[tag]))

    # --- the trained small model ---
    if args.adapter and os.path.isdir(args.adapter):
        try:
            from src.predict_hf import HFPredictor
            hf = HFPredictor(
                args.base, args.adapter, temperature=args.temperature,
                load_4bit=args.load_4bit, max_len=(args.max_len or None))
            results["trained"] = metrics.aggregate(_predict_all(hf, test_items), trues)
            model_of["trained"] = f"{args.base} + LoRA"
            print(metrics.format_report("trained", results["trained"]))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] trained-model eval failed: {e}")
    else:
        print(f"[warn] adapter {args.adapter} not found — skipping trained model")

    # --- persist raw results ---
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset_name, "n_train": len(train_items),
                   "n_test": len(test_items), "models": model_of,
                   "results": results}, f, indent=2)

    # --- markdown report, ranked by MAE ---
    order = ["MAE", "Brier", "TV", "KL", "JS"]
    ranked = sorted(results.items(), key=lambda kv: kv[1]["MAE"])
    base_mae = results["mean"]["MAE"] if "mean" in results else None

    lines = [f"# Answer-distribution results ({args.dataset_name})",
             f"\n**Test items:** {len(test_items)}  |  **Train items:** {len(train_items)}",
             "\nLower is better for every metric. **MAE** = avg percentage-points off "
             "(primary). **Brier** = squared error (penalizes large misses).\n",
             "| Rank | Predictor | Model | " + " | ".join(order) +
             " | ΔMAE vs mean |",
             "|---|---|---|" + "---|" * len(order) + "---|"]
    for rank, (lbl, m) in enumerate(ranked, 1):
        delta = "" if base_mae is None else f"{m['MAE'] - base_mae:+.4f}"
        model = model_of.get(lbl, "—")
        row = (f"| {rank} | {lbl} | {model} | "
               + " | ".join(f"{m[k]:.4f}" for k in order)
               + f" | {delta} |")
        lines.append(row)

    winner = ranked[0][0]
    lines.append(f"\n**Best by MAE:** `{winner}`.")
    if "trained" in results and base_mae is not None:
        d = results["trained"]["MAE"] - base_mae
        verdict = "BEATS" if d < 0 else "does NOT beat"
        lines.append(f"\nTrained small model {verdict} the mean baseline on MAE "
                     f"({results['trained']['MAE']:.4f} vs {base_mae:.4f}, Δ={d:+.4f}).")
        # vs frontier
        for tag in ("sonnet", "haiku", "opus"):
            if tag in results:
                dd = results["trained"]["MAE"] - results[tag]["MAE"]
                v = "beats" if dd < 0 else "loses to"
                lines.append(f"- vs {tag}: trained {v} it "
                             f"({results['trained']['MAE']:.4f} vs {results[tag]['MAE']:.4f}, Δ={dd:+.4f})")

    report = "\n".join(lines) + "\n"
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"saved -> {args.out_dir}/results.json  and  {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
