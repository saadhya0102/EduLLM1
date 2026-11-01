"""Evaluate a predictor (or compare several) on a test set of items.

Usage:
    # baselines only (works today on synthetic or real data):
    python -m src.eval --test data/synth_test.jsonl --train data/synth_train.jsonl \
        --predictors uniform,mean

    # add the Sonnet frontier baseline (uses the API):
    python -m src.eval --test data/synth_test.jsonl --train data/synth_train.jsonl \
        --predictors uniform,mean,sonnet

    # evaluate a fine-tuned model (after training):
    python -m src.eval --test data/dataset_test.jsonl --train data/dataset_train.jsonl \
        --predictors mean,hf --adapter out/qwen1.5b-adapter
"""
from __future__ import annotations

import argparse

from src.schema import read_items
from src.metrics import aggregate, format_report
from src import baselines


def build_predictor(name, train_items, args):
    if name == "uniform":
        return baselines.uniform_predictor()
    if name == "mean":
        return baselines.mean_distribution(train_items)
    if name == "sonnet":
        return baselines.SonnetPredictor(train_items, provider=args.provider,
                                         model=args.model)
    if name == "hf":
        from src.predict_hf import HFPredictor  # heavy import only if needed
        return HFPredictor(args.base, args.adapter)
    raise ValueError(f"unknown predictor: {name}")


def evaluate(predictor, test_items):
    preds, trues = [], []
    for it in test_items:
        preds.append(predictor(it))
        trues.append(it.normalized())
    return aggregate(preds, trues), preds, trues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--predictors", default="uniform,mean")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()

    train_items = read_items(args.train)
    test_items = read_items(args.test)
    print(f"train={len(train_items)}  test={len(test_items)}\n")

    results = {}
    for name in args.predictors.split(","):
        name = name.strip()
        if not name:
            continue
        pred = build_predictor(name, train_items, args)
        m, _, _ = evaluate(pred, test_items)
        results[name] = m
        print(format_report(name, m))

    # headline: lower MAE / KL is better; compare to mean baseline
    if "mean" in results:
        print("\n-- vs mean-distribution baseline (MAE; negative = better) --")
        base = results["mean"]["MAE"]
        for name, m in results.items():
            if name != "mean":
                print(f"  {name:8s}: {m['MAE'] - base:+.4f}")


if __name__ == "__main__":
    main()
