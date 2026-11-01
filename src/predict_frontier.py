"""Run one frontier model (via the gateway) as a distribution predictor over a
test set, and dump its predictions to JSON. Used to compute the frontier
baselines (the models the small fine-tune must beat).

Uses the SAME few-shot predictor as baselines.SonnetPredictor, so every frontier
model is prompted / parsed identically -> apples-to-apples with each other and
with the trained model's scoring.

    python -m src.predict_frontier --model claude-sonnet-5   --tag sonnet
    python -m src.predict_frontier --model claude-haiku-4-5  --tag haiku
    python -m src.predict_frontier --model claude-opus-4-8   --tag opus

Writes results/preds_<tag>.json = {"model":..., "tag":..., "ids":[...], "preds":[[...]]}.
"""
from __future__ import annotations

import argparse
import json
import os

from src.schema import read_items
from src.baselines import SonnetPredictor
from src.progress import Progress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="short name for the output file")
    ap.add_argument("--test", default="data/synth_test.jsonl")
    ap.add_argument("--train", default="data/synth_train.jsonl")
    ap.add_argument("--provider", default="anthropic")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    train = read_items(args.train)
    test = read_items(args.test)
    predictor = SonnetPredictor(train, provider=args.provider, model=args.model)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"preds_{args.tag}.json")
    partial = out + ".partial"
    ids, preds = [], []
    if os.path.isfile(partial):
        with open(partial, encoding="utf-8") as f:
            saved = json.load(f)
        expected = [item.id for item in test[:len(saved.get("ids", []))]]
        if (saved.get("model") == args.model and saved.get("tag") == args.tag
                and saved.get("ids") == expected):
            ids, preds = saved["ids"], saved["preds"]
            print(f"[{args.tag}] resuming after {len(ids)} saved items", flush=True)

    def save(path):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"model": args.model, "tag": args.tag,
                 "ids": ids, "preds": preds}, f)
        os.replace(tmp, path)

    progress = Progress(
        f"{args.tag} predictions", len(test), unit="items", min_interval=10)
    if ids:
        progress.update(len(ids), part="resumed predictions", force=True)
    for i, it in enumerate(test[len(ids):], start=len(ids)):
        preds.append(predictor(it))
        ids.append(it.id)
        save(partial)
        progress.update(i + 1, part=it.id)

    if os.path.isfile(partial):
        os.replace(partial, out)
    else:
        save(out)
    print(f"[{args.tag}] wrote {len(preds)} predictions -> {out}", flush=True)


if __name__ == "__main__":
    main()
