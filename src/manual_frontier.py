"""Export label-free frontier prompts and merge manually generated predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.baselines import SonnetPredictor
from src.schema import read_items


def export_chunk(args):
    train = read_items(args.train)
    test = read_items(args.test)
    if not (0 <= args.start < args.end <= len(test)):
        raise ValueError(f"invalid range [{args.start}, {args.end}) for {len(test)} items")
    predictor = SonnetPredictor(train)
    rows = [
        {"id": item.id, "prompt": predictor._prompt(item)}
        for item in test[args.start:args.end]
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"start": args.start, "end": args.end, "items": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} label-free prompts -> {out}")


def merge_parts(args):
    test = read_items(args.test)
    part_dir = Path(args.part_dir)
    parts = []
    for path in part_dir.glob(f"{args.tag}_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("tag") != args.tag or data.get("model") != args.model:
            continue
        parts.append((int(data["start"]), int(data["end"]), path, data))
    parts.sort()

    ids, preds = [], []
    expected_start = 0
    for start, end, path, data in parts:
        if start != expected_start or end <= start:
            raise ValueError(
                f"non-contiguous part {path}: expected start {expected_start}, "
                f"got [{start}, {end})")
        expected_ids = [item.id for item in test[start:end]]
        if data.get("ids") != expected_ids:
            raise ValueError(f"ID/order mismatch in {path}")
        rows = data.get("preds")
        if rows is None:
            rows = data.get("predictions")
        if rows is None:
            rows = data.get("probabilities")
        if rows is None:
            rows = data.get("probs", [])
        if len(rows) != end - start:
            raise ValueError(f"prediction count mismatch in {path}")
        for row in rows:
            if (
                len(row) != 5
                or any(not math.isfinite(float(x)) or float(x) < 0 for x in row)
                or sum(float(x) for x in row) <= 0
            ):
                raise ValueError(f"invalid probability row in {path}: {row}")
            total = sum(float(x) for x in row)
            preds.append([float(x) / total for x in row])
        ids.extend(data["ids"])
        expected_start = end

    if expected_start != len(test):
        raise ValueError(
            f"parts cover {expected_start}/{len(test)} items for {args.tag}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"model": args.model, "tag": args.tag, "ids": ids, "preds": preds},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"merged {len(parts)} parts / {len(preds)} predictions -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--train", required=True)
    export.add_argument("--test", required=True)
    export.add_argument("--start", type=int, required=True)
    export.add_argument("--end", type=int, required=True)
    export.add_argument("--out", required=True)
    export.set_defaults(func=export_chunk)

    merge = sub.add_parser("merge")
    merge.add_argument("--test", required=True)
    merge.add_argument("--part-dir", required=True)
    merge.add_argument("--tag", required=True)
    merge.add_argument("--model", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=merge_parts)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
