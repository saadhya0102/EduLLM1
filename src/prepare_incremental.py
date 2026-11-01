"""Build a new-item training slice with replay while freezing stage-1 holdouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from src.schema import read_items, write_items


def _hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-train", required=True)
    ap.add_argument("--expanded-train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seen-out", required=True)
    ap.add_argument("--new-items", type=int, default=1000)
    ap.add_argument("--replay-items", type=int, default=250)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    old = read_items(args.stage1_train)
    expanded = read_items(args.expanded_train)
    old_ids = {item.id for item in old}
    candidates = [item for item in expanded if item.id not in old_ids]
    if len(candidates) < args.new_items:
        raise SystemExit(
            f"need {args.new_items} genuinely new items, found {len(candidates)}")

    rng = random.Random(args.seed + 2000)
    candidates.sort(key=lambda item: item.id)
    rng.shuffle(candidates)
    new = candidates[:args.new_items]

    replay_pool = sorted(old, key=lambda item: item.id)
    rng.shuffle(replay_pool)
    replay = replay_pool[:min(args.replay_items, len(replay_pool))]
    incremental = new + replay
    rng.shuffle(incremental)

    seen = old + new
    write_items(args.out, incremental)
    write_items(args.seen_out, seen)
    manifest = {
        "stage1_items": len(old),
        "new_items": len(new),
        "replay_items": len(replay),
        "incremental_examples": len(incremental),
        "cumulative_unique_items": len(seen),
        "stage1_train": args.stage1_train,
        "expanded_train": args.expanded_train,
        "seed": args.seed,
        "sha256": {
            "incremental": _hash(args.out),
            "seen": _hash(args.seen_out),
        },
    }
    manifest_path = str(Path(args.out).with_suffix(".manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(
        f"incremental set: {len(new)} new + {len(replay)} replay; "
        f"{len(seen)} cumulative unique items", flush=True)


if __name__ == "__main__":
    main()
