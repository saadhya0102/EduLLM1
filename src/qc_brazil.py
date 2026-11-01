"""Fail-closed deterministic QC for prepared Brazilian Item splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from src import lp
from src.schema import read_items


MARKERS = re.compile(
    r"VALIDIN(?:E|3)P|\*[Rr]\d+\*|\ufffd|h\u01a9|RASCUNHO|"
    r"AS QUEST(?:O|\u00d5)ES ABAIXO VISAM CONHECER SUA "
    r"OPINI(?:A|\u00c3)O|QUEST(?:A|\u00c3)O DISCURSIVA",
    re.I)


def _canonical(item) -> str:
    values = [item.question, *item.options]
    text = "\n".join(
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value))
        .strip().casefold()
        for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--min-responses", type=int, default=100)
    ap.add_argument("--allow-val-year-train", action="store_true")
    args = ap.parse_args()

    paths = {
        split: f"{args.prefix}_{split}.jsonl"
        for split in ("train", "val", "test")}
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"QC inputs missing: {missing}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    errors: list[dict] = []
    warnings: list[dict] = []
    seen_ids = {}
    seen_content = {}
    split_years = {}
    counts = {}
    token_lengths = []

    def fail(split, item_id, reason):
        errors.append({"split": split, "id": item_id, "reason": reason})

    for split, path in paths.items():
        items = read_items(path)
        counts[split] = len(items)
        split_years[split] = sorted({
            int(item.source.rsplit(" ", 1)[-1]) for item in items})
        for item in items:
            if item.id in seen_ids:
                fail(split, item.id, f"duplicate ID also in {seen_ids[item.id]}")
            seen_ids[item.id] = split
            if not re.fullmatch(r"enade:\d{4}:(?:general|\d+):\d+", item.id):
                fail(split, item.id, "invalid ENADE item ID")
            try:
                id_year = int(item.id.split(":", 2)[1])
                source_year = int(item.source.rsplit(" ", 1)[-1])
                if id_year != source_year:
                    fail(split, item.id, "ID/source year mismatch")
            except (IndexError, ValueError):
                fail(split, item.id, "unparseable source year")
            if item.k() != 5 or any(not str(x).strip() for x in item.options):
                fail(split, item.id, "requires five nonempty options")
            normalized_options = {
                re.sub(r"\s+", " ", unicodedata.normalize("NFKC", x))
                .strip().casefold() for x in item.options}
            if len(normalized_options) != len(item.options):
                fail(split, item.id, "duplicate normalized options")
            if not 0 <= item.correct_index < item.k():
                fail(split, item.id, "correct_index out of range")
            if item.language != "pt" or not item.source or not item.population:
                fail(split, item.id, "missing provenance metadata")
            if len(item.question.strip()) < 20:
                fail(split, item.id, "question too short")
            if MARKERS.search(item.question) or any(
                    MARKERS.search(option) for option in item.options):
                fail(split, item.id, "residual PDF extraction marker")
            if (len(item.dist) != 5 or
                    any(not math.isfinite(x) or x < 0 or x > 1
                        for x in item.dist) or
                    abs(sum(item.dist) - 1) > 1e-6):
                fail(split, item.id, "invalid probability distribution")
            if item.n_responses < args.min_responses:
                fail(split, item.id, "insufficient valid responses")
            reconstructed = [round(x * item.n_responses) for x in item.dist]
            if (any(abs(x * item.n_responses - n) > 1e-5
                    for x, n in zip(item.dist, reconstructed)) or
                    sum(reconstructed) != item.n_responses):
                fail(split, item.id, "distribution/count inconsistency")
            digest = _canonical(item)
            if digest in seen_content:
                fail(
                    split, item.id,
                    f"duplicate content also in {seen_content[digest]}")
            seen_content[digest] = f"{split}:{item.id}"
            token_len = len(lp.prompt_ids(tokenizer, item))
            token_lengths.append(token_len)
            if token_len > args.max_len:
                fail(split, item.id, f"{token_len} tokens exceeds {args.max_len}")
            keyed_share = item.dist[item.correct_index]
            if keyed_share < 0.10:
                warnings.append({
                    "split": split, "id": item.id,
                    "reason": f"keyed option share is only {keyed_share:.1%}"})

    train_max = max(split_years["train"], default=0)
    val_min = min(split_years["val"], default=9999)
    val_max = max(split_years["val"], default=0)
    test_min = min(split_years["test"], default=9999)
    valid_year_order = (
        train_max <= val_min <= val_max < test_min
        if args.allow_val_year_train else
        train_max < val_min <= val_max < test_min)
    if not valid_year_order:
        errors.append({
            "split": "all", "id": "",
            "reason": f"temporal split violation: {split_years}"})

    report = {
        "qc_passed": not errors,
        "counts": counts,
        "split_years": split_years,
        "max_tokens": max(token_lengths, default=0),
        "errors": errors,
        "warnings": warnings,
    }
    out = f"{args.prefix}_qc.json"
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(out)
    print(
        f"QC {'PASSED' if not errors else 'FAILED'}: "
        f"{sum(counts.values())} items, {len(errors)} errors, "
        f"{len(warnings)} warnings -> {out}", flush=True)
    if errors:
        for error in errors[:20]:
            print(f"  {error}", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
