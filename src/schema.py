"""Data schema for the MCQ answer-distribution task.

One record = one multiple-choice item with the *empirical distribution* of which
option students actually picked. The model's job is to predict that distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import json

LETTERS = ["A", "B", "C", "D", "E", "F"]  # supports up to 6 options; usually 4


@dataclass
class Item:
    id: str
    question: str                 # question stem
    options: list[str]            # option texts, in order (index 0 = "A", ...)
    dist: list[float]             # empirical selection fraction per option (sums to 1)
    correct_index: int            # 0-based index of the correct option
    n_responses: int = 0          # how many student responses the dist is built from
    skill: Optional[str] = None   # topic / Common Core skill
    split: Optional[str] = None   # "train" | "test"
    source: Optional[str] = None  # assessment + year (e.g. "ENADE 2023")
    population: Optional[str] = None  # population whose choices define ``dist``
    language: Optional[str] = None    # ISO-ish language tag, e.g. "pt" or "en"

    def k(self) -> int:
        return len(self.options)

    def letters(self) -> list[str]:
        return LETTERS[: self.k()]

    def normalized(self) -> list[float]:
        s = sum(self.dist)
        return [x / s for x in self.dist] if s > 0 else [1 / self.k()] * self.k()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Item":
        return Item(**json.loads(line))


def read_items(path: str) -> list[Item]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Item.from_json(line))
    return out


def write_items(path: str, items: list[Item]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
