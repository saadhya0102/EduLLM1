"""Baselines the fine-tuned model must beat (PRD win condition).

A "predictor" is any callable: Item -> predicted distribution (list of floats
summing to 1, length = item.k()).

  - uniform_predictor:      1/K each (floor).
  - mean_distribution:      predict the average training distribution (strong,
                            position-based; the real bar to beat).
  - SonnetPredictor:        frontier few-shot — ask the model for the percentages.
"""
from __future__ import annotations

import json
import hashlib
import re
from collections import defaultdict

from src.llm import chat
from src.schema import Item


def uniform_predictor(_train=None):
    def predict(item: Item):
        return [1 / item.k()] * item.k()
    return predict


def mean_distribution(train_items: list[Item], k: int | None = None):
    """Average distribution by option count, preferring the same data source."""
    groups = defaultdict(list)
    for it in train_items:
        if k is not None and it.k() != k:
            continue
        groups[("all", None, it.k())].append(it.normalized())
        if it.source:
            groups[("source", it.source, it.k())].append(it.normalized())
        if it.skill:
            groups[("skill", it.skill, it.k())].append(it.normalized())

    means = {}
    for key, rows in groups.items():
        means[key] = [sum(r[j] for r in rows) / len(rows)
                      for j in range(key[2])]

    def predict(item: Item):
        keys = [
            ("skill", item.skill, item.k()),
            ("source", item.source, item.k()),
            ("all", None, item.k()),
        ]
        mean = next((means[key] for key in keys if key in means), None)
        return list(mean or [1 / item.k()] * item.k())
    return predict


def _fewshot_instruction(item: Item):
    population = (item.population or
                  "a class of school students (grades 3-8)")
    example = ", ".join([f"{100 / item.k():.1f}"] * item.k())
    return (
        f"You are predicting how {population} will answer a multiple-choice question. "
        "Many people will pick the correct answer, while others pick tempting wrong "
        "answers matching misconceptions. Estimate the PERCENTAGE choosing each option. "
        "Output ONLY a JSON list of numbers (one per option, summing to about 100), "
        f"for example [{example}]."
    )


class SonnetPredictor:
    """Frontier few-shot distribution predictor (the model to beat)."""

    def __init__(self, train_items, provider="anthropic", model="claude-sonnet-5",
                 k_shot=5, seed=0):
        self.train_items = list(train_items)
        self.k_shot = int(k_shot)
        self.seed = int(seed)
        self.provider, self.model = provider, model

    def _fmt(self, item: Item):
        opts = "\n".join(f"{L}) {t}" for L, t in zip(item.letters(), item.options))
        context = []
        if item.population:
            context.append(f"Population: {item.population}")
        if item.source:
            context.append(f"Assessment source: {item.source}")
        prefix = "\n".join(context)
        return f"{prefix + chr(10) if prefix else ''}Question: {item.question}\n{opts}"

    def _shots(self, item: Item):
        import random
        pool = [it for it in self.train_items
                if it.k() == item.k() and item.skill and
                it.skill == item.skill]
        if not pool:
            pool = [it for it in self.train_items
                    if it.k() == item.k() and
                    (not item.source or it.source == item.source)]
        if not pool:
            pool = [it for it in self.train_items if it.k() == item.k()]
        key = (f"{self.seed}|{item.source or ''}|{item.skill or ''}|"
               f"{item.population or ''}|{item.k()}")
        stable_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
        rng = random.Random(stable_seed)
        return rng.sample(pool, min(self.k_shot, len(pool))) if pool else []

    def _prompt(self, item: Item):
        ex = []
        for s in self._shots(item):
            pct = [round(100 * x) for x in s.normalized()]
            ex.append(f"{self._fmt(s)}\nAnswer: {json.dumps(pct)}")
        shots = "\n\n".join(ex)
        instr = _fewshot_instruction(item)
        return (f"{instr}\n\n{shots}\n\n{self._fmt(item)}\nAnswer:"
                if shots else f"{instr}\n\n{self._fmt(item)}\nAnswer:")

    def __call__(self, item: Item):
        raw = chat(self._prompt(item), provider=self.provider, model=self.model,
                   max_tokens=100, temperature=0.0)
        nums = re.findall(r"-?\d+\.?\d*", raw)
        vals = [max(0.0, float(x)) for x in nums[: item.k()]]
        if len(vals) != item.k() or sum(vals) == 0:
            return [1 / item.k()] * item.k()
        s = sum(vals)
        return [v / s for v in vals]
