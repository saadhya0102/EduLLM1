"""Shared 'letter-probability' formatting for train + inference (must match!).

We frame the task natively for a causal LM: prompt ends right where the answer
letter would go, so the model's next-token distribution over {A,B,C,D} — after
renormalizing over just those tokens — IS the predicted answer distribution.
Training pushes that distribution toward the empirical student distribution.
"""
from __future__ import annotations

from src.schema import Item

LEGACY_SYSTEM = (
    "You model how a class of school students (grades 3-8) answers multiple-choice "
    "math questions. Students often pick tempting wrong answers matching common "
    "misconceptions, not only the correct one."
)

GENERIC_SYSTEM = (
    "You predict the empirical answer-choice behavior of a stated learner population. "
    "Predict what people in that population actually select, including tempting wrong "
    "answers and systematic misconceptions; do not merely solve the question."
)


def build_messages(item: Item):
    opts = "\n".join(f"{L}) {t}" for L, t in zip(item.letters(), item.options))
    context = []
    if item.population:
        context.append(f"Population: {item.population}")
    if item.source:
        context.append(f"Assessment source: {item.source}")
    if item.language:
        context.append(f"Question language: {item.language}")
    prefix = "\n".join(context)
    if prefix:
        prefix += "\n\n"
    subject = "member of this population" if context else "student"
    user = (f"{prefix}Question: {item.question}\n{opts}\n\n"
            f"Which option would a randomly chosen {subject} pick? "
            "Answer with a single letter.")
    system = GENERIC_SYSTEM if context else LEGACY_SYSTEM
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def letter_token_ids(tokenizer, k: int):
    """Token id emitted first for each option letter A..(k). Robust to tokenizers
    that prefix a space: we take the LAST token of the encoded letter."""
    ids = []
    for L in ["A", "B", "C", "D", "E", "F"][:k]:
        enc = tokenizer.encode(L, add_special_tokens=False)
        ids.append(enc[-1])
    return ids


def prompt_ids(tokenizer, item: Item):
    """Tokenized chat prompt ending at the generation point (answer = next token).

    Returns a flat list[int]. Robust to the transformers version: <5.x returned a
    plain list from apply_chat_template(tokenize=True); 5.x returns a BatchEncoding
    (token ids under 'input_ids', possibly nested one level for the single convo).
    """
    out = tokenizer.apply_chat_template(
        build_messages(item), add_generation_prompt=True, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids and isinstance(ids[0], (list, tuple)):   # nested [[...]] -> take first
        ids = ids[0]
    return list(ids)
