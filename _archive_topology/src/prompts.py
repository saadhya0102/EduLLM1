"""Shared prompt builders: the inference protocol (identical for baseline and
fine-tuned model), and the training-target formatter.

Keeping the inference prompt in ONE place guarantees the baseline and the
fine-tuned model are evaluated under an identical protocol (PRD §5.5).
"""
from __future__ import annotations

import json

from src.schema import Problem, RESPONSE_SCHEMA_DOC, gold_decision, KIND_ANSWERABLE

SYSTEM_PROMPT = (
    "You are a careful point-set topology expert. Solve well-posed problems. "
    "If a problem is FALSE, UNDERSPECIFIED, or OUTSIDE point-set topology, you "
    "must DECLINE rather than guess: a calibrated decline is far better than a "
    "confident wrong answer. When you decline, name the specific missing or "
    "false hypothesis (or why it is out of scope). Set 'confidence' to reflect "
    "how sure you are of your answer — lower it when a problem is hard or you are "
    "unsure, rather than always reporting high confidence."
)

# One compact worked example, appended to the prompt so smaller models reliably
# emit clean single-object JSON (PRD risks #8/#9).
_FORMAT_EXAMPLE = (
    'Example (format only):\n'
    '{"decision": "decline", "confidence": 0.9, '
    '"reason": "Missing hypothesis: Hausdorff. Counterexample: in an indiscrete '
    'space a compact set need not be closed.", "answer": ""}'
)


def build_inference_prompt(statement: str) -> str:
    """The user-turn shown to any model at eval/inference time (identical for
    baseline and fine-tuned model, so the comparison is fair — PRD §5.5)."""
    return f"{statement}\n\n{RESPONSE_SCHEMA_DOC}\n\n{_FORMAT_EXAMPLE}"


# Confidence the SFT target expresses, by difficulty tier (answerable items).
# Spread (not a constant) so the model learns to MODULATE confidence, which is
# what makes ECE / risk-coverage meaningful (PRD §5.6). These are a transparent
# stand-in; error-mining (src/data/error_mine.py) can set p.confidence directly
# from the base model's empirical accuracy for true calibration.
_DIFF_CONF = {"easy": 0.92, "medium": 0.84, "hard": 0.74}


def target_confidence(p: Problem) -> float:
    """Confidence value for p's training target."""
    if p.confidence is not None:                      # explicit (e.g. error-mined)
        return round(max(0.0, min(1.0, float(p.confidence))), 2)
    if gold_decision(p.kind) == "answer":
        return _DIFF_CONF.get(p.difficulty or "medium", 0.84)
    # declines: naming a clear missing/false hypothesis is a confident judgment;
    # underspecified / out-of-scope are usually even clearer-cut.
    return 0.90 if p.kind in ("underspecified", "out_of_scope") else 0.88


def build_training_target(p: Problem) -> str:
    """The gold assistant response (structured JSON string) for SFT.

    Answerable -> decision=answer with the reference solution, high confidence.
    Decline kinds -> decision=decline naming the missing hypothesis / reason.
    """
    conf = target_confidence(p)
    if gold_decision(p.kind) == "answer":
        obj = {
            "decision": "answer",
            "confidence": conf,
            "reason": "",
            "answer": p.solution or "",
        }
    else:
        reason_bits = []
        if p.missing_hypothesis:
            reason_bits.append(f"Missing/false hypothesis: {p.missing_hypothesis}.")
        if p.counterexample:
            reason_bits.append(f"Counterexample: {p.counterexample}.")
        if p.kind == "out_of_scope":
            reason_bits.append("This is outside point-set topology.")
        if p.kind == "underspecified":
            reason_bits.append("The problem is underspecified as stated.")
        obj = {
            "decision": "decline",
            "confidence": conf,
            "reason": " ".join(reason_bits) or p.solution or "Ill-posed as stated.",
            "answer": "",
        }
    return json.dumps(obj, ensure_ascii=False)


def build_chat_example(p: Problem) -> dict:
    """A chat-format training example: messages list ready for SFTTrainer."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_inference_prompt(p.statement)},
            {"role": "assistant", "content": build_training_target(p)},
        ]
    }
