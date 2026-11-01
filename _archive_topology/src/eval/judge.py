"""LLM-judge for grading correctness / decline quality (PRD §5.5).

Two jobs:
  1. answer_correct(problem, model_answer) -> bool
        Grade an ANSWERABLE item's answer against the reference solution.
  2. decline_quality(problem, model_reason) -> 0..1
        Grade whether a DECLINE named the right missing hypothesis.

IMPORTANT: run the judge on a DIFFERENT provider than the one used to GENERATE
the dataset (PRD risk #7). Default judge provider here is 'anthropic'; if you
generated with anthropic, pass --provider openai in the eval runner.
"""
from __future__ import annotations

from src.llm import chat, extract_json
from src.schema import Problem

JUDGE_SYSTEM = (
    "You are a strict grader of point-set topology solutions. You compare a "
    "candidate answer against a reference and judge correctness on the "
    "mathematics, not the phrasing."
)

GRADE_ANSWER_PROMPT = """Grade this candidate solution.

PROBLEM: {statement}
REFERENCE SOLUTION: {solution}
CANDIDATE ANSWER: {answer}

Is the candidate mathematically correct (sound and reaching the right conclusion)?
Return ONLY JSON: {{"correct": true|false, "notes": "<brief>"}}"""

GRADE_DECLINE_PROMPT = """A model DECLINED this problem, which is the correct action
because the problem is ill-posed / out of scope.

PROBLEM: {statement}
GOLD MISSING/FALSE HYPOTHESIS (or reason): {missing_hypothesis}
MODEL'S STATED REASON: {reason}

Did the model identify the correct reason (name the right missing/false hypothesis,
or correctly say it is out of scope)? Score 1.0 if it nailed the reason, 0.5 if it
declined for a vague/partially-right reason, 0.0 if the reason is wrong/absent.
Return ONLY JSON: {{"score": 0.0-1.0, "notes": "<brief>"}}"""


def answer_correct(p: Problem, answer: str, provider="anthropic", model=None) -> bool:
    prompt = GRADE_ANSWER_PROMPT.format(
        statement=p.statement, solution=p.solution or "", answer=answer)
    try:
        v = extract_json(chat(prompt, provider=provider, model=model,
                              system=JUDGE_SYSTEM, temperature=0.0))
        return bool(v.get("correct"))
    except Exception as e:  # noqa: BLE001
        print(f"[judge] answer grade failed for {p.id}: {e}")
        return False


def decline_quality(p: Problem, reason: str, provider="anthropic", model=None) -> float:
    prompt = GRADE_DECLINE_PROMPT.format(
        statement=p.statement, missing_hypothesis=p.missing_hypothesis or "", reason=reason)
    try:
        v = extract_json(chat(prompt, provider=provider, model=model,
                              system=JUDGE_SYSTEM, temperature=0.0))
        return float(v.get("score", 0.0))
    except Exception as e:  # noqa: BLE001
        print(f"[judge] decline grade failed for {p.id}: {e}")
        return 0.0
