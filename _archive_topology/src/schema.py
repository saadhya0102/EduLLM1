"""Data schema, taxonomy, and the structured response format.

One dataset record = one problem with a gold label. Gold labels live on a clean
3-way decline taxonomy (PRD risk #5): answerable / false / underspecified /
out_of_scope. The first is "answer"; the rest are "decline".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional
import json

# ---- Taxonomy -------------------------------------------------------------

# Gold "kind" of a problem.
KIND_ANSWERABLE = "answerable"          # well-posed, has a correct solution
KIND_FALSE = "false"                    # ablated hypothesis makes statement false
KIND_UNDERSPECIFIED = "underspecified"  # missing data -> cannot be determined
KIND_OUT_OF_SCOPE = "out_of_scope"      # adjacent field, no point-set content

DECLINE_KINDS = {KIND_FALSE, KIND_UNDERSPECIFIED, KIND_OUT_OF_SCOPE}
ALL_KINDS = {KIND_ANSWERABLE} | DECLINE_KINDS


def gold_decision(kind: str) -> str:
    """Map a gold kind to the correct decision: 'answer' or 'decline'."""
    if kind not in ALL_KINDS:
        raise ValueError(f"Unknown kind: {kind}")
    return "answer" if kind == KIND_ANSWERABLE else "decline"


# ---- Records --------------------------------------------------------------

@dataclass
class Problem:
    """A single dataset item."""
    id: str
    kind: str                       # one of ALL_KINDS
    statement: str                  # the problem text shown to the model
    # For answerable items:
    solution: Optional[str] = None  # reference solution / final answer
    # For ill-posed items:
    missing_hypothesis: Optional[str] = None  # e.g. "compactness"
    counterexample: Optional[str] = None      # witness that the statement fails
    # Calibration (PRD §5.6): the confidence the SFT target should express.
    # Set explicitly (e.g. from error-mining) OR derived from difficulty; see
    # prompts.target_confidence. Avoids the degenerate "always 0.9" target.
    difficulty: Optional[str] = None   # "easy" | "medium" | "hard" (answerable)
    confidence: Optional[float] = None # explicit override in [0,1]
    # Provenance:
    source: Optional[str] = None    # e.g. "Munkres §26 Ex 5"
    parent_id: Optional[str] = None # id of the answerable item this was ablated from
    topic: Optional[str] = None     # e.g. "compactness", "connectedness"
    split: Optional[str] = None     # "train" | "test" | None
    verified: bool = False          # passed auto-verify
    spotchecked: Optional[bool] = None  # human spot-check result (None = not checked)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Problem":
        return Problem(**json.loads(line))


# ---- Structured response format (PRD §5.6) --------------------------------

RESPONSE_SCHEMA_DOC = """Respond with ONLY a single JSON object, no prose, of the form:
{
  "decision": "answer" | "decline",
  "confidence": <float 0.0-1.0>,
  "reason": "<if decline: name the missing/false hypothesis or why out of scope>",
  "answer": "<if answer: the solution; else empty string>"
}"""


@dataclass
class ModelResponse:
    decision: str            # "answer" | "decline"
    confidence: float
    reason: str = ""
    answer: str = ""
    raw: str = ""

    @staticmethod
    def parse(obj: dict, raw: str = "") -> "ModelResponse":
        decision = str(obj.get("decision", "")).strip().lower()
        if decision not in {"answer", "decline"}:
            # be lenient: treat missing/garbled decision as a (low-conf) answer
            decision = "answer"
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        return ModelResponse(
            decision=decision,
            confidence=conf,
            reason=str(obj.get("reason", "")),
            answer=str(obj.get("answer", "")),
            raw=raw,
        )


def parse_model_output(raw: str) -> "ModelResponse":
    """Robustly turn raw model text into a ModelResponse (PRD risks #8/#9).

    Order: (1) parse a clean JSON object; (2) regex-salvage decision/confidence
    from messy output; (3) last resort, infer from decline keywords. We only
    default to a confident 'answer' if there is genuinely no signal — that path
    biases FCR, so we try hard to avoid it.
    """
    import re
    from src.llm import extract_json  # local import avoids cycle at module load

    # (1) clean JSON
    try:
        return ModelResponse.parse(extract_json(raw), raw=raw)
    except Exception:
        pass

    # (2) regex salvage
    low = raw.lower()
    decision = None
    m = re.search(r'"?decision"?\s*[:=]\s*"?(answer|decline)"?', low)
    if m:
        decision = m.group(1)
    conf = None
    mc = re.search(r'"?confidence"?\s*[:=]\s*([01]?\.?\d+)', low)
    if mc:
        try:
            conf = max(0.0, min(1.0, float(mc.group(1))))
        except ValueError:
            conf = None

    # (3) keyword inference if decision still unknown
    if decision is None:
        if any(k in low for k in ("decline", "ill-posed", "ill posed",
                                  "cannot be determined", "out of scope",
                                  "underspecified", "missing hypothesis",
                                  "not enough information", "false as stated")):
            decision = "decline"
        else:
            decision = "answer"

    if conf is None:
        # unknown confidence: neutral, not maximal, so it doesn't dominate FCR
        conf = 0.5
    return ModelResponse(decision=decision, confidence=conf,
                         reason="", answer="", raw=raw)


def read_problems(path: str) -> list[Problem]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Problem.from_json(line))
    return out


def write_problems(path: str, problems: list[Problem]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in problems:
            f.write(p.to_json() + "\n")
