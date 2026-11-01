"""Ablation generator (PRD §5.3 step 2).

Takes answerable, well-posed problems and produces ill-posed variants by
removing a load-bearing hypothesis (making the statement FALSE) or deleting
given data (making it UNDERSPECIFIED). The generator MUST propose the specific
missing hypothesis and a witnessing counterexample — items without a
counterexample are later rejected by the verifier (guards PRD risk #2).

Also emits out-of-scope items from a fixed template list.

Run:
    python -m src.data.generate --in data/seed_answerable.jsonl \
        --out data/generated_raw.jsonl --provider anthropic --n-per 2
"""
from __future__ import annotations

import argparse
import json

from src.llm import chat, extract_json
from src.schema import Problem, KIND_FALSE, KIND_UNDERSPECIFIED, KIND_OUT_OF_SCOPE

ABLATE_SYSTEM = (
    "You are an expert in point-set topology who constructs instructive "
    "ill-posed variants of true theorems for a teaching dataset. You are precise "
    "about which hypotheses are load-bearing and you always supply an explicit "
    "counterexample witnessing the failure."
)

ABLATE_PROMPT = """Here is a TRUE, well-posed point-set topology problem and its solution:

PROBLEM: {statement}
SOLUTION: {solution}

Produce {n} DISTINCT ill-posed variants. Each variant must be obtained by
REMOVING a single load-bearing hypothesis (so the statement becomes FALSE), or
by DELETING given data (so it becomes UNDERSPECIFIED and cannot be determined).

Only remove a hypothesis that is TRULY load-bearing: there must exist an explicit
counterexample showing the weakened statement fails. Do NOT weaken a hypothesis
that the result does not actually need.

Return ONLY a JSON array. Each element:
{{
  "kind": "false" | "underspecified",
  "statement": "<the ill-posed problem, phrased naturally as if a real problem>",
  "missing_hypothesis": "<the specific hypothesis removed / data deleted>",
  "counterexample": "<explicit witness that the weakened statement fails; for underspecified, explain why it cannot be determined>"
}}"""

# Fixed out-of-scope items (adjacent fields, no point-set content). Kept broad
# across many neighboring areas so the model learns to decline "wrong field"
# problems rather than pattern-matching one or two (PRD risk #4).
OUT_OF_SCOPE = [
    ("Compute the fundamental group of the torus.", "algebraic topology"),
    ("Determine the Euler characteristic of a genus-2 surface.", "algebraic topology"),
    ("Compute the singular homology of the 2-sphere.", "algebraic topology"),
    ("Evaluate the contour integral of 1/z around the unit circle.", "complex analysis"),
    ("State and apply the residue theorem to a rational function.", "complex analysis"),
    ("Prove that a bounded entire function is constant (Liouville).", "complex analysis"),
    ("Find the eigenvalues of a given 3x3 real symmetric matrix.", "linear algebra"),
    ("Diagonalize a linear operator with distinct eigenvalues.", "linear algebra"),
    ("Solve the heat equation on an interval with Dirichlet boundary conditions.", "PDE"),
    ("Find the general solution of y'' + y = 0.", "ODE"),
    ("Classify compact orientable surfaces up to diffeomorphism.", "differential topology"),
    ("Compute the derivative of a smooth map between manifolds at a point.", "differential geometry"),
    ("Compute the Gaussian curvature of the unit sphere.", "differential geometry"),
    ("Prove that the rationals are countable.", "set theory / cardinality"),
    ("Show that the power set of the naturals is uncountable.", "set theory / cardinality"),
    ("Prove that every group of prime order is cyclic.", "abstract algebra"),
    ("Determine whether a given polynomial is irreducible over Q.", "abstract algebra"),
    ("Compute the probability of at least one success in n Bernoulli trials.", "probability"),
    ("Find the Fourier series of a square wave.", "harmonic analysis"),
    ("Evaluate the integral of e^{-x^2} over the real line.", "real analysis / integration"),
    ("Prove that sqrt(2) is irrational.", "number theory"),
    ("Determine the convergence of the series sum 1/n^2.", "real analysis / series"),
]


def generate_ablations(problems, provider, model, n_per):
    out = []
    for p in problems:
        prompt = ABLATE_PROMPT.format(statement=p.statement, solution=p.solution or "",
                                      n=n_per)
        try:
            raw = chat(prompt, provider=provider, model=model, system=ABLATE_SYSTEM,
                       temperature=0.7, max_tokens=3000)
            data = extract_json_array(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[generate] {p.id} failed: {e}")
            continue
        for i, item in enumerate(data):
            kind = item.get("kind")
            if kind not in (KIND_FALSE, KIND_UNDERSPECIFIED):
                continue
            out.append(Problem(
                id=f"{p.id}-ab{i}",
                kind=kind,
                statement=item.get("statement", "").strip(),
                missing_hypothesis=item.get("missing_hypothesis", "").strip(),
                counterexample=item.get("counterexample", "").strip(),
                source=f"ablation of {p.id}",
                parent_id=p.id,
                topic=p.topic,
            ))
        print(f"[generate] {p.id}: +{len(data)} variants")
    return out


def generate_out_of_scope():
    out = []
    for i, (stmt, field) in enumerate(OUT_OF_SCOPE):
        out.append(Problem(
            id=f"oos-{i:02d}",
            kind=KIND_OUT_OF_SCOPE,
            statement=stmt,
            missing_hypothesis=f"belongs to {field}, not point-set topology",
            counterexample="",
            source="out-of-scope template",
            topic="out_of_scope",
            verified=True,  # templated, trusted
        ))
    return out


def extract_json_array(text: str):
    """Extract a JSON array (possibly fenced) from model output."""
    t = text.strip()
    if "```" in t:
        for part in t.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                t = part
                break
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1:
        # maybe a single object -> wrap
        return [extract_json(text)]
    return json.loads(t[start : end + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/seed_*.jsonl",
                    help="path or glob; all matching seed files are used as sources")
    ap.add_argument("--out", default="data/generated_raw.jsonl")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n-per", type=int, default=2, help="ill-posed variants per source")
    args = ap.parse_args()

    from src.schema import read_problems, write_problems
    import glob as _glob
    from src.schema import KIND_ANSWERABLE
    files = sorted(_glob.glob(args.inp)) or [args.inp]
    src_problems, seen = [], set()
    for f in files:
        for p in read_problems(f):
            if p.kind == KIND_ANSWERABLE and p.id not in seen:
                seen.add(p.id)
                src_problems.append(p)
    print(f"[generate] {len(src_problems)} source problems from {len(files)} file(s)")
    ablations = generate_ablations(src_problems, args.provider, args.model, args.n_per)
    oos = generate_out_of_scope()
    all_gen = ablations + oos
    write_problems(args.out, all_gen)
    print(f"\nWrote {len(all_gen)} generated items "
          f"({len(ablations)} ablations + {len(oos)} out-of-scope) to {args.out}")


if __name__ == "__main__":
    main()
