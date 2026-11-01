"""Drive a model over the test set, parse structured output, grade, and report
the full selective-prediction metric table (PRD §5.5 / M3).

Two backends:
  --backend api   : a frontier model via API (the baseline). Runs anywhere.
  --backend hf    : a HuggingFace base model + optional LoRA adapter (the
                    fine-tuned model). Run this ON COLAB with a GPU.

Grading uses the LLM-judge on a DIFFERENT provider than dataset generation
(risk #7): pass --judge-provider accordingly.

Examples:
    # baseline (frontier) — pick the weakest/easiest model here at M4
    python -m src.eval.run_eval --backend api --provider anthropic \
        --model claude-opus-4-8 --test data/dataset_test.jsonl --judge-provider openai

    # fine-tuned (Colab)
    python -m src.eval.run_eval --backend hf \
        --base Qwen/Qwen2.5-Math-7B-Instruct --adapter out/adapter \
        --test data/dataset_test.jsonl --judge-provider openai
"""
from __future__ import annotations

import argparse
import json

from src.llm import chat
from src.prompts import SYSTEM_PROMPT, build_inference_prompt
from src.schema import parse_model_output, read_problems, gold_decision, KIND_ANSWERABLE
from src.eval.metrics import EvalRow, compute_metrics, bootstrap_ci, format_report
from src.eval import judge as judgemod


def infer_api(statements, provider, model):
    outs = []
    for s in statements:
        raw = chat(build_inference_prompt(s), provider=provider, model=model,
                   system=SYSTEM_PROMPT, temperature=0.0)
        outs.append(raw)
    return outs


def infer_hf(statements, base, adapter, max_new_tokens=768):
    """Local/Colab inference with transformers (+ optional PEFT adapter)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype="auto", device_map="auto", load_in_4bit=True)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    outs = []
    for s in statements:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_inference_prompt(s)},
        ]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][inputs.shape[1]:], skip_special_tokens=True)
        outs.append(text)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["api", "hf"], required=True)
    ap.add_argument("--test", default="data/dataset_test.jsonl")
    ap.add_argument("--tau", type=float, default=0.5)
    # api backend
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--model", default=None)
    # hf backend
    ap.add_argument("--base", default="Qwen/Qwen2.5-Math-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    # judge (use different provider than dataset generation)
    ap.add_argument("--judge-provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--out", default="data/eval_results.json")
    args = ap.parse_args()

    problems = read_problems(args.test)
    statements = [p.statement for p in problems]

    print(f"[eval] running {args.backend} over {len(problems)} items...")
    if args.backend == "api":
        raws = infer_api(statements, args.provider, args.model)
        label = f"api:{args.provider}:{args.model or 'default'}"
    else:
        raws = infer_hf(statements, args.base, args.adapter)
        label = f"hf:{args.adapter or args.base}"

    rows = []
    for p, raw in zip(problems, raws):
        resp = parse_model_output(raw)

        answer_correct = None
        if p.kind == KIND_ANSWERABLE and resp.decision == "answer":
            answer_correct = judgemod.answer_correct(
                p, resp.answer, provider=args.judge_provider, model=args.judge_model)

        rows.append(EvalRow(
            id=p.id,
            gold_decision=gold_decision(p.kind),
            model_decision=resp.decision,
            confidence=resp.confidence,
            answer_correct=answer_correct,
            kind=p.kind,
        ))

    metrics = compute_metrics(rows, tau=args.tau)
    cis = {k: bootstrap_ci(rows, k, tau=args.tau)
           for k in ["FCR", "answer_accuracy", "coverage", "over_refusal"]}
    print(format_report(label, metrics, cis))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"label": label, "metrics": metrics,
                   "cis": {k: list(v) for k, v in cis.items()},
                   "rows": [r.__dict__ for r in rows]}, f, indent=2)
    print(f"\n[eval] wrote {args.out}")
    print("[eval] Compare baseline vs fine-tuned: the win is LOWER FCR at "
          "matched-or-better answer_accuracy AND acceptable coverage (PRD §5.5).")


if __name__ == "__main__":
    main()
