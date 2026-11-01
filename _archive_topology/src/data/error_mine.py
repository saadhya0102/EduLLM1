"""Error-mining pass for calibrated confidence (PRD §5.3 step 4 / §5.6).

Runs the BASE model over the answerable problems, k times each with sampling,
and records the fraction of attempts that produce a correct answer (graded by
the LLM-judge). That empirical accuracy is the item's calibrated target
confidence: problems the base model reliably solves -> high confidence targets;
problems it often botches -> low confidence, so after fine-tuning the model
learns to hedge (or decline) exactly where it tends to be wrong.

Feed the output to build_dataset:
    python -m src.data.error_mine --base Qwen/Qwen2.5-Math-7B-Instruct \
        --answerable "data/seed_*.jsonl" --k 5 --out data/error_mining.json
    python -m src.data.build_dataset --error-mining data/error_mining.json

Run this ON THE GPU MACHINE (hf backend). Grading uses the LLM-judge on a
DIFFERENT provider than dataset generation (risk #7).
"""
from __future__ import annotations

import argparse
import glob
import json

from src.prompts import SYSTEM_PROMPT, build_inference_prompt
from src.schema import parse_model_output, read_problems, KIND_ANSWERABLE
from src.eval import judge as judgemod


def load_answerable(pattern):
    seen, out = set(), []
    for f in sorted(glob.glob(pattern)) or [pattern]:
        for p in read_problems(f):
            if p.kind == KIND_ANSWERABLE and p.id not in seen:
                seen.add(p.id)
                out.append(p)
    return out


def generate_samples(model, tok, statement, k, max_new_tokens=512):
    import torch
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_inference_prompt(statement)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt").to(model.device)
    outs = []
    for _ in range(k):
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit")
    ap.add_argument("--answerable", default="data/seed_*.jsonl")
    ap.add_argument("--k", type=int, default=5, help="samples per problem")
    ap.add_argument("--out", default="data/error_mining.json")
    ap.add_argument("--judge-provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap #problems (0=all)")
    args = ap.parse_args()

    problems = load_answerable(args.answerable)
    if args.limit:
        problems = problems[:args.limit]
    print(f"[error_mine] {len(problems)} answerable problems, k={args.k}")

    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=1024, load_in_4bit=True, dtype=None)
    FastLanguageModel.for_inference(model)

    acc = {}
    for i, p in enumerate(problems):
        correct = 0
        for raw in generate_samples(model, tok, p.statement, args.k):
            resp = parse_model_output(raw)
            if resp.decision == "answer" and judgemod.answer_correct(
                    p, resp.answer, provider=args.judge_provider,
                    model=args.judge_model):
                correct += 1
        acc[p.id] = correct / args.k
        print(f"[{i+1}/{len(problems)}] {p.id}: acc={acc[p.id]:.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=2)
    mean = sum(acc.values()) / max(1, len(acc))
    print(f"\n[error_mine] wrote {args.out}  (mean base accuracy {mean:.2f})")
    print("[error_mine] next: python -m src.data.build_dataset "
          f"--error-mining {args.out}")


if __name__ == "__main__":
    main()
