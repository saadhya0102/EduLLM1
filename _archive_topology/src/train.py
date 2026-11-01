"""Local QLoRA fine-tune for the RTX 4070 Laptop (8 GB) — the local counterpart
to notebooks/train_qlora.ipynb (PRD §5.2/§5.4).

Runs refusal-aware SFT on data/train_chat.jsonl. Uses Unsloth (required to fit a
7B model in 8 GB VRAM). Your Ada GPU supports bf16 + FlashAttention-2, so this is
actually faster per step than a free Colab T4.

Prereqs: see SETUP_LOCAL.md (WSL2 recommended). Then:
    # 3B smoke test first (fast, comfortable in 8 GB):
    python -m src.train --dev
    # then the 7B target:
    python -m src.train
    # if you hit CUDA OOM on 7B, shrink the context:
    python -m src.train --max-seq-len 512 --grad-accum 16

Adapter is saved to out/<run>-adapter. Checkpoints go to out/checkpoints/<run>;
pass --resume to continue an interrupted run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# 3B dev model and 7B target. If models/qwen2.5-math-7b-4bit exists locally
# (pre-downloaded), we use that path; otherwise Unsloth pulls it from HF.
DEV_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
PROD_LOCAL = "models/qwen2.5-math-7b-4bit"
PROD_HF = "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit"


def resolve_model(args) -> str:
    if args.model_name:
        return args.model_name
    if args.dev:
        return DEV_MODEL
    return PROD_LOCAL if os.path.isdir(PROD_LOCAL) else PROD_HF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="use Qwen2.5-3B (fast smoke test) instead of the 7B target")
    ap.add_argument("--model-name", default=None,
                    help="override the model id / local path")
    ap.add_argument("--data", default="data/train_chat.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-seq-len", type=int, default=1024,
                    help="lower to 512 if you hit CUDA OOM on 7B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.data):
        raise SystemExit(
            f"Training data not found: {args.data}\n"
            "Run the dataset build first: python -m src.data.build_dataset")

    # Heavy imports are local so the module compiles without a GPU stack.
    import time
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback

    def _fmt(s):
        s = int(s)
        return f"{s//60:d}m{s%60:02d}s"

    class ProgressCallback(TrainerCallback):
        """Prints % done + elapsed + ETA + latest loss, ~every 1% of steps, so
        you can watch progress and Ctrl+C to stop early if it's taking too long
        (checkpoints every 25 steps mean you can --resume where you left off)."""
        def on_train_begin(self, a, state, control, **kw):
            self.t0 = time.time()
            self.last_loss = float("nan")
            self.every = max(1, (state.max_steps or 1) // 100)  # ~1% granularity
            print(f"[progress] starting: {state.max_steps} optimizer steps total. "
                  f"Ctrl+C to stop; rerun with --resume to continue.", flush=True)

        def on_log(self, a, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.last_loss = logs["loss"]

        def on_step_end(self, a, state, control, **kw):
            step, total = state.global_step, (state.max_steps or 1)
            if step % self.every and step != total:
                return
            elapsed = time.time() - self.t0
            eta = elapsed / step * (total - step) if step else 0.0
            print(f"[progress] {100*step/total:5.1f}%  step {step}/{total}  "
                  f"elapsed {_fmt(elapsed)}  ETA {_fmt(eta)}  loss {self.last_loss:.4f}",
                  flush=True)

        def on_train_end(self, a, state, control, **kw):
            print(f"[progress] done in {_fmt(time.time() - self.t0)} "
                  f"({state.global_step} steps)", flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible. In WSL2, check `nvidia-smi` works.")
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    bf16 = torch.cuda.is_bf16_supported()
    model_name = resolve_model(args)
    run_name = "qwen3b-dev" if (args.dev or "3B" in model_name) else "qwen7b-math"
    print(f"GPU={gpu} ({vram:.1f} GB) bf16={bf16}")
    print(f"model={model_name}  run={run_name}  seq={args.max_seq_len} "
          f"batch={args.batch} x accum={args.grad_accum}  lr={args.lr}")

    # ---- load base model 4-bit ----
    model, tok = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,  # auto: bf16 on Ada
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,        # alpha=r is a fine default
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",  # required to fit in 8 GB
        random_state=13,
    )

    # ---- dataset -> rendered chat text ----
    ds = load_dataset("json", data_files=args.data, split="train")
    print(f"train examples: {len(ds)}")

    def to_text(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text, remove_columns=[c for c in ds.column_names if c != "text"])

    # ---- trainer ----
    targs = TrainingArguments(
        output_dir=f"{args.out}/checkpoints/{run_name}",
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not bf16,
        bf16=bf16,
        max_grad_norm=0.3,             # stability
        logging_steps=5,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        seed=13,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        report_to="wandb" if args.wandb else "none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=targs,
    )

    # Refusal-aware focus: compute loss only on the assistant response (PRD §5.4).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.add_callback(ProgressCallback())

    resume = args.resume and bool(
        glob.glob(f"{args.out}/checkpoints/{run_name}/checkpoint-*"))
    print("resuming from checkpoint" if resume else "fresh run")

    try:
        trainer.train(resume_from_checkpoint=resume)
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "\nCUDA OOM. On 8 GB try one of:\n"
            "  python -m src.train --max-seq-len 512 --grad-accum 16\n"
            "  python -m src.train --dev            # 3B instead of 7B\n")

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM: {peak:.1f} GB")

    # ---- save adapter ----
    adapter = f"{args.out}/{run_name}-adapter"
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    print(f"saved adapter -> {adapter}")

    # ---- sanity check: decline an ill-posed problem, answer a good one ----
    from src.prompts import SYSTEM_PROMPT, build_inference_prompt
    FastLanguageModel.for_inference(model)
    tests = [
        "Prove that every compact subset of a topological space is closed.",  # FALSE
        "Prove that the continuous image of a compact space is compact.",     # answerable
    ]
    for t in tests:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_inference_prompt(t)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        out = model.generate(ids, max_new_tokens=400, do_sample=False)
        print("\nQ:", t)
        print("A:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
