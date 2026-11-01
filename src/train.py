"""Fine-tune Qwen2.5 with LoRA to predict answer-choice distributions.

Native formulation (see src/lp.py): the model's next-token probabilities over
the option letters, renormalized, are the predicted distribution. Loss = soft
cross-entropy / KL between that and the empirical human distribution.

Runs on the RTX 4070 (8 GB). 1.5B in bf16 + LoRA fits comfortably; add
--load-4bit only if you go bigger.

    python -m src.train --train data/dataset_train.jsonl        # real data
    python -m src.train --train data/synth_train.jsonl --epochs 3   # dev/smoke
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time

from src.schema import read_items
from src import lp
from src.progress import format_duration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/dataset_train.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="out/qwen1.5b-adapter")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument(
        "--init-adapter", default="",
        help="continue training from these LoRA weights with a fresh optimizer")
    ap.add_argument("--options", type=int, default=0,
                    help="required option count; 0 infers it from the data")
    ap.add_argument("--max-items", type=int, default=0,
                    help="deterministically cap usable training items; 0 keeps all")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint under --out")
    ap.add_argument(
        "--save-steps", type=int, default=0,
        help="save a resumable checkpoint every N optimizer steps; 0 saves each epoch")
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument(
        "--control-file", default="",
        help="JSON control file used to request a pause at a safe checkpoint")
    args = ap.parse_args()

    if not os.path.isfile(args.train):
        raise SystemExit(f"missing {args.train} (run make_synthetic or data_prep first)")

    import torch
    import torch.nn.functional as F
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer, TrainerCallback)
    from peft import LoraConfig, PeftModel, get_peft_model

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible (check nvidia-smi / WSL2).")
    bf16 = torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"                       # so logits[:, -1] is the answer slot
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kw = dict(device_map="auto")
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(torch.bfloat16 if bf16 else torch.float16))
    else:
        load_kw["torch_dtype"] = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.base, **load_kw)
    if args.load_4bit:                              # QLoRA: stabilize the 4-bit base
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    if args.init_adapter:
        if not os.path.isfile(
                os.path.join(args.init_adapter, "adapter_model.safetensors")):
            raise SystemExit(f"missing initial adapter: {args.init_adapter}")
        model = PeftModel.from_pretrained(
            model, args.init_adapter, is_trainable=True)
        print(f"initialized LoRA weights from {args.init_adapter}", flush=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    # Fit the 8 GB card: gradient checkpointing trades compute for memory.
    # enable_input_require_grads() is required for checkpointing + PEFT/LoRA.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    items = read_items(args.train)
    if args.options:
        items = [it for it in items if it.k() == args.options]
    option_counts = sorted({it.k() for it in items})
    if len(option_counts) != 1:
        raise SystemExit(
            "training batches require one option count; pass --options K "
            f"(found {option_counts or 'none'})")
    k_options = option_counts[0]
    letter_ids = lp.letter_token_ids(tok, k_options)
    print(f"letter token ids ({','.join(chr(65+i) for i in range(k_options))}):",
          letter_ids)

    def to_example(it):
        ids = lp.prompt_ids(tok, it)
        if len(ids) > args.max_len:
            return None
        return {"input_ids": ids, "target": it.normalized()}

    rng = random.Random(args.seed)
    items = sorted(items, key=lambda x: x.id)
    rng.shuffle(items)
    data = [ex for it in items if (ex := to_example(it)) is not None]
    dropped = len(items) - len(data)
    if args.max_items:
        data = data[:args.max_items]
    if not data:
        raise SystemExit("no usable training items after option/length filtering")
    print(f"training on {len(data)} {k_options}-option items "
          f"({dropped} over {args.max_len} tokens dropped)")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn = [], []
        for b in batch:                              # left-pad
            pad = maxlen - len(b["input_ids"])
            input_ids.append([tok.pad_token_id] * pad + b["input_ids"])
            attn.append([0] * pad + [1] * len(b["input_ids"]))
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
            "target": torch.tensor([b["target"] for b in batch], dtype=torch.float),
        }

    lids = torch.tensor(letter_ids)

    class DistTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            target = inputs.pop("target")
            out = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"])
            last = out.logits[:, -1, :]                       # [B, vocab]
            letter_logits = last[:, lids.to(last.device)]      # [B, 4]
            logp = F.log_softmax(letter_logits, dim=-1)
            loss = -(target.to(logp.device) * logp).sum(-1).mean()  # soft CE = KL+const
            return (loss, out) if return_outputs else loss

    class Progress(TrainerCallback):
        paused = False
        pause_announced = False

        def pause_requested(self):
            if not args.control_file or not os.path.isfile(args.control_file):
                return False
            try:
                with open(args.control_file, encoding="utf-8") as f:
                    request = json.load(f)
                if request.get("action") != "pause_at_checkpoint":
                    return False
                return time.time() >= float(request.get("not_before_unix", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return False

        def on_train_begin(self, a, s, c, **k):
            self.t0 = time.time()
            self.every = max(1, (s.max_steps or 1) // 100)
            print(f"[progress] {s.max_steps} steps; Ctrl+C to stop.", flush=True)

        def on_step_end(self, a, s, c, **k):
            if self.pause_requested() and not self.pause_announced:
                self.pause_announced = True
                print(
                    "[control] pause requested; stopping after the next "
                    "checkpoint is safely written", flush=True)
            if s.global_step % self.every and s.global_step != s.max_steps:
                return
            el = time.time() - self.t0
            eta = el / s.global_step * (s.max_steps - s.global_step) if s.global_step else 0
            print(f"[progress] {100*s.global_step/s.max_steps:5.1f}%  "
                  f"{s.global_step}/{s.max_steps} steps | "
                  f"part: epoch {s.epoch:.2f}/{a.num_train_epochs:g} | "
                  f"elapsed {format_duration(el)} | ETA {format_duration(eta)}",
                  flush=True)

        def on_save(self, a, s, c, **k):
            if not (self.pause_announced or self.pause_requested()):
                return
            self.paused = True
            c.should_training_stop = True
            if args.control_file:
                with open(args.control_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "action": "paused",
                        "checkpoint_step": s.global_step,
                        "acknowledged_unix": time.time(),
                    }, f, indent=2)
            print(
                f"[control] paused safely at checkpoint step {s.global_step}",
                flush=True)

    progress_callback = Progress()
    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=10, bf16=bf16, fp16=not bf16, max_grad_norm=1.0,
        seed=args.seed, data_seed=args.seed,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps or 500,
        save_total_limit=max(2, args.save_total_limit), report_to="none",
        remove_unused_columns=False, label_names=["target"])

    trainer = DistTrainer(model=model, args=targs, train_dataset=data,
                          data_collator=collate,
                          callbacks=[progress_callback])
    checkpoints = sorted(
        glob.glob(os.path.join(args.out, "checkpoints", "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume_from = checkpoints[-1] if args.resume and checkpoints else None
    if resume_from:
        print(f"resuming from {resume_from}", flush=True)
    trainer.train(resume_from_checkpoint=resume_from)
    if progress_callback.paused:
        print(
            "training paused with a complete checkpoint; rerun with --resume "
            "when ready", flush=True)
        raise SystemExit(75)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_file": args.train,
            "base": args.base,
            "n_items": len(data),
            "options": k_options,
            "max_len": args.max_len,
            "epochs": args.epochs,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "learning_rate": args.lr,
            "lora_r": args.lora_r,
            "load_4bit": args.load_4bit,
            "init_adapter": args.init_adapter or None,
            "seed": args.seed,
            "save_steps": args.save_steps,
        }, f, indent=2)
    print(f"saved adapter -> {args.out}")


if __name__ == "__main__":
    main()
