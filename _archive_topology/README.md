# EduLLM1 — Selective-Prediction SLM for Point-Set Topology

Fine-tune a small Qwen model (QLoRA) that **knows when *not* to answer**: it
solves well-posed point-set topology problems and reliably **declines** ill-posed
/ out-of-scope ones instead of hallucinating. See `PRD.md` for the full spec.

**The win:** lower **False Confidence Rate** than a frontier baseline, at
matched-or-better answer accuracy and acceptable coverage.

---

## Layout

```
PRD.md                     # spec (v0.2) — read this first
configs/train.yaml         # hyperparameters (reference)
notebooks/train_qlora.ipynb# Colab QLoRA training (T4-first, 3B dev / 7B prod)
SETUP_LOCAL.md             # local-GPU (RTX 4070 8GB) WSL2 setup + train guide
src/train.py               # local QLoRA training (Unsloth, 8GB-tuned)
data/
  seed_*.jsonl             # 145 curated answerable problems across 8 files
  probe_items.jsonl        # 30 items for the M0.5 frontier probe
src/
  llm.py                   # provider-agnostic client (Anthropic + OpenAI)
  schema.py                # Problem record + taxonomy + response format
  prompts.py               # shared inference prompt + training-target builder
  probe.py                 # M0.5 Tier-1 frontier probe
  data/
    generate.py            # ablation generator (answerable -> ill-posed) + OOS
    verify.py              # frontier auto-verifier (drops non-load-bearing)
    error_mine.py          # base-model pass -> calibrated confidence (optional)
    build_dataset.py       # combine + balance + calibrate + split -> chat format
    spotcheck.py           # sample ~10% for a one-time human spot-check
    paraphrase.py          # rewrite answerable TEST items (reduce contamination)
  eval/
    metrics.py             # FCR, AURC, ECE, risk-coverage, bootstrap CIs
    judge.py               # LLM-judge (correctness + decline quality)
    run_eval.py            # drive api/hf model over test set -> metrics
    compare.py             # baseline vs fine-tuned + risk-coverage plot
```

## Setup (local — data gen / probe / eval)

```bash
pip install -r requirements-local.txt
```

Credentials are read from the environment (already set on this machine):
- `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` (proxy), or `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

Use **different providers** for generation vs. verification/judging (PRD risk #7).

---

## Runbook

### Step 0 — M0.5 probe (DO THIS FIRST — it's the go/no-go)
Measures whether frontier models are already good at declining point-set
problems. If their False Confidence Rate is near zero, there's no headroom and
you should harden the domain *before* building the dataset (PRD risk #1).

```bash
python -m src.probe --provider anthropic --model claude-opus-4-8 --tau 0.5
python -m src.probe --provider openai   --model gpt-4o          --tau 0.5
```
Look for **FCR ≥ ~0.3** → GO.

### Step 1 — Build the dataset (M1)
The seed base is **145 curated answerable problems** across the full point-set
syllabus, split over `data/seed_*.jsonl` (basics, continuity, connectedness,
compactness, separation/countability, metric, advanced). `generate.py` and
`build_dataset.py` glob **all** `data/seed_*.jsonl` by default.

```bash
# generate ill-posed variants from ALL seed files (provider A). --n-per 4
# on 130 seeds -> a few hundred ill-posed candidates.
python -m src.data.generate --out data/generated_raw.jsonl \
    --provider anthropic --n-per 4

# independently verify with the OTHER provider (drops non-load-bearing ablations)
python -m src.data.verify --in data/generated_raw.jsonl \
    --out data/generated_verified.jsonl --provider openai

# one-time human spot-check of ~10%
python -m src.data.spotcheck --in data/generated_verified.jsonl --frac 0.1
#   -> edit data/spotcheck_review.txt (mark y/n), then:
python -m src.data.spotcheck --score data/spotcheck_review.txt

# combine, balance (caps decline share), calibrate confidence, split, emit chat
python -m src.data.build_dataset --test-frac 0.2

# (optional, reduces contamination on the test set — PRD risk #3)
python -m src.data.paraphrase --in data/dataset_test.jsonl \
    --out data/dataset_test.jsonl --provider anthropic
```
Outputs: `data/dataset_train.jsonl`, `data/dataset_test.jsonl`, `data/train_chat.jsonl`.

**Calibrated confidence (optional, best done after M2 base run):** for true
confidence calibration instead of the length-based difficulty heuristic, run the
base model over the answerable pool and feed its empirical accuracy back in:
```bash
python -m src.data.error_mine --base Qwen/Qwen2.5-Math-7B-Instruct --k 5 \
    --out data/error_mining.json --judge-provider openai      # on the GPU box
python -m src.data.build_dataset --error-mining data/error_mining.json
```

> **Toward the PRD's ~2k target:** 130 seeds × `--n-per 4` ≈ 300–350 items after
> balancing — a solid v1. To grow further, add more `data/seed_*.jsonl` files
> (same schema) and/or raise `--n-per`; the ablation step is the volume
> multiplier, so adding correct answerable seeds is the highest-leverage move.

### Step 2 — Train (M2)

**Local (recommended — RTX 4070 8GB):** see `SETUP_LOCAL.md` for the one-time
WSL2 setup, then:
```bash
python -m src.train --dev     # 3B smoke test (~5-10 min on current data)
python -m src.train           # 7B target  (~15-25 min on current data)
#   OOM on 7B?  python -m src.train --max-seq-len 512 --grad-accum 16
```
Adapter saves to `out/<run>-adapter`. It prints live `[progress] N% … ETA …`
lines (~every 1%), and checkpoints every 25 steps — so you can **Ctrl+C to stop
anytime** and rerun with `--resume` to pick up where you left off.

**Colab (fallback):** upload `data/train_chat.jsonl` + `data/dataset_test.jsonl`
to `MyDrive/EduLLM1/data/`, open `notebooks/train_qlora.ipynb`, set runtime to
**T4/L4**, run with `DEV=True` then `DEV=False`.

### Step 3 — Evaluate + compare (M3)
```bash
# frontier baseline (pick the weakest/easiest model here per M4)
python -m src.eval.run_eval --backend api --provider anthropic \
    --model claude-opus-4-8 --test data/dataset_test.jsonl \
    --judge-provider openai --out data/eval_baseline.json

# fine-tuned (run on Colab where the GPU + adapter live)
python -m src.eval.run_eval --backend hf \
    --base Qwen/Qwen2.5-Math-7B-Instruct --adapter out/qwen7b-math-adapter \
    --test data/dataset_test.jsonl --judge-provider openai \
    --out data/eval_finetuned.json

# compare + risk-coverage plot
python -m src.eval.compare --baseline data/eval_baseline.json \
    --model data/eval_finetuned.json --out data/risk_coverage.png
```

---

## Notes
- All model I/O uses a strict JSON response format so "answer vs decline" and
  "confidence" are unambiguous (PRD §5.6, risks #8/#9).
- The `too-hard → decline` bin is intentionally **dropped for v1** (PRD risk #6).
- Metrics report **bootstrap confidence intervals** because per-bin samples are
  small (PRD risk #10).
