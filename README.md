# EduLLM1 — Predicting Student Answer Distributions on MCQs

Fine-tune a small LLM (**Qwen2.5-1.5B**, LoRA) to predict the **distribution of
which option students pick** on a multiple-choice question (% choosing A/B/C/D),
matching real student data — and **beat a frontier model (Sonnet)** at it.

**Why small can win:** frontier models are RLHF'd toward the *correct* answer, so
they mis-calibrate the *percentages* students actually pick. Calibration to a
human distribution is an axis scale doesn't help — same idea as Maia Chess. See
`PRD.md`.

> Pivoted from an earlier topology selective-prediction attempt (archived in
> `_archive_topology/`).

## Layout
```
src/
  llm.py           # provider client (Anthropic/OpenAI, gateway-aware)
  schema.py        # Item: question, options, empirical dist, correct_index
  lp.py            # shared letter-probability prompt (train == inference)
  metrics.py       # MAE / Brier / TV / KL / JS
  make_synthetic.py# synthetic dataset for dev before real data
  data_prep.py     # build Items from FoundationalASSIST (gated)
  baselines.py     # uniform, mean-distribution, Sonnet few-shot
  train.py         # Qwen2.5-1.5B letter-logprob KL LoRA (RTX 4070)
  predict_hf.py    # load base+adapter, predict distributions
  eval.py          # run/compare predictors -> metric table
data/              # synth_*.jsonl (now); dataset_*.jsonl (after data_prep)
```

## Setup
```bash
pip install -r requirements-local.txt      # data prep, baselines, eval, Sonnet
# GPU box (WSL2): pip install torch --index-url .../cu121 && pip install -r requirements-gpu.txt
```
API creds come from env (`ANTHROPIC_AUTH_TOKEN`+`ANTHROPIC_BASE_URL`, `OPENAI_API_KEY`).

## Runbook

### Works today (synthetic — no data access needed)
```bash
python -m src.make_synthetic --n 400 --out-prefix data/synth
python -m src.eval --test data/synth_test.jsonl --train data/synth_train.jsonl \
    --predictors uniform,mean,sonnet          # add 'sonnet' to hit the API
```

### Get the real data (one-time, YOUR action — it's gated)
1. Log in at huggingface.co; open `datasets/ASSISTments/FoundationalASSIST` → **Agree and access**.
2. Create a read token; `setx HF_TOKEN "hf_..."` (reopen shell) or `huggingface-cli login`.
3. Confirm schema, then build:
```bash
python -m src.data_prep --peek            # print CSV headers -> fix COL{} in data_prep.py
python -m src.data_prep --build           # -> data/dataset_{train,test}.jsonl
```

### Headroom probe (M0.5 — go/no-go)
```bash
python -m src.eval --test data/dataset_test.jsonl --train data/dataset_train.jsonl \
    --predictors mean,sonnet
```
If Sonnet barely beats `mean`, a fine-tune has room to win.

### Train + evaluate (GPU box)
```bash
python -m src.train --train data/dataset_train.jsonl        # ~5-15 min on 4070
python -m src.eval  --test data/dataset_test.jsonl --train data/dataset_train.jsonl \
    --predictors mean,sonnet,hf --adapter out/qwen1.5b-adapter
```
**Win = fine-tuned `hf` beats both `mean` and `sonnet` on MAE/KL.**

## Staged 1.5B run: 1,000 items first (recommended)

The fastest path to a complete real-data result now builds 1,000 training
items and trains Qwen2.5-1.5B for two epochs:
```powershell
.\run_staged_1k.ps1 -WithOpus
```

Expected first-run breakdown:
- download and prepare 1,000 items: **1–4 hours**
- train 1.5B for two epochs at 1,024 tokens: **1–2 hours**
- Opus and local evaluation: **0.55–1.6 hours**
- complete run: **about 3.5–8 hours**, typically around **5–6 hours**

Inspect progress from another terminal:
```powershell
.\run_staged_1k.ps1 -Stage status
```

Request a safe pause at the next training checkpoint:
```powershell
.\run_staged_1k.ps1 -Stage pause-next
```

Or schedule the first safe checkpoint after a delay:
```powershell
.\run_staged_1k.ps1 -Stage pause-next -AfterMinutes 60
```

Two checkpoints are retained. A reversible rollback archives the newest one
and makes the preceding checkpoint active for the next resume:
```powershell
.\run_staged_1k.ps1 -Stage rollback
```

Resume with the original run command. Later, `run_staged_2k.ps1` builds the
additional pool from cached source files, selects 1,000 item IDs not used in
stage 1, adds 250 old-item replay examples, initializes from the stage-1 LoRA,
and keeps the original validation/test sets fixed. Existing Opus predictions
are reused because the test set does not change.

## Prepared 3B overnight run

The overnight profile uses recent ENADE releases only, with a deterministic
**2,000 train / 100 validation / 250 test** temporal split. It trains the 3B
model for 2 epochs at 768 tokens and skips the much larger ENEM downloads.

Inspect it without starting downloads:
```powershell
.\run_overnight.ps1 -Stage dry-run
```

Run the complete resumable experiment with Opus:
```powershell
.\run_overnight.ps1 -WithOpus
```

Expected first-run time:
- data download + preparation: **2–6 hours**
- Qwen2.5-3B QLoRA training: **2.5–4.5 hours**
- Opus + local evaluation: **0.65–1.8 hours**
- total: **5.5–12.5 hours**, typically around **7–9 hours**

INEP server latency can exceed the estimate, but downloads and training
checkpoints resume. Outputs use the `brazil2000` prefix under `data/`, `out/`,
`results/`, and `runs/`.

### Stopping, resuming, and progress

Press `Ctrl+C` once to stop. The runner records the interrupted stage and shuts
down its child process. Resume by running the exact same command again:
```powershell
.\run_overnight.ps1 -WithOpus
```
Already completed stages are skipped automatically. During a run, the console
shows the pipeline part, current subtask, percentage, elapsed time, and an ETA
calculated from the measured rate. The ETA is deliberately recalculated as more
work completes, so the first few readings may move substantially. In another
terminal, inspect the last recorded progress with:
```powershell
.\run_overnight.ps1 -Stage status
```

Resume granularity differs by stage:
- data: finished downloads and completed years are cached; interrupted HTTP
  downloads continue from their `.part` file when the server supports ranges
- training: a full Trainer checkpoint is kept every 25 optimizer steps; a stop
  can repeat at most the work since the latest checkpoint
- Opus: every completed API prediction is saved, so paid calls are not repeated
- local evaluation: this shorter stage restarts from its beginning

Do not use `-Force` to resume. It intentionally reruns completed work. Closing
the terminal or powering off is less graceful than `Ctrl+C`, but the latest
completed download, year cache, training checkpoint, or Opus item still remains.

Training time is approximately linear in epoch count. For this profile, one
epoch is roughly **1.25–2.25 hours** after model startup: 1 epoch is about half
the two-epoch training time, while 3 epochs adds about 50%. Epochs do not change
download, data-preparation, Opus, or evaluation time. Two epochs is the default
because one is more likely to underfit, while a third costs substantial time
and may offer only a small gain.

## Optional full 8,000-item run

The Brazil 8k pipeline builds option distributions from official ENADE and
ENEM microdata (`DS_VT_ESC_*` / `TX_RESPOSTAS_*`) and released five-choice
exam content. It does not generate questions, choices, labels, or responses.
The deterministic temporal split is **8,000 train / 200 validation / 400 test**.
Older INEP archives are internally 7z/RAR despite their filenames; the pipeline
uses the locally installed `C:\Program Files\7-Zip\7z.exe` to normalize them.

Inspect the download manifest without starting the run:
```powershell
.\run_8k.ps1 -Stage dry-run
```

Run the complete resumable experiment, including the Opus baseline:
```powershell
.\run_8k.ps1 -WithOpus
```

Individual stages are `data`, `train`, `frontier`, and `evaluate`. Check cached
progress with:
```powershell
.\run_8k.ps1 -Stage status
```

Expected first-run time on the RTX 4070 Laptop GPU (8 GB):
- download + PDF/data preparation: **18–48 hours**
- Qwen2.5-3B QLoRA, 3 epochs at 1,024 tokens: **12–24 hours**
- Opus predictions on 400 test items: **0.5–1.5 hours**
- local inference + temperature selection: **0.5–2 hours**
- full first run: **32–76 hours**; interrupted downloads/checkpoints resume

The training range is anchored to the prior local 3B run (800 sample-passes in
665 seconds at a 164-token mean). A public ENADE sample averages about 401
tokens; linear scaling predicts roughly 13.5 hours for 24,000 sample-passes,
with the wider range covering QLoRA overhead and longer items.

Outputs are `data/brazil8k_{train,val,test}.jsonl`,
`out/brazil8k/qwen3b`, and `results/brazil8k/`.
Allow roughly **25–40 GB** of local disk for cached archives, PDFs, model
weights, checkpoints, and results. The machine currently has about 311 GB free.

## Status
DBE-KT22 training and frontier comparisons are complete. Both the recommended
overnight profile and the larger 8k profile are configured; neither has started
its full data download or training run.
