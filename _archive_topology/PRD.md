# PRD — Selective-Prediction SLM for Point-Set Topology

**Owner:** Saadhya Vijayvargiya
**Status:** Draft v0.2
**Last updated:** 2026-07-08

---

## 1. Summary

Fine-tune a small, open-source Qwen model (QLoRA) into a **point-set topology**
specialist whose distinguishing skill is **knowing when *not* to answer**. The
model should solve well-posed problems it is competent on, and — more
importantly — reliably decline problems that are **ill-posed (false or
underspecified), out of scope, or beyond its ability**, instead of
hallucinating a confident wrong answer.

This is a **selective-prediction** project, not a raw-accuracy project. The
target is an *inverse skill* that frontier models are not specifically trained
for and that is winnable in a narrow domain, whereas raw problem-solving on
frontier math benchmarks is already saturating and not a realistic place to win.

**One-line success criterion:** at matched or higher answer-accuracy on
answerable problems, achieve a **substantially lower False Confidence Rate**
(confident-but-wrong / should-have-declined) than the frontier baseline on a
held-out point-set-topology test set.

---

## 2. Locked Decisions (was §10 Open Questions)

| # | Decision | Value |
|---|---|---|
| Domain | **Point-set / metric-space topology** (Munkres-level) | keeps the "topology" theme but is tractable + verifiable, with crisp load-bearing hypotheses (Hausdorff, compact, connected, metrizable, …). See §11 risk #1 for the tradeoff vs. differential topology. |
| Base model | **Qwen2.5-Math-7B-Instruct** (primary); Qwen2.5-3B for the dev loop | math-pretrained prior; 3B first to shake out the pipeline cheaply |
| GPU | **T4 (free) first, L4 (Pro) as upgrade path** | dev loop + first real runs on T4; move 7B to L4 if wall-clock/VRAM demands |
| Dataset size | **~2k curated** for v1 | quality > quantity; expand only if metrics demand |
| Verification | **generate → frontier auto-verify → ~5–10% human spot-check** | NOT full hand-verification (see §5.3) |
| Baseline | **lock protocol now, pick exact model at M4** | choose whichever frontier model shows the *highest* false-confidence rate (easiest to beat) via the M0.5 probe |
| LLM-judge | **different provider than the generator** | you have both Anthropic + OpenAI keys → generate with one, judge with the other (avoids circularity) |
| "Too-hard" bin | **dropped for v1** | model-relative label is unstable under fine-tuning (see §11 risk #6); revisit later |

---

## 3. Goals & Non-Goals

### Goals
- Fine-tune a Qwen SLM (3B dev, 7B primary) with **QLoRA**.
- Build a clean dataset of **answerable + deliberately ill-posed** point-set-
  topology problems with unambiguous ground truth.
- Train **refusal-aware** (R-Tuning-style): declining when unqualified is scored
  as *much better* than answering wrong.
- Beat a frontier baseline on **selective-prediction metrics** (§5), not on raw
  solve rate.
- Do all training within **Google Colab** budgets (T4 first, L4 later).

### Non-Goals
- General-purpose chatbot quality.
- Multimodal / vision / speech.
- RLHF; training a foundation model from scratch.
- Anything outside point-set topology.
- **Beating frontier models at actually *solving* hard problems.** We beat them
  at *declining* correctly.

---

## 4. Background & Thesis (SPOV)

The consensus race is raw problem-solving, and frontier models are winning it.
Even a cheap SLM that solved well wouldn't be *enough* better to matter. The
non-consensus, winnable target is the inverse skill — **selective prediction** —
in a narrow subdomain where ground-truth training data can be constructed by
ablating hypotheses from known-good problems.

Supporting evidence:
- Small fine-tuned models have beaten frontier models on *narrow* tasks
  (specialization > scale when the domain is tight).
- ~2k high-quality curated CoT examples can move the needle more than raw scale.
- R-Tuning shows refusal-aware tuning improves calibration and transfers as a
  meta-skill.
- Automatic Item Generation (AIG): item models with varied "radicals" generate
  many content-equivalent items with sound psychometric properties — here
  applied to make *un*answerable items via hypothesis ablation.

---

## 5. Approach

### 5.1 Base model
- **Primary:** Qwen2.5-Math-7B-Instruct (Apache-2.0 — verify per checkpoint).
- **Dev loop:** Qwen2.5-3B-Instruct (fast, T4-friendly; also tests the "small
  wins" thesis).
- Download from Hugging Face. Confirm license before any release.

### 5.2 QLoRA configuration (starting point)
- 4-bit **NF4** quant, double quantization, compute dtype **fp16** (T4) / **bf16** (L4).
- LoRA: `r=16–32`, `alpha=16–32`, `dropout=0.05`, target = all linear proj
  (`q,k,v,o,gate,up,down`).
- Optimizer `paged_adamw_8bit`, LR `2e-4` (drop to `1e-4` + grad-clip on T4 if
  loss spikes), cosine schedule, warmup ~3%.
- **Gradient checkpointing ON.**
- Max seq len 1024 (raise to 2048 on L4 if proofs truncate).
- Effective batch via gradient accumulation (per-device batch 1–2).
- **Unsloth** recommended: ~2× throughput, ~40% less VRAM on T4/L4.

### 5.3 Dataset construction (the project lives or dies here)
1. **Source well-posed problems + ground truth** from canonical references:
   Munkres (*Topology*), plus point-set chapters of Willard, Steen & Seebach
   (*Counterexamples in Topology* — invaluable for ablation counterexamples),
   and qualifying-exam sets. These already have solutions → **no verification
   needed for the answerable bin.**
2. **Generate ill-posed variants via hypothesis ablation** — remove a load-
   bearing assumption (Hausdorff, compactness, connectedness, metrizability,
   second-countability, boundary/closure conditions) or delete given data so the
   statement becomes **false** or **underspecified**. Clean label: *correct
   action = decline + name the missing hypothesis (+ cite the witnessing
   counterexample)*.
3. **Out-of-scope items:** adjacent-but-outside problems (algebraic topology
   with no point-set content, pure analysis, etc.) → decline.
4. Target size: **~2k curated** (quality > quantity).

**Verification pipeline (you do NOT hand-verify everything):**
- **Auto-verify** every ablation with a frontier model: "Is this now genuinely
  ill-posed? Which hypothesis is missing, and what explicit counterexample
  witnesses the failure?" Reject any ablation with no witnessing counterexample
  (guards risk #2).
- **Human spot-check ~5–10%** (a few hours, once) to trust the generator — not
  the full set.
- Use a **different provider** for generation vs. later judging.

### 5.4 Training objective (refusal-aware)
SFT on chat-formatted `(problem → response)` where response is **structured**
(see §5.6) and is either a correct solution (answerable) or a structured decline
that **names why** (false hypothesis / underspecified / out of scope). Curriculum
weights a correct decline **>** a wrong answer, rewarding calibrated abstention
over hallucinated iteration.

### 5.5 Evaluation
- Held-out stratified test set, 3 bins: **answerable**, **ill-posed
  (false/underspecified)**, **out-of-scope**. Never seen in training. Prefer
  ablation-generated novel items over verbatim textbook problems (contamination —
  risk #3).
- Frontier baseline run under identical prompt protocol.
- Report the full metric table + risk–coverage curve **with confidence
  intervals** (small-sample noise — risk #10).
- Grade decline-reason quality with an **LLM-judge (different provider) + rubric**;
  hand spot-check a sample.

Metrics (primary axis = selective prediction):

| Metric | Definition | Direction |
|---|---|---|
| **False Confidence Rate (FCR)** | P(confident answer \| should have declined) | **↓ primary** |
| **Answer accuracy** | correctness on *answerable* bin, among attempted | ↑ (must not collapse) |
| **Refusal precision/recall** | correctly declining ill-posed / OOS | ↑ |
| **Coverage** | fraction of all questions answered | tuned, not maximized |
| **Risk** | error rate among answered | ↓ |
| **AURC** | area under risk–coverage curve | ↓ |
| **ECE** | expected calibration error | ↓ |
| **Over-refusal rate** | declining *answerable* problems | ↓ (guardrail) |

**Anti-gaming guardrail:** refusing everything zeroes FCR but destroys coverage.
The win is the *joint* improvement: lower FCR **at matched-or-better answer
accuracy and acceptable coverage** vs. the frontier baseline.

### 5.6 Structured output format (operationalizes "confidence")
Every model response — baseline and fine-tuned — is forced into a strict schema
so "answer vs. decline" and "confidence" are unambiguous (risks #8, #9):

```json
{
  "decision": "answer" | "decline",
  "confidence": 0.0-1.0,
  "reason": "…(for decline: which hypothesis is missing / why out of scope)…",
  "answer": "…(only when decision=answer)…"
}
```

FCR is then defined operationally as: `decision=answer` **and** `confidence ≥ τ`
on an item whose gold label is `decline`, where τ is a fixed threshold reported
alongside results.

**Confidence targets (calibration).** SFT targets do NOT use a constant
confidence (that would make ECE / risk-coverage degenerate). Two sources:
- *Heuristic (v1 default):* answerable items get a difficulty tier by
  solution-length tertiles → confidence 0.92 / 0.84 / 0.74 (easy/medium/hard);
  declines get 0.88–0.90. Gives graded targets so the model learns to modulate.
- *Principled (preferred):* `error_mine.py` runs the base model k times over the
  answerable pool; the fraction graded correct becomes each item's target
  confidence — true calibration to the model's actual ability. This revives the
  error-mining idea (PRD §5.3 step 4) for *calibration* without reintroducing the
  unstable "too-hard → decline" bin.

---

## 6. Tech Stack
- **Local (data gen / eval / probe):** `anthropic`, `openai`, `pydantic`,
  `numpy`, `scikit-learn` (metrics), `matplotlib` (risk–coverage curve).
- **Colab (training):** `transformers`, `peft`, `bitsandbytes`, `trl`
  (SFTTrainer), `accelerate`, `datasets`, **Unsloth**, `wandb`.
- Flash Attention 2 **only on L4/Ampere+** — *not* on T4.
- Checkpoint to Google Drive (Colab sessions are ephemeral — §7).

---

## 7. Hardware Feasibility & Training-Time Estimates

Representative run: **~2,000 examples × ~1,000 tokens × 3 epochs ≈ 6M training
tokens.** Order-of-magnitude estimates, not benchmarks. "Unsloth" assumes ~2×.

| Model | GPU | VRAM fit? | Vanilla QLoRA | With Unsloth |
|---|---|---|---|---|
| Qwen 3B | T4 (16GB) | easy | ~1.5–2 h | ~0.8–1.2 h |
| **Qwen 7B** | **T4 (16GB)** | **tight-ok** | **~3–4 h** | **~2–2.5 h** |
| Qwen 3B | L4 (24GB) | easy | ~0.6–1 h | ~0.4–0.6 h |
| **Qwen 7B** | **L4 (24GB)** | **comfortable** | **~1.5–2 h** | **~1–1.5 h** |

**Colab reality:**
- Free T4: ~12h ceiling, ~90-min idle disconnect, variable availability →
  **checkpoint to Drive + resume**.
- T4 is **fp16 only** (no bf16 on Turing) → watch for loss spikes/NaN; lower LR +
  grad clip if needed. L4 supports bf16 + FA2 → prefer when available.
- Extra time not in table: base-model download (~15 GB for 7B), dataset
  generation (API calls), eval sweeps. Budget separately.

**Recommendation:** dev-loop on **3B/T4**, first real train on **7B/T4 +
Unsloth** (~2–2.5h), move to **7B/L4** if you upgrade to Pro.

---

## 8. Milestones

| # | Milestone | Output |
|---|---|---|
| M0 | Repo + Colab scaffold, base model loads in 4-bit, generates 1 sample | notebook runs end-to-end on seed data |
| **M0.5** | **Tier-1 frontier probe** — measure frontier FCR on ~30 hand-made ill-posed items | go/no-go on domain difficulty *before* full dataset build |
| M1 | Dataset v1: answerable + ablated ill-posed + OOS, auto-verified, spot-checked | ~2k curated items, stratified splits |
| M2 | First QLoRA fine-tune (3B then 7B), refusal-aware | trained adapter + eval on 3 bins |
| M3 | Frontier baseline comparison on selective-prediction metrics | metric table + risk–coverage curve + CIs |
| M4 | Iterate (data + objective + size); lock exact baseline model | final model + report |

---

## 9. Runbook (see README.md for commands)
1. `M0.5` probe → confirm there's headroom.
2. Build dataset (`build_dataset.py`) → spot-check.
3. Push data to Drive → open Colab notebook → train 3B → eval.
4. Train 7B → eval → baseline compare.

---

## 10. Tech-stack env
- `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` (proxy) — set.
- `OPENAI_API_KEY` — set (use as the *other* provider for judge vs. generator).
- `src/llm.py` reads these; supports both providers.

---

## 11. Risks & Mitigations (tiered by threat to the thesis)

### Tier 1 — threaten the whole thesis
| # | Risk | Mitigation |
|---|---|---|
| 1 | **Frontier already good at point-set topology → little headroom to win.** (cost of pivoting off differential topology) | Target the *ill-posed* slice specifically (models fail at spotting *missing* hypotheses even when strong solvers); keep hard sub-topics (quotient/product topologies, nets, non-metrizability). **Probe this at M0.5 before building the dataset.** |
| 2 | **Ablations that aren't actually load-bearing** (theorem still true under weaker hyp → poisoned "decline" labels) | Require an **explicit witnessing counterexample** for every ill-posed item; frontier auto-verify + spot-check. |
| 3 | **Contamination / memorization** (Munkres solutions in pretraining; test set leakage) | Heavily paraphrase; test on **ablation-generated novel variants**, not verbatim problems. |

### Tier 2 — data & labeling
| # | Risk | Mitigation |
|---|---|---|
| 4 | **Shortcut learning** (model detects "a clause is missing" format, not the math) | Normalize/paraphrase both classes so surface form doesn't leak the label. |
| 5 | **Muddy decline taxonomy** (false vs. underspecified vs. OOS conflated) | Clean 3-way label with distinct decline reasons. |
| 6 | **"Too-hard → decline" bin is model-relative & unstable** under fine-tuning | **Dropped for v1;** revisit only if needed. |
| 7 | **LLM-judge circularity** (same model generates, judges, is baseline) | Use **different providers** for generation vs. judging (Anthropic ↔ OpenAI). |

### Tier 3 — measurement (easy to fool yourself)
| # | Risk | Mitigation |
|---|---|---|
| 8 | **Defining "confident" is non-trivial** | Structured output with explicit `confidence`; FCR uses a fixed threshold τ (§5.6). |
| 9 | **Parsing answer-vs-decline from free text is flaky** | Force structured/JSON output. |
| 10 | **Small-sample noise** (~50–100 items/bin) | Size test set adequately; report confidence intervals. |
| 11 | **Metric gaming** (FCR drops by refusing more) | Always report full risk–coverage curve; judge the *joint* win. |

### Tier 4 — engineering / Colab
| # | Risk | Mitigation |
|---|---|---|
| 12 | **Dependency hell** (bitsandbytes/unsloth/transformers) | Pinned `requirements-colab.txt`; Unsloth's Colab image. |
| 13 | **T4 fp16 instability** (loss spike/NaN) | Lower LR, grad clip; 3B dev-loop first. |
| 14 | **Session disconnects / seq-len squeeze** | Checkpoint to Drive + resume; cut seq len on T4 if OOM. |
| 15 | **License of base checkpoint** | Verify HF license per checkpoint before release. |
