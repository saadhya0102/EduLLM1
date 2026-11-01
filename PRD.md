# PRD — Predicting Student Answer *Distributions* on MCQs

**Owner:** Saadhya Vijayvargiya
**Status:** Draft v0.3 (pivot from topology selective-prediction)
**Last updated:** 2026-07-09

---

## 1. Summary

Fine-tune a small LLM (**Qwen2.5-1.5B**, LoRA) to predict, for a multiple-choice
question, the **distribution of which option students pick** — i.e. the % of
students choosing A/B/C/D — matching the *empirical* distribution from real
student responses. Compared against a **frontier baseline (Sonnet)** and a
mean-distribution baseline.

## 2. Thesis (why a small model can win — the spiky point)

A specialized small model beats a frontier model **only when capability is not
the objective.** Here it isn't: predicting *which wrong answer students pick, and
in what proportion* is a task frontier models are **structurally handicapped** on:
- They are RLHF-trained toward the *correct* answer, so they systematically
  under-weight attractive distractors.
- They can often *rank* options but **mis-calibrate the magnitudes** (the
  percentages) to a specific student population — which is exactly what the data
  teaches and the metric measures.

This is the same shape as **Maia Chess** (a small model beats a superhuman engine
at *predicting human moves*). The win is **calibration to a human distribution**,
not raw correctness — an axis scale doesn't help.

## 3. Task

- **Input:** question stem + K option texts (K usually 4).
- **Output:** probability distribution over the K options (predicted selection %).
- **Ground truth:** empirical selection fractions from real student responses.
- **Native formulation (src/lp.py):** the LM's next-token probabilities over the
  tokens {A,B,C,D}, renormalized, *are* the prediction. Training = soft
  cross-entropy / KL between that and the empirical distribution. Keeps it a
  genuine generative LLM, no regression head.

## 4. Data

**Target: FoundationalASSIST** ([HF: ASSISTments/FoundationalASSIST](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST))
— 1.7M interactions, 5k students, grades 3–8 math, **full question text + actual
first-attempt answers (incl. which wrong answer chosen)**. Open but **manually
gated** — requires HF login + accepting access + a token.

Prep (`src/data_prep.py`): keep MCQ items → aggregate first-attempt choices per
question → empirical distribution → filter to ≥50 matched responses → split.
~30% of interactions are MCQ (rest fill-in-the-blank), still ample.

**Alternatives if needed:** Cambridge CMCQRD (reading, license-gated, restrictive);
Eedi 1.4K math (proprietary); NeurIPS-2020 Eedi (no question text).

**Contamination:** low — selection *percentages* aren't memorizable.

## 5. Model & training
- **Qwen2.5-1.5B-Instruct**, LoRA (r=16), bf16 on the RTX 4070 (8 GB) — fits
  without QLoRA; `--load-4bit` optional. Ablation: try 0.5B ("how small still wins").
- Loss: soft cross-entropy over {A,B,C,D} vs. empirical distribution.
- Training time: **~5–15 min** on the 4070 (light task).

## 6. Metrics & baselines (src/metrics.py, src/eval.py)
- **MAE** (primary, interpretable), **Brier** (squared diff), **TV**, **KL**, **JS**.
- **Baselines to beat:** uniform; **global mean-distribution** (real bar); **Sonnet
  few-shot** (the frontier comparison); untuned base 1.5B.
- **Win:** fine-tuned 1.5B beats Sonnet *and* the mean baseline on held-out
  questions (lower MAE/KL), especially on the distractor mass.
- Report a **held-out-skill** split too (generalization to unseen topics).

## 7. Milestones
| # | Milestone | Output |
|---|---|---|
| M0 | Pipeline scaffold + synthetic data + baselines validated | ✅ done |
| M0.5 | **Headroom probe**: Sonnet vs mean-dist MAE on real items | go/no-go |
| M1 | Data prep from FoundationalASSIST → dataset_{train,test} | real items |
| M2 | Train Qwen2.5-1.5B letter-logprob LoRA | adapter |
| M3 | Eval: fine-tuned vs Sonnet vs mean vs base + by-skill | metric table |
| M4 | Ablations (0.5B; k-shot Sonnet; #responses threshold) | final report |

## 8. Risks
| Risk | Mitigation |
|---|---|
| **Data access gated** (all distribution datasets are) | user accepts HF gate; synthetic data unblocks dev meanwhile |
| **Headroom uncertain** — Sonnet may calibrate well few-shot | M0.5 probe *before* training; compare to mean baseline |
| Noisy targets from few responses | filter ≥50 matched responses/question |
| MCQ subset small after filtering | FoundationalASSIST is large; fall back to Cambridge if needed |
| answer_text↔option matching messy | exact-match now; may need normalization (verify at --peek) |
| Position bias makes "mean" trivial/strong | report by-skill split; the content signal is the point |

## 9. Status now
Scaffold + synthetic pipeline built and validated (baselines run end-to-end).
**Blocked on data access** (FoundationalASSIST HF gate — user action). Training
+ real-data prep + probe run once the gate is open.
