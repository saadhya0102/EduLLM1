# Answer-distribution results (DBE-KT22 (real))

**Test items:** 31  |  **Train items:** 123

Lower is better for every metric. **MAE** = avg percentage-points off (primary). **Brier** = squared error (penalizes large misses).

| Rank | Predictor | Model | MAE | Brier | TV | KL | JS | ΔMAE vs mean |
|---|---|---|---|---|---|---|---|---|
| 1 | sonnet | claude-sonnet-5 | 0.1440 | 0.0413 | 0.2880 | 0.3237 | 0.0843 | -0.1090 |
| 2 | opus | claude-opus-4-8 | 0.1473 | 0.0381 | 0.2946 | 0.2972 | 0.0792 | -0.1057 |
| 3 | trained | Qwen/Qwen2.5-1.5B-Instruct + LoRA | 0.1984 | 0.0700 | 0.3968 | 0.5334 | 0.1278 | -0.0546 |
| 4 | haiku | claude-haiku-4-5 | 0.1996 | 0.1047 | 0.3992 | 0.7726 | 0.1596 | -0.0533 |
| 5 | mean | — | 0.2529 | 0.0951 | 0.5059 | 0.6745 | 0.1750 | +0.0000 |
| 6 | uniform | — | 0.2603 | 0.0989 | 0.5207 | 0.7057 | 0.1821 | +0.0074 |

**Best by MAE:** `sonnet`.

Trained small model BEATS the mean baseline on MAE (0.1984 vs 0.2529, Δ=-0.0546).
- vs sonnet: trained loses to it (0.1984 vs 0.1440, Δ=+0.0544)
- vs haiku: trained beats it (0.1984 vs 0.1996, Δ=-0.0012)
- vs opus: trained loses to it (0.1984 vs 0.1473, Δ=+0.0511)
