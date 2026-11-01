# Tuned results — test set (DBE-KT22)

Winner config (by validation MAE): **c5_7b_r16_e2**  (base Qwen/Qwen2.5-7B-Instruct, T=0.6)

Tuned small models scored at their val-selected temperature. Lower is better.

| Rank | Predictor | Model | MAE | Brier | TV | KL | JS | ΔMAE vs mean |
|---|---|---|---|---|---|---|---|---|
| 1 | sonnet | claude-sonnet-5 | 0.1440 | 0.0413 | 0.2880 | 0.3237 | 0.0843 | -0.1090 |
| 2 | opus | claude-opus-4-8 | 0.1473 | 0.0381 | 0.2946 | 0.2972 | 0.0792 | -0.1057 |
| 3 | c5_7b_r16_e2 (T=0.6) | Qwen/Qwen2.5-7B-Instruct | 0.1811 | 0.0620 | 0.3622 | 0.5022 | 0.1232 | -0.0719 |
| 4 | c4_3b_r16_e8 (T=0.95) | Qwen/Qwen2.5-3B-Instruct | 0.1821 | 0.0635 | 0.3642 | 0.4978 | 0.1198 | -0.0709 |
| 5 | c2_1p5b_r32_e12 (T=1.2) | Qwen/Qwen2.5-1.5B-Instruct | 0.1851 | 0.0713 | 0.3702 | 0.5203 | 0.1233 | -0.0678 |
| 6 | c1_1p5b_r16_e5 (T=0.95) | Qwen/Qwen2.5-1.5B-Instruct | 0.1946 | 0.0692 | 0.3892 | 0.5339 | 0.1268 | -0.0583 |
| 7 | c3_1p5b_r32_lr2 (T=1.1) | Qwen/Qwen2.5-1.5B-Instruct | 0.1972 | 0.0798 | 0.3944 | 0.5523 | 0.1318 | -0.0558 |
| 8 | haiku | claude-haiku-4-5 | 0.1996 | 0.1047 | 0.3992 | 0.7726 | 0.1596 | -0.0533 |
| 9 | mean | — | 0.2529 | 0.0951 | 0.5059 | 0.6745 | 0.1750 | +0.0000 |
| 10 | uniform | — | 0.2603 | 0.0989 | 0.5207 | 0.7057 | 0.1821 | +0.0074 |

## Temperature effect (test MAE: T=1 → best T)

| Config | base | T | val MAE (T1→best) | test MAE (T1→best) |
|---|---|---|---|---|
| c1_1p5b_r16_e5 | Qwen/Qwen2.5-1.5B-Instruct | 0.95 | 0.1821→0.1821 | 0.1948→0.1946 |
| c2_1p5b_r32_e12 | Qwen/Qwen2.5-1.5B-Instruct | 1.2 | 0.1816→0.1791 | 0.1851→0.1851 |
| c3_1p5b_r32_lr2 | Qwen/Qwen2.5-1.5B-Instruct | 1.1 | 0.1939→0.1933 | 0.1970→0.1972 |
| c4_3b_r16_e8 | Qwen/Qwen2.5-3B-Instruct | 0.95 | 0.2036→0.2035 | 0.1822→0.1821 |
| c5_7b_r16_e2 | Qwen/Qwen2.5-7B-Instruct | 0.6 | 0.1949→0.1779 | 0.2023→0.1811 |
