# Post‑CV Summary (5 Fits + Family Leaderboard)

# LLM Regression Summary (F0.5)

## Full Pipeline Results (2026-03-19 16:14)

**Run settings:** model=xgboost, CV folds=5, pool=4400 founders (seed excluded)

### Table 1 - Human & Reasoning Combos (model=xgboost, CV=5)
_No Table 1 rows._

### Table HQ - High-Quality Human & Reasoning Combos (model=xgboost, CV=5)
_HQ features = Structured v2 (28 features), with optional repeat_founding_gap and A/B/E reasoning combos._
| Regression | Reasoning Combo | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| XGB HQ Only | — | 0.207+/-0.016 | 0.666+/-0.018 | 0.176+/-0.010 | 0.199+/-0.022 | 0.260+/-0.045 | 0.837+/-0.023 |
| XGB HQ Only (+repeat_founding_gap) | — | 0.223+/-0.011 | 0.665+/-0.019 | 0.176+/-0.009 | 0.226+/-0.019 | 0.230+/-0.054 | 0.858+/-0.016 |
| XGB Reasoning Only | A | 0.268+/-0.030 | 0.641+/-0.049 | 0.173+/-0.016 | 0.262+/-0.031 | 0.298+/-0.043 | 0.861+/-0.009 |
| XGB HQ + Reasoning | A | 0.244+/-0.032 | 0.704+/-0.024 | 0.199+/-0.015 | 0.246+/-0.035 | 0.252+/-0.058 | 0.862+/-0.016 |
| XGB Reasoning Only | A+B | 0.251+/-0.027 | 0.637+/-0.045 | 0.172+/-0.016 | 0.249+/-0.026 | 0.268+/-0.047 | 0.861+/-0.008 |
| XGB HQ + Reasoning | A+B | 0.262+/-0.048 | 0.702+/-0.023 | 0.196+/-0.015 | 0.257+/-0.047 | 0.288+/-0.051 | 0.861+/-0.010 |
| XGB Reasoning Only | A+B+E | 0.251+/-0.045 | 0.640+/-0.032 | 0.176+/-0.016 | 0.257+/-0.039 | 0.235+/-0.064 | 0.872+/-0.003 |
| XGB HQ + Reasoning | A+B+E | 0.257+/-0.020 | 0.706+/-0.026 | 0.199+/-0.015 | 0.256+/-0.025 | 0.273+/-0.042 | 0.862+/-0.012 |
| XGB Reasoning Only | A+E | 0.264+/-0.022 | 0.638+/-0.031 | 0.174+/-0.015 | 0.262+/-0.020 | 0.275+/-0.038 | 0.865+/-0.006 |
| XGB HQ + Reasoning | A+E | 0.262+/-0.027 | 0.708+/-0.029 | 0.200+/-0.015 | 0.272+/-0.013 | 0.240+/-0.058 | 0.875+/-0.007 |
| XGB Reasoning Only | B | 0.179+/-0.033 | 0.543+/-0.027 | 0.116+/-0.009 | 0.234+/-0.051 | 0.096+/-0.017 | 0.889+/-0.009 |
| XGB HQ + Reasoning | B | 0.212+/-0.022 | 0.666+/-0.021 | 0.175+/-0.012 | 0.206+/-0.029 | 0.273+/-0.075 | 0.835+/-0.032 |
| XGB Reasoning Only | B+E | 0.178+/-0.036 | 0.553+/-0.026 | 0.126+/-0.011 | 0.216+/-0.051 | 0.109+/-0.020 | 0.882+/-0.012 |
| XGB HQ + Reasoning | B+E | 0.212+/-0.017 | 0.677+/-0.033 | 0.180+/-0.016 | 0.222+/-0.029 | 0.202+/-0.059 | 0.860+/-0.027 |
| XGB Reasoning Only | E | 0.181+/-0.025 | 0.545+/-0.025 | 0.124+/-0.013 | 0.228+/-0.038 | 0.101+/-0.013 | 0.887+/-0.007 |
| XGB HQ + Reasoning | E | 0.207+/-0.030 | 0.678+/-0.032 | 0.181+/-0.016 | 0.215+/-0.043 | 0.255+/-0.134 | 0.835+/-0.063 |

### Top 3 (Table 1) by F0.5
- (none)

### Top 3 (Table HQ) by F0.5
- XGB Reasoning Only [A]: F0.5=0.268+/-0.030
- XGB Reasoning Only [A+E]: F0.5=0.264+/-0.022
- XGB HQ + Reasoning [A+E]: F0.5=0.262+/-0.027

### Table 2 - Engineered Family (18 rules x 10 sets) (model=xgboost, CV=5)
_No Table 2 rows._

### Top 10 (Table 2) by F0.5
- (none)

*Metrics are 5-fold stratified CV on 4400 founders (seed excluded).*
