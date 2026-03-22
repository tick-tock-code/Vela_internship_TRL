# Paper Pipeline Report
Generated: 2026-03-22T00:12:57.988563

## Part 1 (Pool 4400, XGB only — engineered sets 01/04/05 and A+E only)
| Set ID | Regression | Reasoning Combo | F0.5 (mean+/-std) |
|---|---|---|---:|
| -- | human | n/a | 0.226+/-0.014 |
| -- | human | n/a | 0.226+/-0.015 |
| -- | human | n/a | 0.230+/-0.016 |
| set_05 | llm_engineered | n/a | 0.261+/-0.066 |
| set_08 | llm_engineered | n/a | 0.252+/-0.042 |
| set_07 | llm_engineered | n/a | 0.249+/-0.051 |
| set_05 | llm_engineered_plus_reasoning | A+E | 0.280+/-0.069 |
| set_08 | llm_engineered_plus_reasoning | A+E | 0.274+/-0.032 |
| set_07 | llm_engineered_plus_reasoning | A+E | 0.277+/-0.040 |

**Part 1 family mean +/- std (F0.5):**
- human: 0.227+/-0.002
- llm_engineered: 0.254+/-0.005
- llm_engineered_plus_reasoning: 0.277+/-0.002

## Part 2 (Full 4500, Full Mirror + Reasoning; rule layer — top picks only)
| Model | Combo | Type | F0.5 (mean±std) |
|---|---|---|---:|
| Mirror HQ HQ (LOGISTIC) | HQ | logistic | 0.234+/-0.041 |
| Mirror HQ HQ (XGBOOST) | HQ | xgboost | 0.233+/-0.027 |
| Mirror HQ A (LOGISTIC) | A | logistic | 0.275+/-0.024 |
| Mirror HQ A (XGBOOST) | A | xgboost | 0.272+/-0.042 |
| Mirror HQ A+E (LOGISTIC) | A+E | logistic | 0.281+/-0.042 |
| Mirror HQ A+E (XGBOOST) | A+E | xgboost | 0.280+/-0.055 |
| Mirror HQ A+F (LOGISTIC) | A+F | logistic | 0.308+/-0.056 |
| Mirror HQ A+F (XGBOOST) | A+F | xgboost | 0.295+/-0.030 |
| Mirror HQ A+D+E+F (LOGISTIC) | A+D+E+F | logistic | 0.339+/-0.043 |
| Mirror HQ A+D+E+F (XGBOOST) | A+D+E+F | xgboost | 0.301+/-0.042 |
| Mirror HQ A+B+C+D+E+F (LOGISTIC) | A+B+C+D+E+F | logistic | 0.329+/-0.054 |
| Mirror HQ A+B+C+D+E+F (XGBOOST) | A+B+C+D+E+F | xgboost | 0.300+/-0.040 |

**Test predictions filtered to combos:** HQ,A+E,A+D+E+F; **models:** logistic,xgboost

### Mirror LR/XGB summary
| Combo | LR F0.5 | XGB F0.5 |
|---|---:|---:|
| HQ | 0.234+/-0.041 | 0.233+/-0.027 |
| A | 0.275+/-0.024 | 0.272+/-0.042 |
| A+E | 0.281+/-0.042 | 0.280+/-0.055 |
| A+F | 0.308+/-0.056 | 0.295+/-0.030 |
| A+D+E+F | 0.339+/-0.043 | 0.301+/-0.042 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.300+/-0.040 |

## Max F0.5 on full train (biased upper bound)
| Model | Full-train F0.5 |
|---|---:|
| Human Legacy 1 (scaled durations) | 0.228 |
| Human Legacy 2 (binary equivalents) | 0.210 |
| Human Legacy 3 (mixed durations) | 0.238 |
| Engineered set_05 | 0.268 |
| Engineered set_08 | 0.251 |
| Engineered set_07 | 0.255 |
| Engineered set_05 + A+E | 0.288 |
| Engineered set_08 + A+E | 0.287 |
| Engineered set_07 + A+E | 0.291 |
| Mirror HQ HQ (LOGISTIC) | 0.239 |
| Mirror HQ HQ (XGBOOST) | 0.235 |
| Mirror HQ A (LOGISTIC) | 0.278 |
| Mirror HQ A (XGBOOST) | 0.277 |
| Mirror HQ A+E (LOGISTIC) | 0.291 |
| Mirror HQ A+E (XGBOOST) | 0.285 |
| Mirror HQ A+F (LOGISTIC) | 0.323 |
| Mirror HQ A+F (XGBOOST) | 0.299 |
| Mirror HQ A+D+E+F (LOGISTIC) | 0.337 |
| Mirror HQ A+D+E+F (XGBOOST) | 0.313 |
| Mirror HQ A+B+C+D+E+F (LOGISTIC) | 0.339 |
| Mirror HQ A+B+C+D+E+F (XGBOOST) | 0.306 |

## Test Prediction Distribution Summary (probabilities)

_Generated: 2026-03-22T00:40:12.174665_

### PT1

| Model | n | mean | median | p10 | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Human_Legacy_1_(scaled_durations) | 4500 | 0.0696 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Human_Legacy_2_(binary_equivalents) | 4500 | 0.0596 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Human_Legacy_3_(mixed_durations) | 4500 | 0.0682 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Engineered_set_05 | 4500 | 0.0542 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Engineered_set_08 | 4500 | 0.0571 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Engineered_set_07 | 4500 | 0.0813 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Engineered_set_05___A_E | 4500 | 0.0167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Engineered_set_08___A_E | 4500 | 0.0436 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Engineered_set_07___A_E | 4500 | 0.0636 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### PT2

| Model | n | mean | median | p10 | p25 | p75 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mirror_HQ_HQ_(LOGISTIC) | 4500 | 0.0593 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mirror_HQ_HQ_(XGBOOST) | 4500 | 0.0824 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mirror_HQ_A_E_(LOGISTIC) | 4500 | 0.0624 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mirror_HQ_A_E_(XGBOOST) | 4500 | 0.0616 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mirror_HQ_A_D_E_F_(LOGISTIC) | 4500 | 0.0707 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mirror_HQ_A_D_E_F_(XGBOOST) | 4500 | 0.0649 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Test Prediction Consistency Checks (correlation)

_Generated: 2026-03-22T00:47:16.327953_

### PT2 (Full 4500)

Pairwise Pearson correlation summary (all model columns): min=0.501, median=0.681, mean=0.677, max=0.875


Correlation vs HQ (LOGISTIC):

| Model | r |
|---|---:|
| Mirror_HQ_HQ_(XGBOOST) | 0.746 |
| Mirror_HQ_A_E_(LOGISTIC) | 0.666 |
| Mirror_HQ_A_E_(XGBOOST) | 0.730 |
| Mirror_HQ_A_D_E_F_(LOGISTIC) | 0.573 |
| Mirror_HQ_A_D_E_F_(XGBOOST) | 0.678 |

Correlation vs HQ (XGBOOST):

| Model | r |
|---|---:|
| Mirror_HQ_HQ_(LOGISTIC) | 0.746 |
| Mirror_HQ_A_E_(LOGISTIC) | 0.607 |
| Mirror_HQ_A_E_(XGBOOST) | 0.764 |
| Mirror_HQ_A_D_E_F_(LOGISTIC) | 0.501 |
| Mirror_HQ_A_D_E_F_(XGBOOST) | 0.705 |

### PT1 (Part 1 predictions)

Pairwise Pearson correlation summary (all model columns): min=0.272, median=0.439, mean=0.465, max=0.962


Engineered vs Engineered+Reasoning (A+E) correlations:

| Base | +A+E | r |
|---|---|---:|
| Engineered_set_05 | Engineered_set_05___A_E | 0.390 |
| Engineered_set_08 | Engineered_set_08___A_E | 0.647 |
| Engineered_set_07 | Engineered_set_07___A_E | 0.599 |
