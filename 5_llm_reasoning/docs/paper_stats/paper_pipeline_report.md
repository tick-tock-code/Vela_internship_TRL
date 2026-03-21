# Paper Pipeline Report
Generated: 2026-03-21T13:45:51.494990

## Part 1 (Pool 4400, XGB only — engineered sets 01/04/05 and A+E only)
| Set ID | Regression | Reasoning Combo | F0.5 (mean+/-std) |
|---|---|---|---:|
| -- | human | n/a | 0.226+/-0.014 |
| -- | human | n/a | 0.226+/-0.015 |
| -- | human | n/a | 0.230+/-0.016 |
| set_01 | llm_engineered | n/a | 0.242+/-0.030 |
| set_04 | llm_engineered | n/a | 0.238+/-0.014 |
| set_05 | llm_engineered | n/a | 0.239+/-0.021 |
| set_01 | llm_engineered_plus_reasoning | A+E | 0.272+/-0.034 |
| set_04 | llm_engineered_plus_reasoning | A+E | 0.268+/-0.035 |
| set_05 | llm_engineered_plus_reasoning | A+E | 0.283+/-0.053 |

**Part 1 family mean +/- std (F0.5):**
- human: 0.227+/-0.002
- llm_engineered: 0.240+/-0.002
- llm_engineered_plus_reasoning: 0.274+/-0.007

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

### Full Mirror parity check (vs llm_regression_report)
| Combo | Model | Paper F0.5 | Report F0.5 | Δ | Status |
|---|---|---:|---:|---:|---|
| HQ | logistic | 0.234+/-0.041 | 0.234+/-0.041 | +0.000 | PASS |
| HQ | xgboost | 0.233+/-0.027 | 0.233+/-0.027 | -0.000 | PASS |
| A | logistic | 0.275+/-0.024 | 0.275+/-0.024 | -0.000 | PASS |
| A | xgboost | 0.272+/-0.042 | 0.272+/-0.042 | -0.000 | PASS |
| A+E | logistic | 0.281+/-0.042 | 0.281+/-0.042 | -0.000 | PASS |
| A+E | xgboost | 0.280+/-0.055 | 0.280+/-0.055 | +0.000 | PASS |
| A+F | logistic | 0.308+/-0.056 | 0.308+/-0.056 | +0.000 | PASS |
| A+F | xgboost | 0.295+/-0.030 | 0.295+/-0.030 | -0.000 | PASS |
| A+D+E+F | logistic | 0.339+/-0.043 | 0.339+/-0.043 | +0.000 | PASS |
| A+D+E+F | xgboost | 0.301+/-0.042 | 0.301+/-0.042 | -0.000 | PASS |
| A+B+C+D+E+F | logistic | 0.329+/-0.054 | 0.329+/-0.054 | +0.000 | PASS |
| A+B+C+D+E+F | xgboost | 0.300+/-0.040 | 0.300+/-0.040 | +0.000 | PASS |

## Max F0.5 on full train (biased upper bound)
| Model | Full-train F0.5 |
|---|---:|
| Human Legacy 1 (scaled durations) | 0.228 |
| Human Legacy 2 (binary equivalents) | 0.210 |
| Human Legacy 3 (mixed durations) | 0.238 |
| Engineered set_01 | 0.244 |
| Engineered set_04 | 0.242 |
| Engineered set_05 | 0.246 |
| Engineered set_01 + A+E | 0.270 |
| Engineered set_04 + A+E | 0.287 |
| Engineered set_05 + A+E | 0.292 |
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
