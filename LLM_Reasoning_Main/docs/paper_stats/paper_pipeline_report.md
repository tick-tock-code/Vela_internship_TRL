# Paper Pipeline Report
Generated: 2026-03-26T01:18:06.143215

## Part 1 (Pool 4400) ? CV (OOF)
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

**Part 1 family mean +/- std (F0.5, CV):**
- human: 0.227+/-0.002
- llm_engineered: 0.254+/-0.005
- llm_engineered_plus_reasoning: 0.277+/-0.002

## Part 1 (Pool 4400) ? Train/Val split (80/20)
| Set ID | Regression | Reasoning Combo | F0.5 (val) |
|---|---|---|---:|
| -- | human | n/a | 0.244 |
| -- | human | n/a | 0.238 |
| -- | human | n/a | 0.244 |
| set_05 | llm_engineered | n/a | 0.236 |
| set_08 | llm_engineered | n/a | 0.310 |
| set_07 | llm_engineered | n/a | 0.327 |
| set_05 | llm_engineered_plus_reasoning | A+E | 0.362 |
| set_08 | llm_engineered_plus_reasoning | A+E | 0.382 |
| set_07 | llm_engineered_plus_reasoning | A+E | 0.377 |

## Part 2 (Full 4500, Full Mirror + Reasoning; rule layer) ? CV (OOF)
| Model | Combo | Type | F0.5 (mean+/-std) |
|---|---|---|---:|
| Mirror HQ HQ (LOGISTIC) | HQ | logistic | 0.234+/-0.041 |
| Mirror HQ HQ (XGBOOST) | HQ | xgboost | 0.233+/-0.027 |
| Mirror HQ A (LOGISTIC) | A | logistic | 0.275+/-0.024 |
| Mirror HQ A (XGBOOST) | A | xgboost | 0.272+/-0.042 |
| Mirror HQ B (LOGISTIC) | B | logistic | 0.246+/-0.050 |
| Mirror HQ B (XGBOOST) | B | xgboost | 0.227+/-0.023 |
| Mirror HQ D (LOGISTIC) | D | logistic | 0.249+/-0.021 |
| Mirror HQ D (XGBOOST) | D | xgboost | 0.235+/-0.029 |
| Mirror HQ A+E (LOGISTIC) | A+E | logistic | 0.281+/-0.042 |
| Mirror HQ A+E (XGBOOST) | A+E | xgboost | 0.280+/-0.055 |
| Mirror HQ A+F (LOGISTIC) | A+F | logistic | 0.308+/-0.056 |
| Mirror HQ A+F (XGBOOST) | A+F | xgboost | 0.295+/-0.030 |
| Mirror HQ A+D+E+F (LOGISTIC) | A+D+E+F | logistic | 0.339+/-0.043 |
| Mirror HQ A+D+E+F (XGBOOST) | A+D+E+F | xgboost | 0.301+/-0.042 |
| Mirror HQ A+B+C+D+E+F (LOGISTIC) | A+B+C+D+E+F | logistic | 0.329+/-0.054 |
| Mirror HQ A+B+C+D+E+F (XGBOOST) | A+B+C+D+E+F | xgboost | 0.300+/-0.040 |

**Test predictions filtered to combos:** HQ,A+E,A+D+E+F,D; **models:** logistic,xgboost

### Mirror LR/XGB summary (CV)
| Combo | LR F0.5 | XGB F0.5 |
|---|---:|---:|
| HQ | 0.234+/-0.041 | 0.233+/-0.027 |
| A | 0.275+/-0.024 | 0.272+/-0.042 |
| B | 0.246+/-0.050 | 0.227+/-0.023 |
| D | 0.249+/-0.021 | 0.235+/-0.029 |
| A+E | 0.281+/-0.042 | 0.280+/-0.055 |
| A+F | 0.308+/-0.056 | 0.295+/-0.030 |
| A+D+E+F | 0.339+/-0.043 | 0.301+/-0.042 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.300+/-0.040 |

## Part 2 (Full 4500, Full Mirror + Reasoning; rule layer) ? Train/Val split (80/20)
| Model | Combo | Type | F0.5 (val) |
|---|---|---|---:|
| Mirror HQ HQ (LOGISTIC) | HQ | logistic | 0.294 |
| Mirror HQ HQ (XGBOOST) | HQ | xgboost | 0.292 |
| Mirror HQ A (LOGISTIC) | A | logistic | 0.356 |
| Mirror HQ A (XGBOOST) | A | xgboost | 0.364 |
| Mirror HQ B (LOGISTIC) | B | logistic | 0.300 |
| Mirror HQ B (XGBOOST) | B | xgboost | 0.295 |
| Mirror HQ D (LOGISTIC) | D | logistic | 0.315 |
| Mirror HQ D (XGBOOST) | D | xgboost | 0.303 |
| Mirror HQ A+E (LOGISTIC) | A+E | logistic | 0.361 |
| Mirror HQ A+E (XGBOOST) | A+E | xgboost | 0.353 |
| Mirror HQ A+F (LOGISTIC) | A+F | logistic | 0.391 |
| Mirror HQ A+F (XGBOOST) | A+F | xgboost | 0.347 |
| Mirror HQ A+D+E+F (LOGISTIC) | A+D+E+F | logistic | 0.396 |
| Mirror HQ A+D+E+F (XGBOOST) | A+D+E+F | xgboost | 0.349 |
| Mirror HQ A+B+C+D+E+F (LOGISTIC) | A+B+C+D+E+F | logistic | 0.400 |
| Mirror HQ A+B+C+D+E+F (XGBOOST) | A+B+C+D+E+F | xgboost | 0.350 |

### Mirror LR/XGB summary (Train/Val)
| Combo | LR F0.5 | XGB F0.5 |
|---|---:|---:|
| HQ | 0.294 | 0.292 |
| A | 0.356 | 0.364 |
| B | 0.300 | 0.295 |
| D | 0.315 | 0.303 |
| A+E | 0.361 | 0.353 |
| A+F | 0.391 | 0.347 |
| A+D+E+F | 0.396 | 0.349 |
| A+B+C+D+E+F | 0.400 | 0.350 |

### Full Mirror parity check (vs llm_regression_report)
| Combo | Model | Paper F0.5 | Report F0.5 | Delta | Status |
|---|---|---:|---:|---:|---|
| HQ | logistic | 0.234+/-0.041 | 0.234+/-0.041 | +0.000 | PASS |
| HQ | xgboost | 0.233+/-0.027 | 0.233+/-0.027 | -0.000 | PASS |
| A | logistic | 0.275+/-0.024 | 0.275+/-0.024 | -0.000 | PASS |
| A | xgboost | 0.272+/-0.042 | 0.272+/-0.042 | -0.000 | PASS |
| B | logistic | 0.246+/-0.050 | 0.246+/-0.050 | -0.000 | PASS |
| B | xgboost | 0.227+/-0.023 | 0.227+/-0.023 | +0.000 | PASS |
| D | logistic | 0.249+/-0.021 | 0.249+/-0.021 | -0.000 | PASS |
| D | xgboost | 0.235+/-0.029 | 0.235+/-0.029 | +0.000 | PASS |
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
| Mirror HQ B (LOGISTIC) | 0.248 |
| Mirror HQ B (XGBOOST) | 0.239 |
| Mirror HQ D (LOGISTIC) | 0.263 |
| Mirror HQ D (XGBOOST) | 0.243 |
| Mirror HQ A+E (LOGISTIC) | 0.291 |
| Mirror HQ A+E (XGBOOST) | 0.285 |
| Mirror HQ A+F (LOGISTIC) | 0.323 |
| Mirror HQ A+F (XGBOOST) | 0.299 |
| Mirror HQ A+D+E+F (LOGISTIC) | 0.337 |
| Mirror HQ A+D+E+F (XGBOOST) | 0.313 |
| Mirror HQ A+B+C+D+E+F (LOGISTIC) | 0.339 |
| Mirror HQ A+B+C+D+E+F (XGBOOST) | 0.306 |
