# Model Testing Report
Generated: 2026-03-26T22:18:47.302805+00:00Z

Model variants: logistic, xgb1 (stump).
Model set: simple.
Transform: PCA.
PCA variance: 0.999

## HQ Mirror + Reasoning (rule layer)
| Combo | LOGISTIC | XGB1 |
|---|---:|---:|---:|---:|
| HQ | 0.225+/-0.032 | 0.228+/-0.028 |
| D | 0.238+/-0.034 | 0.245+/-0.039 |
| A+E | 0.238+/-0.042 | 0.237+/-0.033 |
| A+D+E+F | 0.263+/-0.038 | 0.272+/-0.054 |

## CV vs Full-Train (per combo, same model)
### HQ Mirror + Reasoning
| Combo | LOGISTIC CV | LOGISTIC Full | XGB1 CV | XGB1 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | 0.225+/-0.032 | 0.234 | 0.228+/-0.028 | 0.271 |
| D | 0.238+/-0.034 | 0.240 | 0.245+/-0.039 | 0.252 |
| A+E | 0.238+/-0.042 | 0.230 | 0.237+/-0.033 | 0.229 |
| A+D+E+F | 0.263+/-0.038 | 0.260 | 0.272+/-0.054 | 0.276 |

## Interpretability (HQ Mirror only)
_Interpretability disabled for this run._
