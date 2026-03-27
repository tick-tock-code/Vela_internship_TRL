# Model Testing Report
Generated: 2026-03-26T22:19:21.213796+00:00Z

Model variants: logistic, xgb1 (stump).
Model set: simple.
Transform: SFT.
PCA variance: n/a

## HQ Mirror + Reasoning (rule layer)
| Combo | LOGISTIC | XGB1 |
|---|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.233+/-0.027 |
| D | 0.249+/-0.021 | 0.235+/-0.029 |
| A+E | 0.279+/-0.050 | 0.277+/-0.052 |
| A+D+E+F | 0.309+/-0.062 | 0.297+/-0.056 |

## CV vs Full-Train (per combo, same model)
### HQ Mirror + Reasoning
| Combo | LOGISTIC CV | LOGISTIC Full | XGB1 CV | XGB1 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.239 | 0.233+/-0.027 | 0.235 |
| D | 0.249+/-0.021 | 0.263 | 0.235+/-0.029 | 0.243 |
| A+E | 0.279+/-0.050 | 0.267 | 0.277+/-0.052 | 0.281 |
| A+D+E+F | 0.309+/-0.062 | 0.335 | 0.297+/-0.056 | 0.309 |

## Interpretability (HQ Mirror only)
_Interpretability disabled for this run._
