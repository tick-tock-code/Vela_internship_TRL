# Model Testing Report
Generated: 2026-03-26T22:18:41.343365+00:00Z

Model variants: logistic, xgb1 (stump).
Model set: simple.
Transform: BASE.
PCA variance: n/a

## HQ Mirror + Reasoning (rule layer)
| Combo | LOGISTIC | XGB1 |
|---|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.233+/-0.027 |
| D | 0.249+/-0.021 | 0.235+/-0.029 |
| A+E | 0.281+/-0.042 | 0.280+/-0.055 |
| A+D+E+F | 0.339+/-0.043 | 0.301+/-0.042 |

## CV vs Full-Train (per combo, same model)
### HQ Mirror + Reasoning
| Combo | LOGISTIC CV | LOGISTIC Full | XGB1 CV | XGB1 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.239 | 0.233+/-0.027 | 0.235 |
| D | 0.249+/-0.021 | 0.263 | 0.235+/-0.029 | 0.243 |
| A+E | 0.281+/-0.042 | 0.291 | 0.280+/-0.055 | 0.285 |
| A+D+E+F | 0.339+/-0.043 | 0.337 | 0.301+/-0.042 | 0.313 |

## Interpretability (HQ Mirror only)
_Interpretability disabled for this run._
