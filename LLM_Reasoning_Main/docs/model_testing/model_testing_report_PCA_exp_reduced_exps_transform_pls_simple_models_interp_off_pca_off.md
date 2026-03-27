# Model Testing Report
Generated: 2026-03-26T22:18:52.152794+00:00Z

Model variants: logistic, xgb1 (stump).
Model set: simple.
Transform: PLS.
PCA variance: n/a

## HQ Mirror + Reasoning (rule layer)
| Combo | LOGISTIC | XGB1 |
|---|---:|---:|---:|---:|
| HQ | 0.225+/-0.008 | 0.224+/-0.058 |
| D | 0.248+/-0.042 | 0.228+/-0.041 |
| A+E | 0.282+/-0.048 | 0.270+/-0.069 |
| A+D+E+F | 0.325+/-0.044 | 0.305+/-0.048 |

## CV vs Full-Train (per combo, same model)
### HQ Mirror + Reasoning
| Combo | LOGISTIC CV | LOGISTIC Full | XGB1 CV | XGB1 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | 0.225+/-0.008 | 0.231 | 0.224+/-0.058 | 0.264 |
| D | 0.248+/-0.042 | 0.265 | 0.228+/-0.041 | 0.276 |
| A+E | 0.282+/-0.048 | 0.288 | 0.270+/-0.069 | 0.314 |
| A+D+E+F | 0.325+/-0.044 | 0.334 | 0.305+/-0.048 | 0.330 |

## Interpretability (HQ Mirror only)
_Interpretability disabled for this run._
