# Model Testing Report
Generated: 2026-03-27T14:43:16.655338+00:00Z

Model variants: logistic (l2), elasticnet.
Model set: simple.
Transform: BASE.

## HQ Mirror + Reasoning (rule layer)
| Combo | LOGISTIC | ELASTICNET |
|---|---:|---:|
| HQ | 0.234+/-0.041 | 0.230+/-0.046 |
| A | 0.275+/-0.024 | 0.269+/-0.036 |
| B | 0.246+/-0.050 | 0.250+/-0.055 |
| C | 0.253+/-0.030 | 0.243+/-0.059 |
| D | 0.252+/-0.024 | 0.253+/-0.063 |
| E | 0.253+/-0.059 | 0.251+/-0.046 |
| F | 0.284+/-0.038 | 0.289+/-0.053 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.311+/-0.053 |

## CV vs Full-Train (per combo, same model)
### HQ Mirror + Reasoning
| Combo | LOGISTIC CV | LOGISTIC Full | ELASTICNET CV | ELASTICNET Full |
|---|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.239 | 0.230+/-0.046 | 0.232 |
| A | 0.275+/-0.024 | 0.278 | 0.269+/-0.036 | 0.274 |
| B | 0.246+/-0.050 | 0.248 | 0.250+/-0.055 | 0.253 |
| C | 0.253+/-0.030 | 0.277 | 0.243+/-0.059 | 0.249 |
| D | 0.252+/-0.024 | 0.260 | 0.253+/-0.063 | 0.263 |
| E | 0.253+/-0.059 | 0.258 | 0.251+/-0.046 | 0.265 |
| F | 0.284+/-0.038 | 0.292 | 0.289+/-0.053 | 0.292 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.339 | 0.311+/-0.053 | 0.323 |

## Interpretability (HQ Mirror only)
_Interpretability disabled for this run._
