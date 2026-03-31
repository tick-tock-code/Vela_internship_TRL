# Model Testing Report (Finalising Experiments)
Generated: 2026-03-31T02:35:14.973155+00:00Z

Models: LR (BASE), LR (PLS), MLP4 (PLS).
Feature pruning: aggressive_pruning.

## HQ Mirror + Reasoning
### BASE (LR)
| Combo | LR (BASE) CV | LR (BASE) Full |
|---|---:|---:|
| HQ | 0.241+/-0.033 | 0.246 |
| A | 0.286+/-0.028 | 0.275 |
| A+C | 0.292+/-0.033 | 0.296 |

### PLS (LR, MLP4)
| Combo | LR (PLS) CV | LR (PLS) Full | MLP4 (PLS) CV | MLP4 (PLS) Full |
|---|---:|---:|---:|---:|
| F | 0.278+/-0.031 | 0.293 | 0.292+/-0.054 | 0.296 |
| D+E+F | 0.301+/-0.036 | 0.302 | 0.302+/-0.054 | 0.310 |
| A+B+C+D+E+F | 0.337+/-0.033 | 0.344 | 0.333+/-0.042 | 0.352 |
| C+D+E+F | 0.300+/-0.034 | 0.304 | 0.310+/-0.044 | 0.315 |

## Collinearity Diagnostics (All models)
_Model-input stats use transformed features; raw stats use pre-transform standardized features._
_Correlation threshold: 0.9_

| Combo | Family | Model | Transform | Sweep | max_vif | max_abs_corr | cond_num | avg_sign_flip | raw_max_vif | raw_max_abs_corr | raw_cond_num |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | hq_mirror | logistic | BASE | None | 10.986 | 0.934 | 118.570 | 0.072 | 10.986 | 0.934 | 118.570 |
| A+B+C+D+E+F | hq_mirror | logistic | PLS | 6 | 1.000 | 0.000 | 1.000 | 0.067 | 122.516 | 0.993 | 2902.664 |
| A+B+C+D+E+F | hq_mirror | mlp4 | PLS | 6 | 1.000 | 0.000 | 1.000 | -- | 122.516 | 0.993 | 2902.664 |
| A+C | hq_mirror | logistic | BASE | None | 10.990 | 0.934 | 145.765 | 0.086 | 10.990 | 0.934 | 145.765 |
| C+D+E+F | hq_mirror | logistic | PLS | 6 | 1.000 | 0.000 | 1.000 | 0.067 | 122.253 | 0.993 | 2226.939 |
| C+D+E+F | hq_mirror | mlp4 | PLS | 6 | 1.000 | 0.000 | 1.000 | -- | 122.253 | 0.993 | 2226.939 |
| D+E+F | hq_mirror | logistic | PLS | 6 | 1.000 | 0.000 | 1.000 | 0.067 | 122.178 | 0.993 | 1877.442 |
| D+E+F | hq_mirror | mlp4 | PLS | 6 | 1.000 | 0.000 | 1.000 | -- | 122.178 | 0.993 | 1877.442 |
| F | hq_mirror | logistic | PLS | 6 | 1.000 | 0.000 | 1.000 | 0.067 | 122.141 | 0.993 | 1384.393 |
| F | hq_mirror | mlp4 | PLS | 6 | 1.000 | 0.000 | 1.000 | -- | 122.141 | 0.993 | 1384.393 |
| HQ | hq_mirror | logistic | BASE | None | 10.969 | 0.934 | 82.070 | 0.060 | 10.969 | 0.934 | 82.070 |
