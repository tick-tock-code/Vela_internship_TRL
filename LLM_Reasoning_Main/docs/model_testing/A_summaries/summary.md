# Aggressive Pruning + ElasticNet Sweep (Minimal Exps)

**Run:** `exp_scope=minimal_exps`, `transform=Base`, `feature_pruning=aggressive_pruning`, `model_complexity=simple`.

## Snapshot
This is a decision-grade confirmation that reasoning features still improve HQ after aggressive pruning.

**Short answer:**
- Yes — reasoning features improve HQ.
- F is strongest but less stable.
- A is the best balance of gain + stability.

## Performance (F0.5 CV mean ± std)
**Baseline (HQ):**
- Logistic: **0.238 ± 0.028**
- ElasticNet: **0.244 ± 0.026**

**With features:**
| Combo | Logistic | Gain vs HQ | ElasticNet | Gain vs HQ |
| --- | ---: | ---: | ---: | ---: |
| A | 0.284 ± 0.026 | +0.045 | 0.287 ± 0.033 | +0.043 |
| D | 0.252 ± 0.043 | +0.014 | 0.262 ± 0.036 | +0.017 |
| F | 0.289 ± 0.035 | +0.050 | 0.299 ± 0.043 | +0.055 |

**CV vs Full-train (F0.5):**
| Combo | Logistic CV → Full | ElasticNet CV → Full |
| --- | ---: | ---: |
| HQ | 0.238 → 0.247 | 0.244 → 0.243 |
| A | 0.284 → 0.280 | 0.287 → 0.278 |
| D | 0.252 → 0.258 | 0.262 → 0.263 |
| F | 0.289 → 0.299 | 0.299 → 0.292 |

## Key Takeaways
- **Clear structure from aggressive pruning + elasticnet sweep**
- **HQ:** baseline, stable, weaker
- **A:** strong, most stable, clean signal, low overfit
- **F:** strongest, slightly overfits, more collinear/redundant, higher variance
- **D:** weak, noisy, secondary

## Ranking (Signal Strength)
**F > A >> D**

## Collinearity (Logistic, raw-feature diagnostics)
| Combo | raw_max_vif | raw_max_abs_corr | raw_cond_number |
| --- | ---: | ---: | ---: |
| HQ | 11.0 | 0.934 | 82.1 |
| A | 11.0 | 0.934 | 118.6 |
| D | 11.0 | 0.934 | 96.0 |
| F | 122.1 | 0.993 | 1384.4 |

**Interpretation:**
- Pruning reduced collinearity across the board.
- **F** still has extreme internal redundancy, which likely drives higher variance.

## ElasticNet Notes
ElasticNet improves mean scores slightly but increases variance for A and F. It is useful as a secondary model rather than a clear replacement for LR.

## Hyperparameter Snapshot (ElasticNet best from sweep)
| Combo | Best C | Best l1_ratio |
| --- | ---: | ---: |
| HQ | 0.1 | 0.2 |
| A | 0.1 | 0.2 |
| D | 0.1 | 0.5 |
| F | 0.3 | 1.0 |
