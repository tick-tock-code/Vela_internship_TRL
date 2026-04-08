# Reasoning-Block PLS Summary

This note compares raw `HQ + family` routes against leakage-safe blockwise `PLS` routes.

## HQ Benchmarks

| Evaluator | Outer CV F0.5 Mean | Outer CV F0.5 Std |
|---|---:|---:|
| logistic | 0.207 | 0.046 |
| xgb1 | 0.204 | 0.033 |

## Best Blockwise PLS By Family

| Family | Model | Raw F0.5 | Best Block k | Best Block F0.5 | Delta | Threshold Mean | Threshold Std |
|---|---|---:|---:|---:|---:|---:|---:|
| HQ + A | logistic | 0.302 | 3 | 0.299 | -0.002 | 0.275 | 0.036 |
| HQ + A | xgb1 | 0.298 | 3 | 0.297 | -0.001 | 0.742 | 0.053 |
| HQ + B | logistic | 0.206 | 1 | 0.203 | -0.003 | 0.249 | 0.075 |
| HQ + B | xgb1 | 0.202 | 2 | 0.202 | -0.001 | 0.711 | 0.063 |
| HQ + C | logistic | 0.213 | 2 | 0.213 | -0.000 | 0.239 | 0.054 |
| HQ + C | xgb1 | 0.208 | 2 | 0.215 | 0.007 | 0.700 | 0.064 |
| HQ + D | logistic | 0.218 | 2 | 0.218 | 0.000 | 0.234 | 0.045 |
| HQ + D | xgb1 | 0.216 | 2 | 0.226 | 0.011 | 0.705 | 0.058 |
| HQ + E | logistic | 0.204 | 1 | 0.205 | 0.001 | 0.252 | 0.073 |
| HQ + E | xgb1 | 0.205 | 1 | 0.200 | -0.004 | 0.720 | 0.062 |
| HQ + F | logistic | 0.227 | 2 | 0.227 | -0.000 | 0.231 | 0.055 |
| HQ + F | xgb1 | 0.206 | 3 | 0.241 | 0.035 | 0.686 | 0.037 |
