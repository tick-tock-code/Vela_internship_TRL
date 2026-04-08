# Supervised Grouping Summary

This note compares raw routes against supervised grouping routes on the same `3 x 16` outer CV.

## HQ Benchmarks

| Evaluator | Outer CV F0.5 Mean | Outer CV F0.5 Std | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|
| logistic | 0.207 | 0.046 | 0.237 | 0.060 |
| xgb1 | 0.204 | 0.033 | 0.702 | 0.068 |

## Augmentation Track

| Feature Set | Grouping | Model | Raw F0.5 | Grouped F0.5 | Delta | Raw Std | Grouped Std | Original Count | Grouped Count | Final Count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ + A | 2 groups per family | logistic | 0.302 | 0.283 | -0.018 | 0.035 | 0.039 | 5 | 2 | 31 |
| HQ + A | 2 groups per family | xgb1 | 0.298 | 0.302 | 0.004 | 0.033 | 0.029 | 5 | 2 | 31 |
| HQ + A | 3 groups per family | logistic | 0.302 | 0.301 | -0.001 | 0.035 | 0.038 | 5 | 3 | 32 |
| HQ + A | 3 groups per family | xgb1 | 0.298 | 0.308 | 0.009 | 0.033 | 0.031 | 5 | 3 | 32 |
| HQ + A+B+C+D+E | 2 groups per family | logistic | 0.304 | 0.318 | 0.014 | 0.031 | 0.031 | 15 | 10 | 39 |
| HQ + A+B+C+D+E | 2 groups per family | xgb1 | 0.302 | 0.298 | -0.004 | 0.030 | 0.033 | 15 | 10 | 39 |
| HQ + A+B+C+D+E | 3 groups per family | logistic | 0.304 | 0.306 | 0.002 | 0.031 | 0.031 | 15 | 13 | 42 |
| HQ + A+B+C+D+E | 3 groups per family | xgb1 | 0.302 | 0.304 | 0.002 | 0.030 | 0.029 | 15 | 13 | 42 |
| HQ + B+C+D+E+F | 2 groups per family | logistic | 0.252 | 0.247 | -0.004 | 0.045 | 0.041 | 16 | 10 | 39 |
| HQ + B+C+D+E+F | 2 groups per family | xgb1 | 0.215 | 0.221 | 0.006 | 0.035 | 0.040 | 16 | 10 | 39 |
| HQ + B+C+D+E+F | 3 groups per family | logistic | 0.252 | 0.245 | -0.007 | 0.045 | 0.040 | 16 | 13 | 42 |
| HQ + B+C+D+E+F | 3 groups per family | xgb1 | 0.215 | 0.217 | 0.002 | 0.035 | 0.037 | 16 | 13 | 42 |
| HQ + F | 2 groups per family | logistic | 0.227 | 0.228 | 0.000 | 0.043 | 0.038 | 6 | 2 | 31 |
| HQ + F | 2 groups per family | xgb1 | 0.206 | 0.207 | 0.001 | 0.036 | 0.037 | 6 | 2 | 31 |
| HQ + F | 3 groups per family | logistic | 0.227 | 0.227 | -0.000 | 0.043 | 0.041 | 6 | 3 | 32 |
| HQ + F | 3 groups per family | xgb1 | 0.206 | 0.209 | 0.003 | 0.036 | 0.037 | 6 | 3 | 32 |

## Competition Track

| Feature Set | Grouping | Model | Raw F0.5 | Grouped F0.5 | Delta | Raw Std | Grouped Std | Original Count | Grouped Count | Final Count |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | 6 total groups | logistic | 0.207 | 0.206 | -0.001 | 0.046 | 0.038 | 29 | 6 | 6 |
| HQ | 6 total groups | xgb1 | 0.204 | 0.197 | -0.007 | 0.033 | 0.041 | 29 | 6 | 6 |
| HQ + A | 6 total groups | logistic | 0.302 | 0.300 | -0.002 | 0.035 | 0.039 | 34 | 6 | 6 |
| HQ + A | 6 total groups | xgb1 | 0.298 | 0.299 | 0.001 | 0.033 | 0.035 | 34 | 6 | 6 |
| HQ + A+B+C+D+E | 6 total groups | logistic | 0.304 | 0.267 | -0.037 | 0.031 | 0.048 | 44 | 6 | 6 |
| HQ + A+B+C+D+E | 6 total groups | xgb1 | 0.302 | 0.282 | -0.021 | 0.030 | 0.043 | 44 | 6 | 6 |
| HQ + B+C+D+E+F | 6 total groups | logistic | 0.252 | 0.214 | -0.038 | 0.045 | 0.027 | 45 | 6 | 6 |
| HQ + B+C+D+E+F | 6 total groups | xgb1 | 0.215 | 0.218 | 0.003 | 0.035 | 0.038 | 45 | 6 | 6 |
| HQ + F | 6 total groups | logistic | 0.227 | 0.209 | -0.018 | 0.043 | 0.041 | 35 | 6 | 6 |
| HQ + F | 6 total groups | xgb1 | 0.206 | 0.206 | 0.000 | 0.036 | 0.041 | 35 | 6 | 6 |
