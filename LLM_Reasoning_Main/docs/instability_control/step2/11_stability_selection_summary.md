# Stability Selection Summary

This note separates route performance from selector behavior for each `HQ + family` pilot.

## HQ Benchmarks

| Evaluator | Outer CV F0.5 Mean | Outer CV F0.5 Std |
|---|---:|---:|
| logistic | 0.212 | 0.044 |
| xgb1 | 0.202 | 0.041 |

## Route Comparison

| Family | Raw LR | Comp LR | Aug LR | Raw XGB1 | Comp XGB1 | Aug XGB1 | Comp HQ Kept | Comp Reason Kept | Aug Reason Kept | Mean Reason Freq | Mean Sign Consistency | Mean Sign-Consistent Reason | Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ + A | 0.308 +/- 0.026 | 0.300 +/- 0.029 | 0.304 +/- 0.029 | 0.296 +/- 0.029 | 0.292 +/- 0.031 | 0.293 +/- 0.027 | 27.533 | 3.067 | 3.067 | 0.973 | 0.896 | 3.067 | 0.961 |
| HQ + F | 0.229 +/- 0.042 | 0.233 +/- 0.044 | 0.233 +/- 0.041 | 0.201 +/- 0.037 | 0.204 +/- 0.040 | 0.203 +/- 0.040 | 27.533 | 3.200 | 3.200 | 0.894 | 0.855 | 3.267 | 0.830 |
