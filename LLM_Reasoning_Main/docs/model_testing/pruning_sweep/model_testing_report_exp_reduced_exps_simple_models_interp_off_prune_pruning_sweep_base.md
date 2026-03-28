# Model Testing Report
Generated: 2026-03-28T13:45:01.027976+00:00Z

Model variants: logistic (l2).
Model set: simple.
Transform: BASE.
Feature pruning: sweep (none, mild, aggressive).

## Pruning Sweep Results

### LOGISTIC
| Combo | none | mild | aggressive |
|---|---:|---:|---:|
| HQ | 0.236+/-0.037 | 0.244+/-0.048 | 0.238+/-0.028 |
| A | 0.273+/-0.036 | 0.281+/-0.029 | 0.284+/-0.026 |
| B | 0.247+/-0.050 | 0.242+/-0.051 | 0.243+/-0.052 |
| C | 0.251+/-0.032 | 0.257+/-0.047 | 0.250+/-0.038 |
| D | 0.249+/-0.023 | 0.254+/-0.032 | 0.252+/-0.043 |
| E | 0.253+/-0.056 | 0.256+/-0.070 | 0.248+/-0.060 |
| F | 0.282+/-0.038 | 0.283+/-0.034 | 0.289+/-0.035 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.330+/-0.052 | 0.327+/-0.039 |

## Pruning Sweep CV vs Full-Train (per combo, same model)

### LOGISTIC - HQ Mirror + Reasoning
| Combo | none CV | none Full | mild CV | mild Full | aggressive CV | aggressive Full |
|---|---:|---:|---:|---:|---:|---:|
| HQ | 0.236+/-0.037 | 0.238 | 0.244+/-0.048 | 0.243 | 0.238+/-0.028 | 0.247 |
| A | 0.273+/-0.036 | 0.278 | 0.281+/-0.029 | 0.287 | 0.284+/-0.026 | 0.280 |
| B | 0.247+/-0.050 | 0.250 | 0.242+/-0.051 | 0.247 | 0.243+/-0.052 | 0.250 |
| C | 0.251+/-0.032 | 0.271 | 0.257+/-0.047 | 0.264 | 0.250+/-0.038 | 0.261 |
| D | 0.249+/-0.023 | 0.249 | 0.254+/-0.032 | 0.264 | 0.252+/-0.043 | 0.258 |
| E | 0.253+/-0.056 | 0.257 | 0.256+/-0.070 | 0.267 | 0.248+/-0.060 | 0.252 |
| F | 0.282+/-0.038 | 0.281 | 0.283+/-0.034 | 0.292 | 0.289+/-0.035 | 0.299 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.343 | 0.330+/-0.052 | 0.344 | 0.327+/-0.039 | 0.348 |

## Selected Logistic Hyperparameters (HQ Mirror)
| Combo | none | mild | aggressive |
|---|---:|---:|---:|
| HQ | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| A | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| B | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| C | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| D | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| E | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| F | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |
| A+B+C+D+E+F | C=0.3,l1=None | C=0.3,l1=None | C=0.3,l1=None |

## Interpretability (HQ Mirror only)
Interpretability disabled for this run.

## Collinearity Diagnostics (Logistic only)
_Model-input stats use transformed features; raw stats use pre-transform standardized features._
_Correlation threshold: 0.9_

| Combo | Family | Transform | Sweep | max_vif | max_abs_corr | cond_num | avg_sign_flip | raw_max_vif | raw_max_abs_corr | raw_cond_num |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | hq_mirror | BASE | none | inf | 1.000 | 183656586954018400.000 | 0.055 | inf | 1.000 | 183656586954018400.000 |
| A+B+C+D+E+F | hq_mirror | BASE | none | inf | 1.000 | 2698910617561285120.000 | 0.041 | inf | 1.000 | 2698910617561285120.000 |
| B | hq_mirror | BASE | none | inf | 1.000 | 153781113049306176.000 | 0.060 | inf | 1.000 | 153781113049306176.000 |
| C | hq_mirror | BASE | none | inf | 1.000 | 7246409554509645824.000 | 0.065 | inf | 1.000 | 7246409554509645824.000 |
| D | hq_mirror | BASE | none | inf | 1.000 | 287905858601857312.000 | 0.040 | inf | 1.000 | 287905858601857312.000 |
| E | hq_mirror | BASE | none | inf | 1.000 | 217649695201373088.000 | 0.058 | inf | 1.000 | 217649695201373088.000 |
| F | hq_mirror | BASE | none | inf | 1.000 | 134269441586398560.000 | 0.053 | inf | 1.000 | 134269441586398560.000 |
| HQ | hq_mirror | BASE | none | inf | 1.000 | 548977557649435328.000 | 0.071 | inf | 1.000 | 548977557649435328.000 |
