# Step 1: Evidence Map

This Step 1 pass is a clean reproduction layer. It does not admit families or reject them permanently.

## Current Read

- The raw reasoning routes remain unstable relative to the HQ benchmark story.
- The clearest report-linked wins still come from transformed `PLS` routes, especially `F` and `D+E+F`.
- This step therefore exists to map raw failure patterns and transform-sensitive signal before new mathematical methods are added.

## Unit Classification

| Unit | Classification | Raw LR CV | Raw XGB1 CV | PLS LR CV | Legacy Raw Test F0.5 | Legacy PLS Test F0.5 |
|---|---|---:|---:|---:|---:|---:|
| HQ + E | no_reproducible_evidence | 0.229 | 0.212 | 0.228 | -- | -- |
| HQ + A+B+C+D+E+F | raw_fail | 0.332 | 0.319 | 0.333 | -- | 0.225 |
| HQ + A+C | raw_fail | 0.316 | 0.317 | 0.326 | 0.242 | -- |
| HQ + C+D+E+F | raw_fail | 0.269 | 0.230 | 0.271 | -- | -- |
| HQ + Engineered Set 05 | raw_fail | 0.252 | 0.245 | 0.250 | -- | -- |
| HQ + A | raw_fail | 0.307 | 0.317 | 0.312 | 0.240 | 0.233 |
| HQ + B | raw_fail | 0.225 | 0.215 | 0.226 | -- | -- |
| HQ + C | raw_fail | 0.243 | 0.209 | 0.232 | -- | -- |
| HQ + D | raw_fail | 0.241 | 0.221 | 0.238 | -- | -- |
| HQ + D+E+F | transform_sensitive | 0.257 | 0.239 | 0.267 | -- | 0.274 |
| HQ + F | transform_sensitive | 0.244 | 0.217 | 0.249 | -- | 0.271 |

## Whole-Data Route Snapshot

| Unit | Class | Raw LR Mean | Raw LR Std | Raw XGB1 Mean | Raw XGB1 Std | PLS LR Mean | PLS LR Std | Legacy Raw Test | Legacy PLS Test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | anchor_reference | 0.233 | 0.028 | 0.212 | 0.025 | 0.231 | 0.028 | 0.264 | -- |
| HQ + D+E+F | transform_sensitive | 0.257 | 0.036 | 0.239 | 0.025 | 0.267 | 0.051 | -- | 0.274 |
| HQ + F | transform_sensitive | 0.244 | 0.031 | 0.217 | 0.015 | 0.249 | 0.036 | -- | 0.271 |
| HQ + A+B+C+D+E+F | raw_fail | 0.332 | 0.025 | 0.319 | 0.045 | 0.333 | 0.037 | -- | 0.225 |
| HQ + A+C | raw_fail | 0.316 | 0.013 | 0.317 | 0.049 | 0.326 | 0.036 | 0.242 | -- |
| HQ + C+D+E+F | raw_fail | 0.269 | 0.035 | 0.230 | 0.018 | 0.271 | 0.044 | -- | -- |
| HQ + Engineered Set 05 | raw_fail | 0.252 | 0.049 | 0.245 | 0.037 | 0.250 | 0.060 | -- | -- |
| HQ + A | raw_fail | 0.307 | 0.033 | 0.317 | 0.051 | 0.312 | 0.036 | 0.240 | 0.233 |
| HQ + B | raw_fail | 0.225 | 0.043 | 0.215 | 0.031 | 0.226 | 0.033 | -- | -- |
| HQ + C | raw_fail | 0.243 | 0.029 | 0.209 | 0.016 | 0.232 | 0.057 | -- | -- |
| HQ + D | raw_fail | 0.241 | 0.040 | 0.221 | 0.019 | 0.238 | 0.050 | -- | -- |
| HQ + E | no_reproducible_evidence | 0.229 | 0.026 | 0.212 | 0.040 | 0.228 | 0.035 | -- | -- |

## Priority Lists

- Compression priority: combo_D_E_F, reasoning_F
- Stability-selection priority: combo_A_B_C_D_E_F, combo_A_C, reasoning_A, combo_C_D_E_F, engineered_set_05
- Grouped-penalty priority: combo_A_B_C_D_E_F, combo_A_C, reasoning_A, combo_C_D_E_F, reasoning_C

## Redundancy Snapshot

| Unit | Max Cross Corr vs HQ | Raw Cond. No. | Raw Max VIF |
|---|---:|---:|---:|
| reasoning_E | 0.030 | 18.499 | 6.049 |
| combo_A_B_C_D_E_F | 0.041 | 203.258 | 119.258 |
| combo_A_C | 0.041 | 33.397 | 7.289 |
| combo_C_D_E_F | 0.040 | 165.264 | 119.002 |
| engineered_set_05 | -- | 158393570380337824.000 | 134492620601592.531 |
| reasoning_A | 0.041 | 22.930 | 6.255 |
| reasoning_B | 0.031 | 18.196 | 7.705 |
| reasoning_C | 0.040 | 19.540 | 6.852 |
| reasoning_D | 0.031 | 17.462 | 5.636 |
| combo_D_E_F | 0.034 | 147.879 | 118.930 |
| reasoning_F | 0.034 | 110.922 | 118.876 |
