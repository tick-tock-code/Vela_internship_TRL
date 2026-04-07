# Instability-Control Status Report

This is a current-state synthesis for the active study path. It is not a final experimental report.

## Step 1 Outcome

- Transform-sensitive units: 2
- Raw-fail units: 8
- No reproducible evidence: 1

## Whole-Data Snapshot

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

## Next Method Targets

- Compression: combo_D_E_F, reasoning_F
- Stability selection: combo_A_B_C_D_E_F, combo_A_C, reasoning_A, combo_C_D_E_F, engineered_set_05
- Grouped penalty: combo_A_B_C_D_E_F, combo_A_C, reasoning_A, combo_C_D_E_F, reasoning_C

## Scaffold Status

- Active mathematical methods in this pass: 1
- The next concrete implementation target is `stability_selection` on `HQ + A-F`.
