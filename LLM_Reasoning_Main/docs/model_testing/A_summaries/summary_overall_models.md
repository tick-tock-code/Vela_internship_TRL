# Overall Models Summary

## Key Findings
- `evidence_support_rating` appears useful and should remain in the feature set.
- Preferred models: `LR`, `XGB1`, and `MLP4` (especially with `PLS`).
- `PLS` impact on `LR`/`XGB1` is still underdetermined; it may help on the test set.
- `XGB1` is the fallback candidate if simple `LR` underperforms on the test set.

## Stability and Combos
- Most stable combos: `A`, `D`, `F`.
- `A+D` looks strong with `MLP4`.
- Promising LR/XGB1 combos to test: `A`, `F`, `D+E+F`, `A+C+D+E`, `A+B+C+D+E+F`.

## Current Direction
- Redundant features pruned and regularization increased.
- `PLS` remains a potential additive step depending on test‑set behavior.



🧠 8. What I would do (clean decision rule)
Step 1 — Split experiments:

Group A: Clean features

A, A+D, A+E → use BASE

Group B: High-signal features

F, D+E+F, full combo → use ONLY PLS
Step 2 — Final candidates:
LR + A (baseline safe)
LR + PLS(full combo) (likely best linear model)
XGB1 + full combo (raw or PLS)
MLP4 + PLS(full combo) (highest ceiling)

## Collinearity (Base, VIF threshold 50)
- VIF <= 50: HQ, A, B, C, D, E, A+B, A+C, A+D, A+E, A+D+E, A+C+D+E
- VIF > 50: F, D+F, C+F, D+E+F, C+D+E+F, A+B+C+D+E+F
