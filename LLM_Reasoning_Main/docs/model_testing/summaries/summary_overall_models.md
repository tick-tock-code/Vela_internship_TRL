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
