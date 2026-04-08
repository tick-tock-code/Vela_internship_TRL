# Stability Selection Protocol

This Step 2 method pass tests row-subsampled stability selection before any new `PLS` work.

## Study Setup

- Families in scope: A, B, C, D, E, F
- Units are evaluated as `HQ + family`, not family-only models.
- Selector: `LogisticRegression(solver="saga", penalty="l1")`
- C grid: 0.01, 0.03, 0.1, 0.3, 1.0, 3.0
- Outer CV: 3-fold repeated 3 times
- Row subsamples per outer-train fold: 25
- Subsample fraction: 0.50
- Primary stability threshold: 0.80
- Sign-consistency threshold for reasoning features: 0.90
- Report thresholds: 0.6, 0.8, 0.9

## Tracks

- `competition_track`: HQ features survive on selection frequency alone; reasoning features must pass both the frequency and sign-consistency thresholds.
- `augmentation_track`: HQ is always retained in the final refit, while reasoning features are admitted only if they pass both thresholds.

## Evaluation

- Downstream evaluators: `LR` and `XGB1`
- Thresholds are chosen using train-only inner CV inside each outer fold.
- Private-test scoring is not available locally in this pass; the final audit table is reserved for locked candidates and external submission results.
