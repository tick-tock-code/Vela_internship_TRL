# Reasoning-Block PLS Protocol

This Step 3 method applies `PLS` to the reasoning block only, inside each outer fold, before concatenating the latent components with raw `HQ`.

## Study Setup

- Families in scope: A, B, C, D, E, F
- Outer CV: 3-fold repeated 16 times
- Models: `LR`, `XGB1`
- Frozen baseline benchmark: run `HQ` once on the same `3 x 16` outer CV for each evaluator used in this stage.
- Current `HQ` benchmark on this run:
  - `LR`: `0.207 +/- 0.046`
  - `XGB1`: `0.204 +/- 0.033`
- No row subsampling is used in this pass.
- Holdout/test is deferred until a candidate route is locked.

## Component Grids

- `A`: 2, 3
- `B`: 1, 2
- `C`: 2
- `D`: 1, 2
- `E`: 1
- `F`: 2, 3

## Leakage Control

- The reasoning block is filled, standardised, and fit with `PLSRegression` using outer-train rows only.
- Validation rows are transformed using the train-fit scaler and train-fit `PLS` model only.
- Raw `HQ` is concatenated after the reasoning block has been transformed.
- Thresholds are selected by train-only inner CV inside each outer split.
