# Supervised Grouping Protocol

This Step 4 method groups correlated features inside each outer training fold, then collapses each cluster to one latent feature with `1`-component `PLS`.

## Study Setup

- Outer CV: 3-fold repeated 16 times
- Models: `LR`, `XGB1`
- Holdout/test is deferred in this first pass.
- Frozen `HQ` is always rerun on the exact same outer splits.

## Tracks

- `augmentation`: group reasoning families separately, then append grouped reasoning features to raw `HQ`.
- `competition`: cluster the full active feature set together, including `HQ`, and evaluate only the grouped features.

## Augmentation Feature Sets

- `HQ + A`
- `HQ + F`
- `HQ + A+B+C+D+E`
- `HQ + B+C+D+E+F`

- Groups per family explored: 2, 3

## Competition Feature Sets

- `HQ`
- `HQ + A`
- `HQ + F`
- `HQ + A+B+C+D+E`
- `HQ + B+C+D+E+F`

- Total groups explored: 6

## Leakage Control

- Train-only Z-scoring is fit inside each outer fold.
- Supervised distances are computed from train-only correlations.
- Agglomerative clustering is fit on the train-only distance matrix.
- `PLSRegression(n_components=1)` is fit on each train-only cluster.
- Threshold selection uses train-only inner CV after grouping is complete.
