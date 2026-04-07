# Step 2 Method Note

Step 2 is testing one narrow idea first: can row-subsampled stability selection make `HQ + reasoning family` routes less internally inflated before any new transform method is introduced.

## What This Step Does

- Fix the baseline as `HQ`.
- Add one reasoning family at a time on top of `HQ`.
- Run repeated outer CV.
- Inside each outer-train split, repeatedly subsample the training rows.
- Fit an `L1` logistic selector on each subsample across a fixed `C` grid.
- Track how often each feature is selected.
- Track whether selected reasoning features keep the same coefficient sign.
- Refit downstream `LR` and `XGB1` models on the selected subsets.

This step is not using `PLS`, and it is not using family combos.

## Current Selection Logic

- A feature is frequency-stable if it is selected often enough across row subsamples.
- A reasoning feature is sign-stable if its coefficient keeps the same dominant sign often enough across those selections.
- `competition_track`:
  HQ features and reasoning features compete together.
  HQ features are filtered on selection frequency.
  Reasoning features must pass both the frequency threshold and the sign-consistency threshold.
- `augmentation_track`:
  all HQ features are kept.
  reasoning features are added only if they pass both thresholds.

## What Is Being Swept

- Outer CV splits
- Row subsamples inside each outer-train fold
- Selector `C` values
- Prediction thresholds chosen by inner CV

The downstream `LR` and `XGB1` evaluators are not being hyperparameter-searched in this step.

## Parameters

The main tunable parameters in this step are:

- `outer_cv.n_splits`
  What it does: number of folds in the outer evaluation loop.
  Current config default: `3`
  Current reduced pilot: `3`
  Suggested values to explore: `3`, then `5` only after the subsampling settings are stable.
- `outer_cv.n_repeats`
  What it does: number of reshuffled outer-CV repeats.
  Current config default: `10`
  Current reduced pilot: `3`
  Suggested values to explore: `5`, then `10`.
- `subsampling.n_subsamples`
  What it does: number of row subsamples per outer-train fold used for stability estimation.
  Current config default: `100`
  Current reduced pilot: `20`
  Suggested values to explore: `50`, `100`.
- `subsampling.fraction`
  What it does: fraction of outer-train rows drawn into each subsample.
  Current config default: `0.50`
  Current reduced pilot: `0.50`
  Suggested values to explore: keep fixed at `0.50` for now.
- `selector.c_grid`
  What it does: regularisation strengths used by the `L1` logistic selector.
  Current config default: `0.01, 0.03, 0.1, 0.3, 1.0, 3.0`
  Current reduced pilot: same
  Suggested values to explore: keep fixed for now; revisit only if `A` still looks too permissive after the larger run.
- `stability.primary_threshold`
  What it does: minimum selection frequency for a feature to count as stable.
  Current config default: `0.80`
  Current reduced pilot: `0.80`
  Suggested values to explore: `0.70`, `0.80`, `0.90`.
- `stability.sign_consistency_threshold`
  What it does: minimum sign-consistency required for reasoning features.
  Current config default: `0.90`
  Current reduced pilot: `0.90`
  Suggested values to explore: `0.85`, `0.90`, `0.95`.
- `evaluation.inner_threshold_cv_splits`
  What it does: inner CV used to pick the prediction threshold inside each outer-train fold.
  Current config default: `3`
  Current reduced pilot: `3`
  Suggested values to explore: keep fixed at `3` for now.

Recommended compute order:

1. Increase `n_subsamples` first.
2. Increase `n_repeats` second.
3. Increase `n_splits` last.

The next larger pilot is:

- families: `A`, `F`
- outer CV: `3` folds x `5` repeats
- row subsamples per outer split: `100`

## What To Look For

- Does `HQ + A` lose some of its apparent CV edge once the sign-aware selector is applied?
- Do the stabilized routes reduce selected reasoning features while preserving useful HQ structure?
- Does the collapse gap look smaller once external audit results are available?

The main purpose of this step is to establish a clean, interpretable stability-selection baseline before trying more complex mathematical routes.
