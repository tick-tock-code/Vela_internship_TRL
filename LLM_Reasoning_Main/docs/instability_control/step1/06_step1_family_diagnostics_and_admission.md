# Step 1: Evidence Mapping and Transform-Sensitivity Reproduction

This note replaces the earlier admission-first framing.

Step 1 is a clean evidence layer whose purpose is to restate the current result without searching for new methods yet.

## Purpose

Step 1 exists to answer one question:

- which reasoning units fail in raw form, and which ones only show usable signal after transformation?

This step does not admit families permanently and it does not discard transform-sensitive units.

Every non-`HQ` unit in Step 1 is evaluated as `HQ + unit`, not as a family-only model.

## Part 1: Freeze The Anchor

Freeze:

- `HQ_anchor_xgb1_unpruned`

This is the canonical benchmark anchor.

Do not redefine it during Step 1.

## Part 2: Freeze The Evaluation Units

Evaluate two layers:

- atomic families:
  - engineered bank
  - `A`
  - `B`
  - `C`
  - `D`
  - `E`
  - `F`
- fixed legacy combos:
  - `HQ`
  - `A+C`
  - `D+E+F`
  - `C+D+E+F`
  - `A+B+C+D+E+F`

The point is to repeat the known evidence cleanly before exploring new search spaces.

## Part 3: Run The Canonical Step 1 Routes

Use these routes:

- `anchor_xgb1_unpruned`
- `raw_lr_base`
- `raw_xgb1`
- `pls_lr_n6`
- `pls_mlp4_n6`

The `pls_mlp4_n6` route is audit-only and should not drive the main classification decisions.

## Part 4: Produce The Evidence Artifacts

Required artifacts:

- `step1_route_metrics.csv/json`
- `step1_overlap_metrics.csv/json`
- `step1_synthesis.md`
- `step1_legacy_alignment.md`
- `step1_snapshot.csv`

The route-metrics table should capture:

- CV F0.5 mean/std
- per-fold F0.5
- full-train diagnostic metrics
- legacy test precision, recall, and F0.5 where a curated report row exists

The overlap table should capture:

- added feature count
- within-unit correlation burden
- cross-correlation with HQ
- condition number
- VIF-style burden where it can be computed safely

The snapshot should capture, in one wide table:

- raw LR CV mean/std
- raw XGB1 CV mean/std
- PLS LR CV mean/std
- legacy raw and transformed holdout/test values where available

## Step 1 Synthesis Labels

Use these labels:

- `raw_fail`
  - raw LR and raw XGB1 do not produce a reproducible out-of-sample story over HQ
- `transform_sensitive`
  - raw routes fail or collapse, but `PLS` reproduces an improvement over the raw version and clears the report-linked HQ comparison
- `no_reproducible_evidence`
  - neither raw nor transformed routes show evidence worth carrying forward yet

## What Step 1 Must Not Do

Step 1 must not:

- redefine a pruned HQ route as the baseline
- use "positive and stable raw gain" as a keep-or-kill rule
- act as a keep/discard gate based on raw gain
- describe Step 1 as a process that is not a keep/discard gate based on raw gain
- jump straight to new mathematical methods before the current evidence has been restated clearly

## Success Criteria

Step 1 is successful if it leaves the repo with:

- one clean evidence map
- one clean legacy-alignment note
- one classification of units into raw-fail, transform-sensitive, or no reproducible evidence
- one explicit priority list for:
  - compression
  - stability selection
  - grouped penalties

At that point the repo is ready for `stability_selection` as the first new mathematical method.
