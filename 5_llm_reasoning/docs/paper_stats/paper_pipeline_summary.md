# paper_pipeline.py — Long‑Form Summary

This document describes what `paper_pipeline.py` does, end‑to‑end, so you can verify it is executing the exact experiments you intend. It is intentionally detailed and mirrors the code path as of the current version.

## High‑Level Purpose

`paper_pipeline.py` runs a **paper‑focused experiment suite** that:
- Trains a fixed set of model variants (Part 1 + Part 2).
- Computes OOF CV metrics (5‑fold by default).
- Trains full‑data models **for test‑set prediction** (and also reports a biased upper‑bound metric).
- Produces test‑set predictions.
- Enforces strict integrity checks to ensure test predictions come from the exact trained model.

## Inputs (Primary)

- Public dataset CSV:
  - `VCBench-Starter-Kit/vcbench_final_public.csv`
- Private test dataset CSV:
  - `test_dataset/vcbench_final_private (success column removed) - vcbench_final_private.csv`
- Reasoning caches:
  - Pool (4,400): `features_storage/llm_reasoning/currently_in_use/llm_reasoning_full.parquet`
  - Full (4,500): `features_storage/llm_reasoning/full_current/llm_reasoning_full.parquet`
- Engineered feature family:
  - `features_storage/llm_engineered/archives/family_<id>/set_*/current/llm_features.parquet`
- Fixed fold caches:
  - Pool folds: `features_storage/cv_folds/folds_k{K}_seed42.json`
  - Full folds: `features_storage/cv_folds/folds_k{K}_seed42_full.json`
- Reasoning prompt files:
  - `core_prompt.txt`
  - `experiments.json`

**Test parsing (mirror‑aligned):**
- The private test CSV is parsed using the same `_safe_json_parse` logic as `think_reason_learn.datasets._vcbench`.
- HQ test features are extracted directly from the raw test CSV (same as the mirror pipeline).

## Outputs (Primary)

- Main report:
  - `docs/paper_stats/paper_pipeline_report.md`
- Model results table:
  - `docs/paper_stats/paper_pipeline_results.csv`
- Test predictions:
  - `docs/paper_stats/paper_pipeline_test_preds_pt1.csv`
  - `docs/paper_stats/paper_pipeline_test_preds_pt2.csv`
- Per‑model artifacts:
  - `docs/paper_stats/pt1/<family>/<model_name>/`
  - `docs/paper_stats/pt2/<combo>/<model_name>/`
  - Each folder contains `model.joblib`, `metrics_cv.json`, `metrics_full_train.json`, `threshold_oof.json`, `model_manifest.json`
- Prediction verification log:
  - `docs/paper_stats/prediction_verification_report.json`
- Feature snapshots for audit:
  - `docs/paper_stats/features/` (human, llm_engineered, llm_reasoned)

## Configuration (CLI)

- `--cv_folds` (default 5): number of folds used in OOF CV.
- `--test_csv`: path to private test CSV.
- `--test_reasoning_parquet`: merged private test reasoning output.
- `--include_abcdef_test_preds`: allow predictions for ABCDEF in Part 2.
- `--pt2_pred_combos`: filter which Part‑2 combos produce test predictions (comma list).
- `--pt2_pred_models`: filter which Part‑2 model types produce test predictions (comma list).
- `--llm_model`, `--google_model`: LLM providers for test reasoning generation.
- `--engineered_family_id`: pick a specific engineered family archive.
- `--engineered_set_ids`: which engineered sets to use (default `auto`, which loads the top‑3 set IDs from `docs/paper_stats/engineered_top3_xgb.json` if present; otherwise falls back to `set_01,set_04,set_05`).
- `--n_engineered_sets`: number of engineered sets if auto‑generating.
- `--engineered_n_rules`: number of engineered rules if auto‑generating.

**Prediction filtering behavior:**
- If `--pt2_pred_combos` is provided, Part‑2 test predictions are **only** written for those combos.
- If `--pt2_pred_models` is empty but combos are provided, it defaults to `logistic,xgboost`.
- Training, model saving, and reporting are **unchanged**; only the Part‑2 prediction CSV is filtered.

**Engineered test feature reconstruction:**
- If engineered test features are missing or column‑mismatched, the pipeline rebuilds them from the saved `llm_rules.json` for each set (stored in the engineered family archive).

## Part 1: Pool‑Only (4,400) Experiments

**Dataset:** 4,400 pool (seed_100 excluded). Fold cache: `folds_k{K}_seed42.json`.

**Models:** XGBoost only for Part 1.

**Families:**
- Human baselines (legacy feature sets): `LEGACY_HUMAN_FEATURE_SETS`.
- LLM‑engineered sets: `set_01`, `set_04`, `set_05`.
- LLM‑engineered + reasoning: only **A+E** (from `currently_in_use`).

**For each model run:**
- Build feature matrix for pool and test set.
- Compute OOF scores using fixed folds.
- Select OOF‑tuned threshold.
- Train **full‑data model** on entire pool (this is the model used for test predictions).
- Produce test predictions using the **exact trained model object** (see integrity section).

## Part 2: Full‑Dataset (4,500) Experiments

**Dataset:** full 4,500 (seed included). Fold cache: `folds_k{K}_seed42_full.json`.

**Models:** Logistic Regression and XGBoost.

**Features:**
- HQ structured v2 features (28 base features).
- Reasoning combos (numeric only) from full_current.
- Rule layer applied (prior‑exit): `exit_count > 0` overrides predictions.

**Allowed combos:**
- HQ (no reasoning)
- A
- A+E
- A+F
- A+D+E+F
- A+B+C+D+E+F

**For each model run:**
- Build feature matrix from HQ + reasoning combo.
- Compute OOF scores using fixed folds.
- Select OOF‑tuned threshold.
- Train **full‑data model** on 4,500 (this is the model used for test predictions).
- Produce test predictions using the **exact trained model object**.

## Reasoning Feature Handling (Private Test Set)

`_ensure_test_reasoning(...)` guarantees that **A/D/E/F** test reasoning features exist and are clean.

Behavior:
- If merged test reasoning parquet exists, metadata must match:
  - `parse_version: vcbench_safe_json_parse_v1`
  - `records_hash` (hash of parsed test records)
- If metadata mismatches → archive old parquet and regenerate.
- If missing experiments or NaNs → generate only the missing/NaN batches.
- Each experiment is run **separately** (one prompt per experiment).

Generation parameters:
- Batch size: 20
- Concurrency: 10
- Rate‑limit fallback sequence: 8 → 6 → 4 → 2 → 1
- Inline NaN repair enabled
- Targeted repair runs only for affected batches

All test reasoning outputs are written to:
- `test_dataset/exp_<ID>/` (per‑experiment run artifacts)
- `test_dataset/llm_reasoning_private.parquet` (merged output)
- `test_dataset/llm_reasoning_private_meta.json` (parse metadata + records hash)

## OOF Threshold Tuning (Both Parts)

- For each fold:
  - Train model on k‑1 folds.
  - Predict on validation fold.
- Concatenate all out‑of‑fold predictions.
- Sweep threshold to maximize F0.5 across all OOF predictions.
- Use this single threshold for:
  - Fold metrics
  - Full‑train evaluation
  - Test predictions (applied to the full‑data model)

## Integrity Checks: Model–Prediction Consistency

Before predictions are written:
- The pipeline builds a `model_manifest.json` with hashes of:
  - feature_names
  - train matrix (after imputation/standardization)
  - labels
  - model bytes
  - threshold
  - seed
- Immediately prior to prediction, it recomputes these hashes and verifies:
  - all hashes match
  - threshold matches
  - model type matches

If any mismatch is found, the run **fails hard** with a RuntimeError.

Verification status is written to:
- `docs/paper_stats/prediction_verification_report.json`

## Parity Check Against vcbench Pipeline

For the Full Mirror inputs:
- The pipeline compares the numeric full_current reasoning matrix used in `paper_pipeline.py` against the `vcbench_pipeline` reference.
- It fails if row order, dtypes, columns, or content hash differ.

This ensures the Part 2 Full Mirror runs are **identical** to the main pipeline in data terms.

## Output Files (Summary)

- `docs/paper_stats/paper_pipeline_report.md`:
  - Part 1 summary table
  - Part 2 summary table
  - Mirror LR/XGB summary
  - Parity check section
  - Full‑train F0.5 section

- `docs/paper_stats/paper_pipeline_results.csv`:
  - one row per model run, includes CV mean/std + OOF threshold + full‑train F0.5

- `docs/paper_stats/paper_pipeline_test_preds_pt1.csv`:
  - test predictions for Part 1 runs

- `docs/paper_stats/paper_pipeline_test_preds_pt2.csv`:
  - test predictions for Part 2 runs

- Per‑model folders under:
  - `docs/paper_stats/pt1/` and `docs/paper_stats/pt2/`

## Flowchart (Data Flow)

```mermaid
flowchart TD
  A[Public CSV: vcbench_final_public.csv] --> B[Load records + labels]
  B --> C[Split seed_100 + pool_4400]
  B --> D[Full 4500 dataset]

  C --> E[Pool reasoning: currently_in_use]
  C --> F[Engineered features: family sets]
  C --> G[Legacy human features]

  D --> H[Full reasoning: full_current]
  D --> I[HQ features (structured v2)]
  D --> J[Rule mask: exit_count > 0]

  K[Private test CSV] --> L[Parse JSON -> test records]
  L --> M[Ensure test reasoning A–F]
  M --> N[Merged test reasoning parquet]

  E & F & G --> O[Part 1: XGB OOF CV]
  O --> P[Train full model]
  P --> Q[Test predictions]

  H & I & J --> R[Part 2: LR + XGB OOF CV]
  R --> S[Train full model]
  S --> T[Test predictions]

  Q & T --> U[paper_pipeline_report.md + CSVs]
```

## Quick Sanity Checklist

- Part 1 uses pool (4400), XGB only, A+E only for reasoning.
- Part 2 uses full (4500), LR + XGB, top‑pick combos only.
- OOF thresholding is used everywhere.
- Test predictions only use the trained model object.
- Manifests are generated and validated.

