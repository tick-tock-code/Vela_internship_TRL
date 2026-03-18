# Script-by-Script (Step-by-Step)

This document describes what each script/file in `5_llm_reasoning` does, step by step.

## `vcbench_pipeline.py`
1. Load `.env` (if present) to set API keys.
2. Parse CLI flags and read `features.json` (if provided).
3. Resolve dataset CSV (`sample` or `full`) and load records/labels.
4. Resolve LLM-reasoning core prompt and experiments paths.
5. Apply dataset size override for reasoning (if requested).
6. Create train/test split and log split sizes.
7. If `--llm_reasoning_dry_run_fast`:
   1. Generate reasoning features on a small subset (no API calls).
   2. Save parquet + metadata.
   3. Write logs to `training_logs/dry_runs/...`.
   4. Exit (no training).
8. Load baseline 15 features (via example script) and custom registry features.
9. If LLM-engineered features are enabled:
   1. Generate rules from train set.
   2. Evaluate features on train/test/all.
10. If LLM-reasoning features are enabled:
    1. Generate per-founder outputs from core prompt + experiments.
    2. Save parquet + metadata.
11. Build the final feature matrix for the selected mode:
    - `human`, `llm`, `reasoning`, or `hybrid`.
12. Standardize continuous custom features (train stats only).
13. Save the feature dataset to `features_storage/`.
14. If `--extract_only`, write logs and exit.
15. Train scikit-learn logistic regression.
16. Tune threshold on training set for F0.5.
17. Compute metrics and write training logs + run report JSON.

## `llm_reasoning_features.py`
1. Load core prompt template and experiments JSON.
2. Validate experiment definitions.
3. Select dataset size (full or subset).
4. Build output schema (namespaced columns for each experiment key).
5. For each founder:
   1. Create prompt: core prompt + selected experiment instructions + founder JSON.
   2. Call LLM (or mock response in dry-run).
   3. Parse JSON into numeric + text outputs.
   4. Validate numeric ranges.
6. Save parquet to `features_storage/llm_reasoning/`.
7. Write metadata JSON (prompt paths, dataset size, failures).
8. Return the dataframe and numeric feature list for training.

## `llm_feature_generation.py`
1. Load `.env` and refresh LLM settings.
2. Instantiate `FeatureGenerator` (TRL) with schema/helpers.
3. Generate `n_rules` rules using **train** records only.
4. Compile and evaluate rules (train/test/all).
5. Print rule names/descriptions and compilation warnings.
6. Return LLM feature matrices and rule names.

## `feature_registry.py`
1. Define custom feature functions (QS tiers, exits, durations).
2. Register features with type metadata (binary/continuous).
3. Expose `FEATURE_SETS` for simple selection.

## `feature_selector_gui.py`
1. Render a Tkinter GUI with baseline + custom features.
2. Allow selection and saving to `features.json`.
3. Allow loading a saved JSON to restore selections.
4. Provide a checkbox + count for LLM engineered features.

## `tests_reasoning_schema.py`
1. Validates core prompt placeholders.
2. Validates experiments JSON schema.
3. Runs a dry-run generation sanity test.
4. Confirms dry-run fast limits rows.

## `tests_smoke.py`
1. Lightweight smoke checks (project-specific).
2. Sanity checks for imports and basic execution.

## `run_feature_gui.ps1`
1. Launches the feature selector GUI in the correct environment.

## `core_prompt.txt`
1. Defines the fixed prompt template.
2. Includes placeholders for experiments and founder profile.

## `experiments.json`
1. Defines selectable experiments (A–F).
2. Each experiment includes instructions and output keys.

## `features.json`
1. Feature list for custom registry features.
2. LLM toggles and LLM reasoning experiment selections.
3. Paths for prompt/experiments files.

## `README.md`
1. High-level overview of the folder.
2. Basic usage instructions.
