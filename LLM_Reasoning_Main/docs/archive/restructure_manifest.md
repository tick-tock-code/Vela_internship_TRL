# Restructure Manifest

Date: 2026-04-05

## Canonical Data Moves

- `features_storage/cv_folds/` -> `data/vcbench/folds/`
- `features_storage/human/` -> `data/vcbench/features/human/`
- `features_storage/human_high_quality/` -> `data/vcbench/features/human_high_quality/`
- `features_storage/features_full.parquet` -> `data/vcbench/features/combined/features_full.parquet`
- `features_storage/llm_engineered/archives/` -> `data/vcbench/features/llm_engineered/archives/`
- `features_storage/llm_engineered/families/` -> `data/vcbench/features/llm_engineered/families/`
- `features_storage/llm_engineered/seed_100.json` -> `data/vcbench/features/llm_engineered/seed_100.json`
- canonical engineered family duplicated at `data/vcbench/features/llm_engineered/family_20260321_144238/`
- `features_storage/llm_reasoning/currently_in_use/` -> `data/vcbench/features/llm_reasoning/train/combined/currently_in_use/`
- `features_storage/llm_reasoning/full_current/` -> `data/vcbench/features/llm_reasoning/train/combined/full_current/`
- `test_dataset/exp_A`..`exp_F` -> `data/vcbench/features/llm_reasoning/test/exp_A`..`exp_F`
- `test_dataset/llm_reasoning_private.parquet` -> `data/vcbench/features/llm_reasoning/test/combined/llm_reasoning_private.parquet`
- `test_dataset/llm_reasoning_private_meta.json` -> `data/vcbench/features/llm_reasoning/test/combined/llm_reasoning_private_meta.json`
- private test CSV -> `data/vcbench/raw/`
- public VCBench CSVs copied from `../VCBench-Starter-Kit/` -> `data/vcbench/raw/`

## Deprecated Trees Archived

- old `features_storage/` -> `.tmp/migration/deprecated_features_storage/`
- old `test_dataset/` -> `.tmp/migration/deprecated_test_dataset/`
- old `training_logs/` -> `.tmp/migration/legacy_training_logs/`
- old `logging/` -> `.tmp/migration/legacy_logging/`
- duplicate `python/docs/` -> `.tmp/migration/python_docs_duplicate/`
- legacy `docs/model_testing/` export tree -> `.tmp/migration/docs_model_testing_legacy/`
- detailed `docs/paper_stats/` artifacts -> `.tmp/migration/docs_paper_stats_detailed/`

## Curated Docs Restored

- `docs/model_testing/` now contains only curated summaries copied back from the archived legacy export tree
- `docs/paper_stats/` now contains only retained summary files at the top level

## Code Layout Changes

- frozen legacy implementations moved to `python/pipelines/legacy/`
- original legacy entrypoint filenames remain callable as wrappers in `python/pipelines/`
- active instability-control code added under `python/lib/stability/` and `python/pipelines/instability_control/`
- canonical presets added under `configs/presets/`

## Write Paths Going Forward

- curated human-readable outputs stay in `docs/`
- raw run outputs go to `.tmp/runs/`
- canonical reusable datasets and caches live in `data/vcbench/`
