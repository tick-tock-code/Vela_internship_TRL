# 5_llm_reasoning layout

This folder is organized to keep code, configs, prompts, and outputs separate.

**Core pipelines**
- `python/vcbench_pipeline.py` Full public-data pipeline (features, reasoning, CV training, reports, sweeps)
- `python/paper_pipeline.py` Paper-only pipeline (3x3 and 2x5 blocks, CV + test predictions)
- `python/prompt_evolution.py` Reasoning prompt evolution loop
- `python/reasoning_benchmark.py` Batch-size and concurrency benchmarking
- `python/reasoning_preview.py` Small live preview with logging and schema checks

**Key flags and modes (summary)**
`vcbench_pipeline.py`
- `--mode human|llm|reasoning|hybrid`
- `--run_profile full|xgb_mirror|reasoning_xgb_oof`
- `--model_type logistic|xgboost`
- `--cv_folds`, `--cv_use_fixed_folds`, `--cv_folds_path`
- `--llm_sweep`, `--llm_sweep_range`, `--llm_sweep_repeats`
- `--llm_reasoning_batch_size`, `--llm_reasoning_concurrency`, `--llm_reasoning_rate_limit_fallback_sequence`
- `--llm_reasoning_core_prompt`, `--llm_reasoning_experiments`
- `--human_feature_source baseline|high_quality`

`paper_pipeline.py`
- `--cv_folds`
- `--engineered_family_id`, `--engineered_set_ids`, `--engineered_n_rules`
- `--pt2_pred_combos`, `--pt2_pred_models`
- `--include_abcdef_test_preds`

**Other Python utilities**
- `python/llm_reasoning_features.py` Core reasoning engine
- `python/llm_feature_generation.py` LLM rule generation helper
- `python/feature_registry.py` Custom feature formulas and feature sets
- `python/feature_selector_gui.py` GUI for feature selection
- `python/cv_folds.py` Fixed fold cache helpers
- `python/rebuild_currently_in_use.py` Rebuild reasoning cache from latest runs
- `python/sync_currently_in_use.py` Sync latest runs to currently_in_use
- `python/repair_run.py` Targeted NaN repair (per experiment)
- `python/repair_run_a.py` Targeted NaN repair for run A
- `python/backfill_confidence_from_runs.py` Backfill evidence_support_rating into current cache
- `python/summarize_feature_weights.py` Coefficient summary report
- `python/summarize_reasoning_weights_top10.py` Reasoning weight summary
- `python/tests_reasoning_schema.py` Reasoning schema tests (no API)
- `python/tests_prompt_evolution.py` Prompt evolution unit tests
- `python/tests_smoke.py` Smoke test helpers

**Mirror experiments (XGBoost baseline replication)**
- `mirror_experiments/mirror_holdout_xgb.py` 80/20 holdout mirror with rule layer
- `mirror_experiments/mirror_cv_tuned_xgb.py` CV mirror with tuned threshold grid
- `mirror_experiments/mirror_test_predictions.py` Predict test CSV with mirror model

**Configuration and prompts**
- `configs/` JSON configs (features, experiments, prompt evolution)
- `prompts/` Prompt templates

**Scripts**
- `scripts/` PowerShell helpers (GUI launcher, cleanup helper)

**Outputs and data**
- `features_storage/` Cached features and manifests
- `test_dataset/` Private test inputs and cached reasoning outputs
- `training_logs/` and `logging/` Run logs (older sweep logs archived under `_archive/`)

**Notes**
- `docs/` Reports, paper stats, and summaries
- `archive/` Legacy runs and older notes
