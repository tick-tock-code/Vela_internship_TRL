# LLM_Reasoning_Main

This project builds and evaluates founder‑success prediction features for VCBench. It combines three sources of signal: (1) baseline human‑engineered features, (2) LLM‑engineered rule features generated from a small seed set, and (3) LLM reasoning features that score founders using structured prompts. The pipelines run cross‑validation, compare model families (Logistic Regression and XGBoost), and produce both report tables and test‑set predictions.

## Layout
- `python/` All Python code, split into:
  - `python/pipelines/` runnable pipelines
  - `python/lib/` import‑only modules
  - `python/tools/` maintenance/utility scripts
  - `python/tests/` smoke and schema tests
- `configs/` JSON configs (features, experiments, prompt evolution)
- `prompts/` Prompt templates
- `scripts/` PowerShell helpers (GUI launcher, cleanup helper)
- `docs/` Reports, paper stats, and summaries
- `features_storage/` Cached features and manifests
- `test_dataset/` Private test inputs and cached reasoning outputs
- `training_logs/`, `logging/` Run logs (older sweep logs archived under `_archive/`)

## Core pipelines and what they do
- `python/pipelines/vcbench_pipeline.py` Full public‑data pipeline: human features, LLM engineered, LLM reasoning, CV training, reports, and sweeps
- `python/pipelines/paper_pipeline.py` Paper‑only pipeline: 3x3 and 2x5 experiment blocks, CV summaries, and test‑set predictions
- `python/pipelines/prompt_evolution.py` Prompt evolution loop for reasoning prompts with CV evaluation
- `python/pipelines/reasoning_benchmark.py` Batch‑size and concurrency benchmark for reasoning generation
- `python/pipelines/reasoning_preview.py` Small live preview run with logging and schema checks

## Key modes and flags
`python/pipelines/vcbench_pipeline.py`
- `--dataset sample|full` Select dataset size
- `--mode human|llm|reasoning|hybrid` Control which feature sources are used
- `--feature_config configs/features.json` Load config defaults and feature list
- `--feature_set` or `--features` Manual feature selection
- `--llm_features` or `--llm_reasoning` Enable LLM features
- `--model_type logistic|xgboost` Choose model family for CV training
- `--cv_folds` `--cv_use_fixed_folds` `--cv_folds_path` CV control
- `--run_profile full|xgb_mirror|reasoning_xgb_oof` Run a specific profile only
- `--llm_sweep` `--llm_sweep_range` `--llm_sweep_repeats` Engineered rule sweep
- `--llm_reasoning_core_prompt` `--llm_reasoning_experiments` Prompt/experiment files
- `--llm_reasoning_batch_size` `--llm_reasoning_concurrency` Generation controls
- `--llm_reasoning_rate_limit_fallback_sequence` Rate‑limit backoff settings
- `--human_feature_source baseline|high_quality` Use baseline or HQ features

`python/pipelines/paper_pipeline.py`
- `--cv_folds` CV folds for paper runs
- `--test_csv` Path to private test CSV
- `--test_reasoning_parquet` Path to cached test reasoning parquet
- `--engineered_family_id` `--engineered_set_ids` Select engineered family/sets
- `--engineered_n_rules` `--n_engineered_sets` Control engineered generation when needed
- `--pt2_pred_combos` `--pt2_pred_models` Filter which part‑2 models write test predictions
- `--include_abcdef_test_preds` Include ABCDEF in test predictions

## Other Python utilities
- `python/lib/feature_registry.py` Custom feature formulas and named feature sets
- `python/tools/feature_selector_gui.py` GUI to select features and save a config
- `python/lib/llm_feature_generation.py` LLM rule generation and evaluation helper
- `python/lib/llm_reasoning_features.py` Core reasoning generation engine
- `python/lib/cv_folds.py` Fixed fold cache utilities
- `python/tools/rebuild_currently_in_use.py` Rebuild current reasoning cache from latest runs
- `python/tools/sync_currently_in_use.py` Sync latest runs to currently_in_use
- `python/tools/repair_run.py` Targeted NaN repair for a specific experiment run
- `python/tools/repair_run_a.py` Targeted NaN repair for run A
- `python/tools/backfill_confidence_from_runs.py` Backfill evidence_support_rating into current cache
- `python/tools/summarize_feature_weights.py` Coefficient summary report
- `python/tools/summarize_reasoning_weights_top10.py` Reasoning weight summary for top runs
- `python/tests/tests_reasoning_schema.py` Reasoning schema tests (no API)
- `python/tests/tests_prompt_evolution.py` Prompt evolution unit tests
- `python/tests/tests_smoke.py` Small smoke test helpers

## Mirror experiments (XGBoost baseline replication)
- `mirror_experiments/mirror_holdout_xgb.py` 80/20 holdout mirror with rule layer
- `mirror_experiments/mirror_cv_tuned_xgb.py` CV mirror with tuned threshold grid
- `mirror_experiments/mirror_test_predictions.py` Predict test CSV with mirror model
- `docs/paper_stats/mirror_parity_check.md` Test‑side parity note (paper pipeline vs mirror)

## Quick usage
```bash
python python/pipelines/vcbench_pipeline.py --dataset sample
python python/pipelines/vcbench_pipeline.py --dataset full --feature_config configs/features.json
python python/tools/feature_selector_gui.py
```

## Generated artifacts
Parquet outputs and logs are created under `features_storage/`, `training_logs/`,
and `logging/`. These are ignored by git via `LLM_Reasoning_Main/.gitignore`.

## Notes
- Baseline human feature reference is archived under `Archive/Old_Human_features_running_example_script/`.
- For LLM usage, set API keys in `.env` (e.g., `OPENAI_API_KEY`).
