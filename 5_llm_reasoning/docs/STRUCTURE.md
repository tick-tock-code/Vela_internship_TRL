# 5_llm_reasoning layout

This folder is organized to keep code, configs, prompts, and outputs separate.

## Core entrypoints (Python)
- `python/vcbench_pipeline.py`
- `python/paper_pipeline.py`
- `python/llm_reasoning_features.py`
- `python/llm_feature_generation.py`
- `python/prompt_evolution.py`
- `python/reasoning_benchmark.py`
- `python/reasoning_preview.py`

## Configuration & prompts
- `configs/` — JSON configs (features, experiments, prompt evolution)
- `prompts/` — prompt templates

## Scripts
- `scripts/` — helper PowerShell scripts (GUI launcher, cleanup helper)

## Outputs / data
- `features_storage/` — cached features and manifests
- `test_dataset/` — private test inputs + cached reasoning outputs
- `training_logs/` and `logging/` — logs (older sweep logs archived under `_archive/`)

## Notes
- `docs/` — reports, paper stats, and summaries
- `archive/` — legacy runs and older notes
