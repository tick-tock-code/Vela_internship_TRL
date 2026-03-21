# VCBench Custom Feature Pipeline

This folder contains an in-depth pipeline that combines the 15 baseline features
from the example script with your own feature registry, with an option to add
LLM-derived features.

## Structure

- `python/`  
  All Python entrypoints and utilities (pipelines, analyses, repairs, tests).
- `configs/`  
  JSON configs (features, experiments, prompt evolution).
- `prompts/`  
  Core prompt templates.
- `scripts/`  
  PowerShell helpers (e.g., GUI launcher).
- `docs/`  
  Reports, paper stats, and supporting notes.
- `features_storage/`, `test_dataset/`  
  Cached features and test artifacts.
- `training_logs/`, `logging/`  
  Run logs (older sweep logs archived under `_archive/`).

## Custom features included

- QS tiers (binary): `qs_top_25`, `qs_top_50`, `qs_top_100`, `qs_top_200`
- Exits (binary): `prior_ipos`, `prior_acquisitions`
- Tenure (continuous): `large_company_years`

## Usage

Example (default feature set = base + custom):

```bash
python python/vcbench_pipeline.py --dataset sample
```

Override with a named set:

```bash
python python/vcbench_pipeline.py --dataset full --feature_set custom_only
```

Override with explicit custom features:

```bash
python python/vcbench_pipeline.py --dataset full --features qs_top_25,prior_ipos,large_company_years
```

Use the GUI to save a JSON feature list:

```bash
python python/feature_selector_gui.py
```

Then run the pipeline with:

```bash
python python/vcbench_pipeline.py --dataset full --feature_config configs/features.json
```

## Generated artifacts

The pipeline writes Parquet outputs and training logs during runs. These files
are ignored by git via `5_llm_reasoning/.gitignore`.

## Notes

- The baseline features are imported directly from:
  `2_Human_features_running_example_script\vcbench_lambda_features_minimal.py`
- The base script path can be overridden with `--base_script`.
- To include LLM-derived features, add `--llm_features` and ensure your API
  key is set in the environment (e.g., `OPENAI_API_KEY`).
