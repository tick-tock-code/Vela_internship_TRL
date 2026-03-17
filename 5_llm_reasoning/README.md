# VCBench Custom Feature Pipeline

This folder contains an in-depth pipeline that combines the 15 baseline features
from the example script with your own feature registry, with an option to add
LLM-derived features.

## Files

- `feature_registry.py`  
  Defines custom feature formulas and named feature sets.

- `vcbench_pipeline.py`  
  Loads VCBench, imports the baseline 15 features, adds custom features, can
  optionally add LLM features, saves a feature dataset, and trains/evaluates a
  scikit-learn Logistic Regression model.

- `feature_selector_gui.py`  
  A simple GUI to select features and save them to a JSON file.

## Custom features included

- QS tiers (binary): `qs_top_25`, `qs_top_50`, `qs_top_100`, `qs_top_200`
- Exits (binary): `prior_ipos`, `prior_acquisitions`
- Tenure (continuous): `large_company_years`

## Usage

Example (default feature set = base + custom):

```bash
python vcbench_pipeline.py --dataset sample
```

Override with a named set:

```bash
python vcbench_pipeline.py --dataset full --feature_set custom_only
```

Override with explicit custom features:

```bash
python vcbench_pipeline.py --dataset full --features qs_top_25,prior_ipos,large_company_years
```

Use the GUI to save a JSON feature list:

```bash
python feature_selector_gui.py
```

Then run the pipeline with:

```bash
python vcbench_pipeline.py --dataset full --feature_config features.json
```

## Generated artifacts

The pipeline writes Parquet outputs and training logs during runs. These files
are ignored by git via `3_pipeline_for_features/.gitignore`.

## Notes

- The baseline features are imported directly from:
  `2_Human_features_running_example_script\vcbench_lambda_features_minimal.py`
- The base script path can be overridden with `--base_script`.
- To include LLM-derived features, add `--llm_features` and ensure your API
  key is set in the environment (e.g., `OPENAI_API_KEY`).
