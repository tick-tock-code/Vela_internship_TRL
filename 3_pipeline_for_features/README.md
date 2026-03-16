# VCBench Custom Feature Pipeline

This folder contains a custom pipeline that combines the 15 baseline features
from the example script with your own feature registry.

## Files

- `feature_registry.py`  
  Defines custom feature formulas and named feature sets.

- `vcbench_custom_pipeline.py`  
  Loads VCBench, imports the baseline 15 features, adds custom features, and
  trains/evaluates a logistic regression model.

## Custom features included

- QS tiers (binary): `qs_top_25`, `qs_top_50`, `qs_top_100`, `qs_top_200`
- Exits (binary): `prior_ipos`, `prior_acquisitions`
- Tenure (continuous): `large_company_years`

## Usage

Example (default feature set = base + custom):

```bash
python vcbench_custom_pipeline.py --input_csv C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\VCBench-Starter-Kit\vcbench_final_public_sample100.csv
```

Override with a named set:

```bash
python vcbench_custom_pipeline.py --input_csv C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\VCBench-Starter-Kit\vcbench_final_public.csv --feature_set custom_only
```

Override with explicit custom features:

```bash
python vcbench_custom_pipeline.py --input_csv C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\VCBench-Starter-Kit\vcbench_final_public.csv --features qs_top_25,prior_ipos,large_company_years
```

## Notes

- The baseline features are imported directly from:
  `2_Human_features_running_example_script\vcbench_lambda_features_minimal.py`
- The base script path can be overridden with `--base_script`.

