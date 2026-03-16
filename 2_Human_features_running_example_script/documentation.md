# VCBench Human-Only Run Documentation

This document explains how to run the human-only baseline and how to extend
the hand-engineered features.

## How to run (human-only)

Run the script with `--feature_mode human` and a VCBench CSV input.
Example (sample dataset):

```bash
python vcbench_lambda_features_minimal.py \
  --input_csv C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\VCBench-Starter-Kit\vcbench_final_public_sample100.csv \
  --feature_mode human
```

Example (full dataset):

```bash
python vcbench_lambda_features_minimal.py \
  --input_csv C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\VCBench-Starter-Kit\vcbench_final_public.csv \
  --feature_mode human
```

## Expected inputs

- A VCBench CSV file with founder profile data.
- The default label column is `success` (can be overridden via `--label_column`).

## Expected outputs

The script prints:
- The list of human features used.
- Class balance in train and test.
- Logistic regression coefficients for each feature.
- ROC-AUC, PR-AUC, precision, recall, F0.5, accuracy on test data.

## How to extend human features

1. Add new features in `_extract_human_features(record)`.
2. Keep each feature binary (0/1).
3. Update any logic that depends on `HUMAN_FEATURE_NAMES` (it is derived
   automatically from `_extract_human_features({})`).
4. Re-run the script in human-only mode to evaluate changes.

## Notes for smoke testing

- Use the sample CSV for a quick check.
- If metrics are unstable due to small positives, reduce test size or use
  the full dataset.

