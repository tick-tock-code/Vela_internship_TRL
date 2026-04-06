# LLM_Reasoning_Main

This repo now separates frozen legacy pipelines from the new instability-control direction.

## Active Structure

- `python/pipelines/legacy/`: frozen legacy implementations.
- `python/pipelines/`: thin compatibility wrappers that keep the original entrypoint filenames callable.
- `python/pipelines/instability_control/`: active family-diagnostics, admission, route-comparison, and final-report entrypoints.
- `python/lib/shared/`: reusable loading, folds, metrics, and artifact helpers.
- `python/lib/stability/`: instability-control logic.
- `configs/presets/`: canonical legacy presets.
- `configs/instability_control/`: active configs for the new methodology path.
- `data/vcbench/`: canonical datasets, fold caches, feature banks, and train/test reasoning caches.
- `.tmp/runs/`: raw run outputs.
- `docs/`: curated human-readable outputs only.

## Canonical Paths

- Public and private CSVs: `data/vcbench/raw/`
- Fold caches: `data/vcbench/folds/`
- Baseline and HQ features: `data/vcbench/features/human/`, `data/vcbench/features/human_high_quality/`
- Engineered families: `data/vcbench/features/llm_engineered/`
- Reasoning caches: `data/vcbench/features/llm_reasoning/`

Deprecated storage trees such as `features_storage/`, `test_dataset/`, `training_logs/`, and `logging/` were archived under `.tmp/migration/`.

## Legacy Entry Points

These filenames still work and now load canonical presets by default:

- `python/pipelines/vcbench_pipeline.py`
- `python/pipelines/paper_pipeline.py`
- `python/pipelines/model_testing_pipeline.py`
- `python/pipelines/prompt_evolution.py`
- `python/pipelines/reasoning_benchmark.py`
- `python/pipelines/reasoning_preview.py`
- `python/pipelines/sft_selection_summary.py`

Preset files live in `configs/presets/`.

## Active Instability-Control Entry Points

- `python/pipelines/instability_control/family_diagnostics.py`
- `python/pipelines/instability_control/family_admission.py`
- `python/pipelines/instability_control/route_comparison.py`
- `python/pipelines/instability_control/final_report.py`

Default configs live in `configs/instability_control/`.

## Docs

Start with `docs/README.md` for navigation and `docs/next_steps.md` for the week-one execution plan.
