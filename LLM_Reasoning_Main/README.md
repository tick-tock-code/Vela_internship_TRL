# LLM_Reasoning_Main

This repo now separates frozen legacy pipelines from an evidence-first instability-control workflow.

## Active Structure

- `python/pipelines/legacy/`: frozen legacy implementations.
- `python/pipelines/`: thin compatibility wrappers that keep original entrypoint filenames callable.
- `python/pipelines/instability_control/evidence_map.py`: primary Step 1 evidence-mapping entrypoint.
- `python/pipelines/instability_control/method_benchmark.py`: scaffold entrypoint for the next mathematical-method stage.
- `python/pipelines/instability_control/reasoning_block_pls.py`: Step 3 blockwise PLS on reasoning families before HQ concatenation.
- `python/pipelines/instability_control/supervised_grouping.py`: Step 4 supervised grouping with train-only clustering and 1-component cluster PLS.
- `python/pipelines/instability_control/status_report.py`: current-state synthesis for the instability-control track.
- `python/lib/shared/`: reusable loading, folds, metrics, and artifact helpers.
- `python/lib/stability/`: family registry, combo catalog, route evaluators, evidence synthesis, and method scaffold logic.
- `configs/instability_control/`: active configs for atomic families, legacy combos, route controls, reporting, and method order.
- `data/vcbench/`: canonical datasets, fold caches, feature banks, and train/test reasoning caches.
- `.tmp/runs/`: raw run outputs.
- `docs/`: curated human-readable outputs only.

## Instability-Control Workflow

1. Anchor continuity:
   `HQ_anchor_xgb1_unpruned` stays frozen as the benchmark anchor.
2. Step 1 evidence map:
   reproduce the raw failure pattern and transformed `PLS` wins cleanly.
3. Step 2 stability analysis:
   preserve the row-subsampled stability-selection reference on `HQ + A-F`.
4. Step 3 blockwise PLS:
   compress the reasoning block inside each outer fold before joining it to raw `HQ`.
5. Step 4 supervised grouping:
   cluster correlated features inside each outer fold, then collapse each cluster to one latent route feature.
6. Final synthesis:
   maintain one paper-facing summary of what fails raw, what is transform-sensitive, and what to try next.

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

## Docs

Start with `docs/README.md` for navigation and `docs/instability_control/README.md` for the active study path.
