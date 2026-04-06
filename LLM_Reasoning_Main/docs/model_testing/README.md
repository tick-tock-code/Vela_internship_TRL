# Model Testing Docs

This folder is now curated-only.

Retained here:

- `model_testing_notes.md`
- `summary*.md`
- `model_testing_report_*.md`
- final retained summary CSVs for the canonical finalising and sweep runs

Legacy raw run exports and bulky per-run artifacts that previously lived under `docs/model_testing/` were moved to:

- `.tmp/migration/docs_model_testing_legacy/`

The canonical reproducible legacy model-testing entrypoint is:

- `python/pipelines/model_testing_pipeline.py --preset canonical`

The canonical preset is:

- `configs/presets/model_testing_canonical.json`

That preset matches the report-linked finalisation run from March 31, 2026 and now reads canonical inputs from `data/vcbench/` while writing raw outputs to `.tmp/runs/model_testing/`.
