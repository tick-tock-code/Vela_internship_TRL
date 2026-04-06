# Paper Stats Docs

This folder is now summary-only.

Retained here:

- `paper_pipeline_summary.md`
- `paper_pipeline_report.md`
- `paper_pipeline_results.csv`
- `experiment_top_picks.md`
- `mirror_parity_check.md`
- `paper_structure.md`
- `engineered_top3_xgb.json`
- `prediction_verification_report.json`

Detailed per-part outputs, feature dumps, reasoning logs, test predictions, and run logs were moved to:

- `.tmp/migration/docs_paper_stats_detailed/`

The canonical reproducible legacy paper entrypoint is:

- `python/pipelines/paper_pipeline.py --preset canonical`

The canonical preset is:

- `configs/presets/paper_canonical.json`
