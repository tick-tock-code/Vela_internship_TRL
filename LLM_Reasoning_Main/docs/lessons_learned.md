# Lessons Learned (LLM_Reasoning_Main)

## 2026-03-27
- ElasticNet sweep grids were not saved for the Base transform; reports can only show best-per-combo unless we persist full grids.
  - Fix: always write a sweep CSV whenever `logistic_tuning_mode=per_model`, regardless of transform.
  - Use that sweep CSV to render 2D (C × l1_ratio) tables.
- If a 2D hyperparameter sweep is part of the analysis, the markdown report must render the full grid (not just the chosen best).
- Transform sweeps (PCA/PLS/SFT) were implemented, but Base was excluded; this caused repeated rework.
- When a report expects a 2D hyperparameter grid, confirm the pipeline stores *all* grid points, not just the selected best.
- Long runs should be validated via a quick artifact check (CSV contains all grid points) before generating reports.
