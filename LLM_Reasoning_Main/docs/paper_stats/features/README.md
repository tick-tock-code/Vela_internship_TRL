# paper_stats/features

This folder contains feature artifacts duplicated for paper runs.

- human/
  - features_pool.parquet: pooled training features (if available)
  - test_raw.csv: private test input CSV (no success labels)

- llm_engineered/
  - set_01/engineered_pool.parquet
  - set_01/engineered_test.parquet
  - ... (set_02, set_03)
  - engineered_rules.json

- llm_reasoned/
  - reasoning_pool.parquet: pooled reasoning features (A/B/E/F as currently_in_use)
  - reasoning_test.parquet: merged private test reasoning (A/D/E/F)
  - exp_A/, exp_D/, exp_E/, exp_F/: per-experiment test outputs
