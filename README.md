# Project_folder

This repository contains multiple workstreams and some third‑party copies. If you are new to the project, start here:

## Key folders
- **EDA/** — Exploratory data analysis and notebooks.
- **LLM_Reasoning_Main/** — Primary, current LLM reasoning + evaluation pipeline.

## Other folders (mostly copies / legacy work)
These are preserved for reference or provenance and are not the primary pipeline today:
- `Archive/` — archived legacy folders and prior pipelines.
- `High_Quality_human_features/` — upstream high‑quality human feature extraction (Structured v2 + HPO baselines).
- `VCBench-Starter-Kit/` — upstream starter kit and baseline training utilities for VCBench.
- `think-reason-learn/` — upstream research codebase providing LLM engineered feature utilities used in the pipeline.
- `think_reason_learn_docs/` — upstream docs/reference material for the above.

## Pipelines (where to look)
- **VCBench pipeline** (main training + evaluation): `LLM_Reasoning_Main/python/pipelines/vcbench_pipeline.py`.
  - Generates/loads features, runs CV, produces reports.
- **Paper pipeline** (paper‑grade evaluations + test predictions): `LLM_Reasoning_Main/python/pipelines/paper_pipeline.py`.
  - Uses fixed splits, OOF thresholds, and produces test‑set prediction CSVs.
- **LLM‑engineered n_rules sweep**: `LLM_Reasoning_Main/python/lib/llm_feature_generation.py` + sweep scripts in `LLM_Reasoning_Main/python/`.

## Notes
- Many of the non‑key folders are **copied from external repos** or earlier experiments; they remain for comparison and auditability.
- All current work should be documented under **LLM_Reasoning_Main/** and **EDA/**.
