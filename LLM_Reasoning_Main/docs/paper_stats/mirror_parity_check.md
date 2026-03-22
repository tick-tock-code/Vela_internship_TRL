# Mirror Parity Check (Test‑Side)

Last updated: 2026-03-21

This note summarizes the **test‑side parity** between the paper pipeline and the mirror pipeline used for the HQ XGBoost baseline.

## Scope
- **Mirror pipeline:** `mirror_experiments/mirror_test_predictions.py`
- **Paper pipeline:** `python/pipelines/paper_pipeline.py`
- **Focus:** Test‑set feature generation and model inputs (not training).

## ✅ Confirmed Parity

### 1) Raw CSV parsing
- **Mirror:** `pd.read_csv(test_csv)`  
- **Paper:** `pd.read_csv(test_csv)` → then `_safe_json_parse` for JSON fields (reasoning/engineered only).

✅ **Same raw test DataFrame** for HQ features.

### 2) HQ feature extraction
- **Mirror:** calls `High_Quality_human_features/features/extract_structured.py::extract_features` on the raw test DataFrame.
- **Paper:** calls the **same extractor** on the raw test DataFrame via `_extract_hq_from_raw_df`.

✅ **HQ feature matrix identical** (same 28 columns + `fillna(0.0)`).

### 3) Rule layer
- **Mirror:** `apply_rule_override(... exit_count ...)`
- **Paper:** Full‑Mirror variants apply the **same rule override**.

✅ **Rule logic identical.**

## ✅ Reasoning features (paper only)
The mirror pipeline does **not** generate reasoning features.  
Paper pipeline uses:
- **Same core prompt:** `prompts/core_prompt.txt`
- **Same experiments file:** `configs/experiments.json`
- **Same batch settings:** size 20, concurrency 10
- **Same parsing:** `_safe_json_parse` for JSON fields
- **Regen guard:** `llm_reasoning_private_meta.json` stores `parse_version` + `records_hash`

## ✅ Result
Test‑side HQ + rule inputs are **mirror‑equivalent**.  
Any differences between mirror predictions and paper predictions are therefore due to:
1) Reasoning features being appended (paper only), or  
2) Threshold grid differences (if configured).

