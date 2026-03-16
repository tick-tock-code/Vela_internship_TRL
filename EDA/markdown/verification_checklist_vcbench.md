# Verification Checklist ? VCBench EDA

Date: ____
Owner: ____
Project: VCBench EDA
Run ID / Notebook: EDA\vcbench_day1_eda.ipynb

## 1) Inputs
- [ ] Dataset file exists: `VCBench-Starter-Kit\vcbench_final_public.csv`
  - Log: size = ___ MB, rows = ___, cols = ___
- [ ] Expected columns present: `founder_uuid, success, industry, ipos, acquisitions, educations_json, jobs_json, anonymised_prose`
  - Log: missing columns = ___
- [ ] Missingness sanity check
  - Log: max missing % = ___ (column: ___)

## 2) Derived Features
- [ ] Derived columns created
  - `ipos_count`, `acquisitions_count`, `ipos_count_bucket`, `acq_count_bucket`
  - `ipos_best_valuation_bucket`, `ipos_best_amount_bucket`, `acq_best_price_bucket`
  - `ipos_has_any`, `acq_has_any`, `acq_well_known_count`
  - `edu_count`, `edu_any_stem`, `edu_best_qs_bucket`
  - `job_count`, `job_leadership_count`, `job_large_company_count`, `job_total_duration_proxy`
  - `text_char_len`, `text_word_count`, `text_sentence_count`, `text_bullet_count`
  - Log: non-null counts = ___
- [ ] Edge cases handled (empty lists / null JSON)
  - Log: sample row ids checked = ___

## 3) Visual Outputs
- [ ] Plots generated:
  - `success_counts.png` [PASS/FAIL]
  - `missingness_by_column.png` [PASS/FAIL]
  - `industry_top.png` [PASS/FAIL]
  - `industry_success_rate.png` [PASS/FAIL]
  - `ipos_acquisitions_count_hist.png` [PASS/FAIL]
  - `ipos_best_valuation_bucket.png` [PASS/FAIL]
  - `ipos_best_amount_bucket.png` [PASS/FAIL]
  - `acq_best_price_bucket.png` [PASS/FAIL]
  - `ipos_count_success_rate_with_freq.png` [PASS/FAIL]
  - `acq_count_success_rate_with_freq.png` [PASS/FAIL]
  - `edu_job_counts.png` [PASS/FAIL]
  - `edu_qs_bucket.png` [PASS/FAIL]
  - `company_size_role_counts.png` [PASS/FAIL]
  - `job_large_company_count.png` [PASS/FAIL]
  - `qs_success_rate.png` [PASS/FAIL]
  - `text_proxy_hists.png` [PASS/FAIL]
  - `text_length_success_rate.png` [PASS/FAIL]
  - `correlation_matrix.png` [PASS/FAIL]
- [ ] Plots are non-empty (not blank)
  - Log: checked files = ___

## 4) Summary Outputs
- [ ] Summary file created: `EDA\eda_summary.md`
  - Log: lines = ___
- [ ] Includes >= 5 feature ideas
  - Log: count = ___

## 5) Sanity Checks / Spot Review
- [ ] Inspect 5 random rows for JSON parsing correctness
  - Log: row ids = ___, issues = ___
- [ ] Quick distribution check (counts not all zero)
  - Log: feature = ___, min/mean/max = ___

## Notes
- Issues found: ___
- Fixes applied: ___
- Follow-ups: ___
