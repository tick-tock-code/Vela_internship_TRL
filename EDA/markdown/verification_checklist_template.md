# Verification Checklist Template

Date: YYYY-MM-DD
Owner: <name>
Project: <project name>
Run ID / Notebook: <path or run tag>

## 1) Inputs
- [ ] Dataset file exists: `<path>`
  - Log: size = ___ MB, rows = ___, cols = ___
- [ ] Expected columns present: `<col1>, <col2>, ...`
  - Log: missing columns = ___
- [ ] Missingness sanity check
  - Log: max missing % = ___ (column: ___)

## 2) Derived Features
- [ ] Derived columns created: `<colA>, <colB>, ...`
  - Log: non-null counts = ___
- [ ] Edge cases handled (empty lists / null JSON)
  - Log: sample row ids checked = ___

## 3) Visual Outputs
- [ ] Plots generated:
  - `<plot1.png>` [PASS/FAIL]
  - `<plot2.png>` [PASS/FAIL]
  - `<plot3.png>` [PASS/FAIL]
- [ ] Plots are non-empty (not blank)
  - Log: checked files = ___

## 4) Summary Outputs
- [ ] Summary file created: `<path>`
  - Log: lines = ___
- [ ] Includes >= N feature ideas
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
