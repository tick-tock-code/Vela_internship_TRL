# Paper Structure Notes

We evaluate a minimal reasoning baseline (A), a small additive extension (A+E), a rubric-style extension (A+F), a richer multi-signal model (ADEF), and the maximal combination (ABCDEF). This provides a graded complexity spectrum while avoiding post-hoc selection.

## Part 1: Pool experiments (4,400)
- Human-only baselines and Human + reasoning combos.
- LLM-engineered baselines and LLM-engineered + reasoning combos.
- Reported for both Logistic Regression and XGBoost.

## Part 2: HQ mirror experiments (4,500)
- HQ features + reasoning additions on the full dataset.
- Full mirror (rule layer) + reasoning additions on the full dataset.
- Reported for both Logistic Regression and XGBoost.

## Test-set plan (private)
- HQ Only (baseline)
- HQ + A (minimal reasoning)
- HQ + A+E (incremental improvement)
- HQ + A+F (stronger gain, more complexity)
- HQ + A+D+E+F (best-performing complex version)
- HQ + A+B+C+D+E+F (only if added complexity appears to help)

## Private test addendum
- Run 3x human-only feature sets with XGBoost.
- Run top 3 LLM-engineered sets with XGBoost.
- Run top 3 LLM-engineered + A+E with XGBoost.
