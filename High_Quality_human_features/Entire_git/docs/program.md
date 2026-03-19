# program.md — VCBench Phase 4 Overnight Loop

## Current best F₀.₅: 0.3030 (val) / 0.2539 (CV) — XGB stumps + prior_exit rule
## CV method: 5-fold stratified (with rules applied in CV)
## Target: > 0.28 by end of overnight loop. Stretch: > 0.33.
## Status: 120+ experiments run. Structured features plateau at CV ~0.25.

## Observations (do not change these — read and use them)

**What separates TPs from FPs (from manual inspection, N=900 val rows):**
- exit_count: TP avg 0.58 vs FP avg 0.11 — strongest signal
- founding_role_count: TP avg 2.22 vs FP avg 1.84 — serial founding matters
- edu_prestige_tier: TP avg 2.78 vs FP avg 3.10 — INVERTED, prestige is FP noise
- prestige_sacrifice_score: TP avg 21.31 vs FP avg 25.22 — INVERTED, also noise
- The model predicted 20.4% positive at threshold 0.512; after disabling bad rules and raising threshold to 0.923, it predicts 4% positive with much better precision (36.1%)

**Rule layer status:**
- prior_exit: ACTIVE — precision 24.5% on train (2.7x base rate). Fires on ~3% of rows.
- top10_stem_founder: DISABLED — FP amplifier in credential-dense industries.
- clevel_serial_founder: DISABLED — 11.1% precision (barely above 9% base rate).

**FP cluster:** At high threshold (0.923), FPs are distributed across Technology/Internet (5), NaN (4), Software Dev (3). NOT concentrated in biotech/VC as originally hypothesized — that was a lower-threshold artifact.

**Calibration:** Disabling Rules 2+3 improved F0.5 from 0.2203 to 0.2889 (+6.9pp). Platt scaling on same val set is invalid — use 5-fold CV for proper calibration.

**v2 features available (added but not yet in classifier.py):**
- is_serial_founder: binary, 26.2% of founders. Success rate 9.9% vs 8.7%.
- exit_x_serial: exit_count * founding_role_count.
- sacrifice_x_serial: prestige_sacrifice_score * is_serial_founder.
- industry_prestige_penalty: edu_prestige_tier * is_biotech_or_vc.
- persistence_score: longest_founding_tenure / (total_inferred_experience + 0.01). Range 0-1.
- repeat_founding_gap: DROPPED (73.8% null rate).

## Constraints — never violate

- Only modify `classifier.py`. Never modify `evaluate.py`, val split, or train split.
- Every experiment logs to `experiment_log.md` before the next one starts.
- Revert if CV F₀.₅ does not improve.
- random_state=42 everywhere. Do not change.
- No features derived from `anonymised_prose`.
- Do not run `predict.py` on private test set.
- Do not declare done without printing final CV F₀.₅ and val F₀.₅.

## Seed hypotheses — start here, then generate your own from the log

1. Add v2 features to FEATURE_COLS: is_serial_founder, exit_x_serial, sacrifice_x_serial, industry_prestige_penalty, persistence_score. Retrain with same hyperparams.
2. Remove edu_prestige_tier + best_degree_prestige entirely (null hypothesis test — FPs have HIGHER scores).
3. Remove prestige_sacrifice_score (FPs have higher scores than TPs — net-negative?).
4. Increase scale_pos_weight from 10 to 20 — model was predicting 20.4% positive at old threshold, true rate is 9%.
5. Decrease scale_pos_weight to 5 — with high threshold, maybe less class weight is better.
6. Raise threshold to 0.80–0.95 range — F₀.₅ rewards precision > recall.
7. Try max_depth=3 (simpler trees, less overfitting).
8. Try max_depth=6 (deeper trees, more complex interactions).
9. Try n_estimators=500 with learning_rate=0.03 (more trees, lower learning rate).
10. Switch to LightGBM — may produce better-calibrated probabilities.
11. Test minimal feature set: exit_count + founding_role_count + industry_alignment + is_serial_founder only.
12. Try removing the rule layer entirely — let the model handle everything including exits.
13. Try LogisticRegression instead of XGBoost — simpler model may generalize better on small data.
14. Add feature interactions manually: exit_count * industry_alignment, founding_role_count * persistence_score.
15. Try adding colsample_bytree=0.5 (more regularization).
16. Test whether comfort_index adds or subtracts from F0.5 (ablation).
17. Try gamma=1 or gamma=5 in XGBoost (minimum loss reduction for split).
18. Try min_child_weight=5 or 10 (more conservative splits).

After exhausting this list, generate new hypotheses by reading which experiments improved
F₀.₅ and what they have in common. Focus on feature interactions and threshold tuning,
not on adding more raw features.
