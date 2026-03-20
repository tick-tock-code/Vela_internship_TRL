# Experiment Log — VCBench Task IV

All experiments are logged here in append-only format. Never delete entries.

---

## Step 1 — 2026-03-08
**Change:** Created `evaluate.py` (FIXED — never modify)
**Smoke test result:** f05=1.0, precision=1.0, recall=1.0, positive_rate=0.09, n_predicted_positive=9, threshold=0.804
**Verdict:** PASS — evaluate.py locked.

## Step 2 — 2026-03-08
**Change:** Created `data/split.py` — 80/20 stratified train/val split (random_state=42)
**Dataset:** 4,500 rows, 9.0% positive (405 success / 4,095 failure)
**Train:** 3,600 rows, 9.0% positive
**Val:** 900 rows, 9.0% positive
**Output:** `data/public_train.csv`, `data/public_val.csv`
**Verdict:** PASS — split locked. Do not re-run.

## Step 3 — 2026-03-08
**Change:** Created `features/extract_structured.py` — Tier 1–4 feature engineering (23 features)
**Null rates:** 0.0% across all 23 features
**Key findings:**
- exit_count: 106 founders with exits (2.9%). Success rate: 22.8% (1 exit), 60.0% (2 exits) vs 8.5% baseline
- edu_prestige_tier: top-10=421, top-50=479, top-100=219, ranked=2066, null=415
- max_seniority_reached: 67% are founder/C-level (5)
- prestige_sacrifice_score: good spread 0–45
**Bug fixed:** ipos/acquisitions fields use Python single-quote dict syntax, not JSON. Added `ast.literal_eval` fallback.
**Verdict:** PASS — all features extracted, no nulls.

## Step 3b — 2026-03-08
**Change:** Created `tests/test_extract_structured.py` — 90 unit + integration tests for all 23 features
**Bug found & fixed:** `_get_seniority` had substring false positives — "cto" matched inside "director", "ceo" inside "director". Fixed with word-boundary regex matching. Also moved junior/intern check before IC-level to prevent "Junior Developer" matching "developer" first.
**Test results:** 90/90 passed (2.77s)
**Output:** `features/FEATURE_SUMMARY.md` — full distribution and success-rate-by-value for all 23 features
**Verdict:** PASS — all features validated. See `features/FEATURE_SUMMARY.md` for detailed distributions.

## Step 5 — 2026-03-08
**Change:** Zero-shot Claude Sonnet 4.5 baseline on 900 val rows (`anonymised_prose` only)
**Model:** claude-sonnet-4-5-20250929
**F0.5:** 0.1265 (prev best: none — this is the first baseline)
**Precision:** 0.105
**Recall:** 0.7037
**Threshold:** 0.327
**Positive rate:** 60.3% (543/900 predicted positive)
**Notes:** Model outputs are poorly calibrated — 60% of predictions cluster at exactly 0.50, almost nothing above 0.60. The model predicts way too many positives, yielding very low precision. This is significantly worse than the GPT-4o zero-shot baseline on the leaderboard (25.7% F0.5). Possible causes: (1) Sonnet 4.5 is overly generous in success probability estimates, (2) the system prompt calibration is insufficient, (3) the anonymised prose doesn't contain enough discriminative signal for zero-shot approaches. This baseline strongly motivates the structured feature engineering approach.
**Verdict:** LOGGED — baseline established at F0.5=0.1265. All subsequent models must beat this.

## Step 4 — 2026-03-08
**Change:** Created `features/high_precision_rules.py` — deterministic rule layer
**Rule performance on training set:**
- Rule 1 (prior_exit): fired=106, precision=24.5% — KEPT
- Rule 2 (top10_stem_founder): fired=134, precision=21.6% — KEPT
- Rule 3 (clevel_large_company_founder): fired=783, precision=10.0% — DISABLED (below 30% threshold)
**Verdict:** Rules 1 & 2 active. Rule 3 disabled due to low precision.

## Step 6 — 2026-03-08
**Change:** Created `classifier.py` — XGBoost on 23 structured features (Tiers 1–4) + rule layer
**F0.5:** 0.2203 (prev best: 0.1265 zero-shot)
**Precision:** 0.1957
**Recall:** 0.4444
**Threshold:** 0.512
**Positive rate:** 20.4% (184/900)
**Rule layer overrides:** prior_exit=36, top10_stem_founder=35
**Top 5 features by importance:**
1. exit_count (0.078)
2. edu_prestige_tier (0.076)
3. best_degree_prestige (0.070)
4. industry_alignment (0.055)
5. prestige_sacrifice_score (0.046)
**Notes:** Beats zero-shot baseline by +9.4pp. Below Phase 2 target of 28%. Feature importances show exit signals and education prestige dominate. Sacrifice signal (prestige_sacrifice_score) is in top-5. The threshold sweep shows F0.5 is fairly flat across 0.50–0.95, suggesting the model lacks confidence separation. Next steps: hyperparameter tuning, feature interactions, or threshold calibration.
**Verdict:** KEEP — first structured model. Needs improvement to hit 28% target.

## Step 7 — 2026-03-08
**Change:** Manual inspection of val set predictions (TP=36, FN=45, FP=148, TN=671)

**True Positives (36) — what the model gets right:**
Most TPs are caught by the rule layer (exit_count=1 or top10+STEM+founder). The model correctly identifies founders with prior exits, top-10 QS education, and high prestige_x_relevance scores. Average TP profile: exit_count=0.58, edu_prestige_tier=2.78, prestige_x_relevance=11.67. These are founders with clear, measurable pedigree signals.

**False Negatives (45) — successful founders the model misses:**
FNs tend to have low edu_prestige_tier (avg ~1.3), no exits, and lower prestige scores. Many are serial operators at 200+ QS schools with deep domain expertise — their signal comes from career trajectory patterns (long tenures, domain focus) that the current features don't weight enough. Several FNs have high prestige_sacrifice_score (45) but low education prestige — the sacrifice signal alone isn't enough without education prestige to boost it.

**False Positives (148) — the precision problem:**
The FP rate is the core issue (148 FPs vs 36 TPs). Many FPs look indistinguishable from TPs on our features — top-10 QS, STEM, founder role, high sacrifice score. The rule layer is responsible for many FPs: the top10_stem_founder rule fires on 35 val rows but many are failures. The model struggles to separate "impressive resume" from "actually successful founder." FPs cluster in biotech/research and VC/PE industries where elite credentials are common but success rates are still low.

**Feature comparison across groups:**
| Feature | TP avg | FN avg | FP avg |
|---|---|---|---|
| exit_count | 0.58 | 0.04 | 0.11 |
| edu_prestige_tier | 2.78 | 1.31 | 3.10 |
| prestige_sacrifice_score | 21.31 | 18.09 | 25.22 |
| industry_alignment | 0.47 | 0.33 | 0.38 |
| prestige_x_relevance | 11.67 | 3.91 | 13.24 |
| stem_flag | 0.56 | 0.40 | 0.68 |
| max_seniority_reached | 4.64 | 3.67 | 4.78 |
| founding_role_count | 2.22 | 1.11 | 1.84 |

**Key insight:** FPs actually have HIGHER edu_prestige_tier, prestige_sacrifice_score, and prestige_x_relevance than TPs. The model is over-indexing on education prestige. What separates TPs from FPs is primarily exit_count (0.58 vs 0.11) and founding_role_count (2.22 vs 1.84). The rule layer's top10_stem_founder rule is too aggressive.

**Hypotheses for Phase 4:**
1. Tighten the top10_stem_founder rule (add more conditions or disable it)
2. Add industry-specific base rate features (biotech/research has lower success despite elite credentials)
3. Add serial founder signal — founding_role_count interacted with tenure
4. Reduce scale_pos_weight to lower recall and boost precision
5. Add feature: ratio of founding tenure to total experience (commitment signal)

## Step 3c — 2026-03-08
**Change:** Added 6 v2 features to `features/extract_structured.py` (now 29 features total)
**New features:**
- `is_serial_founder`: binary (founding_role_count >= 2). 26.2% serial founders. Success rate: 9.9% vs 8.7% for non-serial.
- `exit_x_serial`: exit_count * founding_role_count. Sparse — most are 0.
- `sacrifice_x_serial`: prestige_sacrifice_score * is_serial_founder. Conditions sacrifice on serial history.
- `industry_prestige_penalty`: edu_prestige_tier * is_biotech_or_vc. Discounts credential-dense industries.
- `persistence_score`: longest_founding_tenure / (total_inferred_experience + 0.01). Range 0–1, median 0.064.
- `repeat_founding_gap`: **DROPPED** — 73.8% null rate (above 35% threshold). Only computable for serial founders.
**Null rates:** 0.0% for all 5 kept features.
**Tests:** 90/90 still passing.
**Verdict:** 5 v2 features added (repeat_founding_gap dropped). Total usable features: 28.

## Step 4b — 2026-03-08
**Change:** Updated `features/high_precision_rules.py` to v2 — disabled Rules 2 and 3
**Rule validation on training set (v2):**
- Rule 1 (prior_exit): fired=106, precision=24.5% — KEPT (2.7x base rate, strongest signal)
- Rule 2 (top10_stem_founder): DISABLED — FP amplifier in biotech/VC/PE
- Rule 3 (clevel_serial_founder): fired=431, precision=11.1% — DISABLED (barely above 9% base rate)
**Verdict:** Only prior_exit rule active. Rules 2+3 disabled.

## Step 6b — 2026-03-08
**Change:** Applied Platt scaling to existing model.pkl (Karpathy priority #1)
**Raw model (only prior_exit rule):** F0.5=0.2889, P=0.3611, R=0.1605, threshold=0.923
**Calibrated (Platt on same val set):** F0.5=0.0 — collapsed distribution, invalid (fitting and evaluating on same data)
**Key finding:** Disabling Rules 2+3 improved F0.5 from 0.2203 to 0.2889 (+6.9pp). The rule layer was injecting more noise than signal. High threshold (0.923) means the model is being very selective — only 36 predicted positive (4% positive rate).
**Platt scaling note:** Same-set calibration is invalid. Proper calibration requires cross-validation (5-fold CV in Phase 4).
**Verdict:** Rule layer fix confirmed. Platt scaling deferred to 5-fold CV approach.

## Step 6c — 2026-03-08
**Change:** Industry-stratified precision analysis on val set (a16z recommendation)
**Threshold used:** 0.923 (optimal from Step 6b)
**Overall:** F0.5=0.2889, P=0.3611, R=0.1605, 36 predicted positive (4% rate)

**Top FP-generating industries (at threshold=0.923):**

| Industry | N | FP | TP | Precision | Base rate |
|---|---|---|---|---|---|
| Technology/Internet Platforms | 100 | 5 | 2 | 28.6% | 11.0% |
| (NaN / missing industry) | 43 | 4 | 2 | 33.3% | 7.0% |
| Software Development | 127 | 3 | 1 | 25.0% | 11.8% |
| Internet Media & Publishing | 5 | 2 | 0 | 0.0% | 20.0% |
| VC & Private Equity | 14 | 1 | 2 | 66.7% | 14.3% |
| Biotech & Nanotech Research | 42 | 1 | 1 | 50.0% | 19.0% |

**Key findings:**
- At high threshold (0.923), FPs are distributed across industries, not concentrated in biotech/VC as expected.
- Biotech/VC actually have decent precision (50-67%) at this threshold — the FP cluster was at lower thresholds.
- Technology/Internet (5 FP) and Software Dev (3 FP) generate the most FPs — these are the largest industry groups.
- Only 36 predictions total at this threshold, making per-industry analysis sparse.
- 23 of the 36 predictions come from the prior_exit rule layer override.
**Verdict:** Industry-stratified thresholds may help at lower thresholds but are less relevant at 0.923. Log for Phase 4 experimentation.

---

# Phase 4 Overnight Loop — Experiments

## Experiment 8 — 2026-03-08
**Change:** Phase 4 baseline — v2 features + 5-fold CV + only prior_exit rule
**Hypothesis:** Adding v2 interaction features (is_serial_founder, exit_x_serial, sacrifice_x_serial, industry_prestige_penalty, persistence_score) should improve discrimination
**CV F₀.₅:** 0.2025 ± 0.0318 (prev best: N/A — first CV measurement)
**Val F₀.₅:** 0.2941 (prev best: 0.2889)
**Precision:** 0.3269
**Recall:** 0.2099
**Threshold:** 0.751
**Top 3 features:** edu_prestige_tier (0.065), best_degree_prestige (0.058), exit_x_serial (0.047)
**Verdict:** KEEP — Phase 4 baseline established. CV is ~9pp below val, indicating high variance.
**Notes:** exit_x_serial entered top-3 importance. edu_prestige_tier still dominates despite being noise. Need to test removing it.

## Experiments 9–100 — 2026-03-08 (Phase 4 batch)
**Change:** Systematic hyperparameter sweep, feature ablations, model comparisons
**Total experiments:** 100+
**Hypotheses tested:**
- Remove edu_prestige features (Exp9): CV=0.1839 — WORSE, features help despite noise
- Remove prestige_sacrifice_score (Exp10): CV=0.1992 — WORSE
- scale_pos_weight sweep (Exp11-12,16,70): spw=5 is marginally best but within noise
- max_depth sweep (Exp13-14,24-25,67,94): depth=4 remains best
- min_child_weight (Exp18,23,29,48): **mcw=5 is a major win, CV=0.2330** (+3pp from baseline)
- Rule layer in CV (Exp33): **CV jumps from 0.2330 to 0.2506** (+1.8pp)
- Two-stage model (Exp71-76): WORSE than single model
- LightGBM (Exp36,109-111): **LGB num_leaves=15 reaches CV=0.2525**
- RandomForest (Exp63,65): CV=0.2361-0.2406
- LogisticRegression (Exp35): CV=0.2277 (competitive!)
- Ensemble XGB+LGB (Exp59,103): CV=0.2463 (no improvement)
- Custom sample weights (Exp101-102): WORSE
- Derived features in classifier (Exp83-85): No improvement
- Permutation importance: sacrifice_x_serial, field_relevance_score, industry_pivot_count are harmful
- Bayesian optimization via Optuna (200 trials each): See below.

**Key finding:** All models converge on **max_depth=1 (decision stumps)** with heavy regularization as the optimal architecture. The dataset is too small (405 positives) for complex models.

## Experiment 120 — 2026-03-08 (Optuna best)
**Change:** Optuna Bayesian optimization — 200 trials XGB + 200 trials LGB
**Hypothesis:** Systematic hyperparameter search may find a better configuration than manual tuning
**XGB best params:** n_estimators=227, max_depth=1, lr=0.0674, subsample=0.949, colsample_bytree=0.413, scale_pos_weight=10, mcw=14, gamma=4.19, reg_alpha=0.73, reg_lambda=15.0
**LGB best params:** n_estimators=612, max_depth=1, lr=0.013, num_leaves=45, subsample=0.849, colsample=0.762, spw=10, mcw=10, reg_alpha=1.11, reg_lambda=9.55
**CV F₀.₅ (XGB):** 0.2549 ± 0.0291 (prev best: 0.2506)
**CV F₀.₅ (LGB):** 0.2589 ± 0.0308
**Val F₀.₅ (XGB):** 0.3030, P=0.3333, R=0.2222, threshold=0.738, n_pos=54
**Val F₀.₅ (LGB):** 0.2889, P=0.3611, R=0.1605
**Ensemble (XGB+LGB) CV:** 0.2560, Val=0.2949
**Final config chosen:** XGB stumps (spw=10 variant) — best val F0.5 at 0.3030
**Top 5 features:** edu_prestige_tier (0.121), best_degree_prestige (0.100), prestige_x_relevance (0.088), prestige_sacrifice_score (0.084), exit_x_serial (0.063)
**Verdict:** KEEP — best configuration found. CV=0.2539, Val=0.3030.

## Phase 4 Summary — 2026-03-08
**Total experiments run:** 120+
**Best CV F₀.₅:** 0.2539 ± 0.0348 (XGB stumps, spw=10, with rules)
**Best Val F₀.₅:** 0.3030 (P=0.3333, R=0.2222, threshold=0.738)
**Phase 2 target (0.28 CV):** NOT reached. Best CV is 0.2539. Gap: ~2.6pp.
**Phase 4 target (0.33 CV):** NOT reached. Gap: ~7.6pp.

**Root cause analysis:**
- 84% of positives (68 of 81 on val) have NO prior exits. The rule layer catches the easy 16%.
- Non-exit positives are statistically almost identical to failures on all structured features.
- The strongest non-exit separator is industry_alignment (1.38x ratio) — too weak alone.
- The structured JSON fields contain limited discriminative signal for non-exit success.
- All models (XGB, LGB, RF, LR) and all architectures (deep trees, stumps, ensembles, stacking) converge on the same ~0.25 CV ceiling.

**Recommendations for next phase:**
1. Phase 3 (LLM feature extraction) is the most promising path forward — the prose may contain signal not captured in structured fields (e.g., domain expertise depth, career narrative type).
2. Industry-specific threshold tuning at lower thresholds.
3. Phase 5 ensemble with SageMaker HPO on the full hyperparameter space.

---

## Phase 3 — LLM Feature Extraction (2026-03-08)

### Step 9: LLM feature extraction from anonymised_prose

**Extraction model:** claude-haiku-4-5-20251001
**Extraction fields:** prior_founding_attempt, domain_expertise_depth, highest_seniority_reached, evidence_of_prior_exit, career_narrative_type, domain_focus_consistency, conviction_indicator
**Mapped to 9 classifier features:** llm_prior_founding, llm_domain_expertise, llm_prior_exit, llm_narrative_builder, llm_narrative_climber, llm_seniority_founder, llm_seniority_clevel, llm_domain_focus, llm_conviction

**Coverage:** 3,034/4,500 (67.4%) — API credit balance exhausted midway. 1,466 rows use default values (3 for numeric scales, 0 for binary).
- Train coverage: 2,464/3,600 (68.4%)
- Val coverage: 570/900 (63.3%)

**Signal analysis (train set, covered rows only):**
- llm_prior_exit: 8.5% True (success) vs 2.4% (failure) — 3.5x ratio (strongest)
- llm_domain_focus: 3.12 (success) vs 2.61 (failure) — +0.51 gap
- llm_domain_expertise: 3.32 vs 2.85 — +0.47 gap
- llm_conviction: 2.97 vs 2.88 — weak (+0.09)
- llm_prior_founding: 36.8% vs 37.4% — no signal

### Experiment: Add 9 LLM features to locked XGB stump config

**Change:** Added 9 LLM features to FEATURE_COLS (28 → 37 features). Locked model config from Experiment 120.
**Hypothesis:** LLM-extracted features from prose capture signal invisible to structured JSON (domain focus, conviction, prior founding).
**CV F₀.₅:** 0.2612 ± 0.0275 (prev best: 0.2539 ± 0.0348)
**Val F₀.₅:** 0.2889 (prev best: 0.3030)
**Precision:** 0.3611 (val)
**Recall:** 0.1605 (val)
**Threshold:** 0.897
**CV delta:** +0.0073 (+0.73pp)
**Val delta:** -0.0141 (-1.41pp)

**Feature importance ranking for LLM features:**
- #3: llm_domain_focus (0.0783) — strongest LLM feature
- #4: llm_domain_expertise (0.0628)
- #5: llm_prior_exit (0.0564)
- #17: llm_conviction (0.0255)
- #21: llm_seniority_clevel (0.0193)
- #23: llm_seniority_founder (0.0148)
- #27-29: llm_narrative_builder, llm_narrative_climber, llm_prior_founding (0.000 — unused by model)
- LLM features capture 25.7% of total feature importance

**Decision gate:** CV delta 0.0073 < 0.01 threshold → **NO IMPROVEMENT**

**Verdict:** REVERT (per decision gate)

**Notes:**
- 33% of rows have default LLM values due to API credit exhaustion — this adds noise and dilutes signal. The true delta with full coverage may be higher.
- CV std improved from 0.0348 to 0.0275 (more stable), suggesting LLM features add some regularization signal.
- Val F₀.₅ dropped 1.4pp despite CV improvement — possible that default values for the 37% uncovered val rows hurt the val model more than the 32% uncovered train rows hurt CV.
- Three LLM features (llm_domain_focus, llm_domain_expertise, llm_prior_exit) ranked in top 5 feature importances — the model WANTS to use them but coverage is insufficient.
- If API credits are topped up and full extraction completes, re-run this experiment. The 0.73pp CV delta with 33% missing data is suggestive of a true delta potentially crossing the 1pp gate.

### Re-run: Full coverage (4,500/4,500, 0 nulls)

**Change:** Re-ran Phase 3 experiment after completing extraction to 100% coverage. Same 9 LLM features, same locked XGB stump config.
**Coverage:** 4,500/4,500 (100%) — train 3,600/3,600, val 900/900.
**CV F₀.₅:** 0.2534 ± 0.0305 (baseline: 0.2539 ± 0.0348)
**Val F₀.₅:** 0.3065 (baseline: 0.3030)
**Precision:** 0.3556 (val)
**Recall:** 0.1975 (val)
**Threshold:** 0.778
**CV delta:** -0.0005 (-0.05pp)
**Val delta:** +0.0035 (+0.35pp)

**Feature importance ranking for LLM features (full coverage):**
- #5: llm_domain_expertise (0.0568)
- #6: llm_domain_focus (0.0539)
- #7: llm_prior_exit (0.0524)
- #14: llm_seniority_clevel (0.0314)
- #18: llm_narrative_climber (0.0222)
- #21: llm_narrative_builder (0.0194)
- #24: llm_conviction (0.0151)
- #26: llm_seniority_founder (0.0123)
- #27: llm_prior_founding (0.000 — unused)
- LLM features capture 26.4% of total feature importance

**Decision gate:** CV delta -0.05pp < 1pp threshold → **NO IMPROVEMENT (DEFINITIVE)**

**Verdict:** REVERT

**Notes:**
- Full coverage CONFIRMS the partial-coverage result: LLM features do not improve CV F₀.₅.
- The earlier +0.73pp with 33% missing data was noise from default values, not signal.
- The model allocates 26.4% of importance budget to LLM features, but this comes at the expense of structured features — it's redistributing, not adding signal.
- Val F₀.₅ improved slightly (+0.35pp) but CV is flat, indicating this is variance, not signal.
- Key finding for paper: LLM-extracted features from prose are REDUNDANT with structured JSON features. The prose is generated from the same underlying data — LLM extraction recovers the same signal, not new signal.
- Phase 3 is complete. The structured ceiling (CV ≈ 0.25) is an information ceiling for this dataset, not a feature-engineering ceiling.
