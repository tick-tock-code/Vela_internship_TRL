# VCBench Task IV — Competition Plan v2
*Enhanced after Steps 6–7 results. Panel-reviewed: Karpathy + a16z synthesis.*

---

## Contest Context

- **Contest:** SecureFinAI Contest 2026 (IEEE / Vela Partners / Trismik)
- **Task:** Task IV — AI for Venture Capital: Prediction of Startup Success
- **Benchmark:** [VCBench](https://www.vcbench.com/) — `vcbench-founder-prediction-v1`
- **Metric:** F₀.₅ — precision counts 2× more than recall
- **Dataset:** 4,500 public rows; private test set for final submission (single blind)
- **Class imbalance:** 91% failure (4,095) / 9% success (405)

---

## Leaderboard (Current State)

| Rank | Model | Org | Precision | Recall | F₀.₅ |
|---|---|---|---|---|---|
| 1 | Verifiable-RL | Vela+Oxford | 42.6% | 23.6% | **36.6%** |
| 2 | Policy-Induction | Vela+Oxford | 41.0% | 20.2% | 34.0% |
| 3 | Random-Rule-Forest | Vela+Oxford | 42.5% | 12.1% | 28.1% |
| 6 | GPT-4o (zero-shot) | OpenAI | 30.0% | 16.3% | 25.7% |
| 11 | Reasoned-Rule-Mining | Vela+Oxford | **87.5%** | 5.0% | 21.0% |

**Target:** F₀.₅ > 33% (top-2). Stretch: > 36.6% (beat Verifiable-RL).

### What the leaderboard tells us

RRF is confirmed prose-only — they explicitly prompt an LLM on `anonymised_prose` for each YES/NO question. Policy Induction and Verifiable-RL describe working from "founder profiles" without confirming JSON parsing. The defensible paper claim: *"To our knowledge, no prior work has directly engineered features from the structured JSON fields."*

The more important point: `anonymised_prose` is generated *from* the structured fields. It is a lossy re-encoding. Signal that degrades in text rendering: ordinal relationships in company size/duration buckets, interaction terms between fields, null values (empty `ipos`/`acquisitions` silently omitted), job sequence ordering. Structured parsing operates on higher-fidelity data. This is the core paper claim.

Reasoned-Rule-Mining (87.5% precision, 5% recall) almost certainly fires only on founders with prior exits/IPOs — near-deterministic rules from the `ipos`/`acquisitions` fields. This is the rule layer ceiling for precision without recall.

---

## Experiment State (as of 2026-03-08)

| Step | Result |
|---|---|
| Step 5 — Zero-shot baseline | F₀.₅ = 0.1265 |
| Step 6 — XGBoost + rule layer (23 structured features) | F₀.₅ = 0.2203, P=0.1957, R=0.4444, threshold=0.512 |
| Step 7 — Manual inspection (TP=36, FN=45, FP=148, TN=671) | Core problem identified: FP rate |
| Step 3c — Added 5 v2 features (repeat_founding_gap dropped, 73.8% null) | 28 usable features total |
| Step 4b — Rule layer v2: only prior_exit active (Rule 3 precision=11.1%, disabled) | Rule layer simplified |
| Step 6b — Platt calibration: invalid on same-set, deferred to CV | Rule-only model: F₀.₅=0.2889 (+6.9pp from disabling Rules 2+3) |
| Step 6c — Industry-stratified analysis at threshold=0.923 | FPs spread across tech/software, not concentrated in biotech/VC at high threshold |
| Phase 4 — 120+ experiments: HPO, ablations, model comparisons | **Best CV F₀.₅: 0.2539 ± 0.0348. Best Val F₀.₅: 0.3030** |

### Structured feature ceiling — confirmed

**Phase 4 is complete. The structured JSON fields have a hard ceiling at CV ≈ 0.25.**

Root cause: 84% of positives (68 of 81 on val) have no prior exits. The rule layer catches the easy 16%. Non-exit positives are statistically near-identical to failures on every structured feature. Every model architecture tested — XGBoost, LightGBM, RandomForest, LogisticRegression, ensembles, two-stage models, stacking — converges on the same CV ceiling. The optimal architecture is decision stumps (max_depth=1) with heavy regularization, which signals the structured features have limited joint discriminative power.

**Best configuration locked (do not iterate further on structured features):**
- Model: XGBoost stumps, n_estimators=227, max_depth=1, lr=0.0674, subsample=0.949, colsample_bytree=0.413, scale_pos_weight=10, min_child_weight=14, gamma=4.19, reg_alpha=0.73, reg_lambda=15.0
- Rule layer: prior_exit only (precision=24.5% on training set, 2.7× base rate)
- Threshold: 0.738
- Val result: F₀.₅=0.3030, P=0.3333, R=0.2222, 54 predicted positive

**What this means for Phase 3:**
Phase 3 (LLM feature extraction) is now the primary path to closing the gap, not a secondary test. The prose field likely contains signal not encoded in the structured JSON — domain expertise depth, career narrative type, conviction language — that the JSON fields cannot represent. The 0.25 CV ceiling is a structured-data ceiling, not an information ceiling.

### Key finding from Step 7 (still valid)

FPs have *higher* education prestige than TPs (edu_prestige_tier: 3.10 FP vs 2.78 TP). The model is over-indexing on credentials. What actually separates TPs from FPs:

| Feature | TP avg | FN avg | FP avg |
|---|---|---|---|
| exit_count | **0.58** | 0.04 | 0.11 |
| founding_role_count | **2.22** | 1.11 | 1.84 |
| industry_alignment | **0.47** | 0.33 | 0.38 |
| edu_prestige_tier | 2.78 | 1.31 | **3.10** ← noise |
| prestige_sacrifice_score | 21.31 | 18.09 | **25.22** ← noise |

**Rule layer update:**
- Rule 1 (prior_exit): KEEP — validated, precision=24.5% on training set (2.7× base rate).
- Rule 2 (top10_stem_founder): DISABLED — FP amplifier.
- Rule 3 (clevel_serial_founder): DISABLED — precision=11.1%, barely above 9% base rate.

---

## Panel Synthesis: What Changed in v2

### From Karpathy

**1. Run calibration now, not in Phase 5.**
A flat threshold-F₀.₅ curve means the model's probability output is not discriminative. Before adding any new features, apply Platt scaling (logistic regression on XGBoost's raw probabilities). Re-sweep threshold on calibrated output. If the curve sharpens, you've found 2–4pp of F₀.₅ for free.

**2. Replace single 80/20 split with 5-fold stratified CV.**
With 405 positives, a single val split has ~±3pp sampling variance. Every experiment result is noisy. 5-fold CV on the training set gives tighter signal. Keep the fixed 20% holdout as a final sanity check — never touch it during experimentation.

**3. The overnight loop should exhaust hypotheses, not pace them.**
20–30 iterations/night is too slow. Each experiment runs in seconds on 4,500 rows. The agent should run 100–200 iterations per session, auto-generating hypotheses from the experiment log pattern rather than executing a pre-written list. program.md should be: constraints + observations + seed hypotheses. Not a task list.

**4. Test the null hypothesis on sacrifice signal.**
prestige_sacrifice_score is in top-5 features but FPs have *higher* sacrifice scores than TPs. Run the experiment: remove edu_prestige_tier, best_degree_prestige, and prestige_sacrifice_score entirely. Does F₀.₅ improve? If yes, these features are net-negative.

### From a16z

**5. Add industry-stratified precision analysis.**
FPs cluster in biotech/research and VC/PE — industries where elite credentials are common but VCBench's definition of "success" (IPO/major acquisition) may not match real-world value creation. Build a per-industry precision table. Apply a higher decision threshold in credential-dense industries.

**6. Build tiered confidence output, not binary predictions.**
Reasoned-Rule-Mining's 87.5% precision at 5% recall is the most strategically interesting point on the leaderboard. It represents near-certain signal. Your system should output three tiers:
- **Tier A (Rule layer fires):** ~85–90% precision, 3–5% recall. Very high confidence.
- **Tier B (XGBoost > 0.80):** ~50–60% precision, moderate recall. High confidence.
- **Tier C (XGBoost 0.65–0.80):** Lower precision, higher recall. Uncertain — include only if recall recovery is worth the precision cost given F₀.₅.

**7. Add forward-looking proxy features.**
Every current feature is backward-looking (what did they do before founding). Add:
- `persistence_score` — longest_founding_tenure / total_inferred_experience (stayed in the fire vs. pivoted out)
- `repeat_founding_gap` — time between founding attempts (serial founders who re-enter quickly); drop if null rate > 35% after data audit

*`founding_age_proxy` was considered and dropped — it requires strict chronological ordering of jobs_json and precise duration start dates, neither of which can be assumed from bucket fields. `founding_timing` in Tier 2 covers the same intuition more reliably.*

**8. Rename the paper.**
"Beyond Prose" frames the contribution as a technical critique. A stronger framing leads with the behavioral insight: *"What VCs Miss: Behavioral Signals in Structured Founder Career Data"*. This is the paper that gets cited by practitioners, not just ML researchers.

---

## Data Schema

### Public dataset
```
founder_uuid, success, industry, ipos, acquisitions,
educations_json, jobs_json, anonymised_prose
```

### Private dataset (submission target)
```
founder_uuid, industry, ipos, acquisitions,
educations_json, jobs_json, anonymised_prose
[success column absent — predict this]
```

---

## Feature Engineering

### Tier 1: Direct Exit Signals — near-deterministic rules

- `has_prior_ipo` — binary: `ipos` not null/empty
- `has_prior_acquisition` — binary: `acquisitions` not null/empty
- `exit_count` — sum of ipos + acquisitions counts
- **Rule:** `exit_count > 0 → predict success` (validated, precision >> 50%)

### Tier 2: Sacrifice Signal — first novel contribution

From `jobs_json`:

- `max_company_size_before_founding` — largest employer before first founding role (ordinal)
- `prestige_sacrifice_score` — max company size rank × max seniority at point of first founding role
- `years_in_large_company` — total duration in 500+ employee companies before founding
- `comfort_index` — big company tenure × senior role × stability industry weight
- `founding_timing` — total inferred experience years before first founding role

**⚠️ Step 7 flag:** FPs have higher prestige_sacrifice_score than TPs (25.22 vs 21.31). Run null-hypothesis experiment: remove this feature and measure delta F₀.₅. If delta ≥ 0, disable it. If it hurts F₀.₅, it's adding noise and should be dropped.

**New — forward-looking proxies (a16z addition):**
- `persistence_score` — longest_founding_tenure / total_inferred_experience (commitment signal)
- `repeat_founding_gap` — time between first and second founding role (serial conviction signal)

*`founding_age_proxy` was considered and dropped. It requires jobs_json to be in strict
chronological order and duration buckets to locate a precise founding start date — neither
can be assumed. `founding_timing` in Tier 2 already captures the same intuition more
reliably. Adding a noisier proxy on top adds null-rate risk for no incremental signal.*

Encoding constants:
```python
COMPANY_SIZE_MAP = {
    "myself only": 1, "2-10": 2, "11-50": 3, "51-200": 4,
    "201-500": 5, "501-1000": 6, "1001-5000": 7, "5001-10000": 8, "10001+": 9
}
DURATION_MIDPOINT = {"<2": 1.0, "2-3": 2.5, "4-5": 4.5, "6+": 7.0}
HIGH_COMFORT_INDUSTRIES = {
    "Financial Services", "Consulting", "Investment Banking",
    "Government", "Law", "Accounting", "Insurance"
}
```

### Tier 3: Education × QS Interaction — second novel contribution

From `educations_json`:

- `edu_prestige_tier` — top-10=4, top-50=3, top-100=2, ranked=1, null=0
- `field_relevance_score` — CS/Eng/Math→tech=5; MBA→any=3; History/Arts→tech=1
- `prestige_x_relevance` — interaction term (the key feature)
- `degree_level` — PhD=4, MBA=3, MS=2, BS/BA=1, other=0
- `stem_flag` — binary
- `best_degree_prestige` — max prestige tier across all education records

**⚠️ Step 7 flag:** edu_prestige_tier is a noise amplifier (FP avg 3.10 > TP avg 2.78). Test experiment: remove edu_prestige_tier and best_degree_prestige, keep only prestige_x_relevance and stem_flag. The interaction term may be informative while the raw prestige feature is not.

### Tier 4: Career Trajectory

From `jobs_json`:

- `max_seniority_reached` — 0-5 scale (Founder/CEO/CTO=5, VP=4, Director=3, Sr IC=2, IC=1, Junior=0)
- `seniority_is_monotone` — did seniority increase over time? Binary
- `company_size_is_growing` — did company size increase over career? Binary
- `restlessness_score` — count of roles with duration < 2 years
- `founding_role_count` — **key separator (TP=2.22 vs FP=1.84)** — count of founding-stage roles
- `is_serial_founder` — binary: founding_role_count ≥ 2 (new — from Step 7)
- `longest_founding_tenure` — max duration in a founding-stage company
- `industry_pivot_count` — count of distinct industries across all jobs
- `industry_alignment` — does any prior job industry match current startup industry?
- `total_inferred_experience` — sum of duration midpoints

**New interaction features (from Step 7 + Karpathy):**
- `exit_x_serial` — exit_count × founding_role_count (serial founders with exits are near-certain TPs)
- `sacrifice_x_serial` — prestige_sacrifice_score × is_serial_founder (conditions sacrifice signal)
- `industry_prestige_penalty` — edu_prestige_tier × is_biotech_or_vc (discounts noise industries)

### Tier 5: LLM-Extracted Features (Phase 3, cached, one-time)

Use Claude as a structured extractor — NOT as a predictor. Extract from `anonymised_prose`:

```python
EXTRACTION_PROMPT = """Given this founder profile, respond ONLY with valid JSON, nothing else.

{
  "prior_founding_attempt": true or false,
  "domain_expertise_depth": integer 1-5,
  "highest_seniority_reached": "founder" | "C-level" | "VP" | "senior-IC" | "IC" | "junior",
  "evidence_of_prior_exit": true or false,
  "career_narrative_type": "builder" | "climber" | "academic" | "hybrid" | "unclear"
}"""
```

Cache all 4,500 results. Add as features. Measure delta F₀.₅ vs. structured-only. If delta < 1pp, deprioritize in paper.

---

## The High-Precision Rule Layer

```python
def apply_rules(row) -> tuple[int | None, str]:
    """
    Returns (prediction, rule_name) or (None, None) to fall through to classifier.
    
    Calibration from Step 7 (2026-03-08):
    - Rule 1 (prior_exit): VALIDATED. Keep.
    - Rule 2 (top10_stem_founder): DISABLED — high FP rate in biotech/VC/PE.
      Re-enable only with founding_role_count >= 2 gate.
    - Rule 3: Tightened — now requires founding_role_count >= 2.
    """
    # Rule 1: Prior exit — near-deterministic (validated on training set)
    if row.get('exit_count', 0) > 0:
        return 1, "prior_exit"

    # Rule 2: DISABLED — education prestige is a FP amplifier
    # Biotech/VC/PE have high credentials but low VCBench success rates.
    # Only re-enable if training set precision > 40% with founding_role_count >= 2 gate.

    # Rule 3: C-level at large company + serial founding history
    if (row.get('max_seniority_reached', 0) >= 4 and
        row.get('max_company_size_before_founding', 0) >= 6 and
        row.get('founding_role_count', 0) >= 2):
        return 1, "clevel_serial_founder"

    return None, None
```

Validate every rule on `public_train.csv`. Print precision per rule. Disable any rule with precision < 35%.

---

## Tiered Confidence Output (a16z addition)

Instead of a flat binary prediction, the final system outputs confidence tiers:

```python
def predict_with_confidence(row, prob, threshold_high=0.80, threshold_mid=0.65):
    rule_pred, rule_name = apply_rules(row)
    if rule_pred == 1:
        return {"prediction": 1, "tier": "A", "source": rule_name, "confidence": "very_high"}
    elif prob >= threshold_high:
        return {"prediction": 1, "tier": "B", "source": "xgboost", "confidence": "high"}
    elif prob >= threshold_mid:
        return {"prediction": 1, "tier": "C", "source": "xgboost", "confidence": "uncertain"}
    else:
        return {"prediction": 0, "tier": None, "source": None, "confidence": "low"}
```

For the contest submission: use Tier A + Tier B only (maximizes F₀.₅). Tier C is for analysis. Thresholds for A and B are tuned on val set CV.

---

## Solution Architecture

### Phase 1: Baseline & Data Profiling — COMPLETE ✅
Zero-shot baseline: F₀.₅ = 0.1265.

### Phase 2: Structured Feature Engineering — IN PROGRESS
Current: F₀.₅ = 0.2203. Target: > 0.28.

**Remaining Phase 2 tasks (run manually, in order):**

**2a. Fix calibration first (Karpathy priority #1).**
Before adding any features, apply Platt scaling to Step 6 model output:
```python
from sklearn.calibration import CalibratedClassifierCV
# or: fit a LogisticRegression on (xgb_probs, y_val) and use its output
# Re-sweep threshold on calibrated probabilities
# If F0.5 curve sharpens, record delta — this is free improvement
```

**2b. Switch to 5-fold stratified CV (Karpathy priority #2).**
Replace the single 80/20 eval loop with:
```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Train on 4 folds, evaluate on 1 fold, average F0.5 across folds
# Keep fixed 20% holdout (public_val.csv) as final sanity check only
```

**2c. Run null-hypothesis experiment on prestige features.**
Remove `edu_prestige_tier`, `best_degree_prestige`, `prestige_sacrifice_score` from FEATURE_COLS. Re-train. If CV F₀.₅ improves or stays same → these features are net-negative noise, drop them permanently.

**2d. Add new features from Step 7 findings.**
Add `is_serial_founder`, `exit_x_serial`, `sacrifice_x_serial`, `industry_prestige_penalty`,
`persistence_score`, and `repeat_founding_gap` (if null rate ≤ 35% from data audit).
Re-train. Record delta CV F₀.₅.

**2e. Run industry-stratified precision analysis (a16z).**
For the best model so far:
```python
for industry in val['industry'].unique():
    subset = val[val['industry'] == industry]
    # Compute precision, recall, F0.5 for this industry
    # Print sorted by precision descending
# Goal: identify which industries have high FP rates → inform threshold adjustment
```

**2f. Run RRF ablation experiment.**
Take Appendix E top-10 questions from the RRF paper. Implement each as a deterministic rule from structured fields. Compare precision of deterministic implementation vs. LLM-answered prose version. This becomes Section 4 of the paper.

### Phase 3: LLM Feature Extraction — NOW PRIMARY PATH ⚠️

**Status change:** Phase 3 is no longer a secondary delta-test. It is the primary remaining lever. Structured features are exhausted at CV=0.2539. The gap to target (CV > 0.28) must come from prose-derived signal.

**Why prose may help for non-exit positives:** The 84% of positives with no prior exits are invisible to structured features — their career JSON looks nearly identical to failures. But their prose may reveal: conviction language, domain expertise depth, the *way* they describe career transitions, and narrative patterns that VCs recognize intuitively but that don't reduce to structured fields. This is signal the JSON encoding cannot represent by construction.

**What to extract (from `anonymised_prose` — do NOT ask "will this founder succeed?"):**

```python
EXTRACTION_PROMPT = """Given this founder profile, respond ONLY with valid JSON. No preamble.

{
  "prior_founding_attempt": true or false,
  "domain_expertise_depth": 1-5,
  "highest_seniority_reached": "founder"|"C-level"|"VP"|"senior-IC"|"IC"|"junior",
  "evidence_of_prior_exit": true or false,
  "career_narrative_type": "builder"|"climber"|"academic"|"hybrid"|"unclear",
  "domain_focus_consistency": 1-5,
  "conviction_indicator": 1-5
}"""
```

Two fields added vs. original spec:
- `domain_focus_consistency` — does the career show deep focus in one domain or broad jumping? 1=scattered, 5=laser-focused. This may be the key signal for non-exit positives.
- `conviction_indicator` — does the prose suggest high personal commitment (left stable career, self-funded, early founding)? 1=low, 5=high. Operationalizes the sacrifice signal via language rather than career structure.

**Execution:**
- Run on all 4,500 public rows. Cache to `features/llm_features_cache.json`.
- One-time cost: ~$5–10 at Sonnet pricing. Never re-run unless cache is deleted.
- Add as features to `classifier.py`. Retrain with locked XGBoost stump config.
- Run 5-fold CV. Record delta vs. CV=0.2539 baseline.
- **Decision gate:** If CV delta ≥ 0.01 (1pp) → include in final model. If delta < 0.01 → prose features do not help; accept structured ceiling and submit best structured model.

**If Phase 3 delta ≥ 1pp:** Run a second Phase 4 overnight loop (50–100 experiments) on the expanded feature set. Threshold re-sweep required since adding LLM features changes probability distribution.

**If Phase 3 delta < 1pp:** Do not continue iterating. The information-theoretic ceiling for this dataset may be near 0.30 Val F₀.₅. Submit the locked XGBoost stump model. Write the paper around the structured ceiling finding — it is itself a publishable result about the limits of career-data signals for founder success prediction.

### Phase 4: Overnight Loop — CLAUDE CODE
**Structure:** Read experiment_log.md → generate hypothesis → modify classifier.py → run evaluate.py (CV) → log → revert if no improvement → repeat.

**Loop speed target (Karpathy):** 100–200 experiments per overnight session. Each experiment runs in ~3 seconds. The agent should exhaust the hypothesis space in one night, not pace across multiple nights.

**program.md structure (Karpathy):** Observations + constraints + seed hypotheses. NOT a task list. The agent generates its own hypotheses from the experiment log pattern.

**Hard constraints — embed verbatim in program.md:**
- Only modify `classifier.py`. Never touch `evaluate.py`, `data/public_val.csv`, `data/public_train.csv`.
- Every experiment must append to `experiment_log.md` before trying the next one.
- Revert any change that does not improve 5-fold CV F₀.₅.
- Do not change random_state=42 anywhere.
- No features derived from `anonymised_prose` during Phase 4 (structured fields only).
- Do not run `predict.py` on the private test set.
- Do not declare a phase complete without printing final F₀.₅.

**Seed hypotheses for program.md (generated from Step 7):**
1. Raise threshold to 0.80 — accept lower recall, gain precision. F₀.₅ rewards this.
2. Add `is_serial_founder` binary (founding_role_count ≥ 2). Clearest TP vs FP separator.
3. Add `exit_x_serial` interaction. Serial founders with exits are near-certain TPs.
4. Remove `edu_prestige_tier` and `best_degree_prestige` entirely — test null hypothesis.
5. Increase `scale_pos_weight` from 10 to 20. Model predicts 20.4% positive; true rate is 9%.
6. Apply industry-stratified thresholds — higher threshold for biotech/VC/PE.
7. Add `sacrifice_x_serial` — condition sacrifice signal on serial founding history.
8. Try LightGBM instead of XGBoost — may produce better-calibrated probabilities.
9. Add `persistence_score` — longest_founding_tenure / total_inferred_experience.
10. Test removing `prestige_sacrifice_score` — FPs have higher scores than TPs.
11. Add `repeat_founding_gap` — only if null rate was ≤ 35% after Step 3 data audit.
12. Test minimal feature set: keep only exit_count + founding_role_count + industry_alignment.

### Phase 5: Ensemble & Final Calibration — AWS SAGEMAKER HPO
After feature set locked (Phase 4 complete):

1. SageMaker HPO sweep: `max_depth`, `min_child_weight`, `scale_pos_weight`, `learning_rate`, `subsample`, `colsample_bytree`.
2. Platt scaling on final model.
3. Stability check: 5 random seeds on train/val. F₀.₅ should be consistent ±1pp.
4. Tiered confidence validation: measure precision for Tier A vs Tier B separately.
5. Set FINAL_THRESHOLD in predict.py. Run predict.py on private test set exactly once.

**Ensemble structure:**
```
Tier A: Rule layer (prior_exit, clevel_serial_founder) → predict success
  ↓ if no rule fires
Tier B: XGBoost + Platt calibration > 0.80 → predict success
  ↓ if below 0.80
Tier C (analysis only): XGBoost 0.65–0.80 → do not include in submission
```

**Fallback (only if structured approach plateaus below 0.26):** Fine-tune 7B LLM on AWS. High overfitting risk on 405 positives — treat as last resort.

---

## Repo Structure

```
vcbench-task4/
├── plan_v2.md                      ← this file
├── program.md                      ← Claude Code overnight loop instructions
├── data/
│   ├── public.csv                  ← original dataset
│   ├── public_train.csv            ← 80% stratified split (seed=42, FIXED)
│   ├── public_val.csv              ← 20% holdout — NEVER touch during experiments
│   └── private_test.csv            ← submission target — NEVER use until Phase 5
├── features/
│   ├── extract_structured.py       ← Tiers 1–4 feature engineering
│   ├── extract_llm.py              ← Tier 5 (cached, one-time)
│   └── high_precision_rules.py     ← deterministic rule layer
├── baselines/
│   ├── zero_shot_baseline.py
│   └── zero_shot_predictions.json  ← cached, do not re-run
├── classifier.py                   ← Phase 4 iterates here
├── evaluate.py                     ← FIXED. Never modify.
├── predict.py                      ← Run only after Phase 5
├── experiment_log.md               ← append-only
└── submissions/
    └── submission_v{n}.csv
```

---

## Claude Code Execution Instructions

Read this entire file before starting. Execute phases sequentially. Do not skip ahead.
After every step, append to `experiment_log.md` before proceeding.

---

### Environment Setup (run once)

```bash
mkdir -p vcbench-task4/{data,features,baselines,submissions}
cd vcbench-task4
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost lightgbm anthropic matplotlib seaborn joblib
```

Confirm `data/public.csv` and `data/private_test.csv` exist. If missing, stop and ask.

---

### Step 1: Create `evaluate.py` — FIXED, never modify

```python
# evaluate.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "f05": round(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "positive_rate": round(float(np.array(y_pred).mean()), 4),
        "n_predicted_positive": int(np.array(y_pred).sum()),
        "threshold": threshold,
    }

def sweep_thresholds(y_true, y_prob, steps=50):
    results = []
    for t in np.linspace(0.3, 0.95, steps):
        results.append(evaluate(y_true, y_prob, threshold=t))
    return sorted(results, key=lambda x: x["f05"], reverse=True)

def best_threshold(y_true, y_prob):
    return sweep_thresholds(y_true, y_prob)[0]

def cv_evaluate(X, y, model_fn, n_splits=5):
    """5-fold stratified CV. Returns mean and std F0.5."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = model_fn()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        result = best_threshold(y_val.tolist(), probs.tolist())
        scores.append(result["f05"])
    return {"cv_mean_f05": round(np.mean(scores), 4), "cv_std_f05": round(np.std(scores), 4)}

if __name__ == "__main__":
    y_true = [0]*91 + [1]*9
    y_prob = [0.1]*85 + [0.8]*6 + [0.9]*9
    result = best_threshold(y_true, y_prob)
    print("Smoke test passed:", result)
```

Run smoke test before proceeding.

---

### Step 2: Generate train/val split (run once only)

```python
# data/split.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/public.csv")
print(f"Total rows: {len(df)}, positive rate: {df['success'].mean():.1%}")

train, val = train_test_split(df, test_size=0.2, stratify=df['success'], random_state=42)
train.to_csv("data/public_train.csv", index=False)
val.to_csv("data/public_val.csv", index=False)
print(f"Train: {len(train)} rows | Val: {len(val)} rows")
print("Saved. Do not re-run.")
```

**If split already exists, skip this step.**

---

### Step 3: Create `features/extract_structured.py`

Implement all Tier 1–4 features. Function signature:

```python
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes raw dataframe. Returns dataframe with all feature columns appended."""
```

Required features (in addition to Tiers 1–4 from previous plan):

```python
# New features added in v2 (Step 7 + panel synthesis)
"is_serial_founder",          # binary: founding_role_count >= 2. No new parsing needed.
"exit_x_serial",              # exit_count * founding_role_count. No new parsing needed.
"sacrifice_x_serial",         # prestige_sacrifice_score * is_serial_founder. No new parsing needed.
"industry_prestige_penalty",  # edu_prestige_tier * is_biotech_or_vc. Requires industry field audit.
"persistence_score",          # longest_founding_tenure / (total_inferred_experience + 0.01). No new parsing.
"repeat_founding_gap",        # years between 1st and 2nd founding role. Requires chronological job order.
                              # DROP this feature if null rate > 35% after audit.
```

*`founding_age_proxy` was dropped — it requires strict chronological ordering of jobs_json
and precise duration start dates, neither of which can be assumed from bucket fields.
`founding_timing` in Tier 2 covers the same intuition more reliably.*

After writing, print null rates and distributions for all new features.
Fix any feature with >50% null rate. Drop `repeat_founding_gap` if null rate > 35%.

---

### Step 4: Create `features/high_precision_rules.py`

```python
# features/high_precision_rules.py
def apply_rules(row) -> tuple:
    """
    Returns (prediction, rule_name) if high-confidence rule fires.
    Returns (None, None) to fall through to classifier.

    Calibration log:
    - 2026-03-08: Rule 2 disabled. FPs have higher edu_prestige than TPs.
      Biotech/VC/PE high-credential profiles are FP clusters.
    """
    if row.get('exit_count', 0) > 0:
        return 1, "prior_exit"

    # Rule 2: DISABLED — re-enable only after training set precision validation >= 40%
    # if (row.get('edu_prestige_tier', 0) >= 4 and row.get('stem_flag', 0) == 1
    #         and row.get('founding_role_count', 0) >= 2):
    #     return 1, "top10_stem_serial_founder"

    if (row.get('max_seniority_reached', 0) >= 4 and
        row.get('max_company_size_before_founding', 0) >= 6 and
        row.get('founding_role_count', 0) >= 2):
        return 1, "clevel_serial_founder"

    return None, None
```

Validate on `public_train.csv`: print precision per rule. Disable any rule with precision < 35%.

---

### Step 5 (COMPLETE): Zero-shot baseline — F₀.₅ = 0.1265 ✅

Skip. Results are cached in `baselines/zero_shot_predictions.json`.

---

### Step 6 (COMPLETE): XGBoost structured baseline — F₀.₅ = 0.2203 ✅

Skip. Results logged. Proceed to Phase 2 improvements.

---

### Step 6b: Apply Platt calibration to existing model (NEW — Karpathy priority)

This should be the first thing run before any new feature work.

```python
# run_calibration.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules
from evaluate import sweep_thresholds, best_threshold

FEATURE_COLS = [
    "has_prior_ipo", "has_prior_acquisition", "exit_count",
    "max_company_size_before_founding", "prestige_sacrifice_score",
    "years_in_large_company", "comfort_index", "founding_timing",
    "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
    "degree_level", "stem_flag", "best_degree_prestige",
    "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
    "restlessness_score", "founding_role_count", "longest_founding_tenure",
    "industry_pivot_count", "industry_alignment", "total_inferred_experience",
]

def calibrate():
    val = extract_features(pd.read_csv("data/public_val.csv"))
    model = joblib.load("model.pkl")

    X_val = val[FEATURE_COLS].fillna(0)
    y_val = val["success"]
    raw_probs = model.predict_proba(X_val)[:, 1]

    # Apply rule layer
    for i, (_, row) in enumerate(val.iterrows()):
        rule_pred, _ = apply_rules(row)
        if rule_pred == 1:
            raw_probs[i] = 1.0

    # Platt scaling
    platt = LogisticRegression()
    platt.fit(raw_probs.reshape(-1, 1), y_val)
    calibrated_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    print("=== Raw probabilities ===")
    raw_result = best_threshold(y_val.tolist(), raw_probs.tolist())
    print(raw_result)

    print("\n=== Calibrated probabilities ===")
    cal_result = best_threshold(y_val.tolist(), calibrated_probs.tolist())
    print(cal_result)

    print(f"\nDelta F0.5 from calibration: {cal_result['f05'] - raw_result['f05']:+.4f}")
    joblib.dump(platt, "platt_scaler.pkl")

if __name__ == "__main__":
    calibrate()
```

Record delta F₀.₅ in experiment_log.md. If delta > 0.005, use calibrated probabilities in all subsequent experiments.

---

### Step 7 (COMPLETE): Manual inspection ✅

Key findings are embedded in this plan (v2 Feature Engineering section and Experiment State).

---

### Step 8: Create `program.md` for Phase 4 overnight loop

Write `program.md` with this exact structure. **This is observations + constraints, not a task list.** The agent generates its own hypotheses from the experiment log.

```markdown
# program.md — VCBench Phase 4 Overnight Loop

## Current best F₀.₅: 0.2203 (pre-calibration) — update after Step 6b
## CV method: 5-fold stratified (not single val split)
## Target: > 0.28 by end of overnight loop. Stretch: > 0.33.

## Observations (do not change these — read and use them)

**What separates TPs from FPs (from manual inspection, N=900 val rows):**
- exit_count: TP avg 0.58 vs FP avg 0.11 — strongest signal
- founding_role_count: TP avg 2.22 vs FP avg 1.84 — serial founding matters
- edu_prestige_tier: TP avg 2.78 vs FP avg 3.10 — INVERTED, prestige is FP noise
- prestige_sacrifice_score: TP avg 21.31 vs FP avg 25.22 — INVERTED, also noise
- The model predicts 20.4% positive; true rate is 9% — model is too liberal

**FP cluster:** Biotech/research, VC/PE industries — high credentials, low VCBench success

**Calibration:** Threshold-F₀.₅ curve is flat from 0.50–0.95. Model probabilities
are not well-separated. Platt calibration was applied in Step 6b (record delta here).

## Constraints — never violate

- Only modify `classifier.py`. Never modify `evaluate.py`, val split, or train split.
- Every experiment logs to `experiment_log.md` before the next one starts.
- Revert if CV F₀.₅ does not improve.
- random_state=42 everywhere. Do not change.
- No features derived from `anonymised_prose`.
- Do not run `predict.py` on private test set.
- Do not declare done without printing final CV F₀.₅ and val F₀.₅.

## Seed hypotheses — start here, then generate your own from the log

1. Raise threshold to 0.80 — F₀.₅ rewards precision > recall
2. Add `is_serial_founder` binary (founding_role_count >= 2)
3. Add `exit_x_serial` interaction term
4. Remove `edu_prestige_tier` + `best_degree_prestige` entirely (null hypothesis test)
5. Increase scale_pos_weight to 20 — model predicts 20.4%, true rate is 9%
6. Add industry-stratified threshold offset for biotech/VC/PE
7. Add `sacrifice_x_serial` interaction
8. Switch to LightGBM — may produce better-calibrated probabilities
9. Add `persistence_score` — longest_founding_tenure / total_inferred_experience
10. Remove `prestige_sacrifice_score` — FPs have higher scores than TPs
11. Add `repeat_founding_gap` (only if null rate was ≤ 35% after Step 3 audit)
12. Test: keep only exit_count + founding_role_count + industry_alignment (minimal feature set)

After exhausting this list, generate new hypotheses by reading which experiments improved
F₀.₅ and what they have in common. Focus on feature interactions and threshold tuning,
not on adding more raw features.
```

---

### Step 9: Run LLM feature extraction (Phase 3)

```python
# features/extract_llm.py
import anthropic, pandas as pd, json
from pathlib import Path

CACHE_FILE = "features/llm_features_cache.json"
EXTRACTION_PROMPT = """Given this founder profile, respond ONLY with valid JSON. No preamble.

{
  "prior_founding_attempt": true or false,
  "domain_expertise_depth": 1-5,
  "highest_seniority_reached": "founder"|"C-level"|"VP"|"senior-IC"|"IC"|"junior",
  "evidence_of_prior_exit": true or false,
  "career_narrative_type": "builder"|"climber"|"academic"|"hybrid"|"unclear",
  "domain_focus_consistency": 1-5,
  "conviction_indicator": 1-5
}

domain_focus_consistency: 1=scattered across many domains, 5=laser-focused in one domain throughout career.
conviction_indicator: 1=low personal commitment signals, 5=high commitment (left stable career, early founding, self-funded signals)."""

def extract_llm_features(df):
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    else:
        client = anthropic.Anthropic()
        cache = {}
        for i, row in df.iterrows():
            try:
                r = client.messages.create(
                    model="claude-sonnet-4-20250514", max_tokens=150,
                    messages=[{"role": "user", "content": EXTRACTION_PROMPT + "\n\nProfile:\n" + row['anonymised_prose']}]
                )
                cache[row['founder_uuid']] = json.loads(r.content[0].text)
            except Exception as e:
                cache[row['founder_uuid']] = None
            if i % 100 == 0:
                cache_path.write_text(json.dumps(cache))
                print(f"  {i}/{len(df)} extracted")
        cache_path.write_text(json.dumps(cache))

    records = []
    for uuid in df['founder_uuid']:
        feat = cache.get(uuid) or {}
        records.append({
            "founder_uuid": uuid,
            "llm_prior_founding": int(feat.get("prior_founding_attempt", False)),
            "llm_domain_expertise": feat.get("domain_expertise_depth", 3),
            "llm_prior_exit": int(feat.get("evidence_of_prior_exit", False)),
            "llm_narrative_builder": int(feat.get("career_narrative_type") == "builder"),
            "llm_narrative_climber": int(feat.get("career_narrative_type") == "climber"),
            "llm_seniority_founder": int(feat.get("highest_seniority_reached") == "founder"),
            "llm_seniority_clevel": int(feat.get("highest_seniority_reached") == "C-level"),
            "llm_domain_focus": feat.get("domain_focus_consistency", 3),
            "llm_conviction": feat.get("conviction_indicator", 3),
        })
    return df.merge(pd.DataFrame(records), on="founder_uuid", how="left")
```

Run on full public dataset. Retrain classifier. Record delta F₀.₅. If delta < 0.01, note in log and exclude from Phase 4 hypothesis space.

---

### Step 10: Create `predict.py` — run only after Phase 5

```python
# predict.py
import pandas as pd, numpy as np, joblib
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules

FINAL_THRESHOLD = None  # Set manually after Phase 5 calibration

def generate_submission(threshold):
    test = extract_features(pd.read_csv("data/private_test.csv"))
    model = joblib.load("model.pkl")
    platt = joblib.load("platt_scaler.pkl")

    feature_cols = [c for c in test.columns if c in model.feature_names_in_]
    X_test = test[feature_cols].fillna(0)
    raw_probs = model.predict_proba(X_test)[:, 1]
    probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    for i, (_, row) in enumerate(test.iterrows()):
        rule_pred, _ = apply_rules(row)
        if rule_pred == 1:
            probs[i] = 1.0

    preds = (probs >= threshold).astype(int)
    submission = pd.DataFrame({"founder_uuid": test["founder_uuid"], "success": preds})
    filename = f"submissions/submission_threshold_{threshold:.2f}.csv"
    submission.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    print(f"Predicted positives: {preds.sum()} / {len(preds)} ({preds.mean():.1%})")
    return filename

if __name__ == "__main__":
    if FINAL_THRESHOLD is None:
        raise ValueError("Set FINAL_THRESHOLD before generating submission.")
    generate_submission(FINAL_THRESHOLD)
```

---

### Step 11: Experiment log format — append-only, every run

```
## Experiment {N} — {YYYY-MM-DD}
**Change:** {what was modified in classifier.py}
**Hypothesis:** {why this should improve F₀.₅}
**CV F₀.₅:** {mean} ± {std} (prev best: {previous best})
**Val F₀.₅:** {result on public_val.csv}
**Precision:** {val precision}
**Recall:** {val recall}
**Threshold:** {optimal threshold}
**Verdict:** KEEP / REVERT
**Notes:** {any observations — feature importance shifts, FP/FN pattern changes, etc.}
```

Never delete entries. CV F₀.₅ is the decision metric. Val F₀.₅ is the sanity check.

---

### What Claude Code must NOT do

- Modify `evaluate.py` under any circumstances
- Modify `data/public_val.csv` or `data/public_train.csv`
- Change random_state=42 in split or model
- Run `predict.py` on private test set before Phase 5
- Add features from `anonymised_prose` during Phase 4
- Skip an experiment log entry
- Declare a phase complete without printing final CV F₀.₅ and val F₀.₅

---

## Paper Strategy

### Title (revised — a16z framing)
*"What VCs Miss: Behavioral and Structural Signals in Founder Career Data"*
*(Working alt: "Beyond Prose: Structured Feature Engineering for Founder Success Prediction")*

### Core argument (updated after Phase 4)

The paper now has two distinct contributions rather than one:

**Contribution 1 (confirmed):** Structured JSON parsing recovers signal lost in the prose rendering — ordinal relationships, interaction terms, null signals — and outperforms zero-shot LLM inference on prose by +17pp Val F₀.₅ (0.1265 → 0.3030). The structured ceiling is real and measurable.

**Contribution 2 (pending Phase 3):** The structured ceiling (CV ≈ 0.25) reveals that 84% of successful founders — those without prior exits — are invisible to career-structure data. Their success signal lives in the prose: conviction language, domain focus consistency, narrative type. We show that LLM-extracted behavioral features from prose recover this signal as a second-stage complement to structured parsing. The two approaches are not alternatives but complements — each covering a distinct population of successful founders.

**If Phase 3 delta < 1pp:** Contribution 2 becomes a null result, which is itself publishable. The paper argues that founder success prediction for non-exit founders is near-random on available VCBench data — structured or unstructured — and that the benchmark's information ceiling for this population needs to be addressed in future dataset design.

### Ablation table (updated with Phase 4 results)

| Variant | CV F₀.₅ | Val F₀.₅ | Δ baseline |
|---|---|---|---|
| Zero-shot LLM on prose (baseline) | — | 0.1265 | — |
| Tier 1 rules only (prior_exit) | — | 0.2889 | +16.2pp |
| Tier 1–4 full structured (v1, 23 features) | — | 0.2203 | +9.4pp |
| Tier 1–4 + v2 features + HPO (XGB stumps) | 0.2539 ± 0.035 | **0.3030** | +17.7pp |
| + LLM extractor (Tier 5) | TBD | TBD | TBD |
| + Phase 3 loop (if delta ≥ 1pp) | TBD | TBD | TBD |
| RRF top-10 as deterministic rules | TBD | TBD | vs. RRF LLM-answered |

**Key architectural finding for paper:** All model families (XGB, LGB, RF, LR, ensembles) converge on decision stumps (max_depth=1) as optimal. This is a dataset-size constraint — 405 positives cannot support complex joint feature interactions. This finding constrains the paper's claims about which features matter and motivates the LLM extraction approach as the path to recovering non-structural signal.

### Framing rule (updated)
Results now dictate a two-population framing:
- **Exit-history founders (~16% of positives):** Fully captured by structured features. Near-deterministic rule layer.
- **Non-exit founders (~84% of positives):** Invisible to structured data. Phase 3 is the test of whether LLM prose extraction can recover signal here.

The paper's contribution depends on Phase 3 results. Do not finalize the narrative until Phase 3 delta is known.

### Target venues
1. **IEEE SecureFinAI 2026** — guaranteed with contest entry; prize + proceedings in one artifact
2. **ACM ICAIF 2026** (AI in Finance, Oct/Nov) — most respected AI+finance venue; extend with deeper ablations
3. **arXiv preprint** — post same week as contest submission to timestamp priority

### O-1A timeline

| Milestone | Target |
|---|---|
| Phase 3 (LLM extraction) | April 2026 |
| Phase 4 loop on expanded features (if Phase 3 delta ≥ 1pp) | April–May 2026 |
| Phase 5 HPO + submission | End of contest window |
| arXiv preprint posted | Same week as submission |
| IEEE proceedings | ~2 months post-contest |
| ACM ICAIF submission | August 2026 |
| ACM ICAIF notification | Oct–Nov 2026 |
| O-1A filing | January 2027 |

O-1A criteria served: Prize/award (top-3 IEEE finish) + Scholarly articles (IEEE + ACM ICAIF + arXiv).
Note: USCIS does not require accepted papers at filing. IEEE submission + arXiv preprint is sufficient for initial filing.

### Conflict of interest
Vela Partners co-sponsors the contest. Keep the paper independent. If formal Vela/Oxford collaboration is pursued, document contribution split formally before filing. Do not mention the family relationship.

---

## Key Numbers

| Metric | Value |
|---|---|
| Dataset (public) | 4,500 rows |
| Positive rate | 9% (405 success) |
| Zero-shot baseline Val F₀.₅ | 0.1265 |
| Structured ceiling CV F₀.₅ | 0.2539 ± 0.035 |
| Best Val F₀.₅ (locked model) | **0.3030** (P=0.3333, R=0.2222, threshold=0.738) |
| Phase 3 decision gate | CV delta ≥ 0.01 → include LLM features |
| Contest target | > 0.33 Val F₀.₅ |
| Stretch goal | > 0.366 (beat Verifiable-RL) |
| Locked model config | XGB stumps, max_depth=1, spw=10, mcw=14 |
| Rule layer | prior_exit only (precision=24.5%, fires on ~106 train rows) |

---

## Key References

- [VCBench paper](https://arxiv.org/abs/2509.14448)
- [Policy Induction](https://arxiv.org/abs/2505.21427)
- [Random Rule Forest](https://arxiv.org/abs/2505.24622)
- [R.A.I.S.E.](https://arxiv.org/abs/2504.12090)
- [Think-Reason-Learn](https://github.com/Vela-Research/think-reason-learn)
- [autoresearch (Karpathy)](https://github.com/karpathy/autoresearch)
- [ACM ICAIF](https://ai-finance.org/)
- `vela_oxford_papers.md` — detailed paper summaries