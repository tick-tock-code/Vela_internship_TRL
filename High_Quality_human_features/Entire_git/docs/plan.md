# VCBench Task IV: Startup Success Prediction — Competition Plan

## Contest Context

- **Contest:** SecureFinAI Contest 2026 (IEEE / Vela Partners / Trismik)
- **Task:** Task IV — AI for Venture Capital: Prediction of Startup Success
- **Benchmark:** [VCBench](https://www.vcbench.com/) — `vcbench-founder-prediction-v1`
- **Metric:** F₀.₅ Score — precision counts 2x more than recall
- **Objective:** Predict binary `success` label from anonymized founder profiles
- **Dataset:** 4,500 public rows (train + local eval), private test set for final submission
- **Class imbalance:** 91% failure (4,095 rows) / 9% success (405 rows)
- **Submission:** Single blind — get it right before submitting

---

## Leaderboard (Current State)

| Rank | Model | Org | Precision | Recall | F₀.₅ |
|---|---|---|---|---|---|
| 1 | Verifiable-RL | Vela+Oxford | 42.6% | 23.6% | **36.6%** |
| 2 | Policy-Induction | Vela+Oxford | 41.0% | 20.2% | 34.0% |
| 3 | Random-Rule-Forest | Vela+Oxford | 42.5% | 12.1% | 28.1% |
| 6 | GPT-4o (zero-shot) | OpenAI | 30.0% | 16.3% | 25.7% |
| 11 | Reasoned-Rule-Mining | Vela+Oxford | **87.5%** | 5.0% | 21.0% |
| 18 | Tier-1 VCs | Humans | 23.0% | 5.2% | 10.7% |
| 19 | Random Classifier | Baseline | 9.0% | 9.0% | 9.0% |

**Target:** F₀.₅ > 33% (top-2 territory). Stretch goal: > 36.6% (beat Verifiable-RL).

### What the leaderboard tells us

The RRF paper explicitly answers YES/NO questions by prompting an LLM on the `anonymised_prose`
text — confirmed prose-only. Policy Induction and Verifiable-RL describe their inputs as
"founder profiles" without explicitly stating whether they parse structured fields; their
published code and paper examples consistently show prose as the input format. To our knowledge,
no published approach has directly engineered features from the structured JSON fields —
but this is an inference, not a confirmed fact. The defensible paper claim is:
*"To our knowledge, no prior work has directly parsed the structured fields for feature
engineering"* — not *"all models use prose only."*

The more important point: `anonymised_prose` is generated *from* the structured fields by the
VCBench anonymization pipeline. It is a lossy re-encoding — not a richer source. Signal that
degrades in the text rendering: ordinal relationships in duration/company size buckets,
interaction terms between fields, null values (empty `ipos`/`acquisitions` silently omitted),
and the precise sequence/ordering of roles. Structured parsing isn't just a different approach
— it operates on higher-fidelity data. This is the core claim of the paper.

The "Reasoned-Rule-Mining" entry (87.5% precision, 5% recall) is almost certainly a handful
of near-deterministic rules for founders with prior exits/IPOs. These should fire on the
`ipos` and `acquisitions` fields directly — near-certain success labels. Build this as a
dedicated high-precision rule layer in the ensemble.

---

## What the Vela+Oxford Papers Tell Us

### Policy Induction (arXiv: 2505.21427) → Rank #2

Iterative natural language policy embedded into an LLM prompt. Few-shot ICL loop with memory:
the model forms explicit IF-THEN rules, tests them, and refines. Fully human-readable output.

**What worked:** Data-efficient, high precision, human-auditable rules.
**What didn't:** Recall is weak (20.2%). Policy convergence can get stuck. Sensitive to seeding
examples. Entirely prose-based — misses all structured-field signal.
**Ceiling:** ~34% F₀.₅ for prose-only rule induction approaches.

### Random Rule Forest (arXiv: 2505.24622) → Rank #3

LLM generates ~100 YES/NO questions, each scored on validation precision, low-performers
filtered, survivors vote threshold-style. Expert-crafted questions lift performance further.

**What worked:** Ensemble diversity, aggressive filtering of noise, transparent decision logic.
Their **Appendix E (top-10 high-precision questions)** is the signal map for this domain —
all about exits, C-level roles, top-ranked education, domain alignment. Their **Appendix D
(bottom-10)** are all prose-inference soft signals ("communicates clearly," "seems passionate").
**What didn't:** Questions are answered from prose by an LLM — lossy. Recall very low (12.1%).
No joint optimization across questions.
**Key actionable insight:** Take their top-10 questions and implement them as deterministic
rules from `jobs_json`/`educations_json`. If deterministic parsing beats LLM-answered prose,
that's a clean publishable ablation.

### Verifiable-RL / Think-Reason-Learn (github.com/Vela-Research/think-reason-learn) → Rank #1

RLVR (Reinforcement Learning with Verifiable Rewards) applied to founder prediction. The LLM
generates a prediction + reasoning chain; reward = 1 if prediction matches ground truth label,
0 otherwise. GRPO-style policy update reinforces reasoning paths that lead to correct outcomes.
The TRL framework also includes GPTree (LLM-guided decision trees) and RRF as modules.

**What worked:** Current ceiling at 36.6% F₀.₅. Best precision-recall balance of any approach.
The RLVR loop discovers non-obvious patterns without being told what to look for.
**What didn't:** Still prose-based. Requires GPU + labeled data for stable RL training. Only
405 positive examples — small for RLVR. Results are stochastic; leaderboard likely represents
a best run.
**Ceiling insight:** The VCBench paper itself acknowledges that anonymization introduces noise
and the 9% base rate means even a perfect classifier is working in difficult conditions.
The 36.6% number likely approaches the information-theoretic ceiling for prose-only approaches.

### The gap no one has (to our knowledge) closed

| Signal | Policy Induction | RRF | Verifiable-RL | This plan |
|---|---|---|---|---|
| Works from prose | ✅ | ✅ (confirmed) | ✅ | ✅ (secondary) |
| Parses structured JSON | unknown | ❌ (confirmed) | unknown | ✅ (primary) |
| Sacrifice / opportunity cost signal | ❌ | ❌ | ❌ | ✅ |
| Education × QS interaction term | ❌ | ❌ | ❌ | ✅ |
| Threshold calibration (>0.5) | Partial | ✅ | Partial | ✅ |
| Gradient-based classifier | ❌ | ❌ | ✅ | ✅ (XGBoost) |
| Ablation study per feature group | ❌ | Partial | ❌ | ✅ |

**The paper no one has written:** `anonymised_prose` is a generated, lossy re-encoding of
the structured fields. A paper that directly engineers features from the source JSON —
and ablates the signal loss introduced by the prose rendering — is both novel and directly
falsifiable. That is the contribution.

---

## Data Schema

### Public dataset (train + local eval)
```
founder_uuid, success, industry, ipos, acquisitions,
educations_json, jobs_json, anonymised_prose
```

### Private dataset (submission target)
```
founder_uuid, industry, ipos, acquisitions,
educations_json, jobs_json, anonymised_prose
[success column empty — this is what we predict]
```

### First thing to run after loading
```python
print(df['success'].value_counts(normalize=True))
# Expected: ~9% positive rate — confirmed from dataset
# success=0: 4,095 rows (91%)
# success=1:   405 rows  (9%)
```

---

## Feature Engineering

### Tier 1: Direct Exit Signals — near-deterministic rules
These fields are the structural basis for the "Reasoned-Rule-Mining" approach (87.5% precision).
Treat as rules, not probabilistic features.

- `has_prior_ipo` — binary flag: `ipos` is not null/empty
- `has_prior_acquisition` — binary flag: `acquisitions` is not null/empty
- `exit_count` — sum of ipos + acquisitions counts
- **Rule:** `if exit_count > 0 → predict success with very high confidence`

Caveat: the `ipos`/`acquisitions` fields in the public training data describe the *founder's
prior history*, not the current startup's outcome. Founders with prior exits are near-certain
success signals on this benchmark by construction.

### Tier 2: Sacrifice Signal — novel contribution
**Core insight:** Opportunity cost = the best measurable proxy for hunger/conviction.
A founder who walked away from a high-prestige, stable career is making a structurally
measurable bet — even in anonymized data.

Features to engineer from `jobs_json`:
- `max_company_size_before_founding` — largest employer (by employee count bucket) before
  first "myself only" or solo founding role
- `prestige_sacrifice_score` — (max company size rank × max seniority rank) at point of
  first founding role; encode company sizes as ordinal (myself=1, 2-10=2, 11-50=3,
  51-200=4, 201-500=5, 501-1000=6, 1001-5000=7, 5000+=8)
- `years_in_large_company` — total `duration` in 500+ employee companies before first
  founding role (encode duration buckets as midpoints: <2=1, 2-3=2.5, 4-5=4.5, 6+=7)
- `comfort_index` — weighted: big company tenure × senior role × stability industry
  (Finance/Consulting/Government = high comfort; Startups/NGO = low comfort)
- `founding_timing` — total inferred experience years before first founding role
  (late founding after prestige accumulation = higher sacrifice signal)

**Hypothesis:** A VP at a 5,000+ person company for 6+ years who then founded a solo startup
has a high sacrifice score. A person who bounced between 2-10 person companies is a different
risk profile — not worse, just different signal.

### Tier 3: Education × QS Rank Interaction — second novel contribution
**Core insight:** CS from QS rank 1 ≠ MBA from QS rank 1 ≠ CS from QS rank 200.
The interaction captures what neither variable does alone — and is exactly the kind of
signal a VC uses intuitively but rarely formalizes.

Features from `educations_json`:
- `edu_prestige_tier` — encode `qs_ranking` as: top-10=4, top-50=3, top-100=2, ranked=1, null=0
- `field_relevance_score` — encode `field` relevance to startup's `industry`:
  CS/Engineering/Math → tech startup = 5; MBA → any = 3; History/Arts → tech = 1
- `prestige_x_relevance` — interaction term: `edu_prestige_tier × field_relevance_score`
- `degree_level` — PhD=4, MBA=3, MS=2, BS/BA=1, other=0 (ordinal)
- `stem_flag` — binary: field is STEM
- `best_degree_prestige` — max prestige tier across all education records (some founders
  have multiple degrees)

### Tier 4: Career Trajectory Features
From `jobs_json`:

- `max_seniority_reached` — encode roles: Founder/CEO/CTO=5, VP=4, Director=3,
  Senior IC=2, IC=1, Junior/Intern=0; take max across all roles
- `seniority_is_monotone` — did seniority increase over time (early→late)? Binary
- `company_size_is_growing` — did company size increase over career trajectory? Binary
- `restlessness_score` — count of roles with duration < 2 years
- `founding_role_count` — count of "myself only" or "2-10" company size roles
- `longest_founding_tenure` — max duration in a founding-stage company (persistence signal)
- `industry_pivot_count` — count of distinct industries across all jobs
- `industry_alignment` — does any prior job industry match the current startup `industry`?
- `total_inferred_experience` — sum of duration midpoints across all jobs

### Tier 5: LLM-Extracted Features (Phase 3 only)
Do NOT ask the LLM "is this founder successful?" — it will answer with baked-in priors.
Use LLM as structured feature extractor only:

```python
EXTRACTION_PROMPT = """
Given this founder profile, respond ONLY with a JSON object and nothing else:
{
  "prior_founding_attempt": true/false,
  "domain_expertise_depth": 1-5,
  "highest_seniority_reached": "founder/C-level/VP/senior-IC/IC/junior",
  "evidence_of_prior_exit": true/false,
  "career_narrative_type": "builder/climber/academic/hybrid/unclear"
}
"""
```

Cache all 4,500 results — this is a one-time cost (~$5-10 at Sonnet pricing).

---

## The High-Precision Rule Layer

Inspired by "Reasoned-Rule-Mining" (87.5% precision). Build a deterministic rule module
that fires on high-confidence cases before the classifier:

```python
def high_precision_rules(row) -> Optional[int]:
    # Rule 1: Prior exit is near-deterministic
    if row['exit_count'] > 0:
        return 1
    # Rule 2: Top-10 QS + Founder role + STEM
    if (row['edu_prestige_tier'] == 4 and
        row['founding_role_count'] > 0 and
        row['stem_flag'] == 1):
        return 1
    # Rule 3: C-level at 500+ company + prior founding attempt
    if (row['max_seniority_reached'] >= 4 and
        row['max_company_size_before_founding'] >= 5 and
        row['prior_founding_attempt'] == True):
        return 1
    return None  # No high-confidence rule fired → fall through to classifier
```

These rules should be derived from and validated against the training set — don't hardcode
thresholds without checking them empirically. The key value: they are deterministic,
interpretable, and contribute to the paper's ablation story.

---

## Solution Architecture

### Phase 1: Baseline & Data Profiling (Week 1) — MANUAL
**Goal:** Build intuition, establish eval pipeline, reproduce GPT-4o baseline (~25% F₀.₅).

Steps:
1. Load public dataset — confirm class distribution (9% positive — already confirmed)
2. Create 80/20 stratified train/val split, stratify on `success`, fix random seed
3. Build `evaluate.py`: compute F₀.₅, precision, recall; sweep threshold 0.3→0.95;
   plot threshold vs. F₀.₅ curve
4. Manually inspect 20 true positives, 20 false positives, 20 false negatives —
   build intuition about what separates them before any feature engineering
5. Zero-shot Claude Sonnet on `anonymised_prose` → record as baseline (~25% F₀.₅ expected)
6. Zero-shot with explicit threshold tuning → check how much threshold alone moves F₀.₅

**Do this phase manually.** You will catch things an agent won't: wrong JSON parsing,
unexpected null patterns, duration encoding issues, QS ranking field quirks.

### Phase 2: Structured Feature Engineering (Week 1–2) — MANUAL
**Goal:** Engineer Tiers 1–4, train XGBoost, beat GPT-4o baseline.

Steps:
1. Parse `educations_json` and `jobs_json` into flat feature columns
2. Engineer all Tier 1–4 features — check for null rates and distributions on each
3. Train XGBoost / LightGBM on structured features only (no prose, no LLM)
4. **Plot feature importances** — this empirically determines whether sacrifice signal
   or education×QS interaction is the primary contribution for the paper
5. Sweep decision threshold 0.3→0.95, plot F₀.₅ curve per threshold
6. Run RRF experiment: implement their top-10 questions as deterministic rules from
   structured fields; compare precision vs. their LLM-answered prose versions
7. Target: F₀.₅ > 28% on val split, with threshold ≈ 0.65–0.80

**Threshold note:** With 9% positive rate, the F₀.₅-optimal threshold will be well above
0.5. Do not use 0.5 default. Expect 10–20% positive prediction rate to be optimal.

### Phase 3: LLM Feature Extraction (Week 2) — MANUAL
**Goal:** Add Tier 5 extracted features; test marginal contribution vs. Phase 2.

Steps:
1. Run extraction prompt on all 4,500 public rows via Anthropic API — cache to JSON file
2. Add as additional columns, retrain XGBoost
3. Measure delta F₀.₅ vs. Phase 2 structured-only
4. If delta < 1pp → deprioritize LLM features; focus paper on structured signal story
5. If delta ≥ 1pp → include as Phase 3 component in paper's ablation table

### Phase 4: Autoresearch-Style Overnight Loop (Week 2–3) — CLAUDE CODE
**Goal:** Autonomous experiment scaling to push F₀.₅ past 33%.

Setup:
```
program.md          ← human writes, refines based on experiment log
classifier.py       ← Claude Code iterates (new features, thresholds, ensembles)
evaluate.py         ← FIXED, never modified — F₀.₅ on val split is the single metric
experiment_log.md   ← append-only; every run logged with F₀.₅, change made, verdict
```

What to put in `program.md` before first overnight run:
- Current best F₀.₅ (from Phase 2/3)
- Top-5 features by importance from XGBoost
- List of failed hypotheses so far
- Constraints: no data leakage, structured features primary, single val split, no
  threshold changes unless paired with feature change

The loop (20–30 iterations per night):
1. Read `program.md` + `experiment_log.md`
2. Propose one hypothesis (new feature, new interaction, new threshold, new ensemble)
3. Modify `classifier.py`
4. Run `evaluate.py`
5. If F₀.₅ improved → keep change, append to log
6. If not → revert, log why and what was learned
7. Repeat

**Adaptation of autoresearch:** This is exactly the Karpathy loop (agent edits one file,
metric is fixed, human edits `program.md` to steer) applied to classification rather than
pretraining. The only difference: instead of 5-minute training runs on a GPU, experiments
run in seconds on CPU with 4,500 rows.

### Phase 5: Ensemble & Calibration (Week 3) — AWS SAGEMAKER HPO
**Goal:** Final architecture lock, threshold tuning, pre-submission validation.

Ensemble structure (conservative AND logic — maximizes precision):
```
Layer 1: High-precision rule layer (Tier 1 exits + top-3 deterministic rules)
  → if rules fire → predict success
  → if no rule fires → pass to Layer 2

Layer 2: Structured XGBoost (Tiers 2–4 features)
  → probability output

Layer 3: LLM extractor features (Tier 5, if delta > 1pp from Phase 3)
  → probability output

Final decision: predict success if XGBoost probability > calibrated threshold
  AND (LLM features agree OR rule layer fired)
```

SageMaker HPO: once feature set is locked, run hyperparameter sweep on XGBoost
(max_depth, min_child_weight, scale_pos_weight, learning_rate, subsample).
4,500 rows → HPO takes minutes, not hours.

Calibration: Platt scaling on probabilities; re-sweep threshold on calibrated output.
Stability check: run 5 random seeds on train/val split; F₀.₅ should be consistent ±1pp.

---

## Iteration Approach by Phase

| Phase | Method | Rationale |
|---|---|---|
| 1–2 | Manual | Build intuition. Catch data issues, null patterns, encoding errors. Non-negotiable. |
| 3 | Manual | One-time LLM extraction run. Cache results. Manual inspection of extraction quality. |
| 4 | Claude Code overnight | Scale experiments once architecture is locked. 20–30/night. |
| 5 | AWS SageMaker HPO | Hyperparameter sweep after feature set is fixed. Minutes on this dataset. |
| Fallback only | Fine-tune 7B LLM on AWS | Only if structured approach plateaus below 28%. High overfitting risk on 405 positives. |

---

## Repo Structure

```
vcbench-task4/
├── plan.md                         ← this file
├── program.md                      ← Claude Code agent instructions (autoresearch style)
├── data/
│   ├── vcbench_final_public.csv    ← 4,500 labeled rows (source)
│   ├── vcbench_final_private.csv   ← submission target
│   ├── public_train.csv            ← 80% stratified split (generated)
│   └── public_val.csv              ← 20% stratified holdout — NEVER touch during training
├── features/
│   ├── extract_structured.py       ← parse educations_json, jobs_json → flat features
│   ├── extract_llm.py              ← Anthropic API Tier 5 extraction (cached to JSON)
│   ├── enrich_industry.py          ← attach industry-level base rates
│   └── high_precision_rules.py     ← deterministic rule layer (Tier 1 + derived rules)
├── classifier.py                   ← main model (Claude Code iterates on this in Phase 4)
├── evaluate.py                     ← F₀.₅, precision, recall, threshold curve — FIXED
├── predict.py                      ← generate final submission CSV
├── experiment_log.md               ← append-only, every experiment logged
└── submissions/
    └── submission_v{n}.csv
```

---

## Submission Format

```csv
founder_uuid,success
ef9c8004-4689-436f-a457-6593bc03d0c2,0
...
```

---

## Paper Strategy (O-1A Artifact)

### Working title
*"Beyond Prose: Behavioral and Structural Signals for Founder Success Prediction"*

### Core argument
The `anonymised_prose` field in VCBench is a generated, lossy re-encoding of the structured
fields (`educations_json`, `jobs_json`, `ipos`, `acquisitions`). Ordinal relationships,
interaction terms, null signals, and field sequencing are all degraded in this text rendering.
To our knowledge, existing approaches treat founder success prediction as text classification
over this prose field. We instead engineer features directly from the structured source data —
recovering signal that the prose rendering discards. We propose a feature engineering framework
extracting sacrifice signals, education×prestige interactions, and trajectory features from
structured JSON, and show via ablation that structured parsing recovers measurable signal lost
in the prose encoding. The resulting system achieves state-of-the-art F₀.₅ on VCBench.

### The key ablation the paper must contain
| Model variant | F₀.₅ | Δ vs. baseline |
|---|---|---|
| Zero-shot LLM on prose (baseline) | ~25% | — |
| Tier 1 rules only (exit signals) | TBD | TBD |
| Tier 1 + Tier 2 (sacrifice signal) | TBD | TBD |
| Tier 1–3 (+ edu×QS interaction) | TBD | TBD |
| Tier 1–4 (full structured features) | TBD | TBD |
| + LLM extractor (Tier 5) | TBD | TBD |
| + Ensemble / calibration | TBD | TBD |
| RRF top-10 as deterministic rules | TBD | vs. RRF LLM-answered |

This table is the paper's Table 2. Fill it in as phases complete.

### Framing rule
**Results dictate framing.** After Phase 4, read feature importance plots:
- Sacrifice signal is top-3 → lead with it as the core contribution
- Education×QS interaction dominates → reframe around that
- The ensemble architecture is the key differentiator → frame as systems paper
- Do not commit to narrative until data tells you what worked

### Draft section structure
1. **Introduction** — why prose-only prediction misses structure; the case for JSON parsing
2. **Related Work** — VCBench, Policy Induction, RRF, Verifiable-RL, R.A.I.S.E.
3. **Methodology** — feature engineering framework (sacrifice, edu×QS, trajectory, rule layer)
4. **The RRF ablation** — deterministic rules vs. LLM-answered prose versions of same questions
5. **Experiments** — ablation table (above); threshold sweep analysis
6. **Results** — leaderboard position + comparison to all baselines
7. **Analysis** — feature importances and behavioral interpretation (the insight section)
8. **Conclusion + Future Work** — generalization to idea-level prediction (VCBench v2)

### Target venues (priority order)
1. **IEEE SecureFinAI 2026 contest proceedings** — guaranteed if competing; establishes IEEE
   stamp and contest placement in the same document
2. **ACM ICAIF 2026** (AI in Finance, typically Oct/Nov) — most respected AI+finance venue;
   extend the contest paper with deeper ablations
3. **arXiv preprint** — post concurrently with contest submission to timestamp priority

### O-1A timeline

| Milestone | Target Date |
|---|---|
| Public dataset analysis + baseline | April 2026 |
| Structured feature engineering complete | April–May 2026 |
| Claude Code overnight loop runs | May 2026 |
| Contest submission | End of contest window |
| arXiv preprint posted | Same week as contest submission |
| IEEE proceedings accepted | ~2 months after contest |
| ACM ICAIF submission | August 2026 |
| ACM ICAIF notification | October–November 2026 |
| O-1A filing | January 2027 |

**Filing note:** USCIS does not require accepted papers — a strong arXiv preprint + IEEE
submission confirmation letter is sufficient for initial filing. Acceptance strengthens the
case but don't wait for it.

### O-1A criteria this project serves
- **Prize or award for excellence:** Top-3 finish at IEEE-affiliated SecureFinAI Contest 2026
- **Scholarly articles in professional publications:** IEEE proceedings + ACM ICAIF paper +
  arXiv preprint (three independent signals for one body of work)
- **Bonus if achievable:** Formal judging/review role for other contest submissions = third
  criterion from the same project

### Conflict of interest note
Vela Partners co-sponsors the contest and your brother is GP there. Keep the paper entirely
independent. If you pursue formal collaboration with Vela's Oxford research partners (which
would add significant O-1A weight as "critical role at a distinguished organization"), document
it as a professional research relationship with a clear contribution split — formal, arm's-length,
and documented before filing. Do not mention the family relationship in the paper.

---

## Key Numbers to Keep in Mind

| Metric | Value |
|---|---|
| Dataset size (public) | 4,500 rows |
| Positive rate | 9% (405 success, 4,095 failure) |
| Optimal threshold (estimated) | 0.65–0.80 (not 0.5) |
| Optimal positive prediction rate | 10–20% of test set |
| Current best F₀.₅ | 36.6% (Verifiable-RL) |
| Gap to close | ~3–4 F₀.₅ points |
| GPT-4o zero-shot baseline | 25.7% |
| Your Phase 2 target | > 28% |
| Your final target | > 33% |

---

## Claude Code Execution Instructions

This section is written for Claude Code. Read the entire plan.md before starting.
Execute phases sequentially. Do not skip ahead. After each phase, append results to
`experiment_log.md` before proceeding.

---

### Environment Setup (do this first, once)

```bash
mkdir -p vcbench-task4/{data,features,submissions}
cd vcbench-task4
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm anthropic matplotlib seaborn joblib
```

Confirm the public dataset CSV is in `data/vcbench_final_public.csv` and the private test set is in
`data/vcbench_final_private.csv` before proceeding. Do not create placeholder files — stop and
ask if either file is missing.

---

### Step 1: Create `evaluate.py` (FIXED — never modify after creation)

This file is the single source of truth for all metrics. Write it exactly as below.
Do not add arguments, flags, or other complexity. It must be importable and callable
from any other file.

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
        "positive_rate": round(float(y_pred.mean()), 4),
        "n_predicted_positive": int(y_pred.sum()),
        "threshold": threshold,
    }

def sweep_thresholds(y_true, y_prob, steps=50):
    results = []
    for t in np.linspace(0.3, 0.95, steps):
        results.append(evaluate(y_true, y_prob, threshold=t))
    return sorted(results, key=lambda x: x["f05"], reverse=True)

def best_threshold(y_true, y_prob):
    return sweep_thresholds(y_true, y_prob)[0]

if __name__ == "__main__":
    # Smoke test
    import numpy as np
    y_true = [0]*91 + [1]*9
    y_prob = [0.1]*85 + [0.8]*6 + [0.9]*9
    result = best_threshold(y_true, y_prob)
    print("Smoke test passed:", result)
```

Run it to confirm the smoke test passes before moving to Step 2.

---

### Step 2: Create `data/split.py` and generate the train/val split (run once only)

```python
# data/split.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/vcbench_final_public.csv")

print("=== Dataset overview ===")
print(f"Total rows: {len(df)}")
print(f"Success rate: {df['success'].mean():.1%}")
print(f"Columns: {list(df.columns)}")
print("\nSuccess distribution:")
print(df['success'].value_counts())
print(df['success'].value_counts(normalize=True).round(3))

train, val = train_test_split(
    df, test_size=0.2, stratify=df['success'], random_state=42
)
train.to_csv("data/public_train.csv", index=False)
val.to_csv("data/public_val.csv", index=False)

print(f"\nTrain: {len(train)} rows, {train['success'].mean():.1%} positive")
print(f"Val:   {len(val)} rows, {val['success'].mean():.1%} positive")
print("Split saved. Do not re-run this script.")
```

Run once. Commit both output CSVs. Never re-run or regenerate the split.

---

### Step 3: Create `features/extract_structured.py`

Parse `educations_json` and `jobs_json` into flat feature columns.
Implement all Tier 1–4 features from the Feature Engineering section of this plan.

Key encoding constants to use:

```python
COMPANY_SIZE_MAP = {
    "myself only": 1, "2-10": 2, "11-50": 3, "51-200": 4,
    "201-500": 5, "501-1000": 6, "1001-5000": 7, "5001-10000": 8, "10001+": 9
}
DURATION_MIDPOINT = {
    "<2": 1.0, "2-3": 2.5, "4-5": 4.5, "6+": 7.0
}
SENIORITY_MAP = {
    # Map common role strings to 0-5 scale
    # 5: founder/CEO/CTO/co-founder
    # 4: VP/SVP/EVP/President
    # 3: Director/Head of/Principal
    # 2: Senior/Lead/Staff
    # 1: Engineer/Manager/Associate/Analyst
    # 0: Intern/Junior/Fellow/Coordinator
}
HIGH_COMFORT_INDUSTRIES = {
    "Financial Services", "Consulting", "Investment Banking",
    "Government", "Law", "Accounting", "Insurance"
}
```

The function signature must be:
```python
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes raw dataframe, returns dataframe with all feature columns appended."""
```

After writing the function, run it on `public_train.csv` and print:
- Total features created
- Null rate for each feature
- Value distribution for `prestige_sacrifice_score`, `edu_prestige_tier`,
  `max_seniority_reached`, and `exit_count`

Fix any features with >50% null rate before proceeding.

---

### Step 4: Create `features/high_precision_rules.py`

```python
# features/high_precision_rules.py
def apply_rules(row) -> tuple[int | None, str]:
    """
    Returns (prediction, rule_name) if a high-confidence rule fires.
    Returns (None, None) if no rule fires — falls through to classifier.
    """
    # Rule 1: Prior exit history — near-deterministic
    if row.get('exit_count', 0) > 0:
        return 1, "prior_exit"

    # Rule 2: Top-10 QS + STEM + founding role
    if (row.get('edu_prestige_tier', 0) >= 4 and
        row.get('stem_flag', 0) == 1 and
        row.get('founding_role_count', 0) > 0):
        return 1, "top10_stem_founder"

    # Rule 3: C-level at large company + prior founding attempt
    if (row.get('max_seniority_reached', 0) >= 4 and
        row.get('max_company_size_before_founding', 0) >= 6 and
        row.get('founding_role_count', 0) > 0):
        return 1, "clevel_large_company_founder"

    return None, None
```

After writing, run the rule layer on `public_train.csv` and print:
- How many rows each rule fires on
- Precision of each rule against `success` label
- If any rule has precision < 30%, flag it and do not use it — report to experiment log

---

### Step 5: Run zero-shot LLM baseline and record result

Create `baselines/zero_shot_baseline.py`:

```python
# baselines/zero_shot_baseline.py
"""
Zero-shot Claude Sonnet baseline using anonymised_prose only.
This establishes the ~25% F0.5 baseline that all structured approaches must beat.
Run once, cache predictions, never re-run.
"""
import anthropic
import pandas as pd
import json
from pathlib import Path
from evaluate import evaluate, sweep_thresholds

CACHE_FILE = "baselines/zero_shot_predictions.json"

SYSTEM_PROMPT = """You are a venture capital analyst predicting startup success.
You will be given a founder profile. Respond with ONLY a JSON object:
{"probability": <float between 0 and 1>, "reasoning": "<one sentence>"}
A probability > 0.5 means you predict success (IPO, major acquisition, or high-tier funding).
Be calibrated: only ~9% of founders in this dataset are successful."""

def predict_one(client, prose: str) -> float:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prose}]
    )
    try:
        return json.loads(response.content[0].text)["probability"]
    except Exception:
        return 0.5

def run_baseline():
    val = pd.read_csv("data/public_val.csv")

    if Path(CACHE_FILE).exists():
        print("Loading cached predictions...")
        probs = json.loads(Path(CACHE_FILE).read_text())
    else:
        client = anthropic.Anthropic()
        probs = []
        for i, row in val.iterrows():
            prob = predict_one(client, row['anonymised_prose'])
            probs.append(prob)
            if i % 50 == 0:
                print(f"  {i}/{len(val)} complete...")
        Path(CACHE_FILE).write_text(json.dumps(probs))

    results = sweep_thresholds(val['success'].tolist(), probs)
    print("\n=== Zero-shot baseline (top 5 thresholds) ===")
    for r in results[:5]:
        print(r)
    print(f"\nBest F0.5: {results[0]['f05']} at threshold {results[0]['threshold']}")
    return results[0]['f05']

if __name__ == "__main__":
    run_baseline()
```

Run this. Record the best F₀.₅ in `experiment_log.md` as "Baseline: zero-shot prose".
This is the number every subsequent experiment must beat.

---

### Step 6: Create `classifier.py` — structured features baseline

This is the first real model. Claude Code will iterate on this file in Phase 4.
Initial version uses XGBoost on Tier 1–4 structured features only.

```python
# classifier.py
"""
Main classifier. This file is iterated on by Claude Code in Phase 4.
Current version: XGBoost on structured features (Tiers 1-4).
evaluate.py is FIXED and defines the metric. Do not modify evaluate.py.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from evaluate import evaluate, sweep_thresholds, best_threshold
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules

FEATURE_COLS = [
    # Tier 1
    "has_prior_ipo", "has_prior_acquisition", "exit_count",
    # Tier 2 — sacrifice signal
    "max_company_size_before_founding", "prestige_sacrifice_score",
    "years_in_large_company", "comfort_index", "founding_timing",
    # Tier 3 — education x QS
    "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
    "degree_level", "stem_flag", "best_degree_prestige",
    # Tier 4 — trajectory
    "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
    "restlessness_score", "founding_role_count", "longest_founding_tenure",
    "industry_pivot_count", "industry_alignment", "total_inferred_experience",
]

def train_and_evaluate():
    train = extract_features(pd.read_csv("data/public_train.csv"))
    val = extract_features(pd.read_csv("data/public_val.csv"))

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["success"]
    X_val = val[FEATURE_COLS].fillna(0)
    y_val = val["success"]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,   # ~91/9 class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Apply high-precision rule layer first
    probs = model.predict_proba(X_val)[:, 1].tolist()
    for i, row in val.iterrows():
        rule_pred, rule_name = apply_rules(row)
        if rule_pred == 1:
            probs[val.index.get_loc(i)] = 1.0  # Override with certainty

    result = best_threshold(y_val.tolist(), probs)
    print(f"\n=== Structured features baseline ===")
    print(result)

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nTop 10 features:")
    print(importances.head(10))

    joblib.dump(model, "model.pkl")
    return result

if __name__ == "__main__":
    train_and_evaluate()
```

Run this. Record F₀.₅ in `experiment_log.md` as "Phase 2: structured features XGBoost".
Print and save the feature importance table — this is critical for paper framing.

**Target:** F₀.₅ > 28%. If below 25%, stop and debug feature extraction before continuing.

---

### Step 7: Manual inspection — REQUIRED before Phase 4

After running classifier.py, identify:
- 20 rows from val set where `success=1` and model predicted correctly (true positives)
- 20 rows where `success=1` and model predicted incorrectly (false negatives)
- 20 rows where `success=0` and model predicted success (false positives)

Print the `anonymised_prose` and top feature values for each group.
Write a 3-5 sentence observation for each group in `experiment_log.md`.
These observations become the hypotheses for the Phase 4 overnight loop.

---

### Step 8: Create `program.md` for Phase 4 overnight loop

After Steps 1–7 are complete and you have a baseline F₀.₅ on record, create `program.md`.
This file tells Claude Code what to do in the autonomous overnight loop.

`program.md` must contain:
1. Current best F₀.₅ on val split
2. Top-5 features by importance
3. Observations from manual inspection (Step 7)
4. List of hypotheses to test (start with 5–10)
5. Hard constraints (listed below)
6. Pointer to evaluate.py as the fixed metric

**Hard constraints for the overnight loop (copy these verbatim into program.md):**
- Only modify `classifier.py` — never modify `evaluate.py` or the val split
- Every experiment must log its F₀.₅ result to `experiment_log.md` before trying the next
- Revert any change that does not improve F₀.₅ on the val split
- Do not change the random seed (42) in any model
- Do not add features derived from `anonymised_prose` — structured fields only in Phase 4
- Do not use the private test set during any experiment

---

### Step 9: Run LLM feature extraction (Phase 3)

Create `features/extract_llm.py`:

```python
# features/extract_llm.py
"""
One-time LLM feature extraction. Caches results to JSON.
Uses Claude Sonnet as structured feature extractor — NOT as predictor.
Run once on full public dataset. Never re-run unless cache is deleted.
"""
import anthropic
import pandas as pd
import json
from pathlib import Path

CACHE_FILE = "features/llm_features_cache.json"

EXTRACTION_PROMPT = """Given this founder profile, respond ONLY with a valid JSON object and nothing else.
No explanation, no markdown, no preamble.

{
  "prior_founding_attempt": true or false,
  "domain_expertise_depth": integer 1-5,
  "highest_seniority_reached": one of "founder" / "C-level" / "VP" / "senior-IC" / "IC" / "junior",
  "evidence_of_prior_exit": true or false,
  "career_narrative_type": one of "builder" / "climber" / "academic" / "hybrid" / "unclear"
}"""

def extract_llm_features(df: pd.DataFrame) -> pd.DataFrame:
    cache_path = Path(CACHE_FILE)

    if cache_path.exists():
        print("Loading cached LLM features...")
        cache = json.loads(cache_path.read_text())
    else:
        client = anthropic.Anthropic()
        cache = {}
        for i, row in df.iterrows():
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=150,
                    messages=[{
                        "role": "user",
                        "content": EXTRACTION_PROMPT + "\n\nProfile:\n" + row['anonymised_prose']
                    }]
                )
                cache[row['founder_uuid']] = json.loads(response.content[0].text)
            except Exception as e:
                cache[row['founder_uuid']] = None
                print(f"  Failed on {row['founder_uuid']}: {e}")
            if i % 100 == 0:
                cache_path.write_text(json.dumps(cache))
                print(f"  {i}/{len(df)} extracted, cached.")
        cache_path.write_text(json.dumps(cache))

    # Flatten to dataframe columns
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
        })
    return df.merge(pd.DataFrame(records), on="founder_uuid", how="left")
```

Run on public train + val. Then re-run `classifier.py` with LLM features added to
`FEATURE_COLS`. Record the delta F₀.₅ in `experiment_log.md` as "Phase 3: + LLM features".
If delta < 0.01 (1pp), note this in the experiment log and do not include LLM features
in the Phase 4 loop.

---

### Step 10: Create `predict.py` — final submission generator

```python
# predict.py
"""
Generates the final submission CSV for the private test set.
Run only after Phase 5 is complete and the final model + threshold are locked.
"""
import pandas as pd
import numpy as np
import joblib
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules
from evaluate import evaluate

FINAL_THRESHOLD = None  # Set this after Phase 5 calibration

def generate_submission(threshold: float):
    test = extract_features(pd.read_csv("data/vcbench_final_private.csv"))
    model = joblib.load("model.pkl")

    feature_cols = [c for c in test.columns if c in model.feature_names_in_]
    X_test = test[feature_cols].fillna(0)
    probs = model.predict_proba(X_test)[:, 1].tolist()

    for i, row in test.iterrows():
        rule_pred, _ = apply_rules(row)
        if rule_pred == 1:
            probs[test.index.get_loc(i)] = 1.0

    preds = (np.array(probs) >= threshold).astype(int)
    submission = pd.DataFrame({
        "founder_uuid": test["founder_uuid"],
        "success": preds
    })

    filename = f"submissions/submission_threshold_{threshold:.2f}.csv"
    submission.to_csv(filename, index=False)
    print(f"Submission saved: {filename}")
    print(f"Predicted positives: {preds.sum()} / {len(preds)} ({preds.mean():.1%})")
    return filename

if __name__ == "__main__":
    if FINAL_THRESHOLD is None:
        raise ValueError("Set FINAL_THRESHOLD before generating submission.")
    generate_submission(FINAL_THRESHOLD)
```

Do not run this until Phase 5 is complete and FINAL_THRESHOLD is set.

---

### Step 11: Append to `experiment_log.md` after every run

Format every entry exactly as:

```
## Experiment {N} — {date}
**Change:** {what was modified in classifier.py}
**Hypothesis:** {why this should help}
**F0.5:** {result} (prev best: {previous best})
**Precision:** {result}
**Recall:** {result}
**Threshold:** {result}
**Verdict:** KEEP / REVERT
**Notes:** {any observations about feature importance, failure patterns, etc.}
```

The experiment log is append-only. Never delete entries.

---

### What Claude Code should NOT do

- Modify `evaluate.py` under any circumstances
- Modify `data/public_val.csv` or `data/public_train.csv`
- Change the random seed (42) in the train/val split or model
- Run `predict.py` on the private test set before Phase 5 is complete
- Add features derived from `anonymised_prose` during the Phase 4 loop
- Skip the experiment log entry for any run
- Declare a phase complete without printing the final F₀.₅ result

---

## Key References

- [VCBench paper](https://arxiv.org/abs/2509.14448) — Chen, Ternasky et al. (Vela+Oxford)
- [Policy Induction](https://arxiv.org/abs/2505.21427) — Mu, Ternasky, Alican, Ihlamur
- [Random Rule Forest](https://arxiv.org/abs/2505.24622) — Griffin, Ternasky, Alican, Ihlamur
- [R.A.I.S.E.](https://arxiv.org/abs/2504.12090) — Preuveneers, Ternasky, Alican, Ihlamur
- [Think-Reason-Learn](https://github.com/Vela-Research/think-reason-learn) — Vela Research
- [autoresearch (Karpathy)](https://github.com/karpathy/autoresearch) — loop design inspiration
- [ACM ICAIF](https://ai-finance.org/) — primary extended paper venue
- `vela_oxford_papers.md` — detailed breakdown of all three Vela+Oxford papers