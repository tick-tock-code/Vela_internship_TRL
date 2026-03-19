# Vela + Oxford VCBench Papers: What They Tried, What Didn't Work, Where the Ceiling Is

These are the three research lineages behind the top 3 entries on the VCBench leaderboard.
Read these before writing a single line of code. They tell you exactly what the current frontier
is, what was tried and failed, and where your differentiation opportunity lies.

---

## Paper 1: Policy Induction
**Full title:** *Policy Induction: Predicting Startup Success via Explainable Memory-Augmented In-Context Learning*
**Authors:** Xianling Mu (Oxford), Joseph Ternasky, Fuat Alican, Yigit Ihlamur (Vela Research)
**arXiv:** [2505.21427](https://arxiv.org/abs/2505.21427) — June 2025
**VCBench rank:** #2 — Precision 41.0%, Recall 20.2%, F₀.₅ 34.0%

---

### The Core Idea

Policy Induction treats the investment decision as a **learnable natural language policy** — a
plain-English set of rules embedded directly in the LLM prompt, like:

> "Predict success if the founder has a prior exit AND a top-50 QS education AND a C-level role
> at a 50+ person company."

The model doesn't fine-tune any weights. Instead, it uses **in-context learning (ICL)** with a
memory-augmented loop: it sees a few labeled examples, forms a hypothesis about what patterns
predict success, encodes that hypothesis as a natural language policy, then iteratively refines the
policy based on structured feedback from prediction errors.

This is essentially the autoresearch loop applied to prompt engineering — the "program" (policy)
is the thing being optimized, not the model weights.

### What They Built

- A **few-shot ICL loop** where the LLM ingests labeled founder profiles and learns to articulate
  explicit rules
- A **memory module** that stores successful and failed prediction examples across iterations,
  so the policy gets reinforced rather than starting fresh each time
- A **natural language policy** as the final output — fully human-readable, auditable, and
  modifiable by domain experts without retraining
- Iterative refinement: the policy is tested, errors are analyzed, and the policy is updated;
  this repeats until performance stabilizes

### What Worked

- The framework is highly **data-efficient**: it achieves strong results with very few labeled
  examples, making it suitable for the VCBench setting where 9% positive rate means positives
  are scarce
- **Human-in-the-loop refinement** is easy — an expert VC can read the policy and say
  "that rule is wrong" and directly edit it, without understanding ML
- **Precision is well-calibrated**: the iterative refinement loop naturally drives the policy
  toward high-precision rules because low-precision rules create noisy signals that get pruned
- The final policy is **transferable**: the same natural language rules can be applied to new
  datasets without retraining

### What Didn't Work / Limitations

- **Recall is the weak point**: F₀.₅ of 34.0% with recall of only 20.2% means the policy is
  conservative. It misses many true positives. The precision-recall tradeoff is hard to resolve
  because the policy either becomes too specific (high precision, low recall) or too vague
  (catches more but with more noise)
- **Policy convergence is fragile**: the iterative refinement loop can get stuck in local optima
  — a policy that works well on the few-shot examples might not generalize to the full dataset
- **Sensitive to example selection**: the few-shot examples seeded into the memory module matter
  a lot. Bad initial examples can lead to degenerate policies that never recover
- **The framework can generate AI slop rules**: without careful prompting, the LLM will produce
  rules that sound like VC wisdom ("founder should have strong domain expertise") but are
  not grounded in the actual dataset's distributional patterns
- No structured feature engineering — the model works entirely from the anonymised prose text,
  which is a lossy representation of the underlying structured data

### Signal Ceiling Insights

The 34.0% F₀.₅ likely represents the ceiling for **prose-only, rule-induction approaches** on
this dataset. The marginal gains from iterating on the policy are diminishing — you can refine
the language but you can't recover signal that isn't in the text.

**Your opportunity:** Policy Induction operates on text. You operate on structure. The sacrifice
signal, education×QS interaction, and trajectory features are not recoverable from prose alone —
they require parsing the JSON fields. If your structured features add 5-10% more separable signal
than the prose, you can outperform this approach even without a memory-augmented loop.

---

## Paper 2: Random Rule Forest (RRF)
**Full title:** *Random Rule Forest (RRF): Interpretable Ensembles of LLM-Generated Questions for Predicting Startup Success*
**Authors:** Ben Griffin (Oxford), Joseph Ternasky, Fuat Alican, Yigit Ihlamur (Vela Research)
**arXiv:** [2505.24622](https://arxiv.org/abs/2505.24622) — May 2025 (v2: September 2025)
**VCBench rank:** #3 — Precision 42.5%, Recall 12.1%, F₀.₅ 28.1%

---

### The Core Idea

Random Rule Forest takes inspiration from ensemble learning (Random Forests, boosting) but
replaces decision tree nodes with **LLM-generated YES/NO questions**. Each question is a weak
heuristic:

> "Does this founder have prior experience as a CTO or CEO?"
> "Did this founder attend a top-25 ranked university?"
> "Has this founder previously founded a company that was acquired?"

The LLM generates ~100 of these questions, they are evaluated on a validation set for precision,
low-precision questions are filtered out, and the surviving questions vote together using a
threshold-based mechanism to produce a final prediction.

The key insight: **diversity of weak heuristics + aggressive filtering beats a single strong model**.

### What They Built

1. **Question generation phase**: prompt GPT-4o to generate 100 YES/NO questions about
   founder success indicators, with no constraints — let the LLM be creative
2. **Question evaluation phase**: run each question against the validation set, compute
   precision for each question individually (see Figure 2 in the paper — questions are scored
   like features)
3. **Filtering phase**: remove questions that don't exceed the random-chance precision
   threshold (~9-10% in this dataset)
4. **Ensemble construction**: combine surviving questions with a threshold-based vote —
   predict success only if K of N questions answer "YES"
5. **Expert-in-the-loop**: add 20 domain-expert-crafted questions to the pool alongside
   LLM-generated ones — this lifted F₀.₅ further

**Key finding from their appendix:** The top-10 high-precision questions are almost entirely
about structured, verifiable signals (prior exits, C-level roles, prestigious education, domain
alignment). The bottom-10 low-precision questions are about soft, prose-inference signals
("does the founder seem passionate?", "does the founder communicate clearly?").

### What Worked

- The filtering mechanism is extremely effective at removing noise — the gap between
  raw LLM questions and filtered questions is very large
- **Expert questions outperform LLM-generated questions on average** — human-crafted
  heuristics are more precision-oriented than LLM-generated ones, which tend toward
  prose-inferrable soft signals
- The threshold-based voting creates a natural precision bias: requiring more "YES" answers
  means fewer positives, but they're more reliable
- The framework is highly transparent and auditable — you can print the full decision logic
- **Precision of 42.5%** is strong, but the cost is very low recall (12.1%)

### What Didn't Work / Limitations

- **Recall is extremely low**: 12.1% recall with 42.5% precision means the model is
  highly conservative. It correctly identifies only 1 in 8 true positives. This is the
  fundamental tension in the RRF design — you can tune K to change the tradeoff but
  you can't escape the underlying signal limitation
- **Questions are redundant**: many LLM-generated questions are semantically similar,
  which reduces ensemble diversity. The filtering step removes many but not all redundancies
- **Questions are prose-dependent**: the YES/NO answering is done by prompting the LLM
  on the prose text, which means questions about structural signals (QS rank, prior exits)
  are answered from prose encoding rather than from the structured fields directly — lossy
- **No gradient signal**: the questions can't be jointly optimized. They're evaluated
  independently and combined by vote, which is less powerful than a learned ensemble
- **Grid search required**: finding the optimal (K questions, vote threshold) combination
  requires a grid search (see Figure 3 in the paper), which can overfit to the validation set

### Signal Ceiling Insights

The RRF paper explicitly shows its top-10 high-precision questions in the appendix. These are
effectively the **signal map for this domain** — they tell you what the LLM (and implicitly,
the dataset) treats as success indicators. The top-performing questions are all about:
- Prior exits or acquisitions
- C-level / founding roles at scale
- Top-ranked university attendance
- Domain expertise alignment

**Your opportunity:** RRF proves that YES/NO questions about structured features work — but
it answers them from prose. You can answer these exact questions directly from `jobs_json` and
`educations_json` with zero ambiguity. Your structured feature engineering is essentially RRF
without the LLM question-answering noise, plus the additional sacrifice signal that RRF's
question generation probably never surfaced (it's too abstract for a YES/NO question).

A concrete experiment to run: take RRF's top-10 questions and implement them as deterministic
rules from your structured features. Compare their precision to RRF's LLM-answered versions.
If deterministic rules are higher precision, that's a publishable finding.

---

## Paper 3: Think-Reason-Learn (Verifiable-RL)
**Full title:** Framework described at [github.com/Vela-Research/think-reason-learn](https://github.com/Vela-Research/think-reason-learn); leaderboard entry is "Verifiable-RL"
**Related paper:** R.A.I.S.E. — *Reasoning-Based AI for Startup Evaluation: A Memory-Augmented, Multi-Step Decision Framework* (Preuveneers, Ternasky, Alican, Ihlamur — arXiv:[2504.12090](https://arxiv.org/abs/2504.12090))
**VCBench rank:** #1 — Precision 42.6%, Recall 23.6%, F₀.₅ 36.6%

---

### The Core Idea

Think-Reason-Learn (TRL) is described as "scikit-learn with reasoning built in" — a modular
ML framework that keeps the interpretability of classical ML while giving models the context
understanding and generalization of LLMs.

The Verifiable-RL approach (which achieves the current top score) applies **Reinforcement
Learning with Verifiable Rewards (RLVR)** — the same paradigm as DeepSeek-R1 — but applied
to the founder prediction task rather than math reasoning.

The RLVR loop:
1. LLM generates a prediction (success/failure) with reasoning chain
2. Reward = 1 if prediction matches ground truth label, 0 otherwise (binary, verifiable)
3. GRPO-style policy gradient update: reinforce reasoning paths that led to correct predictions
4. The model learns *which patterns of reasoning* reliably lead to correct outcomes, without
   being told what those patterns should be — they emerge from the training signal

The R.A.I.S.E. variant (which is likely the predecessor or closely related):
- Uses chain-of-thought prompting to generate detailed reasoning logs
- Distills those logs into structured IF-THEN rules (like Policy Induction but more automated)
- Iteratively refines rules using simulated RL scoring
- Achieves precision of 34.6% (+54% over baseline o3)

### What They Built (TRL Framework)

The TRL framework includes two main algorithms available as open-source:

**GPTree**: LLM-guided decision trees where the LLM dynamically generates features/splits at
each node, rather than using fixed numerical features. The tree is constructed top-down, and at
each node the LLM is asked "what question would best separate these founders into success/failure?"

**RRF (Random Rule Forest)**: The Paper 2 approach above, packaged as a reusable module.

**Verifiable-RL**: The top-performing entry — a trained model (likely fine-tuned or GRPO-trained
on a small LLM) that produces reasoning traces rewarded by prediction accuracy against labels.

### What Worked

- **Verifiable-RL is the current ceiling**: 36.6% F₀.₅ is the best result on the leaderboard
  across any approach. It achieves both the highest F₀.₅ AND reasonable recall (23.6%) —
  a better precision-recall balance than the pure rule-based methods
- **The RLVR loop discovers non-obvious patterns**: because the reward is binary and verifiable,
  the model is forced to find reasoning paths that actually correlate with ground truth, not
  just with prose features that sound like VC wisdom
- The TRL framework is **domain-agnostic** — the same approach applies to law, healthcare,
  finance; the VC results are a proof-of-concept for the broader framework

### What Didn't Work / Limitations

- **Recall is still limited**: even the best model only catches 23.6% of true successes. This
  is a fundamental dataset ceiling — the signal in founder profiles alone (without company
  data, network data, idea data) may not support much higher recall
- **The RL training requires labeled data**: RLVR needs ground truth labels to compute rewards.
  On 4,500 rows with 9% positive rate, you have only ~405 positive examples — this is small
  for stable RL training
- **Reproducibility**: the RLVR training process is stochastic and sensitive to initialization.
  The leaderboard number likely represents a best run, not a stable average
- **Compute requirement**: RLVR fine-tuning requires GPU time and careful hyperparameter
  management — significantly more complex than XGBoost + structured features

### Signal Ceiling Insights

The 36.6% F₀.₅ from Verifiable-RL likely represents something close to the **information-theoretic
ceiling** for founder-profile-only prediction on this dataset. Here's why:

- The dataset was deliberately anonymized to remove company names, exact dates, institutions
- The anonymization pipeline preserves *predictive structure* but degrades *identity leakage*
- The VCBench paper explicitly notes that residual noise from anonymization reduces signal
- At 9% positive rate, even a perfect classifier needs to be very precise — small amounts of
  label noise or feature noise push the precision ceiling down significantly

**What this means for your approach:** The gap between your target (33%+) and the current best
(36.6%) is small — roughly 3-4 F₀.₅ points. That gap can likely be closed by structured feature
engineering (which TRL's current Verifiable-RL entry does not appear to use — it works from
prose like the other methods). If you implement structured features AND an ensemble with an LLM
method, you have a credible path to matching or exceeding 36.6%.

---

## Synthesis: What None of Them (To Our Knowledge) Did — Your Opportunity Space

After reading all three papers, the consistent gap is:

| Signal Type | Policy Induction | RRF | Verifiable-RL | Your Plan |
|---|---|---|---|---|
| Works from prose | ✅ | ✅ (confirmed) | ✅ | ✅ (secondary) |
| Parses structured JSON directly | unknown | ❌ (confirmed) | unknown | ✅ (primary) |
| Sacrifice/opportunity cost signal | ❌ | ❌ | ❌ | ✅ |
| Education × QS interaction term | ❌ | ❌ | ❌ | ✅ |
| Threshold calibration (>0.5) | Partial | ✅ | Partial | ✅ |
| Gradient-based classifier | ❌ | ❌ | ✅ | ✅ (XGBoost) |
| Ablation study of feature groups | ❌ | Partial | ❌ | ✅ (planned) |

**Important epistemic note:** RRF is confirmed prose-only — they explicitly prompt an LLM on
the text to answer YES/NO questions. Policy Induction and Verifiable-RL describe working from
"founder profiles" without confirming whether they parse the JSON fields. The honest paper
claim is: *"To our knowledge, no prior work has directly engineered features from the structured
JSON fields."* Do not write "all models use prose only" — that may be wrong and a Vela reviewer
would flag it immediately.

**The actual core contribution:** `anonymised_prose` is generated *from* the structured fields
by the VCBench pipeline — it is a lossy re-encoding, not a richer source. Ordinal relationships
in company size/duration buckets, field interaction terms, null signals (empty `ipos` silently
omitted from prose), and job sequence information all degrade in text rendering. A paper that
parses the source JSON, ablates the signal loss introduced by the prose rendering, and shows
that structured features recover measurable F₀.₅ is novel, directly falsifiable, and fills a
gap the existing papers acknowledge but don't address.

That is your paper.

---

## Key Numbers to Know

| Paper | Precision | Recall | F₀.₅ | Dataset size | Positive rate |
|---|---|---|---|---|---|
| Policy Induction | 41.0% | 20.2% | 34.0% | 9,000 | 9% |
| Random Rule Forest | 42.5% | 12.1% | 28.1% | 9,892 | 10% |
| Verifiable-RL (best) | 42.6% | 23.6% | 36.6% | (VCBench) | 9% |
| GPT-4o zero-shot | 30.0% | 16.3% | 25.7% | VCBench | 9% |
| Your target | >40% | >20% | >33% | 4,500 | 9% |

Note: The RRF paper reports higher absolute precision numbers in their own test (50–54%) versus
the VCBench leaderboard (42.5%) — the VCBench standardized evaluation is stricter than their
internal test set. Use leaderboard numbers for honest comparison.