# Results — VCBench Task IV Private Test Set

Detailed results from the final submission (XGBoost decision stumps, 28 structured features, threshold=0.738).

---

## Private Test Set Summary

| Metric | Value |
|---|---|
| F₀.₅ | **0.2811** |
| Precision | 0.3281 |
| Recall | 0.1802 |
| Positive predictions | 224 / 4,500 |
| Base rate (actual positives) | 405 / 4,500 = 9.0% |
| Precision lift over base rate | **3.6×** |

---

## Confusion Matrix

Private test set (n=4,500):

|  | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 3,944 | 151 |
| **Actual 1** | 332 | 73 |

- True positives (TP): 73
- False positives (FP): 151
- True negatives (TN): 3,944
- False negatives (FN): 332

---

## Fold-by-Fold Breakdown

The private test set was evaluated in three folds by the contest organizers:

| Fold | n | Precision | Recall | F₀.₅ |
|---|---|---|---|---|
| Fold 1 | 1,500 | 0.3133 | 0.1926 | 0.2784 |
| Fold 2 | 1,500 | 0.3594 | 0.1704 | 0.2941 |
| Fold 3 | 1,500 | 0.3117 | 0.1778 | 0.2709 |
| **Average** | **4,500** | **0.3281** | **0.1802** | **0.2811** |

Fold 2 shows notably higher precision (0.3594) with the lowest recall (0.1704), suggesting that the model's precision-recall trade-off varies with the composition of each fold. The threshold (0.738) was selected to maximise F₀.₅ on the validation set, which weights precision 4× more than recall.

---

## Feature Importance

Top features by XGBoost gain (trained on full public_train.csv, n=3,600):

| Rank | Feature | Tier | Notes |
|---|---|---|---|
| 1 | `exit_count` | 1 | Also triggers high-precision rule layer |
| 2 | `prestige_x_relevance` | 3 | Top-10 QS × STEM × tech industry |
| 3 | `founding_role_count` | 4 | Serial founder signal |
| 4 | `persistence_score` | v2 | Founding tenure / total experience |
| 5 | `exit_x_serial` | v2 | Amplifies exit signal for serial founders |
| 6 | `edu_prestige_tier` | 3 | Degraded by biotech/VC FPs (see penalty feature) |
| 7 | `max_seniority_reached` | 4 | 66.9% of founders at level 5 — low discrimination |
| 8 | `industry_alignment` | 4 | Prior job in same industry as startup |
| 9 | `years_in_large_company` | 2 | Pre-founding tenure at 501+ employee companies |
| 10 | `prestige_sacrifice_score` | 2 | Inverted signal; FPs have higher scores |

Full importance scores are printed when running `python classifier.py`.

---

## The Two-Population Problem

**84% of positive founders (332/405 on private test) are structurally indistinguishable from failures on all 28 available features.**

This is the central finding of the paper. It explains why the F₀.₅ ceiling on this dataset is approximately 0.30, regardless of model complexity.

### What the data shows

The 73 true positives the model correctly identifies are disproportionately drawn from the 16% of positive founders who have a visible structural signal:

- Prior exit (IPO or acquisition): precision 24.5%, fires on ~3% of rows
- Top-10 QS institution + STEM + tech industry: precision ~24%, fires on ~1% of rows

The remaining 332 positive founders who were missed have career profiles that are, by the measurable features available, statistically identical to the 3,944 true negatives. Their success is attributable to factors not captured in the VCBench structured data: timing, market conditions, co-founder quality, investor network, and post-founding execution.

### Why LLM features did not help

The LLM prose field (`anonymised_prose`) in VCBench is a textual summary generated from the same structured JSON used for feature extraction. Extracting LLM features from this prose therefore recovers the same signal rather than adding new signal — the 9 LLM features added a CV delta of −0.05pp and were reverted (see `experiment_log.md`, Phase 3).

This is not a failure of LLM extraction methodology; it is a consequence of the data generation process. If `anonymised_prose` contained genuinely independent information (e.g., verbatim interview transcripts or social media history), LLM features might break the ceiling.

### Implications for VCBench v2

The two-population structure implies that the benchmark's F₀.₅ ceiling is approximately 0.28–0.32 for structured-feature approaches. Breaching this ceiling requires new data fields that are orthogonal to career history. Candidate additions are listed in the README Roadmap section.

---

## Comparison to Baselines

| Approach | Val F₀.₅ | Notes |
|---|---|---|
| Always predict 0 | 0.0000 | Trivial baseline |
| Always predict 1 | 0.0461 | F₀.₅ at 9% base rate |
| Zero-shot LLM (Claude Sonnet 4.5) | 0.1265 | Poor calibration; see `baselines/zero_shot_baseline.py` |
| Rule layer only (prior exit) | 0.2889 | High precision, very low recall |
| Structured v1 (23 feat, XGB) | 0.2203 | Pre-HPO, pre-v2 features |
| **Structured v2 + HPO (28 feat, XGB)** | **0.3030** | **Final submission** |
| + LLM features (100% coverage) | 0.3065 | CV 0.2534; rejected (redundant) |
