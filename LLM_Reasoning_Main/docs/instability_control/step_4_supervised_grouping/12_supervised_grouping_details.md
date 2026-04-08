# Supervised Grouping Details

This note records the grouped-route deltas, representative cluster layouts, and stable co-clustering patterns.

## Augmentation | HQ + A | 2 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.302 | 0.283 | -0.018 | 0.226 | 0.217 | 0.269 | 0.260 |
| xgb1 | 0.298 | 0.302 | 0.004 | 0.234 | 0.231 | 0.711 | 0.716 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `A`:
  A_grp_1: A_career_coherence, A_evidence_support_rating, A_trajectory_strength
  A_grp_2: A_ownership_signal, A_scrappiness

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| A | A_career_coherence | A_evidence_support_rating | 1.000 |
| A | A_career_coherence | A_trajectory_strength | 1.000 |
| A | A_evidence_support_rating | A_trajectory_strength | 1.000 |
| A | A_ownership_signal | A_scrappiness | 1.000 |
| A | A_career_coherence | A_ownership_signal | 0.000 |
| A | A_career_coherence | A_scrappiness | 0.000 |
| A | A_evidence_support_rating | A_ownership_signal | 0.000 |
| A | A_evidence_support_rating | A_scrappiness | 0.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| A | A_career_coherence / A_evidence_support_rating / A_trajectory_strength | A_evidence_support_rating | 0.926 | 0.029 | 0.000 | 0.737 |
| A | A_ownership_signal / A_scrappiness | A_scrappiness | 0.776 | 0.067 | 0.000 | 0.723 |
| A | A_ownership_signal / A_scrappiness | A_ownership_signal | 0.621 | 0.090 | 0.000 | 0.705 |
| A | A_career_coherence / A_evidence_support_rating / A_trajectory_strength | A_career_coherence | 0.330 | 0.058 | 0.000 | 0.674 |
| A | A_career_coherence / A_evidence_support_rating / A_trajectory_strength | A_trajectory_strength | 0.150 | 0.081 | 0.083 | 0.652 |

## Augmentation | HQ + A | 3 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.302 | 0.301 | -0.001 | 0.226 | 0.226 | 0.269 | 0.271 |
| xgb1 | 0.298 | 0.308 | 0.009 | 0.234 | 0.235 | 0.711 | 0.707 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `A`:
  A_grp_1: A_ownership_signal, A_scrappiness
  A_grp_2: A_career_coherence, A_trajectory_strength
  A_grp_3: A_evidence_support_rating

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| A | A_career_coherence | A_trajectory_strength | 1.000 |
| A | A_ownership_signal | A_scrappiness | 1.000 |
| A | A_career_coherence | A_evidence_support_rating | 0.000 |
| A | A_career_coherence | A_ownership_signal | 0.000 |
| A | A_career_coherence | A_scrappiness | 0.000 |
| A | A_evidence_support_rating | A_ownership_signal | 0.000 |
| A | A_evidence_support_rating | A_scrappiness | 0.000 |
| A | A_evidence_support_rating | A_trajectory_strength | 0.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| A | A_evidence_support_rating | A_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| A | A_career_coherence / A_trajectory_strength | A_career_coherence | 0.904 | 0.047 | 0.000 | 0.808 |
| A | A_ownership_signal / A_scrappiness | A_scrappiness | 0.776 | 0.067 | 0.000 | 0.723 |
| A | A_ownership_signal / A_scrappiness | A_ownership_signal | 0.621 | 0.090 | 0.000 | 0.705 |
| A | A_career_coherence / A_trajectory_strength | A_trajectory_strength | 0.375 | 0.202 | 0.083 | 0.749 |

## Augmentation | HQ + A+B+C+D+E | 2 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.304 | 0.318 | 0.014 | 0.234 | 0.230 | 0.268 | 0.285 |
| xgb1 | 0.302 | 0.298 | -0.004 | 0.234 | 0.231 | 0.708 | 0.713 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `A`:
  A_grp_1: A_career_coherence, A_evidence_support_rating, A_trajectory_strength
  A_grp_2: A_ownership_signal, A_scrappiness
- Scope `B`:
  B_grp_1: B_rubric_score
  B_grp_2: B_evidence_support_rating
- Scope `C`:
  C_grp_1: C_evidence_support_rating, C_latent_strength
  C_grp_2: C_latent_risk
- Scope `D`:
  D_grp_1: D_evidence_support_rating
  D_grp_2: D_overall_founder_quality
- Scope `E`:
  E_grp_1: E_evidence_support_rating, E_hidden_upside
  E_grp_2: E_hidden_risk

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| A | A_career_coherence | A_evidence_support_rating | 1.000 |
| A | A_career_coherence | A_trajectory_strength | 1.000 |
| A | A_evidence_support_rating | A_trajectory_strength | 1.000 |
| A | A_ownership_signal | A_scrappiness | 1.000 |
| C | C_evidence_support_rating | C_latent_strength | 1.000 |
| E | E_evidence_support_rating | E_hidden_upside | 1.000 |
| A | A_career_coherence | A_ownership_signal | 0.000 |
| A | A_career_coherence | A_scrappiness | 0.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| B | B_evidence_support_rating | B_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| B | B_rubric_score | B_rubric_score | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_latent_risk | C_latent_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_evidence_support_rating | D_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_overall_founder_quality | D_overall_founder_quality | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_hidden_risk | E_hidden_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| A | A_career_coherence / A_evidence_support_rating / A_trajectory_strength | A_evidence_support_rating | 0.926 | 0.029 | 0.000 | 0.737 |
| C | C_evidence_support_rating / C_latent_strength | C_latent_strength | 0.762 | 0.306 | 0.042 | 0.819 |
| A | A_ownership_signal / A_scrappiness | A_scrappiness | 0.776 | 0.067 | 0.000 | 0.723 |
| E | E_evidence_support_rating / E_hidden_upside | E_hidden_upside | 0.668 | 0.248 | 0.021 | 0.754 |

## Augmentation | HQ + A+B+C+D+E | 3 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.304 | 0.306 | 0.002 | 0.234 | 0.235 | 0.268 | 0.271 |
| xgb1 | 0.302 | 0.304 | 0.002 | 0.234 | 0.235 | 0.708 | 0.713 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `A`:
  A_grp_1: A_ownership_signal, A_scrappiness
  A_grp_2: A_career_coherence, A_trajectory_strength
  A_grp_3: A_evidence_support_rating
- Scope `B`:
  B_grp_1: B_rubric_score
  B_grp_2: B_evidence_support_rating
- Scope `C`:
  C_grp_1: C_evidence_support_rating
  C_grp_2: C_latent_risk
  C_grp_3: C_latent_strength
- Scope `D`:
  D_grp_1: D_evidence_support_rating
  D_grp_2: D_overall_founder_quality
- Scope `E`:
  E_grp_1: E_hidden_risk
  E_grp_2: E_hidden_upside
  E_grp_3: E_evidence_support_rating

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| A | A_career_coherence | A_trajectory_strength | 1.000 |
| A | A_ownership_signal | A_scrappiness | 1.000 |
| A | A_career_coherence | A_evidence_support_rating | 0.000 |
| A | A_career_coherence | A_ownership_signal | 0.000 |
| A | A_career_coherence | A_scrappiness | 0.000 |
| A | A_evidence_support_rating | A_ownership_signal | 0.000 |
| A | A_evidence_support_rating | A_scrappiness | 0.000 |
| A | A_evidence_support_rating | A_trajectory_strength | 0.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| A | A_evidence_support_rating | A_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| B | B_evidence_support_rating | B_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| B | B_rubric_score | B_rubric_score | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_evidence_support_rating | C_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_latent_risk | C_latent_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_latent_strength | C_latent_strength | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_evidence_support_rating | D_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_overall_founder_quality | D_overall_founder_quality | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_evidence_support_rating | E_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_hidden_risk | E_hidden_risk | 1.000 | 0.000 | 0.000 | 1.000 |

## Augmentation | HQ + B+C+D+E+F | 2 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.252 | 0.247 | -0.004 | 0.190 | 0.187 | 0.258 | 0.246 |
| xgb1 | 0.215 | 0.221 | 0.006 | 0.178 | 0.179 | 0.694 | 0.688 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `B`:
  B_grp_1: B_rubric_score
  B_grp_2: B_evidence_support_rating
- Scope `C`:
  C_grp_1: C_evidence_support_rating, C_latent_strength
  C_grp_2: C_latent_risk
- Scope `D`:
  D_grp_1: D_evidence_support_rating
  D_grp_2: D_overall_founder_quality
- Scope `E`:
  E_grp_1: E_evidence_support_rating, E_hidden_upside
  E_grp_2: E_hidden_risk
- Scope `F`:
  F_grp_1: F_career_coherence, F_ownership_signal, F_rubric_score, F_scrappiness, F_trajectory_strength
  F_grp_2: F_evidence_support_rating

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| C | C_evidence_support_rating | C_latent_strength | 1.000 |
| E | E_evidence_support_rating | E_hidden_upside | 1.000 |
| F | F_career_coherence | F_ownership_signal | 1.000 |
| F | F_career_coherence | F_rubric_score | 1.000 |
| F | F_career_coherence | F_scrappiness | 1.000 |
| F | F_career_coherence | F_trajectory_strength | 1.000 |
| F | F_ownership_signal | F_rubric_score | 1.000 |
| F | F_ownership_signal | F_scrappiness | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| B | B_evidence_support_rating | B_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| B | B_rubric_score | B_rubric_score | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_latent_risk | C_latent_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_evidence_support_rating | D_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_overall_founder_quality | D_overall_founder_quality | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_hidden_risk | E_hidden_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| F | F_evidence_support_rating | F_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_evidence_support_rating / C_latent_strength | C_latent_strength | 0.762 | 0.306 | 0.042 | 0.819 |
| E | E_evidence_support_rating / E_hidden_upside | E_hidden_upside | 0.668 | 0.248 | 0.021 | 0.754 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_career_coherence | 0.643 | 0.276 | 0.083 | 0.555 |

## Augmentation | HQ + B+C+D+E+F | 3 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.252 | 0.245 | -0.007 | 0.190 | 0.187 | 0.258 | 0.256 |
| xgb1 | 0.215 | 0.217 | 0.002 | 0.178 | 0.178 | 0.694 | 0.687 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `B`:
  B_grp_1: B_rubric_score
  B_grp_2: B_evidence_support_rating
- Scope `C`:
  C_grp_1: C_evidence_support_rating
  C_grp_2: C_latent_risk
  C_grp_3: C_latent_strength
- Scope `D`:
  D_grp_1: D_evidence_support_rating
  D_grp_2: D_overall_founder_quality
- Scope `E`:
  E_grp_1: E_hidden_risk
  E_grp_2: E_hidden_upside
  E_grp_3: E_evidence_support_rating
- Scope `F`:
  F_grp_1: F_career_coherence, F_ownership_signal, F_rubric_score, F_trajectory_strength
  F_grp_2: F_evidence_support_rating
  F_grp_3: F_scrappiness

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| F | F_career_coherence | F_ownership_signal | 1.000 |
| F | F_career_coherence | F_rubric_score | 1.000 |
| F | F_career_coherence | F_trajectory_strength | 1.000 |
| F | F_ownership_signal | F_rubric_score | 1.000 |
| F | F_ownership_signal | F_trajectory_strength | 1.000 |
| F | F_rubric_score | F_trajectory_strength | 1.000 |
| B | B_evidence_support_rating | B_rubric_score | 0.000 |
| C | C_evidence_support_rating | C_latent_risk | 0.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| B | B_evidence_support_rating | B_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| B | B_rubric_score | B_rubric_score | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_evidence_support_rating | C_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_latent_risk | C_latent_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| C | C_latent_strength | C_latent_strength | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_evidence_support_rating | D_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| D | D_overall_founder_quality | D_overall_founder_quality | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_evidence_support_rating | E_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_hidden_risk | E_hidden_risk | 1.000 | 0.000 | 0.000 | 1.000 |
| E | E_hidden_upside | E_hidden_upside | 1.000 | 0.000 | 0.000 | 1.000 |

## Augmentation | HQ + F | 2 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.227 | 0.228 | 0.000 | 0.178 | 0.174 | 0.231 | 0.229 |
| xgb1 | 0.206 | 0.207 | 0.001 | 0.173 | 0.174 | 0.705 | 0.703 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `F`:
  F_grp_1: F_career_coherence, F_ownership_signal, F_rubric_score, F_scrappiness, F_trajectory_strength
  F_grp_2: F_evidence_support_rating

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| F | F_career_coherence | F_ownership_signal | 1.000 |
| F | F_career_coherence | F_rubric_score | 1.000 |
| F | F_career_coherence | F_scrappiness | 1.000 |
| F | F_career_coherence | F_trajectory_strength | 1.000 |
| F | F_ownership_signal | F_rubric_score | 1.000 |
| F | F_ownership_signal | F_scrappiness | 1.000 |
| F | F_ownership_signal | F_trajectory_strength | 1.000 |
| F | F_rubric_score | F_scrappiness | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| F | F_evidence_support_rating | F_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_career_coherence | 0.643 | 0.276 | 0.083 | 0.555 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_rubric_score | 0.397 | 0.071 | 0.000 | 0.555 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_trajectory_strength | 0.364 | 0.085 | 0.000 | 0.554 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_scrappiness | 0.247 | 0.225 | 0.125 | 0.528 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_ownership_signal | 0.270 | 0.152 | 0.083 | 0.551 |

## Augmentation | HQ + F | 3 groups per family

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.227 | 0.227 | -0.000 | 0.178 | 0.174 | 0.231 | 0.231 |
| xgb1 | 0.206 | 0.209 | 0.003 | 0.173 | 0.174 | 0.705 | 0.701 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `F`:
  F_grp_1: F_career_coherence, F_ownership_signal, F_rubric_score, F_trajectory_strength
  F_grp_2: F_evidence_support_rating
  F_grp_3: F_scrappiness

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| F | F_career_coherence | F_ownership_signal | 1.000 |
| F | F_career_coherence | F_rubric_score | 1.000 |
| F | F_career_coherence | F_trajectory_strength | 1.000 |
| F | F_ownership_signal | F_rubric_score | 1.000 |
| F | F_ownership_signal | F_trajectory_strength | 1.000 |
| F | F_rubric_score | F_trajectory_strength | 1.000 |
| F | F_career_coherence | F_evidence_support_rating | 0.000 |
| F | F_career_coherence | F_scrappiness | 0.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| F | F_evidence_support_rating | F_evidence_support_rating | 1.000 | 0.000 | 0.000 | 1.000 |
| F | F_scrappiness | F_scrappiness | 1.000 | 0.000 | 0.000 | 1.000 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_trajectory_strength | F_career_coherence | 0.667 | 0.298 | 0.083 | 0.576 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_trajectory_strength | F_rubric_score | 0.423 | 0.076 | 0.000 | 0.577 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_trajectory_strength | F_trajectory_strength | 0.389 | 0.094 | 0.000 | 0.576 |
| F | F_career_coherence / F_ownership_signal / F_rubric_score / F_trajectory_strength | F_ownership_signal | 0.295 | 0.184 | 0.083 | 0.573 |

## Competition | HQ | 6 total groups

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.207 | 0.206 | -0.001 | 0.167 | 0.166 | 0.237 | 0.190 |
| xgb1 | 0.204 | 0.197 | -0.007 | 0.171 | 0.163 | 0.702 | 0.715 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `HQ`:
  competition_grp_1: founding_role_count, founding_timing, industry_alignment, industry_pivot_count, is_serial_founder, longest_founding_tenure, max_company_size_before_founding, max_seniority_reached, persistence_score, prestige_sacrifice_score, restlessness_score, sacrifice_x_serial, total_inferred_experience, years_in_large_company
  competition_grp_2: best_degree_prestige, degree_level, edu_prestige_tier, field_relevance_score, industry_prestige_penalty, prestige_x_relevance, stem_flag
  competition_grp_3: company_size_is_growing, seniority_is_monotone
  competition_grp_4: comfort_index
  competition_grp_5: repeat_founding_gap
  competition_grp_6: exit_count, exit_x_serial, has_prior_acquisition, has_prior_ipo

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| HQ | best_degree_prestige | degree_level | 1.000 |
| HQ | best_degree_prestige | edu_prestige_tier | 1.000 |
| HQ | best_degree_prestige | field_relevance_score | 1.000 |
| HQ | best_degree_prestige | industry_prestige_penalty | 1.000 |
| HQ | best_degree_prestige | prestige_x_relevance | 1.000 |
| HQ | best_degree_prestige | stem_flag | 1.000 |
| HQ | company_size_is_growing | seniority_is_monotone | 1.000 |
| HQ | degree_level | edu_prestige_tier | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| HQ | comfort_index | comfort_index | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ | industry_alignment | industry_alignment | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ | repeat_founding_gap | repeat_founding_gap | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ | company_size_is_growing / seniority_is_monotone | company_size_is_growing | 0.756 | 0.222 | 0.026 | 0.751 |
| HQ | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_count | 0.580 | 0.014 | 0.000 | 0.579 |
| HQ | company_size_is_growing / seniority_is_monotone | seniority_is_monotone | 0.570 | 0.235 | 0.000 | 0.640 |
| HQ | company_size_is_growing / founding_role_count / founding_timing / industry_pivot_count / is_serial_founder / longest_founding_tenure / max_company_size_before_founding / max_seniority_reached / persistence_score / prestige_sacrifice_score / restlessness_score / sacrifice_x_serial / seniority_is_monotone / total_inferred_experience / years_in_large_company | prestige_sacrifice_score | 0.566 | 0.026 | 0.000 | 0.594 |
| HQ | founding_role_count / founding_timing / industry_alignment / industry_pivot_count / is_serial_founder / longest_founding_tenure / max_company_size_before_founding / max_seniority_reached / persistence_score / prestige_sacrifice_score / restlessness_score / sacrifice_x_serial / total_inferred_experience / years_in_large_company | prestige_sacrifice_score | 0.521 | 0.027 | 0.000 | 0.541 |
| HQ | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_x_serial | 0.518 | 0.034 | 0.000 | 0.543 |
| HQ | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | has_prior_acquisition | 0.506 | 0.032 | 0.000 | 0.525 |

## Competition | HQ + A | 6 total groups

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.302 | 0.300 | -0.002 | 0.226 | 0.214 | 0.269 | 0.225 |
| xgb1 | 0.298 | 0.299 | 0.001 | 0.234 | 0.220 | 0.711 | 0.757 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `HQ + A`:
  competition_grp_1: company_size_is_growing, founding_role_count, founding_timing, industry_alignment, industry_pivot_count, is_serial_founder, longest_founding_tenure, max_company_size_before_founding, max_seniority_reached, persistence_score, prestige_sacrifice_score, restlessness_score, sacrifice_x_serial, seniority_is_monotone, total_inferred_experience, years_in_large_company
  competition_grp_2: exit_count, exit_x_serial, has_prior_acquisition, has_prior_ipo
  competition_grp_3: best_degree_prestige, degree_level, edu_prestige_tier, field_relevance_score, industry_prestige_penalty, prestige_x_relevance, stem_flag
  competition_grp_4: A_career_coherence, A_evidence_support_rating, A_ownership_signal, A_scrappiness, A_trajectory_strength
  competition_grp_5: comfort_index
  competition_grp_6: repeat_founding_gap

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| HQ + A | A_career_coherence | A_evidence_support_rating | 1.000 |
| HQ + A | A_career_coherence | A_ownership_signal | 1.000 |
| HQ + A | A_career_coherence | A_scrappiness | 1.000 |
| HQ + A | A_career_coherence | A_trajectory_strength | 1.000 |
| HQ + A | A_evidence_support_rating | A_ownership_signal | 1.000 |
| HQ + A | A_evidence_support_rating | A_scrappiness | 1.000 |
| HQ + A | A_evidence_support_rating | A_trajectory_strength | 1.000 |
| HQ + A | A_ownership_signal | A_scrappiness | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| HQ + A | comfort_index | comfort_index | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + A | repeat_founding_gap | repeat_founding_gap | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + A | A_career_coherence / A_evidence_support_rating / A_ownership_signal / A_scrappiness / A_trajectory_strength | A_evidence_support_rating | 0.839 | 0.033 | 0.000 | 0.929 |
| HQ + A | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_count | 0.580 | 0.014 | 0.000 | 0.579 |
| HQ + A | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_x_serial | 0.518 | 0.034 | 0.000 | 0.543 |
| HQ + A | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | has_prior_acquisition | 0.506 | 0.032 | 0.000 | 0.525 |
| HQ + A | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | prestige_x_relevance | 0.504 | 0.019 | 0.000 | 0.495 |
| HQ + A | company_size_is_growing / founding_role_count / founding_timing / industry_alignment / industry_pivot_count / is_serial_founder / longest_founding_tenure / max_company_size_before_founding / max_seniority_reached / persistence_score / prestige_sacrifice_score / restlessness_score / sacrifice_x_serial / seniority_is_monotone / total_inferred_experience / years_in_large_company | prestige_sacrifice_score | 0.501 | 0.030 | 0.000 | 0.533 |
| HQ + A | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | edu_prestige_tier | 0.498 | 0.022 | 0.000 | 0.474 |
| HQ + A | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | best_degree_prestige | 0.498 | 0.022 | 0.000 | 0.474 |

## Competition | HQ + A+B+C+D+E | 6 total groups

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.304 | 0.267 | -0.037 | 0.234 | 0.198 | 0.268 | 0.198 |
| xgb1 | 0.302 | 0.282 | -0.021 | 0.234 | 0.206 | 0.708 | 0.744 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `HQ + A+B+C+D+E`:
  competition_grp_1: company_size_is_growing, founding_role_count, founding_timing, industry_alignment, industry_pivot_count, is_serial_founder, longest_founding_tenure, max_company_size_before_founding, max_seniority_reached, persistence_score, prestige_sacrifice_score, restlessness_score, sacrifice_x_serial, seniority_is_monotone, total_inferred_experience, years_in_large_company
  competition_grp_2: exit_count, exit_x_serial, has_prior_acquisition, has_prior_ipo
  competition_grp_3: best_degree_prestige, degree_level, edu_prestige_tier, field_relevance_score, industry_prestige_penalty, prestige_x_relevance, stem_flag
  competition_grp_4: A_career_coherence, A_evidence_support_rating, A_ownership_signal, A_scrappiness, A_trajectory_strength, B_evidence_support_rating, B_rubric_score, C_evidence_support_rating, C_latent_risk, C_latent_strength, D_evidence_support_rating, D_overall_founder_quality, E_evidence_support_rating, E_hidden_risk, E_hidden_upside
  competition_grp_5: comfort_index
  competition_grp_6: repeat_founding_gap

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| HQ + A+B+C+D+E | A_career_coherence | A_evidence_support_rating | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | A_ownership_signal | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | A_scrappiness | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | A_trajectory_strength | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | B_evidence_support_rating | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | B_rubric_score | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | C_evidence_support_rating | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence | C_latent_risk | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| HQ + A+B+C+D+E | comfort_index | comfort_index | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + A+B+C+D+E | repeat_founding_gap | repeat_founding_gap | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + A+B+C+D+E | A_career_coherence / A_evidence_support_rating / A_ownership_signal / A_scrappiness / A_trajectory_strength / B_evidence_support_rating / B_rubric_score / C_evidence_support_rating / C_latent_risk / C_latent_strength / D_evidence_support_rating / D_overall_founder_quality / E_evidence_support_rating / E_hidden_risk / E_hidden_upside | A_evidence_support_rating | 0.764 | 0.034 | 0.000 | 0.631 |
| HQ + A+B+C+D+E | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_count | 0.580 | 0.014 | 0.000 | 0.579 |
| HQ + A+B+C+D+E | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_x_serial | 0.518 | 0.034 | 0.000 | 0.543 |
| HQ + A+B+C+D+E | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | has_prior_acquisition | 0.506 | 0.032 | 0.000 | 0.525 |
| HQ + A+B+C+D+E | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | prestige_x_relevance | 0.504 | 0.019 | 0.000 | 0.495 |
| HQ + A+B+C+D+E | company_size_is_growing / founding_role_count / founding_timing / industry_alignment / industry_pivot_count / is_serial_founder / longest_founding_tenure / max_company_size_before_founding / max_seniority_reached / persistence_score / prestige_sacrifice_score / restlessness_score / sacrifice_x_serial / seniority_is_monotone / total_inferred_experience / years_in_large_company | prestige_sacrifice_score | 0.501 | 0.030 | 0.000 | 0.533 |
| HQ + A+B+C+D+E | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | edu_prestige_tier | 0.498 | 0.022 | 0.000 | 0.474 |
| HQ + A+B+C+D+E | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | best_degree_prestige | 0.498 | 0.022 | 0.000 | 0.474 |

## Competition | HQ + B+C+D+E+F | 6 total groups

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.252 | 0.214 | -0.038 | 0.190 | 0.168 | 0.258 | 0.182 |
| xgb1 | 0.215 | 0.218 | 0.003 | 0.178 | 0.168 | 0.694 | 0.709 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `HQ + B+C+D+E+F`:
  competition_grp_1: company_size_is_growing, founding_role_count, founding_timing, industry_alignment, industry_pivot_count, is_serial_founder, longest_founding_tenure, max_company_size_before_founding, max_seniority_reached, persistence_score, prestige_sacrifice_score, restlessness_score, sacrifice_x_serial, seniority_is_monotone, total_inferred_experience, years_in_large_company
  competition_grp_2: exit_count, exit_x_serial, has_prior_acquisition, has_prior_ipo
  competition_grp_3: best_degree_prestige, degree_level, edu_prestige_tier, field_relevance_score, industry_prestige_penalty, prestige_x_relevance, stem_flag
  competition_grp_4: B_evidence_support_rating, B_rubric_score, C_evidence_support_rating, C_latent_risk, C_latent_strength, D_evidence_support_rating, D_overall_founder_quality, E_evidence_support_rating, E_hidden_risk, E_hidden_upside, F_career_coherence, F_evidence_support_rating, F_ownership_signal, F_rubric_score, F_scrappiness, F_trajectory_strength
  competition_grp_5: comfort_index
  competition_grp_6: repeat_founding_gap

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| HQ + B+C+D+E+F | B_evidence_support_rating | B_rubric_score | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | C_evidence_support_rating | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | C_latent_risk | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | C_latent_strength | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | D_evidence_support_rating | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | D_overall_founder_quality | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | E_evidence_support_rating | 1.000 |
| HQ + B+C+D+E+F | B_evidence_support_rating | E_hidden_risk | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| HQ + B+C+D+E+F | comfort_index | comfort_index | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + B+C+D+E+F | repeat_founding_gap | repeat_founding_gap | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + B+C+D+E+F | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_count | 0.580 | 0.014 | 0.000 | 0.579 |
| HQ + B+C+D+E+F | B_evidence_support_rating / B_rubric_score / C_evidence_support_rating / C_latent_risk / C_latent_strength / D_evidence_support_rating / D_overall_founder_quality / E_evidence_support_rating / E_hidden_risk / E_hidden_upside / F_career_coherence / F_evidence_support_rating / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | D_overall_founder_quality | 0.466 | 0.352 | 0.146 | 0.365 |
| HQ + B+C+D+E+F | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_x_serial | 0.518 | 0.034 | 0.000 | 0.543 |
| HQ + B+C+D+E+F | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | has_prior_acquisition | 0.506 | 0.032 | 0.000 | 0.525 |
| HQ + B+C+D+E+F | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | prestige_x_relevance | 0.504 | 0.019 | 0.000 | 0.495 |
| HQ + B+C+D+E+F | company_size_is_growing / founding_role_count / founding_timing / industry_alignment / industry_pivot_count / is_serial_founder / longest_founding_tenure / max_company_size_before_founding / max_seniority_reached / persistence_score / prestige_sacrifice_score / restlessness_score / sacrifice_x_serial / seniority_is_monotone / total_inferred_experience / years_in_large_company | prestige_sacrifice_score | 0.501 | 0.030 | 0.000 | 0.533 |
| HQ + B+C+D+E+F | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | edu_prestige_tier | 0.498 | 0.022 | 0.000 | 0.474 |
| HQ + B+C+D+E+F | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | best_degree_prestige | 0.498 | 0.022 | 0.000 | 0.474 |

## Competition | HQ + F | 6 total groups

| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.227 | 0.209 | -0.018 | 0.178 | 0.168 | 0.231 | 0.188 |
| xgb1 | 0.206 | 0.206 | 0.000 | 0.173 | 0.170 | 0.705 | 0.703 |

Representative cluster layout from `repeat_01_fold_01`:

- Scope `HQ + F`:
  competition_grp_1: company_size_is_growing, founding_role_count, founding_timing, industry_alignment, industry_pivot_count, is_serial_founder, longest_founding_tenure, max_company_size_before_founding, max_seniority_reached, persistence_score, prestige_sacrifice_score, restlessness_score, sacrifice_x_serial, seniority_is_monotone, total_inferred_experience, years_in_large_company
  competition_grp_2: exit_count, exit_x_serial, has_prior_acquisition, has_prior_ipo
  competition_grp_3: best_degree_prestige, degree_level, edu_prestige_tier, field_relevance_score, industry_prestige_penalty, prestige_x_relevance, stem_flag
  competition_grp_4: F_career_coherence, F_evidence_support_rating, F_ownership_signal, F_rubric_score, F_scrappiness, F_trajectory_strength
  competition_grp_5: comfort_index
  competition_grp_6: repeat_founding_gap

Top co-clustering pairs:

| Scope | Feature A | Feature B | Same-Cluster Frequency |
|---|---|---|---:|
| HQ + F | F_career_coherence | F_evidence_support_rating | 1.000 |
| HQ + F | F_career_coherence | F_ownership_signal | 1.000 |
| HQ + F | F_career_coherence | F_rubric_score | 1.000 |
| HQ + F | F_career_coherence | F_scrappiness | 1.000 |
| HQ + F | F_career_coherence | F_trajectory_strength | 1.000 |
| HQ + F | F_evidence_support_rating | F_ownership_signal | 1.000 |
| HQ + F | F_evidence_support_rating | F_rubric_score | 1.000 |
| HQ + F | F_evidence_support_rating | F_scrappiness | 1.000 |

Top component weights:

| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |
|---|---|---|---:|---:|---:|---:|
| HQ + F | comfort_index | comfort_index | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + F | repeat_founding_gap | repeat_founding_gap | 1.000 | 0.000 | 0.000 | 1.000 |
| HQ + F | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_count | 0.580 | 0.014 | 0.000 | 0.579 |
| HQ + F | F_career_coherence / F_evidence_support_rating / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_career_coherence | 0.195 | 0.511 | 0.417 | 0.264 |
| HQ + F | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | exit_x_serial | 0.518 | 0.034 | 0.000 | 0.543 |
| HQ + F | F_career_coherence / F_evidence_support_rating / F_ownership_signal / F_rubric_score / F_scrappiness / F_trajectory_strength | F_evidence_support_rating | 0.177 | 0.566 | 0.458 | 0.425 |
| HQ + F | exit_count / exit_x_serial / has_prior_acquisition / has_prior_ipo | has_prior_acquisition | 0.506 | 0.032 | 0.000 | 0.525 |
| HQ + F | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | prestige_x_relevance | 0.504 | 0.019 | 0.000 | 0.495 |
| HQ + F | company_size_is_growing / founding_role_count / founding_timing / industry_alignment / industry_pivot_count / is_serial_founder / longest_founding_tenure / max_company_size_before_founding / max_seniority_reached / persistence_score / prestige_sacrifice_score / restlessness_score / sacrifice_x_serial / seniority_is_monotone / total_inferred_experience / years_in_large_company | prestige_sacrifice_score | 0.501 | 0.030 | 0.000 | 0.533 |
| HQ + F | best_degree_prestige / degree_level / edu_prestige_tier / field_relevance_score / industry_prestige_penalty / prestige_x_relevance / stem_flag | edu_prestige_tier | 0.498 | 0.022 | 0.000 | 0.474 |
