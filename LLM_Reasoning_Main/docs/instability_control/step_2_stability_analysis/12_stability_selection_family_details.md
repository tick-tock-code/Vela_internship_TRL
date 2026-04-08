# Stability Selection Family Details

This note shows, for each family, how route performance changed and which reasoning features survived the sign-aware selector.

## HQ + A

- Mean reasoning selection frequency: 0.973
- Mean reasoning sign consistency: 0.889
- Minimum reasoning sign consistency: 0.660
- Mean sign-consistent reasoning features across outer splits: 3.222
- Mean admitted reasoning features after both thresholds: 3.222
- Mean subsample Jaccard: 0.962

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.310 | 0.030 | 0.297 | 0.034 | 34.000 | 29.000 | 5.000 |
| Competition | 0.302 | 0.031 | 0.294 | 0.028 | 30.333 | 27.111 | 3.222 |
| Augmentation | 0.292 | 0.040 | 0.294 | 0.030 | 32.222 | 29.000 | 3.222 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| A_evidence_support_rating | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | positive |
| A_scrappiness | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | negative |
| A_trajectory_strength | 0.969 | 0.138 | 0.831 | 0.859 | 0.556 | negative |
| A_ownership_signal | 0.960 | 0.284 | 0.676 | 0.735 | 0.111 | negative |
| A_career_coherence | 0.938 | 0.138 | 0.800 | 0.852 | 0.556 | negative |

Reasoning-feature LR coefficient stability:

| Feature | Raw Mean | Raw Std | Raw Flip | Comp Mean | Comp Std | Comp Flip | Aug Mean | Aug Std | Aug Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_career_coherence | -0.144 | 0.045 | 0.000 | -0.219 | 0.049 | 0.000 | -0.217 | 0.049 | 0.000 |
| A_evidence_support_rating | 1.119 | 0.104 | 0.000 | 1.089 | 0.104 | 0.000 | 1.089 | 0.104 | 0.000 |
| A_ownership_signal | -0.092 | 0.068 | 0.111 | -0.206 | 0.000 | 0.000 | -0.206 | 0.000 | 0.000 |
| A_scrappiness | -0.484 | 0.082 | 0.000 | -0.549 | 0.110 | 0.000 | -0.549 | 0.110 | 0.000 |
| A_trajectory_strength | -0.167 | 0.058 | 0.000 | -0.268 | 0.059 | 0.000 | -0.271 | 0.056 | 0.000 |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 1.000 |
| industry_prestige_penalty | 0.991 |
| max_seniority_reached | 0.991 |
| degree_level | 0.982 |
| restlessness_score | 0.982 |
| best_degree_prestige | 0.978 |
| edu_prestige_tier | 0.978 |
| years_in_large_company | 0.978 |

## HQ + B

- Mean reasoning selection frequency: 0.989
- Mean reasoning sign consistency: 0.957
- Minimum reasoning sign consistency: 0.936
- Mean sign-consistent reasoning features across outer splits: 1.778
- Mean admitted reasoning features after both thresholds: 1.778
- Mean subsample Jaccard: 0.978

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.207 | 0.033 | 0.209 | 0.037 | 31.000 | 29.000 | 2.000 |
| Competition | 0.207 | 0.040 | 0.199 | 0.039 | 29.111 | 27.333 | 1.778 |
| Augmentation | 0.207 | 0.040 | 0.205 | 0.040 | 30.778 | 29.000 | 1.778 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| B_evidence_support_rating | 0.991 | 0.969 | 0.022 | 0.977 | 1.000 | positive |
| B_rubric_score | 0.987 | 0.062 | 0.924 | 0.936 | 0.778 | negative |

Reasoning-feature LR coefficient stability:

| Feature | Raw Mean | Raw Std | Raw Flip | Comp Mean | Comp Std | Comp Flip | Aug Mean | Aug Std | Aug Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B_evidence_support_rating | 0.384 | 0.058 | 0.000 | 0.322 | 0.146 | 0.000 | 0.324 | 0.147 | 0.000 |
| B_rubric_score | -0.318 | 0.038 | 0.000 | -0.322 | 0.035 | 0.000 | -0.325 | 0.038 | 0.000 |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 1.000 |
| max_seniority_reached | 0.991 |
| restlessness_score | 0.991 |
| seniority_is_monotone | 0.987 |
| industry_prestige_penalty | 0.982 |
| degree_level | 0.978 |
| best_degree_prestige | 0.973 |
| comfort_index | 0.973 |

## HQ + C

- Mean reasoning selection frequency: 0.953
- Mean reasoning sign consistency: 0.852
- Minimum reasoning sign consistency: 0.664
- Mean sign-consistent reasoning features across outer splits: 1.667
- Mean admitted reasoning features after both thresholds: 1.667
- Mean subsample Jaccard: 0.914

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.215 | 0.047 | 0.202 | 0.041 | 32.000 | 29.000 | 3.000 |
| Competition | 0.223 | 0.049 | 0.214 | 0.043 | 29.000 | 27.333 | 1.667 |
| Augmentation | 0.222 | 0.053 | 0.201 | 0.041 | 30.667 | 29.000 | 1.667 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| C_latent_risk | 0.991 | 0.929 | 0.062 | 0.937 | 0.778 | positive |
| C_latent_strength | 0.969 | 0.880 | 0.089 | 0.904 | 0.778 | positive |
| C_evidence_support_rating | 0.898 | 0.573 | 0.324 | 0.716 | 0.111 | positive |

Reasoning-feature LR coefficient stability:

| Feature | Raw Mean | Raw Std | Raw Flip | Comp Mean | Comp Std | Comp Flip | Aug Mean | Aug Std | Aug Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C_evidence_support_rating | 0.047 | 0.105 | 0.444 | 0.247 | 0.000 | 0.000 | 0.242 | 0.000 | 0.000 |
| C_latent_risk | 0.302 | 0.102 | 0.000 | 0.306 | 0.142 | 0.143 | 0.307 | 0.143 | 0.143 |
| C_latent_strength | 0.285 | 0.120 | 0.000 | 0.301 | 0.103 | 0.000 | 0.302 | 0.102 | 0.000 |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 0.996 |
| industry_prestige_penalty | 0.991 |
| years_in_large_company | 0.991 |
| best_degree_prestige | 0.987 |
| degree_level | 0.987 |
| edu_prestige_tier | 0.987 |
| exit_x_serial | 0.987 |
| max_seniority_reached | 0.982 |

## HQ + D

- Mean reasoning selection frequency: 1.000
- Mean reasoning sign consistency: 1.000
- Minimum reasoning sign consistency: 1.000
- Mean sign-consistent reasoning features across outer splits: 2.000
- Mean admitted reasoning features after both thresholds: 2.000
- Mean subsample Jaccard: 1.000

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.213 | 0.039 | 0.206 | 0.041 | 31.000 | 29.000 | 2.000 |
| Competition | 0.214 | 0.039 | 0.210 | 0.043 | 29.667 | 27.667 | 2.000 |
| Augmentation | 0.213 | 0.039 | 0.206 | 0.041 | 31.000 | 29.000 | 2.000 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| D_evidence_support_rating | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | negative |
| D_overall_founder_quality | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | positive |

Reasoning-feature LR coefficient stability:

| Feature | Raw Mean | Raw Std | Raw Flip | Comp Mean | Comp Std | Comp Flip | Aug Mean | Aug Std | Aug Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D_evidence_support_rating | -0.620 | 0.098 | 0.000 | -0.620 | 0.098 | 0.000 | -0.620 | 0.098 | 0.000 |
| D_overall_founder_quality | 0.772 | 0.080 | 0.000 | 0.771 | 0.081 | 0.000 | 0.772 | 0.080 | 0.000 |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| max_seniority_reached | 1.000 |
| industry_alignment | 0.996 |
| repeat_founding_gap | 0.996 |
| comfort_index | 0.991 |
| industry_prestige_penalty | 0.991 |
| degree_level | 0.987 |
| best_degree_prestige | 0.982 |
| edu_prestige_tier | 0.982 |

## HQ + E

- Mean reasoning selection frequency: 0.957
- Mean reasoning sign consistency: 0.714
- Minimum reasoning sign consistency: 0.583
- Mean sign-consistent reasoning features across outer splits: 0.333
- Mean admitted reasoning features after both thresholds: 0.333
- Mean subsample Jaccard: 0.919

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.199 | 0.042 | 0.199 | 0.039 | 32.000 | 29.000 | 3.000 |
| Competition | 0.210 | 0.045 | 0.197 | 0.038 | 27.556 | 27.222 | 0.333 |
| Augmentation | 0.209 | 0.047 | 0.193 | 0.042 | 29.333 | 29.000 | 0.333 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| E_hidden_risk | 0.978 | 0.609 | 0.369 | 0.735 | 0.111 | positive |
| E_evidence_support_rating | 0.947 | 0.484 | 0.462 | 0.713 | 0.111 | positive |
| E_hidden_upside | 0.947 | 0.596 | 0.351 | 0.695 | 0.111 | positive |

Reasoning-feature LR coefficient stability:

| Feature | Raw Mean | Raw Std | Raw Flip | Comp Mean | Comp Std | Comp Flip | Aug Mean | Aug Std | Aug Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E_evidence_support_rating | 0.010 | 0.089 | 0.333 | -0.028 | 0.000 | 0.000 | -0.028 | 0.000 | 0.000 |
| E_hidden_risk | 0.042 | 0.066 | 0.222 | 0.022 | 0.000 | 0.000 | 0.022 | 0.000 | 0.000 |
| E_hidden_upside | 0.046 | 0.080 | 0.333 | 0.044 | 0.000 | 0.000 | 0.045 | 0.000 | 0.000 |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 1.000 |
| years_in_large_company | 0.996 |
| comfort_index | 0.991 |
| industry_prestige_penalty | 0.991 |
| max_seniority_reached | 0.982 |
| repeat_founding_gap | 0.982 |
| best_degree_prestige | 0.973 |
| edu_prestige_tier | 0.973 |

## HQ + F

- Mean reasoning selection frequency: 0.898
- Mean reasoning sign consistency: 0.869
- Minimum reasoning sign consistency: 0.616
- Mean sign-consistent reasoning features across outer splits: 3.333
- Mean admitted reasoning features after both thresholds: 3.222
- Mean subsample Jaccard: 0.831

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.230 | 0.046 | 0.202 | 0.042 | 35.000 | 29.000 | 6.000 |
| Competition | 0.239 | 0.036 | 0.199 | 0.043 | 30.556 | 27.333 | 3.222 |
| Augmentation | 0.230 | 0.046 | 0.197 | 0.041 | 32.222 | 29.000 | 3.222 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| F_career_coherence | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | negative |
| F_evidence_support_rating | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | positive |
| F_trajectory_strength | 0.876 | 0.813 | 0.062 | 0.912 | 0.667 | positive |
| F_scrappiness | 0.862 | 0.391 | 0.471 | 0.632 | 0.000 | positive |
| F_ownership_signal | 0.858 | 0.720 | 0.138 | 0.837 | 0.333 | positive |
| F_rubric_score | 0.791 | 0.151 | 0.640 | 0.834 | 0.333 | negative |

Reasoning-feature LR coefficient stability:

| Feature | Raw Mean | Raw Std | Raw Flip | Comp Mean | Comp Std | Comp Flip | Aug Mean | Aug Std | Aug Flip |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F_career_coherence | -1.097 | 0.172 | 0.000 | -1.022 | 0.293 | 0.000 | -1.024 | 0.296 | 0.000 |
| F_evidence_support_rating | 0.539 | 0.082 | 0.000 | 0.551 | 0.070 | 0.000 | 0.550 | 0.069 | 0.000 |
| F_ownership_signal | 0.259 | 0.165 | 0.000 | 0.418 | 0.127 | 0.000 | 0.421 | 0.121 | 0.000 |
| F_rubric_score | -0.236 | 0.226 | 0.111 | -0.530 | 0.039 | 0.000 | -0.544 | 0.058 | 0.000 |
| F_scrappiness | -0.022 | 0.040 | 0.333 | -- | -- | -- | -- | -- | -- |
| F_trajectory_strength | 0.566 | 0.357 | 0.000 | 0.690 | 0.302 | 0.000 | 0.698 | 0.307 | 0.000 |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 1.000 |
| max_seniority_reached | 0.991 |
| best_degree_prestige | 0.987 |
| edu_prestige_tier | 0.987 |
| industry_prestige_penalty | 0.987 |
| restlessness_score | 0.982 |
| years_in_large_company | 0.982 |
| exit_x_serial | 0.978 |
