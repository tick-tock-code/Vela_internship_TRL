# Stability Selection Family Details

This note shows, for each family, how route performance changed and which reasoning features survived the sign-aware selector.

## HQ + A

- Mean reasoning selection frequency: 0.973
- Mean reasoning sign consistency: 0.896
- Minimum reasoning sign consistency: 0.684
- Mean sign-consistent reasoning features across outer splits: 3.067
- Mean admitted reasoning features after both thresholds: 3.067
- Mean subsample Jaccard: 0.961

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.308 | 0.026 | 0.296 | 0.029 | 34.000 | 29.000 | 5.000 |
| Competition | 0.300 | 0.029 | 0.292 | 0.031 | 30.600 | 27.533 | 3.067 |
| Augmentation | 0.304 | 0.029 | 0.293 | 0.027 | 32.067 | 29.000 | 3.067 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| A_evidence_support_rating | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | positive |
| A_scrappiness | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | negative |
| A_ownership_signal | 0.966 | 0.248 | 0.718 | 0.777 | 0.267 | negative |
| A_career_coherence | 0.953 | 0.125 | 0.828 | 0.867 | 0.400 | negative |
| A_trajectory_strength | 0.949 | 0.151 | 0.797 | 0.836 | 0.400 | negative |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 0.999 |
| industry_prestige_penalty | 0.994 |
| max_seniority_reached | 0.994 |
| comfort_index | 0.991 |
| degree_level | 0.987 |
| best_degree_prestige | 0.985 |
| edu_prestige_tier | 0.985 |
| repeat_founding_gap | 0.975 |

## HQ + F

- Mean reasoning selection frequency: 0.894
- Mean reasoning sign consistency: 0.855
- Minimum reasoning sign consistency: 0.610
- Mean sign-consistent reasoning features across outer splits: 3.267
- Mean admitted reasoning features after both thresholds: 3.200
- Mean subsample Jaccard: 0.830

| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 0.229 | 0.042 | 0.201 | 0.037 | 35.000 | 29.000 | 6.000 |
| Competition | 0.233 | 0.044 | 0.204 | 0.040 | 30.733 | 27.533 | 3.200 |
| Augmentation | 0.233 | 0.041 | 0.203 | 0.040 | 32.200 | 29.000 | 3.200 |

Top reasoning features:

| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |
|---|---:|---:|---:|---:|---:|---|
| F_evidence_support_rating | 1.000 | 0.999 | 0.001 | 0.999 | 1.000 | positive |
| F_career_coherence | 0.997 | 0.001 | 0.996 | 0.999 | 1.000 | negative |
| F_trajectory_strength | 0.909 | 0.843 | 0.066 | 0.916 | 0.733 | positive |
| F_scrappiness | 0.854 | 0.376 | 0.478 | 0.632 | 0.000 | negative |
| F_ownership_signal | 0.839 | 0.678 | 0.161 | 0.804 | 0.267 | positive |
| F_rubric_score | 0.763 | 0.223 | 0.540 | 0.781 | 0.267 | negative |

Top HQ features in the competition-track selector view:

| Feature | Mean Selection Frequency |
|---|---:|
| industry_alignment | 1.000 |
| industry_prestige_penalty | 0.989 |
| max_seniority_reached | 0.988 |
| comfort_index | 0.985 |
| degree_level | 0.983 |
| years_in_large_company | 0.977 |
| repeat_founding_gap | 0.976 |
| best_degree_prestige | 0.975 |
