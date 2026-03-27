# Model Testing Report
Generated: 2026-03-26T03:38:56.698714+00:00Z

Model variants: logistic, xgb1 (stump), xgb3 (depth=3), mlp32 (1 hidden layer).

## HQ Mirror + Reasoning (rule layer)
| Combo | LOGISTIC | XGB1 | XGB3 | MLP32 |
|---|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.233+/-0.027 | 0.229+/-0.035 | 0.210+/-0.042 |
| A | 0.275+/-0.024 | 0.272+/-0.042 | 0.282+/-0.049 | 0.240+/-0.058 |
| B | 0.246+/-0.050 | 0.227+/-0.023 | 0.238+/-0.055 | 0.211+/-0.063 |
| D | 0.249+/-0.021 | 0.235+/-0.029 | 0.249+/-0.047 | 0.227+/-0.069 |
| A+E | 0.281+/-0.042 | 0.280+/-0.055 | 0.294+/-0.061 | 0.233+/-0.033 |
| A+F | 0.308+/-0.056 | 0.295+/-0.030 | 0.327+/-0.049 | 0.264+/-0.046 |
| A+D+E+F | 0.339+/-0.043 | 0.301+/-0.042 | 0.327+/-0.050 | 0.281+/-0.050 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.300+/-0.040 | 0.343+/-0.064 | 0.296+/-0.044 |

## CV vs Full-Train (per combo, same model)
### HQ Mirror + Reasoning
| Combo | LOGISTIC CV | LOGISTIC Full | XGB1 CV | XGB1 Full | XGB3 CV | XGB3 Full | MLP32 CV | MLP32 Full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HQ | 0.234+/-0.041 | 0.239 | 0.233+/-0.027 | 0.235 | 0.229+/-0.035 | 0.331 | 0.210+/-0.042 | 0.324 |
| A | 0.275+/-0.024 | 0.278 | 0.272+/-0.042 | 0.277 | 0.282+/-0.049 | 0.402 | 0.240+/-0.058 | 0.308 |
| B | 0.246+/-0.050 | 0.248 | 0.227+/-0.023 | 0.239 | 0.238+/-0.055 | 0.352 | 0.211+/-0.063 | 0.276 |
| D | 0.249+/-0.021 | 0.263 | 0.235+/-0.029 | 0.243 | 0.249+/-0.047 | 0.365 | 0.227+/-0.069 | 0.319 |
| A+E | 0.281+/-0.042 | 0.291 | 0.280+/-0.055 | 0.285 | 0.294+/-0.061 | 0.408 | 0.233+/-0.033 | 0.370 |
| A+F | 0.308+/-0.056 | 0.323 | 0.295+/-0.030 | 0.299 | 0.327+/-0.049 | 0.428 | 0.264+/-0.046 | 0.356 |
| A+D+E+F | 0.339+/-0.043 | 0.337 | 0.301+/-0.042 | 0.313 | 0.327+/-0.050 | 0.456 | 0.281+/-0.050 | 0.432 |
| A+B+C+D+E+F | 0.329+/-0.054 | 0.339 | 0.300+/-0.040 | 0.306 | 0.343+/-0.064 | 0.457 | 0.296+/-0.044 | 0.489 |

## Interpretability (HQ Mirror only)

### HQ
**LOGISTIC — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| edu_prestige_tier | 0.0257 | 0.0241 |
| years_in_large_company | 0.0202 | 0.0155 |
| industry_alignment | 0.0165 | 0.0199 |
| sacrifice_x_serial | 0.0150 | 0.0151 |
| best_degree_prestige | 0.0144 | 0.0247 |
| prestige_sacrifice_score | 0.0135 | 0.0134 |
| degree_level | 0.0134 | 0.0127 |
| industry_prestige_penalty | 0.0125 | 0.0280 |
| industry_pivot_count | 0.0108 | 0.0116 |
| total_inferred_experience | 0.0097 | 0.0191 |
**LOGISTIC — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| total_inferred_experience | 0.2503 | 0.0868 |
| industry_alignment | 0.1924 | 0.0311 |
| max_seniority_reached | 0.1859 | 0.0170 |
| edu_prestige_tier | 0.1350 | 0.0205 |
| best_degree_prestige | 0.1350 | 0.0205 |
| is_serial_founder | 0.1222 | 0.0492 |
| sacrifice_x_serial | 0.1188 | 0.0432 |
| years_in_large_company | 0.1003 | 0.0234 |
| prestige_sacrifice_score | 0.0795 | 0.0525 |
| degree_level | 0.0793 | 0.0234 |
**XGB1 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| edu_prestige_tier | 0.0195 | 0.0235 |
| best_degree_prestige | 0.0182 | 0.0119 |
| years_in_large_company | 0.0123 | 0.0138 |
| total_inferred_experience | 0.0113 | 0.0182 |
| industry_alignment | 0.0101 | 0.0141 |
| field_relevance_score | 0.0085 | 0.0117 |
| degree_level | 0.0071 | 0.0069 |
| prestige_x_relevance | 0.0061 | 0.0152 |
| restlessness_score | 0.0055 | 0.0025 |
| longest_founding_tenure | 0.0052 | 0.0044 |
**XGB1 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| edu_prestige_tier | 0.1558 | 0.0175 |
| industry_alignment | 0.1511 | 0.0334 |
| total_inferred_experience | 0.1500 | 0.0389 |
| prestige_sacrifice_score | 0.1285 | 0.0237 |
| prestige_x_relevance | 0.0921 | 0.0224 |
| years_in_large_company | 0.0726 | 0.0295 |
| max_seniority_reached | 0.0710 | 0.0129 |
| field_relevance_score | 0.0631 | 0.0262 |
| best_degree_prestige | 0.0620 | 0.0150 |
| exit_x_serial | 0.0560 | 0.0120 |
**XGB3 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| edu_prestige_tier | 0.0263 | 0.0157 |
| industry_alignment | 0.0215 | 0.0039 |
| best_degree_prestige | 0.0144 | 0.0085 |
| industry_prestige_penalty | 0.0140 | 0.0127 |
| prestige_sacrifice_score | 0.0136 | 0.0196 |
| years_in_large_company | 0.0135 | 0.0222 |
| degree_level | 0.0123 | 0.0127 |
| prestige_x_relevance | 0.0098 | 0.0096 |
| total_inferred_experience | 0.0061 | 0.0348 |
| is_serial_founder | 0.0057 | 0.0048 |
**XGB3 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| total_inferred_experience | 0.1782 | 0.0351 |
| edu_prestige_tier | 0.1775 | 0.0145 |
| industry_alignment | 0.1676 | 0.0268 |
| prestige_sacrifice_score | 0.1517 | 0.0134 |
| max_seniority_reached | 0.1379 | 0.0231 |
| prestige_x_relevance | 0.1186 | 0.0169 |
| years_in_large_company | 0.0916 | 0.0259 |
| founding_timing | 0.0873 | 0.0125 |
| field_relevance_score | 0.0818 | 0.0231 |
| max_company_size_before_founding | 0.0761 | 0.0172 |
**MLP32 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| prestige_sacrifice_score | 0.0480 | 0.0403 |
| total_inferred_experience | 0.0421 | 0.0309 |
| prestige_x_relevance | 0.0285 | 0.0353 |
| founding_timing | 0.0252 | 0.0353 |
| sacrifice_x_serial | 0.0224 | 0.0336 |
| best_degree_prestige | 0.0205 | 0.0231 |
| industry_alignment | 0.0203 | 0.0156 |
| restlessness_score | 0.0200 | 0.0233 |
| founding_role_count | 0.0183 | 0.0102 |
| industry_prestige_penalty | 0.0182 | 0.0194 |
**MLP32 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| prestige_sacrifice_score | 0.0270 | 0.0115 |
| founding_timing | 0.0184 | 0.0040 |
| best_degree_prestige | 0.0151 | 0.0020 |
| prestige_x_relevance | 0.0145 | 0.0045 |
| total_inferred_experience | 0.0130 | 0.0017 |
| years_in_large_company | 0.0106 | 0.0017 |
| industry_alignment | 0.0102 | 0.0039 |
| sacrifice_x_serial | 0.0098 | 0.0048 |
| max_company_size_before_founding | 0.0096 | 0.0030 |
| field_relevance_score | 0.0089 | 0.0020 |

### D
**LOGISTIC — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| D_evidence_support_rating | 0.1015 | 0.0120 |
| D_overall_founder_quality | 0.0992 | 0.0268 |
| best_degree_prestige | 0.0312 | 0.0235 |
| edu_prestige_tier | 0.0268 | 0.0218 |
| industry_alignment | 0.0210 | 0.0074 |
| total_inferred_experience | 0.0202 | 0.0129 |
| max_seniority_reached | 0.0124 | 0.0235 |
| prestige_x_relevance | 0.0113 | 0.0204 |
| exit_count | 0.0071 | 0.0091 |
| sacrifice_x_serial | 0.0064 | 0.0083 |
**LOGISTIC — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| D_evidence_support_rating | 0.9657 | 0.1118 |
| D_overall_founder_quality | 0.8059 | 0.0947 |
| industry_alignment | 0.2097 | 0.0318 |
| max_seniority_reached | 0.1952 | 0.0155 |
| best_degree_prestige | 0.1851 | 0.0231 |
| edu_prestige_tier | 0.1851 | 0.0231 |
| total_inferred_experience | 0.1850 | 0.0859 |
| max_company_size_before_founding | 0.1089 | 0.0196 |
| prestige_sacrifice_score | 0.1007 | 0.0458 |
| is_serial_founder | 0.0960 | 0.0504 |
**XGB1 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| D_evidence_support_rating | 0.0176 | 0.0078 |
| edu_prestige_tier | 0.0168 | 0.0373 |
| industry_alignment | 0.0108 | 0.0047 |
| comfort_index | 0.0107 | 0.0139 |
| industry_prestige_penalty | 0.0079 | 0.0133 |
| total_inferred_experience | 0.0077 | 0.0124 |
| years_in_large_company | 0.0076 | 0.0257 |
| field_relevance_score | 0.0056 | 0.0154 |
| exit_x_serial | 0.0051 | 0.0020 |
| max_company_size_before_founding | 0.0049 | 0.0190 |
**XGB1 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| edu_prestige_tier | 0.1489 | 0.0109 |
| industry_alignment | 0.1477 | 0.0280 |
| prestige_sacrifice_score | 0.1362 | 0.0180 |
| prestige_x_relevance | 0.1303 | 0.0256 |
| total_inferred_experience | 0.1216 | 0.0329 |
| D_evidence_support_rating | 0.1150 | 0.0198 |
| max_seniority_reached | 0.0724 | 0.0110 |
| years_in_large_company | 0.0663 | 0.0238 |
| max_company_size_before_founding | 0.0650 | 0.0243 |
| exit_x_serial | 0.0543 | 0.0165 |
**XGB3 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| D_evidence_support_rating | 0.0479 | 0.0255 |
| industry_alignment | 0.0247 | 0.0112 |
| edu_prestige_tier | 0.0206 | 0.0203 |
| prestige_x_relevance | 0.0135 | 0.0247 |
| restlessness_score | 0.0135 | 0.0135 |
| persistence_score | 0.0115 | 0.0185 |
| degree_level | 0.0115 | 0.0093 |
| founding_timing | 0.0106 | 0.0198 |
| industry_prestige_penalty | 0.0100 | 0.0103 |
| total_inferred_experience | 0.0063 | 0.0202 |
**XGB3 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| D_evidence_support_rating | 0.2066 | 0.0291 |
| prestige_x_relevance | 0.1755 | 0.0232 |
| prestige_sacrifice_score | 0.1684 | 0.0091 |
| industry_alignment | 0.1649 | 0.0231 |
| total_inferred_experience | 0.1623 | 0.0411 |
| edu_prestige_tier | 0.1560 | 0.0102 |
| max_seniority_reached | 0.1392 | 0.0241 |
| D_overall_founder_quality | 0.0996 | 0.0260 |
| years_in_large_company | 0.0899 | 0.0305 |
| max_company_size_before_founding | 0.0897 | 0.0258 |
**MLP32 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| prestige_sacrifice_score | 0.0892 | 0.0563 |
| sacrifice_x_serial | 0.0598 | 0.0326 |
| D_evidence_support_rating | 0.0329 | 0.0307 |
| D_overall_founder_quality | 0.0327 | 0.0308 |
| max_company_size_before_founding | 0.0311 | 0.0241 |
| founding_timing | 0.0289 | 0.0437 |
| best_degree_prestige | 0.0287 | 0.0231 |
| years_in_large_company | 0.0275 | 0.0203 |
| total_inferred_experience | 0.0264 | 0.0341 |
| prestige_x_relevance | 0.0165 | 0.0270 |
**MLP32 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| prestige_sacrifice_score | 0.0538 | 0.0146 |
| founding_timing | 0.0400 | 0.0066 |
| D_evidence_support_rating | 0.0377 | 0.0080 |
| max_company_size_before_founding | 0.0328 | 0.0068 |
| D_overall_founder_quality | 0.0200 | 0.0102 |
| sacrifice_x_serial | 0.0200 | 0.0109 |
| best_degree_prestige | 0.0172 | 0.0065 |
| years_in_large_company | 0.0162 | 0.0040 |
| total_inferred_experience | 0.0141 | 0.0042 |
| prestige_x_relevance | 0.0129 | 0.0050 |

### A+B+C+D+E+F
**LOGISTIC — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| D_evidence_support_rating | 0.1400 | 0.0486 |
| D_overall_founder_quality | 0.1328 | 0.0550 |
| F_evidence_support_rating | 0.1116 | 0.0467 |
| F_rubric_score | 0.0714 | 0.0393 |
| A_scrappiness | 0.0413 | 0.0361 |
| best_degree_prestige | 0.0374 | 0.0300 |
| edu_prestige_tier | 0.0347 | 0.0272 |
| B_evidence_support_rating | 0.0295 | 0.0167 |
| C_latent_risk | 0.0280 | 0.0277 |
| E_evidence_support_rating | 0.0265 | 0.0327 |
**LOGISTIC — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| D_overall_founder_quality | 0.9793 | 0.1044 |
| D_evidence_support_rating | 0.9768 | 0.1093 |
| F_evidence_support_rating | 0.7311 | 0.0384 |
| F_rubric_score | 0.5218 | 0.1087 |
| A_scrappiness | 0.3337 | 0.0420 |
| C_latent_risk | 0.2871 | 0.0240 |
| max_seniority_reached | 0.2633 | 0.0126 |
| A_career_coherence | 0.2621 | 0.0182 |
| industry_alignment | 0.2238 | 0.0377 |
| best_degree_prestige | 0.1996 | 0.0215 |
**XGB1 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| A_scrappiness | 0.0305 | 0.0130 |
| best_degree_prestige | 0.0254 | 0.0194 |
| edu_prestige_tier | 0.0242 | 0.0229 |
| A_ownership_signal | 0.0201 | 0.0127 |
| F_scrappiness | 0.0194 | 0.0119 |
| A_career_coherence | 0.0192 | 0.0225 |
| F_evidence_support_rating | 0.0160 | 0.0060 |
| F_career_coherence | 0.0138 | 0.0091 |
| field_relevance_score | 0.0097 | 0.0144 |
| industry_alignment | 0.0094 | 0.0057 |
**XGB1 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| A_scrappiness | 0.1999 | 0.0317 |
| edu_prestige_tier | 0.1350 | 0.0205 |
| prestige_sacrifice_score | 0.1339 | 0.0213 |
| industry_alignment | 0.1331 | 0.0360 |
| prestige_x_relevance | 0.1272 | 0.0338 |
| max_seniority_reached | 0.0825 | 0.0142 |
| A_ownership_signal | 0.0765 | 0.0119 |
| best_degree_prestige | 0.0725 | 0.0125 |
| F_career_coherence | 0.0725 | 0.0317 |
| A_career_coherence | 0.0673 | 0.0190 |
**XGB3 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| edu_prestige_tier | 0.0347 | 0.0386 |
| F_evidence_support_rating | 0.0329 | 0.0223 |
| A_career_coherence | 0.0303 | 0.0222 |
| F_career_coherence | 0.0288 | 0.0276 |
| prestige_sacrifice_score | 0.0282 | 0.0220 |
| A_scrappiness | 0.0281 | 0.0125 |
| C_latent_risk | 0.0279 | 0.0139 |
| A_ownership_signal | 0.0230 | 0.0098 |
| total_inferred_experience | 0.0162 | 0.0222 |
| D_evidence_support_rating | 0.0161 | 0.0059 |
**XGB3 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| A_scrappiness | 0.2226 | 0.0479 |
| prestige_sacrifice_score | 0.1633 | 0.0142 |
| industry_alignment | 0.1600 | 0.0367 |
| max_seniority_reached | 0.1599 | 0.0160 |
| edu_prestige_tier | 0.1541 | 0.0111 |
| F_evidence_support_rating | 0.1439 | 0.0236 |
| prestige_x_relevance | 0.1418 | 0.0305 |
| total_inferred_experience | 0.1228 | 0.0321 |
| F_career_coherence | 0.1227 | 0.0217 |
| F_scrappiness | 0.1175 | 0.0214 |
**MLP32 — Permutation Importance (Top 10)**
| Feature | Mean drop | Std |
|---|---:|---:|
| prestige_x_relevance | 0.0572 | 0.0242 |
| F_evidence_support_rating | 0.0469 | 0.0230 |
| D_evidence_support_rating | 0.0429 | 0.0219 |
| prestige_sacrifice_score | 0.0426 | 0.0305 |
| sacrifice_x_serial | 0.0364 | 0.0344 |
| D_overall_founder_quality | 0.0319 | 0.0337 |
| A_scrappiness | 0.0291 | 0.0151 |
| max_company_size_before_founding | 0.0233 | 0.0124 |
| total_inferred_experience | 0.0232 | 0.0216 |
| years_in_large_company | 0.0221 | 0.0348 |
**MLP32 — SHAP (Mean |value|, Top 10)**
| Feature | Mean | Std |
|---|---:|---:|
| F_evidence_support_rating | 0.0305 | 0.0075 |
| D_evidence_support_rating | 0.0236 | 0.0061 |
| A_scrappiness | 0.0185 | 0.0036 |
| D_overall_founder_quality | 0.0179 | 0.0051 |
| prestige_sacrifice_score | 0.0109 | 0.0045 |
| A_career_coherence | 0.0106 | 0.0013 |
| prestige_x_relevance | 0.0094 | 0.0045 |
| C_evidence_support_rating | 0.0088 | 0.0023 |
| edu_prestige_tier | 0.0079 | 0.0037 |
| industry_alignment | 0.0078 | 0.0039 |
