# Reasoning-Block PLS Family Details

This note shows raw versus blockwise-PLS route performance and the component diagnostics for each reasoning family.

## HQ + A

- Original reasoning feature count: 5
- Best blockwise `PLS` k by LR CV: 3
- Best blockwise `PLS` k by XGB1 CV: 3

| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | -- | 0.302 | 0.035 | 0.298 | 0.033 | 0.226 | 0.269 | 0.034 |
| Block PLS | 2 | 0.293 | 0.037 | 0.294 | 0.032 | 0.220 | 0.271 | 0.041 |
| Block PLS | 3 | 0.299 | 0.038 | 0.297 | 0.035 | 0.226 | 0.275 | 0.036 |

Component diagnostics for k = 2:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| A_pls_1 | A_evidence_support_rating | 0.839 | 0.033 | 0.000 | 0.929 | 0.119 | 0.000 |
| A_pls_1 | A_scrappiness | -0.307 | 0.082 | 0.000 | 0.297 | 0.251 | 0.104 |
| A_pls_1 | A_career_coherence | 0.302 | 0.065 | 0.000 | 0.801 | 0.136 | 0.000 |
| A_pls_1 | A_ownership_signal | -0.254 | 0.090 | 0.000 | 0.291 | 0.260 | 0.104 |
| A_pls_1 | A_trajectory_strength | 0.140 | 0.077 | 0.083 | 0.673 | 0.179 | 0.021 |
| A_pls_2 | A_scrappiness | 0.525 | 0.122 | 0.021 | 0.563 | 0.177 | 0.021 |
| A_pls_2 | A_ownership_signal | 0.470 | 0.156 | 0.021 | 0.570 | 0.183 | 0.021 |
| A_pls_2 | A_trajectory_strength | 0.471 | 0.057 | 0.000 | 0.429 | 0.074 | 0.000 |
| A_pls_2 | A_career_coherence | 0.443 | 0.120 | 0.021 | 0.319 | 0.080 | 0.000 |
| A_pls_2 | A_evidence_support_rating | 0.087 | 0.143 | 0.188 | 0.184 | 0.184 | 0.062 |

Component diagnostics for k = 3:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| A_pls_1 | A_evidence_support_rating | 0.839 | 0.033 | 0.000 | 0.929 | 0.119 | 0.000 |
| A_pls_1 | A_scrappiness | -0.307 | 0.082 | 0.000 | 0.297 | 0.251 | 0.104 |
| A_pls_1 | A_career_coherence | 0.302 | 0.065 | 0.000 | 0.801 | 0.136 | 0.000 |
| A_pls_1 | A_ownership_signal | -0.254 | 0.090 | 0.000 | 0.291 | 0.260 | 0.104 |
| A_pls_1 | A_trajectory_strength | 0.140 | 0.077 | 0.083 | 0.673 | 0.179 | 0.021 |
| A_pls_2 | A_scrappiness | 0.525 | 0.122 | 0.021 | 0.563 | 0.177 | 0.021 |
| A_pls_2 | A_ownership_signal | 0.470 | 0.156 | 0.021 | 0.570 | 0.183 | 0.021 |
| A_pls_2 | A_trajectory_strength | 0.471 | 0.057 | 0.000 | 0.429 | 0.074 | 0.000 |
| A_pls_2 | A_career_coherence | 0.443 | 0.120 | 0.021 | 0.319 | 0.080 | 0.000 |
| A_pls_2 | A_evidence_support_rating | 0.087 | 0.143 | 0.188 | 0.184 | 0.184 | 0.062 |
| A_pls_3 | A_career_coherence | 0.540 | 0.352 | 0.104 | 0.527 | 0.348 | 0.104 |
| A_pls_3 | A_evidence_support_rating | -0.353 | 0.340 | 0.125 | -0.373 | 0.354 | 0.125 |
| A_pls_3 | A_ownership_signal | -0.340 | 0.341 | 0.125 | -0.271 | 0.296 | 0.146 |
| A_pls_3 | A_trajectory_strength | 0.195 | 0.148 | 0.083 | 0.254 | 0.169 | 0.104 |
| A_pls_3 | A_scrappiness | -0.125 | 0.193 | 0.146 | -0.222 | 0.249 | 0.125 |

## HQ + B

- Original reasoning feature count: 2
- Best blockwise `PLS` k by LR CV: 1
- Best blockwise `PLS` k by XGB1 CV: 2

| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | -- | 0.206 | 0.044 | 0.202 | 0.037 | 0.168 | 0.245 | 0.059 |
| Block PLS | 1 | 0.203 | 0.045 | 0.201 | 0.036 | 0.167 | 0.249 | 0.075 |
| Block PLS | 2 | 0.202 | 0.044 | 0.202 | 0.035 | 0.168 | 0.241 | 0.056 |

Component diagnostics for k = 1:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| B_pls_1 | B_evidence_support_rating | 0.758 | 0.396 | 0.104 | 0.968 | 0.325 | 0.000 |
| B_pls_1 | B_rubric_score | 0.339 | 0.392 | 0.208 | 0.947 | 0.342 | 0.000 |

Component diagnostics for k = 2:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| B_pls_1 | B_evidence_support_rating | 0.758 | 0.396 | 0.104 | 0.968 | 0.325 | 0.000 |
| B_pls_1 | B_rubric_score | 0.339 | 0.392 | 0.208 | 0.947 | 0.342 | 0.000 |
| B_pls_2 | B_rubric_score | 0.802 | 0.298 | 0.042 | 0.802 | 0.298 | 0.042 |
| B_pls_2 | B_evidence_support_rating | -0.065 | 0.514 | 0.354 | -0.065 | 0.514 | 0.354 |

## HQ + C

- Original reasoning feature count: 3
- Best blockwise `PLS` k by LR CV: 2
- Best blockwise `PLS` k by XGB1 CV: 2

| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | -- | 0.213 | 0.045 | 0.208 | 0.038 | 0.170 | 0.240 | 0.048 |
| Block PLS | 2 | 0.213 | 0.045 | 0.215 | 0.045 | 0.170 | 0.239 | 0.054 |

Component diagnostics for k = 2:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| C_pls_1 | C_latent_strength | 0.566 | 0.363 | 0.125 | 0.308 | 0.828 | 0.354 |
| C_pls_1 | C_latent_risk | 0.414 | 0.422 | 0.208 | -0.042 | 0.896 | 0.354 |
| C_pls_1 | C_evidence_support_rating | 0.259 | 0.362 | 0.271 | 0.261 | 0.842 | 0.354 |
| C_pls_2 | C_latent_risk | 0.566 | 0.498 | 0.208 | 0.599 | 0.503 | 0.208 |
| C_pls_2 | C_latent_strength | 0.276 | 0.417 | 0.271 | 0.226 | 0.414 | 0.292 |
| C_pls_2 | C_evidence_support_rating | -0.016 | 0.427 | 0.375 | 0.067 | 0.419 | 0.417 |

## HQ + D

- Original reasoning feature count: 2
- Best blockwise `PLS` k by LR CV: 2
- Best blockwise `PLS` k by XGB1 CV: 2

| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | -- | 0.218 | 0.036 | 0.216 | 0.033 | 0.175 | 0.234 | 0.044 |
| Block PLS | 1 | 0.212 | 0.046 | 0.215 | 0.037 | 0.170 | 0.249 | 0.057 |
| Block PLS | 2 | 0.218 | 0.039 | 0.226 | 0.034 | 0.175 | 0.234 | 0.045 |

Component diagnostics for k = 1:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| D_pls_1 | D_overall_founder_quality | 0.929 | 0.048 | 0.000 | 0.818 | 0.114 | 0.000 |
| D_pls_1 | D_evidence_support_rating | 0.322 | 0.175 | 0.042 | 0.775 | 0.081 | 0.000 |

Component diagnostics for k = 2:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| D_pls_1 | D_overall_founder_quality | 0.929 | 0.048 | 0.000 | 0.818 | 0.114 | 0.000 |
| D_pls_1 | D_evidence_support_rating | 0.322 | 0.175 | 0.042 | 0.775 | 0.081 | 0.000 |
| D_pls_2 | D_evidence_support_rating | 0.929 | 0.048 | 0.000 | 0.929 | 0.048 | 0.000 |
| D_pls_2 | D_overall_founder_quality | -0.322 | 0.175 | 0.042 | -0.322 | 0.175 | 0.042 |

## HQ + E

- Original reasoning feature count: 3
- Best blockwise `PLS` k by LR CV: 1
- Best blockwise `PLS` k by XGB1 CV: 1

| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | -- | 0.204 | 0.043 | 0.205 | 0.035 | 0.165 | 0.258 | 0.078 |
| Block PLS | 1 | 0.205 | 0.045 | 0.200 | 0.035 | 0.166 | 0.252 | 0.073 |

Component diagnostics for k = 1:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| E_pls_1 | E_hidden_upside | 0.340 | 0.470 | 0.250 | 0.294 | 0.612 | 0.292 |
| E_pls_1 | E_evidence_support_rating | 0.329 | 0.472 | 0.250 | 0.291 | 0.618 | 0.292 |
| E_pls_1 | E_hidden_risk | 0.111 | 0.565 | 0.458 | -0.120 | 0.601 | 0.333 |

## HQ + F

- Original reasoning feature count: 6
- Best blockwise `PLS` k by LR CV: 2
- Best blockwise `PLS` k by XGB1 CV: 3

| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw | -- | 0.227 | 0.043 | 0.206 | 0.036 | 0.178 | 0.231 | 0.055 |
| Block PLS | 2 | 0.227 | 0.038 | 0.235 | 0.038 | 0.176 | 0.231 | 0.055 |
| Block PLS | 3 | 0.224 | 0.044 | 0.241 | 0.032 | 0.177 | 0.240 | 0.058 |

Component diagnostics for k = 2:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| F_pls_1 | F_career_coherence | 0.195 | 0.511 | 0.417 | 0.264 | 0.714 | 0.229 |
| F_pls_1 | F_evidence_support_rating | 0.177 | 0.566 | 0.458 | 0.425 | 0.590 | 0.167 |
| F_pls_1 | F_rubric_score | 0.181 | 0.286 | 0.333 | 0.269 | 0.717 | 0.229 |
| F_pls_1 | F_trajectory_strength | 0.178 | 0.263 | 0.333 | 0.266 | 0.718 | 0.229 |
| F_pls_1 | F_scrappiness | 0.196 | 0.175 | 0.208 | 0.285 | 0.677 | 0.208 |
| F_pls_2 | F_evidence_support_rating | 0.629 | 0.345 | 0.083 | 0.683 | 0.406 | 0.083 |
| F_pls_2 | F_ownership_signal | 0.320 | 0.085 | 0.000 | 0.266 | 0.138 | 0.000 |
| F_pls_2 | F_scrappiness | 0.295 | 0.089 | 0.000 | 0.287 | 0.103 | 0.000 |
| F_pls_2 | F_trajectory_strength | 0.268 | 0.126 | 0.021 | 0.237 | 0.171 | 0.062 |
| F_pls_2 | F_rubric_score | 0.258 | 0.136 | 0.042 | 0.243 | 0.163 | 0.062 |

Component diagnostics for k = 3:

| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| F_pls_1 | F_career_coherence | 0.195 | 0.511 | 0.417 | 0.264 | 0.714 | 0.229 |
| F_pls_1 | F_evidence_support_rating | 0.177 | 0.566 | 0.458 | 0.425 | 0.590 | 0.167 |
| F_pls_1 | F_rubric_score | 0.181 | 0.286 | 0.333 | 0.269 | 0.717 | 0.229 |
| F_pls_1 | F_trajectory_strength | 0.178 | 0.263 | 0.333 | 0.266 | 0.718 | 0.229 |
| F_pls_1 | F_scrappiness | 0.196 | 0.175 | 0.208 | 0.285 | 0.677 | 0.208 |
| F_pls_2 | F_evidence_support_rating | 0.629 | 0.345 | 0.083 | 0.683 | 0.406 | 0.083 |
| F_pls_2 | F_ownership_signal | 0.320 | 0.085 | 0.000 | 0.266 | 0.138 | 0.000 |
| F_pls_2 | F_scrappiness | 0.295 | 0.089 | 0.000 | 0.287 | 0.103 | 0.000 |
| F_pls_2 | F_trajectory_strength | 0.268 | 0.126 | 0.021 | 0.237 | 0.171 | 0.062 |
| F_pls_2 | F_rubric_score | 0.258 | 0.136 | 0.042 | 0.243 | 0.163 | 0.062 |
| F_pls_3 | F_career_coherence | 0.679 | 0.249 | 0.042 | 0.698 | 0.278 | 0.042 |
| F_pls_3 | F_ownership_signal | -0.376 | 0.216 | 0.042 | -0.281 | 0.188 | 0.042 |
| F_pls_3 | F_evidence_support_rating | 0.329 | 0.117 | 0.042 | 0.388 | 0.142 | 0.042 |
| F_pls_3 | F_trajectory_strength | -0.255 | 0.178 | 0.042 | -0.027 | 0.084 | 0.250 |
| F_pls_3 | F_scrappiness | -0.149 | 0.131 | 0.062 | -0.614 | 0.332 | 0.042 |
