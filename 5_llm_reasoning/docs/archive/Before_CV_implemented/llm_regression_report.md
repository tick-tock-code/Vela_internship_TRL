# LLM Regression Summary (F0.5)

| Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |
|---|---:|---:|---:|---:|---:|---:|
| Human Only | 0.248 | 0.730 | 0.215 | 0.233 | 0.333 | 0.841 |
| LLM Reasoning Only | 0.158 | 0.615 | 0.129 | 0.134 | 0.556 | 0.638 |
| LLM Reasoning + Human | 0.245 | 0.724 | 0.224 | 0.235 | 0.296 | 0.850 |
| LLM Engineered Only | 0.251 | 0.679 | 0.224 | 0.227 | 0.432 | 0.817 |
| LLM Engineered + Reasoning | 0.304 | 0.691 | 0.248 | 0.293 | 0.358 | 0.864 |

## LLM Engineered Run-Family Leaderboard

| Set ID | Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| set_01 | LLM Engineered Only | 0.262 | 0.663 | 0.195 | 0.267 | 0.247 | 0.871 |
| set_01 | LLM Engineered + Reasoning | 0.261 | 0.706 | 0.227 | 0.310 | 0.160 | 0.892 |
| set_02 | LLM Engineered Only | 0.214 | 0.664 | 0.186 | 0.204 | 0.272 | 0.839 |
| set_02 | LLM Engineered + Reasoning | 0.296 | 0.681 | 0.210 | 0.317 | 0.235 | 0.886 |
| set_03 | LLM Engineered Only | 0.289 | 0.675 | 0.232 | 0.327 | 0.198 | 0.891 |
| set_03 | LLM Engineered + Reasoning | 0.287 | 0.695 | 0.252 | 0.333 | 0.185 | 0.893 |
| set_04 | LLM Engineered Only | 0.279 | 0.640 | 0.184 | 0.304 | 0.210 | 0.886 |
| set_04 | LLM Engineered + Reasoning | 0.299 | 0.665 | 0.199 | 0.303 | 0.284 | 0.877 |
| set_05 | LLM Engineered Only | 0.232 | 0.661 | 0.179 | 0.282 | 0.136 | 0.891 |
| set_05 | LLM Engineered + Reasoning | 0.247 | 0.677 | 0.200 | 0.244 | 0.259 | 0.861 |
| set_06 | LLM Engineered Only | 0.226 | 0.663 | 0.205 | 0.212 | 0.309 | 0.834 |
| set_06 | LLM Engineered + Reasoning | 0.265 | 0.668 | 0.219 | 0.258 | 0.296 | 0.860 |
| set_07 | LLM Engineered Only | 0.180 | 0.666 | 0.184 | 0.196 | 0.136 | 0.872 |
| set_07 | LLM Engineered + Reasoning | 0.201 | 0.673 | 0.194 | 0.229 | 0.136 | 0.881 |
| set_08 | LLM Engineered Only | 0.277 | 0.697 | 0.211 | 0.295 | 0.222 | 0.882 |
| set_08 | LLM Engineered + Reasoning | 0.293 | 0.699 | 0.239 | 0.333 | 0.198 | 0.892 |
| set_09 | LLM Engineered Only | 0.208 | 0.680 | 0.199 | 0.205 | 0.222 | 0.852 |
| set_09 | LLM Engineered + Reasoning | 0.276 | 0.698 | 0.232 | 0.280 | 0.259 | 0.873 |
| set_10 | LLM Engineered Only | 0.246 | 0.679 | 0.203 | 0.231 | 0.333 | 0.840 |
| set_10 | LLM Engineered + Reasoning | 0.284 | 0.688 | 0.232 | 0.284 | 0.284 | 0.871 |

### Average Improvement (Engineered + Reasoning vs Engineered Only)
- F0.5: +0.030
- ROC-AUC: +0.016
- PR-AUC: +0.023
- Prec: +0.037
- Rec: +0.001
- Acc: +0.013
