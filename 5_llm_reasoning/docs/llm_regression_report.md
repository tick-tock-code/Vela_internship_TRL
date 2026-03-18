# LLM Regression Summary (F0.5)

| Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |
|---|---:|---:|---:|---:|---:|---:|
| Human Only | 0.206+/-0.055 | 0.672+/-0.041 | 0.189+/-0.034 | 0.202+/-0.050 | 0.235+/-0.091 | 0.850+/-0.014 |
| LLM Reasoning Only | 0.143+/-0.019 | 0.596+/-0.031 | 0.131+/-0.018 | 0.124+/-0.015 | 0.419+/-0.142 | 0.683+/-0.069 |
| LLM Reasoning + Human | 0.214+/-0.078 | 0.671+/-0.041 | 0.194+/-0.036 | 0.226+/-0.077 | 0.194+/-0.095 | 0.869+/-0.017 |
| LLM Engineered Only | 0.164+/-0.066 | 0.631+/-0.048 | 0.166+/-0.050 | 0.156+/-0.057 | 0.224+/-0.119 | 0.830+/-0.026 |
| LLM Engineered + Reasoning | 0.210+/-0.051 | 0.644+/-0.039 | 0.177+/-0.043 | 0.215+/-0.053 | 0.197+/-0.053 | 0.863+/-0.012 |


*CV: 10-fold stratified on 4400 founders (seed excluded).*

## LLM Engineered Run-Family Leaderboard

| Set ID | Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |
|---|---|---:|---:|---:|---:|---:|---:|
| set_01 | LLM Engineered Only | 0.150 | 0.564 | 0.133 | 0.162 | 0.119 | 0.866 |
| set_01 | LLM Engineered + Reasoning | 0.173 | 0.614 | 0.159 | 0.187 | 0.147 | 0.864 |
| set_02 | LLM Engineered Only | 0.183 | 0.581 | 0.142 | 0.174 | 0.234 | 0.830 |
| set_02 | LLM Engineered + Reasoning | 0.200 | 0.633 | 0.172 | 0.215 | 0.166 | 0.871 |
| set_03 | LLM Engineered Only | 0.191 | 0.595 | 0.153 | 0.194 | 0.187 | 0.856 |
| set_03 | LLM Engineered + Reasoning | 0.186 | 0.637 | 0.176 | 0.206 | 0.156 | 0.867 |
| set_04 | LLM Engineered Only | 0.195 | 0.578 | 0.153 | 0.241 | 0.121 | 0.887 |
| set_04 | LLM Engineered + Reasoning | 0.187 | 0.636 | 0.176 | 0.226 | 0.124 | 0.883 |
| set_05 | LLM Engineered Only | 0.176 | 0.603 | 0.153 | 0.189 | 0.144 | 0.868 |
| set_05 | LLM Engineered + Reasoning | 0.169 | 0.648 | 0.181 | 0.194 | 0.116 | 0.878 |
| set_06 | LLM Engineered Only | 0.173 | 0.600 | 0.156 | 0.179 | 0.159 | 0.860 |
| set_06 | LLM Engineered + Reasoning | 0.179 | 0.636 | 0.176 | 0.192 | 0.152 | 0.870 |
| set_07 | LLM Engineered Only | 0.187 | 0.614 | 0.165 | 0.264 | 0.089 | 0.896 |
| set_07 | LLM Engineered + Reasoning | 0.198 | 0.648 | 0.189 | 0.216 | 0.156 | 0.874 |
| set_08 | LLM Engineered Only | 0.213 | 0.622 | 0.164 | 0.229 | 0.189 | 0.867 |
| set_08 | LLM Engineered + Reasoning | 0.231 | 0.648 | 0.185 | 0.235 | 0.220 | 0.865 |
| set_09 | LLM Engineered Only | 0.164 | 0.615 | 0.162 | 0.159 | 0.211 | 0.839 |
| set_09 | LLM Engineered + Reasoning | 0.176 | 0.650 | 0.176 | 0.175 | 0.186 | 0.850 |
| set_10 | LLM Engineered Only | 0.169 | 0.617 | 0.161 | 0.159 | 0.240 | 0.818 |
| set_10 | LLM Engineered + Reasoning | 0.182 | 0.645 | 0.178 | 0.188 | 0.167 | 0.859 |

### Average Improvement (Engineered + Reasoning vs Engineered Only)
- F0.5: +0.008
- ROC-AUC: +0.040
- PR-AUC: +0.022
- Prec: +0.008
- Rec: -0.010
- Acc: +0.009

*Metrics are 10-fold stratified CV on 4400 founders (seed excluded).*
