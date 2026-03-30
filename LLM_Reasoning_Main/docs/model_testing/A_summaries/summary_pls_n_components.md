# PLS Sweep Summary (No-LLM, Simple Models)

**Context:** `exp_scope=no_llm_exps`, `model_complexity=simple` (LR + XGB1), `feature_pruning=aggressive_pruning`, `transform=PLS`, sweep over `n_components`.

## Why we choose n_components = 6
- **Best average CV performance** across LR and XGB1.
- **Most frequent winner** for LR across combos.
- **Stable variance** with no obvious overfitting spikes.

### Aggregate evidence
- **Mean CV F0.5 by n_components (LR):**
  - k=6: **0.273**
  - k=8: 0.271
  - k=10: 0.271
  - k=4: 0.265
  - k=2: 0.248
- **Mean CV F0.5 by n_components (XGB1):**
  - k=6: **0.270**
  - k=8: 0.267
  - k=4: 0.266
  - k=10: 0.266
  - k=2: 0.247

### Winner counts (best CV per combo)
- **LR:** k=6 wins 7/10 combos
- **XGB1:** k=6 wins 3/10 combos (k=4 and k=8 split the rest)

## Representative results (best sweep per combo)
Format: `CV F0.5 ± std → Full-train F0.5 (best k)`

**LR (Logistic)**
- HQ: **0.240 ± 0.041 → 0.242 (k=6)**
- A: **0.287 ± 0.036 → 0.284 (k=6)**
- D: **0.258 ± 0.039 → 0.262 (k=6)**
- F: **0.288 ± 0.034 → 0.301 (k=10)**
- A+E: **0.299 ± 0.059 → 0.289 (k=6)**
- A+D: **0.290 ± 0.046 → 0.283 (k=6)**
- A+B+C+D+E+F: **0.337 ± 0.033 → 0.344 (k=6)**

**XGB1**
- HQ: **0.250 ± 0.055 → 0.252 (k=4)**
- A: **0.293 ± 0.038 → 0.307 (k=6)**
- D: **0.259 ± 0.027 → 0.264 (k=4)**
- F: **0.289 ± 0.042 → 0.319 (k=8)**
- A+E: **0.283 ± 0.032 → 0.332 (k=6)**
- A+D: **0.300 ± 0.040 → 0.318 (k=6)**
- A+B+C+D+E+F: **0.331 ± 0.036 → 0.359 (k=8)**

## Decision
We set **`PLS_COMPONENTS_DEFAULT = 6`** as the default. This is the strongest and most stable choice across models and combos, and aligns with the best mean performance.

