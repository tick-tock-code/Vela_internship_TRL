# F0.5 Summary — Structured v2 (28 feats) + XGB stumps + prior_exit rule

Common setup: 28 structured v2 features, XGBoost stumps (HPO config), prior_exit rule layer.

- Strict CV (5-fold)
  Threshold tuning: on train folds only
  F0.5: 0.216 +/- 0.027

- Mirror Holdout (fixed 80/20 split, seed=42)
  Threshold tuning: on validation split
  F0.5: 0.2918

- Mirror CV Tuned (5-fold)
  Threshold tuning: on each validation fold
  F0.5: 0.2592 +/- 0.0390
