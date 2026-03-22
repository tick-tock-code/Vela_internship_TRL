# VCBench Human-Feature Script Functionality

This document explains what `vcbench_lambda_features_minimal.py` does after it
identifies the 15 human-made features.

## Step-by-step flow (human-only mode)

1. Load data
   - Reads the VCBench CSV into a list of founder records and labels.

2. Split into train/test
   - Performs a stratified train/test split before any model training.
   - This prevents leakage from test into training.

3. Extract 15 human features
   - For each record, the script computes these 15 binary features:
     - `top_university`
     - `has_phd`
     - `has_mba`
     - `stem_degree`
     - `prior_exit`
     - `multiple_exits`
     - `senior_leadership`
     - `large_company_exp`
     - `startup_exp`
     - `long_experience`
     - `serial_founder`
     - `technical_role`
     - `many_prior_roles`
     - `short_tenure_pattern`
     - `industry_match`

4. Build feature matrices
   - Creates `X_train` and `X_test` from the extracted features.
   - Values are converted to float for model fitting.

5. Train the model
   - Fits a logistic regression classifier on `X_train` and `y_train`.

6. Tune a decision threshold on the training set
   - Sweeps thresholds from 0.05 to 0.95.
   - Picks the threshold that maximizes F0.5 on training data.

7. Evaluate on test set
   - Uses the chosen threshold on `X_test` to produce predictions.
   - Reports ROC-AUC and PR-AUC using predicted probabilities.
   - Reports precision, recall, F0.5, and accuracy using predicted labels.

8. Report coefficients
   - Prints logistic regression coefficients per feature (sorted by magnitude).
   - This provides a simple interpretability view of which features matter most.

## Output interpretation

- ROC-AUC and PR-AUC measure ranking performance.
- F0.5 emphasizes precision over recall.
- Coefficients indicate direction and strength of each human feature.

