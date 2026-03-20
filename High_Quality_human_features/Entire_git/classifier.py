# classifier.py
"""
Main classifier. Iterated on by Claude Code in Phase 4.
Final version after 120+ experiments and Optuna Bayesian optimization.

Architecture: XGBoost decision stumps (max_depth=1) + prior_exit rule layer.
Key finding: Heavy regularization (mcw=14, gamma=4.19, reg_lambda=15) forces
the model into simple additive decision stumps, which generalize best on this
small, imbalanced dataset (405 positives in 4500 rows).

Phase 3 result (full coverage): 9 LLM features tested, CV delta -0.05pp.
LLM features are redundant with structured features. Reverted to structured-only.

evaluate.py is FIXED. Do not modify.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import StratifiedKFold
from evaluate import evaluate, sweep_thresholds, best_threshold
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules

FEATURE_COLS = [
    # Tier 1
    "has_prior_ipo", "has_prior_acquisition", "exit_count",
    # Tier 2 — sacrifice signal
    "max_company_size_before_founding", "prestige_sacrifice_score",
    "years_in_large_company", "comfort_index", "founding_timing",
    # Tier 3 — education x QS
    "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
    "degree_level", "stem_flag", "best_degree_prestige",
    # Tier 4 — trajectory
    "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
    "restlessness_score", "founding_role_count", "longest_founding_tenure",
    "industry_pivot_count", "industry_alignment", "total_inferred_experience",
    # v2 interaction features
    "is_serial_founder", "exit_x_serial", "sacrifice_x_serial",
    "industry_prestige_penalty", "persistence_score",
]

# Optuna-optimized: decision stumps with heavy regularization
MODEL_PARAMS = dict(
    n_estimators=227,
    max_depth=1,
    learning_rate=0.0674,
    subsample=0.949,
    colsample_bytree=0.413,
    scale_pos_weight=10,
    min_child_weight=14,
    gamma=4.19,
    reg_alpha=0.73,
    reg_lambda=15.0,
    eval_metric="logloss",
    random_state=42,
)


def make_model():
    """Return a fresh XGBClassifier with the Optuna-tuned hyperparameters."""
    return xgb.XGBClassifier(**MODEL_PARAMS)


def cv_evaluate_with_rules(df, features, n_splits=5):
    """Run stratified k-fold CV with the rule layer applied inside each fold.

    The rule layer (high_precision_rules.apply_rules) is applied to validation
    rows *after* the model scores them, overriding probabilities to 1.0 where a
    rule fires. This mirrors the exact inference path used in production.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-extracted dataframe including a ``success`` column.
    features : list[str]
        Feature column names to pass to the model.
    n_splits : int
        Number of CV folds (default 5).

    Returns
    -------
    dict
        ``{"cv_mean_f05": float, "cv_std_f05": float}``
    """
    X = df[features].fillna(0)
    y = df["success"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = make_model()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1].tolist()
        val_rows = df.iloc[val_idx]
        for j, (_, row) in enumerate(val_rows.iterrows()):
            rp, _ = apply_rules(row)
            if rp == 1:
                probs[j] = 1.0
        scores.append(best_threshold(y_val.tolist(), probs)["f05"])
    return {"cv_mean_f05": round(np.mean(scores), 4), "cv_std_f05": round(np.std(scores), 4)}


def train_and_evaluate():
    """Train the final model on public_train.csv and evaluate on public_val.csv.

    Workflow:
    1. Run 5-fold CV on the training set (with rule layer) to get CV F₀.₅.
    2. Train the final model on the full training set.
    3. Apply the rule layer to validation probabilities.
    4. Sweep thresholds to find the best F₀.₅ on the validation set.
    5. Print feature importances and save the trained model to model.pkl.

    Returns
    -------
    tuple[dict, dict]
        ``(cv_result, val_result)`` where each dict contains f05, precision,
        recall, and threshold keys.
    """
    train = extract_features(pd.read_csv("data/public_train.csv"))
    val = extract_features(pd.read_csv("data/public_val.csv"))

    X_train = train[FEATURE_COLS].fillna(0)
    y_train = train["success"]
    X_val = val[FEATURE_COLS].fillna(0)
    y_val = val["success"]

    # 5-fold CV on training set (with rules)
    cv_result = cv_evaluate_with_rules(train, FEATURE_COLS)
    print(f"=== 5-fold CV (with rules) ===")
    print(f"CV F0.5: {cv_result['cv_mean_f05']:.4f} ± {cv_result['cv_std_f05']:.4f}")

    # Train final model on full training set
    model = make_model()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Get base probabilities
    probs = model.predict_proba(X_val)[:, 1].tolist()

    # Apply rule layer
    rule_stats = {}
    for idx_pos, (idx, row) in enumerate(val.iterrows()):
        rule_pred, rule_name = apply_rules(row)
        if rule_pred == 1:
            probs[idx_pos] = 1.0
            if rule_name not in rule_stats:
                rule_stats[rule_name] = 0
            rule_stats[rule_name] += 1

    print(f"\nRule layer overrides: {rule_stats}")

    result = best_threshold(y_val.tolist(), probs)
    print(f"\n=== Val set result ===")
    print(result)

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nTop 10 features:")
    print(importances.head(10))

    # Threshold sweep
    print("\n=== Threshold sweep (top 5) ===")
    sweep = sweep_thresholds(y_val.tolist(), probs)
    for r in sweep[:5]:
        print(r)

    joblib.dump(model, "model.pkl")
    return cv_result, result


if __name__ == "__main__":
    train_and_evaluate()
