from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _oof_scores(feature_df: pd.DataFrame, labels: pd.Series, fold_ids: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(feature_df), dtype=float)
    X = feature_df.copy()
    y = labels.to_numpy(dtype=int)
    for fold_id in sorted(set(int(value) for value in fold_ids.tolist())):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        X_train = X.iloc[train_idx].fillna(X.iloc[train_idx].mean(numeric_only=True))
        X_test = X.iloc[test_idx].fillna(X.iloc[train_idx].mean(numeric_only=True))
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train.values.astype(float), y_train)
        scores[test_idx] = model.predict_proba(X_test.values.astype(float))[:, 1]
    return scores


def summarize_residual_gain(
    baseline_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    labels: pd.Series,
    fold_ids: np.ndarray,
    *,
    uncertainty_quantile: float = 0.2,
) -> dict[str, Any]:
    y = labels.to_numpy(dtype=int)
    baseline_scores = _oof_scores(baseline_df, labels, fold_ids)
    combined_scores = _oof_scores(combined_df, labels, fold_ids)
    baseline_preds = (baseline_scores >= 0.5).astype(int)
    combined_preds = (combined_scores >= 0.5).astype(int)

    baseline_false_negatives = np.where((y == 1) & (baseline_preds == 0))[0]
    recovered = np.where((y == 1) & (baseline_preds == 0) & (combined_preds == 1))[0]
    uncertainty = np.abs(baseline_scores - 0.5)
    cutoff = np.quantile(uncertainty, uncertainty_quantile)
    borderline = np.where(uncertainty <= cutoff)[0]

    return {
        "baseline_false_negatives": int(len(baseline_false_negatives)),
        "recovered_false_negatives": int(len(recovered)),
        "borderline_case_count": int(len(borderline)),
        "borderline_recovered": int(
            np.sum((y[borderline] == 1) & (baseline_preds[borderline] == 0) & (combined_preds[borderline] == 1))
        ),
        "mean_positive_score_shift": float(np.mean(combined_scores[y == 1] - baseline_scores[y == 1])),
        "mean_negative_score_shift": float(np.mean(combined_scores[y == 0] - baseline_scores[y == 0])),
    }
