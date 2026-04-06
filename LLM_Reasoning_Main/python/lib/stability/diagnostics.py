from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lib.shared.metrics import binary_metrics
from lib.stability.family_registry import AlignedFamilyData


def _fill_missing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill_values = train_df.mean(numeric_only=True)
    return train_df.fillna(fill_values), test_df.fillna(fill_values)


def evaluate_logistic_cv(feature_df: pd.DataFrame, labels: pd.Series, fold_ids: np.ndarray) -> dict[str, Any]:
    metrics_by_fold: list[dict[str, float]] = []
    X = feature_df.copy()
    y = labels.to_numpy(dtype=int)

    for fold_id in sorted(set(int(value) for value in fold_ids.tolist())):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        X_train, X_test = _fill_missing(X_train, X_test)
        model = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train.values.astype(float), y_train)
        train_scores = model.predict_proba(X_train.values.astype(float))[:, 1]
        test_scores = model.predict_proba(X_test.values.astype(float))[:, 1]
        metrics_by_fold.append(binary_metrics(y_train, train_scores, y_test, test_scores))

    if not metrics_by_fold:
        raise RuntimeError("No valid CV folds were available for evaluation.")

    summary: dict[str, float] = {}
    metric_keys = [key for key in metrics_by_fold[0] if key != "threshold"]
    for key in metric_keys:
        values = [fold[key] for fold in metrics_by_fold]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["fold_count"] = float(len(metrics_by_fold))
    return {"summary": summary, "per_fold": metrics_by_fold}


def _correlation_summary(feature_df: pd.DataFrame) -> dict[str, float]:
    if feature_df.shape[1] < 2:
        return {"max_abs_corr": 0.0, "mean_abs_corr": 0.0}
    corr = feature_df.corr(numeric_only=True).abs()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr.where(mask).stack().to_numpy()
    if values.size == 0:
        return {"max_abs_corr": 0.0, "mean_abs_corr": 0.0}
    return {
        "max_abs_corr": float(values.max()),
        "mean_abs_corr": float(values.mean()),
    }


def _cross_correlation_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    if left.empty or right.empty:
        return {"max_abs_cross_corr": 0.0, "mean_abs_cross_corr": 0.0}
    left_filled = left.fillna(left.mean(numeric_only=True))
    right_filled = right.fillna(right.mean(numeric_only=True))
    matrix = pd.concat({"left": left_filled, "right": right_filled}, axis=1).corr().abs()
    cross = matrix.loc["left", "right"]
    values = cross.to_numpy().reshape(-1)
    if values.size == 0:
        return {"max_abs_cross_corr": 0.0, "mean_abs_cross_corr": 0.0}
    return {
        "max_abs_cross_corr": float(values.max()),
        "mean_abs_cross_corr": float(values.mean()),
    }


def compute_family_diagnostics(
    baseline_df: pd.DataFrame,
    family_df: pd.DataFrame,
    labels: pd.Series,
    fold_ids: np.ndarray,
) -> dict[str, Any]:
    baseline_eval = evaluate_logistic_cv(baseline_df, labels, fold_ids)
    combined_eval = evaluate_logistic_cv(pd.concat([baseline_df, family_df], axis=1), labels, fold_ids)
    baseline_summary = baseline_eval["summary"]
    combined_summary = combined_eval["summary"]
    return {
        "family_feature_count": int(family_df.shape[1]),
        "baseline_feature_count": int(baseline_df.shape[1]),
        "family_corr": _correlation_summary(family_df),
        "cross_corr": _cross_correlation_summary(baseline_df, family_df),
        "baseline_metrics": baseline_summary,
        "combined_metrics": combined_summary,
        "delta_f0_5_mean": float(combined_summary["f0_5_mean"] - baseline_summary["f0_5_mean"]),
        "delta_f0_5_std": float(combined_summary["f0_5_std"] - baseline_summary["f0_5_std"]),
        "delta_pr_auc_mean": float(combined_summary["pr_auc_mean"] - baseline_summary["pr_auc_mean"]),
        "delta_precision_at_10_mean": float(
            combined_summary["precision_at_10_mean"] - baseline_summary["precision_at_10_mean"]
        ),
    }


def run_family_diagnostics(data: AlignedFamilyData, fold_ids: np.ndarray) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for family_id, family_df in data.candidate_frames.items():
        results[family_id] = compute_family_diagnostics(
            data.baseline,
            family_df,
            data.labels,
            fold_ids,
        )
    return results
