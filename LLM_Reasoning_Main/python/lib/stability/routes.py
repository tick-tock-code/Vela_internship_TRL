from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

PRIMARY_METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f0_5",
    "brier",
    "precision_at_01",
    "precision_at_05",
    "precision_at_10",
)

XGB_PARAMS = {
    "n_estimators": 227,
    "max_depth": 1,
    "learning_rate": 0.0674,
    "subsample": 0.949,
    "colsample_bytree": 0.413,
    "scale_pos_weight": 10,
    "min_child_weight": 14,
    "gamma": 4.19,
    "reg_alpha": 0.73,
    "reg_lambda": 15.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
}


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, pct: float) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(y_true) * pct)))
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[order]))


def _metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, preds, zero_division=0.0)),
        "recall": float(recall_score(y_true, preds, zero_division=0.0)),
        "f0_5": float(fbeta_score(y_true, preds, beta=0.5, zero_division=0.0)),
        "brier": float(brier_score_loss(y_true, scores)),
        "precision_at_01": _precision_at_k(y_true, scores, 0.01),
        "precision_at_05": _precision_at_k(y_true, scores, 0.05),
        "precision_at_10": _precision_at_k(y_true, scores, 0.10),
        "threshold": float(threshold),
    }


def _threshold_grid(mode: str) -> np.ndarray:
    if mode == "mirror":
        return np.linspace(0.30, 0.95, 50)
    return np.arange(0.05, 0.96, 0.01)


def _select_threshold(y_true: np.ndarray, scores: np.ndarray, mode: str) -> float:
    best_threshold = 0.5
    best_value = -1.0
    for threshold in _threshold_grid(mode):
        value = fbeta_score(y_true, (scores >= threshold).astype(int), beta=0.5, zero_division=0.0)
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def _fill_missing(train_df: pd.DataFrame, test_df: pd.DataFrame, *, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model == "xgb1":
        return train_df.fillna(0.0), test_df.fillna(0.0)
    fill_values = train_df.mean(numeric_only=True)
    train_filled = train_df.fillna(fill_values).fillna(0.0)
    test_filled = test_df.fillna(fill_values).fillna(0.0)
    return train_filled, test_filled


def _standardize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = train_df.mean(numeric_only=True)
    stds = train_df.std(numeric_only=True).replace(0.0, 1.0)
    train_scaled = (train_df - means) / stds
    test_scaled = (test_df - means) / stds
    return train_scaled.fillna(0.0), test_scaled.fillna(0.0)


def _apply_rule_override(scores: np.ndarray, rule_mask: np.ndarray | None) -> np.ndarray:
    if rule_mask is None:
        return scores
    adjusted = scores.copy()
    adjusted[rule_mask.astype(bool)] = 1.0
    return adjusted


def _build_model(route_spec: dict[str, Any], random_state: int) -> Any:
    model_type = str(route_spec.get("model", "logistic"))
    if model_type == "logistic":
        return LogisticRegression(
            solver="lbfgs",
            C=float(route_spec.get("c", 1.0)),
            max_iter=3000,
            random_state=random_state,
        )
    if model_type == "mlp4":
        return MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="relu",
            alpha=float(route_spec.get("mlp_alpha", 0.01)),
            max_iter=2000,
            random_state=random_state,
            early_stopping=False,
        )
    if model_type == "xgb1":
        try:
            import xgboost as xgb  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "xgboost is required for XGB routes. Install with: pip install xgboost"
            ) from exc
        return xgb.XGBClassifier(**XGB_PARAMS)
    raise ValueError(f"Unsupported route model: {model_type}")


def _preprocess_fold(
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    extra_train: pd.DataFrame,
    extra_test: pd.DataFrame,
    y_train: np.ndarray,
    route_spec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    drop_baseline_features = set(str(value) for value in route_spec.get("drop_baseline_features", []))
    baseline_train_use = baseline_train.drop(columns=[c for c in drop_baseline_features if c in baseline_train.columns])
    baseline_test_use = baseline_test.drop(columns=[c for c in drop_baseline_features if c in baseline_test.columns])
    X_train = pd.concat([baseline_train_use, extra_train], axis=1)
    X_test = pd.concat([baseline_test_use, extra_test], axis=1)
    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_test = X_test.loc[:, X_train.columns].copy()

    model_type = str(route_spec.get("model", "logistic"))
    X_train, X_test = _fill_missing(X_train, X_test, model=model_type)

    transform = str(route_spec.get("transform", "BASE")).upper()
    if transform == "PLS" or model_type in {"logistic", "mlp4"}:
        X_train, X_test = _standardize(X_train, X_test)

    if transform == "PLS":
        n_components = min(
            int(route_spec.get("pls_components", 6)),
            X_train.shape[1],
            max(1, len(y_train) - 1),
        )
        pls = PLSRegression(n_components=n_components)
        X_train_arr = pls.fit_transform(X_train.to_numpy(dtype=float), y_train)[0]
        X_test_arr = pls.transform(X_test.to_numpy(dtype=float))
        return X_train_arr, X_test_arr

    return X_train.to_numpy(dtype=float), X_test.to_numpy(dtype=float)


def evaluate_route_cv(
    baseline_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    labels: pd.Series,
    fold_ids: np.ndarray,
    route_spec: dict[str, Any],
    *,
    rule_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    y = labels.to_numpy(dtype=int)
    all_oof_scores = np.zeros(len(labels), dtype=float)
    all_oof_labels = np.zeros(len(labels), dtype=int)
    fold_results: list[dict[str, float]] = []
    fold_ids_seen: list[int] = []

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for fold_id in sorted(set(int(value) for value in fold_ids.tolist())):
        train_idx = np.where(fold_ids != fold_id)[0]
        test_idx = np.where(fold_ids == fold_id)[0]
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        X_train, X_test = _preprocess_fold(
            baseline_df.iloc[train_idx].copy(),
            baseline_df.iloc[test_idx].copy(),
            extra_df.iloc[train_idx].copy(),
            extra_df.iloc[test_idx].copy(),
            y_train,
            route_spec,
        )
        model = _build_model(route_spec, random_state=42)
        model.fit(X_train, y_train)
        test_scores = model.predict_proba(X_test)[:, 1]
        if bool(route_spec.get("apply_rule_override", False)):
            test_scores = _apply_rule_override(test_scores, None if rule_mask is None else rule_mask[test_idx])

        all_oof_scores[test_idx] = test_scores
        all_oof_labels[test_idx] = y_test
        fold_ids_seen.append(fold_id)

    if not fold_ids_seen:
        raise RuntimeError(f"No valid folds available for route {route_spec.get('id', 'unknown')}")

    oof_threshold = _select_threshold(
        all_oof_labels,
        all_oof_scores,
        mode=str(route_spec.get("threshold_grid", "default")),
    )
    for fold_id in fold_ids_seen:
        test_idx = np.where(fold_ids == fold_id)[0]
        fold_results.append(_metric_row(y[test_idx], all_oof_scores[test_idx], oof_threshold))

    full_train_X, _ = _preprocess_fold(
        baseline_df.copy(),
        baseline_df.copy(),
        extra_df.copy(),
        extra_df.copy(),
        y,
        route_spec,
    )
    full_model = _build_model(route_spec, random_state=42)
    full_model.fit(full_train_X, y)
    full_scores = full_model.predict_proba(full_train_X)[:, 1]
    if bool(route_spec.get("apply_rule_override", False)):
        full_scores = _apply_rule_override(full_scores, rule_mask)
    full_metrics = _metric_row(y, full_scores, oof_threshold)

    summary: dict[str, float] = {}
    for key in PRIMARY_METRIC_KEYS:
        values = [fold[key] for fold in fold_results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["fold_count"] = float(len(fold_results))
    summary["threshold_oof"] = float(oof_threshold)
    summary["full_train_f0_5"] = float(full_metrics["f0_5"])
    summary["full_train_precision"] = float(full_metrics["precision"])
    summary["full_train_recall"] = float(full_metrics["recall"])
    summary["full_train_pr_auc"] = float(full_metrics["pr_auc"])

    return {
        "route_id": str(route_spec.get("id", "")),
        "route_label": str(route_spec.get("label", route_spec.get("id", ""))),
        "model": str(route_spec.get("model", "logistic")),
        "transform": str(route_spec.get("transform", "BASE")).upper(),
        "summary": summary,
        "per_fold": fold_results,
    }
