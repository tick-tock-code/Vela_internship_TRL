from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression

from lib.shared.metrics import binary_metrics


def _fill_missing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fill_values = train_df.mean(numeric_only=True)
    return train_df.fillna(fill_values), test_df.fillna(fill_values)


def evaluate_route_cv(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    fold_ids: np.ndarray,
    route_spec: dict[str, Any],
) -> dict[str, Any]:
    route_type = str(route_spec.get("type", "logistic"))
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

        if route_type == "elasticnet_logistic":
            model = LogisticRegression(
                max_iter=4000,
                solver="saga",
                penalty="elasticnet",
                l1_ratio=float(route_spec.get("l1_ratio", 0.5)),
                C=float(route_spec.get("c", 1.0)),
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_train.values.astype(float), y_train)
            train_scores = model.predict_proba(X_train.values.astype(float))[:, 1]
            test_scores = model.predict_proba(X_test.values.astype(float))[:, 1]
        elif route_type == "pls_logistic":
            n_components = min(
                int(route_spec.get("n_components", 6)),
                X_train.shape[1],
                max(1, len(train_idx) - 1),
            )
            pls = PLSRegression(n_components=n_components)
            X_train_pls = pls.fit_transform(X_train.values.astype(float), y_train)[0]
            X_test_pls = pls.transform(X_test.values.astype(float))
            model = LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                random_state=42,
            )
            model.fit(X_train_pls, y_train)
            train_scores = model.predict_proba(X_train_pls)[:, 1]
            test_scores = model.predict_proba(X_test_pls)[:, 1]
        else:
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
        raise RuntimeError(f"No valid folds available for route {route_type}.")

    summary: dict[str, float] = {}
    metric_keys = [key for key in metrics_by_fold[0] if key != "threshold"]
    for key in metric_keys:
        values = [fold[key] for fold in metrics_by_fold]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))

    return {
        "route_id": str(route_spec.get("id", route_type)),
        "route_type": route_type,
        "parameters": route_spec,
        "summary": summary,
    }
