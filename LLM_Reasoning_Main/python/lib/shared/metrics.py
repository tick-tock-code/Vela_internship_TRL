from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, precision_score, recall_score, roc_auc_score


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, pct: float) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(math.ceil(len(y_true) * pct)))
    order = np.argsort(scores)[::-1][:k]
    return float(np.mean(y_true[order]))


def select_f05_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_value = -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        value = fbeta_score(y_true, (scores >= threshold).astype(int), beta=0.5, zero_division=0.0)
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def binary_metrics(y_train: np.ndarray, train_scores: np.ndarray, y_test: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        threshold = 0.5
    else:
        threshold = select_f05_threshold(y_train, train_scores)
    predictions = (test_scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, test_scores)) if len(np.unique(y_test)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "precision": float(precision_score(y_test, predictions, zero_division=0.0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0.0)),
        "f0_5": float(fbeta_score(y_test, predictions, beta=0.5, zero_division=0.0)),
        "brier": float(brier_score_loss(y_test, test_scores)),
        "precision_at_01": precision_at_k(y_test, test_scores, 0.01),
        "precision_at_05": precision_at_k(y_test, test_scores, 0.05),
        "precision_at_10": precision_at_k(y_test, test_scores, 0.10),
        "threshold": float(threshold),
    }
