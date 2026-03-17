"""Lightweight smoke tests for vcbench_pipeline helpers.

These tests are intentionally small and fast; they do not touch the dataset.
"""

from __future__ import annotations

import numpy as np

import vcbench_pipeline as vp


def test_precision_at_k_basic() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.1, 0.0])
    # Top 50% => top 2 scores: [1,0] => precision 0.5
    assert abs(vp._precision_at_k(y_true, y_scores, 0.5) - 0.5) < 1e-9


def test_report_metrics_keys() -> None:
    y_train = np.array([0, 1, 0, 1])
    train_scores = np.array([0.1, 0.9, 0.2, 0.8])
    y_test = np.array([0, 1])
    test_scores = np.array([0.2, 0.7])
    metrics = vp._report_metrics(y_train, train_scores, y_test, test_scores)
    for key in (
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f0.5",
        "threshold",
        "tp",
        "fn",
        "tn",
        "fp",
        "fnr",
        "precision@1%",
        "precision@5%",
        "precision@10%",
    ):
        assert key in metrics


def test_train_sklearn_shapes() -> None:
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[1.5], [2.5]])
    train_scores, test_scores, model = vp._train_sklearn(
        X_train, y_train, X_test, random_state=42
    )
    assert train_scores.shape == (4,)
    assert test_scores.shape == (2,)
    assert model.coef_.shape == (1, 1)


if __name__ == "__main__":
    test_precision_at_k_basic()
    test_report_metrics_keys()
    test_train_sklearn_shapes()
    print("Smoke tests passed.")
