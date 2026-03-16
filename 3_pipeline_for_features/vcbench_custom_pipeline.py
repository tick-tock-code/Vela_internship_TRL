"""VCBench custom feature pipeline.

Builds a feature matrix from:
- 15 baseline features (imported from the example script)
- custom features defined in feature_registry.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from think_reason_learn.datasets import load_vcbench

from feature_registry import FEATURE_REGISTRY, FEATURE_SETS


def _load_base_feature_extractor(script_path: Path):
    spec = importlib.util.spec_from_file_location("vcbench_base_features", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base feature script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    if not hasattr(module, "_extract_human_features"):
        raise AttributeError(
            f"Base script missing _extract_human_features: {script_path}"
        )
    return module._extract_human_features  # type: ignore[attr-defined]


def _custom_feature_df(
    records: Iterable[dict[str, Any]], feature_names: list[str]
) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {}
        for name in feature_names:
            spec = FEATURE_REGISTRY.get(name)
            if spec is None:
                raise KeyError(f"Unknown feature: {name}")
            row[name] = spec.func(r)
        rows.append(row)
    return pd.DataFrame(rows)


def _report(
    label: str,
    feature_names: list[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # F0.5-optimal threshold on training set (handles class imbalance).
    y_prob_tr = clf.predict_proba(X_train)[:, 1]
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(
            y_train,
            (y_prob_tr >= t).astype(int),
            beta=0.5,
            zero_division=0.0,  # type: ignore[arg-type]
        )
        if f > best_f:
            best_f, best_t = f, float(t)

    y_pred = (y_prob >= best_t).astype(int)
    roc = roc_auc_score(y_test, y_prob)
    pr = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0.0)  # type: ignore[arg-type]
    rec = recall_score(y_test, y_pred, zero_division=0.0)  # type: ignore[arg-type]
    f05 = fbeta_score(y_test, y_pred, beta=0.5, zero_division=0.0)  # type: ignore[arg-type]

    coefs = sorted(
        zip(feature_names, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True
    )
    print(f"\n  [{label}] — {len(feature_names)} features, threshold={best_t:.2f}")
    for name, c in coefs:
        print(f"    {'+' if c >= 0 else '-'}{abs(c):.3f}  {name}")

    print(
        f"\n  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  "
        f"Prec={prec:.3f}  Rec={rec:.3f}  F0.5={f05:.3f}  Acc={acc:.3f}"
    )

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "precision": prec,
        "recall": rec,
        "f0.5": f05,
        "accuracy": acc,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VCBench custom feature pipeline.")
    p.add_argument("--input_csv", required=True, help="VCBench CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--max_rows", type=int, default=0, help="0 = all data (default).")
    p.add_argument("--test_size", type=float, default=0.40)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--feature_set",
        choices=sorted(FEATURE_SETS.keys()),
        default="base_plus_custom",
        help="Named feature set from feature_registry.py",
    )
    p.add_argument(
        "--features",
        default="",
        help="Optional comma-separated custom feature names (overrides feature_set)",
    )
    p.add_argument(
        "--base_script",
        default=str(
            Path(
                r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
            )
            / "2_Human_features_running_example_script"
            / "vcbench_lambda_features_minimal.py"
        ),
        help="Path to base feature script with _extract_human_features.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rs = args.random_state

    print(f"\n{'=' * 60}")
    print("  VCBench Custom Feature Pipeline")
    print(f"{'=' * 60}\n")

    records, labels = load_vcbench(
        args.input_csv,
        args.label_column,
        args.max_rows,
        rs,
    )

    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=labels,
        random_state=rs,
    )
    train_recs = [records[i] for i in train_idx]
    test_recs = [records[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    n_pos_test = int(y_test.sum())
    print(f"  {len(records)} rows -> {len(train_idx)} train / {len(test_idx)} test")
    print(f"  Train: {int(y_train.sum())} pos, {len(y_train) - int(y_train.sum())} neg")
    print(f"  Test:  {n_pos_test} pos, {len(y_test) - n_pos_test} neg")
    if n_pos_test < 10:
        print(
            f"  WARNING: only {n_pos_test} test positives — "
            "try --max_rows 0 or --test_size 0.5"
        )

    base_script = Path(args.base_script)
    extract_base = _load_base_feature_extractor(base_script)
    base_train = pd.DataFrame([extract_base(r) for r in train_recs])
    base_test = pd.DataFrame([extract_base(r) for r in test_recs])
    base_feature_names = list(base_train.columns)

    if args.features.strip():
        custom_features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        custom_features = FEATURE_SETS[args.feature_set]

    custom_train = _custom_feature_df(train_recs, custom_features)
    custom_test = _custom_feature_df(test_recs, custom_features)

    full_train = pd.concat([base_train, custom_train], axis=1)
    full_test = pd.concat([base_test, custom_test], axis=1)
    full_feature_names = base_feature_names + custom_features

    print("\n  Base features:", ", ".join(base_feature_names))
    print("  Custom features:", ", ".join(custom_features))

    _report(
        "Base + Custom",
        full_feature_names,
        full_train.values.astype(float),
        full_test.values.astype(float),
        y_train,
        y_test,
        rs,
    )


if __name__ == "__main__":
    main()

