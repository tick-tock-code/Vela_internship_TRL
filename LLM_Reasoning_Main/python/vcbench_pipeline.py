"""VCBench in-depth pipeline (human + optional LLM features).

Pipeline steps:
1. Load VCBench data (sample or full).
2. Extract baseline features (15) from example script.
3. Extract custom features from registry.
4. Optionally add LLM-derived features.
5. Save full feature dataset (founder_uuid, label, features) to Parquet.
6. Train logistic regression and evaluate metrics.
"""

from __future__ import annotations

import argparse
import time
import re
import os
import shutil
import json
import hashlib
import importlib.util
import math
import asyncio
import sys
import shutil
import copy
from pathlib import Path
from datetime import datetime
import traceback
import time
from typing import Any, Iterable, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

from cv_folds import load_or_create_folds, resolve_folds_path
from think_reason_learn.datasets import load_vcbench

from feature_registry import FEATURE_REGISTRY, FEATURE_SETS
from llm_feature_generation import generate_llm_features
from llm_reasoning_features import (
    ReasoningConfig,
    build_experiment_key_map,
    generate_reasoning_features,
    write_per_experiment_parquets,
)
from paths import BASE_DIR, PROJECT_ROOT, CONFIG_DIR, PROMPT_DIR

RUN_LOG_PATH: Path | None = None

HQ_FEATURES_BASE = [
    "has_prior_ipo",
    "has_prior_acquisition",
    "exit_count",
    "max_company_size_before_founding",
    "prestige_sacrifice_score",
    "years_in_large_company",
    "comfort_index",
    "founding_timing",
    "edu_prestige_tier",
    "field_relevance_score",
    "prestige_x_relevance",
    "degree_level",
    "stem_flag",
    "best_degree_prestige",
    "max_seniority_reached",
    "seniority_is_monotone",
    "company_size_is_growing",
    "restlessness_score",
    "founding_role_count",
    "longest_founding_tenure",
    "industry_pivot_count",
    "industry_alignment",
    "total_inferred_experience",
    "is_serial_founder",
    "exit_x_serial",
    "sacrifice_x_serial",
    "industry_prestige_penalty",
    "persistence_score",
]
HQ_FEATURES_WITH_GAP = HQ_FEATURES_BASE + ["repeat_founding_gap"]

LEGACY_HUMAN_FEATURE_SETS = [
    {
        "name": "Human Legacy 1 (scaled durations)",
        "features": [
            "has_phd",
            "has_mba",
            "stem_degree",
            "multiple_exits",
            "senior_leadership",
            "startup_exp",
            "short_tenure_pattern",
            "industry_match",
            "qs_top_25",
            "qs_top_200",
            "qs_inverse_best",
            "prior_ipos",
            "prior_acquisitions",
            "large_company_years",
            "experience_duration",
            "technical_experience_duration",
        ],
    },
    {
        "name": "Human Legacy 2 (binary equivalents)",
        "features": [
            "has_phd",
            "has_mba",
            "stem_degree",
            "multiple_exits",
            "senior_leadership",
            "startup_exp",
            "short_tenure_pattern",
            "industry_match",
            "long_experience",
            "technical_role",
            "qs_top_25",
            "qs_top_200",
            "qs_inverse_best",
            "prior_ipos",
            "prior_acquisitions",
            "large_company_years",
        ],
    },
    {
        "name": "Human Legacy 3 (mixed durations)",
        "features": [
            "has_phd",
            "has_mba",
            "stem_degree",
            "multiple_exits",
            "senior_leadership",
            "startup_exp",
            "short_tenure_pattern",
            "industry_match",
            "technical_role",
            "qs_top_25",
            "qs_top_200",
            "qs_inverse_best",
            "prior_ipos",
            "prior_acquisitions",
            "large_company_years",
            "experience_duration",
        ],
    },
]


def _install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        if RUN_LOG_PATH is not None:
            try:
                with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write("\n[ERROR] Unhandled exception:\n")
                    f.write("".join(traceback.format_exception(exc_type, exc, tb)))
            except Exception:
                pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


_install_excepthook()


def _validate_fold_cache(path: Path, expected_rows: int, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing fold cache ({label}): {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    fold_map = data.get("fold_map", {})
    if not isinstance(fold_map, dict):
        raise RuntimeError(f"Invalid fold cache format ({label}): {path}")
    if len(fold_map) != expected_rows:
        raise RuntimeError(
            f"Fold cache size mismatch ({label}): expected {expected_rows}, got {len(fold_map)}"
        )
    missing_uuid_count = int(data.get("missing_uuid_count", 0))
    if missing_uuid_count:
        try:
            print(
                f"  Validation: {label} missing_uuid_count={missing_uuid_count} (row_index fallback)"
            )
        except OSError:
            pass


def _validate_run_state(
    expected_pool: int,
    expected_full: int,
    pool_cache: Path,
    full_cache: Path,
) -> None:
    _validate_fold_cache(pool_cache, expected_pool, "pool")
    _validate_fold_cache(full_cache, expected_full, "full")


def _validate_reasoning_full_df(df: pd.DataFrame, expected_rows: int) -> None:
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Reasoning full_current row mismatch: expected {expected_rows}, got {len(df)}"
        )
    num = df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
    if not num.empty and num.isna().any(axis=1).any():
        raise RuntimeError("NaNs detected in full_current reasoning features.")


def _hash_seed(indices: Sequence[int], uuids: Sequence[str]) -> str:
    payload = {
        "indices": [int(i) for i in list(indices)],
        "uuids": [str(u) for u in list(uuids)],
    }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _load_or_create_seed(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    founder_ids: list[str],
    seed_size: int,
    random_state: int,
    seed_path: Path,
) -> tuple[np.ndarray, list[str], str]:
    if seed_path.exists():
        data = json.loads(seed_path.read_text(encoding="utf-8"))
        indices = data.get("indices", [])
        if not isinstance(indices, list) or len(indices) != seed_size:
            raise RuntimeError(
                f"Seed file invalid or wrong size ({seed_path}). "
                "Delete it to regenerate."
            )
        seed_idx = np.array(indices, dtype=int)
        seed_uuids = [founder_ids[i] for i in seed_idx]
        seed_hash = data.get("seed_hash") or _hash_seed(seed_idx, seed_uuids)
        return seed_idx, seed_uuids, seed_hash

    if seed_size <= 0 or seed_size >= len(records):
        raise ValueError("Seed size must be >0 and < dataset size.")
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=seed_size, random_state=random_state
    )
    seed_idx, _ = next(sss.split(np.zeros(len(labels)), labels))
    seed_idx = np.array(seed_idx, dtype=int)
    seed_uuids = [founder_ids[i] for i in seed_idx]
    seed_hash = _hash_seed(seed_idx, seed_uuids)
    seed_meta = {
        "seed_size": seed_size,
        "random_state": random_state,
        "dataset_size": len(records),
        "indices": seed_idx.tolist(),
        "uuids": seed_uuids,
        "label_counts": {
            "positive": int(labels[seed_idx].sum()),
            "negative": int(seed_size - int(labels[seed_idx].sum())),
        },
        "seed_hash": seed_hash,
        "created_at": datetime.now().isoformat(),
    }
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps(seed_meta, indent=2), encoding="utf-8")
    return seed_idx, seed_uuids, seed_hash


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


def _load_high_quality_extractor(script_path: Path):
    spec = importlib.util.spec_from_file_location("vcbench_hq_features", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load HQ feature script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    if not hasattr(module, "extract_features"):
        raise AttributeError(
            f"HQ script missing extract_features: {script_path}"
        )
    return module.extract_features  # type: ignore[attr-defined]


def _build_high_quality_features(
    records: list[dict[str, Any]],
    script_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        rows.append(
            {
                "industry": rec.get("industry", "") or "",
                "educations_json": json.dumps(rec.get("educations", [])),
                "jobs_json": json.dumps(rec.get("jobs", [])),
                "ipos": json.dumps(rec.get("ipos", [])),
                "acquisitions": json.dumps(rec.get("acquisitions", [])),
            }
        )
    base_df = pd.DataFrame(rows)
    extractor = _load_high_quality_extractor(script_path)
    return extractor(base_df)


def _save_high_quality_features(
    hq_df: pd.DataFrame,
    founder_ids: list[str | None],
    labels: np.ndarray,
    out_dir: Path,
) -> None:
    save_df = hq_df.copy()
    save_df.insert(0, "founder_uuid", founder_ids)
    save_df.insert(1, "success", labels)
    save_df.insert(2, "row_index", np.arange(len(save_df)))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_df.to_parquet(out_dir / "features_full.parquet", index=False)
    meta = {
        "features_no_gap": HQ_FEATURES_BASE,
        "features_with_gap": HQ_FEATURES_WITH_GAP,
        "repeat_founding_gap_impute": "0.0 for HQ-only+gap",
        "n_rows": len(save_df),
    }
    (out_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_llm_engineered_cache(
    cache_dir: Path,
    expected_rows: int,
    expected_n: int,
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
    seed_hash: str | None = None,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    current_dir = cache_dir / "current"
    cache_path = current_dir / "llm_features.parquet"
    meta_path = current_dir / "llm_features_meta.json"
    if not cache_path.exists() or not meta_path.exists():
        # Backward-compat: check legacy location
        legacy_cache = cache_dir / "llm_features.parquet"
        legacy_meta = cache_dir / "llm_features_meta.json"
        if not legacy_cache.exists() or not legacy_meta.exists():
            return None, None
        cache_path = legacy_cache
        meta_path = legacy_meta
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("n_rows") == expected_rows
            and meta.get("n_features") == expected_n
            and meta.get("model") == model
            and meta.get("providers") == providers
            and meta.get("google_model") == google_model
            and isinstance(meta.get("feature_names"), list)
            and (seed_hash is None or meta.get("seed_hash") == seed_hash)
        ):
            df = pd.read_parquet(cache_path)
            return df, list(meta.get("feature_names"))
    except Exception:
        return None, None
    return None, None


def _save_llm_engineered_cache(
    cache_dir: Path,
    df: pd.DataFrame,
    feature_names: list[str],
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
    n_features: int,
    seed_hash: str | None = None,
    rules: list[dict[str, str]] | None = None,
) -> None:
    current_dir = cache_dir / "current"
    current_dir.mkdir(parents=True, exist_ok=True)
    cache_path = current_dir / "llm_features.parquet"
    meta_path = current_dir / "llm_features_meta.json"
    df.to_parquet(cache_path, index=False)
    meta = {
        "n_rows": len(df),
        "n_features": n_features,
        "feature_names": feature_names,
        "model": model,
        "providers": providers,
        "google_model": google_model,
        "seed_hash": seed_hash,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if rules:
        rules_path = current_dir / "llm_rules.json"
        rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def _rotate_llm_engineered_cache(cache_dir: Path) -> None:
    current_dir = cache_dir / "current"
    cache_path = current_dir / "llm_features.parquet"
    meta_path = current_dir / "llm_features_meta.json"
    legacy_cache = cache_dir / "llm_features.parquet"
    legacy_meta = cache_dir / "llm_features_meta.json"
    has_current = cache_path.exists() and meta_path.exists()
    has_legacy = legacy_cache.exists() and legacy_meta.exists()
    if not has_current and not has_legacy:
        return
    old_root = cache_dir / "old"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = old_root / f"run_{stamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if has_current:
        shutil.move(str(cache_path), str(dest_dir / "llm_features.parquet"))
        shutil.move(str(meta_path), str(dest_dir / "llm_features_meta.json"))
    if has_legacy:
        shutil.move(str(legacy_cache), str(dest_dir / "llm_features_legacy.parquet"))
        shutil.move(str(legacy_meta), str(dest_dir / "llm_features_legacy_meta.json"))


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


def _standardize_continuous(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cont_names = [
        n
        for n in feature_names
        if FEATURE_REGISTRY.get(n, None) is not None
        and FEATURE_REGISTRY[n].feature_type == "continuous"
    ]
    if not cont_names:
        return train_df, test_df
    train = train_df.copy()
    test = test_df.copy()
    for name in cont_names:
        mean = float(train[name].mean())
        std = float(train[name].std())
        if std <= 0:
            std = 1.0
        train[name] = (train[name] - mean) / std
        test[name] = (test[name] - mean) / std
    return train, test


def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, pct: float) -> float:
    n = len(y_true)
    k = max(1, int(math.ceil(pct * n)))
    order = np.argsort(y_scores)[::-1]
    top_k = order[:k]
    return float(np.sum(y_true[top_k]) / k)


def _train_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    clf = LogisticRegression(max_iter=3000, random_state=random_state)
    clf.fit(X_train, y_train)
    train_scores = clf.predict_proba(X_train)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]
    return train_scores, test_scores, clf


def _train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover - import-time failure
        raise RuntimeError(
            "xgboost is required for model_type=xgboost. "
            "Install with: pip install xgboost"
        ) from exc

    params = {
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
        "random_state": random_state,
        "n_jobs": 1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    return train_scores, test_scores, model


def _apply_rule_override(scores: np.ndarray, rule_mask: np.ndarray | None) -> np.ndarray:
    if rule_mask is None:
        return scores
    if scores.shape[0] != rule_mask.shape[0]:
        raise ValueError("Rule mask length mismatch for score override.")
    adjusted = scores.copy()
    adjusted[rule_mask] = 1.0
    return adjusted


def _train_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if model_type == "xgboost":
        return _train_xgboost(X_train, y_train, X_test, random_state)
    return _train_sklearn(X_train, y_train, X_test, random_state)


def _select_threshold(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, float]:
    best_t = 0.5
    best_f = 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(
            y_true,
            (scores >= t).astype(int),
            beta=0.5,
            zero_division=0.0,  # type: ignore[arg-type]
        )
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def _report_metrics(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    threshold_tuning: str = "train",
) -> dict[str, float]:
    if threshold_tuning not in ("train", "val"):
        raise ValueError(f"Unknown threshold_tuning={threshold_tuning}")

    if threshold_tuning == "val":
        best_t, _ = _select_threshold(y_test, test_scores)
    else:
        best_t, _ = _select_threshold(y_train, train_scores)

    return _metrics_from_scores(y_test, test_scores, best_t)


def _metrics_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fnt = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0.0),  # type: ignore[arg-type]
        "recall": recall_score(y_true, y_pred, zero_division=0.0),  # type: ignore[arg-type]
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0.0),  # type: ignore[arg-type]
        "precision@1%": _precision_at_k(y_true, scores, 0.01),
        "precision@5%": _precision_at_k(y_true, scores, 0.05),
        "precision@10%": _precision_at_k(y_true, scores, 0.10),
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "fnr": fnt,
    }
    return metrics


def _format_topk(metrics: dict[str, float]) -> list[str]:
    return [
        "\n  Top-k precision:",
        f"    precision@1%={metrics['precision@1%']:.3f}",
        f"    precision@5%={metrics['precision@5%']:.3f}",
        f"    precision@10%={metrics['precision@10%']:.3f}",
    ]


def _format_mean_std(mean: float, std: float) -> str:
    if std != std:
        return f"{mean:.3f}"
    return f"{mean:.3f}+/-{std:.3f}"


def _compute_cv_splits(
    y: np.ndarray, n_splits: int, random_state: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        splits.append((train_idx, test_idx))
        fold_ids[test_idx] = fold_idx
    if (fold_ids < 0).any():
        raise RuntimeError("Failed to assign fold ids for all records.")
    return splits, fold_ids


def _cv_evaluate(
    feature_df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
    n_splits: int,
    random_state: int,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    rule_mask: np.ndarray | None = None,
    threshold_tuning: str = "oof",
) -> tuple[dict[str, float], dict[str, float]]:
    metrics_list: list[dict[str, float]] = []
    oof_scores: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    fold_scores: list[tuple[np.ndarray, np.ndarray]] = []
    if splits is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(feature_df, y))
    for train_idx, test_idx in splits:
        X_train = feature_df.iloc[train_idx].copy()
        X_test = feature_df.iloc[test_idx].copy()
        if X_train.isna().any().any() or X_test.isna().any().any():
            if args.model_type == "xgboost":
                X_train = X_train.fillna(0.0)
                X_test = X_test.fillna(0.0)
            else:
                fill_values = X_train.mean()
                X_train = X_train.fillna(fill_values)
                X_test = X_test.fillna(fill_values)
        if args.model_type == "logistic":
            X_train, X_test = _standardize_continuous(X_train, X_test, feature_names)
        train_scores, test_scores, _ = _train_model(
            args.model_type,
            X_train.values.astype(float),
            y[train_idx],
            X_test.values.astype(float),
            args.random_state,
        )
        if rule_mask is not None:
            train_scores = _apply_rule_override(train_scores, rule_mask[train_idx])
            test_scores = _apply_rule_override(test_scores, rule_mask[test_idx])
        if threshold_tuning == "oof":
            oof_scores.append(test_scores)
            oof_labels.append(y[test_idx])
            fold_scores.append((y[test_idx], test_scores))
        else:
            metrics = _report_metrics(
                y[train_idx],
                train_scores,
                y[test_idx],
                test_scores,
                threshold_tuning=threshold_tuning,
            )
            acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y[test_idx]))
            metrics["accuracy"] = acc
            metrics_list.append(metrics)

    if threshold_tuning == "oof":
        if not oof_scores:
            raise RuntimeError("OOF threshold tuning requested but no fold scores were collected.")
        all_scores = np.concatenate(oof_scores)
        all_labels = np.concatenate(oof_labels)
        oof_threshold, _ = _select_threshold(all_labels, all_scores)
        for y_fold, scores_fold in fold_scores:
            metrics = _metrics_from_scores(y_fold, scores_fold, oof_threshold)
            acc = float(np.mean((scores_fold >= metrics["threshold"]).astype(int) == y_fold))
            metrics["accuracy"] = acc
            metrics_list.append(metrics)

    keys = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f0.5",
        "accuracy",
        "precision@1%",
        "precision@5%",
        "precision@10%",
    ]
    means = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in keys}
    stds = {k: float(np.nanstd([m[k] for m in metrics_list])) for k in keys}
    return means, stds


def _write_cv_log(
    log_lines: list[str],
    args: argparse.Namespace,
    input_csv: str,
    feature_names: list[str],
    n_samples: int,
    pos_count: int,
    cv_folds: int,
    metrics_mean: dict[str, float],
    metrics_std: dict[str, float],
    log_dir: Path | None = None,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = BASE_DIR / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_log_{ts}.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    try:
        print(f"\n  Training log saved to: {log_path}")
    except OSError:
        pass

    report = {
        "dataset": input_csv,
        "random_state": args.random_state,
        "cv_folds": cv_folds,
        "n_samples": n_samples,
        "positives": pos_count,
        "features": feature_names,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
    }
    report_path = log_dir / f"run_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        print(f"  Run report saved to: {report_path}")
    except OSError:
        pass


def _train_and_log_cv(
    feature_names: list[str],
    full_df: pd.DataFrame,
    y: np.ndarray,
    args: argparse.Namespace,
    input_csv: str,
    mode_label: str,
    cv_folds: int,
    log_dir: Path | None = None,
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    rule_mask: np.ndarray | None = None,
    threshold_tuning: str = "oof",
) -> tuple[dict[str, float], dict[str, float]]:
    means, stds = _cv_evaluate(
        full_df,
        y,
        feature_names,
        args,
        cv_folds,
        args.random_state,
        splits=cv_splits,
        rule_mask=rule_mask,
        threshold_tuning=threshold_tuning,
    )
    log_lines: list[str] = []
    log_lines.append(f"Features used: {', '.join(feature_names)}")
    log_lines.append(
        f"\n[{mode_label}]   {len(feature_names)} features, CV={cv_folds} folds, "
        f"threshold_tuning={threshold_tuning}"
    )
    log_lines.append(
        "ROC-AUC={roc}  PR-AUC={pr}  Prec={prec}  Rec={rec}  F0.5={f0}  Acc={acc}".format(
            roc=_format_mean_std(means["roc_auc"], stds["roc_auc"]),
            pr=_format_mean_std(means["pr_auc"], stds["pr_auc"]),
            prec=_format_mean_std(means["precision"], stds["precision"]),
            rec=_format_mean_std(means["recall"], stds["recall"]),
            f0=_format_mean_std(means["f0.5"], stds["f0.5"]),
            acc=_format_mean_std(means["accuracy"], stds["accuracy"]),
        )
    )
    log_lines.append(
        "\n  Top-k precision:"
    )
    log_lines.append(
        f"    precision@1%={_format_mean_std(means['precision@1%'], stds['precision@1%'])}"
    )
    log_lines.append(
        f"    precision@5%={_format_mean_std(means['precision@5%'], stds['precision@5%'])}"
    )
    log_lines.append(
        f"    precision@10%={_format_mean_std(means['precision@10%'], stds['precision@10%'])}"
    )

    _write_cv_log(
        log_lines,
        args,
        input_csv,
        feature_names,
        len(y),
        int(y.sum()),
        cv_folds,
        means,
        stds,
        log_dir=log_dir,
    )
    return means, stds

def _train_and_log(
    feature_names: list[str],
    full_train: pd.DataFrame,
    full_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    input_csv: str,
    mode_label: str,
    log_dir: Path | None = None,
    rule_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], LogisticRegression]:
    # Impute missing values using training means (prevents leakage).
    if full_train.isna().any().any() or full_test.isna().any().any():
        fill_values = full_train.mean()
        full_train = full_train.fillna(fill_values)
        full_test = full_test.fillna(fill_values)

    if args.model_type == "logistic":
        full_train, full_test = _standardize_continuous(full_train, full_test, feature_names)

    X_train = full_train.values.astype(float)
    X_test = full_test.values.astype(float)

    train_scores, test_scores, model = _train_model(
        args.model_type, X_train, y_train, X_test, args.random_state
    )
    if rule_mask is not None:
        train_scores = _apply_rule_override(train_scores, rule_mask[: len(y_train)])
        test_scores = _apply_rule_override(test_scores, rule_mask[len(y_train) :])
    metrics = _report_metrics(y_train, train_scores, y_test, test_scores)
    acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
    metrics["accuracy"] = acc

    log_lines: list[str] = []
    log_lines.append(f"Features used: {', '.join(feature_names)}")
    log_lines.append(
        f"\n[{mode_label}]   {len(feature_names)} features, threshold={metrics['threshold']:.2f}"
    )
    if args.model_type == "logistic":
        coef = model.coef_[0]
        ranked = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)
        for name, c in ranked:
            sign = "+" if c >= 0 else "-"
            log_lines.append(f"  {sign}{abs(c):.3f}  {name}")
    log_lines.append(
        f"\nROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
        f"F0.5={metrics['f0.5']:.3f}  Acc={acc:.3f}  "
        f"FNR={metrics['fnr']:.3f}  TP={int(metrics['tp'])}  FN={int(metrics['fn'])}"
    )
    log_lines.extend(_format_topk(metrics))

    _write_log(
        log_lines,
        args,
        input_csv,
        feature_names,
        len(y_train),
        len(y_test),
        int(y_train.sum()),
        int(y_test.sum()),
        metrics,
        log_dir=log_dir,
    )
    return metrics, model


def _parse_training_log(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    metrics_line = None
    for line in lines:
        if line.startswith("ROC-AUC="):
            metrics_line = line
            break
    if not metrics_line:
        return None
    metrics: dict[str, float] = {}
    for part in metrics_line.split():
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            metrics[key] = float(val)
        except ValueError:
            continue
    return metrics


def _write_f05_table(rows: list[dict[str, float]], report_path: Path) -> str:
    header = "| Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        f0 = _format_mean_std(row["F0.5"], row.get("F0.5_std", float("nan")))
        roc = _format_mean_std(row["ROC-AUC"], row.get("ROC-AUC_std", float("nan")))
        pr = _format_mean_std(row["PR-AUC"], row.get("PR-AUC_std", float("nan")))
        prec = _format_mean_std(row["Prec"], row.get("Prec_std", float("nan")))
        rec = _format_mean_std(row["Rec"], row.get("Rec_std", float("nan")))
        acc = _format_mean_std(row["Acc"], row.get("Acc_std", float("nan")))
        lines.append(
            "| {name} | {f0} | {roc} | {pr} | {prec} | {rec} | {acc} |".format(
                name=row["name"],
                f0=f0,
                roc=roc,
                pr=pr,
                prec=prec,
                rec=rec,
                acc=acc,
            )
        )
    table = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# LLM Regression Summary (F0.5)\n\n" + table + "\n",
        encoding="utf-8",
    )
    return table


def _write_family_leaderboard(
    rows: list[dict[str, Any]],
    report_path: Path,
    csv_path: Path,
    cv_folds: int,
    pool_size: int,
) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    header = "| Set ID | Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append(
            "| {set_id} | {name} | {f0:.3f} | {roc:.3f} | {pr:.3f} | {prec:.3f} | {rec:.3f} | {acc:.3f} |".format(
                set_id=row.get("set_id", ""),
                name=row.get("regression", ""),
                f0=float(row.get("F0.5", float("nan"))),
                roc=float(row.get("ROC-AUC", float("nan"))),
                pr=float(row.get("PR-AUC", float("nan"))),
                prec=float(row.get("Prec", float("nan"))),
                rec=float(row.get("Rec", float("nan"))),
                acc=float(row.get("Acc", float("nan"))),
            )
        )
    leaderboard_table = "\n".join(lines)

    section = (
        "## LLM Engineered Run-Family Leaderboard\n\n"
        + leaderboard_table
        + "\n\n"
        + f"*Metrics are {cv_folds}-fold stratified CV on {pool_size} founders (seed excluded).*\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        base = report_path.read_text(encoding="utf-8").rstrip()
        report_path.write_text(base + "\n\n" + section, encoding="utf-8")
    else:
        report_path.write_text(
            "# LLM Regression Summary (F0.5)\n\n" + section, encoding="utf-8"
        )


def _refresh_family_leaderboard_from_full_csv(
    full_csv: Path,
    report_path: Path,
    csv_path: Path,
    cv_folds: int,
    pool_size: int,
) -> None:
    if not full_csv.exists():
        return
    df = pd.read_csv(full_csv)
    if df.empty:
        return
    df = df[df["set_id"].notna() & (df["set_id"].astype(str) != "")]
    df = df[df["regression"].astype(str).str.contains("LLM Engineered", na=False)]
    if df.empty:
        return
    # Normalize columns to legacy leaderboard schema
    out = pd.DataFrame(
        {
            "set_id": df["set_id"].astype(str),
            "regression": df["regression"].astype(str),
            "F0.5": df["F0.5_mean"].astype(float),
            "ROC-AUC": df["ROC-AUC_mean"].astype(float),
            "PR-AUC": df["PR-AUC_mean"].astype(float),
            "Prec": df["Prec_mean"].astype(float),
            "Rec": df["Rec_mean"].astype(float),
            "Acc": df["Acc_mean"].astype(float),
            "reasoning_experiment": df["reasoning_combo"].fillna("").astype(str),
        }
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    header = "| Set ID | Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for _, row in out.iterrows():
        lines.append(
            "| {set_id} | {name} | {f0:.3f} | {roc:.3f} | {pr:.3f} | {prec:.3f} | {rec:.3f} | {acc:.3f} |".format(
                set_id=row.get("set_id", ""),
                name=row.get("regression", ""),
                f0=float(row.get("F0.5", float("nan"))),
                roc=float(row.get("ROC-AUC", float("nan"))),
                pr=float(row.get("PR-AUC", float("nan"))),
                prec=float(row.get("Prec", float("nan"))),
                rec=float(row.get("Rec", float("nan"))),
                acc=float(row.get("Acc", float("nan"))),
            )
        )
    leaderboard_table = "\n".join(lines)
    section = (
        "## LLM Engineered Run-Family Leaderboard (from latest full results)\n\n"
        + leaderboard_table
        + "\n\n"
        + f"*Metrics are {cv_folds}-fold stratified CV on {pool_size} founders (seed excluded).*\\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        base = report_path.read_text(encoding="utf-8").rstrip()
        report_path.write_text(base + "\n\n" + section, encoding="utf-8")
    else:
        report_path.write_text(
            "# LLM Regression Summary (F0.5)\n\n" + section, encoding="utf-8"
        )


def _build_reasoning_combos(
    reasoning_df: pd.DataFrame,
    experiment_ids: list[str],
) -> dict[str, list[str]]:
    if reasoning_df is None or reasoning_df.empty:
        return {}
    numeric_cols = {
        c
        for c in reasoning_df.columns
        if c not in ("founder_uuid", "success")
        and pd.api.types.is_numeric_dtype(reasoning_df[c])
    }
    exp_to_cols: dict[str, list[str]] = {}
    for exp_id in experiment_ids:
        if not exp_id:
            continue
        cols = [c for c in reasoning_df.columns if c.startswith(f"{exp_id}_") and c in numeric_cols]
        if cols:
            exp_to_cols[exp_id] = cols
    combos: dict[str, list[str]] = {}
    exp_keys = list(exp_to_cols.keys())
    for r in range(1, len(exp_keys) + 1):
        for subset in combinations(exp_keys, r):
            combo = "+".join(subset)
            cols: list[str] = []
            for exp_id in subset:
                cols.extend(exp_to_cols.get(exp_id, []))
            if cols:
                combos[combo] = sorted(cols)
    return combos


def _filter_reasoning_combos_no_f_with_ab(
    combos: dict[str, list[str]],
) -> dict[str, list[str]]:
    filtered: dict[str, list[str]] = {}
    for combo, cols in combos.items():
        parts = set(combo.split("+"))
        if "F" in parts and ("A" in parts or "B" in parts):
            continue
        filtered[combo] = cols
    return filtered


def _load_reasoning_numeric_map(
    experiments_path: Path,
    exp_list: list[str],
) -> dict[str, list[str]]:
    data = json.loads(experiments_path.read_text(encoding="utf-8-sig"))
    exp_set = set(exp_list)
    mapping: dict[str, list[str]] = {}
    for exp in data:
        exp_id = str(exp.get("id"))
        if exp_id not in exp_set:
            continue
        keys = [f"{exp_id}_{k}" for k in exp.get("numeric_keys", [])]
        keys.append(f"{exp_id}_evidence_support_rating")
        mapping[exp_id] = keys
    return mapping


def _write_full_results_report(
    lr_table1_rows: list[dict[str, Any]],
    lr_table2_rows: list[dict[str, Any]],
    lr_table_hq_rows: list[dict[str, Any]],
    xgb_table1_rows: list[dict[str, Any]],
    xgb_table2_rows: list[dict[str, Any]],
    xgb_table_hq_rows: list[dict[str, Any]],
    full_lr_rows: list[dict[str, Any]],
    full_xgb_rows: list[dict[str, Any]],
    report_path: Path,
    csv_path: Path,
    cv_folds: int,
    pool_size: int,
    full_size: int,
) -> None:
    all_rows = (
        lr_table1_rows
        + lr_table2_rows
        + lr_table_hq_rows
        + xgb_table1_rows
        + xgb_table2_rows
        + xgb_table_hq_rows
        + full_lr_rows
        + full_xgb_rows
    )
    if not all_rows:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    def _add_csv_rows(rows: list[dict[str, Any]], model_type: str, dataset_size: int) -> None:
        for row in rows:
            csv_rows.append(
                {
                    "model_type": model_type,
                    "table": row.get("table", ""),
                    "regression": row.get("regression", ""),
                    "set_id": row.get("set_id", ""),
                    "reasoning_combo": row.get("reasoning_combo", ""),
                    "variant": row.get("variant", ""),
                    "threshold_tuning": row.get("tuning", "train"),
                    "dataset_size": dataset_size,
                    "F0.5_mean": row.get("F0.5", float("nan")),
                    "F0.5_std": row.get("F0.5_std", float("nan")),
                    "ROC-AUC_mean": row.get("ROC-AUC", float("nan")),
                    "ROC-AUC_std": row.get("ROC-AUC_std", float("nan")),
                    "PR-AUC_mean": row.get("PR-AUC", float("nan")),
                    "PR-AUC_std": row.get("PR-AUC_std", float("nan")),
                    "Prec_mean": row.get("Prec", float("nan")),
                    "Prec_std": row.get("Prec_std", float("nan")),
                    "Rec_mean": row.get("Rec", float("nan")),
                    "Rec_std": row.get("Rec_std", float("nan")),
                    "Acc_mean": row.get("Acc", float("nan")),
                    "Acc_std": row.get("Acc_std", float("nan")),
                }
            )

    _add_csv_rows(lr_table1_rows + lr_table2_rows + lr_table_hq_rows, "logistic", pool_size)
    _add_csv_rows(xgb_table1_rows + xgb_table2_rows + xgb_table_hq_rows, "xgboost", pool_size)
    _add_csv_rows(full_lr_rows, "logistic", full_size)
    _add_csv_rows(full_xgb_rows, "xgboost", full_size)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    paper_stats_dir = BASE_DIR / "docs" / "paper_stats"
    _write_experiment_top_picks(csv_path, paper_stats_dir / "experiment_top_picks.md", full_size)
    _write_paper_structure(paper_stats_dir / "paper_structure.md")

    def _fmt(row: dict[str, Any], key: str) -> str:
        return _format_mean_std(float(row.get(key, float("nan"))), float(row.get(f"{key}_std", float("nan"))))

    def _render_table(rows: list[dict[str, Any]], include_set: bool) -> str:
        if not rows:
            return "_No rows._"
        if include_set:
            header = "| Set ID | Regression | Reasoning Combo | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
            sep = "|---|---|---|---:|---:|---:|---:|---:|---:|"
        else:
            header = "| Regression | Reasoning Combo | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
            sep = "|---|---|---:|---:|---:|---:|---:|---:|"
        lines = [header, sep]
        for row in rows:
            combo = row.get("reasoning_combo", "") or "n/a"
            if include_set:
                lines.append(
                    "| {set_id} | {name} | {combo} | {f0} | {roc} | {pr} | {prec} | {rec} | {acc} |".format(
                        set_id=row.get("set_id", ""),
                        name=row.get("regression", ""),
                        combo=combo,
                        f0=_fmt(row, "F0.5"),
                        roc=_fmt(row, "ROC-AUC"),
                        pr=_fmt(row, "PR-AUC"),
                        prec=_fmt(row, "Prec"),
                        rec=_fmt(row, "Rec"),
                        acc=_fmt(row, "Acc"),
                    )
                )
            else:
                lines.append(
                    "| {name} | {combo} | {f0} | {roc} | {pr} | {prec} | {rec} | {acc} |".format(
                        name=row.get("regression", ""),
                        combo=combo,
                        f0=_fmt(row, "F0.5"),
                        roc=_fmt(row, "ROC-AUC"),
                        pr=_fmt(row, "PR-AUC"),
                        prec=_fmt(row, "Prec"),
                        rec=_fmt(row, "Rec"),
                        acc=_fmt(row, "Acc"),
                    )
                )
        return "\n".join(lines)

    def _filter_rows(rows: list[dict[str, Any]], *, variant: str | None = None, tuning: str | None = None) -> list[dict[str, Any]]:
        filtered = rows
        if variant is not None:
            filtered = [r for r in filtered if (r.get("variant", "") == variant)]
        if tuning is not None:
            filtered = [r for r in filtered if (r.get("tuning", "oof") == tuning)]
        return filtered

    def _top_k_block(
        rows: list[dict[str, Any]],
        *,
        title: str,
        include_set: bool,
        k: int = 5,
    ) -> str:
        if not rows:
            return "_No rows to summarize._"
        def _score(row: dict[str, Any]) -> float:
            try:
                val = float(row.get("F0.5", float("nan")))
                return val if not math.isnan(val) else -1e9
            except Exception:
                return -1e9

        ranked = sorted(rows, key=_score, reverse=True)[:k]
        lines = [f"**{title}**"]
        for idx, row in enumerate(ranked, 1):
            combo = row.get("reasoning_combo", "") or "n/a"
            f0 = _fmt(row, "F0.5")
            if include_set:
                lines.append(
                    f"{idx}. {row.get('set_id', '')} | {row.get('regression', '')} | {combo} | {f0}"
                )
            else:
                lines.append(
                    f"{idx}. {row.get('regression', '')} | {combo} | {f0}"
                )
        return "\n".join(lines)

    def _avg_improvement_table(
        rows: list[dict[str, Any]],
        *,
        title: str,
    ) -> str:
        if not rows:
            return "_No rows to summarize._"
        base_by_set: dict[str, float] = {}
        for row in rows:
            if row.get("regression") == "LLM Engineered Only":
                set_id = row.get("set_id", "")
                try:
                    base_by_set[set_id] = float(row.get("F0.5", float("nan")))
                except Exception:
                    continue
        delta_by_combo: dict[str, list[float]] = {}
        for row in rows:
            if row.get("regression") != "LLM Engineered + Reasoning":
                continue
            combo = row.get("reasoning_combo", "") or ""
            set_id = row.get("set_id", "")
            if not combo or set_id not in base_by_set:
                continue
            try:
                delta = float(row.get("F0.5", float("nan"))) - base_by_set[set_id]
            except Exception:
                continue
            delta_by_combo.setdefault(combo, []).append(delta)
        if not delta_by_combo:
            return "_No rows to summarize._"
        header = "| Reasoning Combo | Avg dF0.5 | Std dF0.5 | N |"
        sep = "|---|---:|---:|---:|"
        lines = [f"**{title}**", header, sep]
        for combo in sorted(delta_by_combo.keys()):
            vals = np.array(delta_by_combo[combo], dtype=float)
            avg = float(np.nanmean(vals)) if vals.size else float("nan")
            std = float(np.nanstd(vals)) if vals.size else float("nan")
            lines.append(f"| {combo} | {avg:+.3f} | {std:.3f} | {vals.size} |")
        return "\n".join(lines)

    def _legacy_summary_block(
        lr_rows: list[dict[str, Any]],
        xgb_rows: list[dict[str, Any]],
    ) -> str:
        legacy_names = {s["name"] for s in LEGACY_HUMAN_FEATURE_SETS}
        legacy_lr = [r for r in lr_rows if r.get("regression") in legacy_names]
        legacy_xgb = [r for r in xgb_rows if r.get("regression") in legacy_names]
        if not legacy_lr and not legacy_xgb:
            return "_No legacy human-only rows to summarize._"
        lines = ["**Legacy human-only (LR + XGB)**"]
        for rows, label in ((legacy_lr, "LR"), (legacy_xgb, "XGB")):
            if not rows:
                continue
            lines.append(f"- {label}:")
            for row in rows:
                lines.append(
                    f"  - {row.get('regression','')}: { _fmt(row, 'F0.5') }"
                )
        return "\n".join(lines)

    section_lines: list[str] = [
        f"## Part 1: Pool ({pool_size})",
        "",
        f"**CV folds:** {cv_folds}",
        "",
        "### Logistic Regression (oof-tuned)",
        "#### Human + Reasoning",
        _render_table(lr_table1_rows, include_set=False),
        _top_k_block(lr_table1_rows, title="Top 5 by F0.5", include_set=False),
        _legacy_summary_block(lr_table1_rows, xgb_table1_rows),
        "#### LLM-Engineered + Reasoning",
        _render_table(lr_table2_rows, include_set=True),
        _avg_improvement_table(lr_table2_rows, title="Average ΔF0.5 vs Engineered Only (LR)"),
        _top_k_block(lr_table2_rows, title="Top 5 by F0.5", include_set=True),
        "#### HQ Human + Reasoning",
        "_HQ features = Structured v2 (28 features)._\n",
        _render_table(lr_table_hq_rows, include_set=False),
        _top_k_block(lr_table_hq_rows, title="Top 5 by F0.5", include_set=False),
        "",
        "### XGBoost (oof-tuned)",
        "#### Human + Reasoning",
        _render_table(xgb_table1_rows, include_set=False),
        _top_k_block(xgb_table1_rows, title="Top 5 by F0.5", include_set=False),
        "#### LLM-Engineered + Reasoning",
        _render_table(xgb_table2_rows, include_set=True),
        _avg_improvement_table(xgb_table2_rows, title="Average ΔF0.5 vs Engineered Only (XGB)"),
        _top_k_block(xgb_table2_rows, title="Top 5 by F0.5", include_set=True),
        "#### HQ Human + Reasoning",
        "_HQ features = Structured v2 (28 features)._\n",
        _render_table(xgb_table_hq_rows, include_set=False),
        _top_k_block(xgb_table_hq_rows, title="Top 5 by F0.5", include_set=False),
        "",
        f"## Part 2: Full ({full_size})",
        "### HQ + Reasoning (no rule layer)",
        "#### Logistic (oof-tuned)",
        _render_table(_filter_rows(full_lr_rows, variant="full_hq_no_rule", tuning="oof"), include_set=False),
        _top_k_block(_filter_rows(full_lr_rows, variant="full_hq_no_rule", tuning="oof"), title="Top 5 by F0.5", include_set=False),
        "#### XGBoost (oof-tuned)",
        _render_table(_filter_rows(full_xgb_rows, variant="full_hq_no_rule", tuning="oof"), include_set=False),
        _top_k_block(_filter_rows(full_xgb_rows, variant="full_hq_no_rule", tuning="oof"), title="Top 5 by F0.5", include_set=False),
        "",
        "### Full Mirror + Reasoning (rule layer)",
        "#### Logistic (oof-tuned)",
        _render_table(_filter_rows(full_lr_rows, variant="full_mirror_rule", tuning="oof"), include_set=False),
        _top_k_block(_filter_rows(full_lr_rows, variant="full_mirror_rule", tuning="oof"), title="Top 5 by F0.5", include_set=False),
        "#### XGBoost (oof-tuned)",
        _render_table(_filter_rows(full_xgb_rows, variant="full_mirror_rule", tuning="oof"), include_set=False),
        _top_k_block(_filter_rows(full_xgb_rows, variant="full_mirror_rule", tuning="oof"), title="Top 5 by F0.5", include_set=False),
        "",
        f"*Metrics are {cv_folds}-fold stratified CV. Pool excludes seed_100; full uses 4,500 founders.*",
    ]

    section = "\n".join(section_lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# LLM Regression Summary (F0.5)\n\n" + section + "\n", encoding="utf-8")


def _write_experiment_top_picks(csv_path: Path, report_path: Path, full_size: int) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    df = df[(df["dataset_size"] == full_size) & (df["threshold_tuning"] == "oof")].copy()
    if df.empty:
        return

    df["reasoning_combo"] = df["reasoning_combo"].fillna("")
    variant_map = {
        "full_hq_no_rule": "HQ (no rule)",
        "full_mirror_rule": "Mirror (rule)",
    }
    df = df[df["variant"].isin(variant_map.keys())].copy()
    if df.empty:
        return
    df["variant_label"] = df["variant"].map(variant_map)

    combos = {
        "HQ Only": "",
        "A": "A",
        "A+E": "A+E",
        "A+F": "A+F",
        "A+D+E+F": "A+D+E+F",
        "A+B+C+D+E+F": "A+B+C+D+E+F",
    }

    def _fmt(mean: float, std: float) -> str:
        if math.isnan(mean) or math.isnan(std):
            return "—"
        return f"{mean:.3f}+/-{std:.3f}"

    rows = []
    for label, combo in combos.items():
        row = {"Experiment": label}
        for model in ["logistic", "xgboost"]:
            for variant_label in ["HQ (no rule)", "Mirror (rule)"]:
                key = f"{model.upper()} | {variant_label}"
                sub = df[
                    (df["model_type"] == model)
                    & (df["variant_label"] == variant_label)
                    & (df["reasoning_combo"] == combo)
                ]
                if sub.empty:
                    row[key] = "—"
                else:
                    row[key] = _fmt(float(sub.iloc[0]["F0.5_mean"]), float(sub.iloc[0]["F0.5_std"]))
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = [
        "Experiment",
        "LOGISTIC | HQ (no rule)",
        "LOGISTIC | Mirror (rule)",
        "XGBOOST | HQ (no rule)",
        "XGBOOST | Mirror (rule)",
    ]
    out = out[cols]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    header = "| Experiment | LOGISTIC | HQ (no rule) | LOGISTIC | Mirror (rule) | XGBOOST | HQ (no rule) | XGBOOST | Mirror (rule) |"
    sep = "|---|---:|---:|---:|---:|"
    lines = [header, sep]
    for _, row in out.iterrows():
        lines.append(
            "| {exp} | {l_hq} | {l_m} | {x_hq} | {x_m} |".format(
                exp=row["Experiment"],
                l_hq=row["LOGISTIC | HQ (no rule)"],
                l_m=row["LOGISTIC | Mirror (rule)"],
                x_hq=row["XGBOOST | HQ (no rule)"],
                x_m=row["XGBOOST | Mirror (rule)"],
            )
        )
    report_path.write_text(
        "\n".join(
            [
                "# Experiment Top Picks (Full 4,500 | OOF-tuned)",
                "",
                *lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_paper_structure(report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Paper Structure Notes",
                "",
                "We evaluate a minimal reasoning baseline (A), a small additive extension (A+E), a rubric-style extension (A+F), a richer multi-signal model (ADEF), and the maximal combination (ABCDEF). This provides a graded complexity spectrum while avoiding post-hoc selection.",
                "",
                "## Part 1: Pool experiments (4,400)",
                "- Human-only baselines and Human + reasoning combos.",
                "- LLM-engineered baselines and LLM-engineered + reasoning combos.",
                "- Reported for both Logistic Regression and XGBoost.",
                "",
                "## Part 2: HQ mirror experiments (4,500)",
                "- HQ features + reasoning additions on the full dataset.",
                "- Full mirror (rule layer) + reasoning additions on the full dataset.",
                "- Reported for both Logistic Regression and XGBoost.",
                "",
                "## Test-set plan (private)",
                "- HQ Only (baseline)",
                "- HQ + A (minimal reasoning)",
                "- HQ + A+E (incremental improvement)",
                "- HQ + A+F (stronger gain, more complexity)",
                "- HQ + A+D+E+F (best-performing complex version)",
                "- HQ + A+B+C+D+E+F (only if added complexity appears to help)",
                "",
                "## Private test addendum",
                "- Run 3x human-only feature sets with XGBoost.",
                "- Run top 3 LLM-engineered sets with XGBoost.",
                "- Run top 3 LLM-engineered + A+E with XGBoost.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_reasoning_xgb_oof_report(
    hq_rows: list[dict[str, Any]],
    mirror_rows: list[dict[str, Any]],
    report_path: Path,
    csv_path: Path,
    cv_folds: int,
    full_size: int,
) -> None:
    def _fmt(row: dict[str, Any], key: str) -> str:
        return _format_mean_std(float(row.get(key, float("nan"))), float(row.get(f"{key}_std", float("nan"))))

    def _render_table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "_No rows._"
        header = "| Reasoning Combo | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
        sep = "|---|---:|---:|---:|---:|---:|---:|"
        lines = [header, sep]
        for row in rows:
            combo = row.get("reasoning_combo", "") or "n/a"
            lines.append(
                "| {combo} | {f0} | {roc} | {pr} | {prec} | {rec} | {acc} |".format(
                    combo=combo,
                    f0=_fmt(row, "F0.5"),
                    roc=_fmt(row, "ROC-AUC"),
                    pr=_fmt(row, "PR-AUC"),
                    prec=_fmt(row, "Prec"),
                    rec=_fmt(row, "Rec"),
                    acc=_fmt(row, "Acc"),
                )
            )
        return "\n".join(lines)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    for variant, rows in (("hq_no_rule", hq_rows), ("full_mirror_rule", mirror_rows)):
        for row in rows:
            csv_rows.append(
                {
                    "variant": variant,
                    "reasoning_combo": row.get("reasoning_combo", ""),
                    "F0.5_mean": row.get("F0.5", float("nan")),
                    "F0.5_std": row.get("F0.5_std", float("nan")),
                    "ROC-AUC_mean": row.get("ROC-AUC", float("nan")),
                    "ROC-AUC_std": row.get("ROC-AUC_std", float("nan")),
                    "PR-AUC_mean": row.get("PR-AUC", float("nan")),
                    "PR-AUC_std": row.get("PR-AUC_std", float("nan")),
                    "Prec_mean": row.get("Prec", float("nan")),
                    "Prec_std": row.get("Prec_std", float("nan")),
                    "Rec_mean": row.get("Rec", float("nan")),
                    "Rec_std": row.get("Rec_std", float("nan")),
                    "Acc_mean": row.get("Acc", float("nan")),
                    "Acc_std": row.get("Acc_std", float("nan")),
                    "threshold_tuning": row.get("tuning", "oof"),
                    "cv_folds": cv_folds,
                    "dataset_size": full_size,
                }
            )
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    section_lines = [
        "# XGBoost Reasoning Sweep (OOF-tuned)",
        "",
        f"**Dataset:** full ({full_size})",
        f"**CV folds:** {cv_folds}",
        "",
        "## HQ + Reasoning (no rule layer)",
        _render_table(hq_rows),
        "",
        "## Full Mirror + Reasoning (rule layer)",
        _render_table(mirror_rows),
        "",
        "*All rows are XGBoost with OOF-tuned thresholds.*",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(section_lines) + "\n", encoding="utf-8")


def _write_snapshot_combined_report() -> None:
    snapshot_dir = BASE_DIR / "docs" / "post_CV_implementation_before_recalculating_features"
    report_path = BASE_DIR / "docs" / "llm_regression_report.md"
    if not snapshot_dir.exists() or not report_path.exists():
        return
    combined_path = snapshot_dir / "post_cv_combined_report.md"
    text = report_path.read_text(encoding="utf-8")
    combined = "# Post-CV Summary (5 Fits + Family Leaderboard)\n\n" + text.strip() + "\n"
    combined_path.write_text(combined, encoding="utf-8")
    for entry in snapshot_dir.iterdir():
        if entry.is_file() and entry.name != combined_path.name:
            entry.unlink()


def _write_run_snapshot(label: str) -> Path:
    docs_dir = BASE_DIR / "docs"
    snapshot_root = docs_dir / "run_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    if not safe_label:
        safe_label = "run"
    snapshot_dir = snapshot_root / f"{ts}_{safe_label}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    candidate_names = [
        "llm_regression_report.md",
        "llm_full_results.csv",
        "llm_engineered_family_leaderboard.md",
        "llm_engineered_family_leaderboard.csv",
        "llm_engineered_family_leaderboard_meta.json",
        "feature_weight_summary.md",
    ]
    copied: list[str] = []
    for name in candidate_names:
        src = docs_dir / name
        if src.exists():
            shutil.copy2(src, snapshot_dir / name)
            copied.append(name)
    manifest = {
        "label": label,
        "timestamp": ts,
        "files": copied,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return snapshot_dir

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VCBench in-depth pipeline.")
    p.add_argument("--dataset", choices=["sample", "full"], default="sample")
    p.add_argument("--input_csv", default="", help="Optional override CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--model_type",
        choices=["logistic", "xgboost"],
        default="logistic",
        help="Model type: logistic (default) or xgboost.",
    )
    p.add_argument(
        "--cv_folds",
        type=int,
        default=None,
        help="Optional override for CV folds (e.g., 5).",
    )
    p.add_argument(
        "--feature_set",
        choices=sorted(FEATURE_SETS.keys()),
        default="base_plus_custom",
    )
    p.add_argument(
        "--features",
        default="",
        help="Optional comma-separated custom feature names (overrides feature_set)",
    )
    p.add_argument(
        "--human_feature_source",
        choices=["baseline", "high_quality"],
        default=None,
        help="Select human feature source: baseline or high_quality (CLI overrides config).",
    )
    p.add_argument(
        "--feature_config",
        default=str(CONFIG_DIR / "features.json"),
        help="Optional JSON file with {\"features\": [...]} to select custom features.",
    )
    p.add_argument(
        "--extract_only",
        action="store_true",
        help="Extract features and save Parquet, then exit before training.",
    )
    p.add_argument(
        "--snapshot_label",
        default="",
        help="If set, save a run snapshot with this label.",
    )
    p.add_argument(
        "--snapshot_only",
        action="store_true",
        help="Only write a snapshot (requires --snapshot_label) and exit.",
    )
    p.add_argument(
        "--run_profile",
        choices=["full", "xgb_mirror", "reasoning_xgb_oof"],
        default="full",
        help="Run profile: full (default), xgb_mirror (HQ+reasoning only), or reasoning_xgb_oof.",
    )
    p.add_argument(
        "--dry_validate",
        action="store_true",
        help="Perform validation checks and exit without API calls.",
    )
    p.add_argument(
        "--mode",
        choices=["human", "llm", "reasoning", "hybrid"],
        default="human",
        help="human=base+custom, llm=LLM only, reasoning=LLM reasoning only, hybrid=base+custom+LLM+reasoning",
    )
    p.add_argument(
        "--llm_features",
        action="store_true",
        help="(Deprecated) Use --mode llm or --mode hybrid instead.",
    )
    p.add_argument("--llm_model", default="gpt-4.1-nano")
    p.add_argument("--llm_n_features", type=int, default=8)
    p.add_argument(
        "--llm_reasoning",
        action="store_true",
        help="Enable LLM reasoning features (can be combined with other modes).",
    )
    p.add_argument(
        "--llm_reasoning_core_prompt",
        default=str(PROMPT_DIR / "core_prompt.txt"),
        help="Path to core prompt template for LLM reasoning features.",
    )
    p.add_argument(
        "--llm_reasoning_experiments",
        default=str(CONFIG_DIR / "experiments.json"),
        help="Path to experiments JSON for LLM reasoning features.",
    )
    p.add_argument(
        "--llm_reasoning_sequential",
        default="",
        help="Comma-separated list of sequential LLM reasoning experiments.",
    )
    p.add_argument(
        "--llm_reasoning_combined",
        default="",
        help="Comma-separated list of combined LLM reasoning experiments.",
    )
    p.add_argument(
        "--llm_reasoning_dataset_size",
        default="full",
        help="Dataset size for reasoning features: full, 200, 400, 1000.",
    )
    p.add_argument(
        "--llm_reasoning_batch_size",
        type=int,
        default=20,
        help="Batch size for LLM reasoning (default 20).",
    )
    p.add_argument(
        "--llm_reasoning_dry_run",
        action="store_true",
        help="Run LLM reasoning without API calls (mock outputs).",
    )
    p.add_argument(
        "--llm_reasoning_dry_run_fast",
        action="store_true",
        help="Run a fast dry-run (small subset, no sleep, no training).",
    )
    p.add_argument(
        "--reasoning_seed_only",
        action="store_true",
        help="Generate seed_100 reasoning for full_current and exit.",
    )
    p.add_argument(
        "--llm_sweep_repeats",
        type=int,
        default=10,
        help="Repeat each LLM sweep n_rules value this many times.",
    )
    p.add_argument(
        "--llm_sweep_range",
        default="1-20",
        help="Range for LLM sweep when --llm_sweep is set (default: 1-20).",
    )
    p.add_argument(
        "--llm_sweep_seed_holdout_pct",
        type=float,
        default=0.2,
        help="Holdout fraction of seed_100 for LLM rule validation during sweep (default: 0.2).",
    )
    p.add_argument(
        "--llm_retry_attempts",
        type=int,
        default=2,
        help="Retry LLM generation this many times on failure during sweep.",
    )
    p.add_argument(
        "--llm_retry_sleep",
        type=float,
        default=10.0,
        help="Seconds to sleep between LLM generation retries.",
    )
    p.add_argument(
        "--llm_timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each LLM generation call during sweep.",
    )
    p.add_argument(
        "--llm_sweep_start_rule",
        type=int,
        default=1,
        help="Start sweep at this n_rules value (default: 1).",
    )
    p.add_argument(
        "--llm_sweep_start_repeat",
        type=int,
        default=1,
        help="Start sweep at this repeat index for the start rule (default: 1).",
    )
    p.add_argument(
        "--llm_sweep",
        action="store_true",
        help=(
            "Sweep LLM n_rules over a range (default 1-15) in LLM-only mode."
        ),
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
    p.add_argument(
        "--output_parquet",
        default="",
        help="Optional output parquet path for extracted features.",
    )
    return p.parse_args()


def _resolve_input_csv(dataset: str, override: str) -> str:
    if override:
        return override
    base = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit"
    if dataset == "sample":
        return str(base / "vcbench_final_public_sample100.csv")
    return str(base / "vcbench_final_public.csv")


def _parse_sweep_range(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return list(range(1, 16))
    if "-" in spec:
        parts = spec.split("-", 1)
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if start <= 0 or end <= 0:
                raise ValueError
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))
        except Exception:
            return list(range(1, 16))
    # comma-separated list
    vals = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = int(piece)
            if v > 0:
                vals.append(v)
        except Exception:
            continue
    return sorted(set(vals)) if vals else list(range(1, 21))


def _select_dataset(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    dataset_size: str,
    random_state: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if dataset_size == "full":
        return records, labels
    try:
        n = int(dataset_size)
    except Exception as exc:
        raise ValueError(f"Invalid dataset_size: {dataset_size}") from exc
    n = max(1, min(n, len(records)))
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(records), size=n, replace=False)
    return [records[i] for i in idx], labels[idx]


def _load_env_if_present() -> None:
    # Load .env from parent of Project_folder if present (one level above this file's parent).
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from think_reason_learn.core import _config as trl_config
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            val = val.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = val
            if key in ("OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY"):
                if hasattr(trl_config, "settings") and not getattr(trl_config.settings, key, ""):
                    setattr(trl_config.settings, key, val)
        # Reset LLM singleton so it re-reads updated settings
        from think_reason_learn.core.llms._ask import LLM
        from think_reason_learn.core._singleton import SingletonMeta
        SingletonMeta._instances.pop(LLM, None)
        import think_reason_learn.core.llms as trl_llms
        trl_llms.llm = trl_llms.LLM()
    except Exception:
        # Best-effort only; do not crash if .env is malformed.
        return


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    rs = args.random_state
    input_csv = _resolve_input_csv(args.dataset, args.input_csv)

    # Load config early for reasoning paths/options
    selected_features: list[str] = []
    cfg_use_llm: bool | None = None
    cfg_llm_n: int | None = None
    cfg_use_llm_reasoning: bool | None = None
    cfg_llm_reasoning_features: list[str] | None = None
    cfg_llm_reasoning_experiments: list[str] | None = None
    cfg_llm_reasoning_mode: str | None = None
    cfg_llm_reasoning_sequential: list[str] | None = None
    cfg_llm_reasoning_combined: list[str] | None = None
    cfg_llm_temperature: float | None = None
    cfg_llm_reasoning_dataset_size: str | None = None
    cfg_llm_reasoning_prompts_path: str | None = None
    cfg_llm_reasoning_core_prompt_path: str | None = None
    cfg_llm_reasoning_experiments_path: str | None = None
    cfg_llm_reasoning_dry_run: bool | None = None
    cfg_llm_reasoning_dry_run_fast: bool | None = None
    cfg_llm_providers: dict[str, bool] | None = None
    cfg_llm_google_model: str | None = None
    cfg_llm_reasoning_batch_size: int | None = None
    cfg_llm_reasoning_log_every: int | None = None
    cfg_llm_reasoning_concurrency: int | None = None
    cfg_llm_reasoning_rate_limit_fallback_concurrency: int | None = None
    cfg_llm_reasoning_rate_limit_fallback_windows: int | None = None
    cfg_llm_reasoning_rate_limit_fallback_sequence: list[int] | None = None
    cfg_llm_reasoning_inline_repair: bool | None = None
    cfg_llm_reasoning_inline_repair_max_attempts: int | None = None
    cfg_llm_reasoning_repair_nan: bool | None = None
    cfg_llm_reasoning_repair_existing: bool | None = None
    cfg_llm_engineered_for_reasoning: bool | None = None
    cfg_human_feature_source: str | None = None
    cfg_llm_reasoning_split_batches: bool | None = None
    cfg_llm_reasoning_batch_within_folds: bool | None = None
    cfg_llm_engineered_cache: bool | None = None
    cfg_llm_engineered_freeze: bool | None = None
    cfg_llm_engineered_force_recompute: bool | None = None
    cfg_llm_engineered_run_family: bool | None = None
    cfg_llm_engineered_run_family_size: int | None = None
    cfg_llm_engineered_run_family_n: int | None = None
    cfg_llm_engineered_run_family_id: str | None = None
    cfg_llm_engineered_family_allow_seed_mismatch: bool | None = None
    cfg_llm_engineered_seed_size: int | None = None
    cfg_cv_folds: int | None = None
    cfg_cv_use_fixed_folds: bool | None = None
    cfg_cv_folds_path: str | None = None
    cfg_model_type: str | None = None
    cfg_llm_sweep_range: str | None = None
    cfg_llm_sweep_repeats: int | None = None
    cfg_llm_sweep_seed_holdout_pct: float | None = None
    cfg_path = Path(args.feature_config) if args.feature_config else None
    if cfg_path is not None and not cfg_path.is_absolute() and not cfg_path.exists():
        alt_cfg = CONFIG_DIR / cfg_path
        if alt_cfg.exists():
            cfg_path = alt_cfg
    if cfg_path is not None and cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        selected_features = [f for f in data.get("features", []) if isinstance(f, str)]
        if "use_llm" in data:
            cfg_use_llm = bool(data.get("use_llm"))
        if "llm_n_features" in data:
            try:
                cfg_llm_n = int(data.get("llm_n_features"))
            except Exception:
                cfg_llm_n = None
        if "use_llm_reasoning" in data:
            cfg_use_llm_reasoning = bool(data.get("use_llm_reasoning"))
        if isinstance(data.get("llm_reasoning_features"), list):
            cfg_llm_reasoning_features = [
                f for f in data.get("llm_reasoning_features", []) if isinstance(f, str)
            ]
        if isinstance(data.get("llm_reasoning_dataset_size"), str):
            cfg_llm_reasoning_dataset_size = data.get("llm_reasoning_dataset_size")
        if isinstance(data.get("llm_reasoning_prompts_path"), str):
            cfg_llm_reasoning_prompts_path = data.get("llm_reasoning_prompts_path")
        if isinstance(data.get("llm_reasoning_core_prompt_path"), str):
            cfg_llm_reasoning_core_prompt_path = data.get("llm_reasoning_core_prompt_path")
        if isinstance(data.get("llm_reasoning_experiments_path"), str):
            cfg_llm_reasoning_experiments_path = data.get("llm_reasoning_experiments_path")
        if isinstance(data.get("llm_reasoning_experiments"), list):
            cfg_llm_reasoning_experiments = [
                e for e in data.get("llm_reasoning_experiments", []) if isinstance(e, str)
            ]
        if isinstance(data.get("human_feature_source"), str):
            cfg_human_feature_source = data.get("human_feature_source")
        if isinstance(data.get("llm_reasoning_mode"), str):
            cfg_llm_reasoning_mode = data.get("llm_reasoning_mode")
        if isinstance(data.get("llm_reasoning_sequential"), list):
            cfg_llm_reasoning_sequential = [
                e for e in data.get("llm_reasoning_sequential", []) if isinstance(e, str)
            ]
        if isinstance(data.get("llm_reasoning_combined"), list):
            cfg_llm_reasoning_combined = [
                e for e in data.get("llm_reasoning_combined", []) if isinstance(e, str)
            ]
        if "llm_reasoning_dry_run" in data:
            cfg_llm_reasoning_dry_run = bool(data.get("llm_reasoning_dry_run"))
        if "llm_reasoning_dry_run_fast" in data:
            cfg_llm_reasoning_dry_run_fast = bool(data.get("llm_reasoning_dry_run_fast"))
        if isinstance(data.get("llm_providers"), dict):
            cfg_llm_providers = {
                "openai": bool(data.get("llm_providers", {}).get("openai", False)),
                "google": bool(data.get("llm_providers", {}).get("google", False)),
            }
        if isinstance(data.get("llm_google_model"), str):
            cfg_llm_google_model = data.get("llm_google_model")
        if isinstance(data.get("llm_sweep_range"), str):
            cfg_llm_sweep_range = data.get("llm_sweep_range")
        if "llm_sweep_repeats" in data:
            try:
                cfg_llm_sweep_repeats = int(data.get("llm_sweep_repeats"))
            except Exception:
                cfg_llm_sweep_repeats = None
        if "llm_sweep_seed_holdout_pct" in data:
            try:
                cfg_llm_sweep_seed_holdout_pct = float(data.get("llm_sweep_seed_holdout_pct"))
            except Exception:
                cfg_llm_sweep_seed_holdout_pct = None
        if isinstance(data.get("llm_reasoning_batch_size"), int):
            cfg_llm_reasoning_batch_size = data.get("llm_reasoning_batch_size")
        if isinstance(data.get("llm_reasoning_log_every"), int):
            cfg_llm_reasoning_log_every = data.get("llm_reasoning_log_every")
        if isinstance(data.get("llm_reasoning_concurrency"), int):
            cfg_llm_reasoning_concurrency = data.get("llm_reasoning_concurrency")
        if isinstance(data.get("llm_reasoning_rate_limit_fallback_concurrency"), int):
            cfg_llm_reasoning_rate_limit_fallback_concurrency = data.get(
                "llm_reasoning_rate_limit_fallback_concurrency"
            )
        if isinstance(data.get("llm_reasoning_rate_limit_fallback_windows"), int):
            cfg_llm_reasoning_rate_limit_fallback_windows = data.get(
                "llm_reasoning_rate_limit_fallback_windows"
            )
        if isinstance(data.get("llm_reasoning_rate_limit_fallback_sequence"), list):
            seq: list[int] = []
            for item in data.get("llm_reasoning_rate_limit_fallback_sequence", []):
                try:
                    seq.append(int(item))
                except Exception:
                    continue
            if seq:
                cfg_llm_reasoning_rate_limit_fallback_sequence = seq
        if "llm_reasoning_inline_repair" in data:
            cfg_llm_reasoning_inline_repair = bool(data.get("llm_reasoning_inline_repair"))
        if "llm_reasoning_inline_repair_max_attempts" in data:
            try:
                cfg_llm_reasoning_inline_repair_max_attempts = int(
                    data.get("llm_reasoning_inline_repair_max_attempts")
                )
            except Exception:
                cfg_llm_reasoning_inline_repair_max_attempts = None
        if "llm_reasoning_repair_nan" in data:
            cfg_llm_reasoning_repair_nan = bool(data.get("llm_reasoning_repair_nan"))
        if "llm_reasoning_repair_existing" in data:
            cfg_llm_reasoning_repair_existing = bool(data.get("llm_reasoning_repair_existing"))
        if "llm_reasoning_split_batches" in data:
            cfg_llm_reasoning_split_batches = bool(data.get("llm_reasoning_split_batches"))
        if "llm_reasoning_batch_within_folds" in data:
            cfg_llm_reasoning_batch_within_folds = bool(
                data.get("llm_reasoning_batch_within_folds")
            )
        if "llm_engineered_for_reasoning" in data:
            cfg_llm_engineered_for_reasoning = bool(data.get("llm_engineered_for_reasoning"))
        if "llm_engineered_cache" in data:
            cfg_llm_engineered_cache = bool(data.get("llm_engineered_cache"))
        if "llm_engineered_freeze" in data:
            cfg_llm_engineered_freeze = bool(data.get("llm_engineered_freeze"))
        if "llm_engineered_force_recompute" in data:
            cfg_llm_engineered_force_recompute = bool(data.get("llm_engineered_force_recompute"))
        if "llm_engineered_run_family" in data:
            cfg_llm_engineered_run_family = bool(data.get("llm_engineered_run_family"))
        if "llm_engineered_run_family_size" in data:
            try:
                cfg_llm_engineered_run_family_size = int(data.get("llm_engineered_run_family_size"))
            except Exception:
                cfg_llm_engineered_run_family_size = None
        if "llm_engineered_run_family_n_features" in data:
            try:
                cfg_llm_engineered_run_family_n = int(data.get("llm_engineered_run_family_n_features"))
            except Exception:
                cfg_llm_engineered_run_family_n = None
        if "llm_engineered_run_family_id" in data:
            cfg_llm_engineered_run_family_id = str(data.get("llm_engineered_run_family_id"))
        if "llm_engineered_family_allow_seed_mismatch" in data:
            cfg_llm_engineered_family_allow_seed_mismatch = bool(
                data.get("llm_engineered_family_allow_seed_mismatch")
            )
        if "llm_engineered_seed_size" in data:
            try:
                cfg_llm_engineered_seed_size = int(data.get("llm_engineered_seed_size"))
            except Exception:
                cfg_llm_engineered_seed_size = None
        if "cv_folds" in data:
            try:
                cfg_cv_folds = int(data.get("cv_folds"))
            except Exception:
                cfg_cv_folds = None
        if "cv_use_fixed_folds" in data:
            cfg_cv_use_fixed_folds = bool(data.get("cv_use_fixed_folds"))
        if "cv_folds_path" in data:
            cfg_cv_folds_path = str(data.get("cv_folds_path") or "")
        if "model_type" in data:
            cfg_model_type = str(data.get("model_type") or "") or None
        if "llm_temperature" in data:
            try:
                cfg_llm_temperature = float(data.get("llm_temperature"))
            except Exception:
                cfg_llm_temperature = None

    def _parse_comma_list(value: str | None) -> list[str]:
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]

    if args.llm_reasoning_sequential:
        cfg_llm_reasoning_sequential = _parse_comma_list(args.llm_reasoning_sequential)
    if args.llm_reasoning_combined:
        cfg_llm_reasoning_combined = _parse_comma_list(args.llm_reasoning_combined)
    # Set up run-level logging early for full traceability
    run_log = None
    log_root = None
    use_llm_reasoning_early = (
        args.llm_reasoning
        or (cfg_use_llm_reasoning is True)
        or (args.mode in ("reasoning", "hybrid"))
    )
    if use_llm_reasoning_early:
        log_root = BASE_DIR / "logging" / f"llm_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_root.mkdir(parents=True, exist_ok=True)
        run_log = log_root / "run_log.txt"
        run_log.write_text("run_log started\n", encoding="utf-8")
        global RUN_LOG_PATH
        RUN_LOG_PATH = run_log

    def _log_run(msg: str) -> None:
        if run_log is not None:
            with run_log.open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    log_lines: list[str] = []
    sweep_terminal_log: Path | None = None

    def _log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)
        if sweep_terminal_log is not None:
            sweep_terminal_log.parent.mkdir(parents=True, exist_ok=True)
            with sweep_terminal_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    def _log_exception(err: BaseException) -> None:
        _log_run("Unhandled exception:\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__)))

    mode = args.mode
    _log(f"\n{'=' * 60}")
    _log("  VCBench In-Depth Pipeline")
    _log(f"{'=' * 60}\n")
    _log(f"  Dataset: {args.dataset}")
    _log(f"  Input CSV: {input_csv}")

    _log_run(f"Loading dataset from: {input_csv}")
    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        rs,
    )
    _log_run(f"Loaded records: {len(records)} labels: {len(labels)}")

    # Resolve LLM reasoning settings early to allow dataset sizing
    reasoning_core_prompt_path = (
        Path(cfg_llm_reasoning_core_prompt_path)
        if cfg_llm_reasoning_core_prompt_path
        else Path(args.llm_reasoning_core_prompt)
    )
    reasoning_experiments_path = (
        Path(cfg_llm_reasoning_experiments_path)
        if cfg_llm_reasoning_experiments_path
        else Path(args.llm_reasoning_experiments)
    )
    if not reasoning_core_prompt_path.is_absolute():
        reasoning_core_prompt_path = PROMPT_DIR / reasoning_core_prompt_path
    if not reasoning_experiments_path.is_absolute():
        reasoning_experiments_path = CONFIG_DIR / reasoning_experiments_path
    llm_providers = cfg_llm_providers or {"openai": True, "google": False}
    llm_google_model = cfg_llm_google_model
    llm_reasoning_batch_size = (
        cfg_llm_reasoning_batch_size
        if cfg_llm_reasoning_batch_size is not None
        else args.llm_reasoning_batch_size
    )
    llm_reasoning_log_every = cfg_llm_reasoning_log_every if cfg_llm_reasoning_log_every is not None else 10
    llm_reasoning_concurrency = cfg_llm_reasoning_concurrency if cfg_llm_reasoning_concurrency is not None else 1
    llm_reasoning_rate_limit_fallback_concurrency = (
        cfg_llm_reasoning_rate_limit_fallback_concurrency
        if cfg_llm_reasoning_rate_limit_fallback_concurrency is not None
        else 5
    )
    llm_reasoning_rate_limit_fallback_windows = (
        cfg_llm_reasoning_rate_limit_fallback_windows
        if cfg_llm_reasoning_rate_limit_fallback_windows is not None
        else 1
    )
    llm_reasoning_rate_limit_fallback_sequence = (
        cfg_llm_reasoning_rate_limit_fallback_sequence
        if cfg_llm_reasoning_rate_limit_fallback_sequence is not None
        else [8, 6, 4, 2, 1]
    )
    llm_reasoning_inline_repair = (
        cfg_llm_reasoning_inline_repair
        if cfg_llm_reasoning_inline_repair is not None
        else True
    )
    llm_reasoning_inline_repair_max_attempts = (
        cfg_llm_reasoning_inline_repair_max_attempts
        if cfg_llm_reasoning_inline_repair_max_attempts is not None
        else 1
    )
    llm_reasoning_repair_nan = cfg_llm_reasoning_repair_nan if cfg_llm_reasoning_repair_nan is not None else True
    llm_reasoning_repair_existing = cfg_llm_reasoning_repair_existing if cfg_llm_reasoning_repair_existing is not None else False
    llm_engineered_for_reasoning = cfg_llm_engineered_for_reasoning if cfg_llm_engineered_for_reasoning is not None else False
    llm_reasoning_split_batches = (
        cfg_llm_reasoning_split_batches if cfg_llm_reasoning_split_batches is not None else True
    )
    llm_reasoning_batch_within_folds = (
        cfg_llm_reasoning_batch_within_folds
        if cfg_llm_reasoning_batch_within_folds is not None
        else llm_reasoning_split_batches
    )
    llm_engineered_cache = cfg_llm_engineered_cache if cfg_llm_engineered_cache is not None else True
    llm_engineered_freeze = cfg_llm_engineered_freeze if cfg_llm_engineered_freeze is not None else False
    llm_engineered_force_recompute = (
        cfg_llm_engineered_force_recompute if cfg_llm_engineered_force_recompute is not None else False
    )
    llm_engineered_run_family = cfg_llm_engineered_run_family if cfg_llm_engineered_run_family is not None else False
    llm_engineered_run_family_size = cfg_llm_engineered_run_family_size if cfg_llm_engineered_run_family_size is not None else 10
    llm_engineered_run_family_n = cfg_llm_engineered_run_family_n if cfg_llm_engineered_run_family_n is not None else None
    llm_engineered_run_family_id = cfg_llm_engineered_run_family_id
    llm_engineered_seed_size = cfg_llm_engineered_seed_size if cfg_llm_engineered_seed_size is not None else 100
    llm_engineered_family_allow_seed_mismatch = (
        cfg_llm_engineered_family_allow_seed_mismatch
        if cfg_llm_engineered_family_allow_seed_mismatch is not None
        else False
    )
    human_feature_source = (
        args.human_feature_source
        if args.human_feature_source is not None
        else (cfg_human_feature_source if cfg_human_feature_source is not None else "baseline")
    )
    if human_feature_source not in ("baseline", "high_quality"):
        human_feature_source = "baseline"
    llm_engineered_rotated = False
    cache_dir = BASE_DIR / "features_storage" / "llm_engineered"
    cv_folds = (
        int(args.cv_folds)
        if args.cv_folds is not None
        else (cfg_cv_folds if cfg_cv_folds is not None else 5)
    )
    cv_use_fixed_folds = cfg_cv_use_fixed_folds if cfg_cv_use_fixed_folds is not None else True
    cv_folds_path = cfg_cv_folds_path if cfg_cv_folds_path else ""
    llm_temperature = cfg_llm_temperature if cfg_llm_temperature is not None else 0.0
    llm_reasoning_dry_run = bool(args.llm_reasoning_dry_run) or bool(cfg_llm_reasoning_dry_run)
    llm_reasoning_dry_run_fast = bool(args.llm_reasoning_dry_run_fast) or bool(cfg_llm_reasoning_dry_run_fast)
    reasoning_dataset_size = (
        cfg_llm_reasoning_dataset_size
        if cfg_llm_reasoning_dataset_size is not None
        else args.llm_reasoning_dataset_size
    )
    use_llm_reasoning = (
        args.llm_reasoning
        or (cfg_use_llm_reasoning is True)
        or (args.mode in ("reasoning", "hybrid"))
    )
    _log_run(f"Use LLM reasoning: {use_llm_reasoning}")
    _log_run(
        "Config flags: "
        f"llm_engineered_run_family={llm_engineered_run_family} "
        f"llm_engineered_freeze={llm_engineered_freeze} "
        f"llm_engineered_force_recompute={llm_engineered_force_recompute} "
        f"llm_engineered_run_family_id={llm_engineered_run_family_id}"
    )
    _log(f"  Human feature source: {human_feature_source}")
    _log_run(f"Human feature source: {human_feature_source}")
    model_type = (
        args.model_type
        if args.model_type != "logistic" or cfg_model_type is None
        else cfg_model_type
    )
    if model_type not in ("logistic", "xgboost"):
        model_type = "logistic"
    args.model_type = model_type
    _log(f"  Model type: {model_type}")
    _log_run(f"Model type: {model_type}")

    if args.snapshot_only:
        if not args.snapshot_label:
            raise ValueError("--snapshot_only requires --snapshot_label.")
        snapshot_dir = _write_run_snapshot(args.snapshot_label)
        _log(f"Snapshot written to: {snapshot_dir}")
        return

    run_profile = args.run_profile
    dry_validate = bool(args.dry_validate)
    _log_run(f"Run profile: {run_profile}")
    _log(f"  Run profile: {run_profile}")
    skip_llm_engineered = run_profile in ("xgb_mirror", "reasoning_xgb_oof")
    if run_profile == "reasoning_xgb_oof":
        use_llm_reasoning = True
        reasoning_dataset_size = "full"
    if llm_reasoning_dry_run_fast:
        llm_reasoning_dry_run = True
        use_llm_reasoning = True
        if mode == "human":
            mode = "reasoning"

    if use_llm_reasoning and args.dataset == "sample":
        raise RuntimeError("LLM reasoning does not allow dataset=sample. Use full or a size override.")

    if use_llm_reasoning and reasoning_dataset_size != "full":
        records, labels = _select_dataset(records, labels, reasoning_dataset_size, rs)

    all_records = records
    all_labels = labels
    all_founder_ids = [r.get("founder_uuid") for r in all_records]
    seed_path = (
        BASE_DIR
        / "features_storage"
        / "llm_engineered"
        / f"seed_{llm_engineered_seed_size}.json"
    )
    seed_idx, seed_uuids, seed_hash = _load_or_create_seed(
        all_records,
        all_labels,
        all_founder_ids,
        llm_engineered_seed_size,
        rs,
        seed_path,
    )
    seed_recs = [all_records[i] for i in seed_idx]
    seed_labels = all_labels[seed_idx]
    pool_mask = np.ones(len(all_records), dtype=bool)
    pool_mask[seed_idx] = False
    pool_idx = np.where(pool_mask)[0]
    records = [all_records[i] for i in pool_idx]
    labels = all_labels[pool_idx]
    founder_ids = [all_founder_ids[i] for i in pool_idx]
    _log(f"  Seed set: {len(seed_idx)} founders (excluded from training)")
    _log(f"  Pool set: {len(records)} founders for CV")
    _log_run(f"Seed size={len(seed_idx)} pool size={len(records)}")
    _log_run(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")

    folds_path = resolve_folds_path(BASE_DIR, cv_folds, rs, cv_folds_path)
    cv_splits, fold_ids, folds_path = load_or_create_folds(
        founder_ids=founder_ids,
        labels=labels,
        cv_folds=cv_folds,
        random_state=rs,
        folds_path=folds_path,
        dataset_label=input_csv,
        use_fixed=cv_use_fixed_folds,
    )
    fold_sizes = np.bincount(fold_ids, minlength=cv_folds)
    fold_sizes_str = ", ".join(str(int(s)) for s in fold_sizes)
    _log(f"  CV folds: {cv_folds} (sizes: {fold_sizes_str})")
    _log_run(f"CV fold sizes: {fold_sizes_str}")
    _log_run(f"CV fold cache: {folds_path}")

    folds_path_full = (
        BASE_DIR
        / "features_storage"
        / "cv_folds"
        / f"folds_k{cv_folds}_seed{rs}_full.json"
    )
    cv_splits_full, fold_ids_full, folds_path_full = load_or_create_folds(
        founder_ids=all_founder_ids,
        labels=all_labels,
        cv_folds=cv_folds,
        random_state=rs,
        folds_path=folds_path_full,
        dataset_label=f"{input_csv}_full",
        use_fixed=cv_use_fixed_folds,
    )
    _validate_run_state(
        expected_pool=len(records),
        expected_full=len(all_records),
        pool_cache=folds_path,
        full_cache=folds_path_full,
    )

    if llm_reasoning_dry_run_fast:
        _log("  Dry-run fast enabled: generating reasoning features only.")
        output_dir = BASE_DIR / "features_storage" / "llm_reasoning"
        meta_path = output_dir / f"llm_reasoning_dry_run_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config = ReasoningConfig(
            model=args.llm_model,
            dataset_size=reasoning_dataset_size,
            random_state=rs,
            core_prompt_path=reasoning_core_prompt_path,
            experiments_path=reasoning_experiments_path,
            providers=llm_providers,
            google_model=llm_google_model,
            batch_size=llm_reasoning_batch_size,
            concurrency=llm_reasoning_concurrency,
            experiments=cfg_llm_reasoning_experiments,
            dry_run=True,
            dry_run_fast=True,
            repair_nan=llm_reasoning_repair_nan,
            repair_existing=llm_reasoning_repair_existing,
            rate_limit_fallback_concurrency=llm_reasoning_rate_limit_fallback_concurrency,
            rate_limit_fallback_windows=llm_reasoning_rate_limit_fallback_windows,
            rate_limit_fallback_sequence=llm_reasoning_rate_limit_fallback_sequence,
            inline_repair=llm_reasoning_inline_repair,
            inline_repair_max_attempts=llm_reasoning_inline_repair_max_attempts,
        )
        reasoning_df, all_reasoning_names = generate_reasoning_features(
            records=records,
            labels=labels,
            config=config,
            output_dir=output_dir,
            metadata_path=meta_path,
        )
        _log(f"  Dry-run fast output rows: {len(reasoning_df)}")
        _log("  Dry-run fast complete; skipping training.")
        _write_log(
            log_lines,
            args,
            input_csv,
            all_reasoning_names,
            len(records),
            0,
            int(labels.sum()),
            0,
            {"threshold": None},
            log_dir=_log_dir_for_mode("reasoning", True, True),
        )
        return

    # Always build baseline human features for pooled evaluation.
    base_script = Path(args.base_script)
    extract_base = _load_base_feature_extractor(base_script)
    base_all_baseline = pd.DataFrame([extract_base(r) for r in records])
    base_feature_names_baseline = list(base_all_baseline.columns)

    if selected_features:
        base_selected = [f for f in selected_features if f in base_feature_names_baseline]
        custom_features_baseline = [f for f in selected_features if f in FEATURE_REGISTRY]
        unknown = [
            f
            for f in selected_features
            if f not in base_feature_names_baseline and f not in FEATURE_REGISTRY
        ]
        if unknown:
            print(f"  WARNING: Unknown features in config: {', '.join(unknown)}")
    elif args.features.strip():
        base_selected = base_feature_names_baseline
        custom_features_baseline = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        base_selected = base_feature_names_baseline
        custom_features_baseline = FEATURE_SETS[args.feature_set]

    # Apply baseline selection if provided
    if selected_features:
        base_all_baseline = base_all_baseline[base_selected]
        base_feature_names_baseline = list(base_all_baseline.columns)

    custom_all_baseline = (
        _custom_feature_df(records, custom_features_baseline)
        if custom_features_baseline
        else pd.DataFrame(index=range(len(records)))
    )

    # High-quality feature extraction for pooled evaluation.
    hq_full_no_gap: pd.DataFrame | None = None
    hq_full_with_gap: pd.DataFrame | None = None
    hq_df_full: pd.DataFrame | None = None
    hq_script = (
        PROJECT_ROOT
        / "High_Quality_human_features"
        / "features"
        / "extract_structured.py"
    )
    hq_df_full = _build_high_quality_features(records, hq_script)
    missing = [f for f in HQ_FEATURES_WITH_GAP if f not in hq_df_full.columns]
    if missing:
        raise RuntimeError(
            "High-quality feature extraction missing columns: "
            + ", ".join(missing)
        )
    hq_full_no_gap = hq_df_full[HQ_FEATURES_BASE].copy()
    hq_full_with_gap = hq_df_full[HQ_FEATURES_WITH_GAP].copy()
    hq_full_with_gap["repeat_founding_gap"] = hq_full_with_gap["repeat_founding_gap"].fillna(0.0)
    hq_out_dir = BASE_DIR / "features_storage" / "human_high_quality"
    _save_high_quality_features(
        hq_df_full[HQ_FEATURES_WITH_GAP],
        founder_ids,
        labels,
        hq_out_dir,
    )

    # Select the active human feature source for single-mode runs.
    if human_feature_source == "high_quality":
        base_all = hq_full_no_gap
        base_feature_names = list(base_all.columns)
        custom_features = []
        custom_all = pd.DataFrame(index=range(len(records)))
        selected_features = []
    else:
        base_all = base_all_baseline
        base_feature_names = list(base_all.columns)
        custom_features = list(custom_features_baseline)
        custom_all = custom_all_baseline.copy()

    llm_feature_names: list[str] = []
    llm_all = pd.DataFrame(index=range(len(records)))

    reasoning_feature_names: list[str] = []
    reasoning_all = pd.DataFrame(index=range(len(records)))
    llm_engineered_feature_names: list[str] = []

    mode = args.mode
    if human_feature_source != "high_quality":
        if cfg_use_llm is True and mode == "human":
            mode = "hybrid"
        if cfg_use_llm_reasoning is True and mode == "human":
            mode = "hybrid"
    else:
        mode = "human"

    _log(f"  Mode: {mode}\n")

    llm_n = cfg_llm_n if cfg_llm_n is not None else args.llm_n_features
    sweep_enabled = args.llm_sweep
    sweep_range_spec = cfg_llm_sweep_range or args.llm_sweep_range
    sweep_rules = _parse_sweep_range(sweep_range_spec)
    sweep_repeats = max(1, int(cfg_llm_sweep_repeats if cfg_llm_sweep_repeats is not None else args.llm_sweep_repeats))
    sweep_seed_holdout_pct = (
        cfg_llm_sweep_seed_holdout_pct
        if cfg_llm_sweep_seed_holdout_pct is not None
        else args.llm_sweep_seed_holdout_pct
    )
    use_llm = mode in ("llm", "hybrid") or args.llm_features
    use_llm_reasoning = use_llm_reasoning or (mode in ("reasoning", "hybrid"))
    exp_list_for_combo = [
        e
        for e in (cfg_llm_reasoning_sequential or cfg_llm_reasoning_experiments or [])
        if isinstance(e, str)
    ]
    if run_profile == "reasoning_xgb_oof":
        exp_list_for_combo = ["A", "B", "C", "D", "E", "F"]
        use_llm_reasoning = True

    def _dry_validate_xgb_mirror(exp_list: list[str]) -> None:
        _log_run("DRY_VALIDATE: xgb_mirror started")
        use_root = BASE_DIR / "features_storage" / "llm_reasoning" / "currently_in_use"
        use_path = use_root / "llm_reasoning_full.parquet"
        if use_path.exists():
            df_use = pd.read_parquet(use_path)
            if len(df_use) == len(all_records) and len(records) != len(all_records):
                df_use = df_use.iloc[pool_idx].reset_index(drop=True)
            num = df_use.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            nan_count = int(num.isna().any(axis=1).sum()) if not num.empty else 0
            _log_run(f"DRY_VALIDATE: currently_in_use rows={len(df_use)} nan_rows={nan_count}")
        else:
            _log_run("DRY_VALIDATE: currently_in_use missing.")
        full_root = BASE_DIR / "features_storage" / "llm_reasoning" / "full_current"
        full_path = full_root / "llm_reasoning_full.parquet"
        if not full_path.exists():
            _log_run("DRY_VALIDATE: full_current missing. Seed_100 reasoning would be generated.")
        else:
            df_full = pd.read_parquet(full_path)
            num_full = df_full.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            nan_full = int(num_full.isna().any(axis=1).sum()) if not num_full.empty else 0
            _log_run(f"DRY_VALIDATE: full_current rows={len(df_full)} nan_rows={nan_full}")
        _log_run(f"DRY_VALIDATE: experiments={exp_list}")
        _log_run("DRY_VALIDATE: xgb_mirror complete")

    if run_profile == "xgb_mirror" and dry_validate:
        _dry_validate_xgb_mirror(exp_list_for_combo)
        return
    if run_profile in ("xgb_mirror", "reasoning_xgb_oof"):
        use_llm = False
        args.llm_features = False
        llm_engineered_run_family = False
        llm_engineered_cache = False
        llm_engineered_for_reasoning = False
    if use_llm and not skip_llm_engineered:
        _log(f"  OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
        if sweep_enabled:
            sweep_dir = BASE_DIR / "training_logs" / f"llm_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            features_storage = BASE_DIR / "features_storage" / "llm_engineered"
            sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_features_root = features_storage / "sweeps" / f"sweep_{sweep_id}"
            sweep_features_root.mkdir(parents=True, exist_ok=True)
            sweep_terminal_log = sweep_dir / "terminal_log.txt"
            _log("Sweep terminal log started.")
            summary_rows = []
            # Seed-only holdout for rule generation (do not use pool data here).
            holdout_pct = float(sweep_seed_holdout_pct)
            if holdout_pct <= 0 or holdout_pct >= 1:
                holdout_pct = 0.2
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=holdout_pct, random_state=rs
            )
            seed_idx_arr = np.arange(len(seed_recs))
            seed_train_idx, seed_holdout_idx = next(sss.split(seed_idx_arr, seed_labels))
            seed_train_recs = [seed_recs[i] for i in seed_train_idx]
            seed_train_labels = seed_labels[seed_train_idx]
            seed_holdout_recs = [seed_recs[i] for i in seed_holdout_idx]
            start_rule = max(1, int(args.llm_sweep_start_rule))
            start_repeat = max(1, int(args.llm_sweep_start_repeat))
            for n_rules in sweep_rules:
                n_dir = sweep_features_root / f"n_rules_{n_rules}"
                n_dir.mkdir(parents=True, exist_ok=True)
                features_path = n_dir / "features.parquet"
                meta_path = n_dir / "meta.json"
                existing_df: pd.DataFrame | None = None
                existing_meta: dict[str, Any] = {}
                if features_path.exists():
                    existing_df = pd.read_parquet(features_path)
                    if meta_path.exists():
                        try:
                            existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        except Exception:
                            existing_meta = {}
                for repeat_idx in range(1, sweep_repeats + 1):
                    if n_rules < start_rule:
                        continue
                    if n_rules == start_rule and repeat_idx < start_repeat:
                        continue
                    set_id = f"set_{repeat_idx:02d}"
                    # Skip generation if this set already exists in consolidated parquet
                    if existing_df is not None and "set_id" in existing_df.columns:
                        if (existing_df["set_id"] == set_id).any():
                            llm_feature_names = []
                            if isinstance(existing_meta.get("set_features"), dict):
                                llm_feature_names = list(
                                    existing_meta.get("set_features", {}).get(set_id, [])
                                )
                            if not llm_feature_names:
                                # Fall back to using all non-id columns if metadata is missing
                                llm_feature_names = [
                                    c
                                    for c in existing_df.columns
                                    if c not in ("founder_uuid", "success", "set_id")
                                ]
                            llm_set = existing_df[existing_df["set_id"] == set_id].reset_index(drop=True)
                            llm_all = llm_set[llm_feature_names]
                            feature_names = llm_feature_names
                            mode_label = "LLM Only"
                            metrics_mean, metrics_std = _cv_evaluate(
                                llm_all,
                                labels,
                                feature_names,
                                args,
                                cv_folds,
                                rs,
                                splits=cv_splits,
                            )
                            row = {
                                "run_id": f"run_{n_rules}_{repeat_idx}",
                                "repeat": repeat_idx,
                                "set_id": set_id,
                                "n_rules": n_rules,
                                "resultant_features": len(feature_names),
                                "roc_auc": metrics_mean["roc_auc"],
                                "roc_auc_std": metrics_std["roc_auc"],
                                "f0.5": metrics_mean["f0.5"],
                                "f0.5_std": metrics_std["f0.5"],
                                "precision": metrics_mean["precision"],
                                "precision_std": metrics_std["precision"],
                                "recall": metrics_mean["recall"],
                                "recall_std": metrics_std["recall"],
                                "accuracy": metrics_mean["accuracy"],
                                "accuracy_std": metrics_std["accuracy"],
                                "precision@1%": metrics_mean["precision@1%"],
                                "precision@1%_std": metrics_std["precision@1%"],
                                "precision@5%": metrics_mean["precision@5%"],
                                "precision@5%_std": metrics_std["precision@5%"],
                                "precision@10%": metrics_mean["precision@10%"],
                                "precision@10%_std": metrics_std["precision@10%"],
                                "cv_folds": cv_folds,
                                "cv_folds_path": str(folds_path),
                            }
                            summary_rows.append(row)
                            continue
                    _log(
                        f"\n  Generating {n_rules} LLM features with {args.llm_model}... (repeat {repeat_idx}/{sweep_repeats})"
                    )
                    attempt = 0
                    while True:
                        try:
                            llm_all, _, _, llm_feature_names, _ = asyncio.run(
                                asyncio.wait_for(
                                    generate_llm_features(
                                        train_recs=seed_train_recs,
                                        y_train=seed_train_labels,
                                        test_recs=seed_holdout_recs,
                                        model=args.llm_model,
                                        n_features=n_rules,
                                        all_recs=records,
                                        providers=llm_providers,
                                        google_model=llm_google_model,
                                    ),
                                    timeout=float(args.llm_timeout),
                                )
                            )
                            break
                        except Exception:
                            attempt += 1
                            err = traceback.format_exc()
                            _log(f"\n  ERROR during LLM generation (attempt {attempt}):")
                            _log(err)
                            if attempt > max(0, int(args.llm_retry_attempts)):
                                _log("  Exceeded retry attempts. Skipping this run.")
                                llm_all = pd.DataFrame(index=range(len(records)))
                                llm_feature_names = []
                                break
                            _log(f"  Retrying in {args.llm_retry_sleep:.1f}s...")
                            import time as _time
                            _time.sleep(max(0.0, float(args.llm_retry_sleep)))
                    if not llm_feature_names:
                        continue

                    full_all = llm_all
                    feature_names = llm_feature_names
                    mode_label = "LLM Only"

                    metrics_mean, metrics_std = _cv_evaluate(
                        full_all,
                        labels,
                        feature_names,
                        args,
                        cv_folds,
                        rs,
                        splits=cv_splits,
                    )

                    model = LogisticRegression(max_iter=1000, random_state=rs)
                    model.fit(full_all.values.astype(float), labels)

                    # Save features into consolidated parquet for this n_rules
                    llm_df = pd.concat(
                        [
                            pd.Series([r.get("founder_uuid") for r in records], name="founder_uuid"),
                            pd.Series(labels, name="success"),
                            pd.Series([set_id] * len(records), name="set_id"),
                            full_all,
                        ],
                        axis=1,
                    )
                    if existing_df is None:
                        combined = llm_df
                    else:
                        combined = pd.concat([existing_df, llm_df], ignore_index=True, sort=False)
                    combined.to_parquet(features_path, index=False)
                    existing_df = combined
                    existing_meta.setdefault("set_features", {})
                    existing_meta["set_features"][set_id] = list(feature_names)
                    existing_meta["model"] = args.llm_model
                    existing_meta["providers"] = llm_providers
                    existing_meta["google_model"] = llm_google_model
                    existing_meta["seed_hash"] = seed_hash
                    existing_meta["seed_size"] = llm_engineered_seed_size
                    existing_meta["timestamp"] = datetime.now().isoformat()
                    meta_path.write_text(json.dumps(existing_meta, indent=2), encoding="utf-8")

                    # Write run log
                    run_id = f"run_{n_rules}_{repeat_idx}"
                    run_lines = []
                    run_lines.append(f"Requested n_rules: {n_rules}")
                    run_lines.append(f"Repeat: {repeat_idx}")
                    run_lines.append(f"Set ID: {set_id}")
                    run_lines.append(f"Resultant features: {len(feature_names)}")
                    run_lines.append(f"Features used: {', '.join(feature_names)}")
                    run_lines.append(f"[{mode_label}] {len(feature_names)} features, CV={cv_folds} folds")
                    run_lines.append("\nWeights:")
                    run_lines.append(f"{'feature':<40} {'coef':>8}")
                    run_lines.append("-" * 50)
                    for name, c in sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
                        run_lines.append(f"{name:<40} {c:>8.3f}")
                    run_lines.append(
                        "ROC-AUC={roc} PR-AUC={pr} Prec={prec} Rec={rec} F0.5={f0} Acc={acc}".format(
                            roc=_format_mean_std(metrics_mean["roc_auc"], metrics_std["roc_auc"]),
                            pr=_format_mean_std(metrics_mean["pr_auc"], metrics_std["pr_auc"]),
                            prec=_format_mean_std(metrics_mean["precision"], metrics_std["precision"]),
                            rec=_format_mean_std(metrics_mean["recall"], metrics_std["recall"]),
                            f0=_format_mean_std(metrics_mean["f0.5"], metrics_std["f0.5"]),
                            acc=_format_mean_std(metrics_mean["accuracy"], metrics_std["accuracy"]),
                        )
                    )
                    run_lines.append("\n  Top-k precision:")
                    run_lines.append(
                        f"    precision@1%={_format_mean_std(metrics_mean['precision@1%'], metrics_std['precision@1%'])}"
                    )
                    run_lines.append(
                        f"    precision@5%={_format_mean_std(metrics_mean['precision@5%'], metrics_std['precision@5%'])}"
                    )
                    run_lines.append(
                        f"    precision@10%={_format_mean_std(metrics_mean['precision@10%'], metrics_std['precision@10%'])}"
                    )
                    (sweep_dir / f"{run_id}.txt").write_text("\n".join(run_lines), encoding="utf-8")

                    row = {
                        "run_id": run_id,
                        "repeat": repeat_idx,
                        "set_id": set_id,
                        "n_rules": n_rules,
                        "resultant_features": len(feature_names),
                        "roc_auc": metrics_mean["roc_auc"],
                        "roc_auc_std": metrics_std["roc_auc"],
                        "f0.5": metrics_mean["f0.5"],
                        "f0.5_std": metrics_std["f0.5"],
                        "precision": metrics_mean["precision"],
                        "precision_std": metrics_std["precision"],
                        "recall": metrics_mean["recall"],
                        "recall_std": metrics_std["recall"],
                        "accuracy": metrics_mean["accuracy"],
                        "accuracy_std": metrics_std["accuracy"],
                        "precision@1%": metrics_mean["precision@1%"],
                        "precision@1%_std": metrics_std["precision@1%"],
                        "precision@5%": metrics_mean["precision@5%"],
                        "precision@5%_std": metrics_std["precision@5%"],
                        "precision@10%": metrics_mean["precision@10%"],
                        "precision@10%_std": metrics_std["precision@10%"],
                        "cv_folds": cv_folds,
                        "cv_folds_path": str(folds_path),
                    }
                    for name, c in sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
                        row[f"w_{name}"] = c
                    summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(sweep_dir / "llm_sweep_summary.csv", index=False)
            if not summary_df.empty:
                agg = (
                    summary_df.groupby("n_rules")
                    .agg(
                        roc_auc=("roc_auc", "mean"),
                        roc_auc_std=("roc_auc", "std"),
                        f0_5=("f0.5", "mean"),
                        f0_5_std=("f0.5", "std"),
                        precision=("precision", "mean"),
                        precision_std=("precision", "std"),
                        recall=("recall", "mean"),
                        recall_std=("recall", "std"),
                        accuracy=("accuracy", "mean"),
                        accuracy_std=("accuracy", "std"),
                        precision_at_1=("precision@1%", "mean"),
                        precision_at_1_std=("precision@1%", "std"),
                        precision_at_5=("precision@5%", "mean"),
                        precision_at_5_std=("precision@5%", "std"),
                        precision_at_10=("precision@10%", "mean"),
                        precision_at_10_std=("precision@10%", "std"),
                        resultant_features=("resultant_features", "mean"),
                    )
                    .reset_index()
                )
                agg.to_csv(sweep_dir / "llm_sweep_summary_agg.csv", index=False)
            try:
                import matplotlib.pyplot as plt  # type: ignore
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                x = summary_df["n_rules"]
                axes[0, 0].scatter(x, summary_df["roc_auc"])
                axes[0, 0].set_title("ROC-AUC")
                axes[0, 1].scatter(x, summary_df["f0.5"])
                axes[0, 1].set_title("F0.5")
                axes[1, 0].scatter(x, summary_df["recall"], label="Recall")
                if "precision" in summary_df.columns:
                    axes[1, 0].scatter(x, summary_df["precision"], label="Precision")
                    axes[1, 0].legend()
                axes[1, 0].set_title("Recall + Precision")
                axes[1, 1].scatter(x, summary_df["accuracy"])
                axes[1, 1].set_title("Accuracy")
                for ax in axes.flat:
                    ax.set_xlabel("n_rules")
                plt.tight_layout()
                plt.savefig(sweep_dir / "llm_sweep_metrics.png", dpi=150)
                plt.close(fig)
            except Exception:
                pass

            return
        else:
            cache_dir = BASE_DIR / "features_storage" / "llm_engineered"
            if llm_engineered_force_recompute and llm_engineered_cache and not llm_engineered_rotated:
                _rotate_llm_engineered_cache(cache_dir)
                llm_engineered_rotated = True
                _log("  Rotated existing LLM-engineered cache to old/.")
            llm_cached_all = None
            llm_cached_names = None
            if llm_engineered_cache:
                llm_cached_all, llm_cached_names = _load_llm_engineered_cache(
                    cache_dir=cache_dir,
                    expected_rows=len(records),
                    expected_n=llm_n,
                    model=args.llm_model,
                    providers=llm_providers,
                    google_model=llm_google_model,
                    seed_hash=seed_hash,
                )
            if llm_cached_all is not None and llm_cached_names is not None:
                llm_all = llm_cached_all
                llm_feature_names = llm_cached_names
                _log("  Loaded cached LLM-engineered features.")
            else:
                _log(f"\n  Generating {llm_n} LLM features with {args.llm_model}...")
                llm_all, _, _, llm_feature_names, llm_rules = asyncio.run(
                    generate_llm_features(
                        train_recs=seed_recs,
                        y_train=seed_labels,
                        test_recs=records,
                        model=args.llm_model,
                        n_features=llm_n,
                        all_recs=records,
                        providers=llm_providers,
                        google_model=llm_google_model,
                    )
                )
                if llm_engineered_cache:
                    _save_llm_engineered_cache(
                        cache_dir=cache_dir,
                        df=llm_all,
                        feature_names=llm_feature_names,
                        model=args.llm_model,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        n_features=llm_n,
                        seed_hash=seed_hash,
                        rules=llm_rules,
                    )

    def _load_currently_in_use_df() -> pd.DataFrame | None:
        use_root = BASE_DIR / "features_storage" / "llm_reasoning" / "currently_in_use"
        use_path = use_root / "llm_reasoning_full.parquet"
        if not use_path.exists():
            return None
        df = pd.read_parquet(use_path)
        if len(df) == len(all_records) and len(records) != len(all_records):
            df = df.iloc[pool_idx].reset_index(drop=True)
        return df

    def _combo_is_subset(combo_label: str, allowed: set[str]) -> bool:
        if not combo_label:
            return False
        if combo_label.lower() in {"n/a", "na", "none"}:
            return False
        parts = [p for p in combo_label.replace("_", "+").split("+") if p]
        return bool(parts) and set(parts).issubset(allowed)

    if use_llm_reasoning:
        output_root = BASE_DIR / "features_storage" / "llm_reasoning"
        current_root = output_root / "current"
        runs_root = output_root / "runs"
        if log_root is None:
            log_root = BASE_DIR / "logging" / f"llm_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_root.mkdir(parents=True, exist_ok=True)
            run_log = log_root / "run_log.txt"

        def _load_existing_reasoning(
            exp_list: list[str] | None,
        ) -> tuple[pd.DataFrame | None, list[str]]:
            # Prefer current/ if present
            current_path = current_root / "llm_reasoning_full.parquet"
            if current_path.exists():
                df = pd.read_parquet(current_path)
                if len(df) == len(records):
                    feature_cols = [c for c in df.columns if c not in ("founder_uuid", "success")]
                    return df, feature_cols
                if len(df) == len(all_records) and len(records) != len(all_records):
                    df = df.iloc[pool_idx].reset_index(drop=True)
                    feature_cols = [c for c in df.columns if c not in ("founder_uuid", "success")]
                    return df, feature_cols
            # Fallback: most recent run in runs/
            if runs_root.exists():
                run_dirs = sorted(
                    [p for p in runs_root.glob("run_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for run_dir in run_dirs:
                    run_path = run_dir / "llm_reasoning_full.parquet"
                    if run_path.exists():
                        df = pd.read_parquet(run_path)
                        if len(df) == len(records):
                            feature_cols = [c for c in df.columns if c not in ("founder_uuid", "success")]
                            return df, feature_cols
                        if len(df) == len(all_records) and len(records) != len(all_records):
                            df = df.iloc[pool_idx].reset_index(drop=True)
                            feature_cols = [c for c in df.columns if c not in ("founder_uuid", "success")]
                            return df, feature_cols
            return None, []

        def _sync_reasoning_current(run_dir: Path, exp_list: list[str] | None) -> None:
            import time as _time
            current_root.mkdir(parents=True, exist_ok=True)
            src_full = run_dir / "llm_reasoning_full.parquet"
            if src_full.exists():
                shutil.copy2(src_full, current_root / "llm_reasoning_full.parquet")
            # Copy per-experiment outputs if present
            exp_src = run_dir / "experiments"
            exp_dst = current_root / "experiments"
            if exp_src.exists():
                if exp_dst.exists():
                    # Best-effort safe remove; avoid crashing on transient locks (e.g. OneDrive).
                    removed = False
                    for attempt in range(3):
                        try:
                            shutil.rmtree(exp_dst)
                            removed = True
                            break
                        except PermissionError:
                            _time.sleep(0.5 * (attempt + 1))
                        except FileNotFoundError:
                            removed = True
                            break
                    if not removed:
                        try:
                            backup = exp_dst.with_name(
                                exp_dst.name + f"_stale_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            )
                            shutil.move(str(exp_dst), str(backup))
                            removed = True
                        except Exception:
                            # Leave the old folder and continue to avoid crashing the run.
                            removed = False
                    if not removed:
                        # Skip experiment sync but still keep full parquet + manifest.
                        exp_src = None
                if exp_src is not None:
                    shutil.copytree(exp_src, exp_dst)
            # Write a minimal manifest
            manifest = {
                "source_run": run_dir.name,
                "experiments": exp_list or [],
                "timestamp": datetime.now().isoformat(),
            }
            (current_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        def _update_currently_in_use(
            exp_list: list[str],
            selected_runs: dict[str, Path] | None = None,
        ) -> None:
            runs_root = BASE_DIR / "features_storage" / "llm_reasoning" / "runs"
            use_root = BASE_DIR / "features_storage" / "llm_reasoning" / "currently_in_use"
            use_root.mkdir(parents=True, exist_ok=True)
            exp_root = use_root / "experiments"
            exp_root.mkdir(parents=True, exist_ok=True)
            join_key = "founder_uuid"
            if selected_runs is None:
                selected_runs = {}
                for exp_id in exp_list:
                    candidates = sorted(
                        [p for p in runs_root.glob(f"run_{exp_id}_*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for candidate in candidates:
                        run_path = candidate / "llm_reasoning_full.parquet"
                        if run_path.exists():
                            selected_runs[exp_id] = candidate
                            break
            if not selected_runs:
                return
            # Determine join key (founder_uuid vs row_index)
            try:
                sample_run = next(iter(selected_runs.values()))
                sample_df = pd.read_parquet(sample_run / "llm_reasoning_full.parquet")
                if "founder_uuid" not in sample_df.columns or not sample_df["founder_uuid"].notna().any():
                    join_key = "row_index"
            except Exception:
                join_key = "row_index"
            # Copy per-experiment parquets into currently_in_use/experiments
            for exp_id, run_dir in selected_runs.items():
                src = run_dir / "llm_reasoning_full.parquet"
                dst_dir = exp_root / exp_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / "llm_reasoning_full.parquet"
                shutil.copy2(src, dst)
            # Build merged full parquet with disambiguated global keys
            merged = None
            run_map: dict[str, str] = {}
            for exp_id, run_dir in selected_runs.items():
                run_map[exp_id] = run_dir.name
                df = pd.read_parquet(run_dir / "llm_reasoning_full.parquet")
                if len(df) == len(all_records) and len(records) != len(all_records):
                    df = df.iloc[pool_idx].reset_index(drop=True)
                if join_key == "row_index":
                    df = df.reset_index(drop=True)
                    df["row_index"] = df.index
                    if "founder_uuid" in df.columns:
                        df = df.drop(columns=["founder_uuid"])
                if "evidence_support_rating" in df.columns and f"{exp_id}_evidence_support_rating" in df.columns:
                    df = df.drop(columns=["evidence_support_rating"])
                df = df.drop_duplicates(join_key)
                cols = []
                for c in df.columns:
                    if c in ("founder_uuid", "success", join_key):
                        cols.append(c)
                        continue
                    if c.startswith(f"{exp_id}_"):
                        cols.append(c)
                    else:
                        cols.append(f"{exp_id}_{c}")
                df.columns = cols
                df = df.set_index(join_key)
                if merged is None:
                    merged = df
                else:
                    merged = merged.join(df.drop(columns=["success"], errors="ignore"), how="inner")
            if merged is not None:
                if "success" not in merged.columns:
                    raise RuntimeError("Merged reasoning frame missing success column.")
                merged = merged.reset_index()
                merged.to_parquet(use_root / "llm_reasoning_full.parquet", index=False)
            manifest = {
                "timestamp": datetime.now().isoformat(),
                "experiments": exp_list,
                "runs": run_map,
            }
            (use_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        def _reasoning_has_nans(path: Path) -> bool:
            if not path.exists():
                return False
            df = pd.read_parquet(path)
            num = df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            if num.empty:
                return False
            return bool(num.isna().any(axis=1).any())

        def _reasoning_run_valid(run_dir: Path, exp_id: str) -> bool:
            run_path = run_dir / "llm_reasoning_full.parquet"
            if not run_path.exists():
                return False
            df = pd.read_parquet(run_path)
            if len(df) == len(all_records) and len(records) != len(all_records):
                df = df.iloc[pool_idx].reset_index(drop=True)
            if not any(c.startswith(f"{exp_id}_") for c in df.columns):
                return False
            num = df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            if num.empty:
                return True
            return not num.isna().any(axis=1).any()

        def _select_latest_valid_runs(exp_list: list[str]) -> dict[str, Path]:
            selected: dict[str, Path] = {}
            if not runs_root.exists():
                return selected
            for exp_id in exp_list:
                candidates = sorted(
                    [p for p in runs_root.glob(f"run_{exp_id}_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for candidate in candidates:
                    if _reasoning_run_valid(candidate, exp_id):
                        selected[exp_id] = candidate
                        break
            return selected

        def _generate_reasoning_fold_batches(
            experiments_list: list[str] | None,
            output_dir: Path,
            meta_path: Path,
            log_label: str,
            repair_batches_by_fold: dict[int, list[int]] | None = None,
            existing_full_df: pd.DataFrame | None = None,
            records_override: list[dict[str, Any]] | None = None,
            labels_override: np.ndarray | None = None,
            founder_ids_override: list[str | None] | None = None,
            fold_ids_override: np.ndarray | None = None,
        ) -> tuple[pd.DataFrame, list[str]]:
            local_records = records_override if records_override is not None else records
            local_labels = labels_override if labels_override is not None else labels
            local_founder_ids = (
                founder_ids_override if founder_ids_override is not None else founder_ids
            )
            local_fold_ids = fold_ids_override if fold_ids_override is not None else fold_ids
            folds_root = output_dir / "folds"
            fold_meta_paths: list[str] = []
            fallback_total = 0
            feature_cols: list[str] | None = None
            full_features: pd.DataFrame | None = None
            _log_run(
                f"[{log_label}] Starting fold-batched reasoning. "
                f"experiments={experiments_list} folds={cv_folds} "
                f"records={len(local_records)}"
            )
            if repair_batches_by_fold is not None and existing_full_df is not None:
                feature_cols = [
                    c for c in existing_full_df.columns if c not in ("founder_uuid", "success")
                ]
                full_features = existing_full_df[feature_cols].copy()
            for fold_id in range(cv_folds):
                fold_idx = np.where(local_fold_ids == fold_id)[0]
                if fold_idx.size == 0:
                    _log_run(f"[{log_label}] Fold {fold_id}: empty, skipping.")
                    continue
                if repair_batches_by_fold is not None:
                    target_batches = repair_batches_by_fold.get(fold_id, [])
                    if not target_batches:
                        _log_run(f"[{log_label}] Fold {fold_id}: no target batches, skipping.")
                        continue
                else:
                    target_batches = None
                _log_run(
                    f"[{log_label}] Fold {fold_id}: start "
                    f"(records={len(fold_idx)} target_batches={len(target_batches or [])})"
                )
                fold_records = [local_records[i] for i in fold_idx]
                fold_labels = local_labels[fold_idx]
                fold_existing = None
                if existing_full_df is not None:
                    fold_existing = existing_full_df.iloc[fold_idx].reset_index(drop=True)
                fold_dir = folds_root / f"fold_{fold_id}"
                fold_meta = meta_path.with_name(meta_path.stem + f"_fold{fold_id}.json")
                fold_meta_paths.append(str(fold_meta))
                fold_config = ReasoningConfig(
                    model=args.llm_model,
                    dataset_size=reasoning_dataset_size,
                    random_state=rs,
                    core_prompt_path=reasoning_core_prompt_path,
                    experiments_path=reasoning_experiments_path,
                    providers=llm_providers,
                    google_model=llm_google_model,
                    batch_size=llm_reasoning_batch_size,
                    concurrency=llm_reasoning_concurrency,
                    experiments=experiments_list,
                    dry_run=llm_reasoning_dry_run,
                    dry_run_fast=llm_reasoning_dry_run_fast,
                    log_dir=log_root / f"{log_label}_fold{fold_id}",
                    log_every=llm_reasoning_log_every,
                    repair_nan=llm_reasoning_repair_nan,
                    repair_existing=True if repair_batches_by_fold is not None else llm_reasoning_repair_existing,
                    skip_select=True,
                    rate_limit_fallback_concurrency=llm_reasoning_rate_limit_fallback_concurrency,
                    rate_limit_fallback_windows=llm_reasoning_rate_limit_fallback_windows,
                    rate_limit_fallback_sequence=llm_reasoning_rate_limit_fallback_sequence,
                    inline_repair=llm_reasoning_inline_repair,
                    inline_repair_max_attempts=llm_reasoning_inline_repair_max_attempts,
                    target_batch_indices=target_batches,
                )
                fold_df, _ = generate_reasoning_features(
                    records=fold_records,
                    labels=fold_labels,
                    config=fold_config,
                    output_dir=fold_dir,
                    metadata_path=fold_meta,
                    existing_df=fold_existing,
                )
                _log_run(f"[{log_label}] Fold {fold_id}: completed ({len(fold_df)} rows).")
                try:
                    fold_meta_data = json.loads(fold_meta.read_text(encoding="utf-8"))
                    fallback_total += int(fold_meta_data.get("rate_limit_fallbacks", 0))
                except Exception:
                    pass
                fold_cols = [c for c in fold_df.columns if c not in ("founder_uuid", "success")]
                if feature_cols is None:
                    feature_cols = sorted(fold_cols)
                    full_features = pd.DataFrame(index=range(len(local_records)), columns=feature_cols)
                elif set(fold_cols) != set(feature_cols):
                    raise RuntimeError("Fold reasoning feature columns do not match.")
                full_features.iloc[fold_idx] = fold_df[feature_cols].to_numpy()

            if feature_cols is None or full_features is None:
                raise RuntimeError("No reasoning features were generated.")

            combined_df = pd.DataFrame(
                {
                    "founder_uuid": [r.get("founder_uuid") for r in local_records],
                    "success": local_labels,
                }
            )
            for col in feature_cols:
                combined_df[col] = full_features[col].values

            output_dir.mkdir(parents=True, exist_ok=True)
            combined_path = output_dir / f"llm_reasoning_{reasoning_dataset_size}.parquet"
            combined_df.to_parquet(combined_path, index=False)
            _log_run(
                f"[{log_label}] Combined parquet saved: {combined_path} "
                f"(rows={len(combined_df)})"
            )

            _, exp_to_keys = build_experiment_key_map(
                reasoning_experiments_path, experiments_list
            )
            write_per_experiment_parquets(
                combined_df,
                exp_to_keys,
                output_dir,
                reasoning_dataset_size,
                overwrite=True,
            )

            meta = {
                "model": args.llm_model,
                "dataset_size": reasoning_dataset_size,
                "random_state": rs,
                "core_prompt_path": str(reasoning_core_prompt_path),
                "experiments_path": str(reasoning_experiments_path),
                "experiments": experiments_list,
                "output_parquet": str(combined_path),
                "fold_metas": fold_meta_paths,
                "batch_within_folds": True,
                "folds": cv_folds,
                "fold_sizes": [int(s) for s in np.bincount(local_fold_ids, minlength=cv_folds)],
                "batch_size": llm_reasoning_batch_size,
                "concurrency": llm_reasoning_concurrency,
                "dry_run": llm_reasoning_dry_run,
                "dry_run_fast": llm_reasoning_dry_run_fast,
                "rate_limit_fallbacks": fallback_total,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return combined_df, feature_cols

        reasoning_mode = cfg_llm_reasoning_mode or "single"
        sequential = cfg_llm_reasoning_sequential or []
        combined = cfg_llm_reasoning_combined or []
        reasoning_df: pd.DataFrame | None = None
        skip_sequential = False
        if run_profile == "reasoning_xgb_oof":
            reasoning_df = _load_currently_in_use_df()
            if reasoning_df is None:
                raise RuntimeError("No reasoning features found in currently_in_use.")
            skip_sequential = True
        if reasoning_mode == "sequential_and_combined" and sequential and not combined:
            selected_runs = _select_latest_valid_runs(sequential)
            if len(selected_runs) == len(sequential):
                _log_run("A/B/E nan-free found; skipping reasoning generation.")
                _update_currently_in_use(sequential, selected_runs)
                reasoning_df = _load_currently_in_use_df()
                if reasoning_df is not None:
                    skip_sequential = True
        def _repair_current_experiment_a_if_needed() -> bool:
            current_path = current_root / "llm_reasoning_full.parquet"
            runs_root = BASE_DIR / "features_storage" / "llm_reasoning" / "runs"
            if not current_path.exists():
                # Try to sync the most recent run_A into current/ for repair
                if runs_root.exists():
                    candidates = sorted(
                        [p for p in runs_root.glob("run_A_*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for candidate in candidates:
                        run_path = candidate / "llm_reasoning_full.parquet"
                        if run_path.exists():
                            _sync_reasoning_current(candidate, ["A"])
                            current_path = current_root / "llm_reasoning_full.parquet"
                            break
                if not current_path.exists():
                    return False
            manifest_path = current_root / "manifest.json"
            experiments = []
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    experiments = manifest.get("experiments", [])
                except Exception:
                    experiments = []
            df = pd.read_parquet(current_path)
            if len(df) == len(all_records) and len(records) != len(all_records):
                df = df.iloc[pool_idx].reset_index(drop=True)
            if not experiments:
                # fallback: infer from column names
                if any(c.startswith("A_") for c in df.columns):
                    experiments = ["A"]
            if "A" not in experiments:
                _log_run("Repair check: manifest missing A, attempting fallback.")
                experiments = ["A"]
            # If current parquet doesn't actually contain A columns, re-sync from latest run_A
            if not any(c.startswith("A_") for c in df.columns):
                if runs_root.exists():
                    candidates = sorted(
                        [p for p in runs_root.glob("run_A_*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for candidate in candidates:
                        run_path = candidate / "llm_reasoning_full.parquet"
                        if run_path.exists():
                            df_run = pd.read_parquet(run_path)
                            if len(df_run) == len(all_records) and len(records) != len(all_records):
                                df_run = df_run.iloc[pool_idx].reset_index(drop=True)
                            num_run = df_run.select_dtypes(include=[np.number]).drop(
                                columns=["success"], errors="ignore"
                            )
                            nan_free_run = True if num_run.empty else not num_run.isna().any(axis=1).any()
                            if nan_free_run and any(c.startswith("A_") for c in df_run.columns):
                                _sync_reasoning_current(candidate, ["A"])
                                _log_run(f"Repair check: synced run_A {candidate.name} into current.")
                                return True
                            # If not nan-free, use this as repair source
                            df = df_run
                            break
            num = df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            if num.empty:
                _log_run("Repair check: no numeric columns found for A.")
                return False
            nan_rows = int(num.isna().any(axis=1).sum())
            _log_run(f"Repair check: A nan_rows={nan_rows}")
            nan_mask = num.isna().any(axis=1).to_numpy()
            if not nan_mask.any():
                return True
            if not llm_reasoning_repair_existing:
                _log_run("Repair check: NaNs found but repair_existing=false.")
                return False
            nan_rows = np.where(nan_mask)[0].tolist()
            repair_batches_by_fold: dict[int, list[int]] = {}
            for fold_id in range(cv_folds):
                fold_idx = np.where(fold_ids == fold_id)[0]
                if fold_idx.size == 0:
                    continue
                pos_map = {int(idx): pos for pos, idx in enumerate(fold_idx)}
                batch_ids: set[int] = set()
                for row_idx in nan_rows:
                    if row_idx in pos_map:
                        pos = pos_map[row_idx]
                        batch_ids.add(int(pos // llm_reasoning_batch_size))
                if batch_ids:
                    repair_batches_by_fold[fold_id] = sorted(batch_ids)
            if not repair_batches_by_fold:
                return
            _log_run("Repairing current Experiment A NaNs with targeted batches.")
            repair_meta = current_root / f"llm_reasoning_{reasoning_dataset_size}_A_repair.json"
            _generate_reasoning_fold_batches(
                experiments_list=["A"],
                output_dir=current_root,
                meta_path=repair_meta,
                log_label="A_repair",
                repair_batches_by_fold=repair_batches_by_fold,
                existing_full_df=df,
            )
            # Refresh df after repair
            df_repaired = pd.read_parquet(current_path)
            if len(df_repaired) == len(all_records) and len(records) != len(all_records):
                df_repaired = df_repaired.iloc[pool_idx].reset_index(drop=True)
            num_repaired = df_repaired.select_dtypes(include=[np.number]).drop(
                columns=["success"], errors="ignore"
            )
            nan_after = int(num_repaired.isna().any(axis=1).sum()) if not num_repaired.empty else 0
            _log_run(f"Experiment A nan_rows_after_repair={nan_after}")
            _log_run("Completed targeted repair for Experiment A.")
            return nan_after == 0

        a_ready = False
        if "A" in sequential:
            a_ready = _repair_current_experiment_a_if_needed()
        if reasoning_mode == "sequential_and_combined" and (sequential or combined) and not skip_sequential:
            _log_run(f"Reasoning mode: sequential_and_combined")
            # Sequential runs
            for exp_id in sequential:
                _log_run(f"Starting sequential experiment {exp_id}")
                runs_root.mkdir(parents=True, exist_ok=True)
                if exp_id == "A":
                    if a_ready:
                        _log_run("Skipping Experiment A (ready after repair check).")
                        continue
                    current_path = current_root / "llm_reasoning_full.parquet"
                    manifest_path = current_root / "manifest.json"
                    current_experiments: list[str] = []
                    if manifest_path.exists():
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                            current_experiments = manifest.get("experiments", []) or []
                        except Exception:
                            current_experiments = []
                    if current_path.exists():
                        try:
                            df_current = pd.read_parquet(current_path)
                            if len(df_current) == len(all_records) and len(records) != len(all_records):
                                df_current = df_current.iloc[pool_idx].reset_index(drop=True)
                            has_a = any(c.startswith("A_") for c in df_current.columns)
                            num = df_current.select_dtypes(include=[np.number]).drop(
                                columns=["success"], errors="ignore"
                            )
                            nan_free = True if num.empty else not num.isna().any(axis=1).any()
                            if nan_free and ("A" in current_experiments or has_a):
                                _log_run("Reusing current Experiment A (nan-free).")
                                continue
                        except Exception:
                            pass
                # Reuse latest completed run for this experiment if available
                cached_dir = None
                if runs_root.exists():
                    candidates = sorted(
                        [p for p in runs_root.glob(f"run_{exp_id}_*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for candidate in candidates:
                        run_path = candidate / "llm_reasoning_full.parquet"
                        if run_path.exists():
                            cached_dir = candidate
                            break
                if cached_dir is not None:
                    run_parquet = cached_dir / f"llm_reasoning_{reasoning_dataset_size}.parquet"
                    if not (llm_reasoning_repair_existing and _reasoning_has_nans(run_parquet)):
                        _log_run(f"Reusing cached experiment {exp_id} from {cached_dir.name}")
                        _sync_reasoning_current(cached_dir, [exp_id])
                        _log_run(f"Completed sequential experiment {exp_id} (cached)")
                        continue

                exp_dir = runs_root / f"run_{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                meta_path = exp_dir / f"llm_reasoning_{reasoning_dataset_size}_{exp_id}.json"
                if llm_reasoning_batch_within_folds:
                    _generate_reasoning_fold_batches([exp_id], exp_dir, meta_path, exp_id)
                else:
                    config = ReasoningConfig(
                        model=args.llm_model,
                        dataset_size=reasoning_dataset_size,
                        random_state=rs,
                        core_prompt_path=reasoning_core_prompt_path,
                        experiments_path=reasoning_experiments_path,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        batch_size=llm_reasoning_batch_size,
                        concurrency=llm_reasoning_concurrency,
                        experiments=[exp_id],
                        dry_run=llm_reasoning_dry_run,
                        dry_run_fast=llm_reasoning_dry_run_fast,
                        log_dir=log_root / f"{exp_id}_progress",
                        log_every=llm_reasoning_log_every,
                        repair_nan=llm_reasoning_repair_nan,
                        repair_existing=llm_reasoning_repair_existing,
                        rate_limit_fallback_concurrency=llm_reasoning_rate_limit_fallback_concurrency,
                        rate_limit_fallback_windows=llm_reasoning_rate_limit_fallback_windows,
                        rate_limit_fallback_sequence=llm_reasoning_rate_limit_fallback_sequence,
                        inline_repair=llm_reasoning_inline_repair,
                        inline_repair_max_attempts=llm_reasoning_inline_repair_max_attempts,
                    )
                    generate_reasoning_features(
                        records=records,
                        labels=labels,
                        config=config,
                        output_dir=exp_dir,
                        metadata_path=meta_path,
                    )
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    _log_run(
                        f"Experiment {exp_id} rate_limit_fallbacks="
                        f"{int(meta.get('rate_limit_fallbacks', 0))}"
                    )
                except Exception:
                    pass
                _sync_reasoning_current(exp_dir, [exp_id])
                _log_run(f"Completed sequential experiment {exp_id}")

            # Combined run
            if combined:
                combo_id = "combined_" + "".join(combined)
                _log_run(f"Starting combined experiment {combo_id}")
                runs_root.mkdir(parents=True, exist_ok=True)
                exp_dir = runs_root / f"run_{combo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                meta_path = exp_dir / f"llm_reasoning_{reasoning_dataset_size}_{combo_id}.json"
                if llm_reasoning_batch_within_folds:
                    _generate_reasoning_fold_batches(combined, exp_dir, meta_path, combo_id)
                else:
                    config = ReasoningConfig(
                        model=args.llm_model,
                        dataset_size=reasoning_dataset_size,
                        random_state=rs,
                        core_prompt_path=reasoning_core_prompt_path,
                        experiments_path=reasoning_experiments_path,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        batch_size=llm_reasoning_batch_size,
                        concurrency=llm_reasoning_concurrency,
                        experiments=combined,
                        dry_run=llm_reasoning_dry_run,
                        dry_run_fast=llm_reasoning_dry_run_fast,
                        log_dir=log_root / f"{combo_id}_progress",
                        log_every=llm_reasoning_log_every,
                        repair_nan=llm_reasoning_repair_nan,
                        repair_existing=llm_reasoning_repair_existing,
                        rate_limit_fallback_concurrency=llm_reasoning_rate_limit_fallback_concurrency,
                        rate_limit_fallback_windows=llm_reasoning_rate_limit_fallback_windows,
                        rate_limit_fallback_sequence=llm_reasoning_rate_limit_fallback_sequence,
                        inline_repair=llm_reasoning_inline_repair,
                        inline_repair_max_attempts=llm_reasoning_inline_repair_max_attempts,
                    )
                    generate_reasoning_features(
                        records=records,
                        labels=labels,
                        config=config,
                        output_dir=exp_dir,
                        metadata_path=meta_path,
                    )
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    _log_run(
                        f"Experiment {combo_id} rate_limit_fallbacks="
                        f"{int(meta.get('rate_limit_fallbacks', 0))}"
                    )
                except Exception:
                    pass
                _sync_reasoning_current(exp_dir, combined)
                _log_run(f"Completed combined experiment {combo_id}")
            if sequential:
                selected_runs = _select_latest_valid_runs(sequential)
                _update_currently_in_use(sequential, selected_runs if selected_runs else None)
            reasoning_df = _load_currently_in_use_df()
            if reasoning_df is None:
                raise RuntimeError("No reasoning features found after sequential run.")
        elif not skip_sequential:
            runs_root.mkdir(parents=True, exist_ok=True)
            run_dir = runs_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_dir = run_dir
            meta_path = run_dir / f"llm_reasoning_{reasoning_dataset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            exp_list = cfg_llm_reasoning_experiments
            exp_id = exp_list[0] if exp_list and len(exp_list) == 1 else None
            run_parquet = run_dir / f"llm_reasoning_{reasoning_dataset_size}.parquet"
            if exp_id and run_dir.exists() and llm_reasoning_repair_existing and _reasoning_has_nans(run_parquet):
                reasoning_df, all_reasoning_names = _generate_reasoning_fold_batches(
                    exp_list,
                    run_dir,
                    meta_path,
                    f"{exp_id}_repair",
                )
                _log("  Repaired cached LLM reasoning features for run_A.")
            else:
                cached_df, cached_names = _load_existing_reasoning(exp_list)
                if cached_df is not None and cached_names:
                    reasoning_df = cached_df
                    all_reasoning_names = cached_names
                    _log("  Loaded cached LLM reasoning features from run_A output.")
                elif llm_reasoning_batch_within_folds:
                    reasoning_df, all_reasoning_names = _generate_reasoning_fold_batches(
                        exp_list,
                        output_dir,
                        meta_path,
                        "single_progress",
                    )
                else:
                    config = ReasoningConfig(
                        model=args.llm_model,
                        dataset_size=reasoning_dataset_size,
                        random_state=rs,
                        core_prompt_path=reasoning_core_prompt_path,
                        experiments_path=reasoning_experiments_path,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        batch_size=llm_reasoning_batch_size,
                        concurrency=llm_reasoning_concurrency,
                        experiments=exp_list,
                        dry_run=llm_reasoning_dry_run,
                        dry_run_fast=llm_reasoning_dry_run_fast,
                        log_dir=log_root / "single_progress",
                        log_every=llm_reasoning_log_every,
                        repair_nan=llm_reasoning_repair_nan,
                        repair_existing=llm_reasoning_repair_existing,
                        rate_limit_fallback_concurrency=llm_reasoning_rate_limit_fallback_concurrency,
                        rate_limit_fallback_windows=llm_reasoning_rate_limit_fallback_windows,
                        rate_limit_fallback_sequence=llm_reasoning_rate_limit_fallback_sequence,
                        inline_repair=llm_reasoning_inline_repair,
                        inline_repair_max_attempts=llm_reasoning_inline_repair_max_attempts,
                    )
                    reasoning_df, all_reasoning_names = generate_reasoning_features(
                        records=records,
                        labels=labels,
                        config=config,
                        output_dir=output_dir,
                        metadata_path=meta_path,
                    )
            if "reasoning_df" in locals():
                _sync_reasoning_current(run_dir, exp_list or [])

        if reasoning_df is not None and not reasoning_feature_names:
            # Use numeric reasoning features for training; keep text in parquet only.
            for col in reasoning_df.columns:
                if col not in ("founder_uuid", "success"):
                    reasoning_df[col] = pd.to_numeric(reasoning_df[col], errors="ignore")
            reasoning_feature_names = [
                c
                for c in reasoning_df.columns
                if c not in ("founder_uuid", "success") and pd.api.types.is_numeric_dtype(reasoning_df[c])
            ]
            if not reasoning_feature_names:
                raise RuntimeError(
                    "No numeric LLM reasoning features detected. "
                    "Check cached reasoning parquet column types."
                )
            reasoning_all = reasoning_df[reasoning_feature_names]

        def _load_full_current_df(exp_list: list[str]) -> pd.DataFrame | None:
            full_root = BASE_DIR / "features_storage" / "llm_reasoning" / "full_current"
            full_path = full_root / "llm_reasoning_full.parquet"
            if not full_path.exists():
                return None
            df_full = pd.read_parquet(full_path)
            if len(df_full) != len(all_records):
                return None
            for exp_id in exp_list:
                if not any(c.startswith(f"{exp_id}_") for c in df_full.columns):
                    return None
            num = df_full.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            if not num.empty and num.isna().any(axis=1).any():
                return None
            return df_full

        def _build_full_current_df(exp_list: list[str], *, seed_separate: bool = False) -> pd.DataFrame:
            if reasoning_df is None or not reasoning_feature_names:
                raise RuntimeError("Pool reasoning features are required to build full_current.")
            full_root = BASE_DIR / "features_storage" / "llm_reasoning" / "full_current"
            full_root.mkdir(parents=True, exist_ok=True)
            new_cols = [
                c
                for c in reasoning_df.columns
                if c not in ("founder_uuid", "success", "row_index")
            ]
            existing_cols: list[str] = []
            existing_full: pd.DataFrame | None = None
            existing_path = full_root / "llm_reasoning_full.parquet"
            seed_missing_exp_ids: set[str] = set()
            if existing_path.exists():
                try:
                    existing_full = pd.read_parquet(existing_path)
                    if len(existing_full) == len(all_records):
                        existing_cols = [
                            c
                            for c in existing_full.columns
                            if c not in ("founder_uuid", "success", "row_index")
                        ]
                        try:
                            # Detect missing experiments in seed rows using numeric keys from experiments.json.
                            numeric_map = _load_reasoning_numeric_map(reasoning_experiments_path, exp_list)
                            seed_slice = existing_full.iloc[seed_idx]
                            for exp_id in exp_list:
                                exp_cols = [c for c in numeric_map.get(exp_id, []) if c in existing_full.columns]
                                if not exp_cols:
                                    seed_missing_exp_ids.add(exp_id)
                                    continue
                                if seed_slice[exp_cols].isna().any(axis=1).any():
                                    seed_missing_exp_ids.add(exp_id)
                        except Exception:
                            seed_missing_exp_ids = set()
                    else:
                        existing_full = None
                        existing_cols = []
                except Exception:
                    existing_full = None
                    existing_cols = []
            # Preserve existing columns, append new ones (missing only).
            new_cols_to_add = [c for c in new_cols if c not in existing_cols]
            if not new_cols_to_add and existing_full is not None:
                num_existing = existing_full.select_dtypes(include=[np.number]).drop(
                    columns=["success"], errors="ignore"
                )
                if num_existing.empty or not num_existing.isna().any(axis=1).any():
                    return existing_full
            full_cols = existing_cols + new_cols_to_add
            full_features = pd.DataFrame(index=range(len(all_records)), columns=full_cols)
            if existing_full is not None and existing_cols:
                existing_full = existing_full.set_index("founder_uuid")
                existing_full = existing_full.reindex(all_founder_ids)
                full_features[existing_cols] = existing_full[existing_cols].values
            if new_cols_to_add:
                full_features.loc[pool_idx, new_cols_to_add] = reasoning_df[new_cols_to_add].values
            # Infer experiment IDs directly from pool reasoning columns to ensure
            # seed generation matches the exact column schema.
            seed_exp_list: list[str] = []
            seen_exp = set()
            for col in new_cols_to_add:
                if "_" not in col:
                    continue
                exp_id = col.split("_", 1)[0]
                if exp_id in seen_exp:
                    continue
                seen_exp.add(exp_id)
                seed_exp_list.append(exp_id)
            if not seed_exp_list and seed_missing_exp_ids:
                seed_exp_list = sorted(seed_missing_exp_ids)
            if not seed_exp_list:
                seed_exp_list = list(exp_list)
            seed_cols_target = (
                [c for c in full_cols if "_" in c and c.split("_", 1)[0] in seed_exp_list]
                if not new_cols_to_add
                else list(new_cols_to_add)
            )
            mode_label = "separate" if seed_separate else "combined"
            _log_run(
                f"Seed reasoning generation for full_current ({mode_label}). "
                f"new_cols={len(new_cols_to_add)} seed_exp_list={seed_exp_list} "
                f"seed_cols_target={len(seed_cols_target)}"
            )

            seed_records = [all_records[i] for i in seed_idx]
            seed_labels = all_labels[seed_idx]
            seed_founder_ids = [all_founder_ids[i] for i in seed_idx]
            seed_fold_ids = fold_ids_full[seed_idx]

            seed_tmp = full_root / "_seed"
            seed_tmp.mkdir(parents=True, exist_ok=True)
            seed_meta = seed_tmp / f"llm_reasoning_{reasoning_dataset_size}_seed.json"
            if seed_separate:
                seed_frames: list[pd.DataFrame] = []
                join_key = "founder_uuid"
                if not seed_founder_ids or all(v is None for v in seed_founder_ids):
                    join_key = "row_index"
                use_row_index = join_key == "row_index"
                for exp_id in seed_exp_list:
                    exp_tmp = seed_tmp / exp_id
                    exp_tmp.mkdir(parents=True, exist_ok=True)
                    exp_meta = seed_tmp / f"llm_reasoning_{reasoning_dataset_size}_seed_{exp_id}.json"
                    exp_parquet = exp_tmp / f"llm_reasoning_{reasoning_dataset_size}.parquet"
                    if exp_parquet.exists():
                        exp_df = pd.read_parquet(exp_parquet)
                        _log_run(f"[seed_{exp_id}] Reusing existing seed parquet ({len(exp_df)} rows).")
                    else:
                        exp_df, _ = _generate_reasoning_fold_batches(
                            experiments_list=[exp_id],
                            output_dir=exp_tmp,
                            meta_path=exp_meta,
                            log_label=f"seed_{exp_id}",
                            records_override=seed_records,
                            labels_override=seed_labels,
                            founder_ids_override=seed_founder_ids,
                            fold_ids_override=seed_fold_ids,
                        )
                    if join_key == "row_index":
                        exp_df = exp_df.reset_index(drop=True)
                        exp_df["row_index"] = exp_df.index
                    seed_frames.append(exp_df)
                seed_df = seed_frames[0]
                for extra_df in seed_frames[1:]:
                    extra_df = extra_df.drop(columns=["success"], errors="ignore")
                    overlap = [c for c in extra_df.columns if c in seed_df.columns and c != join_key]
                    if overlap:
                        extra_df = extra_df.drop(columns=overlap, errors="ignore")
                    seed_df = seed_df.merge(extra_df, on=join_key, how="inner")
                dup_cols = [c for c in seed_df.columns if c.endswith("_x") or c.endswith("_y")]
                if dup_cols:
                    for col in dup_cols:
                        base = col[:-2]
                        if base in seed_df.columns:
                            continue
                        if col.endswith("_x"):
                            seed_df[base] = seed_df[col]
                    seed_df = seed_df.drop(columns=dup_cols, errors="ignore")
                if "success" not in seed_df.columns:
                    seed_df["success"] = seed_frames[0]["success"].values
            else:
                use_row_index = not seed_founder_ids or all(v is None for v in seed_founder_ids)
                seed_df, _ = _generate_reasoning_fold_batches(
                    experiments_list=seed_exp_list,
                    output_dir=seed_tmp,
                    meta_path=seed_meta,
                    log_label="seed_reasoning",
                    records_override=seed_records,
                    labels_override=seed_labels,
                    founder_ids_override=seed_founder_ids,
                    fold_ids_override=seed_fold_ids,
                )
            _log_run(
                f"Seed reasoning completed. Rows={len(seed_df)} cols={len(seed_df.columns)}"
            )
            seed_cols = [
                c
                for c in seed_df.columns
                if c not in ("founder_uuid", "success", "row_index")
            ]
            missing_cols = sorted(set(seed_cols_target) - set(seed_cols))
            extra_cols = sorted(set(seed_cols) - set(seed_cols_target))
            if missing_cols or extra_cols:
                _log_run(
                    "Seed reasoning column mismatch; aligning to pool columns. "
                    f"missing={missing_cols} extra={extra_cols}"
                )
                for col in missing_cols:
                    seed_df[col] = np.nan
                if extra_cols:
                    seed_df = seed_df.drop(columns=extra_cols, errors="ignore")
            # Align to pool column order (new columns only)
            keep_cols = ["founder_uuid", "success"]
            if use_row_index and "row_index" in seed_df.columns:
                keep_cols.append("row_index")
            seed_df = seed_df[[c for c in (keep_cols + seed_cols_target) if c in seed_df.columns]]
            seed_cols = list(seed_cols_target)

            use_row_index = not all_founder_ids or all(v is None for v in all_founder_ids)
            if use_row_index:
                if "row_index" in seed_df.columns:
                    seed_df = seed_df.sort_values("row_index").reset_index(drop=True)
                for idx, row in seed_df.iterrows():
                    if idx >= len(seed_idx):
                        break
                    if seed_cols:
                        full_features.loc[seed_idx[idx], seed_cols] = row[seed_cols].values
            else:
                idx_map = {uid: idx for idx, uid in enumerate(all_founder_ids)}
                for _, row in seed_df.iterrows():
                    uid = row.get("founder_uuid")
                    if uid not in idx_map:
                        continue
                    if seed_cols:
                        full_features.loc[idx_map[uid], seed_cols] = row[seed_cols].values

            combined_df = pd.DataFrame(
                {
                    "founder_uuid": all_founder_ids,
                    "success": all_labels,
                }
            )
            for col in full_cols:
                combined_df[col] = full_features[col].values

            out_path = full_root / "llm_reasoning_full.parquet"
            combined_df.to_parquet(out_path, index=False)
            _log_run(f"full_current parquet written: {out_path} (rows={len(combined_df)})")

            _, exp_to_keys = build_experiment_key_map(reasoning_experiments_path, exp_list)
            write_per_experiment_parquets(
                combined_df,
                exp_to_keys,
                full_root,
                reasoning_dataset_size,
                overwrite=True,
            )
            manifest = {
                "timestamp": datetime.now().isoformat(),
                "experiments": sorted({c.split("_", 1)[0] for c in full_cols if "_" in c}),
                "source": "seed_merge",
                "pool_records": len(records),
                "seed_records": len(seed_records),
            }
            (full_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return combined_df

        def _dry_validate_xgb_mirror(exp_list: list[str]) -> None:
            _log_run("DRY_VALIDATE: xgb_mirror started")
            runs_root = BASE_DIR / "features_storage" / "llm_reasoning" / "runs"
            if not runs_root.exists():
                _log_run("DRY_VALIDATE: runs_root missing; cannot validate A/B/E.")
            selected_runs = _select_latest_valid_runs(exp_list) if exp_list else {}
            _log_run(f"DRY_VALIDATE: valid runs found for {sorted(selected_runs.keys())}")
            use_root = BASE_DIR / "features_storage" / "llm_reasoning" / "currently_in_use"
            use_path = use_root / "llm_reasoning_full.parquet"
            if use_path.exists():
                df_use = pd.read_parquet(use_path)
                if len(df_use) == len(all_records) and len(records) != len(all_records):
                    df_use = df_use.iloc[pool_idx].reset_index(drop=True)
                num = df_use.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
                nan_count = int(num.isna().any(axis=1).sum()) if not num.empty else 0
                _log_run(f"DRY_VALIDATE: currently_in_use rows={len(df_use)} nan_rows={nan_count}")
            else:
                _log_run("DRY_VALIDATE: currently_in_use missing.")
            full_df = _load_full_current_df(exp_list)
            if full_df is None:
                _log_run("DRY_VALIDATE: full_current missing or invalid. Seed_100 reasoning would be generated.")
            else:
                num_full = full_df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
                nan_full = int(num_full.isna().any(axis=1).sum()) if not num_full.empty else 0
                _log_run(f"DRY_VALIDATE: full_current rows={len(full_df)} nan_rows={nan_full}")
            _log_run("DRY_VALIDATE: xgb_mirror complete")

    # Optional: LLM-engineered features for reasoning experiments
    if llm_engineered_for_reasoning:
        if llm_feature_names:
            llm_engineered_feature_names = llm_feature_names
        else:
            cache_dir = BASE_DIR / "features_storage" / "llm_engineered"
            if llm_engineered_force_recompute and llm_engineered_cache and not llm_engineered_rotated:
                _rotate_llm_engineered_cache(cache_dir)
                llm_engineered_rotated = True
                _log("  Rotated existing LLM-engineered cache to old/.")
            llm_cached_all = None
            llm_cached_names = None
            if llm_engineered_cache:
                llm_cached_all, llm_cached_names = _load_llm_engineered_cache(
                    cache_dir=cache_dir,
                    expected_rows=len(records),
                    expected_n=llm_n,
                    model=args.llm_model,
                    providers=llm_providers,
                    google_model=llm_google_model,
                    seed_hash=seed_hash,
                )
            if llm_cached_all is not None and llm_cached_names is not None:
                llm_all = llm_cached_all
                llm_feature_names = llm_cached_names
                _log("  Loaded cached LLM-engineered features.")
            else:
                if llm_engineered_freeze:
                    raise RuntimeError(
                        "LLM-engineered features are frozen but cache is missing. "
                        "Set llm_engineered_freeze=false or regenerate the cache."
                    )
                _log(f"\n  Generating {llm_n} LLM-engineered features with {args.llm_model}...")
                llm_all, _, _, llm_feature_names = asyncio.run(
                    generate_llm_features(
                        train_recs=seed_recs,
                        y_train=seed_labels,
                        test_recs=records,
                        model=args.llm_model,
                        n_features=llm_n,
                        all_recs=records,
                        providers=llm_providers,
                        google_model=llm_google_model,
                    )
                )
                if llm_engineered_cache:
                    _save_llm_engineered_cache(
                        cache_dir=cache_dir,
                        df=llm_all,
                        feature_names=llm_feature_names,
                        model=args.llm_model,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        n_features=llm_n,
                        seed_hash=seed_hash,
                    )
            llm_engineered_feature_names = llm_feature_names


    if mode == "human":
        full_all = pd.concat([base_all, custom_all], axis=1)
        feature_names = base_feature_names + custom_features
        mode_label = "Human Only"
    elif mode == "llm":
        full_all = llm_all
        feature_names = llm_feature_names
        mode_label = "LLM Only"
    elif mode == "reasoning":
        full_all = reasoning_all
        feature_names = reasoning_feature_names
        mode_label = "LLM Reasoning Only"
    else:
        full_all = pd.concat([base_all, custom_all, llm_all, reasoning_all], axis=1)
        feature_names = base_feature_names + custom_features + llm_feature_names + reasoning_feature_names
        mode_label = "Hybrid"

    # Continuous standardization happens inside CV folds.

    # Verify selected features match config (when provided)
    if selected_features and mode in ("human", "hybrid"):
        expected_set = {f for f in selected_features if f in feature_names}
        actual_set = set(feature_names)
        if mode == "human" and expected_set != actual_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            raise RuntimeError(
                "Selected feature list mismatch. "
                f"Missing: {missing} Extra: {extra}"
            )
        if mode == "hybrid" and not expected_set.issubset(actual_set):
            missing = sorted(expected_set - actual_set)
            raise RuntimeError(
                "Selected feature list mismatch. "
                f"Missing: {missing}"
            )

    # NaN/Inf checks (full dataset; per-fold imputation handled in CV).
    if not np.isfinite(full_all.values).all():
        raise RuntimeError("NaN or Inf detected in feature matrix.")

    if run_profile != "xgb_mirror":
        # Save feature dataset (founder_uuid, label, features)
        founder_ids = [r.get("founder_uuid") for r in records]
        save_df = pd.concat(
            [
                pd.Series(founder_ids, name="founder_uuid"),
                pd.Series(labels, name="success"),
                full_all,
            ],
            axis=1,
        )
        features_storage = BASE_DIR / "features_storage"
        features_storage.mkdir(parents=True, exist_ok=True)
        if args.output_parquet:
            out_path = Path(args.output_parquet)
        else:
            if mode == "human":
                if human_feature_source == "high_quality":
                    out_path = features_storage / "human_high_quality" / "features_full.parquet"
                else:
                    out_path = features_storage / "human" / "features_full.parquet"
            elif mode == "llm":
                out_path = features_storage / "llm_engineered" / "llm_features.parquet"
            elif mode == "reasoning":
                out_path = features_storage / "llm_reasoning" / "current" / "llm_reasoning_full.parquet"
            else:
                out_path = features_storage / "features_full.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(out_path, index=False)
        _log(f"  Saved features to: {out_path}")

        if args.extract_only:
            _log("  extract_only enabled; skipping training.")
            _write_log(
                log_lines,
                args,
                input_csv,
                feature_names,
                len(records),
                0,
                int(labels.sum()),
                0,
                {"threshold": None},
                log_dir=_log_dir_for_mode(mode, llm_reasoning_dry_run, llm_reasoning_dry_run_fast),
            )
            return

    # Expanded reasoning + engineered evaluations (CV).
    lr_table1_rows: list[dict[str, Any]] = []
    lr_table2_rows: list[dict[str, Any]] = []
    lr_table_hq_rows: list[dict[str, Any]] = []
    xgb_table1_rows: list[dict[str, Any]] = []
    xgb_table2_rows: list[dict[str, Any]] = []
    xgb_table_hq_rows: list[dict[str, Any]] = []
    full_lr_rows: list[dict[str, Any]] = []
    full_xgb_rows: list[dict[str, Any]] = []
    reasoning_xgb_oof_hq_rows: list[dict[str, Any]] = []
    reasoning_xgb_oof_mirror_rows: list[dict[str, Any]] = []
    reasoning_combo_cols: dict[str, list[str]] = {}
    reasoning_combo_frames: dict[str, pd.DataFrame] = {}
    reasoning_df_part1: pd.DataFrame | None = None
    reasoning_combo_cols_part1: dict[str, list[str]] = {}
    reasoning_combo_frames_part1: dict[str, pd.DataFrame] = {}

    args_lr = copy.copy(args)
    args_lr.model_type = "logistic"
    args_xgb = copy.copy(args)
    args_xgb.model_type = "xgboost"

    def _append_row(
        target: list[dict[str, Any]],
        table: str,
        regression: str,
        set_id: str,
        combo: str,
        means: dict[str, float],
        stds: dict[str, float],
        *,
        variant: str = "",
        tuning: str = "oof",
    ) -> None:
        target.append(
            {
                "table": table,
                "regression": regression,
                "set_id": set_id,
                "reasoning_combo": combo,
                "variant": variant,
                "tuning": tuning,
                "F0.5": means.get("f0.5", float("nan")),
                "F0.5_std": stds.get("f0.5", float("nan")),
                "ROC-AUC": means.get("roc_auc", float("nan")),
                "ROC-AUC_std": stds.get("roc_auc", float("nan")),
                "PR-AUC": means.get("pr_auc", float("nan")),
                "PR-AUC_std": stds.get("pr_auc", float("nan")),
                "Prec": means.get("precision", float("nan")),
                "Prec_std": stds.get("precision", float("nan")),
                "Rec": means.get("recall", float("nan")),
                "Rec_std": stds.get("recall", float("nan")),
                "Acc": means.get("accuracy", float("nan")),
                "Acc_std": stds.get("accuracy", float("nan")),
            }
        )

    if use_llm_reasoning and reasoning_feature_names:
        if reasoning_df is not None:
            reasoning_combo_cols = _build_reasoning_combos(reasoning_df, exp_list_for_combo)
            reasoning_combo_frames = {
                combo: reasoning_df[cols].copy() for combo, cols in reasoning_combo_cols.items()
            }
        if not reasoning_combo_cols:
            _log("  WARNING: No reasoning combos found; check reasoning columns.")

        if run_profile == "reasoning_xgb_oof":
            exp_list_for_combo = ["A", "B", "C", "D", "E", "F"]
            reasoning_full_df = _load_full_current_df(exp_list_for_combo)
            if reasoning_full_df is None:
                _log_run("Full reasoning cache missing; generating seed_100 reasoning for full dataset.")
                reasoning_full_df = _build_full_current_df(exp_list_for_combo, seed_separate=True)
            _validate_reasoning_full_df(reasoning_full_df, len(all_records))
            for col in reasoning_full_df.columns:
                if col not in ("founder_uuid", "success"):
                    reasoning_full_df[col] = pd.to_numeric(reasoning_full_df[col], errors="ignore")
            if args.reasoning_seed_only:
                _log_run("reasoning_seed_only: full_current ready; exiting before evaluation.")
                return
            reasoning_full_combo_cols = _build_reasoning_combos(reasoning_full_df, exp_list_for_combo)
            reasoning_full_combo_cols = _filter_reasoning_combos_no_f_with_ab(reasoning_full_combo_cols)
            if not reasoning_full_combo_cols:
                raise RuntimeError("No reasoning combos found after filtering F with A/B.")
            reasoning_full_combo_frames = {
                combo: reasoning_full_df[cols].copy() for combo, cols in reasoning_full_combo_cols.items()
            }
            _log_run(
                f"reasoning_xgb_oof: total combos={len(reasoning_full_combo_cols)}"
            )

            hq_script_full = (
                PROJECT_ROOT
                / "High_Quality_human_features"
                / "features"
                / "extract_structured.py"
            )
            hq_df_full_all = _build_high_quality_features(all_records, hq_script_full)
            missing_full = [f for f in HQ_FEATURES_BASE if f not in hq_df_full_all.columns]
            if missing_full:
                raise RuntimeError(
                    "High-quality feature extraction missing columns (full): "
                    + ", ".join(missing_full)
                )
            hq_full_no_gap_all = hq_df_full_all[HQ_FEATURES_BASE].copy()
            rule_mask_full = (
                hq_df_full_all["exit_count"].fillna(0.0).astype(float).values > 0
            )

            for combo, cols in reasoning_full_combo_cols.items():
                combo_tag = combo.replace("+", "_")
                combo_df_full = reasoning_full_combo_frames[combo]
                combo_plus = pd.concat([hq_full_no_gap_all, combo_df_full], axis=1)
                _log_run(f"reasoning_xgb_oof: combo {combo} start")

                means, stds = _train_and_log_cv(
                    HQ_FEATURES_BASE + cols,
                    combo_plus,
                    all_labels,
                    args_xgb,
                    input_csv,
                    f"HQ + Reasoning {combo} (Full, XGB, oof)",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR
                    / "training_logs"
                    / "reasoning_xgb_oof"
                    / "hq_plus"
                    / combo_tag,
                    cv_splits=cv_splits_full,
                    threshold_tuning="oof",
                )
                _append_row(
                    reasoning_xgb_oof_hq_rows,
                    "Reasoning XGB OOF",
                    "HQ + Reasoning",
                    "",
                    combo,
                    means,
                    stds,
                    variant="full_hq_no_rule",
                    tuning="oof",
                )

                means, stds = _train_and_log_cv(
                    HQ_FEATURES_BASE + cols,
                    combo_plus,
                    all_labels,
                    args_xgb,
                    input_csv,
                    f"Mirror HQ + Reasoning {combo} (Full, XGB, oof)",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR
                    / "training_logs"
                    / "reasoning_xgb_oof"
                    / "mirror_plus"
                    / combo_tag,
                    cv_splits=cv_splits_full,
                    rule_mask=rule_mask_full,
                    threshold_tuning="oof",
                )
                _append_row(
                    reasoning_xgb_oof_mirror_rows,
                    "Reasoning XGB OOF",
                    "Full Mirror + Reasoning",
                    "",
                    combo,
                    means,
                    stds,
                    variant="full_mirror_rule",
                    tuning="oof",
                )
                _log_run(f"reasoning_xgb_oof: combo {combo} done")

            report_path = BASE_DIR / "docs" / "llm_reasoning_xgb_oof_report.md"
            csv_path = BASE_DIR / "docs" / "llm_reasoning_xgb_oof_results.csv"
            _write_reasoning_xgb_oof_report(
                reasoning_xgb_oof_hq_rows,
                reasoning_xgb_oof_mirror_rows,
                report_path,
                csv_path,
                cv_folds,
                len(all_labels),
            )
            _log(f"\nUpdated report: {report_path}")
            _log(f"Full results CSV: {csv_path}")
            return

        if run_profile != "xgb_mirror":
            # Baseline human features + reasoning (Logistic only).
            human_full = pd.concat([base_all_baseline, custom_all_baseline], axis=1)
            means, stds = _train_and_log_cv(
                base_feature_names_baseline + list(custom_features_baseline),
                human_full,
                labels,
                args_lr,
                input_csv,
                "Human Only",
                cv_folds=cv_folds,
                log_dir=BASE_DIR / "training_logs" / "human" / "only",
                cv_splits=cv_splits,
            )
            _append_row(lr_table1_rows, "Table 1", "Human Only", "", "", means, stds)

            means, stds = _train_and_log_cv(
                base_feature_names_baseline + list(custom_features_baseline),
                human_full,
                labels,
                args_xgb,
                input_csv,
                "Human Only (XGB)",
                cv_folds=cv_folds,
                log_dir=BASE_DIR / "training_logs" / "human" / "only_xgb",
                cv_splits=cv_splits,
            )
            _append_row(xgb_table1_rows, "Table 1", "Human Only", "", "", means, stds)

            # Legacy human-only feature sets (LR + XGB).
            legacy_union = set()
            for legacy in LEGACY_HUMAN_FEATURE_SETS:
                legacy_union.update(legacy["features"])
            legacy_custom = [f for f in legacy_union if f not in base_feature_names_baseline]
            legacy_custom_df = _custom_feature_df(records, legacy_custom) if legacy_custom else pd.DataFrame(index=base_all_baseline.index)
            legacy_full = pd.concat([base_all_baseline, legacy_custom_df], axis=1)
            for legacy in LEGACY_HUMAN_FEATURE_SETS:
                legacy_name = legacy["name"]
                legacy_features = legacy["features"]
                missing = [f for f in legacy_features if f not in legacy_full.columns]
                if missing:
                    _log_run(f"Legacy set '{legacy_name}' missing columns: {missing}. Skipping.")
                    continue
                legacy_df = legacy_full[legacy_features].copy()
                means, stds = _train_and_log_cv(
                    legacy_features,
                    legacy_df,
                    labels,
                    args_lr,
                    input_csv,
                    legacy_name,
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "human" / "legacy" / legacy_name.replace(" ", "_"),
                    cv_splits=cv_splits,
                )
                _append_row(lr_table1_rows, "Table 1", legacy_name, "", "", means, stds)

                means, stds = _train_and_log_cv(
                    legacy_features,
                    legacy_df,
                    labels,
                    args_xgb,
                    input_csv,
                    f"{legacy_name} (XGB)",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "human" / "legacy_xgb" / legacy_name.replace(" ", "_"),
                    cv_splits=cv_splits,
                )
                _append_row(xgb_table1_rows, "Table 1", legacy_name, "", "", means, stds)

            part1_exp_list = ["A", "B", "E"]
            selected_part1 = _select_latest_valid_runs(part1_exp_list)
            if len(selected_part1) == len(part1_exp_list):
                _update_currently_in_use(part1_exp_list, selected_part1)
            reasoning_df_part1 = _load_currently_in_use_df()
            if reasoning_df_part1 is None:
                raise RuntimeError("currently_in_use missing; cannot build Part 1 reasoning combos.")
            reasoning_combo_cols_part1 = _build_reasoning_combos(
                reasoning_df_part1, part1_exp_list
            )
            reasoning_combo_cols_part1 = {
                combo: cols
                for combo, cols in reasoning_combo_cols_part1.items()
                if _combo_is_subset(combo, {"A", "B", "E"})
            }
            reasoning_combo_frames_part1 = {
                combo: reasoning_df_part1[cols].copy()
                for combo, cols in reasoning_combo_cols_part1.items()
            }
            _log_run(
                "Part1 prefix counts: "
                + ", ".join(
                    f"{p}{sum(c.startswith(p) for c in reasoning_df_part1.columns)}"
                    for p in ("A_", "B_", "E_")
                )
            )
            _log_run(
                f"Part1 combos: total={len(reasoning_combo_cols_part1)} "
                f"examples={list(reasoning_combo_cols_part1.keys())[:10]}"
            )
            for combo, cols in reasoning_combo_cols_part1.items():
                combo_tag = combo.replace("+", "_")
                combo_df = reasoning_df_part1[cols].copy()
                means, stds = _train_and_log_cv(
                    cols,
                    combo_df,
                    labels,
                    args_lr,
                    input_csv,
                    f"Reasoning {combo}",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "llm_reasoning" / "only" / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(lr_table1_rows, "Table 1", "Reasoning Only", "", combo, means, stds)

                means, stds = _train_and_log_cv(
                    cols,
                    combo_df,
                    labels,
                    args_xgb,
                    input_csv,
                    f"Reasoning {combo} (XGB)",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "llm_reasoning" / "only_xgb" / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(xgb_table1_rows, "Table 1", "Reasoning Only", "", combo, means, stds)

                combo_plus = pd.concat([human_full, combo_df], axis=1)
                means, stds = _train_and_log_cv(
                    base_feature_names_baseline + list(custom_features_baseline) + cols,
                    combo_plus,
                    labels,
                    args_lr,
                    input_csv,
                    f"Human + Reasoning {combo}",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "llm_reasoning" / "human_plus" / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(lr_table1_rows, "Table 1", "Human + Reasoning", "", combo, means, stds)

                means, stds = _train_and_log_cv(
                    base_feature_names_baseline + list(custom_features_baseline) + cols,
                    combo_plus,
                    labels,
                    args_xgb,
                    input_csv,
                    f"Human + Reasoning {combo} (XGB)",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "llm_reasoning" / "human_plus_xgb" / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(xgb_table1_rows, "Table 1", "Human + Reasoning", "", combo, means, stds)

        # HQ features + reasoning (Logistic only).
        if run_profile != "xgb_mirror":
            if hq_full_no_gap is None or hq_full_with_gap is None:
                raise RuntimeError("High-quality feature frames missing.")

            means, stds = _train_and_log_cv(
                HQ_FEATURES_BASE,
                hq_full_no_gap,
                labels,
                args_lr,
                input_csv,
                "HQ Only",
                cv_folds=cv_folds,
                log_dir=BASE_DIR / "training_logs" / "human_high_quality" / "only",
                cv_splits=cv_splits,
            )
            _append_row(lr_table_hq_rows, "Table HQ", "HQ Only", "", "", means, stds)

            means, stds = _train_and_log_cv(
                HQ_FEATURES_WITH_GAP,
                hq_full_with_gap,
                labels,
                args_lr,
                input_csv,
                "HQ Only (+repeat_founding_gap)",
                cv_folds=cv_folds,
                log_dir=BASE_DIR / "training_logs" / "human_high_quality" / "only_with_gap",
                cv_splits=cv_splits,
            )
            _append_row(
                lr_table_hq_rows,
                "Table HQ",
                "HQ Only (+repeat_founding_gap)",
                "",
                "",
                means,
                stds,
            )

            means, stds = _train_and_log_cv(
                HQ_FEATURES_BASE,
                hq_full_no_gap,
                labels,
                args_xgb,
                input_csv,
                "HQ Only (XGB)",
                cv_folds=cv_folds,
                log_dir=BASE_DIR / "training_logs" / "human_high_quality" / "only_xgb",
                cv_splits=cv_splits,
            )
            _append_row(xgb_table_hq_rows, "Table HQ", "HQ Only", "", "", means, stds)

            hq_exp_list_part1 = ["A", "B", "E", "F"]
            selected_hq_part1 = _select_latest_valid_runs(hq_exp_list_part1)
            if len(selected_hq_part1) == len(hq_exp_list_part1):
                _update_currently_in_use(hq_exp_list_part1, selected_hq_part1)
            hq_reasoning_df_part1 = _load_currently_in_use_df()
            if hq_reasoning_df_part1 is None:
                raise RuntimeError("currently_in_use missing; cannot build HQ Part 1 reasoning combos.")
            hq_combo_cols_part1 = _build_reasoning_combos(
                hq_reasoning_df_part1, hq_exp_list_part1
            )
            hq_combo_cols_part1 = {
                combo: cols
                for combo, cols in hq_combo_cols_part1.items()
                if _combo_is_subset(combo, {"A", "B", "E", "F"})
            }
            hq_combo_frames_part1 = {
                combo: hq_reasoning_df_part1[cols].copy()
                for combo, cols in hq_combo_cols_part1.items()
            }
            if not hq_combo_cols_part1:
                _log_run("HQ Part1: No A/B/E combos found in currently_in_use; skipping HQ+Reasoning combos.")
            for combo, cols in hq_combo_cols_part1.items():
                combo_tag = combo.replace("+", "_")
                combo_df = hq_combo_frames_part1[combo]
                combo_plus = pd.concat([hq_full_no_gap, combo_df], axis=1)
                means, stds = _train_and_log_cv(
                    HQ_FEATURES_BASE + cols,
                    combo_plus,
                    labels,
                    args_lr,
                    input_csv,
                    f"HQ + Reasoning {combo}",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "llm_reasoning" / "hq_plus" / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(lr_table_hq_rows, "Table HQ", "HQ + Reasoning", "", combo, means, stds)

                means, stds = _train_and_log_cv(
                    HQ_FEATURES_BASE + cols,
                    combo_plus,
                    labels,
                    args_xgb,
                    input_csv,
                    f"HQ + Reasoning {combo} (XGB)",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR / "training_logs" / "llm_reasoning" / "hq_plus_xgb" / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(xgb_table_hq_rows, "Table HQ", "HQ + Reasoning", "", combo, means, stds)

    # Run-family mode (multiple engineered feature sets + leaderboard)
    if llm_engineered_run_family and not skip_llm_engineered:
        # Run-family should only use A/B/E combos from currently_in_use.
        run_family_df = _load_currently_in_use_df()
        if run_family_df is None:
            raise RuntimeError("Run-family mode requires currently_in_use reasoning data.")
        run_family_combo_cols = _build_reasoning_combos(run_family_df, ["A", "B", "E"])
        run_family_combo_cols = {
            combo: cols
            for combo, cols in run_family_combo_cols.items()
            if _combo_is_subset(combo, {"A", "B", "E"})
        }
        if not run_family_combo_cols:
            raise RuntimeError("Run-family mode found no A/B/E combos in currently_in_use.")
        run_family_combo_frames = {combo: run_family_df[cols].copy() for combo, cols in run_family_combo_cols.items()}
        _log_run(
            f"Run-family combos (A/B/E only): total={len(run_family_combo_cols)} "
            f"examples={list(run_family_combo_cols.keys())[:10]}"
        )
        families_dir = cache_dir / "families"
        archives_dir = cache_dir / "archives"
        family_dir: Path | None = None
        family_id = llm_engineered_run_family_id or ""
        if llm_engineered_freeze and not family_id:
            if families_dir.exists():
                consolidated = sorted(
                    families_dir.glob("family_*_features.parquet"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if consolidated:
                    latest = consolidated[0]
                    family_id = latest.stem.replace("family_", "").replace("_features", "")
                    _log(f"  [Run-family] Reusing consolidated family: family_{family_id}")
            if not family_id:
                existing = sorted(
                    [p for p in cache_dir.glob("family_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for candidate in existing:
                    set_dirs = list(candidate.glob("set_*"))
                    if not set_dirs:
                        continue
                    has_cache = any(
                        (sd / "llm_features.parquet").exists()
                        and (sd / "llm_features_meta.json").exists()
                        for sd in set_dirs
                    )
                    if has_cache:
                        family_dir = candidate
                        family_id = candidate.name.replace("family_", "")
                        _log(f"  [Run-family] Reusing existing family cache: {candidate.name}")
                        break
        if family_dir is None:
            if not family_id:
                family_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            family_dir = cache_dir / f"family_{family_id}"
            family_dir.mkdir(parents=True, exist_ok=True)
        family_features_path = families_dir / f"family_{family_id}_features.parquet"
        family_meta_path = families_dir / f"family_{family_id}_meta.json"
        run_n = llm_engineered_run_family_n or llm_n
        skipped_sets: list[str] = []
        founder_ids = [r.get("founder_uuid") for r in records]
        family_frames: list[pd.DataFrame] = []
        family_set_features: dict[str, list[str]] = {}
        _log_run(
            f"Run-family start: id={family_id} n_sets={llm_engineered_run_family_size} n_features={run_n}"
        )

        def _evaluate_family_set(set_id: str, features: list[str], set_all: pd.DataFrame) -> None:
            _log_run(f"[Run-family] {set_id}: feature_count={len(features)} rows={len(set_all)}")
            _log_run(f"[Run-family] {set_id}: start engineered-only (LR)")
            metrics_only_lr, stds_only_lr = _train_and_log_cv(
                features,
                set_all,
                labels,
                args_lr,
                input_csv,
                "LLM Engineered Only",
                cv_folds=cv_folds,
                log_dir=BASE_DIR
                / "training_logs"
                / "llm_engineered"
                / f"family_{family_id}"
                / set_id
                / "only_lr",
                cv_splits=cv_splits,
            )
            _append_row(lr_table2_rows, "Table 2", "LLM Engineered Only", set_id, "", metrics_only_lr, stds_only_lr)
            _log_run(f"[Run-family] {set_id}: done engineered-only (LR)")

            _log_run(f"[Run-family] {set_id}: start engineered-only (XGB)")
            metrics_only_xgb, stds_only_xgb = _train_and_log_cv(
                features,
                set_all,
                labels,
                args_xgb,
                input_csv,
                "LLM Engineered Only",
                cv_folds=cv_folds,
                log_dir=BASE_DIR
                / "training_logs"
                / "llm_engineered"
                / f"family_{family_id}"
                / set_id
                / "only_xgb",
                cv_splits=cv_splits,
            )
            _append_row(xgb_table2_rows, "Table 2", "LLM Engineered Only", set_id, "", metrics_only_xgb, stds_only_xgb)
            _log_run(f"[Run-family] {set_id}: done engineered-only (XGB)")

            for combo, cols in run_family_combo_cols.items():
                combo_tag = combo.replace("+", "_")
                combo_df = run_family_combo_frames[combo]
                set_plus_all = pd.concat([set_all, combo_df], axis=1)
                _log_run(f"[Run-family] {set_id}: start engineered+reasoning {combo} (LR)")
                metrics_plus_lr, stds_plus_lr = _train_and_log_cv(
                    features + cols,
                    set_plus_all,
                    labels,
                    args_lr,
                    input_csv,
                    f"LLM Engineered + Reasoning {combo}",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR
                    / "training_logs"
                    / "llm_engineered"
                    / f"family_{family_id}"
                    / set_id
                    / "plus_reasoning_lr"
                    / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(
                    lr_table2_rows,
                    "Table 2",
                    "LLM Engineered + Reasoning",
                    set_id,
                    combo,
                    metrics_plus_lr,
                    stds_plus_lr,
                )
                _log_run(f"[Run-family] {set_id}: done engineered+reasoning {combo} (LR)")

                _log_run(f"[Run-family] {set_id}: start engineered+reasoning {combo} (XGB)")
                metrics_plus_xgb, stds_plus_xgb = _train_and_log_cv(
                    features + cols,
                    set_plus_all,
                    labels,
                    args_xgb,
                    input_csv,
                    f"LLM Engineered + Reasoning {combo}",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR
                    / "training_logs"
                    / "llm_engineered"
                    / f"family_{family_id}"
                    / set_id
                    / "plus_reasoning_xgb"
                    / combo_tag,
                    cv_splits=cv_splits,
                )
                _append_row(
                    xgb_table2_rows,
                    "Table 2",
                    "LLM Engineered + Reasoning",
                    set_id,
                    combo,
                    metrics_plus_xgb,
                    stds_plus_xgb,
                )
                _log_run(f"[Run-family] {set_id}: done engineered+reasoning {combo} (XGB)")

        if llm_engineered_freeze and family_features_path.exists() and family_meta_path.exists():
            df = pd.read_parquet(family_features_path)
            meta = json.loads(family_meta_path.read_text(encoding="utf-8"))
            if meta.get("seed_hash") != seed_hash:
                if not llm_engineered_family_allow_seed_mismatch:
                    raise RuntimeError(
                        "Run-family cache seed hash mismatch. "
                        "Delete the family cache or reset the seed file to regenerate."
                    )
                _log(
                    "  WARNING: Run-family seed hash mismatch; proceeding because "
                    "llm_engineered_family_allow_seed_mismatch=true."
                )
            set_features: dict[str, list[str]] = meta.get("set_features", {})
            for set_id, features in set_features.items():
                df_set = df[df["set_id"] == set_id].set_index("row_index").sort_index()
                set_all = df_set[features].copy()
                set_all.index = range(len(records))
                _evaluate_family_set(set_id, features, set_all)
        else:
            for idx in range(1, llm_engineered_run_family_size + 1):
                set_id = f"set_{idx:02d}"
                set_dir = family_dir / set_id
                set_dir.mkdir(parents=True, exist_ok=True)
                _log_run(f"Run-family processing {set_id}")
                set_all = None
                set_names = None
                set_rules = None
                if llm_engineered_cache:
                    set_all, set_names = _load_llm_engineered_cache(
                        cache_dir=set_dir,
                        expected_rows=len(records),
                        expected_n=run_n,
                        model=args.llm_model,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        seed_hash=seed_hash,
                    )
                if set_all is None or set_names is None:
                    if llm_engineered_freeze:
                        _log(
                            f"\n  [Run-family] Cache missing for {set_id}; "
                            "freeze enabled so skipping generation."
                        )
                        skipped_sets.append(set_id)
                        _log_run(f"Run-family skipped {set_id} (cache missing)")
                        continue
                    _log(f"\n  [Run-family] Generating set {set_id} ({run_n} features)...")
                    attempt = 0
                    max_attempts = max(1, int(args.llm_retry_attempts) + 1)
                    while attempt < max_attempts:
                        attempt += 1
                        try:
                            async def _run_generate():
                                return await generate_llm_features(
                                    train_recs=seed_recs,
                                    y_train=seed_labels,
                                    test_recs=records,
                                    model=args.llm_model,
                                    n_features=run_n,
                                    all_recs=records,
                                    providers=llm_providers,
                                    google_model=llm_google_model,
                                )

                            if float(args.llm_timeout) > 0:
                                set_all, _, _, set_names, set_rules = asyncio.run(
                                    asyncio.wait_for(
                                        _run_generate(), timeout=float(args.llm_timeout)
                                    )
                                )
                            else:
                                set_all, _, _, set_names, set_rules = asyncio.run(_run_generate())
                            break
                        except Exception:
                            err = traceback.format_exc()
                            _log(f"\n  [Run-family] ERROR during LLM generation for {set_id} (attempt {attempt}):")
                            _log(err)
                            _log_run(
                                f"[Run-family] ERROR {set_id} attempt {attempt}: "
                                f"{err.splitlines()[-1] if err else 'unknown error'}"
                            )
                            if attempt >= max_attempts:
                                _log(
                                    f"  [Run-family] Exceeded retry attempts for {set_id}. Skipping."
                                )
                                _log_run(
                                    f"Run-family skipped {set_id} (generation failed after {attempt} attempts)"
                                )
                                skipped_sets.append(set_id)
                                set_all = None
                                set_names = None
                                break
                            _log(f"  [Run-family] Retrying in {args.llm_retry_sleep:.1f}s...")
                            time.sleep(max(0.0, float(args.llm_retry_sleep)))
                    if (
                        llm_engineered_cache
                        and set_all is not None
                        and set_names is not None
                    ):
                        _save_llm_engineered_cache(
                            cache_dir=set_dir,
                            df=set_all,
                            feature_names=set_names,
                            model=args.llm_model,
                            providers=llm_providers,
                            google_model=llm_google_model,
                            n_features=run_n,
                            seed_hash=seed_hash,
                            rules=set_rules,
                        )

                if set_all is None or set_names is None:
                    continue

                set_df = set_all.copy()
                set_df.insert(0, "row_index", np.arange(len(records)))
                set_df.insert(1, "founder_uuid", founder_ids)
                set_df.insert(2, "success", labels)
                set_df.insert(3, "set_id", set_id)
                family_frames.append(set_df)
                family_set_features[set_id] = set_names

                _evaluate_family_set(set_id, set_names, set_all)
                _log_run(f"Run-family completed {set_id}")

            if family_frames:
                families_dir.mkdir(parents=True, exist_ok=True)
                combined = pd.concat(family_frames, ignore_index=True, sort=False)
                combined.to_parquet(family_features_path, index=False)
                meta = {
                    "family_id": family_id,
                    "model": args.llm_model,
                    "n_features": run_n,
                    "providers": llm_providers,
                    "google_model": llm_google_model,
                    "seed_hash": seed_hash,
                    "seed_size": llm_engineered_seed_size,
                    "set_features": family_set_features,
                    "timestamp": datetime.now().isoformat(),
                }
                family_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                if family_dir.exists():
                    archives_dir.mkdir(parents=True, exist_ok=True)
                    archive_target = archives_dir / f"family_{family_id}"
                    if archive_target.exists():
                        shutil.rmtree(archive_target)
                    shutil.move(str(family_dir), str(archive_target))

            meta_path = BASE_DIR / "docs" / "llm_engineered_family_leaderboard_meta.json"
            meta = {
                "family_id": family_id,
                "n_sets": llm_engineered_run_family_size,
                "n_features": run_n,
                "model": args.llm_model,
                "cv_folds": cv_folds,
                "pool_size": len(labels),
                "seed_hash": seed_hash,
                "seed_size": llm_engineered_seed_size,
                "timestamp": datetime.now().isoformat(),
                "skipped_sets": skipped_sets,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            if lr_table2_rows:
                leaderboard_path = BASE_DIR / "docs" / "llm_engineered_family_leaderboard.md"
                leaderboard_csv = BASE_DIR / "docs" / "llm_engineered_family_leaderboard.csv"
                _write_family_leaderboard(
                    lr_table2_rows,
                    leaderboard_path,
                    leaderboard_csv,
                    cv_folds,
                    len(labels),
                )
            if xgb_table2_rows:
                leaderboard_path = BASE_DIR / "docs" / "llm_engineered_family_leaderboard_xgb.md"
                leaderboard_csv = BASE_DIR / "docs" / "llm_engineered_family_leaderboard_xgb.csv"
                _write_family_leaderboard(
                    xgb_table2_rows,
                    leaderboard_path,
                    leaderboard_csv,
                    cv_folds,
                    len(labels),
                )
                # Persist top-3 engineered-only sets by XGB F0.5 for paper pipeline selection.
                xgb_only = [
                    row
                    for row in xgb_table2_rows
                    if row.get("regression") == "LLM Engineered Only"
                ]
                if xgb_only:
                    xgb_only_sorted = sorted(
                        xgb_only,
                        key=lambda r: float(r.get("F0.5", float("-inf"))),
                        reverse=True,
                    )
                    top3 = [r.get("set_id", "") for r in xgb_only_sorted[:3] if r.get("set_id")]
                    top3_path = BASE_DIR / "docs" / "paper_stats" / "engineered_top3_xgb.json"
                    top3_path.parent.mkdir(parents=True, exist_ok=True)
                    top3_path.write_text(
                        json.dumps(
                            {
                                "family_id": family_id,
                                "metric": "XGB engineered-only F0.5",
                                "set_ids": top3,
                                "timestamp": datetime.now().isoformat(),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

    # Full-dataset HQ + Reasoning (no rule layer) and Full Mirror + Reasoning (rule layer).
    if use_llm_reasoning and reasoning_feature_names:
        exp_list_for_combo = ["A", "B", "C", "D", "E", "F"]
        if exp_list_for_combo:
            reasoning_full_df = _load_full_current_df(exp_list_for_combo)
            if reasoning_full_df is None:
                _log_run("Full reasoning cache missing; generating seed_100 reasoning for full dataset.")
                reasoning_full_df = _build_full_current_df(exp_list_for_combo)
            _validate_reasoning_full_df(reasoning_full_df, len(all_records))
            for col in reasoning_full_df.columns:
                if col not in ("founder_uuid", "success"):
                    reasoning_full_df[col] = pd.to_numeric(reasoning_full_df[col], errors="ignore")
            reasoning_full_combo_cols = _build_reasoning_combos(reasoning_full_df, exp_list_for_combo)
            reasoning_full_combo_frames = {
                combo: reasoning_full_df[cols].copy() for combo, cols in reasoning_full_combo_cols.items()
            }

            hq_script_full = (
                PROJECT_ROOT
                / "High_Quality_human_features"
                / "features"
                / "extract_structured.py"
            )
            hq_df_full_all = _build_high_quality_features(all_records, hq_script_full)
            missing_full = [f for f in HQ_FEATURES_BASE if f not in hq_df_full_all.columns]
            if missing_full:
                raise RuntimeError(
                    "High-quality feature extraction missing columns (full): "
                    + ", ".join(missing_full)
                )
            hq_full_no_gap_all = hq_df_full_all[HQ_FEATURES_BASE].copy()
            rule_mask_full = (
                hq_df_full_all["exit_count"].fillna(0.0).astype(float).values > 0
            )

            if run_profile != "xgb_mirror":
                # Full HQ + Reasoning (no rule layer).
                for tuning in ("oof",):
                    means, stds = _train_and_log_cv(
                        HQ_FEATURES_BASE,
                        hq_full_no_gap_all,
                        all_labels,
                        args_lr,
                        input_csv,
                        f"HQ Only (Full, LR, {tuning})",
                        cv_folds=cv_folds,
                        log_dir=BASE_DIR
                        / "training_logs"
                        / "full"
                        / "hq_only"
                        / f"lr_{tuning}",
                        cv_splits=cv_splits_full,
                        threshold_tuning=tuning,
                    )
                    _append_row(
                        full_lr_rows,
                        "Full HQ",
                        "HQ Only",
                        "",
                        "",
                        means,
                        stds,
                        variant="full_hq_no_rule",
                        tuning=tuning,
                    )

                    means, stds = _train_and_log_cv(
                        HQ_FEATURES_BASE,
                        hq_full_no_gap_all,
                        all_labels,
                        args_xgb,
                        input_csv,
                        f"HQ Only (Full, XGB, {tuning})",
                        cv_folds=cv_folds,
                        log_dir=BASE_DIR
                        / "training_logs"
                        / "full"
                        / "hq_only"
                        / f"xgb_{tuning}",
                        cv_splits=cv_splits_full,
                        threshold_tuning=tuning,
                    )
                    _append_row(
                        full_xgb_rows,
                        "Full HQ",
                        "HQ Only",
                        "",
                        "",
                        means,
                        stds,
                        variant="full_hq_no_rule",
                        tuning=tuning,
                    )

                    for combo, cols in reasoning_full_combo_cols.items():
                        combo_tag = combo.replace("+", "_")
                        combo_df_full = reasoning_full_combo_frames[combo]
                        combo_plus = pd.concat([hq_full_no_gap_all, combo_df_full], axis=1)
                        means, stds = _train_and_log_cv(
                            HQ_FEATURES_BASE + cols,
                            combo_plus,
                            all_labels,
                            args_lr,
                            input_csv,
                            f"HQ + Reasoning {combo} (Full, LR, {tuning})",
                            cv_folds=cv_folds,
                            log_dir=BASE_DIR
                            / "training_logs"
                            / "full"
                            / "hq_plus"
                            / combo_tag
                            / f"lr_{tuning}",
                            cv_splits=cv_splits_full,
                            threshold_tuning=tuning,
                        )
                        _append_row(
                            full_lr_rows,
                            "Full HQ",
                            "HQ + Reasoning",
                            "",
                            combo,
                            means,
                            stds,
                            variant="full_hq_no_rule",
                            tuning=tuning,
                        )

                        means, stds = _train_and_log_cv(
                            HQ_FEATURES_BASE + cols,
                            combo_plus,
                            all_labels,
                            args_xgb,
                            input_csv,
                            f"HQ + Reasoning {combo} (Full, XGB, {tuning})",
                            cv_folds=cv_folds,
                            log_dir=BASE_DIR
                            / "training_logs"
                            / "full"
                            / "hq_plus"
                            / combo_tag
                            / f"xgb_{tuning}",
                            cv_splits=cv_splits_full,
                            threshold_tuning=tuning,
                        )
                        _append_row(
                            full_xgb_rows,
                            "Full HQ",
                            "HQ + Reasoning",
                            "",
                            combo,
                            means,
                            stds,
                            variant="full_hq_no_rule",
                            tuning=tuning,
                        )

            # Full mirror (rule layer).
            for tuning in ("oof",):
                if run_profile != "xgb_mirror":
                    means, stds = _train_and_log_cv(
                        HQ_FEATURES_BASE,
                        hq_full_no_gap_all,
                        all_labels,
                        args_lr,
                        input_csv,
                        f"Mirror HQ Only (LR, {tuning})",
                        cv_folds=cv_folds,
                        log_dir=BASE_DIR
                        / "training_logs"
                        / "xgb_mirror"
                        / "hq_only"
                        / f"lr_{tuning}",
                        cv_splits=cv_splits_full,
                        rule_mask=rule_mask_full,
                        threshold_tuning=tuning,
                    )
                    _append_row(
                        full_lr_rows,
                        "Full Mirror",
                        "HQ Only",
                        "",
                        "",
                        means,
                        stds,
                        variant="full_mirror_rule",
                        tuning=tuning,
                    )

                means, stds = _train_and_log_cv(
                    HQ_FEATURES_BASE,
                    hq_full_no_gap_all,
                    all_labels,
                    args_xgb,
                    input_csv,
                    f"Mirror HQ Only (XGB, {tuning})",
                    cv_folds=cv_folds,
                    log_dir=BASE_DIR
                    / "training_logs"
                    / "xgb_mirror"
                    / "hq_only"
                    / f"xgb_{tuning}",
                    cv_splits=cv_splits_full,
                    rule_mask=rule_mask_full,
                    threshold_tuning=tuning,
                )
                _append_row(
                    full_xgb_rows,
                    "Full Mirror",
                    "HQ Only",
                    "",
                    "",
                    means,
                    stds,
                    variant="full_mirror_rule",
                    tuning=tuning,
                )

                for combo, cols in reasoning_full_combo_cols.items():
                    combo_tag = combo.replace("+", "_")
                    combo_df_full = reasoning_full_combo_frames[combo]
                    combo_plus = pd.concat([hq_full_no_gap_all, combo_df_full], axis=1)
                    if run_profile != "xgb_mirror":
                        means, stds = _train_and_log_cv(
                            HQ_FEATURES_BASE + cols,
                            combo_plus,
                            all_labels,
                            args_lr,
                            input_csv,
                            f"Mirror HQ + Reasoning {combo} (LR, {tuning})",
                            cv_folds=cv_folds,
                            log_dir=BASE_DIR
                            / "training_logs"
                            / "xgb_mirror"
                            / "hq_plus"
                            / combo_tag
                            / f"lr_{tuning}",
                            cv_splits=cv_splits_full,
                            rule_mask=rule_mask_full,
                            threshold_tuning=tuning,
                        )
                        _append_row(
                            full_lr_rows,
                            "Full Mirror",
                            "HQ + Reasoning",
                            "",
                            combo,
                            means,
                            stds,
                            variant="full_mirror_rule",
                            tuning=tuning,
                        )

                    means, stds = _train_and_log_cv(
                        HQ_FEATURES_BASE + cols,
                        combo_plus,
                        all_labels,
                        args_xgb,
                        input_csv,
                        f"Mirror HQ + Reasoning {combo} (XGB, {tuning})",
                        cv_folds=cv_folds,
                        log_dir=BASE_DIR
                        / "training_logs"
                        / "xgb_mirror"
                        / "hq_plus"
                        / combo_tag
                        / f"xgb_{tuning}",
                        cv_splits=cv_splits_full,
                        rule_mask=rule_mask_full,
                        threshold_tuning=tuning,
                    )
                    _append_row(
                        full_xgb_rows,
                        "Full Mirror",
                        "HQ + Reasoning",
                        "",
                        combo,
                        means,
                        stds,
                        variant="full_mirror_rule",
                        tuning=tuning,
                    )

    if (
        lr_table1_rows
        or lr_table2_rows
        or lr_table_hq_rows
        or xgb_table1_rows
        or xgb_table2_rows
        or xgb_table_hq_rows
        or full_lr_rows
        or full_xgb_rows
    ):
        report_path = BASE_DIR / "docs" / "llm_regression_report.md"
        full_csv = BASE_DIR / "docs" / "llm_full_results.csv"
        _write_full_results_report(
            lr_table1_rows,
            lr_table2_rows,
            lr_table_hq_rows,
            xgb_table1_rows,
            xgb_table2_rows,
            xgb_table_hq_rows,
            full_lr_rows,
            full_xgb_rows,
            report_path,
            full_csv,
            cv_folds,
            len(labels),
            len(all_labels),
        )
        _log(f"\nUpdated report: {report_path}")
        _log(f"Full results CSV: {full_csv}")
        _refresh_family_leaderboard_from_full_csv(
            full_csv,
            BASE_DIR / "docs" / "llm_engineered_family_leaderboard.md",
            BASE_DIR / "docs" / "llm_engineered_family_leaderboard.csv",
            cv_folds,
            len(labels),
        )
        _write_snapshot_combined_report()
        if args.snapshot_label:
            snapshot_dir = _write_run_snapshot(args.snapshot_label)
            _log(f"Snapshot saved to: {snapshot_dir}")
        return

    # Placeholder for future multiple training loops over feature subsets.
    # TODO: add loop over named feature sets and aggregate metrics.

    metrics_mean, metrics_std = _train_and_log_cv(
        feature_names,
        full_all,
        labels,
        args,
        input_csv,
        mode_label,
        cv_folds=cv_folds,
        log_dir=_log_dir_for_mode(mode, llm_reasoning_dry_run, llm_reasoning_dry_run_fast),
        cv_splits=cv_splits,
        rule_mask=None,
    )
    _log(f"\n  Features used: {', '.join(feature_names)}")
    _log(f"\n  [{mode_label}]   {len(feature_names)} features, CV={cv_folds} folds")
    _log(
        "  ROC-AUC={roc}  PR-AUC={pr}  Prec={prec}  Rec={rec}  F0.5={f0}  Acc={acc}".format(
            roc=_format_mean_std(metrics_mean["roc_auc"], metrics_std["roc_auc"]),
            pr=_format_mean_std(metrics_mean["pr_auc"], metrics_std["pr_auc"]),
            prec=_format_mean_std(metrics_mean["precision"], metrics_std["precision"]),
            rec=_format_mean_std(metrics_mean["recall"], metrics_std["recall"]),
            f0=_format_mean_std(metrics_mean["f0.5"], metrics_std["f0.5"]),
            acc=_format_mean_std(metrics_mean["accuracy"], metrics_std["accuracy"]),
        )
    )


def _write_log(
    lines: list[str],
    args: argparse.Namespace,
    input_csv: str,
    feature_names: list[str],
    n_train: int,
    n_test: int,
    pos_train: int,
    pos_test: int,
    metrics: dict[str, float],
    log_dir: Path | None = None,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = BASE_DIR / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_log_{ts}.txt"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        print(f"\n  Training log saved to: {log_path}")
    except OSError:
        pass

    report = {
        "dataset": input_csv,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "features": feature_names,
        "split": {"train": n_train, "test": n_test},
        "positives": {"train": pos_train, "test": pos_test},
        "threshold": metrics.get("threshold"),
        "metrics": {
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f0.5": metrics.get("f0.5"),
            "accuracy": metrics.get("accuracy"),
            "precision@1%": metrics.get("precision@1%"),
            "precision@5%": metrics.get("precision@5%"),
            "precision@10%": metrics.get("precision@10%"),
            "fnr": metrics.get("fnr"),
            "tp": metrics.get("tp"),
            "fn": metrics.get("fn"),
        },
    }
    report_path = log_dir / f"run_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        print(f"  Run report saved to: {report_path}")
    except OSError:
        pass


def _log_dir_for_mode(
    mode: str,
    dry_run: bool,
    dry_run_fast: bool,
) -> Path:
    root = BASE_DIR / "training_logs"
    if dry_run or dry_run_fast:
        return root / "dry_runs" / mode
    if mode == "human":
        return root / "human"
    if mode == "llm":
        return root / "llm_engineered"
    if mode == "reasoning":
        return root / "llm_reasoning"
    if mode == "hybrid":
        return root / "hybrid"
    return root


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        try:
            _log_exception(exc)
        except Exception:
            pass
        raise




