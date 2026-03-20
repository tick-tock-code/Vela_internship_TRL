from __future__ import annotations

import json
import math
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, fbeta_score


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


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_csv_path() -> Path:
    root = project_root().parent
    return root / "VCBench-Starter-Kit" / "vcbench_final_public.csv"


def _load_high_quality_extractor(script_path: Path):
    spec = importlib.util.spec_from_file_location("hq_extract_structured", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load HQ feature script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    if not hasattr(module, "extract_features"):
        raise AttributeError(f"HQ script missing extract_features: {script_path}")
    return module.extract_features  # type: ignore[attr-defined]


def load_structured_features(csv_path: Path) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    df = pd.read_csv(csv_path)
    script_path = (
        project_root().parent
        / "High_Quality_human_features"
        / "features"
        / "extract_structured.py"
    )
    extractor = _load_high_quality_extractor(script_path)
    hq_df = extractor(df)
    missing = [f for f in HQ_FEATURES_BASE if f not in hq_df.columns]
    if missing:
        raise RuntimeError("Missing HQ features: " + ", ".join(missing))
    X = hq_df[HQ_FEATURES_BASE].fillna(0.0)
    y = df["success"].to_numpy()
    founder_ids = df["founder_uuid"]
    return X, y, founder_ids


def apply_rule_override(probs: np.ndarray, exit_count: pd.Series) -> np.ndarray:
    adjusted = probs.copy()
    mask = exit_count.fillna(0.0).astype(float).values > 0
    adjusted[mask] = 1.0
    return adjusted


def train_xgb(X_train: pd.DataFrame, y_train: np.ndarray):
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xgboost is required for mirror experiments. Install with: pip install xgboost"
        ) from exc
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_min: float = 0.30,
    t_max: float = 0.95,
    steps: int = 50,
    grid_mode: str = "custom",
) -> list[dict[str, Any]]:
    results = []
    if grid_mode == "mirror":
        thresholds = np.linspace(0.30, 0.95, 50)
    else:
        thresholds = np.linspace(t_min, t_max, steps)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
        positive_rate = float(np.mean(y_pred))
        results.append(
            {
                "f05": float(f05),
                "precision": float(precision),
                "recall": float(recall),
                "positive_rate": float(positive_rate),
                "n_predicted_positive": int(np.sum(y_pred)),
                "threshold": float(t),
            }
        )
    return results


def best_threshold(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No threshold results to select from.")
    return sorted(results, key=lambda r: r["f05"], reverse=True)[0]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_fold_cache(path: Path, expected_rows: int, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing fold cache ({label}): {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    fold_map = data.get("fold_map", {})
    if not isinstance(fold_map, dict):
        raise RuntimeError(f"Invalid fold cache format ({label}): {path}")
    if len(fold_map) != expected_rows:
        raise RuntimeError(
            f"Fold cache size mismatch ({label}): "
            f"expected {expected_rows}, got {len(fold_map)}"
        )
    missing_uuid_count = int(data.get("missing_uuid_count", 0))
    if missing_uuid_count:
        print(f"[validation] {label}: missing_uuid_count={missing_uuid_count} (row_index fallback)")


def validate_run_state(expected_pool: int, expected_full: int) -> None:
    folds_dir = project_root() / "features_storage" / "cv_folds"
    pool_path = folds_dir / "folds_k5_seed42.json"
    full_path = folds_dir / "folds_k5_seed42_full.json"
    _validate_fold_cache(pool_path, expected_pool, "pool")
    _validate_fold_cache(full_path, expected_full, "full")
