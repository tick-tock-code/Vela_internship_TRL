from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from mirror_common import (
    HQ_FEATURES_BASE,
    apply_rule_override,
    best_threshold,
    default_csv_path,
    load_structured_features,
    sweep_thresholds,
    train_xgb,
    write_json,
)

_EXTRACTOR = None


def _load_hq_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is not None:
        return _EXTRACTOR
    script_path = (
        Path(__file__).resolve().parents[1]
        / "High_Quality_human_features"
        / "features"
        / "extract_structured.py"
    )
    if not script_path.exists():
        raise FileNotFoundError(f"HQ feature extractor not found: {script_path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("hq_extract_structured", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load HQ feature script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    if not hasattr(module, "extract_features"):
        raise AttributeError("HQ script missing extract_features")
    _EXTRACTOR = module.extract_features  # type: ignore[attr-defined]
    return _EXTRACTOR


def _extract_hq_features(df: pd.DataFrame) -> pd.DataFrame:
    extractor = _load_hq_extractor()
    hq_df = extractor(df)
    missing = [f for f in HQ_FEATURES_BASE if f not in hq_df.columns]
    if missing:
        raise RuntimeError("Missing HQ features: " + ", ".join(missing))
    return hq_df[HQ_FEATURES_BASE].fillna(0.0)


def _auto_find_test_csv(test_dir: Path) -> Path:
    csvs = sorted([p for p in test_dir.glob("*.csv") if p.is_file()])
    if len(csvs) == 1:
        return csvs[0]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {test_dir}")
    names = ", ".join([c.name for c in csvs])
    raise RuntimeError(f"Multiple CSV files found in {test_dir}: {names}")


def _load_or_create_public_splits(
    public_csv: Path, split_dir: Path, seed: int, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_csv = split_dir / "public_train.csv"
    val_csv = split_dir / "public_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        full_df = pd.read_csv(public_csv)
        train_df, val_df = train_test_split(
            full_df,
            test_size=test_size,
            stratify=full_df["success"],
            random_state=seed,
        )
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
    return pd.read_csv(train_csv), pd.read_csv(val_csv)


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib  # type: ignore

        joblib.dump(model, path)
    except Exception:
        import pickle

        with path.open("wb") as f:
            pickle.dump(model, f)


def _apply_rule(scores: np.ndarray, feature_df: pd.DataFrame) -> np.ndarray:
    return apply_rule_override(scores, feature_df["exit_count"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test-set predictions for mirror XGBoost variants."
    )
    parser.add_argument("--public_csv", default=str(default_csv_path()))
    parser.add_argument("--test_csv", default="")
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--t_min", type=float, default=0.30)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--mirror_grid",
        action="store_true",
        help="Use exact mirror grid 0.30-0.95 (50 steps).",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    test_dir = base / "test_dataset"
    split_dir = Path(__file__).resolve().parent / "splits"
    outputs_dir = Path(__file__).resolve().parent / "outputs" / "oof_model"

    public_csv = Path(args.public_csv)
    test_csv = Path(args.test_csv) if args.test_csv else _auto_find_test_csv(test_dir)
    out_csv = Path(args.out_csv) if args.out_csv else test_dir / "xgb_mirror_test_predictions.csv"

    if not public_csv.exists():
        raise FileNotFoundError(f"Public CSV not found: {public_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    test_df = pd.read_csv(test_csv)
    if "founder_uuid" not in test_df.columns:
        raise RuntimeError("Test CSV missing founder_uuid column.")
    test_ids = test_df["founder_uuid"].astype(str)
    X_test = _extract_hq_features(test_df)

    # Full public data for CV + OOF tuning
    X_full, y_full, _ = load_structured_features(public_csv)

    # 1) val_split_tune (train on public_train, tune on public_val)
    train_df, val_df = _load_or_create_public_splits(
        public_csv=public_csv,
        split_dir=split_dir,
        seed=args.seed,
        test_size=args.test_size,
    )
    X_train = _extract_hq_features(train_df)
    y_train = train_df["success"].to_numpy()
    X_val = _extract_hq_features(val_df)
    y_val = val_df["success"].to_numpy()

    val_model = train_xgb(X_train, y_train)
    val_probs = _apply_rule(val_model.predict_proba(X_val)[:, 1], X_val)
    val_sweep = sweep_thresholds(
        y_val,
        val_probs,
        t_min=args.t_min,
        t_max=args.t_max,
        steps=args.steps,
        grid_mode="mirror" if args.mirror_grid else "custom",
    )
    val_best = best_threshold(val_sweep)
    val_threshold = float(val_best["threshold"])
    val_test_probs = _apply_rule(val_model.predict_proba(X_test)[:, 1], X_test)
    val_split_preds = (val_test_probs >= val_threshold).astype(int)

    # Shared CV splits
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(X_full, y_full))

    # 2) CV_val_tune (per-fold threshold, majority vote)
    votes = np.zeros(len(X_test), dtype=int)
    cv_val_thresholds: list[float] = []
    for train_idx, val_idx in splits:
        X_tr = X_full.iloc[train_idx].copy()
        y_tr = y_full[train_idx]
        X_va = X_full.iloc[val_idx].copy()
        y_va = y_full[val_idx]

        fold_model = train_xgb(X_tr, y_tr)
        fold_val_probs = _apply_rule(fold_model.predict_proba(X_va)[:, 1], X_va)
        fold_sweep = sweep_thresholds(
            y_va,
            fold_val_probs,
            t_min=args.t_min,
            t_max=args.t_max,
            steps=args.steps,
            grid_mode="mirror" if args.mirror_grid else "custom",
        )
        fold_best = best_threshold(fold_sweep)
        fold_threshold = float(fold_best["threshold"])
        cv_val_thresholds.append(fold_threshold)

        fold_test_probs = _apply_rule(fold_model.predict_proba(X_test)[:, 1], X_test)
        votes += (fold_test_probs >= fold_threshold).astype(int)

    majority_cutoff = (args.folds // 2) + 1
    cv_val_preds = (votes >= majority_cutoff).astype(int)

    # 3) CV_OOF_tune (global OOF threshold, full-data model)
    oof_scores: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    for train_idx, val_idx in splits:
        X_tr = X_full.iloc[train_idx].copy()
        y_tr = y_full[train_idx]
        X_va = X_full.iloc[val_idx].copy()
        y_va = y_full[val_idx]

        fold_model = train_xgb(X_tr, y_tr)
        fold_val_probs = _apply_rule(fold_model.predict_proba(X_va)[:, 1], X_va)
        oof_scores.append(fold_val_probs)
        oof_labels.append(y_va)

    all_scores = np.concatenate(oof_scores)
    all_labels = np.concatenate(oof_labels)
    oof_sweep = sweep_thresholds(
        all_labels,
        all_scores,
        t_min=args.t_min,
        t_max=args.t_max,
        steps=args.steps,
        grid_mode="mirror" if args.mirror_grid else "custom",
    )
    oof_best = best_threshold(oof_sweep)
    oof_threshold = float(oof_best["threshold"])

    oof_model = train_xgb(X_full, y_full)
    oof_test_probs = _apply_rule(oof_model.predict_proba(X_test)[:, 1], X_test)
    oof_preds = (oof_test_probs >= oof_threshold).astype(int)

    # Save OOF artifacts for reuse
    model_path = outputs_dir / "xgb_model.pkl"
    threshold_path = outputs_dir / "oof_threshold.json"
    _save_model(oof_model, model_path)
    write_json(
        threshold_path,
        {
            "threshold": oof_threshold,
            "folds": args.folds,
            "seed": args.seed,
            "grid": "mirror" if args.mirror_grid else "custom",
            "t_min": args.t_min,
            "t_max": args.t_max,
            "steps": args.steps,
            "public_csv": str(public_csv),
            "saved_at": datetime.now().isoformat(),
        },
    )

    out_df = pd.DataFrame(
        {
            "founder_uuid": test_ids,
            "val_split_tune": val_split_preds.astype(int),
            "CV_val_tune": cv_val_preds.astype(int),
            "CV_OOF_tune": oof_preds.astype(int),
        }
    )
    out_df.to_csv(out_csv, index=False)

    print("Mirror XGB test predictions complete.")
    print(f"Test CSV: {test_csv}")
    print(f"Output CSV: {out_csv}")
    print(f"val_split threshold: {val_threshold:.4f}")
    print(
        f"CV_val thresholds (mean): {float(np.mean(cv_val_thresholds)):.4f} "
        f"[min={float(np.min(cv_val_thresholds)):.4f}, max={float(np.max(cv_val_thresholds)):.4f}]"
    )
    print(f"OOF threshold: {oof_threshold:.4f}")
    print(f"OOF model saved: {model_path}")
    print(f"OOF threshold saved: {threshold_path}")


if __name__ == "__main__":
    main()
