from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from mirror_common import (
    apply_rule_override,
    best_threshold,
    default_csv_path,
    load_structured_features,
    sweep_thresholds,
    train_xgb,
    validate_run_state,
    write_json,
    write_text,
)


def _load_or_create_split(
    y: np.ndarray,
    split_path: Path,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if split_path.exists():
        data = json.loads(split_path.read_text(encoding="utf-8"))
        train_idx = np.array(data["train_idx"], dtype=int)
        val_idx = np.array(data["val_idx"], dtype=int)
        return train_idx, val_idx, data

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
    meta = {
        "random_state": random_state,
        "test_size": test_size,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "positive_rate": float(y.mean()),
    }
    write_json(split_path, meta)
    return train_idx, val_idx, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror holdout (XGB stumps + prior_exit rule).")
    parser.add_argument("--csv_path", default=str(default_csv_path()))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--t_min", type=float, default=0.30)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mirror_grid", action="store_true", help="Use exact mirror grid 0.30-0.95 (50 steps).")
    parser.add_argument(
        "--use_split_csv",
        action="store_true",
        help="Create/use public_train.csv and public_val.csv in mirror_experiments/splits (seed=42, 80/20).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    base = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else base / "outputs" / "holdout"
    split_dir = base / "splits"
    split_path = split_dir / f"holdout_seed{args.seed}.json"

    X, y, founder_ids = load_structured_features(csv_path)
    exit_count = X["exit_count"]
    validate_run_state(expected_pool=len(y) - 100, expected_full=len(y))

    if args.use_split_csv:
        split_dir.mkdir(parents=True, exist_ok=True)
        train_csv = split_dir / "public_train.csv"
        val_csv = split_dir / "public_val.csv"
        if not train_csv.exists() or not val_csv.exists():
            from sklearn.model_selection import train_test_split

            full_df = pd.read_csv(csv_path)
            train_df, val_df = train_test_split(
                full_df,
                test_size=args.test_size,
                stratify=full_df["success"],
                random_state=args.seed,
            )
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)
        # Rebuild indices from saved CSVs to ensure exact mirror behavior
        full_df = pd.read_csv(csv_path)
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        full_df["_row_index"] = np.arange(len(full_df))
        train_idx = (
            full_df.merge(train_df[["founder_uuid"]], on="founder_uuid", how="inner")
            ["_row_index"]
            .to_numpy()
        )
        val_idx = (
            full_df.merge(val_df[["founder_uuid"]], on="founder_uuid", how="inner")
            ["_row_index"]
            .to_numpy()
        )
        split_meta = {
            "mode": "csv_split",
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "random_state": args.seed,
            "test_size": args.test_size,
        }
    else:
        train_idx, val_idx, split_meta = _load_or_create_split(
            y=y,
            split_path=split_path,
            test_size=args.test_size,
            random_state=args.seed,
        )

    X_train = X.iloc[train_idx].copy()
    y_train = y[train_idx]
    X_val = X.iloc[val_idx].copy()
    y_val = y[val_idx]

    model = train_xgb(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]
    val_probs = apply_rule_override(val_probs, exit_count.iloc[val_idx])

    sweep = sweep_thresholds(
        y_val,
        val_probs,
        t_min=args.t_min,
        t_max=args.t_max,
        steps=args.steps,
        grid_mode="mirror" if args.mirror_grid else "custom",
    )
    best = best_threshold(sweep)

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(out_dir / "holdout_sweep.csv", index=False)

    summary = {
        "mode": "holdout",
        "csv_path": str(csv_path),
        "seed": args.seed,
        "test_size": args.test_size,
        "mirror_grid": bool(args.mirror_grid),
        "split_file": str(split_path),
        "split_mode": split_meta.get("mode", "indices"),
        "n_rows": int(len(y)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "positive_rate": float(y.mean()),
        "best": best,
    }
    write_json(out_dir / "holdout_metrics.json", summary)

    lines = [
        "Mirror Holdout (XGB stumps + prior_exit rule)",
        f"CSV: {csv_path}",
        f"Seed: {args.seed}  Test size: {args.test_size}",
        f"Train rows: {len(train_idx)}  Val rows: {len(val_idx)}",
        f"Grid: {'mirror(0.30-0.95,50)' if args.mirror_grid else f'custom({args.t_min}-{args.t_max},{args.steps})'}",
        f"Split mode: {split_meta.get('mode', 'indices')}",
        "",
        "Best threshold (tuned on val):",
        f"  F0.5={best['f05']:.4f}",
        f"  Precision={best['precision']:.4f}",
        f"  Recall={best['recall']:.4f}",
        f"  Positive rate={best['positive_rate']:.4f}",
        f"  N predicted positive={best['n_predicted_positive']}",
        f"  Threshold={best['threshold']:.4f}",
    ]
    write_text(out_dir / "holdout_metrics.txt", lines)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
