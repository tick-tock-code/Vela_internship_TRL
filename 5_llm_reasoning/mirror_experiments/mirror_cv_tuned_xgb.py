from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Mirror CV tuned (XGB stumps + prior_exit rule).")
    parser.add_argument("--csv_path", default=str(default_csv_path()))
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_min", type=float, default=0.30)
    parser.add_argument("--t_max", type=float, default=0.95)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--mirror_grid", action="store_true", help="Use exact mirror grid 0.30-0.95 (50 steps).")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    base = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir) if args.out_dir else base / "outputs" / "cv_tuned"
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_structured_features(csv_path)
    exit_count = X["exit_count"]
    validate_run_state(expected_pool=len(y) - 100, expected_full=len(y))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
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
        row = {
            "fold": float(fold_idx),
            "f05": float(best["f05"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "positive_rate": float(best["positive_rate"]),
            "threshold": float(best["threshold"]),
        }
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "cv_tuned_folds.csv", index=False)

    summary = {
        "mode": "cv_tuned",
        "csv_path": str(csv_path),
        "folds": args.folds,
        "seed": args.seed,
        "mirror_grid": bool(args.mirror_grid),
        "n_rows": int(len(y)),
        "mean": fold_df.mean(numeric_only=True).to_dict(),
        "std": fold_df.std(numeric_only=True).to_dict(),
    }
    write_json(out_dir / "cv_tuned_summary.json", summary)

    lines = [
        "Mirror CV Tuned (XGB stumps + prior_exit rule)",
        f"CSV: {csv_path}",
        f"Folds: {args.folds}  Seed: {args.seed}",
        f"Grid: {'mirror(0.30-0.95,50)' if args.mirror_grid else f'custom({args.t_min}-{args.t_max},{args.steps})'}",
        "",
        "Mean +/- Std (threshold tuned on each validation fold):",
        f"  F0.5={summary['mean']['f05']:.4f} +/- {summary['std']['f05']:.4f}",
        f"  Precision={summary['mean']['precision']:.4f} +/- {summary['std']['precision']:.4f}",
        f"  Recall={summary['mean']['recall']:.4f} +/- {summary['std']['recall']:.4f}",
        f"  Positive rate={summary['mean']['positive_rate']:.4f} +/- {summary['std']['positive_rate']:.4f}",
        f"  Threshold={summary['mean']['threshold']:.4f} +/- {summary['std']['threshold']:.4f}",
    ]
    write_text(out_dir / "cv_tuned_summary.txt", lines)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
