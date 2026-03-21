from __future__ import annotations

from pathlib import Path
import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from paths import BASE_DIR

DOCS_DIR = BASE_DIR / "docs" / "proper_run"
RESULTS_CSV = BASE_DIR / "docs" / "llm_full_results.csv"
FOLDS_PATH = BASE_DIR / "features_storage" / "cv_folds" / "folds_k10_seed42.json"
REASONING_PATH = (
    BASE_DIR
    / "features_storage"
    / "llm_reasoning"
    / "currently_in_use"
    / "llm_reasoning_full.parquet"
)
FAMILY_DIR = BASE_DIR / "features_storage" / "llm_engineered" / "families"

RANDOM_STATE = 42
TOP_K = 10


def _latest_family() -> tuple[Path, Path]:
    candidates = sorted(
        FAMILY_DIR.glob("family_*_features.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("No engineered family features found.")
    features_path = candidates[0]
    meta_path = features_path.with_name(features_path.name.replace("_features.parquet", "_meta.json"))
    if not meta_path.exists():
        raise RuntimeError("Family meta file not found for latest family.")
    return features_path, meta_path


def _load_fold_ids(path: Path, n_rows: int) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    fold_map = data.get("fold_map", {})
    fold_ids = np.full(n_rows, -1, dtype=int)
    for i in range(n_rows):
        fold_ids[i] = int(fold_map.get(f"row_{i}", -1))
    if (fold_ids < 0).any():
        raise RuntimeError("Fold map is missing entries for some rows.")
    return fold_ids


def _cv_coef_stats(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    fold_ids: np.ndarray,
) -> pd.DataFrame:
    coefs: list[np.ndarray] = []
    for fold in sorted(set(fold_ids.tolist())):
        train_idx = np.where(fold_ids != fold)[0]
        test_idx = np.where(fold_ids == fold)[0]
        X_train = df.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()
        if X_train.isna().any().any() or X_test.isna().any().any():
            fill_values = X_train.mean()
            X_train = X_train.fillna(fill_values)
            X_test = X_test.fillna(fill_values)
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train.values.astype(float), y[train_idx])
        coefs.append(model.coef_[0])
    coef_mat = np.vstack(coefs)
    mean = coef_mat.mean(axis=0)
    std = coef_mat.std(axis=0)
    out = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_mean": mean,
            "coef_std": std,
            "abs_mean": np.abs(mean),
        }
    )
    out.sort_values("abs_mean", ascending=False, inplace=True)
    return out


def _load_reasoning_df() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = pd.read_parquet(REASONING_PATH)
    if "row_index" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_index"] = df.index
    numeric_cols = [
        c
        for c in df.columns
        if c not in ("founder_uuid", "success", "row_index")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    exp_ids = ["A", "B", "E"]
    exp_to_cols: dict[str, list[str]] = {}
    for exp_id in exp_ids:
        cols = [c for c in numeric_cols if c.startswith(f"{exp_id}_")]
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
    return df[["row_index", "success"] + numeric_cols], combos


def _load_engineered_family() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    features_path, meta_path = _latest_family()
    df = pd.read_parquet(features_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    set_features = meta.get("set_features", {})
    if "row_index" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_index"] = df.index
    return df, set_features


def _format_reasoning_weights(df: pd.DataFrame) -> list[str]:
    lines = [
        "| feature | coef_mean | coef_std |",
        "|---|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['feature']} | {row['coef_mean']:+.4f} | {row['coef_std']:.4f} |"
        )
    return lines


def main() -> int:
    results = pd.read_csv(RESULTS_CSV)
    table2 = results[results["table"] == "Table 2"].copy()
    table2["reasoning_combo"] = table2["reasoning_combo"].fillna("")
    table2 = table2[table2["reasoning_combo"] != ""]
    table2 = table2.sort_values("F0.5_mean", ascending=False).head(TOP_K)

    engineered_df, set_features = _load_engineered_family()
    reasoning_df, reasoning_combos = _load_reasoning_df()

    fold_ids = _load_fold_ids(FOLDS_PATH, len(reasoning_df))

    lines: list[str] = []
    lines.append("# Reasoning Feature Weights (Top 10 Runs)")
    lines.append("")
    lines.append("This report lists only the reasoning-feature weights for the Top 10 engineered+reasoning runs by F0.5 (Table 2).")
    lines.append("")

    for rank, (_, row) in enumerate(table2.iterrows(), start=1):
        set_id = str(row["set_id"])
        combo = str(row["reasoning_combo"])
        f05_mean = float(row["F0.5_mean"])
        f05_std = float(row.get("F0.5_std", np.nan))
        f05_str = f"{f05_mean:.3f}"
        if not np.isnan(f05_std):
            f05_str = f"{f05_mean:.3f}+/-{f05_std:.3f}"
        lines.append(f"## {rank}. {set_id} | LLM Engineered + Reasoning [{combo}] | F0.5={f05_str}")

        combo_cols = reasoning_combos.get(combo, [])
        if not combo_cols:
            lines.append("No reasoning columns found for this combo.")
            lines.append("")
            continue

        set_df = engineered_df[engineered_df["set_id"] == set_id].copy()
        set_df = set_df.sort_values("row_index").reset_index(drop=True)
        eng_features = set_features.get(set_id, [])
        merged = set_df.merge(
            reasoning_df[["row_index"] + combo_cols], on="row_index", how="inner"
        )
        merged = merged.sort_values("row_index").reset_index(drop=True)
        all_features = eng_features + combo_cols
        stats = _cv_coef_stats(
            merged[all_features],
            merged["success"].to_numpy(),
            all_features,
            fold_ids,
        )
        reasoning_stats = stats[stats["feature"].isin(combo_cols)].copy()
        lines.extend(_format_reasoning_weights(reasoning_stats))
        lines.append("")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "reasoning_feature_weights_top10.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

