from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from itertools import combinations

from lib.feature_registry import FEATURE_REGISTRY
from lib.paths import BASE_DIR, CONFIG_DIR


DOCS_DIR = BASE_DIR / "docs"
RESULTS_CSV = DOCS_DIR / "llm_full_results.csv"
FOLDS_PATH = BASE_DIR / "features_storage" / "cv_folds" / "folds_k10_seed42.json"
HUMAN_PATH = BASE_DIR / "features_storage" / "features_full.parquet"
REASONING_PATH = (
    BASE_DIR
    / "features_storage"
    / "llm_reasoning"
    / "currently_in_use"
    / "llm_reasoning_full.parquet"
)
FAMILY_DIR = BASE_DIR / "features_storage" / "llm_engineered" / "families"
FEATURES_JSON = CONFIG_DIR / "features.json"
RANDOM_STATE = 42


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
    standardize: bool = False,
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
        if standardize:
            X_train, X_test = _standardize_continuous(X_train, X_test, feature_names)
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


def _format_full_weights(df: pd.DataFrame) -> list[str]:
    lines = [
        "| feature | coef_mean | coef_std |",
        "|---|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['feature']} | {row['coef_mean']:+.4f} | {row['coef_std']:.4f} |"
        )
    return lines


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


def _load_human_df() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(HUMAN_PATH)
    if "row_index" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_index"] = df.index
    config = json.loads(FEATURES_JSON.read_text(encoding="utf-8-sig"))
    feature_names = config.get("features", [])
    if not feature_names:
        raise RuntimeError("No human features found in features.json.")
    return df[["row_index", "success"] + feature_names], feature_names


def _load_engineered_family() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    features_path, meta_path = _latest_family()
    df = pd.read_parquet(features_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    set_features = meta.get("set_features", {})
    if "row_index" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_index"] = df.index
    return df, set_features


def _best_sets(results: pd.DataFrame) -> tuple[str, str, dict[str, str]]:
    table2 = results[results["table"] == "Table 2"].copy()
    table2["reasoning_combo"] = table2["reasoning_combo"].fillna("")
    best_engineered = table2[table2["reasoning_combo"] == ""].sort_values(
        "F0.5_mean", ascending=False
    ).iloc[0]["set_id"]
    best_overall_row = table2[table2["reasoning_combo"] != ""].sort_values(
        "F0.5_mean", ascending=False
    ).iloc[0]
    best_combo = str(best_overall_row["reasoning_combo"])
    best_combo_set = str(best_overall_row["set_id"])
    best_per_combo: dict[str, str] = {}
    for combo in sorted(table2["reasoning_combo"].unique()):
        if not combo:
            continue
        subset = table2[table2["reasoning_combo"] == combo]
        best_per_combo[combo] = str(
            subset.sort_values("F0.5_mean", ascending=False).iloc[0]["set_id"]
        )
    return best_engineered, best_combo_set, best_per_combo


def main() -> int:
    results = pd.read_csv(RESULTS_CSV)
    engineered_df, set_features = _load_engineered_family()
    reasoning_df, reasoning_combos = _load_reasoning_df()
    human_df, human_features = _load_human_df()

    best_engineered_set, best_combo_set, best_per_combo = _best_sets(results)

    # Build fold ids
    fold_ids = _load_fold_ids(FOLDS_PATH, len(human_df))

    lines: list[str] = []
    lines.append("# Feature Weight Summary (CV fold coefficients)")
    lines.append("")
    lines.append("*CV: 10-fold stratified on 4400 founders (seed excluded).*")
    lines.append("")

    # Human weights
    human_X = human_df[human_features].copy()
    y = human_df["success"].to_numpy()
    human_stats = _cv_coef_stats(human_X, y, human_features, fold_ids, standardize=True)
    lines.append("## Human Features")
    lines.extend(_format_full_weights(human_stats))
    lines.append("")

    # Best engineered-only
    lines.append("## Best Engineered-Only Set")
    lines.append(f"Set: {best_engineered_set}")
    set_df = engineered_df[engineered_df["set_id"] == best_engineered_set].copy()
    set_df = set_df.sort_values("row_index").reset_index(drop=True)
    eng_features = set_features.get(best_engineered_set, [])
    eng_stats = _cv_coef_stats(
        set_df[eng_features], set_df["success"].to_numpy(), eng_features, fold_ids, standardize=False
    )
    lines.extend(_format_full_weights(eng_stats))
    lines.append("")

    # Best engineered + reasoning overall
    best_combo = results[results["table"] == "Table 2"]
    best_combo = best_combo[best_combo["reasoning_combo"].notna()]
    best_combo = best_combo[best_combo["reasoning_combo"] != ""].sort_values(
        "F0.5_mean", ascending=False
    ).iloc[0]
    best_combo_name = str(best_combo["reasoning_combo"])
    lines.append("## Best Engineered + Reasoning")
    lines.append(f"Set: {best_combo_set}  Combo: {best_combo_name}")
    best_combo_df = engineered_df[engineered_df["set_id"] == best_combo_set].copy()
    best_combo_df = best_combo_df.sort_values("row_index").reset_index(drop=True)
    best_combo_features = set_features.get(best_combo_set, [])
    combo_cols = reasoning_combos.get(best_combo_name, [])
    merged = best_combo_df.merge(
        reasoning_df[["row_index"] + combo_cols], on="row_index", how="inner"
    )
    combo_features = best_combo_features + combo_cols
    combo_stats = _cv_coef_stats(
        merged[combo_features],
        merged["success"].to_numpy(),
        combo_features,
        fold_ids,
        standardize=False,
    )
    lines.extend(_format_full_weights(combo_stats))
    lines.append("")

    # Best engineered + reasoning per combo
    lines.append("## Best Engineered + Reasoning per Combo")
    for combo in sorted(reasoning_combos.keys()):
        best_set = best_per_combo.get(combo)
        if not best_set:
            continue
        lines.append(f"### Combo: {combo}")
        lines.append(f"Set: {best_set}")
        set_df = engineered_df[engineered_df["set_id"] == best_set].copy()
        set_df = set_df.sort_values("row_index").reset_index(drop=True)
        eng_feats = set_features.get(best_set, [])
        cols = eng_feats + reasoning_combos[combo]
        merged = set_df.merge(
            reasoning_df[["row_index"] + reasoning_combos[combo]], on="row_index", how="inner"
        )
        stats = _cv_coef_stats(
            merged[cols],
            merged["success"].to_numpy(),
            cols,
            fold_ids,
            standardize=False,
        )
        lines.extend(_format_full_weights(stats))
        lines.append("")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "feature_weight_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
