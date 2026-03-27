import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif

BASE_DIR = Path(__file__).resolve().parents[2]
PYTHON_DIR = BASE_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.append(str(PYTHON_DIR))

from think_reason_learn.datasets import load_vcbench
from lib.cv_folds import load_or_create_folds
from pipelines import model_testing_pipeline as mtp


def _select_counts_for_combo(
    base_df: pd.DataFrame,
    combo_cols: list[str],
    reasoning_df: pd.DataFrame,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    model_type: str,
) -> tuple[Counter, list[str]]:
    if combo_cols:
        df = pd.concat([base_df, reasoning_df[combo_cols]], axis=1)
    else:
        df = base_df.copy()
    feature_names = list(df.columns)
    counts: Counter = Counter()

    for train_idx, _ in splits:
        X_train = df.iloc[train_idx].copy()
        X_train, _, selected = mtp._preprocess_features(
            X_train,
            X_train.copy(),
            feature_names,
            model_type,
            "SFT",
            labels[train_idx],
            mtp.PCA_VARIANCE_DEFAULT,
            mtp.PLS_COMPONENTS_DEFAULT,
            k,
        )
        counts.update(selected)

    return counts, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--model_type", type=str, default="logistic")
    args = parser.parse_args()
    model_type = str(args.model_type).strip().lower()
    if model_type not in {"logistic", "xgb1"}:
        raise RuntimeError("--model_type must be 'logistic' or 'xgb1'")

    base_dir = BASE_DIR
    input_csv = base_dir.parent / "VCBench-Starter-Kit" / "vcbench_final_public.csv"
    records, labels = load_vcbench(input_csv, "success", 0, 42)
    founder_ids = [r.get("founder_uuid") for r in records]

    reasoning_full = pd.read_parquet(
        base_dir / "features_storage" / "llm_reasoning" / "full_current" / "llm_reasoning_full.parquet"
    )
    reasoning_df = mtp._reindex_reasoning(reasoning_full, founder_ids)
    for col in reasoning_df.columns:
        if col not in ("founder_uuid", "success", "row_index"):
            reasoning_df[col] = pd.to_numeric(reasoning_df[col], errors="ignore")

    combos = {"HQ": []}
    combos.update(mtp._build_reasoning_combos_numeric(reasoning_df, ["A", "B", "C", "D", "E", "F"]))
    allowed_combos = ["HQ", "D", "A+E", "A+D+E+F"]
    combos = {k: v for k, v in combos.items() if k in allowed_combos}

    hq_script = base_dir.parent / "High_Quality_human_features" / "features" / "extract_structured.py"
    hq_df_full = mtp._build_high_quality_features(records, hq_script)
    hq_full_no_gap = hq_df_full[mtp.HQ_FEATURES_BASE].copy()

    folds_path = base_dir / "features_storage" / "cv_folds" / "folds_k5_seed42_full.json"
    splits, _, _ = load_or_create_folds(
        founder_ids=founder_ids,
        labels=labels,
        cv_folds=5,
        random_state=42,
        folds_path=folds_path,
        dataset_label=f"{input_csv}_full",
        use_fixed=True,
    )

    out_dir = base_dir / "docs" / "model_testing" / "SFT_selection_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [f"# SFT Selection Summary (k={args.k}, model={model_type})", ""]

    for combo, combo_cols in combos.items():
        counts, feature_names = _select_counts_for_combo(
            hq_full_no_gap,
            combo_cols,
            reasoning_df,
            labels,
            splits,
            args.k,
            model_type,
        )
        full_df = (
            pd.concat([hq_full_no_gap, reasoning_df[combo_cols]], axis=1)
            if combo_cols
            else hq_full_no_gap.copy()
        )
        X_full, _, selected = mtp._preprocess_features(
            full_df,
            full_df.copy(),
            feature_names,
            model_type,
            "SFT",
            labels,
            mtp.PCA_VARIANCE_DEFAULT,
            mtp.PLS_COMPONENTS_DEFAULT,
            args.k,
        )
        _, _, model = mtp._train_model_local(
            model_type,
            X_full.values.astype(float),
            labels,
            X_full.values.astype(float),
            42,
        )
        coef_map: dict[str, float] = {}
        if model_type == "logistic":
            coef_vals = model.coef_.reshape(-1)
            coef_map = {name: float(val) for name, val in zip(selected, coef_vals)}
        total_folds = len(splits)
        rows = []
        for name in feature_names:
            rows.append(
                {
                    "feature": name,
                    "selected_count": counts.get(name, 0),
                    "selected_frac": counts.get(name, 0) / total_folds,
                    "weight": coef_map.get(name, 0.0),
                }
            )
        df = pd.DataFrame(rows).sort_values(
            ["selected_count", "feature"], ascending=[False, True]
        )
        csv_path = out_dir / f"sft_selection_{combo.replace('+','_')}_k{args.k}.csv"
        df.to_csv(csv_path, index=False)

        never_selected = df[df["selected_count"] == 0]["feature"].tolist()
        summary_lines += [f"## {combo}", "", f"Output: `{csv_path.name}`", ""]
        summary_lines += ["Top selected features (by fold count):"]
        top_block = df.head(15)
        for _, row in top_block.iterrows():
            summary_lines.append(f"- {row['feature']}: {int(row['selected_count'])}/{total_folds}")
        summary_lines += ["", "Never selected:"]
        if never_selected:
            summary_lines.append(", ".join(never_selected))
        else:
            summary_lines.append("(none)")
        summary_lines.append("")

    summary_path = out_dir / f"sft_selection_summary_k{args.k}.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
