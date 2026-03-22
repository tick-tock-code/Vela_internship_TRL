"""Compute correlation matrix for all human-made features and save outputs."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pandas as pd
import numpy as np

from think_reason_learn.datasets import load_vcbench
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_registry import FEATURE_REGISTRY


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


def _custom_feature_df(records: list[dict], feature_names: list[str]) -> pd.DataFrame:
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


def main() -> None:
    base_script = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "2_Human_features_running_example_script" / "vcbench_lambda_features_minimal.py"
    extract_base = _load_base_feature_extractor(base_script)

    input_csv = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit" / "vcbench_final_public.csv"

    records, labels = load_vcbench(str(input_csv), "success", 0, 42)
    idx = np.arange(len(records))
    train_idx, _ = train_test_split(
        idx,
        test_size=0.20,
        stratify=labels,
        random_state=42,
    )
    train_recs = [records[i] for i in train_idx]

    base_df = pd.DataFrame([extract_base(r) for r in train_recs])
    custom_features = list(FEATURE_REGISTRY.keys())
    custom_df = _custom_feature_df(train_recs, custom_features)

    full_df = pd.concat([base_df, custom_df], axis=1)
    corr = full_df.corr()

    out_dir = Path(__file__).parent
    csv_path = out_dir / "human_feature_correlations_train.csv"
    png_path = out_dir / "human_feature_correlations_train.png"

    corr.to_csv(csv_path, index=True)
    print(f"Saved correlation CSV to: {csv_path}")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(12, 10))
        plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Pearson r")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
        plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print(f"Saved correlation heatmap to: {png_path}")
    except Exception as exc:
        print(f"Heatmap not saved (matplotlib missing?): {exc}")


if __name__ == "__main__":
    main()
