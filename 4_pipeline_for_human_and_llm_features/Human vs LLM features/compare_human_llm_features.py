"""Compare human features vs LLM features (n_rules=10, repeat 1).

Outputs:
  - human_llm_corr.csv
  - human_llm_corr.png
  - human_llm_name_mapping.csv
  - llm_top_k_correlations.csv
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\4_pipeline_for_human_and_llm_features")
OUT_DIR = BASE_DIR / "Human vs LLM features"
HUMAN_PARQUET = BASE_DIR / "features_storage" / "features_full.parquet"
LLM_PARQUET = BASE_DIR / "features_storage" / "llm_features_10_r1.parquet"


def _tokenize(name: str) -> set[str]:
    parts = re.split(r"[^a-zA-Z0-9]+", name.lower())
    return {p for p in parts if p}


def main() -> None:
    if not HUMAN_PARQUET.exists():
        raise FileNotFoundError(f"Missing {HUMAN_PARQUET}")
    if not LLM_PARQUET.exists():
        raise FileNotFoundError(f"Missing {LLM_PARQUET}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    human_df = pd.read_parquet(HUMAN_PARQUET)
    llm_df = pd.read_parquet(LLM_PARQUET)

    # Align on founder_uuid (preferred). If missing/empty, align on row index with equal length.
    if "founder_uuid" not in human_df.columns or "founder_uuid" not in llm_df.columns:
        raise KeyError("founder_uuid column missing in one of the inputs.")

    if human_df["founder_uuid"].isna().all() or llm_df["founder_uuid"].isna().all():
        if len(human_df) != len(llm_df):
            raise RuntimeError(
                "founder_uuid is empty in at least one file and row counts differ. "
                "Regenerate features_full.parquet for the same dataset as the LLM features."
            )
        human_df = human_df.reset_index(drop=True)
        llm_df = llm_df.reset_index(drop=True)
    else:
        human_df = human_df.dropna(subset=["founder_uuid"]).set_index("founder_uuid")
        llm_df = llm_df.dropna(subset=["founder_uuid"]).set_index("founder_uuid")
        common_idx = human_df.index.intersection(llm_df.index)
        if len(common_idx) == 0:
            raise RuntimeError("No overlapping founder_uuid between human and LLM datasets.")
        human_df = human_df.loc[common_idx]
        llm_df = llm_df.loc[common_idx]

    # Drop non-feature columns
    human_features = human_df.drop(columns=[c for c in ["success"] if c in human_df.columns])
    llm_features = llm_df.drop(columns=[c for c in ["success"] if c in llm_df.columns])

    if human_features.empty or llm_features.empty:
        raise RuntimeError("No features found after filtering.")

    # Correlation matrix: human rows x LLM columns
    corr = pd.DataFrame(index=human_features.columns, columns=llm_features.columns, dtype=float)
    for h in human_features.columns:
        corr[h] = llm_features.corrwith(human_features[h])
    corr = corr.T  # rows = human, cols = llm

    corr.to_csv(OUT_DIR / "human_llm_corr.csv", index=True)

    # Heatmap
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * corr.shape[1]), max(6, 0.4 * corr.shape[0])))
    im = ax.imshow(corr.values, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Human vs LLM Feature Correlation (Pearson)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "human_llm_corr.png", dpi=150)
    plt.close(fig)

    # Top-k matches by correlation for each LLM feature
    top_rows = []
    for llm_col in corr.columns:
        top = corr[llm_col].sort_values(ascending=False).head(5)
        for rank, (h_name, val) in enumerate(top.items(), 1):
            top_rows.append(
                {
                    "llm_feature": llm_col,
                    "rank": rank,
                    "human_feature": h_name,
                    "correlation": float(val),
                }
            )
    pd.DataFrame(top_rows).to_csv(OUT_DIR / "llm_top_k_correlations.csv", index=False)

    # Name-based mapping (token overlap)
    mapping_rows = []
    human_tokens = {h: _tokenize(h) for h in human_features.columns}
    for llm_name in llm_features.columns:
        llm_toks = _tokenize(llm_name)
        best = None
        best_score = -1.0
        for h_name, h_toks in human_tokens.items():
            if not llm_toks and not h_toks:
                score = 0.0
            else:
                score = len(llm_toks & h_toks) / max(1, len(llm_toks | h_toks))
            if score > best_score:
                best_score = score
                best = h_name
        mapping_rows.append(
            {
                "llm_feature": llm_name,
                "best_human_feature": best,
                "name_similarity": float(best_score),
            }
        )
    pd.DataFrame(mapping_rows).to_csv(OUT_DIR / "human_llm_name_mapping.csv", index=False)

    print(f"Wrote correlation CSV to: {OUT_DIR / 'human_llm_corr.csv'}")
    print(f"Wrote heatmap PNG to: {OUT_DIR / 'human_llm_corr.png'}")
    print(f"Wrote name mapping CSV to: {OUT_DIR / 'human_llm_name_mapping.csv'}")
    print(f"Wrote top-k correlation CSV to: {OUT_DIR / 'llm_top_k_correlations.csv'}")


if __name__ == "__main__":
    main()
