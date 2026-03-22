import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from lib.paths import BASE_DIR


def _reasoning_run_valid(run_dir: Path, exp_id: str) -> bool:
    run_path = run_dir / "llm_reasoning_full.parquet"
    if not run_path.exists():
        return False
    df = pd.read_parquet(run_path)
    if not any(c.startswith(f"{exp_id}_") for c in df.columns):
        return False
    num = df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
    if num.empty:
        return True
    return not num.isna().any(axis=1).any()


def _select_latest_valid_runs(runs_root: Path, exp_list: list[str]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
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


def _detect_join_key(sample_df: pd.DataFrame) -> str:
    if "founder_uuid" in sample_df.columns and sample_df["founder_uuid"].notna().any():
        return "founder_uuid"
    return "row_index"


def _update_currently_in_use(exp_list: list[str], selected_runs: dict[str, Path]) -> None:
    base_root = BASE_DIR
    use_root = base_root / "features_storage" / "llm_reasoning" / "currently_in_use"
    exp_root = use_root / "experiments"
    use_root.mkdir(parents=True, exist_ok=True)
    exp_root.mkdir(parents=True, exist_ok=True)

    sample_df = pd.read_parquet(next(iter(selected_runs.values())) / "llm_reasoning_full.parquet")
    join_key = _detect_join_key(sample_df)

    # Refresh per-experiment folder
    if exp_root.exists():
        shutil.rmtree(exp_root, ignore_errors=True)
    exp_root.mkdir(parents=True, exist_ok=True)

    for exp_id, run_dir in selected_runs.items():
        src = run_dir / "llm_reasoning_full.parquet"
        dst_dir = exp_root / exp_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "llm_reasoning_full.parquet")

    merged = None
    run_map: dict[str, str] = {}
    for exp_id, run_dir in selected_runs.items():
        run_map[exp_id] = run_dir.name
        df = pd.read_parquet(run_dir / "llm_reasoning_full.parquet")
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
    if merged is None:
        raise RuntimeError("No experiments merged for currently_in_use.")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync latest LLM reasoning runs into currently_in_use.")
    parser.add_argument(
        "--experiments",
        default="A,B,C,D,E,F",
        help="Comma-separated list of experiments to sync (default: A,B,C,D,E,F).",
    )
    args = parser.parse_args()
    exp_list = [e.strip() for e in str(args.experiments).split(",") if e.strip()]
    runs_root = BASE_DIR / "features_storage" / "llm_reasoning" / "runs"
    selected_runs = _select_latest_valid_runs(runs_root, exp_list)
    missing = [e for e in exp_list if e not in selected_runs]
    if missing:
        raise RuntimeError(f"Missing valid runs for experiments: {missing}")
    _update_currently_in_use(exp_list, selected_runs)
    print(f"Synced currently_in_use for experiments: {', '.join(exp_list)}")


if __name__ == "__main__":
    main()
