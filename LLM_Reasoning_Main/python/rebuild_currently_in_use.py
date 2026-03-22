"""Rebuild current/currently_in_use from latest run_A/run_B/run_E parquets."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd
from paths import BASE_DIR


def _latest_run(runs_root: Path, exp_id: str) -> Path | None:
    candidates = sorted(
        [p for p in runs_root.glob(f"run_{exp_id}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        run_path = candidate / "llm_reasoning_full.parquet"
        if run_path.exists():
            return candidate
    return None


def _load_run(run_dir: Path, exp_id: str, join_key: str) -> pd.DataFrame:
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
    return df


def _write_outputs(target_root: Path, merged: pd.DataFrame, run_map: dict[str, str], per_exp: dict[str, pd.DataFrame]) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(target_root / "llm_reasoning_full.parquet", index=False)
    exp_root = target_root / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)
    for exp_id, df in per_exp.items():
        exp_dir = exp_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(exp_dir / "llm_reasoning_full.parquet", index=False)
    manifest = {
        "source": "runs",
        "runs": run_map,
    }
    (target_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    root = BASE_DIR
    runs_root = root / "features_storage" / "llm_reasoning" / "runs"
    if not runs_root.exists():
        print("Runs folder not found.", file=sys.stderr)
        return 1

    exp_ids = ["A", "B", "E"]
    join_key = "founder_uuid"
    sample_run = _latest_run(runs_root, exp_ids[0])
    if sample_run is None:
        print("No run_A found.", file=sys.stderr)
        return 1
    sample_df = pd.read_parquet(sample_run / "llm_reasoning_full.parquet")
    if "founder_uuid" not in sample_df.columns or not sample_df["founder_uuid"].notna().any():
        join_key = "row_index"
    run_map: dict[str, str] = {}
    merged = None
    per_exp: dict[str, pd.DataFrame] = {}

    for exp_id in exp_ids:
        run_dir = _latest_run(runs_root, exp_id)
        if run_dir is None:
            print(f"No run_{exp_id}_* parquet found.", file=sys.stderr)
            return 1
        run_map[exp_id] = run_dir.name
        df = _load_run(run_dir, exp_id, join_key)
        per_exp[exp_id] = df.copy()
        df = df.set_index(join_key)
        if merged is None:
            merged = df
        else:
            merged = merged.join(df.drop(columns=["success"], errors="ignore"), how="inner")

    if merged is None:
        print("No data merged.", file=sys.stderr)
        return 1
    if "success" not in merged.columns:
        raise RuntimeError("Merged frame missing success column.")
    merged = merged.reset_index()

    current_root = root / "features_storage" / "llm_reasoning" / "current"
    use_root = root / "features_storage" / "llm_reasoning" / "currently_in_use"
    _write_outputs(current_root, merged, run_map, per_exp)
    _write_outputs(use_root, merged, run_map, per_exp)

    print("Rebuilt current and currently_in_use.")
    for exp_id, run_name in run_map.items():
        print(f"{exp_id}: {run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
