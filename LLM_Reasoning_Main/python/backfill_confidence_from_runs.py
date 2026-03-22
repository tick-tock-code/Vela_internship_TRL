"""Backfill per-experiment confidence columns from latest run_A/run_B/run_E."""

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


def _load_confidence(run_dir: Path, exp_id: str, join_key: str) -> pd.DataFrame:
    df = pd.read_parquet(run_dir / "llm_reasoning_full.parquet")
    if join_key == "row_index":
        df = df.reset_index(drop=True)
        df["row_index"] = df.index
    if "evidence_support_rating" not in df.columns:
        raise RuntimeError(f"{run_dir.name} missing evidence_support_rating.")
    cols = [join_key, "evidence_support_rating"]
    if "founder_uuid" in df.columns and join_key != "founder_uuid":
        cols = ["founder_uuid"] + cols
    out = df[cols].copy()
    out.rename(columns={"evidence_support_rating": f"{exp_id}_evidence_support_rating"}, inplace=True)
    out = out.drop_duplicates(join_key)
    return out


def main() -> int:
    root = BASE_DIR
    runs_root = root / "features_storage" / "llm_reasoning" / "runs"
    use_candidates = [
        root / "features_storage" / "llm_reasoning" / "currently_in_use",
        root / "features_storage" / "llm_reasoning" / "current",
    ]
    use_targets: list[Path] = []
    for candidate in use_candidates:
        if (candidate / "llm_reasoning_full.parquet").exists():
            use_targets.append(candidate)

    if not runs_root.exists():
        print("No runs folder found.", file=sys.stderr)
        return 1
    if not use_targets:
        print("No current reasoning parquet found to backfill.", file=sys.stderr)
        return 1

    exp_ids = ["A", "B", "E"]
    join_key = "founder_uuid"
    # Detect whether founder_uuid is usable
    sample_run = _latest_run(runs_root, exp_ids[0])
    if sample_run is None:
        print("No run_A found.", file=sys.stderr)
        return 1
    sample_df = pd.read_parquet(sample_run / "llm_reasoning_full.parquet")
    if "founder_uuid" not in sample_df.columns or not sample_df["founder_uuid"].notna().any():
        join_key = "row_index"
    run_map: dict[str, str] = {}
    conf_frames: list[pd.DataFrame] = []
    for exp_id in exp_ids:
        run_dir = _latest_run(runs_root, exp_id)
        if run_dir is None:
            print(f"No run_{exp_id}_* parquet found.", file=sys.stderr)
            return 1
        run_map[exp_id] = run_dir.name
        conf_frames.append(_load_confidence(run_dir, exp_id, join_key))

    merged_conf = conf_frames[0]
    for df in conf_frames[1:]:
        merged_conf = merged_conf.merge(df, on=join_key, how="inner")

    for use_root in use_targets:
        use_path = use_root / "llm_reasoning_full.parquet"
        use_df = pd.read_parquet(use_path)
        if join_key == "row_index":
            use_df = use_df.reset_index(drop=True)
            use_df["row_index"] = use_df.index
        use_df = use_df.drop_duplicates(join_key)
        for exp_id in exp_ids:
            for old_col in (f"{exp_id}_confidence", f"{exp_id}_evidence_support_rating"):
                if old_col in use_df.columns:
                    use_df = use_df.drop(columns=[old_col])
        updated = use_df.merge(merged_conf, on=join_key, how="left")
        updated.to_parquet(use_path, index=False)

        exp_root = use_root / "experiments"
        for exp_id in exp_ids:
            exp_path = exp_root / exp_id / "llm_reasoning_full.parquet"
            if not exp_path.exists():
                continue
            exp_df = pd.read_parquet(exp_path)
            if join_key == "row_index":
                exp_df = exp_df.reset_index(drop=True)
                exp_df["row_index"] = exp_df.index
            conf_col = f"{exp_id}_evidence_support_rating"
            if conf_col in exp_df.columns:
                exp_df = exp_df.drop(columns=[conf_col])
            exp_df = exp_df.merge(
                merged_conf[[join_key, conf_col]], on=join_key, how="left"
            )
            exp_df.to_parquet(exp_path, index=False)

        manifest_path = use_root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
        else:
            manifest = {}
        manifest["confidence_backfill"] = {"runs": run_map, "column": "evidence_support_rating"}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Backfill complete for:")
    for use_root in use_targets:
        print(f"- {use_root}")
    for exp_id, run_name in run_map.items():
        print(f"{exp_id}: {run_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
