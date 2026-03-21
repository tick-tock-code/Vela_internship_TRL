"""Repair NaN batches for a specified run using batch-level re-queries."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from think_reason_learn.datasets import load_vcbench

from llm_reasoning_features import ReasoningConfig, generate_reasoning_features, LABEL_FIELDS
from paths import BASE_DIR, CONFIG_DIR, PROMPT_DIR


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


def _strip_label_fields(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in LABEL_FIELDS}


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair NaN batches for an experiment run.")
    parser.add_argument("--experiment", default="F", help="Experiment id to repair (default: F).")
    parser.add_argument("--concurrency", type=int, default=None, help="Override concurrency for repair.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size for repair.")
    args = parser.parse_args()

    exp_id = str(args.experiment).strip().upper()
    if not exp_id:
        raise ValueError("Experiment id cannot be empty.")

    root = BASE_DIR
    for candidate in (root / ".env", root.parent / ".env", root.parent.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            break

    features_path = CONFIG_DIR / "features.json"
    if not features_path.exists():
        raise FileNotFoundError("features.json not found.")
    config_data = json.loads(features_path.read_text(encoding="utf-8-sig"))

    runs_root = root / "features_storage" / "llm_reasoning" / "runs"
    run_dir = _latest_run(runs_root, exp_id)
    if run_dir is None:
        raise RuntimeError(f"No run_{exp_id}_* parquet found.")

    run_parquet = run_dir / "llm_reasoning_full.parquet"
    df_run = pd.read_parquet(run_parquet)
    if "founder_uuid" not in df_run.columns or not df_run["founder_uuid"].notna().any():
        df_run = df_run.reset_index(drop=True)
        df_run["row_index"] = df_run.index

    numeric = df_run.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
    nan_rows = numeric.isna().any(axis=1)
    nan_count = int(nan_rows.sum())
    print(f"run_{exp_id} NaN rows: {nan_count}")
    if nan_count == 0:
        print("No NaNs detected; nothing to repair.")
        return 0

    batch_size = int(config_data.get("llm_reasoning_batch_size", 20))
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    target_batches = sorted({int(i // batch_size) for i in np.where(nan_rows)[0].tolist()})
    print(f"Target batches to repair: {len(target_batches)}")
    print(f"NaN indices: {np.where(nan_rows)[0].tolist()}")

    dataset_path = root.parent / "VCBench-Starter-Kit" / "vcbench_final_public.csv"
    records, labels = load_vcbench(str(dataset_path))
    labels = np.asarray(labels)

    seed_path = root / "features_storage" / "llm_engineered" / "seed_100.json"
    seed_data = json.loads(seed_path.read_text(encoding="utf-8"))
    seed_idx = set(int(i) for i in seed_data.get("indices", []))
    pool_idx = [i for i in range(len(records)) if i not in seed_idx]
    pool_records = [_strip_label_fields(records[i]) for i in pool_idx]
    pool_labels = labels[pool_idx]

    providers = config_data.get("llm_providers", {"openai": True, "google": False})
    concurrency = int(config_data.get("llm_reasoning_concurrency", 10))
    if args.concurrency is not None:
        concurrency = int(args.concurrency)

    cfg = ReasoningConfig(
        model=str(config_data.get("llm_model", "gpt-4.1-mini")),
        dataset_size=str(config_data.get("llm_reasoning_dataset_size", "full")),
        random_state=int(config_data.get("random_state", 42)),
        core_prompt_path=(
            PROMPT_DIR
            / str(config_data.get("llm_reasoning_core_prompt_path", "core_prompt.txt"))
        ),
        experiments_path=(
            CONFIG_DIR
            / str(config_data.get("llm_reasoning_experiments_path", "experiments.json"))
        ),
        providers=providers,
        google_model=str(config_data.get("llm_google_model", "gemini-2.0-flash")),
        batch_size=batch_size,
        concurrency=concurrency,
        experiments=[exp_id],
        dry_run=bool(config_data.get("llm_reasoning_dry_run", False)),
        dry_run_fast=bool(config_data.get("llm_reasoning_dry_run_fast", False)),
        log_dir=root / "logging" / f"repair_run_{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        log_every=int(config_data.get("llm_reasoning_log_every", 10)),
        repair_nan=True,
        repair_existing=True,
        skip_select=True,
        inline_repair=bool(config_data.get("llm_reasoning_inline_repair", True)),
        inline_repair_max_attempts=int(config_data.get("llm_reasoning_inline_repair_max_attempts", 1)),
        target_batch_indices=target_batches,
    )

    generate_reasoning_features(
        records=pool_records,
        labels=pool_labels,
        config=cfg,
        output_dir=run_dir,
        metadata_path=run_dir / f"llm_reasoning_full_{exp_id}_repair_targeted.json",
        existing_df=df_run,
    )
    print(f"Repaired run_{exp_id} in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
