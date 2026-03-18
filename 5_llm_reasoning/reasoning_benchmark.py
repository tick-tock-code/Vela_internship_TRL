"""Benchmark LLM reasoning speed/quality over batch size × concurrency settings."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from think_reason_learn.datasets import load_vcbench

from llm_reasoning_features import ReasoningConfig, generate_reasoning_features


@dataclass
class RunResult:
    batch_size: int
    concurrency: int
    repeat: int
    runtime_s: float
    runtime_per_founder_s: float
    invalid_batch_count: int
    retry_count: int
    error_count: int
    error_rate: float
    nan_pct: float
    mean_feature_std: float
    mean_feature_mad: float
    output_dir: str


def _resolve_input_csv(dataset: str, override: str) -> str:
    if override:
        return override
    base = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit"
    if dataset == "sample":
        return str(base / "vcbench_final_public_sample100.csv")
    return str(base / "vcbench_final_public.csv")


def _load_env_if_present() -> None:
    env_path = Path(__file__).resolve().parents[1].parent / ".env"
    if not env_path.exists():
        return
    try:
        from think_reason_learn.core import _config as trl_config
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            val = val.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = val
            if key in ("OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY"):
                if hasattr(trl_config, "settings") and not getattr(trl_config.settings, key, ""):
                    setattr(trl_config.settings, key, val)
        # Reset LLM singleton so it re-reads updated settings
        from think_reason_learn.core.llms._ask import LLM
        from think_reason_learn.core._singleton import SingletonMeta
        SingletonMeta._instances.pop(LLM, None)
        import think_reason_learn.core.llms as trl_llms
        trl_llms.llm = trl_llms.LLM()
    except Exception:
        return


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _stratified_sample(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    n: int,
    random_state: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rng = np.random.RandomState(random_state)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    target_pos = n // 2
    n_pos = min(len(pos_idx), target_pos)
    n_neg = min(len(neg_idx), n - n_pos)
    selected = list(pos_idx[:n_pos]) + list(neg_idx[:n_neg])
    remaining = [i for i in range(len(labels)) if i not in selected]
    rng.shuffle(remaining)
    if len(selected) < n:
        selected += remaining[: n - len(selected)]
    rng.shuffle(selected)
    return [records[i] for i in selected], labels[selected]


def _parse_errors(errors_path: Path) -> tuple[int, int]:
    if not errors_path.exists():
        return 0, 0
    invalid = 0
    total = 0
    for line in errors_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        total += 1
        if "initial_parse" in line:
            invalid += 1
    return invalid, total


def _count_retries(raw_path: Path) -> int:
    if not raw_path.exists():
        return 0
    text = raw_path.read_text(encoding="utf-8", errors="ignore")
    return text.count("--- RETRY ---") + text.count("--- RETRY 2 ---")


def _nan_pct(df: pd.DataFrame) -> float:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "success"]
    if not numeric_cols:
        return 0.0
    return float(df[numeric_cols].isna().any(axis=1).mean())


def _consistency_metrics(repeat_dfs: list[pd.DataFrame]) -> tuple[float, float]:
    numeric_cols = repeat_dfs[0].select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "success"]
    if not numeric_cols:
        return 0.0, 0.0

    feature_std_means = []
    feature_mad_means = []
    for col in numeric_cols:
        stacked = []
        for df in repeat_dfs:
            series = df.set_index("founder_uuid")[col]
            stacked.append(series)
        wide = pd.concat(stacked, axis=1)
        values = wide.to_numpy(dtype=float)
        mean = np.nanmean(values, axis=1, keepdims=True)
        std = np.nanstd(values, axis=1)
        mad = np.nanmean(np.abs(values - mean), axis=1)
        feature_std_means.append(float(np.nanmean(std)))
        feature_mad_means.append(float(np.nanmean(mad)))
    return float(np.mean(feature_std_means)), float(np.mean(feature_mad_means))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LLM reasoning batch size × concurrency.")
    p.add_argument("--dataset", choices=["sample", "full"], default="full")
    p.add_argument("--input_csv", default="", help="Optional override CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--sample_size", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--batch_sizes", default="5,10,20,30")
    p.add_argument("--concurrencies", default="5,10,15,20")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Reduce grid size to batch_sizes=10,20 and concurrencies=5,10.",
    )
    p.add_argument("--llm_model", default="gpt-4.1-nano")
    p.add_argument(
        "--feature_config",
        default=str(Path(__file__).parent / "features.json"),
        help="Path to features.json (providers + prompt paths).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Run without API calls (mock outputs).",
    )
    return p.parse_args()


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    input_csv = _resolve_input_csv(args.dataset, args.input_csv)

    cfg = _load_config(Path(args.feature_config))
    core_prompt_path = cfg.get("llm_reasoning_core_prompt_path", "core_prompt.txt")
    experiments_path = cfg.get("llm_reasoning_experiments_path", "experiments.json")
    llm_providers = cfg.get("llm_providers", {"openai": True, "google": False})
    llm_google_model = cfg.get("llm_google_model", None)

    core_prompt_path = Path(core_prompt_path)
    experiments_path = Path(experiments_path)
    if not core_prompt_path.is_absolute():
        core_prompt_path = Path(__file__).parent / core_prompt_path
    if not experiments_path.is_absolute():
        experiments_path = Path(__file__).parent / experiments_path

    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        args.random_state,
    )
    sample_records, sample_labels = _stratified_sample(
        records, labels, args.sample_size, args.random_state
    )

    if args.quick:
        batch_sizes = [10, 20]
        concurrencies = [5, 10]
    else:
        batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
        concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]

    root = Path(__file__).parent / "benchmark_raw"
    root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[RunResult] = []
    per_config_feature = {}

    for batch_size in batch_sizes:
        for concurrency in concurrencies:
            repeat_dfs: list[pd.DataFrame] = []
            for repeat in range(1, args.repeats + 1):
                rng = np.random.RandomState(args.random_state + repeat)
                perm = rng.permutation(len(sample_records))
                rep_records = [sample_records[i] for i in perm]
                rep_labels = sample_labels[perm]

                run_id = f"bs{batch_size}_c{concurrency}_r{repeat}"
                run_dir = root / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                log_dir = run_dir / "logs"
                output_dir = run_dir / "outputs"
                output_dir.mkdir(parents=True, exist_ok=True)
                meta_path = run_dir / f"meta_{run_id}.json"

                config = ReasoningConfig(
                    model=args.llm_model,
                    dataset_size="full",
                    random_state=args.random_state + repeat,
                    core_prompt_path=core_prompt_path,
                    experiments_path=experiments_path,
                    providers=llm_providers,
                    google_model=llm_google_model,
                    batch_size=batch_size,
                    concurrency=concurrency,
                    experiments=["A"],
                    dry_run=args.dry_run,
                    dry_run_fast=args.dry_run,
                    log_dir=log_dir,
                    log_every=1,
                )

                start = time.time()
                df, _ = generate_reasoning_features(
                    records=rep_records,
                    labels=rep_labels,
                    config=config,
                    output_dir=output_dir,
                    metadata_path=meta_path,
                )
                runtime_s = time.time() - start

                errors_path = log_dir / "errors.jsonl"
                invalid_count, error_count = _parse_errors(errors_path)
                raw_path = meta_path.with_name(meta_path.stem + "_raw_responses.txt")
                retry_count = _count_retries(raw_path)
                nan_pct = _nan_pct(df)
                n_batches = int(np.ceil(len(rep_records) / batch_size))
                error_rate = invalid_count / n_batches if n_batches else 0.0

                repeat_dfs.append(df)
                summary_rows.append(
                    RunResult(
                        batch_size=batch_size,
                        concurrency=concurrency,
                        repeat=repeat,
                        runtime_s=runtime_s,
                        runtime_per_founder_s=runtime_s / len(rep_records),
                        invalid_batch_count=invalid_count,
                        retry_count=retry_count,
                        error_count=error_count,
                        error_rate=error_rate,
                        nan_pct=nan_pct,
                        mean_feature_std=0.0,
                        mean_feature_mad=0.0,
                        output_dir=str(run_dir),
                    )
                )

            mean_std, mean_mad = _consistency_metrics(repeat_dfs)
            per_config_feature[(batch_size, concurrency)] = (mean_std, mean_mad)

    # Merge consistency metrics into summary
    updated_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        mean_std, mean_mad = per_config_feature[(row.batch_size, row.concurrency)]
        row_dict = row.__dict__.copy()
        row_dict["mean_feature_std"] = mean_std
        row_dict["mean_feature_mad"] = mean_mad
        updated_rows.append(row_dict)

    summary_df = pd.DataFrame(updated_rows)
    summary_path = root / "benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    agg = (
        summary_df.groupby(["batch_size", "concurrency"])
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_path = root / "benchmark_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    print(f"Summary saved to: {summary_path}")
    print(f"Aggregate saved to: {agg_path}")


if __name__ == "__main__":
    main()
