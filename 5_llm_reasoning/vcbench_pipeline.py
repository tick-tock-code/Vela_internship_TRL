"""VCBench in-depth pipeline (human + optional LLM features).

Pipeline steps:
1. Load VCBench data (sample or full).
2. Extract baseline features (15) from example script.
3. Extract custom features from registry.
4. Optionally add LLM-derived features.
5. Save full feature dataset (founder_uuid, label, features) to Parquet.
6. Train logistic regression and evaluate metrics.
"""

from __future__ import annotations

import argparse
import os
import json
import importlib.util
import math
import asyncio
import sys
import shutil
from pathlib import Path
from datetime import datetime
import traceback
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from think_reason_learn.datasets import load_vcbench

from feature_registry import FEATURE_REGISTRY, FEATURE_SETS
from llm_feature_generation import generate_llm_features
from llm_reasoning_features import (
    ReasoningConfig,
    build_experiment_key_map,
    generate_reasoning_features,
    write_per_experiment_parquets,
)

RUN_LOG_PATH: Path | None = None


def _install_excepthook() -> None:
    def _hook(exc_type, exc, tb):
        if RUN_LOG_PATH is not None:
            try:
                with RUN_LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write("\n[ERROR] Unhandled exception:\n")
                    f.write("".join(traceback.format_exception(exc_type, exc, tb)))
            except Exception:
                pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


_install_excepthook()


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


def _load_llm_engineered_cache(
    cache_dir: Path,
    expected_rows: int,
    expected_n: int,
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    cache_path = cache_dir / "llm_features.parquet"
    meta_path = cache_dir / "llm_features_meta.json"
    if not cache_path.exists() or not meta_path.exists():
        return None, None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("n_rows") == expected_rows
            and meta.get("n_features") == expected_n
            and meta.get("model") == model
            and meta.get("providers") == providers
            and meta.get("google_model") == google_model
            and isinstance(meta.get("feature_names"), list)
        ):
            df = pd.read_parquet(cache_path)
            return df, list(meta.get("feature_names"))
    except Exception:
        return None, None
    return None, None


def _save_llm_engineered_cache(
    cache_dir: Path,
    df: pd.DataFrame,
    feature_names: list[str],
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
    n_features: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "llm_features.parquet"
    meta_path = cache_dir / "llm_features_meta.json"
    df.to_parquet(cache_path, index=False)
    meta = {
        "n_rows": len(df),
        "n_features": n_features,
        "feature_names": feature_names,
        "model": model,
        "providers": providers,
        "google_model": google_model,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _custom_feature_df(
    records: Iterable[dict[str, Any]], feature_names: list[str]
) -> pd.DataFrame:
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


def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, pct: float) -> float:
    n = len(y_true)
    k = max(1, int(math.ceil(pct * n)))
    order = np.argsort(y_scores)[::-1]
    top_k = order[:k]
    return float(np.sum(y_true[top_k]) / k)


def _train_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, LogisticRegression]:
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    train_scores = clf.predict_proba(X_train)[:, 1]
    test_scores = clf.predict_proba(X_test)[:, 1]
    return train_scores, test_scores, clf


def _report_metrics(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    # Threshold tuning on training set for F0.5 (match example script)
    best_t = 0.5
    best_f = 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(
            y_train,
            (train_scores >= t).astype(int),
            beta=0.5,
            zero_division=0.0,  # type: ignore[arg-type]
        )
        if f > best_f:
            best_f, best_t = f, float(t)

    y_pred = (test_scores >= best_t).astype(int)
    tp = float(np.sum((y_test == 1) & (y_pred == 1)))
    fn = float(np.sum((y_test == 1) & (y_pred == 0)))
    tn = float(np.sum((y_test == 0) & (y_pred == 0)))
    fp = float(np.sum((y_test == 0) & (y_pred == 1)))
    fnt = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "roc_auc": roc_auc_score(y_test, test_scores),
        "pr_auc": average_precision_score(y_test, test_scores),
        "precision": precision_score(y_test, y_pred, zero_division=0.0),  # type: ignore[arg-type]
        "recall": recall_score(y_test, y_pred, zero_division=0.0),  # type: ignore[arg-type]
        "f0.5": fbeta_score(y_test, y_pred, beta=0.5, zero_division=0.0),  # type: ignore[arg-type]
        "precision@1%": _precision_at_k(y_test, test_scores, 0.01),
        "precision@5%": _precision_at_k(y_test, test_scores, 0.05),
        "precision@10%": _precision_at_k(y_test, test_scores, 0.10),
        "threshold": best_t,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "fnr": fnt,
    }
    return metrics


def _format_topk(metrics: dict[str, float]) -> list[str]:
    return [
        "\n  Top-k precision:",
        f"    precision@1%={metrics['precision@1%']:.3f}",
        f"    precision@5%={metrics['precision@5%']:.3f}",
        f"    precision@10%={metrics['precision@10%']:.3f}",
    ]


def _train_and_log(
    feature_names: list[str],
    full_train: pd.DataFrame,
    full_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    input_csv: str,
    mode_label: str,
    log_dir: Path | None = None,
) -> tuple[dict[str, float], LogisticRegression]:
    # Impute missing values using training means (prevents leakage).
    if full_train.isna().any().any() or full_test.isna().any().any():
        fill_values = full_train.mean()
        full_train = full_train.fillna(fill_values)
        full_test = full_test.fillna(fill_values)

    X_train = full_train.values.astype(float)
    X_test = full_test.values.astype(float)

    train_scores, test_scores, model = _train_sklearn(
        X_train, y_train, X_test, args.random_state
    )
    metrics = _report_metrics(y_train, train_scores, y_test, test_scores)
    acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
    metrics["accuracy"] = acc

    log_lines: list[str] = []
    log_lines.append(f"Features used: {', '.join(feature_names)}")
    log_lines.append(
        f"\n[{mode_label}]   {len(feature_names)} features, threshold={metrics['threshold']:.2f}"
    )
    coef = model.coef_[0]
    ranked = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)
    for name, c in ranked:
        sign = "+" if c >= 0 else "-"
        log_lines.append(f"  {sign}{abs(c):.3f}  {name}")
    log_lines.append(
        f"\nROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
        f"F0.5={metrics['f0.5']:.3f}  Acc={acc:.3f}  "
        f"FNR={metrics['fnr']:.3f}  TP={int(metrics['tp'])}  FN={int(metrics['fn'])}"
    )
    log_lines.extend(_format_topk(metrics))

    _write_log(
        log_lines,
        args,
        input_csv,
        feature_names,
        len(y_train),
        len(y_test),
        int(y_train.sum()),
        int(y_test.sum()),
        metrics,
        log_dir=log_dir,
    )
    return metrics, model


def _parse_training_log(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    metrics_line = None
    for line in lines:
        if line.startswith("ROC-AUC="):
            metrics_line = line
            break
    if not metrics_line:
        return None
    metrics: dict[str, float] = {}
    for part in metrics_line.split():
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        try:
            metrics[key] = float(val)
        except ValueError:
            continue
    return metrics


def _write_f05_table(rows: list[dict[str, float]], report_path: Path) -> str:
    header = "| Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            "| {name} | {f0:.3f} | {roc:.3f} | {pr:.3f} | {prec:.3f} | {rec:.3f} | {acc:.3f} |".format(
                name=row["name"],
                f0=row["F0.5"],
                roc=row["ROC-AUC"],
                pr=row["PR-AUC"],
                prec=row["Prec"],
                rec=row["Rec"],
                acc=row["Acc"],
            )
        )
    table = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# LLM Regression Summary (F0.5)\n\n" + table + "\n",
        encoding="utf-8",
    )
    return table


def _write_family_leaderboard(rows: list[dict[str, Any]], report_path: Path, csv_path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    header = "| Set ID | Regression | F0.5 | ROC-AUC | PR-AUC | Prec | Rec | Acc |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append(
            "| {set_id} | {name} | {f0:.3f} | {roc:.3f} | {pr:.3f} | {prec:.3f} | {rec:.3f} | {acc:.3f} |".format(
                set_id=row.get("set_id", ""),
                name=row.get("regression", ""),
                f0=float(row.get("F0.5", float("nan"))),
                roc=float(row.get("ROC-AUC", float("nan"))),
                pr=float(row.get("PR-AUC", float("nan"))),
                prec=float(row.get("Prec", float("nan"))),
                rec=float(row.get("Rec", float("nan"))),
                  acc=float(row.get("Acc", float("nan"))),
              )
          )
    leaderboard_table = "\n".join(lines)

    metrics = ["F0.5", "ROC-AUC", "PR-AUC", "Prec", "Rec", "Acc"]
    improvements: dict[str, list[float]] = {m: [] for m in metrics}
    for set_id in df["set_id"].unique():
        subset = df[df["set_id"] == set_id]
        only = subset[subset["regression"] == "LLM Engineered Only"]
        plus = subset[subset["regression"] == "LLM Engineered + Reasoning"]
        if only.empty or plus.empty:
            continue
        for m in metrics:
            try:
                delta = float(plus.iloc[0][m]) - float(only.iloc[0][m])
                improvements[m].append(delta)
            except Exception:
                continue

    avg_lines = ["### Average Improvement (Engineered + Reasoning vs Engineered Only)"]
    for m in metrics:
        vals = improvements.get(m, [])
        avg = float(np.nanmean(vals)) if vals else float("nan")
        avg_lines.append(f"- {m}: {avg:+.3f}")

    section = (
        "## LLM Engineered Run-Family Leaderboard\n\n"
        + leaderboard_table
        + "\n\n"
        + "\n".join(avg_lines)
        + "\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        base = report_path.read_text(encoding="utf-8").rstrip()
        report_path.write_text(base + "\n\n" + section, encoding="utf-8")
    else:
        report_path.write_text(
            "# LLM Regression Summary (F0.5)\n\n" + section, encoding="utf-8"
        )

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VCBench in-depth pipeline.")
    p.add_argument("--dataset", choices=["sample", "full"], default="sample")
    p.add_argument("--input_csv", default="", help="Optional override CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--feature_set",
        choices=sorted(FEATURE_SETS.keys()),
        default="base_plus_custom",
    )
    p.add_argument(
        "--features",
        default="",
        help="Optional comma-separated custom feature names (overrides feature_set)",
    )
    p.add_argument(
        "--feature_config",
        default=str(Path(__file__).parent / "features.json"),
        help="Optional JSON file with {\"features\": [...]} to select custom features.",
    )
    p.add_argument(
        "--extract_only",
        action="store_true",
        help="Extract features and save Parquet, then exit before training.",
    )
    p.add_argument(
        "--mode",
        choices=["human", "llm", "reasoning", "hybrid"],
        default="human",
        help="human=base+custom, llm=LLM only, reasoning=LLM reasoning only, hybrid=base+custom+LLM+reasoning",
    )
    p.add_argument(
        "--llm_features",
        action="store_true",
        help="(Deprecated) Use --mode llm or --mode hybrid instead.",
    )
    p.add_argument("--llm_model", default="gpt-4.1-nano")
    p.add_argument("--llm_n_features", type=int, default=8)
    p.add_argument(
        "--llm_reasoning",
        action="store_true",
        help="Enable LLM reasoning features (can be combined with other modes).",
    )
    p.add_argument(
        "--llm_reasoning_core_prompt",
        default=str(Path(__file__).parent / "core_prompt.txt"),
        help="Path to core prompt template for LLM reasoning features.",
    )
    p.add_argument(
        "--llm_reasoning_experiments",
        default=str(Path(__file__).parent / "experiments.json"),
        help="Path to experiments JSON for LLM reasoning features.",
    )
    p.add_argument(
        "--llm_reasoning_dataset_size",
        default="full",
        help="Dataset size for reasoning features: full, 200, 400, 1000.",
    )
    p.add_argument(
        "--llm_reasoning_batch_size",
        type=int,
        default=20,
        help="Batch size for LLM reasoning (default 20).",
    )
    p.add_argument(
        "--llm_reasoning_dry_run",
        action="store_true",
        help="Run LLM reasoning without API calls (mock outputs).",
    )
    p.add_argument(
        "--llm_reasoning_dry_run_fast",
        action="store_true",
        help="Run a fast dry-run (small subset, no sleep, no training).",
    )
    p.add_argument(
        "--llm_sweep_repeats",
        type=int,
        default=3,
        help="Repeat each LLM sweep n_rules value this many times.",
    )
    p.add_argument(
        "--llm_sweep_range",
        default="1-15",
        help="Range for LLM sweep when --llm_sweep is set (default: 1-15).",
    )
    p.add_argument(
        "--llm_retry_attempts",
        type=int,
        default=2,
        help="Retry LLM generation this many times on failure during sweep.",
    )
    p.add_argument(
        "--llm_retry_sleep",
        type=float,
        default=10.0,
        help="Seconds to sleep between LLM generation retries.",
    )
    p.add_argument(
        "--llm_timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for each LLM generation call during sweep.",
    )
    p.add_argument(
        "--llm_sweep_start_rule",
        type=int,
        default=1,
        help="Start sweep at this n_rules value (default: 1).",
    )
    p.add_argument(
        "--llm_sweep_start_repeat",
        type=int,
        default=1,
        help="Start sweep at this repeat index for the start rule (default: 1).",
    )
    p.add_argument(
        "--llm_sweep",
        action="store_true",
        help=(
            "Sweep LLM n_rules over a range (default 1-15) in LLM-only mode."
        ),
    )
    p.add_argument(
        "--base_script",
        default=str(
            Path(
                r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
            )
            / "2_Human_features_running_example_script"
            / "vcbench_lambda_features_minimal.py"
        ),
        help="Path to base feature script with _extract_human_features.",
    )
    p.add_argument(
        "--output_parquet",
        default="",
        help="Optional output parquet path for extracted features.",
    )
    return p.parse_args()


def _resolve_input_csv(dataset: str, override: str) -> str:
    if override:
        return override
    base = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit"
    if dataset == "sample":
        return str(base / "vcbench_final_public_sample100.csv")
    return str(base / "vcbench_final_public.csv")


def _parse_sweep_range(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return list(range(1, 16))
    if "-" in spec:
        parts = spec.split("-", 1)
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            if start <= 0 or end <= 0:
                raise ValueError
            if start > end:
                start, end = end, start
            return list(range(start, end + 1))
        except Exception:
            return list(range(1, 16))
    # comma-separated list
    vals = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            v = int(piece)
            if v > 0:
                vals.append(v)
        except Exception:
            continue
    return sorted(set(vals)) if vals else list(range(1, 16))


def _select_dataset(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    dataset_size: str,
    random_state: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if dataset_size == "full":
        return records, labels
    try:
        n = int(dataset_size)
    except Exception as exc:
        raise ValueError(f"Invalid dataset_size: {dataset_size}") from exc
    n = max(1, min(n, len(records)))
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(records), size=n, replace=False)
    return [records[i] for i in idx], labels[idx]


def _load_env_if_present() -> None:
    # Load .env from parent of Project_folder if present (one level above this file's parent).
    env_path = Path(__file__).resolve().parents[1].parent / ".env"
    if not env_path.exists():
        return
    try:
        from think_reason_learn.core import _config as trl_config
        for line in env_path.read_text(encoding="utf-8").splitlines():
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
        # Best-effort only; do not crash if .env is malformed.
        return


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    rs = args.random_state
    input_csv = _resolve_input_csv(args.dataset, args.input_csv)

    # Load config early for reasoning paths/options
    selected_features: list[str] = []
    cfg_use_llm: bool | None = None
    cfg_llm_n: int | None = None
    cfg_use_llm_reasoning: bool | None = None
    cfg_llm_reasoning_features: list[str] | None = None
    cfg_llm_reasoning_experiments: list[str] | None = None
    cfg_llm_reasoning_mode: str | None = None
    cfg_llm_reasoning_sequential: list[str] | None = None
    cfg_llm_reasoning_combined: list[str] | None = None
    cfg_llm_temperature: float | None = None
    cfg_llm_reasoning_dataset_size: str | None = None
    cfg_llm_reasoning_prompts_path: str | None = None
    cfg_llm_reasoning_core_prompt_path: str | None = None
    cfg_llm_reasoning_experiments_path: str | None = None
    cfg_llm_reasoning_dry_run: bool | None = None
    cfg_llm_reasoning_dry_run_fast: bool | None = None
    cfg_llm_providers: dict[str, bool] | None = None
    cfg_llm_google_model: str | None = None
    cfg_llm_reasoning_batch_size: int | None = None
    cfg_llm_reasoning_log_every: int | None = None
    cfg_llm_reasoning_concurrency: int | None = None
    cfg_llm_reasoning_repair_nan: bool | None = None
    cfg_llm_reasoning_repair_existing: bool | None = None
    cfg_llm_engineered_for_reasoning: bool | None = None
    cfg_llm_reasoning_split_batches: bool | None = None
    cfg_llm_engineered_cache: bool | None = None
    cfg_llm_engineered_freeze: bool | None = None
    cfg_llm_engineered_run_family: bool | None = None
    cfg_llm_engineered_run_family_size: int | None = None
    cfg_llm_engineered_run_family_n: int | None = None
    cfg_llm_engineered_run_family_id: str | None = None
    cfg_path = Path(args.feature_config) if args.feature_config else None
    if cfg_path is not None and cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        selected_features = [f for f in data.get("features", []) if isinstance(f, str)]
        if "use_llm" in data:
            cfg_use_llm = bool(data.get("use_llm"))
        if "llm_n_features" in data:
            try:
                cfg_llm_n = int(data.get("llm_n_features"))
            except Exception:
                cfg_llm_n = None
        if "use_llm_reasoning" in data:
            cfg_use_llm_reasoning = bool(data.get("use_llm_reasoning"))
        if isinstance(data.get("llm_reasoning_features"), list):
            cfg_llm_reasoning_features = [
                f for f in data.get("llm_reasoning_features", []) if isinstance(f, str)
            ]
        if isinstance(data.get("llm_reasoning_dataset_size"), str):
            cfg_llm_reasoning_dataset_size = data.get("llm_reasoning_dataset_size")
        if isinstance(data.get("llm_reasoning_prompts_path"), str):
            cfg_llm_reasoning_prompts_path = data.get("llm_reasoning_prompts_path")
        if isinstance(data.get("llm_reasoning_core_prompt_path"), str):
            cfg_llm_reasoning_core_prompt_path = data.get("llm_reasoning_core_prompt_path")
        if isinstance(data.get("llm_reasoning_experiments_path"), str):
            cfg_llm_reasoning_experiments_path = data.get("llm_reasoning_experiments_path")
        if isinstance(data.get("llm_reasoning_experiments"), list):
            cfg_llm_reasoning_experiments = [
                e for e in data.get("llm_reasoning_experiments", []) if isinstance(e, str)
            ]
        if isinstance(data.get("llm_reasoning_mode"), str):
            cfg_llm_reasoning_mode = data.get("llm_reasoning_mode")
        if isinstance(data.get("llm_reasoning_sequential"), list):
            cfg_llm_reasoning_sequential = [
                e for e in data.get("llm_reasoning_sequential", []) if isinstance(e, str)
            ]
        if isinstance(data.get("llm_reasoning_combined"), list):
            cfg_llm_reasoning_combined = [
                e for e in data.get("llm_reasoning_combined", []) if isinstance(e, str)
            ]
        if "llm_reasoning_dry_run" in data:
            cfg_llm_reasoning_dry_run = bool(data.get("llm_reasoning_dry_run"))
        if "llm_reasoning_dry_run_fast" in data:
            cfg_llm_reasoning_dry_run_fast = bool(data.get("llm_reasoning_dry_run_fast"))
        if isinstance(data.get("llm_providers"), dict):
            cfg_llm_providers = {
                "openai": bool(data.get("llm_providers", {}).get("openai", False)),
                "google": bool(data.get("llm_providers", {}).get("google", False)),
            }
        if isinstance(data.get("llm_google_model"), str):
            cfg_llm_google_model = data.get("llm_google_model")
        if isinstance(data.get("llm_reasoning_batch_size"), int):
            cfg_llm_reasoning_batch_size = data.get("llm_reasoning_batch_size")
        if isinstance(data.get("llm_reasoning_log_every"), int):
            cfg_llm_reasoning_log_every = data.get("llm_reasoning_log_every")
        if isinstance(data.get("llm_reasoning_concurrency"), int):
            cfg_llm_reasoning_concurrency = data.get("llm_reasoning_concurrency")
        if "llm_reasoning_repair_nan" in data:
            cfg_llm_reasoning_repair_nan = bool(data.get("llm_reasoning_repair_nan"))
        if "llm_reasoning_repair_existing" in data:
            cfg_llm_reasoning_repair_existing = bool(data.get("llm_reasoning_repair_existing"))
        if "llm_reasoning_split_batches" in data:
            cfg_llm_reasoning_split_batches = bool(data.get("llm_reasoning_split_batches"))
        if "llm_engineered_for_reasoning" in data:
            cfg_llm_engineered_for_reasoning = bool(data.get("llm_engineered_for_reasoning"))
        if "llm_engineered_cache" in data:
            cfg_llm_engineered_cache = bool(data.get("llm_engineered_cache"))
        if "llm_engineered_freeze" in data:
            cfg_llm_engineered_freeze = bool(data.get("llm_engineered_freeze"))
        if "llm_engineered_run_family" in data:
            cfg_llm_engineered_run_family = bool(data.get("llm_engineered_run_family"))
        if "llm_engineered_run_family_size" in data:
            try:
                cfg_llm_engineered_run_family_size = int(data.get("llm_engineered_run_family_size"))
            except Exception:
                cfg_llm_engineered_run_family_size = None
        if "llm_engineered_run_family_n_features" in data:
            try:
                cfg_llm_engineered_run_family_n = int(data.get("llm_engineered_run_family_n_features"))
            except Exception:
                cfg_llm_engineered_run_family_n = None
        if "llm_engineered_run_family_id" in data:
            cfg_llm_engineered_run_family_id = str(data.get("llm_engineered_run_family_id"))
        if "llm_temperature" in data:
            try:
                cfg_llm_temperature = float(data.get("llm_temperature"))
            except Exception:
                cfg_llm_temperature = None
    # Set up run-level logging early for full traceability
    run_log = None
    log_root = None
    use_llm_reasoning_early = (
        args.llm_reasoning
        or (cfg_use_llm_reasoning is True)
        or (args.mode in ("reasoning", "hybrid"))
    )
    if use_llm_reasoning_early:
        log_root = Path(__file__).parent / "logging" / f"llm_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_root.mkdir(parents=True, exist_ok=True)
        run_log = log_root / "run_log.txt"
        run_log.write_text("run_log started\n", encoding="utf-8")
        global RUN_LOG_PATH
        RUN_LOG_PATH = run_log

    def _log_run(msg: str) -> None:
        if run_log is not None:
            with run_log.open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    log_lines: list[str] = []
    sweep_terminal_log: Path | None = None

    def _log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)
        if sweep_terminal_log is not None:
            sweep_terminal_log.parent.mkdir(parents=True, exist_ok=True)
            with sweep_terminal_log.open("a", encoding="utf-8") as f:
                f.write(msg + "\n")

    mode = args.mode
    _log(f"\n{'=' * 60}")
    _log("  VCBench In-Depth Pipeline")
    _log(f"{'=' * 60}\n")
    _log(f"  Dataset: {args.dataset}")
    _log(f"  Input CSV: {input_csv}")

    _log_run(f"Loading dataset from: {input_csv}")
    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        rs,
    )
    _log_run(f"Loaded records: {len(records)} labels: {len(labels)}")

    # Resolve LLM reasoning settings early to allow dataset sizing
    reasoning_core_prompt_path = (
        Path(cfg_llm_reasoning_core_prompt_path)
        if cfg_llm_reasoning_core_prompt_path
        else Path(args.llm_reasoning_core_prompt)
    )
    reasoning_experiments_path = (
        Path(cfg_llm_reasoning_experiments_path)
        if cfg_llm_reasoning_experiments_path
        else Path(args.llm_reasoning_experiments)
    )
    if not reasoning_core_prompt_path.is_absolute():
        reasoning_core_prompt_path = Path(__file__).parent / reasoning_core_prompt_path
    if not reasoning_experiments_path.is_absolute():
        reasoning_experiments_path = Path(__file__).parent / reasoning_experiments_path
    llm_providers = cfg_llm_providers or {"openai": True, "google": False}
    llm_google_model = cfg_llm_google_model
    llm_reasoning_batch_size = (
        cfg_llm_reasoning_batch_size
        if cfg_llm_reasoning_batch_size is not None
        else args.llm_reasoning_batch_size
    )
    llm_reasoning_log_every = cfg_llm_reasoning_log_every if cfg_llm_reasoning_log_every is not None else 10
    llm_reasoning_concurrency = cfg_llm_reasoning_concurrency if cfg_llm_reasoning_concurrency is not None else 1
    llm_reasoning_repair_nan = cfg_llm_reasoning_repair_nan if cfg_llm_reasoning_repair_nan is not None else True
    llm_reasoning_repair_existing = cfg_llm_reasoning_repair_existing if cfg_llm_reasoning_repair_existing is not None else False
    llm_engineered_for_reasoning = cfg_llm_engineered_for_reasoning if cfg_llm_engineered_for_reasoning is not None else False
    llm_reasoning_split_batches = (
        cfg_llm_reasoning_split_batches if cfg_llm_reasoning_split_batches is not None else True
    )
    llm_engineered_cache = cfg_llm_engineered_cache if cfg_llm_engineered_cache is not None else True
    llm_engineered_freeze = cfg_llm_engineered_freeze if cfg_llm_engineered_freeze is not None else False
    llm_engineered_run_family = cfg_llm_engineered_run_family if cfg_llm_engineered_run_family is not None else False
    llm_engineered_run_family_size = cfg_llm_engineered_run_family_size if cfg_llm_engineered_run_family_size is not None else 10
    llm_engineered_run_family_n = cfg_llm_engineered_run_family_n if cfg_llm_engineered_run_family_n is not None else None
    llm_engineered_run_family_id = cfg_llm_engineered_run_family_id
    llm_temperature = cfg_llm_temperature if cfg_llm_temperature is not None else 0.0
    llm_reasoning_dry_run = bool(args.llm_reasoning_dry_run) or bool(cfg_llm_reasoning_dry_run)
    llm_reasoning_dry_run_fast = bool(args.llm_reasoning_dry_run_fast) or bool(cfg_llm_reasoning_dry_run_fast)
    reasoning_dataset_size = (
        cfg_llm_reasoning_dataset_size
        if cfg_llm_reasoning_dataset_size is not None
        else args.llm_reasoning_dataset_size
    )
    use_llm_reasoning = (
        args.llm_reasoning
        or (cfg_use_llm_reasoning is True)
        or (args.mode in ("reasoning", "hybrid"))
    )
    _log_run(f"Use LLM reasoning: {use_llm_reasoning}")
    _log_run(
        "Config flags: "
        f"llm_engineered_run_family={llm_engineered_run_family} "
        f"llm_engineered_freeze={llm_engineered_freeze} "
        f"llm_engineered_run_family_id={llm_engineered_run_family_id}"
    )
    if llm_reasoning_dry_run_fast:
        llm_reasoning_dry_run = True
        use_llm_reasoning = True
        if mode == "human":
            mode = "reasoning"

    if use_llm_reasoning and args.dataset == "sample":
        raise RuntimeError("LLM reasoning does not allow dataset=sample. Use full or a size override.")

    if use_llm_reasoning and reasoning_dataset_size != "full":
        records, labels = _select_dataset(records, labels, reasoning_dataset_size, rs)

    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=labels,
        random_state=rs,
    )
    if set(train_idx) & set(test_idx):
        raise RuntimeError("Train/test split overlap detected.")
    train_recs = [records[i] for i in train_idx]
    test_recs = [records[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    _log(f"  Split: {len(train_idx)} train / {len(test_idx)} test")
    _log(
        f"  Positives: train={int(y_train.sum())}, test={int(y_test.sum())}"
    )
    _log_run(f"Split train={len(train_idx)} test={len(test_idx)}")
    _log_run(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")

    if llm_reasoning_dry_run_fast:
        _log("  Dry-run fast enabled: generating reasoning features only.")
        output_dir = Path(__file__).parent / "features_storage" / "llm_reasoning"
        meta_path = output_dir / f"llm_reasoning_dry_run_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config = ReasoningConfig(
            model=args.llm_model,
            dataset_size=reasoning_dataset_size,
            random_state=rs,
            core_prompt_path=reasoning_core_prompt_path,
            experiments_path=reasoning_experiments_path,
            providers=llm_providers,
            google_model=llm_google_model,
            batch_size=llm_reasoning_batch_size,
            concurrency=llm_reasoning_concurrency,
            experiments=cfg_llm_reasoning_experiments,
            dry_run=True,
            dry_run_fast=True,
            repair_nan=llm_reasoning_repair_nan,
            repair_existing=llm_reasoning_repair_existing,
        )
        reasoning_df, all_reasoning_names = generate_reasoning_features(
            records=records,
            labels=labels,
            config=config,
            output_dir=output_dir,
            metadata_path=meta_path,
        )
        _log(f"  Dry-run fast output rows: {len(reasoning_df)}")
        _log("  Dry-run fast complete; skipping training.")
        _write_log(
            log_lines,
            args,
            input_csv,
            all_reasoning_names,
            len(train_idx),
            len(test_idx),
            int(y_train.sum()),
            int(y_test.sum()),
            {"threshold": None},
            log_dir=_log_dir_for_mode("reasoning", True, True),
        )
        return

    base_script = Path(args.base_script)
    extract_base = _load_base_feature_extractor(base_script)
    base_all = pd.DataFrame([extract_base(r) for r in records])
    base_train = pd.DataFrame([extract_base(r) for r in train_recs])
    base_test = pd.DataFrame([extract_base(r) for r in test_recs])
    base_feature_names = list(base_train.columns)

    if selected_features:
        base_selected = [f for f in selected_features if f in base_feature_names]
        custom_features = [f for f in selected_features if f in FEATURE_REGISTRY]
        unknown = [
            f
            for f in selected_features
            if f not in base_feature_names and f not in FEATURE_REGISTRY
        ]
        if unknown:
            print(f"  WARNING: Unknown features in config: {', '.join(unknown)}")
    elif args.features.strip():
        base_selected = base_feature_names
        custom_features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        base_selected = base_feature_names
        custom_features = FEATURE_SETS[args.feature_set]

    # Apply baseline selection if provided
    if selected_features:
        base_all = base_all[base_selected]
        base_train = base_train[base_selected]
        base_test = base_test[base_selected]
        base_feature_names = list(base_train.columns)

    custom_all = (
        _custom_feature_df(records, custom_features)
        if custom_features
        else pd.DataFrame(index=range(len(records)))
    )
    custom_train = (
        _custom_feature_df(train_recs, custom_features)
        if custom_features
        else pd.DataFrame(index=range(len(train_recs)))
    )
    custom_test = (
        _custom_feature_df(test_recs, custom_features)
        if custom_features
        else pd.DataFrame(index=range(len(test_recs)))
    )

    llm_feature_names: list[str] = []
    llm_all = pd.DataFrame(index=range(len(records)))
    llm_train = pd.DataFrame(index=range(len(train_recs)))
    llm_test = pd.DataFrame(index=range(len(test_recs)))

    reasoning_feature_names: list[str] = []
    reasoning_all = pd.DataFrame(index=range(len(records)))
    reasoning_train = pd.DataFrame(index=range(len(train_recs)))
    reasoning_test = pd.DataFrame(index=range(len(test_recs)))
    llm_engineered_feature_names: list[str] = []

    mode = args.mode
    if cfg_use_llm is True and mode == "human":
        mode = "hybrid"
    if cfg_use_llm_reasoning is True and mode == "human":
        mode = "hybrid"

    _log(f"  Mode: {mode}\n")

    llm_n = cfg_llm_n if cfg_llm_n is not None else args.llm_n_features
    sweep_enabled = args.llm_sweep
    sweep_rules = _parse_sweep_range(args.llm_sweep_range)
    sweep_repeats = max(1, int(args.llm_sweep_repeats))
    use_llm = mode in ("llm", "hybrid") or args.llm_features
    use_llm_reasoning = use_llm_reasoning or (mode in ("reasoning", "hybrid"))
    if use_llm:
        _log(f"  OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
        if sweep_enabled:
            sweep_dir = Path(__file__).parent / "training_logs" / f"llm_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            features_storage = Path(__file__).parent / "features_storage" / "llm_engineered"
            features_storage.mkdir(parents=True, exist_ok=True)
            sweep_terminal_log = sweep_dir / "terminal_log.txt"
            _log("Sweep terminal log started.")
            summary_rows = []
            start_rule = max(1, int(args.llm_sweep_start_rule))
            start_repeat = max(1, int(args.llm_sweep_start_repeat))
            for n_rules in sweep_rules:
                for repeat_idx in range(1, sweep_repeats + 1):
                    if n_rules < start_rule:
                        continue
                    if n_rules == start_rule and repeat_idx < start_repeat:
                        continue
                    _log(
                        f"\n  Generating {n_rules} LLM features with {args.llm_model}... (repeat {repeat_idx}/{sweep_repeats})"
                    )
                    attempt = 0
                    while True:
                        try:
                            llm_all, llm_train, llm_test, llm_feature_names = asyncio.run(
                                asyncio.wait_for(
                                    generate_llm_features(
                                        train_recs=train_recs,
                                        y_train=y_train,
                                        test_recs=test_recs,
                                        model=args.llm_model,
                                        n_features=n_rules,
                                        all_recs=records,
                                        providers=llm_providers,
                                        google_model=llm_google_model,
                                    ),
                                    timeout=float(args.llm_timeout),
                                )
                            )
                            break
                        except Exception:
                            attempt += 1
                            err = traceback.format_exc()
                            _log(f"\n  ERROR during LLM generation (attempt {attempt}):")
                            _log(err)
                            if attempt > max(0, int(args.llm_retry_attempts)):
                                _log("  Exceeded retry attempts. Skipping this run.")
                                llm_all = llm_train = llm_test = pd.DataFrame(index=range(len(records)))
                                llm_feature_names = []
                                break
                            _log(f"  Retrying in {args.llm_retry_sleep:.1f}s...")
                            import time
                            time.sleep(max(0.0, float(args.llm_retry_sleep)))
                    if not llm_feature_names:
                        continue

                    full_all = llm_all
                    full_train = llm_train
                    full_test = llm_test
                    feature_names = llm_feature_names
                    mode_label = "LLM Only"

                    X_train = full_train.values.astype(float)
                    X_test = full_test.values.astype(float)

                    train_scores, test_scores, model = _train_sklearn(
                        X_train, y_train, X_test, rs
                    )
                    metrics = _report_metrics(y_train, train_scores, y_test, test_scores)
                    acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
                    metrics["accuracy"] = acc

                    # Save features for this n_rules + repeat
                    llm_df = pd.concat(
                        [
                            pd.Series([r.get("founder_uuid") for r in records], name="founder_uuid"),
                            pd.Series(labels, name="success"),
                            full_all,
                        ],
                        axis=1,
                    )
                    llm_df.to_parquet(
                        features_storage / f"llm_features_{n_rules}_r{repeat_idx}.parquet",
                        index=False,
                    )

                    # Write run log
                    run_id = f"run_{n_rules}_{repeat_idx}"
                    run_lines = []
                    run_lines.append(f"Requested n_rules: {n_rules}")
                    run_lines.append(f"Repeat: {repeat_idx}")
                    run_lines.append(f"Resultant features: {len(feature_names)}")
                    run_lines.append(f"Features used: {', '.join(feature_names)}")
                    run_lines.append(f"[{mode_label}] {len(feature_names)} features, threshold={metrics['threshold']:.2f}")
                    run_lines.append("\nWeights:")
                    run_lines.append(f"{'feature':<40} {'coef':>8}")
                    run_lines.append("-" * 50)
                    for name, c in sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
                        run_lines.append(f"{name:<40} {c:>8.3f}")
                    run_lines.append(
                        f"ROC-AUC={metrics['roc_auc']:.3f} PR-AUC={metrics['pr_auc']:.3f} "
                        f"Prec={metrics['precision']:.3f} Rec={metrics['recall']:.3f} "
                        f"F0.5={metrics['f0.5']:.3f} Acc={acc:.3f}"
                    )
                    run_lines.extend(_format_topk(metrics))
                    (sweep_dir / f"{run_id}.txt").write_text("\n".join(run_lines), encoding="utf-8")

                    row = {
                        "run_id": run_id,
                        "repeat": repeat_idx,
                        "n_rules": n_rules,
                        "resultant_features": len(feature_names),
                        "roc_auc": metrics["roc_auc"],
                        "f0.5": metrics["f0.5"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "accuracy": acc,
                        "precision@1%": metrics["precision@1%"],
                        "precision@5%": metrics["precision@5%"],
                        "precision@10%": metrics["precision@10%"],
                    }
                    for name, c in sorted(zip(feature_names, model.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
                        row[f"w_{name}"] = c
                    summary_rows.append(row)

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(sweep_dir / "llm_sweep_summary.csv", index=False)
            try:
                import matplotlib.pyplot as plt  # type: ignore
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                x = summary_df["resultant_features"]
                axes[0, 0].scatter(x, summary_df["roc_auc"])
                axes[0, 0].set_title("ROC-AUC")
                axes[0, 1].scatter(x, summary_df["f0.5"])
                axes[0, 1].set_title("F0.5")
                axes[1, 0].scatter(x, summary_df["recall"], label="Recall")
                if "precision" in summary_df.columns:
                    axes[1, 0].scatter(x, summary_df["precision"], label="Precision")
                    axes[1, 0].legend()
                axes[1, 0].set_title("Recall + Precision")
                axes[1, 1].scatter(x, summary_df["accuracy"])
                axes[1, 1].set_title("Accuracy")
                for ax in axes.flat:
                    ax.set_xlabel("Resultant features")
                plt.tight_layout()
                plt.savefig(sweep_dir / "llm_sweep_metrics.png", dpi=150)
                plt.close(fig)
            except Exception:
                pass

            return
        else:
            _log(f"\n  Generating {llm_n} LLM features with {args.llm_model}...")
            llm_all, llm_train, llm_test, llm_feature_names = asyncio.run(
                generate_llm_features(
                    train_recs=train_recs,
                    y_train=y_train,
                    test_recs=test_recs,
                    model=args.llm_model,
                    n_features=llm_n,
                    all_recs=records,
                    providers=llm_providers,
                    google_model=llm_google_model,
                )
            )

    if use_llm_reasoning:
        output_root = Path(__file__).parent / "features_storage" / "llm_reasoning"
        current_root = output_root / "current"
        runs_root = output_root / "runs"
        if log_root is None:
            log_root = Path(__file__).parent / "logging" / f"llm_reasoning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            log_root.mkdir(parents=True, exist_ok=True)
            run_log = log_root / "run_log.txt"

        def _load_existing_reasoning(
            exp_list: list[str] | None,
        ) -> tuple[pd.DataFrame | None, list[str]]:
            # Prefer current/ if present
            current_path = current_root / "llm_reasoning_full.parquet"
            if current_path.exists():
                df = pd.read_parquet(current_path)
                if len(df) == len(records):
                    feature_cols = [c for c in df.columns if c not in ("founder_uuid", "success")]
                    return df, feature_cols
            # Fallback: most recent run in runs/
            if runs_root.exists():
                run_dirs = sorted(
                    [p for p in runs_root.glob("run_*") if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for run_dir in run_dirs:
                    run_path = run_dir / "llm_reasoning_full.parquet"
                    if run_path.exists():
                        df = pd.read_parquet(run_path)
                        if len(df) == len(records):
                            feature_cols = [c for c in df.columns if c not in ("founder_uuid", "success")]
                            return df, feature_cols
            return None, []

        def _sync_reasoning_current(run_dir: Path, exp_list: list[str] | None) -> None:
            current_root.mkdir(parents=True, exist_ok=True)
            src_full = run_dir / "llm_reasoning_full.parquet"
            if src_full.exists():
                shutil.copy2(src_full, current_root / "llm_reasoning_full.parquet")
            # Copy per-experiment outputs if present
            exp_src = run_dir / "experiments"
            exp_dst = current_root / "experiments"
            if exp_src.exists():
                if exp_dst.exists():
                    shutil.rmtree(exp_dst)
                shutil.copytree(exp_src, exp_dst)
            # Write a minimal manifest
            manifest = {
                "source_run": run_dir.name,
                "experiments": exp_list or [],
                "timestamp": datetime.now().isoformat(),
            }
            (current_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        def _reasoning_has_nans(path: Path) -> bool:
            if not path.exists():
                return False
            df = pd.read_parquet(path)
            num = df.select_dtypes(include=[np.number]).drop(columns=["success"], errors="ignore")
            if num.empty:
                return False
            return bool(num.isna().any(axis=1).any())

        def _generate_reasoning_split_batches(
            experiments_list: list[str] | None,
            output_dir: Path,
            meta_path: Path,
            log_label: str,
        ) -> tuple[pd.DataFrame, list[str]]:
            train_dir = output_dir / "split_train"
            test_dir = output_dir / "split_test"
            train_meta = meta_path.with_name(meta_path.stem + "_train.json")
            test_meta = meta_path.with_name(meta_path.stem + "_test.json")

            train_config = ReasoningConfig(
                model=args.llm_model,
                dataset_size=reasoning_dataset_size,
                random_state=rs,
                core_prompt_path=reasoning_core_prompt_path,
                experiments_path=reasoning_experiments_path,
                providers=llm_providers,
                google_model=llm_google_model,
                batch_size=llm_reasoning_batch_size,
                concurrency=llm_reasoning_concurrency,
                experiments=experiments_list,
                dry_run=llm_reasoning_dry_run,
                dry_run_fast=llm_reasoning_dry_run_fast,
                log_dir=log_root / f"{log_label}_train",
                log_every=llm_reasoning_log_every,
                repair_nan=llm_reasoning_repair_nan,
                repair_existing=llm_reasoning_repair_existing,
                skip_select=True,
            )
            test_config = ReasoningConfig(
                model=args.llm_model,
                dataset_size=reasoning_dataset_size,
                random_state=rs,
                core_prompt_path=reasoning_core_prompt_path,
                experiments_path=reasoning_experiments_path,
                providers=llm_providers,
                google_model=llm_google_model,
                batch_size=llm_reasoning_batch_size,
                concurrency=llm_reasoning_concurrency,
                experiments=experiments_list,
                dry_run=llm_reasoning_dry_run,
                dry_run_fast=llm_reasoning_dry_run_fast,
                log_dir=log_root / f"{log_label}_test",
                log_every=llm_reasoning_log_every,
                repair_nan=llm_reasoning_repair_nan,
                repair_existing=llm_reasoning_repair_existing,
                skip_select=True,
            )

            train_df, _ = generate_reasoning_features(
                records=train_recs,
                labels=y_train,
                config=train_config,
                output_dir=train_dir,
                metadata_path=train_meta,
            )
            test_df, _ = generate_reasoning_features(
                records=test_recs,
                labels=y_test,
                config=test_config,
                output_dir=test_dir,
                metadata_path=test_meta,
            )

            feature_cols = [c for c in train_df.columns if c not in ("founder_uuid", "success")]
            test_cols = [c for c in test_df.columns if c not in ("founder_uuid", "success")]
            if set(feature_cols) != set(test_cols):
                raise RuntimeError("Train/test reasoning feature columns do not match.")
            feature_cols = sorted(feature_cols)
            full_features = pd.DataFrame(index=range(len(records)), columns=feature_cols)
            full_features.iloc[train_idx] = train_df[feature_cols].to_numpy()
            full_features.iloc[test_idx] = test_df[feature_cols].to_numpy()
            combined_df = pd.DataFrame(
                {
                    "founder_uuid": [r.get("founder_uuid") for r in records],
                    "success": labels,
                }
            )
            for col in feature_cols:
                combined_df[col] = full_features[col].values

            output_dir.mkdir(parents=True, exist_ok=True)
            combined_path = output_dir / f"llm_reasoning_{reasoning_dataset_size}.parquet"
            combined_df.to_parquet(combined_path, index=False)

            _, exp_to_keys = build_experiment_key_map(
                reasoning_experiments_path, experiments_list
            )
            write_per_experiment_parquets(
                combined_df,
                exp_to_keys,
                output_dir,
                reasoning_dataset_size,
                overwrite=True,
            )

            meta = {
                "model": args.llm_model,
                "dataset_size": reasoning_dataset_size,
                "random_state": rs,
                "core_prompt_path": str(reasoning_core_prompt_path),
                "experiments_path": str(reasoning_experiments_path),
                "experiments": experiments_list,
                "output_parquet": str(combined_path),
                "train_meta": str(train_meta),
                "test_meta": str(test_meta),
                "split_batches": True,
                "batch_size": llm_reasoning_batch_size,
                "concurrency": llm_reasoning_concurrency,
                "dry_run": llm_reasoning_dry_run,
                "dry_run_fast": llm_reasoning_dry_run_fast,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return combined_df, feature_cols

        reasoning_mode = cfg_llm_reasoning_mode or "single"
        sequential = cfg_llm_reasoning_sequential or []
        combined = cfg_llm_reasoning_combined or []
        if reasoning_mode == "sequential_and_combined" and (sequential or combined):
            _log_run(f"Reasoning mode: sequential_and_combined")
            # Sequential runs
            for exp_id in sequential:
                _log_run(f"Starting sequential experiment {exp_id}")
                runs_root.mkdir(parents=True, exist_ok=True)
                exp_dir = runs_root / f"run_{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                meta_path = exp_dir / f"llm_reasoning_{reasoning_dataset_size}_{exp_id}.json"
                if llm_reasoning_split_batches:
                    _generate_reasoning_split_batches([exp_id], exp_dir, meta_path, exp_id)
                else:
                    config = ReasoningConfig(
                        model=args.llm_model,
                        dataset_size=reasoning_dataset_size,
                        random_state=rs,
                        core_prompt_path=reasoning_core_prompt_path,
                        experiments_path=reasoning_experiments_path,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        batch_size=llm_reasoning_batch_size,
                        concurrency=llm_reasoning_concurrency,
                        experiments=[exp_id],
                        dry_run=llm_reasoning_dry_run,
                        dry_run_fast=llm_reasoning_dry_run_fast,
                        log_dir=log_root / f"{exp_id}_progress",
                        log_every=llm_reasoning_log_every,
                        repair_nan=llm_reasoning_repair_nan,
                        repair_existing=llm_reasoning_repair_existing,
                    )
                    generate_reasoning_features(
                        records=records,
                        labels=labels,
                        config=config,
                        output_dir=exp_dir,
                        metadata_path=meta_path,
                    )
                _sync_reasoning_current(exp_dir, [exp_id])
                _log_run(f"Completed sequential experiment {exp_id}")

            # Combined run
            if combined:
                combo_id = "combined_" + "".join(combined)
                _log_run(f"Starting combined experiment {combo_id}")
                runs_root.mkdir(parents=True, exist_ok=True)
                exp_dir = runs_root / f"run_{combo_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                exp_dir.mkdir(parents=True, exist_ok=True)
                meta_path = exp_dir / f"llm_reasoning_{reasoning_dataset_size}_{combo_id}.json"
                if llm_reasoning_split_batches:
                    _generate_reasoning_split_batches(combined, exp_dir, meta_path, combo_id)
                else:
                    config = ReasoningConfig(
                        model=args.llm_model,
                        dataset_size=reasoning_dataset_size,
                        random_state=rs,
                        core_prompt_path=reasoning_core_prompt_path,
                        experiments_path=reasoning_experiments_path,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        batch_size=llm_reasoning_batch_size,
                        concurrency=llm_reasoning_concurrency,
                        experiments=combined,
                        dry_run=llm_reasoning_dry_run,
                        dry_run_fast=llm_reasoning_dry_run_fast,
                        log_dir=log_root / f"{combo_id}_progress",
                        log_every=llm_reasoning_log_every,
                        repair_nan=llm_reasoning_repair_nan,
                        repair_existing=llm_reasoning_repair_existing,
                    )
                    generate_reasoning_features(
                        records=records,
                        labels=labels,
                        config=config,
                        output_dir=exp_dir,
                        metadata_path=meta_path,
                    )
                _sync_reasoning_current(exp_dir, combined)
                _log_run(f"Completed combined experiment {combo_id}")
            # Skip training path for sequential/combined batch generation
            return
        else:
            runs_root.mkdir(parents=True, exist_ok=True)
            run_dir = runs_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_dir = run_dir
            meta_path = run_dir / f"llm_reasoning_{reasoning_dataset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            exp_list = cfg_llm_reasoning_experiments
            exp_id = exp_list[0] if exp_list and len(exp_list) == 1 else None
            run_parquet = run_dir / f"llm_reasoning_{reasoning_dataset_size}.parquet"
            if exp_id and run_dir.exists() and llm_reasoning_repair_existing and _reasoning_has_nans(run_parquet):
                reasoning_df, all_reasoning_names = _generate_reasoning_split_batches(
                    exp_list,
                    run_dir,
                    meta_path,
                    f"{exp_id}_repair",
                )
                _log("  Repaired cached LLM reasoning features for run_A.")
            else:
                cached_df, cached_names = _load_existing_reasoning(exp_list)
                if cached_df is not None and cached_names:
                    reasoning_df = cached_df
                    all_reasoning_names = cached_names
                    _log("  Loaded cached LLM reasoning features from run_A output.")
                elif llm_reasoning_split_batches:
                    reasoning_df, all_reasoning_names = _generate_reasoning_split_batches(
                        exp_list,
                        output_dir,
                        meta_path,
                        "single_progress",
                    )
                else:
                    config = ReasoningConfig(
                        model=args.llm_model,
                        dataset_size=reasoning_dataset_size,
                        random_state=rs,
                        core_prompt_path=reasoning_core_prompt_path,
                        experiments_path=reasoning_experiments_path,
                        providers=llm_providers,
                        google_model=llm_google_model,
                        batch_size=llm_reasoning_batch_size,
                        concurrency=llm_reasoning_concurrency,
                        experiments=exp_list,
                        dry_run=llm_reasoning_dry_run,
                        dry_run_fast=llm_reasoning_dry_run_fast,
                        log_dir=log_root / "single_progress",
                        log_every=llm_reasoning_log_every,
                        repair_nan=llm_reasoning_repair_nan,
                        repair_existing=llm_reasoning_repair_existing,
                    )
                    reasoning_df, all_reasoning_names = generate_reasoning_features(
                        records=records,
                        labels=labels,
                        config=config,
                        output_dir=output_dir,
                        metadata_path=meta_path,
                    )
            if "reasoning_df" in locals():
                _sync_reasoning_current(run_dir, exp_list or [])
            # Use numeric reasoning features for training; keep text in parquet only.
            for col in reasoning_df.columns:
                if col not in ("founder_uuid", "success"):
                    reasoning_df[col] = pd.to_numeric(reasoning_df[col], errors="ignore")
            reasoning_feature_names = [
                c
                for c in reasoning_df.columns
                if c not in ("founder_uuid", "success") and pd.api.types.is_numeric_dtype(reasoning_df[c])
            ]
            if not reasoning_feature_names:
                raise RuntimeError(
                    "No numeric LLM reasoning features detected. "
                    "Check cached reasoning parquet column types."
                )
            reasoning_all = reasoning_df[reasoning_feature_names]
            reasoning_train = reasoning_all.iloc[train_idx].reset_index(drop=True)
            reasoning_test = reasoning_all.iloc[test_idx].reset_index(drop=True)

    # Optional: LLM-engineered features for reasoning experiments
    if llm_engineered_for_reasoning:
        cache_dir = Path(__file__).parent / "features_storage" / "llm_engineered"
        llm_cached_all = None
        llm_cached_names = None
        if llm_engineered_cache:
            llm_cached_all, llm_cached_names = _load_llm_engineered_cache(
                cache_dir=cache_dir,
                expected_rows=len(records),
                expected_n=llm_n,
                model=args.llm_model,
                providers=llm_providers,
                google_model=llm_google_model,
            )
        if llm_cached_all is not None and llm_cached_names is not None:
            llm_all = llm_cached_all
            llm_feature_names = llm_cached_names
            llm_train = llm_all.iloc[train_idx].reset_index(drop=True)
            llm_test = llm_all.iloc[test_idx].reset_index(drop=True)
            _log("  Loaded cached LLM-engineered features.")
        else:
            if llm_engineered_freeze:
                raise RuntimeError(
                    "LLM-engineered features are frozen but cache is missing. "
                    "Set llm_engineered_freeze=false or regenerate the cache."
                )
            _log(f"\n  Generating {llm_n} LLM-engineered features with {args.llm_model}...")
            llm_all, llm_train, llm_test, llm_feature_names = asyncio.run(
                generate_llm_features(
                    train_recs=train_recs,
                    y_train=y_train,
                    test_recs=test_recs,
                    model=args.llm_model,
                    n_features=llm_n,
                    all_recs=records,
                    providers=llm_providers,
                    google_model=llm_google_model,
                )
            )
            if llm_engineered_cache:
                _save_llm_engineered_cache(
                    cache_dir=cache_dir,
                    df=llm_all,
                    feature_names=llm_feature_names,
                    model=args.llm_model,
                    providers=llm_providers,
                    google_model=llm_google_model,
                    n_features=llm_n,
                )
        llm_engineered_feature_names = llm_feature_names


    if mode == "human":
        full_all = pd.concat([base_all, custom_all], axis=1)
        full_train = pd.concat([base_train, custom_train], axis=1)
        full_test = pd.concat([base_test, custom_test], axis=1)
        feature_names = base_feature_names + custom_features
        mode_label = "Human Only"
    elif mode == "llm":
        full_all = llm_all
        full_train = llm_train
        full_test = llm_test
        feature_names = llm_feature_names
        mode_label = "LLM Only"
    elif mode == "reasoning":
        full_all = reasoning_all
        full_train = reasoning_train
        full_test = reasoning_test
        feature_names = reasoning_feature_names
        mode_label = "LLM Reasoning Only"
    else:
        full_all = pd.concat([base_all, custom_all, llm_all, reasoning_all], axis=1)
        full_train = pd.concat([base_train, custom_train, llm_train, reasoning_train], axis=1)
        full_test = pd.concat([base_test, custom_test, llm_test, reasoning_test], axis=1)
        feature_names = base_feature_names + custom_features + llm_feature_names + reasoning_feature_names
        mode_label = "Hybrid"

    # Standardize continuous custom features only (training stats)
    if mode in ("human", "hybrid"):
        full_train, full_test = _standardize_continuous(
            full_train, full_test, custom_features
        )

    # Verify selected features match config (when provided)
    if selected_features and mode in ("human", "hybrid"):
        expected_set = {f for f in selected_features if f in feature_names}
        actual_set = set(feature_names)
        if expected_set != actual_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            raise RuntimeError(
                "Selected feature list mismatch. "
                f"Missing: {missing} Extra: {extra}"
            )

    # Impute missing values using training means (prevents leakage).
    if full_train.isna().any().any() or full_test.isna().any().any():
        fill_values = full_train.mean()
        full_train = full_train.fillna(fill_values)
        full_test = full_test.fillna(fill_values)
        full_all = full_all.fillna(fill_values)

    # NaN/Inf checks
    if not np.isfinite(full_train.values).all() or not np.isfinite(full_test.values).all():
        raise RuntimeError("NaN or Inf detected in feature matrices.")

    # Save feature dataset (founder_uuid, label, features)
    founder_ids = [r.get("founder_uuid") for r in records]
    save_df = pd.concat(
        [
            pd.Series(founder_ids, name="founder_uuid"),
            pd.Series(labels, name="success"),
            full_all,
        ],
        axis=1,
    )
    features_storage = Path(__file__).parent / "features_storage"
    features_storage.mkdir(parents=True, exist_ok=True)
    if args.output_parquet:
        out_path = Path(args.output_parquet)
    else:
        if mode == "human":
            out_path = features_storage / "human" / "features_full.parquet"
        elif mode == "llm":
            out_path = features_storage / "llm_engineered" / "llm_features.parquet"
        elif mode == "reasoning":
            out_path = features_storage / "llm_reasoning" / "current" / "llm_reasoning_full.parquet"
        else:
            out_path = features_storage / "features_full.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_parquet(out_path, index=False)
    _log(f"  Saved features to: {out_path}")

    if args.extract_only:
        _log("  extract_only enabled; skipping training.")
        _write_log(
            log_lines,
            args,
            input_csv,
            feature_names,
            len(train_idx),
            len(test_idx),
            int(y_train.sum()),
            int(y_test.sum()),
            {"threshold": None},
            log_dir=_log_dir_for_mode(mode, llm_reasoning_dry_run, llm_reasoning_dry_run_fast),
        )
        return

    # Optional reasoning-only and reasoning+human runs for consistent comparison.
    if use_llm_reasoning and reasoning_feature_names:
        if not llm_engineered_run_family:
            human_only_log_dir = Path(__file__).parent / "training_logs" / "human" / "only"
            human_train = pd.concat([base_train, custom_train], axis=1)
            human_test = pd.concat([base_test, custom_test], axis=1)
            human_train, human_test = _standardize_continuous(
                human_train, human_test, custom_features
            )
            _train_and_log(
                base_feature_names + custom_features,
                human_train,
                human_test,
                y_train,
                y_test,
                args,
                input_csv,
                "Human Only",
                log_dir=human_only_log_dir,
            )

            _train_and_log(
                reasoning_feature_names,
                reasoning_train,
                reasoning_test,
                y_train,
                y_test,
                args,
                input_csv,
                "LLM Reasoning Only",
                log_dir=Path(__file__).parent / "training_logs" / "llm_reasoning" / "only",
            )

            reasoning_plus_human_train = pd.concat([human_train, reasoning_train], axis=1)
            reasoning_plus_human_test = pd.concat([human_test, reasoning_test], axis=1)
            reasoning_plus_human_names = base_feature_names + custom_features + reasoning_feature_names
            _train_and_log(
                reasoning_plus_human_names,
                reasoning_plus_human_train,
                reasoning_plus_human_test,
                y_train,
                y_test,
                args,
                input_csv,
                "LLM Reasoning + Human",
                log_dir=Path(__file__).parent / "training_logs" / "llm_reasoning" / "plus_human",
            )
            if llm_engineered_feature_names:
                _train_and_log(
                    llm_engineered_feature_names,
                    llm_train,
                    llm_test,
                    y_train,
                    y_test,
                    args,
                    input_csv,
                    "LLM Engineered Only",
                    log_dir=Path(__file__).parent / "training_logs" / "llm_engineered" / "only",
                )
                llm_plus_reasoning_train = pd.concat([llm_train, reasoning_train], axis=1)
                llm_plus_reasoning_test = pd.concat([llm_test, reasoning_test], axis=1)
                llm_plus_reasoning_names = llm_engineered_feature_names + reasoning_feature_names
                _train_and_log(
                    llm_plus_reasoning_names,
                    llm_plus_reasoning_train,
                    llm_plus_reasoning_test,
                    y_train,
                    y_test,
                    args,
                    input_csv,
                    "LLM Engineered + Reasoning",
                    log_dir=Path(__file__).parent / "training_logs" / "llm_engineered" / "plus_reasoning",
                )

            # F0.5 summary table (latest logs)
            report_path = Path(__file__).parent / "docs" / "llm_regression_report.md"
            rows: list[dict[str, float]] = []
            log_paths = [
                (human_only_log_dir / "training_log_*.txt", "Human Only"),
                (Path(__file__).parent / "training_logs" / "llm_reasoning" / "only" / "training_log_*.txt", "LLM Reasoning Only"),
                (Path(__file__).parent / "training_logs" / "llm_reasoning" / "plus_human" / "training_log_*.txt", "LLM Reasoning + Human"),
                (Path(__file__).parent / "training_logs" / "llm_engineered" / "only" / "training_log_*.txt", "LLM Engineered Only"),
                (Path(__file__).parent / "training_logs" / "llm_engineered" / "plus_reasoning" / "training_log_*.txt", "LLM Engineered + Reasoning"),
            ]
            for pattern, name in log_paths:
                candidates = sorted(pattern.parent.glob(pattern.name))
                if not candidates:
                    continue
                metrics = _parse_training_log(candidates[-1])
                if not metrics:
                    continue
                rows.append(
                    {
                        "name": name,
                        "F0.5": metrics.get("F0.5", float("nan")),
                        "ROC-AUC": metrics.get("ROC-AUC", float("nan")),
                        "PR-AUC": metrics.get("PR-AUC", float("nan")),
                        "Prec": metrics.get("Prec", float("nan")),
                        "Rec": metrics.get("Rec", float("nan")),
                        "Acc": metrics.get("Acc", float("nan")),
                    }
                )
            if rows:
                table = _write_f05_table(rows, report_path)
                _log("\nF0.5 summary:\n" + table)

        # Run-family mode (multiple engineered feature sets + leaderboard)
        if llm_engineered_run_family:
            if not reasoning_feature_names:
                raise RuntimeError("Run-family mode requires reasoning features to be loaded.")
            families_dir = cache_dir / "families"
            archives_dir = cache_dir / "archives"
            family_dir: Path | None = None
            family_id = llm_engineered_run_family_id or ""
            if llm_engineered_freeze and not family_id:
                if families_dir.exists():
                    consolidated = sorted(
                        families_dir.glob("family_*_features.parquet"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    if consolidated:
                        latest = consolidated[0]
                        family_id = latest.stem.replace("family_", "").replace("_features", "")
                        _log(f"  [Run-family] Reusing consolidated family: family_{family_id}")
                if not family_id:
                    existing = sorted(
                        [p for p in cache_dir.glob("family_*") if p.is_dir()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for candidate in existing:
                        set_dirs = list(candidate.glob("set_*"))
                        if not set_dirs:
                            continue
                        has_cache = any(
                            (sd / "llm_features.parquet").exists()
                            and (sd / "llm_features_meta.json").exists()
                            for sd in set_dirs
                        )
                        if has_cache:
                            family_dir = candidate
                            family_id = candidate.name.replace("family_", "")
                            _log(f"  [Run-family] Reusing existing family cache: {candidate.name}")
                            break
            if family_dir is None:
                if not family_id:
                    family_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                family_dir = cache_dir / f"family_{family_id}"
                family_dir.mkdir(parents=True, exist_ok=True)
            family_features_path = families_dir / f"family_{family_id}_features.parquet"
            family_meta_path = families_dir / f"family_{family_id}_meta.json"
            run_n = llm_engineered_run_family_n or llm_n
            rows: list[dict[str, Any]] = []
            skipped_sets: list[str] = []
            reasoning_label = "+".join(cfg_llm_reasoning_experiments or [])
            founder_ids = [r.get("founder_uuid") for r in records]
            family_frames: list[pd.DataFrame] = []
            family_set_features: dict[str, list[str]] = {}
            _log_run(
                f"Run-family start: id={family_id} n_sets={llm_engineered_run_family_size} n_features={run_n}"
            )
            def _evaluate_from_consolidated(df: pd.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
                set_features: dict[str, list[str]] = meta.get("set_features", {})
                out_rows: list[dict[str, Any]] = []
                for set_id, features in set_features.items():
                    df_set = df[df["set_id"] == set_id].set_index("row_index")
                    set_train = df_set.loc[train_idx]
                    set_test = df_set.loc[test_idx]
                    metrics_only, _ = _train_and_log(
                        features,
                        set_train[features],
                        set_test[features],
                        y_train,
                        y_test,
                        args,
                        input_csv,
                        "LLM Engineered Only",
                        log_dir=Path(__file__).parent / "training_logs" / "llm_engineered" / f"family_{family_id}" / set_id / "only",
                    )
                    out_rows.append(
                        {
                            "set_id": set_id,
                            "regression": "LLM Engineered Only",
                            "F0.5": metrics_only.get("f0.5", float("nan")),
                            "ROC-AUC": metrics_only.get("roc_auc", float("nan")),
                            "PR-AUC": metrics_only.get("pr_auc", float("nan")),
                            "Prec": metrics_only.get("precision", float("nan")),
                            "Rec": metrics_only.get("recall", float("nan")),
                            "Acc": metrics_only.get("accuracy", float("nan")),
                            "reasoning_experiment": reasoning_label,
                        }
                    )
                    set_plus_train = pd.concat([set_train[features], reasoning_train], axis=1)
                    set_plus_test = pd.concat([set_test[features], reasoning_test], axis=1)
                    metrics_plus, _ = _train_and_log(
                        features + reasoning_feature_names,
                        set_plus_train,
                        set_plus_test,
                        y_train,
                        y_test,
                        args,
                        input_csv,
                        "LLM Engineered + Reasoning",
                        log_dir=Path(__file__).parent / "training_logs" / "llm_engineered" / f"family_{family_id}" / set_id / "plus_reasoning",
                    )
                    out_rows.append(
                        {
                            "set_id": set_id,
                            "regression": "LLM Engineered + Reasoning",
                            "F0.5": metrics_plus.get("f0.5", float("nan")),
                            "ROC-AUC": metrics_plus.get("roc_auc", float("nan")),
                            "PR-AUC": metrics_plus.get("pr_auc", float("nan")),
                            "Prec": metrics_plus.get("precision", float("nan")),
                            "Rec": metrics_plus.get("recall", float("nan")),
                            "Acc": metrics_plus.get("accuracy", float("nan")),
                            "reasoning_experiment": reasoning_label,
                        }
                    )
                return out_rows

            if llm_engineered_freeze and family_features_path.exists() and family_meta_path.exists():
                df = pd.read_parquet(family_features_path)
                meta = json.loads(family_meta_path.read_text(encoding="utf-8"))
                rows = _evaluate_from_consolidated(df, meta)
                report_path = Path(__file__).parent / "docs" / "llm_regression_report.md"
                leaderboard_csv = Path(__file__).parent / "docs" / "llm_engineered_family_leaderboard.csv"
                _write_family_leaderboard(rows, report_path, leaderboard_csv)
                _log(f"\nRun-family leaderboard appended to: {report_path}")
                return
            for idx in range(1, llm_engineered_run_family_size + 1):
                set_id = f"set_{idx:02d}"
                set_dir = family_dir / set_id
                set_dir.mkdir(parents=True, exist_ok=True)
                _log_run(f"Run-family processing {set_id}")
                set_all = None
                set_names = None
                if llm_engineered_cache:
                    set_all, set_names = _load_llm_engineered_cache(
                        cache_dir=set_dir,
                        expected_rows=len(records),
                        expected_n=run_n,
                        model=args.llm_model,
                        providers=llm_providers,
                        google_model=llm_google_model,
                    )
                if set_all is None or set_names is None:
                    if llm_engineered_freeze:
                        _log(
                            f"\n  [Run-family] Cache missing for {set_id}; "
                            "freeze enabled so skipping generation."
                        )
                        skipped_sets.append(set_id)
                        _log_run(f"Run-family skipped {set_id} (cache missing)")
                        continue
                    _log(f"\n  [Run-family] Generating set {set_id} ({run_n} features)...")
                    attempt = 0
                    max_attempts = max(1, int(args.llm_retry_attempts) + 1)
                    while attempt < max_attempts:
                        attempt += 1
                        try:
                            async def _run_generate():
                                return await generate_llm_features(
                                    train_recs=train_recs,
                                    y_train=y_train,
                                    test_recs=test_recs,
                                    model=args.llm_model,
                                    n_features=run_n,
                                    all_recs=records,
                                    providers=llm_providers,
                                    google_model=llm_google_model,
                                )

                            if float(args.llm_timeout) > 0:
                                set_all, set_train, set_test, set_names = asyncio.run(
                                    asyncio.wait_for(
                                        _run_generate(), timeout=float(args.llm_timeout)
                                    )
                                )
                            else:
                                set_all, set_train, set_test, set_names = asyncio.run(
                                    _run_generate()
                                )
                            break
                        except Exception:
                            err = traceback.format_exc()
                            _log(f"\n  [Run-family] ERROR during LLM generation for {set_id} (attempt {attempt}):")
                            _log(err)
                            _log_run(
                                f"[Run-family] ERROR {set_id} attempt {attempt}: "
                                f"{err.splitlines()[-1] if err else 'unknown error'}"
                            )
                            if attempt >= max_attempts:
                                _log(
                                    f"  [Run-family] Exceeded retry attempts for {set_id}. Skipping."
                                )
                                _log_run(
                                    f"Run-family skipped {set_id} (generation failed after {attempt} attempts)"
                                )
                                skipped_sets.append(set_id)
                                set_all = None
                                set_names = None
                                break
                            _log(f"  [Run-family] Retrying in {args.llm_retry_sleep:.1f}s...")
                            time.sleep(max(0.0, float(args.llm_retry_sleep)))
                    if (
                        llm_engineered_cache
                        and set_all is not None
                        and set_names is not None
                    ):
                        _save_llm_engineered_cache(
                            cache_dir=set_dir,
                            df=set_all,
                            feature_names=set_names,
                            model=args.llm_model,
                            providers=llm_providers,
                            google_model=llm_google_model,
                            n_features=run_n,
                        )
                else:
                    set_train = set_all.iloc[train_idx].reset_index(drop=True)
                    set_test = set_all.iloc[test_idx].reset_index(drop=True)

                if set_all is None or set_names is None:
                    continue

                if set_all is not None and set_names is not None:
                    set_df = set_all.copy()
                    set_df.insert(0, "row_index", np.arange(len(records)))
                    set_df.insert(1, "founder_uuid", founder_ids)
                    set_df.insert(2, "success", labels)
                    set_df.insert(3, "set_id", set_id)
                    family_frames.append(set_df)
                    family_set_features[set_id] = set_names

                set_only_log = Path(__file__).parent / "training_logs" / "llm_engineered" / f"family_{family_id}" / set_id / "only"
                set_plus_log = Path(__file__).parent / "training_logs" / "llm_engineered" / f"family_{family_id}" / set_id / "plus_reasoning"

                metrics_only, _ = _train_and_log(
                    set_names,
                    set_train,
                    set_test,
                    y_train,
                    y_test,
                    args,
                    input_csv,
                    "LLM Engineered Only",
                    log_dir=set_only_log,
                )
                rows.append(
                    {
                        "set_id": set_id,
                        "regression": "LLM Engineered Only",
                        "F0.5": metrics_only.get("f0.5", float("nan")),
                        "ROC-AUC": metrics_only.get("roc_auc", float("nan")),
                        "PR-AUC": metrics_only.get("pr_auc", float("nan")),
                        "Prec": metrics_only.get("precision", float("nan")),
                        "Rec": metrics_only.get("recall", float("nan")),
                        "Acc": metrics_only.get("accuracy", float("nan")),
                        "reasoning_experiment": reasoning_label,
                    }
                )

                set_plus_train = pd.concat([set_train, reasoning_train], axis=1)
                set_plus_test = pd.concat([set_test, reasoning_test], axis=1)
                metrics_plus, _ = _train_and_log(
                    set_names + reasoning_feature_names,
                    set_plus_train,
                    set_plus_test,
                    y_train,
                    y_test,
                    args,
                    input_csv,
                    "LLM Engineered + Reasoning",
                    log_dir=set_plus_log,
                )
                rows.append(
                    {
                        "set_id": set_id,
                        "regression": "LLM Engineered + Reasoning",
                        "F0.5": metrics_plus.get("f0.5", float("nan")),
                        "ROC-AUC": metrics_plus.get("roc_auc", float("nan")),
                        "PR-AUC": metrics_plus.get("pr_auc", float("nan")),
                        "Prec": metrics_plus.get("precision", float("nan")),
                        "Rec": metrics_plus.get("recall", float("nan")),
                        "Acc": metrics_plus.get("accuracy", float("nan")),
                        "reasoning_experiment": reasoning_label,
                    }
                )
                _log_run(f"Run-family completed {set_id}")

            if family_frames:
                families_dir.mkdir(parents=True, exist_ok=True)
                combined = pd.concat(family_frames, ignore_index=True, sort=False)
                combined.to_parquet(family_features_path, index=False)
                meta = {
                    "family_id": family_id,
                    "model": args.llm_model,
                    "n_features": run_n,
                    "providers": llm_providers,
                    "google_model": llm_google_model,
                    "set_features": family_set_features,
                    "timestamp": datetime.now().isoformat(),
                }
                family_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                # Archive per-set caches to reduce clutter
                if family_dir.exists():
                    archives_dir.mkdir(parents=True, exist_ok=True)
                    archive_target = archives_dir / f"family_{family_id}"
                    if archive_target.exists():
                        shutil.rmtree(archive_target)
                    shutil.move(str(family_dir), str(archive_target))

            report_path = Path(__file__).parent / "docs" / "llm_regression_report.md"
            leaderboard_csv = Path(__file__).parent / "docs" / "llm_engineered_family_leaderboard.csv"
            _write_family_leaderboard(rows, report_path, leaderboard_csv)
            meta_path = Path(__file__).parent / "docs" / "llm_engineered_family_leaderboard_meta.json"
            meta = {
                "family_id": family_id,
                "n_sets": llm_engineered_run_family_size,
                "n_features": run_n,
                "reasoning_experiment": reasoning_label,
                "model": args.llm_model,
                "timestamp": datetime.now().isoformat(),
                "skipped_sets": skipped_sets,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            _log(f"\nRun-family leaderboard appended to: {report_path}")
            return

    # Placeholder for future multiple training loops over feature subsets.
    # TODO: add loop over named feature sets and aggregate metrics.

    X_train = full_train.values.astype(float)
    X_test = full_test.values.astype(float)

    train_scores, test_scores, model = _train_sklearn(
        X_train, y_train, X_test, rs
    )

    metrics = _report_metrics(y_train, train_scores, y_test, test_scores)

    # Output format
    _log(f"\n  Features used: {', '.join(feature_names)}")
    _log(
        f"\n  [{mode_label}]   {len(feature_names)} features, threshold={metrics['threshold']:.2f}"
    )
    coef = model.coef_[0]
    ranked = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)
    for name, c in ranked:
        sign = "+" if c >= 0 else "-"
        _log(f"    {sign}{abs(c):.3f}  {name}")

    acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
    _log(
        f"\n  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
        f"F0.5={metrics['f0.5']:.3f}  Acc={acc:.3f}  "
        f"FNR={metrics['fnr']:.3f}  TP={int(metrics['tp'])}  FN={int(metrics['fn'])}"
    )
    for line in _format_topk(metrics):
        _log(line)

    metrics["accuracy"] = acc
    _write_log(
        log_lines,
        args,
        input_csv,
        feature_names,
        len(train_idx),
        len(test_idx),
        int(y_train.sum()),
        int(y_test.sum()),
        metrics,
        log_dir=_log_dir_for_mode(mode, llm_reasoning_dry_run, llm_reasoning_dry_run_fast),
    )


def _write_log(
    lines: list[str],
    args: argparse.Namespace,
    input_csv: str,
    feature_names: list[str],
    n_train: int,
    n_test: int,
    pos_train: int,
    pos_test: int,
    metrics: dict[str, float],
    log_dir: Path | None = None,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is None:
        log_dir = Path(__file__).parent / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"training_log_{ts}.txt"
    log_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        print(f"\n  Training log saved to: {log_path}")
    except OSError:
        pass

    report = {
        "dataset": input_csv,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "features": feature_names,
        "split": {"train": n_train, "test": n_test},
        "positives": {"train": pos_train, "test": pos_test},
        "threshold": metrics.get("threshold"),
        "metrics": {
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f0.5": metrics.get("f0.5"),
            "accuracy": metrics.get("accuracy"),
            "precision@1%": metrics.get("precision@1%"),
            "precision@5%": metrics.get("precision@5%"),
            "precision@10%": metrics.get("precision@10%"),
            "fnr": metrics.get("fnr"),
            "tp": metrics.get("tp"),
            "fn": metrics.get("fn"),
        },
    }
    report_path = log_dir / f"run_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        print(f"  Run report saved to: {report_path}")
    except OSError:
        pass


def _log_dir_for_mode(
    mode: str,
    dry_run: bool,
    dry_run_fast: bool,
) -> Path:
    root = Path(__file__).parent / "training_logs"
    if dry_run or dry_run_fast:
        return root / "dry_runs" / mode
    if mode == "human":
        return root / "human"
    if mode == "llm":
        return root / "llm_engineered"
    if mode == "reasoning":
        return root / "llm_reasoning"
    if mode == "hybrid":
        return root / "hybrid"
    return root


if __name__ == "__main__":
    main()
