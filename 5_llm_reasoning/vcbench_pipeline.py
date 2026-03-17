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
from pathlib import Path
from datetime import datetime
import traceback
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
from llm_reasoning_features import ReasoningConfig, generate_reasoning_features


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
        "--llm_reasoning_prompts",
        default=str(Path(__file__).parent / "llm_reasoning_prompts.json"),
        help="Path to JSON prompts for LLM reasoning features.",
    )
    p.add_argument(
        "--llm_reasoning_dataset_size",
        default="full",
        help="Dataset size for reasoning features: full, 200, 400, 1000.",
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

    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        rs,
    )

    # Resolve LLM reasoning settings early to allow dataset sizing
    reasoning_prompts_path = (
        Path(cfg_llm_reasoning_prompts_path)
        if cfg_llm_reasoning_prompts_path
        else Path(args.llm_reasoning_prompts)
    )
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

    base_script = Path(args.base_script)
    extract_base = _load_base_feature_extractor(base_script)
    base_all = pd.DataFrame([extract_base(r) for r in records])
    base_train = pd.DataFrame([extract_base(r) for r in train_recs])
    base_test = pd.DataFrame([extract_base(r) for r in test_recs])
    base_feature_names = list(base_train.columns)

    selected_features: list[str] = []
    cfg_use_llm: bool | None = None
    cfg_llm_n: int | None = None
    cfg_use_llm_reasoning: bool | None = None
    cfg_llm_reasoning_features: list[str] | None = None
    cfg_llm_reasoning_dataset_size: str | None = None
    cfg_llm_reasoning_prompts_path: str | None = None
    cfg_path = Path(args.feature_config) if args.feature_config else None
    if cfg_path is not None and cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
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
        import asyncio
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
                )
            )

    if use_llm_reasoning:
        reasoning_features = (
            cfg_llm_reasoning_features
            if cfg_llm_reasoning_features is not None
            else []
        )
        prompts_path = reasoning_prompts_path
        output_dir = Path(__file__).parent / "features_storage" / "llm_reasoning"
        meta_path = output_dir / f"llm_reasoning_{reasoning_dataset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config = ReasoningConfig(
            model=args.llm_model,
            dataset_size=reasoning_dataset_size,
            random_state=rs,
            prompts_path=prompts_path,
        )
        reasoning_df, all_reasoning_names = generate_reasoning_features(
            records=records,
            labels=labels,
            config=config,
            output_dir=output_dir,
            metadata_path=meta_path,
        )
        if reasoning_features:
            reasoning_feature_names = [n for n in all_reasoning_names if n in reasoning_features]
        else:
            reasoning_feature_names = all_reasoning_names
        reasoning_all = reasoning_df.drop(columns=[c for c in ["founder_uuid", "success"] if c in reasoning_df.columns])
        reasoning_train = reasoning_all.iloc[train_idx].reset_index(drop=True)
        reasoning_test = reasoning_all.iloc[test_idx].reset_index(drop=True)


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
    if selected_features and mode != "llm":
        expected_set = {f for f in selected_features if f in feature_names}
        actual_set = set(feature_names)
        if expected_set != actual_set:
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            raise RuntimeError(
                "Selected feature list mismatch. "
                f"Missing: {missing} Extra: {extra}"
            )

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
            out_path = features_storage / "llm_reasoning" / "llm_reasoning_features.parquet"
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
        )
        return

    # Optional reasoning-only and reasoning+human runs for consistent comparison.
    if use_llm_reasoning and reasoning_feature_names:
        _train_and_log(
            reasoning_feature_names,
            reasoning_train,
            reasoning_test,
            y_train,
            y_test,
            args,
            input_csv,
            "LLM Reasoning Only",
            log_dir=Path(__file__).parent / "training_logs" / "llm_reasoning_only",
        )

        human_train = pd.concat([base_train, custom_train], axis=1)
        human_test = pd.concat([base_test, custom_test], axis=1)
        human_train, human_test = _standardize_continuous(
            human_train, human_test, custom_features
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
            log_dir=Path(__file__).parent / "training_logs" / "llm_reasoning_plus_human",
        )

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
    print(f"\n  Training log saved to: {log_path}")

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
    print(f"  Run report saved to: {report_path}")


if __name__ == "__main__":
    main()
