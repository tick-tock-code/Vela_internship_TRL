"""Train-only GEPA-style prompt evolution for merged Experiment A+B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice
from think_reason_learn.core.llms import llm as trl_llm

from llm_reasoning_features import ReasoningConfig, generate_reasoning_features, _assert_no_label_fields
from vcbench_pipeline import _report_metrics, _train_sklearn
from cv_folds import build_splits_from_fold_ids, load_or_create_folds, resolve_folds_path


DEFAULT_CONFIG_PATH = Path(__file__).parent / "prompt_evolution_config.json"
DEFAULT_CORE_PROMPT = Path(__file__).parent / "core_prompt_evolve.txt"
DEFAULT_EXPERIMENTS = Path(__file__).parent / "experiments_evolve.json"

FORBIDDEN_MUTATION_MARKERS = [
    "output keys",
    "output format",
    "json",
    "{",
    "}",
    "schema",
    "{{",
    "}}",
    "evidence_support_rating",
    "trajectory_strength",
    "ownership_signal",
    "career_coherence",
    "scrappiness",
    "rubric_score",
    "justification",
]

REQUIRED_COLUMNS = {
    "founder_uuid",
    "industry",
    "educations_json",
    "jobs_json",
    "success",
}

DEFAULTS: dict[str, Any] = {
    "experiment": "AB",
    "pool_size": 10,
    "sample_size": 200,
    "batch_size": 20,
    "concurrency": 10,
    "iterations": 10,
    "save_every": 2,
    "selection_metric": "f0.5",
    "random_state": 42,
    "dataset": "full",
    "input_csv": "",
    "test_size": 0.20,
    "llm_model": "gpt-4.1-nano",
    "llm_google_model": "gemini-2.0-flash",
    "critic_model": "gpt-4.1-nano",
    "critic_provider": "openai",
    "critic_temperature": 0.2,
    "core_prompt_path": str(DEFAULT_CORE_PROMPT),
    "experiments_path": str(DEFAULT_EXPERIMENTS),
    "output_root": "prompt_evolution",
    "dry_run": False,
    "initial_mutations": 9,
    "critic_sample_size": 20,
    "cv_folds": 4,
    "cv_use_fixed_folds": True,
    "cv_folds_path": "",
}


def _safe_json_parse(value: Any) -> list[dict]:
    if pd.isna(value) or value == "":
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else [result]
    except Exception:
        try:
            import ast
            result = ast.literal_eval(str(value))
            return result if isinstance(result, list) else [result]
        except Exception:
            return []


def _load_vcbench_local(
    csv_path: str,
    label_column: str = "success",
) -> tuple[list[dict[str, Any]], np.ndarray]:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if label_column != "success":
        missing.discard("success")
        if label_column not in df.columns:
            missing.add(label_column)
    if missing:
        raise RuntimeError(
            "Missing required columns in CSV: "
            f"{sorted(missing)}. Available: {sorted(df.columns)}"
        )
    df = df.dropna(subset=[label_column]).copy()
    df[label_column] = df[label_column].astype(int)

    records: list[dict[str, Any]] = []
    has_prose = "anonymised_prose" in df.columns
    for _, row in df.iterrows():
        rec: dict[str, Any] = {
            "founder_uuid": row.get("founder_uuid", "") or "",
            "industry": row.get("industry", "") or "",
            "educations": _safe_json_parse(row.get("educations_json", "")),
            "jobs": _safe_json_parse(row.get("jobs_json", "")),
            "ipos": _safe_json_parse(row.get("ipos", "")),
            "acquisitions": _safe_json_parse(row.get("acquisitions", "")),
        }
        if has_prose:
            rec["anonymised_prose"] = row.get("anonymised_prose", "") or ""
        records.append(rec)
    labels: np.ndarray = df[label_column].values  # type: ignore[assignment]
    return records, labels


@dataclass(frozen=True)
class PromptVariant:
    prompt_id: str
    instructions: str
    experiments_path: Path
    parent_id: str | None = None
    created_iter: int | None = None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        global trl_llm
        trl_llm = trl_llms.llm
    except Exception:
        return


def _resolve_input_csv(dataset: str, input_csv: str | None) -> str:
    if input_csv:
        return input_csv
    base = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit"
    if dataset == "sample":
        return str(base / "vcbench_final_public_sample100.csv")
    return str(base / "vcbench_final_public.csv")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_mutation(text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return False, "Mutation is empty."
    lowered = stripped.lower()
    for marker in FORBIDDEN_MUTATION_MARKERS:
        if marker in lowered:
            return False, f"Forbidden marker detected: {marker}"
    if len(stripped) < 40:
        return False, "Mutation too short."
    return True, ""


def _load_base_experiment(experiments_path: Path, experiment_id: str) -> dict[str, Any]:
    data = _read_json(experiments_path)
    if not isinstance(data, list):
        raise ValueError("Experiments file must contain a list.")
    for exp in data:
        if str(exp.get("id")) == experiment_id:
            return exp
    raise ValueError(f"Experiment {experiment_id} not found in {experiments_path}.")


def _persist_prompt_variant(
    root: Path,
    base_experiment: dict[str, Any],
    instructions: str,
    parent_id: str | None,
    created_iter: int | None,
) -> PromptVariant:
    prompt_hash = _hash_text(instructions)
    prompt_id = prompt_hash[:12]
    prompt_dir = root / "prompts" / prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "instructions.txt").write_text(instructions, encoding="utf-8")
    _write_json(prompt_dir / "hash.json", {"hash": prompt_hash})
    _write_json(
        prompt_dir / "metadata.json",
        {
            "prompt_id": prompt_id,
            "hash": prompt_hash,
            "parent_id": parent_id,
            "created_iter": created_iter,
            "created_at": datetime.now().isoformat(),
        },
    )
    exp_payload = [
        {
            "id": base_experiment["id"],
            "instructions": instructions,
            "numeric_keys": list(base_experiment.get("numeric_keys", [])),
            "text_keys": list(base_experiment.get("text_keys", [])),
        }
    ]
    exp_path = prompt_dir / "experiments.json"
    _write_json(exp_path, exp_payload)
    return PromptVariant(
        prompt_id=prompt_id,
        instructions=instructions,
        experiments_path=exp_path,
        parent_id=parent_id,
        created_iter=created_iter,
    )


def _build_critic_payload(
    df: pd.DataFrame,
    numeric_keys: list[str],
    sample_size: int,
    rng: np.random.RandomState,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    numeric_keys = [k for k in numeric_keys if k in df.columns]
    text_keys = [
        c
        for c in df.columns
        if c not in numeric_keys and c not in {"founder_uuid", "success"}
    ]
    total = len(df)
    idx = np.arange(total)
    sample_n = min(sample_size, total)
    sample_idx = rng.choice(idx, size=sample_n, replace=False) if total else np.array([])
    outputs: list[dict[str, Any]] = []
    for i in sample_idx:
        row = df.iloc[int(i)]
        item: dict[str, Any] = {}
        for key in numeric_keys:
            item[key] = float(row[key]) if pd.notna(row[key]) else None
        for key in text_keys:
            val = row[key]
            item[key] = "" if pd.isna(val) else str(val)
        _assert_no_label_fields(item, "critic output item")
        outputs.append(item)

    summary: dict[str, Any] = {"total_rows": total}
    if numeric_keys:
        summary["numeric"] = {}
        for key in numeric_keys:
            series = pd.to_numeric(df[key], errors="coerce")
            summary["numeric"][key] = {
                "mean": float(series.mean()) if len(series) else None,
                "std": float(series.std()) if len(series) else None,
                "min": float(series.min()) if len(series) else None,
                "max": float(series.max()) if len(series) else None,
                "nan_count": int(series.isna().sum()),
            }
    if text_keys:
        summary["text"] = {}
        for key in text_keys:
            series = df[key].fillna("").astype(str)
            summary["text"][key] = {
                "empty_count": int((series.str.len() == 0).sum()),
            }
    return outputs, summary


def _critic_prompt(
    current_instructions: str,
    sample_outputs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    return (
        "You are a prompt critic improving Experiment AB instructions.\n"
        "You MUST NOT change output keys, schema, or formatting rules. "
        "Do NOT mention JSON, output keys, or formatting in your response.\n"
        "Return ONLY the new instruction text, no quotes, no markdown.\n\n"
        "Current instructions:\n"
        f"{current_instructions}\n\n"
        "Summary statistics:\n"
        f"{json.dumps(summary, indent=2)}\n\n"
        "Sample outputs (scores + justifications only):\n"
        f"{json.dumps(sample_outputs, indent=2)}\n\n"
        "Provide improved instructions that reduce ambiguity and improve consistency."
    )


def _mutate_with_critic(
    instructions: str,
    model: str,
    provider: str,
    temperature: float,
    sample_outputs: list[dict[str, Any]],
    summary: dict[str, Any],
    dry_run: bool,
    max_attempts: int = 3,
) -> str:
    if dry_run:
        return instructions + "\n\nMake the rubric more concise and evidence-based."

    prompt = _critic_prompt(instructions, sample_outputs, summary)
    if provider == "google":
        choice = GoogleChoice(model=model)
    else:
        choice = OpenAIChoice(model=model)
    last_text = ""
    for attempt in range(max_attempts):
        resp = trl_llm.respond_sync(
            llm_priority=[choice],
            query=prompt,
            response_format=str,
            temperature=temperature,
        )
        last_text = str(resp.response).strip()
        if last_text.startswith("```"):
            last_text = last_text.strip("`").strip()
        ok, _ = _validate_mutation(last_text)
        if ok:
            return last_text
        prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY the revised instruction text. "
            "Do NOT mention JSON, output keys, or formatting."
        )
    return last_text if last_text else instructions


def _cv_metrics_from_fold_ids(
    X: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    n_folds: int,
    random_state: int,
) -> dict[str, float]:
    splits = build_splits_from_fold_ids(fold_ids, n_folds)
    metrics_list: list[dict[str, float]] = []
    for train_idx, test_idx in splits:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        train_scores, test_scores, _ = _train_sklearn(
            X[train_idx], y_train, X[test_idx], random_state
        )
        metrics = _report_metrics(y_train, train_scores, y_test, test_scores)
        acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
        metrics["accuracy"] = acc
        metrics_list.append(metrics)

    if not metrics_list:
        return {
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
            "accuracy": 0.0,
            "precision@1%": 0.0,
            "precision@5%": 0.0,
            "precision@10%": 0.0,
            "cv_folds_used": 0.0,
            "note": "single_class_folds",
        }

    keys = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f0.5",
        "accuracy",
        "precision@1%",
        "precision@5%",
        "precision@10%",
    ]
    means = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in keys}
    stds = {k: float(np.nanstd([m[k] for m in metrics_list])) for k in keys}
    metrics: dict[str, float] = dict(means)
    for key in keys:
        metrics[f"{key}_std"] = stds[key]
    metrics["cv_folds_used"] = float(len(metrics_list))
    return metrics


def _cv_metrics_from_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    random_state: int,
) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
            "accuracy": 0.0,
            "precision@1%": 0.0,
            "precision@5%": 0.0,
            "precision@10%": 0.0,
            "cv_folds_used": 0.0,
            "note": "single_class_sample",
        }
    counts = np.bincount(y.astype(int))
    min_count = int(counts.min()) if counts.size else 0
    use_folds = min(n_folds, min_count) if min_count >= 2 else 1
    if use_folds < 2:
        return {
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
            "accuracy": 0.0,
            "precision@1%": 0.0,
            "precision@5%": 0.0,
            "precision@10%": 0.0,
            "cv_folds_used": 0.0,
            "note": "insufficient_class_counts",
        }
    skf = StratifiedKFold(n_splits=use_folds, shuffle=True, random_state=random_state)
    metrics_list: list[dict[str, float]] = []
    for train_idx, test_idx in skf.split(X, y):
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        train_scores, test_scores, _ = _train_sklearn(
            X[train_idx], y_train, X[test_idx], random_state
        )
        metrics = _report_metrics(y_train, train_scores, y_test, test_scores)
        acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
        metrics["accuracy"] = acc
        metrics_list.append(metrics)
    if not metrics_list:
        return {
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
            "accuracy": 0.0,
            "precision@1%": 0.0,
            "precision@5%": 0.0,
            "precision@10%": 0.0,
            "cv_folds_used": 0.0,
            "note": "single_class_folds",
        }
    keys = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f0.5",
        "accuracy",
        "precision@1%",
        "precision@5%",
        "precision@10%",
    ]
    means = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in keys}
    stds = {k: float(np.nanstd([m[k] for m in metrics_list])) for k in keys}
    metrics: dict[str, float] = dict(means)
    for key in keys:
        metrics[f"{key}_std"] = stds[key]
    metrics["cv_folds_used"] = float(len(metrics_list))
    if use_folds != n_folds:
        metrics["note"] = f"cv_folds_reduced_to_{use_folds}"
    return metrics


def _evaluate_prompt(
    prompt: PromptVariant,
    records: list[dict[str, Any]],
    labels: np.ndarray,
    cv_folds: int,
    output_root: Path,
    iter_idx: int,
    args: argparse.Namespace,
    llm_providers: dict[str, bool],
    llm_google_model: str | None,
    rng: np.random.RandomState,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    iter_label = f"iter_{iter_idx:03d}"
    output_dir = output_root / "outputs" / prompt.prompt_id / iter_label
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "reasoning_meta.json"
    log_dir = output_dir / "logs"
    config = ReasoningConfig(
        model=args.llm_model,
        dataset_size=str(args.sample_size),
        random_state=args.random_state,
        core_prompt_path=Path(args.core_prompt_path),
        experiments_path=prompt.experiments_path,
        providers=llm_providers,
        google_model=llm_google_model,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        experiments=[args.experiment],
        dry_run=args.dry_run,
        dry_run_fast=False,
        log_dir=log_dir,
        log_every=1,
        repair_nan=False,
        repair_existing=False,
        skip_select=True,
    )
    df, numeric_keys = generate_reasoning_features(
        records=records,
        labels=labels,
        config=config,
        output_dir=output_dir,
        metadata_path=meta_path,
    )

    X = df[numeric_keys].values.astype(float)
    metrics = _cv_metrics_from_sample(
        X=X,
        y=labels,
        n_folds=cv_folds,
        random_state=args.random_state + iter_idx,
    )

    outputs, summary = _build_critic_payload(
        df=df,
        numeric_keys=numeric_keys,
        sample_size=args.critic_sample_size,
        rng=rng,
    )
    _write_json(output_dir / "outputs.json", outputs)
    _write_json(output_dir / "summary.json", summary)
    return metrics, outputs, summary


def _full_eval(
    prompt: PromptVariant,
    records: list[dict[str, Any]],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    cv_folds: int,
    output_root: Path,
    iter_idx: int,
    args: argparse.Namespace,
    llm_providers: dict[str, bool],
    llm_google_model: str | None,
) -> dict[str, float]:
    iter_label = f"iter_{iter_idx:03d}"
    output_dir = output_root / "full_eval" / iter_label
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "reasoning_meta.json"
    log_dir = output_dir / "logs"
    config = ReasoningConfig(
        model=args.llm_model,
        dataset_size="full",
        random_state=args.random_state,
        core_prompt_path=Path(args.core_prompt_path),
        experiments_path=prompt.experiments_path,
        providers=llm_providers,
        google_model=llm_google_model,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        experiments=[args.experiment],
        dry_run=args.dry_run,
        dry_run_fast=False,
        log_dir=log_dir,
        log_every=1,
        repair_nan=False,
        repair_existing=False,
        skip_select=True,
    )
    df, numeric_keys = generate_reasoning_features(
        records=records,
        labels=labels,
        config=config,
        output_dir=output_dir,
        metadata_path=meta_path,
    )
    X = df[numeric_keys].values.astype(float)
    metrics = _cv_metrics_from_fold_ids(
        X=X,
        y=labels,
        fold_ids=fold_ids,
        n_folds=cv_folds,
        random_state=args.random_state,
    )
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "prompt.json", {"prompt_id": prompt.prompt_id})
    return metrics


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        return _read_json(cfg_path)
    return {}


def _apply_config(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    for key, value in cfg.items():
        if hasattr(args, key) and key in DEFAULTS:
            if getattr(args, key) == DEFAULTS[key]:
                setattr(args, key, value)
    return args


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-only prompt evolution for Experiment AB.")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--experiment", type=str, default=DEFAULTS["experiment"])
    p.add_argument("--pool_size", type=int, default=DEFAULTS["pool_size"])
    p.add_argument("--sample_size", type=int, default=DEFAULTS["sample_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--concurrency", type=int, default=DEFAULTS["concurrency"])
    p.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    p.add_argument("--save_every", type=int, default=DEFAULTS["save_every"])
    p.add_argument("--selection_metric", type=str, default=DEFAULTS["selection_metric"])
    p.add_argument("--random_state", type=int, default=DEFAULTS["random_state"])
    p.add_argument("--dataset", type=str, default=DEFAULTS["dataset"], choices=["full", "sample"])
    p.add_argument("--input_csv", type=str, default=DEFAULTS["input_csv"])
    p.add_argument("--test_size", type=float, default=DEFAULTS["test_size"])
    p.add_argument("--llm_model", type=str, default=DEFAULTS["llm_model"])
    p.add_argument("--llm_google_model", type=str, default=DEFAULTS["llm_google_model"])
    p.add_argument("--critic_model", type=str, default=DEFAULTS["critic_model"])
    p.add_argument("--critic_provider", type=str, default=DEFAULTS["critic_provider"], choices=["openai", "google"])
    p.add_argument("--critic_temperature", type=float, default=DEFAULTS["critic_temperature"])
    p.add_argument("--core_prompt_path", type=str, default=DEFAULTS["core_prompt_path"])
    p.add_argument("--experiments_path", type=str, default=DEFAULTS["experiments_path"])
    p.add_argument("--output_root", type=str, default=DEFAULTS["output_root"])
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--initial_mutations", type=int, default=DEFAULTS["initial_mutations"])
    p.add_argument("--critic_sample_size", type=int, default=DEFAULTS["critic_sample_size"])
    p.add_argument("--cv_folds", type=int, default=DEFAULTS["cv_folds"])
    p.add_argument("--cv_use_fixed_folds", type=int, default=DEFAULTS["cv_use_fixed_folds"])
    p.add_argument("--cv_folds_path", type=str, default=DEFAULTS["cv_folds_path"])
    return p.parse_args()


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    cfg = _load_config(args)
    args = _apply_config(args, cfg)

    core_prompt_path = Path(args.core_prompt_path)
    if not core_prompt_path.is_absolute():
        core_prompt_path = Path(__file__).parent / core_prompt_path
    args.core_prompt_path = str(core_prompt_path)
    experiments_path = Path(args.experiments_path)
    if not experiments_path.is_absolute():
        experiments_path = Path(__file__).parent / experiments_path
    args.experiments_path = str(experiments_path)
    root = Path(args.output_root)
    root = root if root.is_absolute() else Path(__file__).parent / root
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "run_config.json", vars(args))

    input_csv = _resolve_input_csv(args.dataset, args.input_csv or None)
    records, labels = _load_vcbench_local(input_csv)
    founder_ids = [r.get("founder_uuid") for r in records]

    features_cfg = Path(__file__).parent / "features.json"
    llm_providers = {"openai": True, "google": False}
    llm_google_model = args.llm_google_model
    cv_folds = int(args.cv_folds)
    cv_use_fixed_folds = bool(args.cv_use_fixed_folds)
    cv_folds_path = args.cv_folds_path
    seed_size = 100
    if features_cfg.exists():
        data = _read_json(features_cfg)
        if isinstance(data.get("llm_providers"), dict):
            llm_providers = dict(data.get("llm_providers"))
        if data.get("llm_google_model"):
            llm_google_model = str(data.get("llm_google_model"))
        if "cv_use_fixed_folds" in data:
            cv_use_fixed_folds = bool(data.get("cv_use_fixed_folds"))
        if "cv_folds_path" in data:
            cv_folds_path = str(data.get("cv_folds_path") or "")
        if data.get("llm_engineered_seed_size") is not None:
            try:
                seed_size = int(data.get("llm_engineered_seed_size"))
            except Exception:
                seed_size = 100

    if isinstance(cfg.get("llm_providers"), dict):
        llm_providers = dict(cfg.get("llm_providers"))
    if cfg.get("llm_google_model"):
        llm_google_model = str(cfg.get("llm_google_model"))

    seed_path = Path(__file__).parent / "features_storage" / "llm_engineered" / f"seed_{seed_size}.json"
    if seed_path.exists():
        try:
            seed_payload = _read_json(seed_path)
            seed_uuids = set(seed_payload.get("uuids", []))
        except Exception:
            seed_uuids = set()
        if seed_uuids:
            keep_mask = [fid not in seed_uuids for fid in founder_ids]
            records = [r for r, keep in zip(records, keep_mask) if keep]
            labels = labels[np.array(keep_mask)]
            founder_ids = [fid for fid, keep in zip(founder_ids, keep_mask) if keep]

    base_experiment = _load_base_experiment(Path(args.experiments_path), args.experiment)
    base_instructions = str(base_experiment.get("instructions", "")).strip()
    if not base_instructions:
        raise RuntimeError("Base instructions are empty.")

    folds_path = resolve_folds_path(Path(__file__).parent, cv_folds, args.random_state, cv_folds_path)
    _, fold_ids, folds_path = load_or_create_folds(
        founder_ids=founder_ids,
        labels=labels,
        cv_folds=cv_folds,
        random_state=args.random_state,
        folds_path=folds_path,
        dataset_label=input_csv,
        use_fixed=cv_use_fixed_folds,
    )

    prompt_pool: list[PromptVariant] = []
    base_prompt = _persist_prompt_variant(
        root=root,
        base_experiment=base_experiment,
        instructions=base_instructions,
        parent_id=None,
        created_iter=None,
    )
    prompt_pool.append(base_prompt)

    initial_mutations = max(0, int(args.initial_mutations))
    for i in range(min(initial_mutations, args.pool_size - 1)):
        mutated = _mutate_with_critic(
            instructions=base_instructions,
            model=args.critic_model,
            provider=args.critic_provider,
            temperature=args.critic_temperature,
            sample_outputs=[],
            summary={"note": "initial mutation"},
            dry_run=args.dry_run,
        )
        ok, _ = _validate_mutation(mutated)
        if not ok:
            mutated = base_instructions + "\n\nEmphasize consistency and evidence."
        prompt_id = _hash_text(mutated)[:12]
        if any(p.prompt_id == prompt_id for p in prompt_pool):
            mutated = mutated + f"\n\nVariation init {i+1}."
        prompt_pool.append(
            _persist_prompt_variant(
                root=root,
                base_experiment=base_experiment,
                instructions=mutated,
                parent_id=base_prompt.prompt_id,
                created_iter=0,
            )
        )

    while len(prompt_pool) < args.pool_size:
        prompt_pool.append(base_prompt)

    for iter_idx in range(1, args.iterations + 1):
        iter_label = f"iter_{iter_idx:03d}"
        iter_root = run_root / iter_label
        iter_root.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(args.random_state + iter_idx)
        sample_n = min(args.sample_size, len(records))
        sample_indices = rng.choice(len(records), size=sample_n, replace=False)
        sample_records = [records[i] for i in sample_indices]
        sample_labels = labels[sample_indices]
        _write_json(iter_root / "sample_indices.json", [int(i) for i in sample_indices])

        metrics_by_prompt: dict[str, dict[str, float]] = {}
        outputs_by_prompt: dict[str, list[dict[str, Any]]] = {}
        summary_by_prompt: dict[str, dict[str, Any]] = {}

        for prompt in prompt_pool:
            metrics, outputs, summary = _evaluate_prompt(
                prompt=prompt,
                records=sample_records,
                labels=sample_labels,
                cv_folds=cv_folds,
                output_root=root,
                iter_idx=iter_idx,
                args=args,
                llm_providers=llm_providers,
                llm_google_model=llm_google_model,
                rng=rng,
            )
            metrics_by_prompt[prompt.prompt_id] = metrics
            outputs_by_prompt[prompt.prompt_id] = outputs
            summary_by_prompt[prompt.prompt_id] = summary

        _write_json(iter_root / "metrics.json", metrics_by_prompt)

        ranked = sorted(
            prompt_pool,
            key=lambda p: metrics_by_prompt[p.prompt_id].get(args.selection_metric, 0.0),
            reverse=True,
        )
        top_keep = ranked[: min(2, len(ranked))]
        selected_ids = [p.prompt_id for p in top_keep]
        _write_json(iter_root / "selected_ids.json", selected_ids)
        _write_json(iter_root / "pool.json", [p.prompt_id for p in prompt_pool])

        next_pool = list(top_keep)
        parent_cycle = ranked[: min(2, len(ranked))]
        if not parent_cycle:
            parent_cycle = [base_prompt]

        desired_new = args.pool_size - len(next_pool)
        for i in range(desired_new):
            parent = parent_cycle[i % len(parent_cycle)]
            outputs = outputs_by_prompt.get(parent.prompt_id, [])
            summary = summary_by_prompt.get(parent.prompt_id, {})
            mutated = _mutate_with_critic(
                instructions=parent.instructions,
                model=args.critic_model,
                provider=args.critic_provider,
                temperature=args.critic_temperature,
                sample_outputs=outputs,
                summary=summary,
                dry_run=args.dry_run,
            )
            ok, _ = _validate_mutation(mutated)
            if not ok:
                mutated = parent.instructions + "\n\nFavor concise, evidence-backed scoring."
            prompt_id = _hash_text(mutated)[:12]
            if any(p.prompt_id == prompt_id for p in next_pool):
                mutated = mutated + f"\n\nVariation {iter_idx}-{i+1}."
            next_pool.append(
                _persist_prompt_variant(
                    root=root,
                    base_experiment=base_experiment,
                    instructions=mutated,
                    parent_id=parent.prompt_id,
                    created_iter=iter_idx,
                )
            )

        prompt_pool = next_pool

        if args.save_every > 0 and iter_idx % args.save_every == 0:
            best = ranked[0]
            _full_eval(
                prompt=best,
                records=records,
                labels=labels,
                fold_ids=fold_ids,
                cv_folds=cv_folds,
                output_root=root,
                iter_idx=iter_idx,
                args=args,
                llm_providers=llm_providers,
                llm_google_model=llm_google_model,
            )


if __name__ == "__main__":
    main()
