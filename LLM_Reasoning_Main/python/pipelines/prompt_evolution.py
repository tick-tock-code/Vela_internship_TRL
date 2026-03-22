from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Train-only GEPA-style prompt evolution for merged Experiment A+B."""


import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice
from think_reason_learn.core.llms import llm as trl_llm

from lib.llm_reasoning_features import ReasoningConfig, generate_reasoning_features, _assert_no_label_fields
from pipelines.vcbench_pipeline import _report_metrics, _train_sklearn
from lib.cv_folds import build_splits_from_fold_ids, load_or_create_folds, resolve_folds_path
from lib.paths import BASE_DIR, PROJECT_ROOT, CONFIG_DIR, PROMPT_DIR


DEFAULT_CONFIG_PATH = CONFIG_DIR / "prompt_evolution_config.json"
DEFAULT_CORE_PROMPT = PROMPT_DIR / "core_prompt_evolve.txt"
DEFAULT_EXPERIMENTS = CONFIG_DIR / "experiments_evolve.json"

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

FALLBACK_MUTATION_TAILS = [
    "Prioritize concrete evidence and avoid speculation.",
    "Be conservative unless signals are strong and consistent.",
    "When evidence conflicts, average rather than polarize.",
    "Treat missing evidence as neutral, not negative.",
    "Focus on leadership scope and ownership of outcomes.",
    "Emphasize scrappiness and resourcefulness over pedigree.",
    "Weight sustained progression more than one-off spikes.",
    "Favor clear cause-and-effect in career impact.",
    "Discount titles without demonstrated responsibility.",
    "Prefer specific, verifiable signals over vague claims.",
]


def _sanitize_instruction_text(text: str) -> str:
    lines = []
    seen = set()
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.lower().startswith("variation"):
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        lines.append(stripped)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _ensure_unique_instruction(
    text: str,
    existing_hashes: set[str],
    salt: str,
) -> str:
    base = _sanitize_instruction_text(text)
    if _hash_text(base) not in existing_hashes:
        return base
    start = abs(hash(salt)) % len(FALLBACK_MUTATION_TAILS)
    for offset in range(len(FALLBACK_MUTATION_TAILS)):
        tail = FALLBACK_MUTATION_TAILS[(start + offset) % len(FALLBACK_MUTATION_TAILS)]
        if tail.lower() in base.lower():
            continue
        candidate = _sanitize_instruction_text(base + "\n\n" + tail)
        if _hash_text(candidate) not in existing_hashes:
            return candidate
    return base

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
    "keep_top": 2,
    "sample_size": 200,
    "batch_size": 20,
    "concurrency": 10,
    "concurrency_ladder": [10, 8, 6, 4, 2, 1],
    "iterations": 10,
    "save_every": 0,
    "full_eval_enabled": False,
    "full_eval_concurrency": 2,
    "full_eval_max_batches": 0,
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
    "budget_usd_per_iteration": 1.0,
    "usd_per_1k_input": 0.001,
    "usd_per_1k_output": 0.002,
    "critic_usd_per_1k_input": 0.001,
    "critic_usd_per_1k_output": 0.002,
    "token_chars_per_token": 4.0,
    "max_rate_limit_retries_per_batch": 2,
    "max_rate_limit_errors": 20,
    "rate_limit_sleep_min": 10.0,
    "rate_limit_sleep_max": 120.0,
    "max_batches": 0,
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


@dataclass
class BudgetTracker:
    budget_usd: float
    usd_per_1k_input: float
    usd_per_1k_output: float
    chars_per_token: float
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def add_tokens(self, input_tokens: int, output_tokens: int) -> float:
        if input_tokens < 0:
            input_tokens = 0
        if output_tokens < 0:
            output_tokens = 0
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        cost = 0.0
        if self.usd_per_1k_input > 0:
            cost += (input_tokens / 1000.0) * self.usd_per_1k_input
        if self.usd_per_1k_output > 0:
            cost += (output_tokens / 1000.0) * self.usd_per_1k_output
        self.spent_usd += cost
        return cost

    def exceeded(self) -> bool:
        return self.budget_usd > 0 and self.spent_usd >= self.budget_usd


def _estimate_tokens_from_text(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    denom = chars_per_token if chars_per_token > 0 else 4.0
    return int((len(text) + denom - 1) // denom)


def _estimate_tokens_from_chars(char_count: int, chars_per_token: float) -> int:
    if char_count <= 0:
        return 0
    denom = chars_per_token if chars_per_token > 0 else 4.0
    return int((char_count + denom - 1) // denom)


def _estimate_cost_from_text(
    prompt_text: str,
    response_text: str,
    usd_per_1k_input: float,
    usd_per_1k_output: float,
    chars_per_token: float,
) -> tuple[int, int, float]:
    input_tokens = _estimate_tokens_from_text(prompt_text, chars_per_token)
    output_tokens = _estimate_tokens_from_text(response_text, chars_per_token)
    cost = 0.0
    if usd_per_1k_input > 0:
        cost += (input_tokens / 1000.0) * usd_per_1k_input
    if usd_per_1k_output > 0:
        cost += (output_tokens / 1000.0) * usd_per_1k_output
    return input_tokens, output_tokens, cost


def _apply_budget(
    tracker: BudgetTracker,
    input_tokens: int,
    output_tokens: int,
    usd_per_1k_input: float,
    usd_per_1k_output: float,
) -> float:
    if input_tokens < 0:
        input_tokens = 0
    if output_tokens < 0:
        output_tokens = 0
    tracker.input_tokens += input_tokens
    tracker.output_tokens += output_tokens
    cost = 0.0
    if usd_per_1k_input > 0:
        cost += (input_tokens / 1000.0) * usd_per_1k_input
    if usd_per_1k_output > 0:
        cost += (output_tokens / 1000.0) * usd_per_1k_output
    tracker.spent_usd += cost
    return cost


def _is_rate_limit_error_message(exc: Exception) -> bool:
    msg = str(exc).lower()
    tokens = ("rate limit", "429", "quota", "resource exhausted", "resource_exhausted")
    return any(token in msg for token in tokens)


def _fallback_mutation(instructions: str, salt: str) -> str:
    tail = FALLBACK_MUTATION_TAILS[hash(salt) % len(FALLBACK_MUTATION_TAILS)]
    return instructions.rstrip() + "\n\n" + tail


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_env_if_present() -> None:
    # Load .env from parent of Project_folder if present (one level above this file's parent).
    env_path = PROJECT_ROOT / ".env"
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


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


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
    instructions = _sanitize_instruction_text(instructions)
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


def _load_prompt_variant(root: Path, prompt_id: str) -> PromptVariant:
    prompt_dir = root / "prompts" / prompt_id
    instructions_path = prompt_dir / "instructions.txt"
    meta_path = prompt_dir / "metadata.json"
    exp_path = prompt_dir / "experiments.json"
    if not instructions_path.exists():
        raise FileNotFoundError(f"Missing instructions for prompt {prompt_id}")
    instructions = instructions_path.read_text(encoding="utf-8")
    parent_id = None
    created_iter = None
    if meta_path.exists():
        meta = _read_json(meta_path)
        parent_id = meta.get("parent_id")
        created_iter = meta.get("created_iter")
    return PromptVariant(
        prompt_id=prompt_id,
        instructions=instructions,
        experiments_path=exp_path,
        parent_id=parent_id,
        created_iter=created_iter,
    )


def _load_prompt_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    prompts_root = root / "prompts"
    if not prompts_root.exists():
        return hashes
    for hash_path in prompts_root.glob("*/hash.json"):
        try:
            payload = _read_json(hash_path)
            value = payload.get("hash")
            if isinstance(value, str) and value:
                hashes.add(value)
        except Exception:
            continue
    return hashes


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
        "You MAY rephrase the dimension labels to reduce bias, while keeping their intent.\n"
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
    fallback_salt: str,
    rate_limit_sleep_min: float,
    rate_limit_sleep_max: float,
    max_rate_limit_retries: int,
    max_attempts: int = 3,
) -> tuple[str, dict[str, int]]:
    if dry_run:
        mutated = instructions + "\n\nMake the rubric more concise and evidence-based."
        return mutated, {
            "prompt_chars": 0,
            "response_chars": len(mutated),
        }

    prompt = _critic_prompt(instructions, sample_outputs, summary)
    if provider == "google":
        choice = GoogleChoice(model=model)
    else:
        choice = OpenAIChoice(model=model)
    last_text = ""
    prompt_used = prompt
    rate_limit_attempts = 0
    for attempt in range(max_attempts):
        try:
            resp = trl_llm.respond_sync(
                llm_priority=[choice],
                query=prompt,
                response_format=str,
                temperature=temperature,
            )
        except Exception as exc:
            if _is_rate_limit_error_message(exc):
                rate_limit_attempts += 1
                if max_rate_limit_retries >= 0 and rate_limit_attempts > max_rate_limit_retries:
                    break
                sleep_seconds = max(rate_limit_sleep_min, 2 ** min(rate_limit_attempts, 5))
                sleep_seconds = min(rate_limit_sleep_max, sleep_seconds)
                time.sleep(sleep_seconds)
                continue
            break
        last_text = str(resp.response).strip()
        prompt_used = prompt
        if last_text.startswith("```"):
            last_text = last_text.strip("`").strip()
        ok, _ = _validate_mutation(last_text)
        if ok:
            return last_text, {
                "prompt_chars": len(prompt_used),
                "response_chars": len(last_text),
            }
        prompt = (
            prompt
            + "\n\nIMPORTANT: Return ONLY the revised instruction text. "
            "Do NOT mention JSON, output keys, or formatting."
        )
    fallback = _fallback_mutation(instructions, fallback_salt)
    return fallback, {
        "prompt_chars": len(prompt_used),
        "response_chars": len(fallback),
    }


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
    concurrency: int,
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
        concurrency=concurrency,
        experiments=[args.experiment],
        dry_run=args.dry_run,
        dry_run_fast=False,
        log_dir=log_dir,
        log_every=1,
        repair_nan=False,
        repair_existing=False,
        skip_select=True,
        rate_limit_sleep_min=args.rate_limit_sleep_min,
        rate_limit_sleep_max=args.rate_limit_sleep_max,
        max_rate_limit_retries_per_batch=args.max_rate_limit_retries_per_batch,
        max_rate_limit_errors=args.max_rate_limit_errors,
        max_batches=args.max_batches,
        token_chars_per_token=args.token_chars_per_token,
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
    concurrency: int,
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
        concurrency=concurrency,
        experiments=[args.experiment],
        dry_run=args.dry_run,
        dry_run_fast=False,
        log_dir=log_dir,
        log_every=1,
        repair_nan=False,
        repair_existing=False,
        skip_select=True,
        rate_limit_sleep_min=args.rate_limit_sleep_min,
        rate_limit_sleep_max=args.rate_limit_sleep_max,
        max_rate_limit_retries_per_batch=args.max_rate_limit_retries_per_batch,
        max_rate_limit_errors=args.max_rate_limit_errors,
        max_batches=args.full_eval_max_batches,
        token_chars_per_token=args.token_chars_per_token,
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


def _parse_ladder(value: Any, fallback: int) -> list[int]:
    if isinstance(value, list):
        ladder = []
        for v in value:
            try:
                iv = int(v)
                if iv > 0:
                    ladder.append(iv)
            except Exception:
                continue
        return ladder if ladder else [fallback]
    if isinstance(value, str):
        parts = []
        for chunk in value.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                iv = int(chunk)
                if iv > 0:
                    parts.append(iv)
            except Exception:
                continue
        return parts if parts else [fallback]
    return [fallback]


def _apply_config(args: argparse.Namespace, cfg: dict[str, Any]) -> argparse.Namespace:
    for key, value in cfg.items():
        if hasattr(args, key) and key in DEFAULTS:
            if getattr(args, key) == DEFAULTS[key]:
                setattr(args, key, value)
        if key == "concurrency_ladder":
            setattr(args, key, value)
    return args


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train-only prompt evolution for Experiment AB.")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--experiment", type=str, default=DEFAULTS["experiment"])
    p.add_argument("--pool_size", type=int, default=DEFAULTS["pool_size"])
    p.add_argument("--keep_top", type=int, default=DEFAULTS["keep_top"])
    p.add_argument("--sample_size", type=int, default=DEFAULTS["sample_size"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--concurrency", type=int, default=DEFAULTS["concurrency"])
    p.add_argument("--concurrency_ladder", type=str, default="10,8,6,4,2,1")
    p.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    p.add_argument("--run_id", type=str, default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--save_every", type=int, default=DEFAULTS["save_every"])
    p.add_argument("--full_eval_enabled", action="store_true")
    p.add_argument("--full_eval_concurrency", type=int, default=DEFAULTS["full_eval_concurrency"])
    p.add_argument("--full_eval_max_batches", type=int, default=DEFAULTS["full_eval_max_batches"])
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
    p.add_argument("--budget_usd_per_iteration", type=float, default=DEFAULTS["budget_usd_per_iteration"])
    p.add_argument("--usd_per_1k_input", type=float, default=DEFAULTS["usd_per_1k_input"])
    p.add_argument("--usd_per_1k_output", type=float, default=DEFAULTS["usd_per_1k_output"])
    p.add_argument("--critic_usd_per_1k_input", type=float, default=DEFAULTS["critic_usd_per_1k_input"])
    p.add_argument("--critic_usd_per_1k_output", type=float, default=DEFAULTS["critic_usd_per_1k_output"])
    p.add_argument("--token_chars_per_token", type=float, default=DEFAULTS["token_chars_per_token"])
    p.add_argument("--max_rate_limit_retries_per_batch", type=int, default=DEFAULTS["max_rate_limit_retries_per_batch"])
    p.add_argument("--max_rate_limit_errors", type=int, default=DEFAULTS["max_rate_limit_errors"])
    p.add_argument("--rate_limit_sleep_min", type=float, default=DEFAULTS["rate_limit_sleep_min"])
    p.add_argument("--rate_limit_sleep_max", type=float, default=DEFAULTS["rate_limit_sleep_max"])
    p.add_argument("--max_batches", type=int, default=DEFAULTS["max_batches"])
    return p.parse_args()


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    cfg = _load_config(args)
    args = _apply_config(args, cfg)
    concurrency_ladder = _parse_ladder(
        getattr(args, "concurrency_ladder", None),
        fallback=int(args.concurrency),
    )
    if int(args.concurrency) not in concurrency_ladder:
        concurrency_ladder = [int(args.concurrency)] + [
            v for v in concurrency_ladder if v != int(args.concurrency)
        ]

    core_prompt_path = Path(args.core_prompt_path)
    if not core_prompt_path.is_absolute():
        core_prompt_path = PROMPT_DIR / core_prompt_path
    args.core_prompt_path = str(core_prompt_path)
    experiments_path = Path(args.experiments_path)
    if not experiments_path.is_absolute():
        experiments_path = CONFIG_DIR / experiments_path
    args.experiments_path = str(experiments_path)
    root = Path(args.output_root)
    root = root if root.is_absolute() else BASE_DIR / root
    root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id.strip() if str(args.run_id).strip() else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = root / "runs" / run_id
    if args.resume and not run_root.exists():
        raise RuntimeError(f"Cannot resume; run directory not found: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    run_cfg = dict(vars(args))
    run_cfg["concurrency_ladder"] = concurrency_ladder
    run_cfg_path = run_root / "run_config.json"
    if not (args.resume and run_cfg_path.exists()):
        _write_json(run_cfg_path, run_cfg)
    _write_json(
        run_root / "run_start.json",
        {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "resume": bool(args.resume),
            "iterations": int(args.iterations),
            "pool_size": int(args.pool_size),
            "sample_size": int(args.sample_size),
        },
    )

    def _log_unhandled(exc_type, exc, tb) -> None:
        _write_json(
            run_root / "run_error.json",
            {
                "error": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
                "timestamp": datetime.now().isoformat(),
            },
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_unhandled

    input_csv = _resolve_input_csv(args.dataset, args.input_csv or None)
    records, labels = _load_vcbench_local(input_csv)
    founder_ids = [r.get("founder_uuid") for r in records]

    features_cfg = CONFIG_DIR / "features.json"
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

    seed_path = BASE_DIR / "features_storage" / "llm_engineered" / f"seed_{seed_size}.json"
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

    folds_path = resolve_folds_path(BASE_DIR, cv_folds, args.random_state, cv_folds_path)
    _, fold_ids, folds_path = load_or_create_folds(
        founder_ids=founder_ids,
        labels=labels,
        cv_folds=cv_folds,
        random_state=args.random_state,
        folds_path=folds_path,
        dataset_label=input_csv,
        use_fixed=cv_use_fixed_folds,
    )

    start_iter = 1
    prompt_pool: list[PromptVariant] = []
    prompt_hashes: set[str] = set()

    critic_input_rate = (
        args.critic_usd_per_1k_input
        if args.critic_usd_per_1k_input > 0
        else args.usd_per_1k_input
    )
    critic_output_rate = (
        args.critic_usd_per_1k_output
        if args.critic_usd_per_1k_output > 0
        else args.usd_per_1k_output
    )

    if args.resume:
        completed_iters: list[int] = []
        for iter_dir in sorted(run_root.glob("iter_*")):
            if not iter_dir.is_dir():
                continue
            metrics_path = iter_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                idx = int(iter_dir.name.split("_")[-1])
            except Exception:
                continue
            completed_iters.append(idx)
        if not completed_iters:
            raise RuntimeError("Resume requested but no completed iterations found.")
        last_iter = max(completed_iters)
        start_iter = last_iter + 1
        if start_iter > args.iterations:
            print(
                f"[prompt_evolution] Resume requested but iterations already complete "
                f"(last={last_iter}, target={args.iterations})."
            )
            return
        selected_path = run_root / f"iter_{last_iter:03d}" / "selected_ids.json"
        selected_ids: list[str] = []
        if selected_path.exists():
            selected_ids = [str(pid) for pid in _read_json(selected_path)]
        created_ids: list[str] = []
        prompts_root = root / "prompts"
        if prompts_root.exists():
            for meta_path in prompts_root.glob("*/metadata.json"):
                try:
                    meta = _read_json(meta_path)
                except Exception:
                    continue
                if meta.get("created_iter") == last_iter:
                    pid = str(meta.get("prompt_id", "")).strip()
                    if pid:
                        created_ids.append(pid)
        pool_ids: list[str] = []
        for pid in selected_ids + created_ids:
            if pid not in pool_ids:
                pool_ids.append(pid)
        if not pool_ids:
            raise RuntimeError("Resume requested but prompt pool could not be reconstructed.")
        for pid in pool_ids:
            prompt_pool.append(_load_prompt_variant(root, pid))
        prompt_hashes = _load_prompt_hashes(root)
        if len(prompt_pool) < args.pool_size and selected_ids:
            while len(prompt_pool) < args.pool_size:
                prompt_pool.append(
                    _load_prompt_variant(root, selected_ids[len(prompt_pool) % len(selected_ids)])
                )
        base_prompt_candidates = [p for p in prompt_pool if p.created_iter is None]
        base_prompt = base_prompt_candidates[0] if base_prompt_candidates else prompt_pool[0]
        _write_json(
            run_root / "resume.json",
            {
                "resumed_at": datetime.now().isoformat(),
                "last_completed_iter": last_iter,
                "start_iter": start_iter,
                "target_iterations": args.iterations,
            },
        )
    else:
        base_instructions_clean = _ensure_unique_instruction(base_instructions, prompt_hashes, "base")
        base_prompt = _persist_prompt_variant(
            root=root,
            base_experiment=base_experiment,
            instructions=base_instructions_clean,
            parent_id=None,
            created_iter=None,
        )
        prompt_hashes.add(_hash_text(base_instructions_clean))
        prompt_pool.append(base_prompt)

        seed_budget = BudgetTracker(
            budget_usd=float(args.budget_usd_per_iteration),
            usd_per_1k_input=float(critic_input_rate),
            usd_per_1k_output=float(critic_output_rate),
            chars_per_token=float(args.token_chars_per_token),
        )

        initial_mutations = max(0, int(args.initial_mutations))
        for i in range(min(initial_mutations, args.pool_size - 1)):
            mutated, usage = _mutate_with_critic(
                instructions=base_instructions,
                model=args.critic_model,
                provider=args.critic_provider,
                temperature=args.critic_temperature,
                sample_outputs=[],
                summary={"note": "initial mutation"},
                dry_run=args.dry_run,
                fallback_salt=f"init-{i}",
                rate_limit_sleep_min=args.rate_limit_sleep_min,
                rate_limit_sleep_max=args.rate_limit_sleep_max,
                max_rate_limit_retries=args.max_rate_limit_retries_per_batch,
            )
            mutated = _ensure_unique_instruction(mutated, prompt_hashes, f"init-{i}")
            if usage:
                prompt_tokens = _estimate_tokens_from_chars(
                    usage.get("prompt_chars", 0), args.token_chars_per_token
                )
                response_tokens = _estimate_tokens_from_chars(
                    usage.get("response_chars", 0), args.token_chars_per_token
                )
                _apply_budget(
                    seed_budget,
                    prompt_tokens,
                    response_tokens,
                    critic_input_rate,
                    critic_output_rate,
                )
                if seed_budget.exceeded():
                    break
            ok, _ = _validate_mutation(mutated)
            if not ok:
                mutated = base_instructions + "\n\nEmphasize consistency and evidence."
            mutated = _ensure_unique_instruction(mutated, prompt_hashes, f"init-{i}-fallback")
            prompt_hashes.add(_hash_text(mutated))
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

    current_concurrency = int(args.concurrency)
    stop_run = False
    for iter_idx in range(start_iter, args.iterations + 1):
        iter_label = f"iter_{iter_idx:03d}"
        iter_root = run_root / iter_label
        iter_root.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(args.random_state + iter_idx)
        sample_n = min(args.sample_size, len(records))
        sample_indices = rng.choice(len(records), size=sample_n, replace=False)
        sample_records = [records[i] for i in sample_indices]
        sample_labels = labels[sample_indices]
        _write_json(iter_root / "sample_indices.json", [int(i) for i in sample_indices])

        iter_budget = BudgetTracker(
            budget_usd=float(args.budget_usd_per_iteration),
            usd_per_1k_input=float(args.usd_per_1k_input),
            usd_per_1k_output=float(args.usd_per_1k_output),
            chars_per_token=float(args.token_chars_per_token),
        )
        budget_records: list[dict[str, Any]] = []
        budget_exceeded = False

        metrics_by_prompt: dict[str, dict[str, float]] = {}
        outputs_by_prompt: dict[str, list[dict[str, Any]]] = {}
        summary_by_prompt: dict[str, dict[str, Any]] = {}

        for prompt in prompt_pool:
            ladder_index = (
                concurrency_ladder.index(current_concurrency)
                if current_concurrency in concurrency_ladder
                else 0
            )
            prompt_error_logged = False
            while True:
                try:
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
                        concurrency=current_concurrency,
                    )
                    metrics_by_prompt[prompt.prompt_id] = metrics
                    outputs_by_prompt[prompt.prompt_id] = outputs
                    summary_by_prompt[prompt.prompt_id] = summary
                    meta_path = root / "outputs" / prompt.prompt_id / iter_label / "reasoning_meta.json"
                    if meta_path.exists():
                        try:
                            meta = _read_json(meta_path)
                            input_tokens = int(meta.get("estimated_prompt_tokens", 0))
                            output_tokens = int(meta.get("estimated_completion_tokens", 0))
                        except Exception:
                            input_tokens = 0
                            output_tokens = 0
                        cost = _apply_budget(
                            iter_budget,
                            input_tokens,
                            output_tokens,
                            float(args.usd_per_1k_input),
                            float(args.usd_per_1k_output),
                        )
                        budget_records.append(
                            {
                                "stage": "generation",
                                "prompt_id": prompt.prompt_id,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "cost_usd": cost,
                                "spent_usd": iter_budget.spent_usd,
                            }
                        )
                    if iter_budget.exceeded():
                        budget_exceeded = True
                    break
                except Exception as exc:
                    msg = str(exc).lower()
                    should_downshift = any(
                        token in msg
                        for token in ("rate limit", "429", "quota", "resource", "timeout", "socket", "connection")
                    )
                    if should_downshift and ladder_index < len(concurrency_ladder) - 1:
                        ladder_index += 1
                        current_concurrency = concurrency_ladder[ladder_index]
                        _write_json(
                            iter_root / "concurrency_fallback.json",
                            {
                                "iteration": iter_idx,
                                "prompt_id": prompt.prompt_id,
                                "error": str(exc),
                                "concurrency_now": current_concurrency,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        continue
                    if not prompt_error_logged:
                        _append_jsonl(
                            iter_root / "prompt_errors.jsonl",
                            {
                                "iteration": iter_idx,
                                "prompt_id": prompt.prompt_id,
                                "error": str(exc),
                                "concurrency": current_concurrency,
                                "timestamp": datetime.now().isoformat(),
                            },
                        )
                        prompt_error_logged = True
                    break
            if budget_exceeded:
                break

        _write_json(iter_root / "metrics.json", metrics_by_prompt)

        available_prompts = [p for p in prompt_pool if p.prompt_id in metrics_by_prompt]
        missing_prompts = [p.prompt_id for p in prompt_pool if p.prompt_id not in metrics_by_prompt]
        if missing_prompts:
            _write_json(
                iter_root / "missing_metrics.json",
                {"missing_prompt_ids": missing_prompts, "timestamp": datetime.now().isoformat()},
            )

        if not available_prompts:
            _write_json(
                iter_root / "iteration_error.json",
                {
                    "error": "No prompts completed with metrics in this iteration.",
                    "timestamp": datetime.now().isoformat(),
                },
            )
            break

        ranked = sorted(
            available_prompts,
            key=lambda p: metrics_by_prompt.get(p.prompt_id, {}).get(args.selection_metric, 0.0),
            reverse=True,
        )
        keep_top = max(1, min(int(args.keep_top), len(ranked)))
        top_keep = ranked[:keep_top]
        selected_ids = [p.prompt_id for p in top_keep]
        _write_json(iter_root / "selected_ids.json", selected_ids)
        _write_json(iter_root / "pool.json", [p.prompt_id for p in prompt_pool])

        next_pool = list(top_keep)
        parent_cycle = ranked[:keep_top]
        if not parent_cycle:
            parent_cycle = [base_prompt]

        desired_new = args.pool_size - len(next_pool)
        for i in range(desired_new):
            if iter_budget.exceeded():
                budget_exceeded = True
                break
            parent = parent_cycle[i % len(parent_cycle)]
            outputs = outputs_by_prompt.get(parent.prompt_id, [])
            summary = summary_by_prompt.get(parent.prompt_id, {})
            mutated, usage = _mutate_with_critic(
                instructions=parent.instructions,
                model=args.critic_model,
                provider=args.critic_provider,
                temperature=args.critic_temperature,
                sample_outputs=outputs,
                summary=summary,
                dry_run=args.dry_run,
                fallback_salt=f"{iter_idx}-{i}",
                rate_limit_sleep_min=args.rate_limit_sleep_min,
                rate_limit_sleep_max=args.rate_limit_sleep_max,
                max_rate_limit_retries=args.max_rate_limit_retries_per_batch,
            )
            mutated = _ensure_unique_instruction(mutated, prompt_hashes, f"{iter_idx}-{i}")
            if usage:
                prompt_tokens = _estimate_tokens_from_chars(
                    usage.get("prompt_chars", 0), args.token_chars_per_token
                )
                response_tokens = _estimate_tokens_from_chars(
                    usage.get("response_chars", 0), args.token_chars_per_token
                )
                cost = _apply_budget(
                    iter_budget,
                    prompt_tokens,
                    response_tokens,
                    critic_input_rate,
                    critic_output_rate,
                )
                budget_records.append(
                    {
                        "stage": "critic",
                        "prompt_id": parent.prompt_id,
                        "input_tokens": prompt_tokens,
                        "output_tokens": response_tokens,
                        "cost_usd": cost,
                        "spent_usd": iter_budget.spent_usd,
                    }
                )
                if iter_budget.exceeded():
                    budget_exceeded = True
                    break
            ok, _ = _validate_mutation(mutated)
            if not ok:
                mutated = parent.instructions + "\n\nFavor concise, evidence-backed scoring."
            mutated = _ensure_unique_instruction(mutated, prompt_hashes, f"{iter_idx}-{i}-fallback")
            prompt_hashes.add(_hash_text(mutated))
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

        _write_json(
            iter_root / "budget.json",
            {
                "budget_usd": iter_budget.budget_usd,
                "spent_usd": iter_budget.spent_usd,
                "input_tokens": iter_budget.input_tokens,
                "output_tokens": iter_budget.output_tokens,
                "records": budget_records,
                "exceeded": budget_exceeded,
            },
        )

        if budget_exceeded:
            stop_run = True
            _write_json(
                iter_root / "budget_exceeded.json",
                {
                    "iteration": iter_idx,
                    "spent_usd": iter_budget.spent_usd,
                    "budget_usd": iter_budget.budget_usd,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            break

        if args.full_eval_enabled and args.save_every > 0 and iter_idx % args.save_every == 0:
            best = ranked[0]
            full_eval_concurrency = (
                int(args.full_eval_concurrency) if args.full_eval_concurrency else current_concurrency
            )
            if full_eval_concurrency < 1:
                full_eval_concurrency = 1
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
                concurrency=full_eval_concurrency,
            )

        if stop_run:
            break
if __name__ == "__main__":
    main()
