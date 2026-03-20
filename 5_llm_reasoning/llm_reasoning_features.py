"""LLM reasoning feature generation for VCBench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import time
import hashlib
import threading
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI

from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice
from think_reason_learn.core.llms import llm as trl_llm


GLOBAL_NUMERIC_KEYS = ["evidence_support_rating"]
GLOBAL_TEXT_KEYS: list[str] = []
LABEL_FIELDS = {"success", "label", "y"}
_THREAD_LOCAL = threading.local()


@dataclass
class ReasoningConfig:
    model: str
    dataset_size: str  # "full", "200", "400", "1000"
    random_state: int
    core_prompt_path: Path
    experiments_path: Path
    providers: dict[str, bool]
    google_model: str | None = None
    batch_size: int = 20
    concurrency: int = 1
    experiments: list[str] | None = None
    dry_run: bool = False
    dry_run_fast: bool = False
    log_dir: Path | None = None
    log_every: int = 1
    repair_nan: bool = True
    repair_existing: bool = False
    skip_select: bool = False
    rate_limit_fallback_concurrency: int = 5
    rate_limit_fallback_windows: int = 1
    rate_limit_fallback_sequence: list[int] | None = None
    inline_repair: bool = True
    inline_repair_max_attempts: int = 1
    target_batch_indices: list[int] | None = None
    rate_limit_sleep_min: float = 5.0
    rate_limit_sleep_max: float = 120.0
    max_rate_limit_retries_per_batch: int = 3
    max_rate_limit_errors: int = 100
    retryable_requeue_max_attempts: int = 2
    max_batches: int = 0
    token_chars_per_token: float = 4.0


def _load_core_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if "{{EXPERIMENT_INSTRUCTIONS}}" not in text or "{{FOUNDER}}" not in text:
        raise ValueError("Core prompt must include {{EXPERIMENT_INSTRUCTIONS}} and {{FOUNDER}} placeholders.")
    return text


def _load_experiments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list) or not data:
        raise ValueError("Experiments JSON must be a non-empty list.")
    return data


def _validate_experiments(experiments: list[dict[str, Any]]) -> None:
    ids = []
    for exp in experiments:
        if "id" not in exp or "instructions" not in exp:
            raise ValueError("Each experiment must include id and instructions.")
        exp_id = str(exp.get("id"))
        if exp_id in ids:
            raise ValueError(f"Duplicate experiment id: {exp_id}")
        ids.append(exp_id)
        numeric_keys = exp.get("numeric_keys", [])
        text_keys = exp.get("text_keys", [])
        if not isinstance(numeric_keys, list) or not isinstance(text_keys, list):
            raise ValueError(f"Experiment {exp_id} keys must be lists.")
        if not numeric_keys and not text_keys:
            raise ValueError(f"Experiment {exp_id} must define numeric_keys or text_keys.")


def _select_records(
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


def _sanitize(text: str) -> str:
    return "".join(ch if ch.isprintable() else " " for ch in text)


def _parse_numeric(text: str) -> float:
    match = re.search(r"[-+]?[0-9]*\\.?[0-9]+", text)
    if not match:
        return float("nan")
    return float(match.group(0))


def _expected_range(key: str) -> tuple[float, float]:
    return (1.0, 5.0)


def _format_record(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False)


def _parse_json_scores(text: str) -> list[dict[str, float]]:
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    items = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "index" not in item or "score" not in item:
            continue
        try:
            idx = int(item["index"])
            score = float(item["score"])
        except Exception:
            continue
        items.append({"index": idx, "score": score})
    return items


def _parse_batch_response(
    raw_text: str,
    batch_len: int,
) -> dict[int, dict[str, Any]]:
    try:
        data = json.loads(raw_text)
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    items: dict[int, dict[str, Any]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        if "index" not in item:
            continue
        try:
            idx = int(item["index"])
        except Exception:
            continue
        if 0 <= idx < batch_len:
            items[idx] = item
    return items


def _refresh_llm_from_env() -> None:
    try:
        from think_reason_learn.core import _config as trl_config
        def _env(key: str) -> str:
            val = os.getenv(key, "")
            if val:
                return val
            bom_key = "\ufeff" + key
            return os.getenv(bom_key, "")

        if _env("OPENAI_API_KEY"):
            trl_config.settings.OPENAI_API_KEY = _env("OPENAI_API_KEY")
        if _env("GOOGLE_AI_API_KEY"):
            trl_config.settings.GOOGLE_AI_API_KEY = _env("GOOGLE_AI_API_KEY")
        if _env("ANTHROPIC_API_KEY"):
            trl_config.settings.ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
        if _env("XAI_API_KEY"):
            trl_config.settings.XAI_API_KEY = _env("XAI_API_KEY")
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


def _get_openai_client() -> OpenAI:
    client = getattr(_THREAD_LOCAL, "openai_client", None)
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            api_key = os.getenv("\ufeffOPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)
        _THREAD_LOCAL.openai_client = client
    return client


def _get_google_client():
    client = getattr(_THREAD_LOCAL, "google_client", None)
    if client is None:
        try:
            from google import genai
        except Exception as exc:
            raise RuntimeError("Google genai client not available.") from exc
        api_key = os.getenv("GOOGLE_AI_API_KEY", "")
        client = genai.Client(api_key=api_key)
        _THREAD_LOCAL.google_client = client
    return client


def _google_output_text(response: Any) -> str:
    try:
        candidates = getattr(response, "candidates", None)
        if candidates:
            content = getattr(candidates[0], "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    part = parts[0]
                    text = getattr(part, "text", "")
                    if text is not None:
                        return str(text)
    except Exception:
        return str(response)
    return ""


def _short_key(key: str) -> str:
    if key in GLOBAL_NUMERIC_KEYS or key in GLOBAL_TEXT_KEYS:
        return key
    if "_" in key:
        return key.split("_", 1)[1]
    return key


def _assert_no_label_fields(obj: dict[str, Any], context: str) -> None:
    bad = [k for k in obj.keys() if k in LABEL_FIELDS]
    if bad:
        raise RuntimeError(
            f"Label field(s) {bad} found in {context}. "
            "Remove outcome labels before LLM prompting."
        )


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    if exc.__class__.__name__.lower().find("ratelimit") >= 0:
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    return False


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_tokens = (
        "connection error",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "service unavailable",
        "socket",
        "eof",
        "name or service not known",
        "dns",
        "ssl",
        "tls",
        "proxy",
    )
    if any(token in msg for token in retry_tokens):
        return True
    name = exc.__class__.__name__.lower()
    if any(token in name for token in ("timeout", "connection", "transport", "proxy", "socket")):
        return True
    return False


def _extract_retry_delay_seconds(message: str) -> float | None:
    msg = message.lower()
    match = re.search(r"retry in ([0-9]+(?:\\.[0-9]+)?)s", msg)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    match = re.search(r"retrydelay['\\\"]?:\\s*['\\\"]?([0-9]+(?:\\.[0-9]+)?)s", msg)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    match = re.search(r"retrydelay['\\\"]?:\\s*['\\\"]?([0-9]+(?:\\.[0-9]+)?)", msg)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    denom = chars_per_token if chars_per_token > 0 else 4.0
    return int(math.ceil(len(text) / denom))


def write_per_experiment_parquets(
    df: pd.DataFrame,
    exp_to_keys: dict[str, list[str]],
    output_dir: Path,
    dataset_size: str,
    overwrite: bool,
) -> None:
    exp_root = output_dir / "experiments"
    for exp_id, exp_keys in exp_to_keys.items():
        evidence_col = f"{exp_id}_evidence_support_rating"
        if evidence_col in df.columns and evidence_col not in exp_keys:
            exp_keys = exp_keys + [evidence_col]
        exp_cols = ["founder_uuid", "success"]
        exp_cols += [c for c in GLOBAL_NUMERIC_KEYS if c in df.columns]
        exp_cols += [c for c in GLOBAL_TEXT_KEYS if c in df.columns]
        exp_cols += [c for c in exp_keys if c in df.columns]
        exp_df = df[exp_cols].copy()
        exp_dir = exp_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_path = exp_dir / f"llm_reasoning_{dataset_size}.parquet"
        if exp_path.exists() and not overwrite:
            continue
        exp_df.to_parquet(exp_path, index=False)


def build_experiment_key_map(
    experiments_path: Path,
    selected_experiments: list[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    experiments = _load_experiments(experiments_path)
    if selected_experiments:
        experiments = [e for e in experiments if e.get("id") in set(selected_experiments)]
    if not experiments:
        raise RuntimeError("No experiments selected for LLM reasoning.")
    _validate_experiments(experiments)

    feature_keys: list[str] = []
    exp_to_keys: dict[str, list[str]] = {}
    for k in GLOBAL_NUMERIC_KEYS:
        feature_keys.append(k)
    for k in GLOBAL_TEXT_KEYS:
        feature_keys.append(k)
    for exp in experiments:
        exp_id = str(exp.get("id"))
        exp_keys: list[str] = []
        for k in exp.get("numeric_keys", []):
            key = f"{exp_id}_{k}"
            feature_keys.append(key)
            exp_keys.append(key)
        for k in exp.get("text_keys", []):
            key = f"{exp_id}_{k}"
            feature_keys.append(key)
            exp_keys.append(key)
        exp_keys.append(f"{exp_id}_evidence_support_rating")
        exp_to_keys[exp_id] = exp_keys
    return feature_keys, exp_to_keys


def generate_reasoning_features(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    config: ReasoningConfig,
    output_dir: Path,
    metadata_path: Path,
    existing_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    core_prompt = _load_core_prompt(config.core_prompt_path)
    experiments = _load_experiments(config.experiments_path)
    if config.experiments:
        experiments = [e for e in experiments if e.get("id") in set(config.experiments)]
    if not experiments:
        raise RuntimeError("No experiments selected for LLM reasoning.")
    _validate_experiments(experiments)
    dataset_size = config.dataset_size
    if config.dry_run_fast and dataset_size == "full":
        dataset_size = "50"

    # Build output schema from experiments
    feature_keys = []
    numeric_feature_keys = []
    numeric_ranges: dict[str, tuple[float, float]] = {}
    text_keys: set[str] = set()
    exp_to_keys: dict[str, list[str]] = {}
    if config.log_every < 1:
        config.log_every = 1
    log_dir = config.log_dir
    progress_log = None
    error_log = None
    summary_path = None
    exp_label = "combined"
    if config.experiments:
        exp_label = "+".join(config.experiments)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        progress_log = log_dir / "progress.jsonl"
        error_log = log_dir / "errors.jsonl"
        summary_path = log_dir / "run_summary.json"
    for k in GLOBAL_NUMERIC_KEYS:
        feature_keys.append(k)
        numeric_feature_keys.append(k)
        numeric_ranges[k] = _expected_range(k)
    for k in GLOBAL_TEXT_KEYS:
        feature_keys.append(k)
        text_keys.add(k)
    for exp in experiments:
        exp_id = str(exp.get("id"))
        exp_keys: list[str] = []
        for k in exp.get("numeric_keys", []):
            key = f"{exp_id}_{k}"
            feature_keys.append(key)
            numeric_feature_keys.append(key)
            numeric_ranges[key] = _expected_range(k)
            exp_keys.append(key)
        for k in exp.get("text_keys", []):
            key = f"{exp_id}_{k}"
            feature_keys.append(key)
            text_keys.add(key)
            exp_keys.append(key)
        exp_to_keys[exp_id] = exp_keys

    if config.skip_select:
        selected_records, selected_labels = records, labels
    else:
        selected_records, selected_labels = _select_records(
            records, labels, dataset_size, config.random_state
        )
    total_records = len(selected_records)

    batch_size = max(1, int(config.batch_size))
    n_batches = int(np.ceil(total_records / batch_size))

    if summary_path is not None:
        summary = {
            "status": "started",
            "experiment": exp_label,
            "total_records": total_records,
            "processed": 0,
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "timestamp": time.time(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if progress_log is not None:
        entry = {
            "index": -1,
            "stage": "start_run",
            "experiment": exp_label,
            "processed": 0,
            "total": total_records,
            "batch_size": batch_size,
            "n_batches": n_batches,
            "timestamp": time.time(),
        }
        progress_log.open("a", encoding="utf-8").write(json.dumps(entry) + "\n")

    # Cache reuse (skip API calls if already computed)
    combined_path = output_dir / f"llm_reasoning_{dataset_size}.parquet"
    manifest_path = output_dir / f"llm_reasoning_{dataset_size}_manifest.json"
    in_preview_dir = "previews" in output_dir.parts
    existing_df_local: pd.DataFrame | None = None
    if existing_df is not None:
        if len(existing_df) == total_records:
            existing_df_local = existing_df.copy()
    if existing_df_local is None and not config.dry_run and not config.dry_run_fast and not in_preview_dir and combined_path.exists():
        expected_manifest = {
            "dataset_size": dataset_size,
            "random_state": config.random_state,
            "model": config.model,
            "providers": config.providers,
            "google_model": config.google_model,
            "experiments": [e.get("id") for e in experiments],
            "core_prompt_hash": _hash_text(core_prompt),
            "experiments_hash": _hash_text(json.dumps(experiments, sort_keys=True)),
            "global_numeric_keys": GLOBAL_NUMERIC_KEYS,
            "global_text_keys": GLOBAL_TEXT_KEYS,
        }
        manifest_ok = False
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_ok = all(manifest.get(k) == v for k, v in expected_manifest.items())
            except Exception:
                manifest_ok = False
        if manifest_ok or not manifest_path.exists():
            df = pd.read_parquet(combined_path)
            required_cols = set(["founder_uuid", "success"] + feature_keys)
            if required_cols.issubset(set(df.columns)):
                if not config.repair_existing:
                    write_per_experiment_parquets(
                        df,
                        exp_to_keys,
                        output_dir,
                        dataset_size,
                        overwrite=False,
                    )
                    if summary_path is not None:
                        summary = {
                            "status": "cached",
                            "experiment": exp_label,
                            "total_records": total_records,
                            "processed": total_records,
                            "dataset_size": dataset_size,
                            "timestamp": time.time(),
                        }
                        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    return df, numeric_feature_keys
                existing_df_local = df.copy()

    features: dict[str, list[Any]] = {}
    if existing_df_local is not None and len(existing_df_local) == total_records:
        for key in feature_keys:
            if key in existing_df_local.columns:
                features[key] = existing_df_local[key].tolist()
            else:
                if key in text_keys:
                    features[key] = [""] * total_records
                else:
                    features[key] = [float("nan")] * total_records
    else:
        for key in feature_keys:
            if key in text_keys:
                features[key] = [""] * total_records
            else:
                features[key] = [float("nan")] * total_records

    failures = 0
    raw_responses: list[tuple[int, str]] = []
    processed = 0
    completed_batches = 0
    log_lock = threading.Lock()
    usage_lock = threading.Lock()
    usage = {
        "prompt_chars": 0,
        "completion_chars": 0,
        "prompt_tokens_est": 0,
        "completion_tokens_est": 0,
        "requests": 0,
    }
    inline_repair_retries = 0
    inline_attempts: dict[int, int] = {}

    use_openai = bool(config.providers.get("openai", False))
    use_google = bool(config.providers.get("google", False))
    if not config.dry_run:
        _refresh_llm_from_env()
        if use_openai:
            model_choice = OpenAIChoice(model=config.model)
        elif use_google:
            model_choice = GoogleChoice(model=config.google_model or "gemini-2.0-flash")
        else:
            raise RuntimeError("No LLM providers enabled for reasoning features.")
    else:
        model_choice = None

    short_keys = [_short_key(k) for k in feature_keys]
    short_counts = Counter(short_keys)
    key_alias: dict[str, str] = {}
    for key in feature_keys:
        short = _short_key(key)
        if short_counts.get(short, 0) > 1:
            key_alias[key] = key
        else:
            key_alias[key] = short
    required_short_keys: list[str] = []
    for key in feature_keys:
        alias = key_alias[key]
        if alias not in required_short_keys:
            required_short_keys.append(alias)
    text_short_keys = {key_alias[k] for k in text_keys}
    experiment_instructions = "\n\n".join([str(exp.get("instructions", "")) for exp in experiments])
    use_batch = batch_size > 1
    concurrency = max(1, int(config.concurrency))
    fallback_concurrency = max(1, int(config.rate_limit_fallback_concurrency))
    fallback_windows = max(0, int(config.rate_limit_fallback_windows))
    fallback_sequence = list(config.rate_limit_fallback_sequence or [])
    if not fallback_sequence:
        fallback_sequence = [fallback_concurrency]
    fallback_sequence = [max(1, int(v)) for v in fallback_sequence]
    rate_limit_fallbacks = 0
    google_model = config.google_model or "gemini-2.0-flash"
    chars_per_token = float(config.token_chars_per_token) if config.token_chars_per_token else 4.0
    max_rate_limit_retries = max(0, int(config.max_rate_limit_retries_per_batch))
    max_rate_limit_errors = max(0, int(config.max_rate_limit_errors))
    max_retryable_requeues = max(0, int(config.retryable_requeue_max_attempts))
    rate_limit_sleep_min = max(0.0, float(config.rate_limit_sleep_min))
    rate_limit_sleep_max = max(rate_limit_sleep_min, float(config.rate_limit_sleep_max))
    rate_limit_attempts: dict[int, int] = {}
    rate_limit_error_count = 0
    retryable_requeues: dict[int, int] = {}

    def _accumulate_usage(prompt_text: str, response_text: str) -> None:
        if not prompt_text and not response_text:
            return
        prompt_len = len(prompt_text or "")
        completion_len = len(response_text or "")
        prompt_tokens = _estimate_tokens(prompt_text or "", chars_per_token)
        completion_tokens = _estimate_tokens(response_text or "", chars_per_token)
        with usage_lock:
            usage["prompt_chars"] += prompt_len
            usage["completion_chars"] += completion_len
            usage["prompt_tokens_est"] += prompt_tokens
            usage["completion_tokens_est"] += completion_tokens
            usage["requests"] += 1

    def _append_jsonl(path: Path | None, entry: dict[str, Any]) -> None:
        if path is None:
            return
        with log_lock:
            path.open("a", encoding="utf-8").write(json.dumps(entry) + "\n")

    def _call_batch(
        batch_idx: int,
        batch_start: int,
        batch_recs: list[dict[str, Any]],
        strict_retry: bool = False,
    ) -> tuple[int, int, list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any] | None, str, int]:
        batch_len = len(batch_recs)
        for rec in batch_recs:
            if isinstance(rec, dict):
                _assert_no_label_fields(rec, "founder record")

        if use_batch:
            batch_payload = [
                {"index": i, "founder": rec} for i, rec in enumerate(batch_recs)
            ]
            founder_text = json.dumps(batch_payload, ensure_ascii=False)
            keys_list = ", ".join(required_short_keys)
            example_count = 2 if batch_len >= 2 else 1
            example_items: list[str] = []
            for ex_idx in range(example_count):
                fields: list[str] = [f"\"index\": {ex_idx}"]
                for key in required_short_keys:
                    if key in text_short_keys:
                        fields.append(f"\"{key}\": \"example\"")
                    else:
                        fields.append(f"\"{key}\": 3")
                example_items.append("{ " + ", ".join(fields) + " }")
            example_json = "[\n  " + ",\n  ".join(example_items) + "\n]"
            batch_prefix = (
                f"You will receive {batch_len} founder profiles in a JSON list. "
                "Score each founder independently; do not compare or rank founders. "
                "Return ONLY valid JSON. Do NOT include any surrounding text or markdown. "
                f"Return a JSON list of length {batch_len}. "
                "Each list item must be a JSON object with:\n"
                "  - index: integer (0-based within this batch)\n"
                f"  - keys: {keys_list}\n"
                "Every list item MUST include every key. "
                "If you are unsure, still provide a score within the allowed range.\n"
                "Example JSON (structure only; fill values for all items):\n"
                f"{example_json}\n"
                "If any experiment instructions show a single-object example, treat that as the per-founder object schema.\n\n"
            )
        else:
            founder_text = _format_record(batch_recs[0])
            batch_prefix = ""

        base_prompt = core_prompt.replace("{{EXPERIMENT_INSTRUCTIONS}}", experiment_instructions)
        base_prompt = base_prompt.replace("{{FOUNDER}}", founder_text)
        if batch_prefix:
            base_prompt = batch_prefix + base_prompt
        prompt = base_prompt

        parsed_items: dict[int, dict[str, Any]] = {}
        parsed_single: dict[str, Any] | None = None
        raw_text = ""
        batch_failures = 0
        if use_batch:
            skeleton = (
                "[\n"
                "  {\"index\": 0, "
                + ", ".join([f"\"{k}\": 1" for k in required_short_keys])
                + "}\n"
                "]"
            )
            strict_prefix = (
                "You MUST return ONLY valid JSON. "
                "Return a JSON list of length "
                + str(batch_len)
                + " with ALL required keys. "
                "Use this exact structure (fill values for each index):\n"
                + skeleton
                + "\n\n"
            )
        else:
            strict_prefix = (
                "You MUST return ONLY valid JSON with ALL required keys. "
                "Return only the JSON object.\n\n"
            )
        if strict_retry:
            prompt = strict_prefix + base_prompt

        if config.dry_run:
            if use_batch:
                mock_items: list[dict[str, Any]] = []
                for i in range(batch_len):
                    mock = {"index": i}
                    for key in feature_keys:
                        short_key = key_alias[key]
                        if key in text_keys:
                            mock[short_key] = f"dry_run: {short_key}"
                        else:
                            min_v, max_v = numeric_ranges.get(key, (1.0, 5.0))
                            mock[short_key] = float((min_v + max_v) / 2)
                    mock_items.append(mock)
                parsed_items = {item["index"]: item for item in mock_items}
            else:
                mock = {}
                for key in feature_keys:
                    short_key = key_alias[key]
                    if key in text_keys:
                        mock[short_key] = f"dry_run: {short_key}"
                    else:
                        min_v, max_v = numeric_ranges.get(key, (1.0, 5.0))
                        mock[short_key] = float((min_v + max_v) / 2)
                parsed_single = mock
            return batch_idx, batch_start, batch_recs, parsed_items, parsed_single, raw_text, batch_failures

        response = None
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                if use_openai and concurrency > 1:
                    client = _get_openai_client()
                    resp = client.responses.create(
                        model=config.model,
                        input=prompt,
                        temperature=0.0,
                    )
                    raw_text = resp.output_text
                    response = True
                elif use_google and concurrency > 1:
                    client = _get_google_client()
                    try:
                        from google.genai import types as gtypes
                        config_obj = gtypes.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                        )
                        resp = client.models.generate_content(
                            model=google_model,
                            contents=prompt,
                            config=config_obj,
                        )
                    except Exception:
                        resp = client.models.generate_content(
                            model=google_model,
                            contents=prompt,
                        )
                    raw_text = _google_output_text(resp)
                    response = True
                else:
                    resp = trl_llm.respond_sync(
                        llm_priority=[model_choice],
                        query=prompt,
                        response_format=str,
                        temperature=0.0,
                    )
                    raw_text = str(resp.response)
                    response = True
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                _append_jsonl(
                    error_log,
                    {
                        "index": batch_idx,
                        "error": str(exc),
                        "stage": "llm_call",
                        "attempt": attempt + 1,
                        "timestamp": time.time(),
                    },
                )
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if not response:
            raise last_exc if last_exc else RuntimeError("LLM call failed.")
        _accumulate_usage(prompt, raw_text)
        if use_batch:
            parsed_items = _parse_batch_response(raw_text, batch_len)
            invalid = False
            if len(parsed_items) != batch_len:
                invalid = True
            else:
                for i in range(batch_len):
                    item = parsed_items.get(i)
                    if not isinstance(item, dict):
                        invalid = True
                        break
                    _assert_no_label_fields(item, "LLM output")
                    missing = [k for k in required_short_keys if k not in item]
                    if missing:
                        invalid = True
                        break
            if invalid:
                batch_failures += 1
                _append_jsonl(
                    error_log,
                    {
                        "index": batch_idx,
                        "error": "Invalid batch response (missing keys or length mismatch)",
                        "stage": "initial_parse",
                        "timestamp": time.time(),
                    },
                )
                retry_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Return ONLY a JSON list of length "
                    + str(batch_len)
                    + ". Each item must include 'index' and ALL required keys: "
                    + ", ".join(required_short_keys)
                    + "."
                )
                if use_openai and concurrency > 1:
                    client = _get_openai_client()
                    resp = client.responses.create(
                        model=config.model,
                        input=retry_prompt,
                        temperature=0.0,
                    )
                    raw_text_retry = resp.output_text
                elif use_google and concurrency > 1:
                    client = _get_google_client()
                    try:
                        from google.genai import types as gtypes
                        config_obj = gtypes.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                        )
                        resp = client.models.generate_content(
                            model=google_model,
                            contents=retry_prompt,
                            config=config_obj,
                        )
                    except Exception:
                        resp = client.models.generate_content(
                            model=google_model,
                            contents=retry_prompt,
                        )
                    raw_text_retry = _google_output_text(resp)
                else:
                    resp = trl_llm.respond_sync(
                        llm_priority=[model_choice],
                        query=retry_prompt,
                        response_format=str,
                        temperature=0.0,
                    )
                    raw_text_retry = str(resp.response)
                _accumulate_usage(retry_prompt, raw_text_retry)
                raw_text = raw_text + "\n\n--- RETRY ---\n\n" + raw_text_retry
                parsed_items = _parse_batch_response(raw_text_retry, batch_len)
                if len(parsed_items) != batch_len:
                    skeleton = (
                        "[\n"
                        "  {\"index\": 0, "
                        + ", ".join([f"\"{k}\": 1" for k in required_short_keys])
                        + "}\n"
                        "]"
                    )
                    retry_prompt_2 = strict_prefix + base_prompt
                    if use_openai and concurrency > 1:
                        client = _get_openai_client()
                        resp = client.responses.create(
                            model=config.model,
                            input=retry_prompt_2,
                            temperature=0.0,
                        )
                        raw_text_retry2 = resp.output_text
                    elif use_google and concurrency > 1:
                        client = _get_google_client()
                        try:
                            from google.genai import types as gtypes
                            config_obj = gtypes.GenerateContentConfig(
                                temperature=0.0,
                                response_mime_type="application/json",
                            )
                            resp = client.models.generate_content(
                                model=google_model,
                                contents=retry_prompt_2,
                                config=config_obj,
                            )
                        except Exception:
                            resp = client.models.generate_content(
                                model=google_model,
                                contents=retry_prompt_2,
                            )
                        raw_text_retry2 = _google_output_text(resp)
                    else:
                        resp = trl_llm.respond_sync(
                            llm_priority=[model_choice],
                            query=retry_prompt_2,
                            response_format=str,
                            temperature=0.0,
                        )
                        raw_text_retry2 = str(resp.response)
                    _accumulate_usage(retry_prompt_2, raw_text_retry2)
                    raw_text = raw_text + "\n\n--- RETRY 2 ---\n\n" + raw_text_retry2
                    parsed_items = _parse_batch_response(raw_text_retry2, batch_len)
        else:
            try:
                parsed_single = json.loads(raw_text)
            except Exception as exc:
                batch_failures += 1
                _append_jsonl(
                    error_log,
                    {
                        "index": batch_idx,
                        "error": str(exc),
                        "stage": "initial_parse",
                        "timestamp": time.time(),
                    },
                )
                if use_openai and concurrency > 1:
                    client = _get_openai_client()
                    resp = client.responses.create(
                        model=config.model,
                        input=prompt,
                        temperature=0.0,
                    )
                    raw_text_retry = resp.output_text
                elif use_google and concurrency > 1:
                    client = _get_google_client()
                    try:
                        from google.genai import types as gtypes
                        config_obj = gtypes.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                        )
                        resp = client.models.generate_content(
                            model=google_model,
                            contents=prompt,
                            config=config_obj,
                        )
                    except Exception:
                        resp = client.models.generate_content(
                            model=google_model,
                            contents=prompt,
                        )
                    raw_text_retry = _google_output_text(resp)
                else:
                    resp = trl_llm.respond_sync(
                        llm_priority=[model_choice],
                        query=prompt,
                        response_format=str,
                        temperature=0.0,
                    )
                    raw_text_retry = str(resp.response)
                _accumulate_usage(prompt, raw_text_retry)
                raw_text = raw_text + "\n\n--- RETRY ---\n\n" + raw_text_retry
                try:
                    parsed_single = json.loads(raw_text_retry)
                except Exception:
                    parsed_single = None
                    batch_failures += 1
            if isinstance(parsed_single, dict):
                _assert_no_label_fields(parsed_single, "LLM output")
                missing = [k for k in required_short_keys if k not in parsed_single]
                if missing:
                    retry_prompt = (
                        prompt
                        + "\n\nIMPORTANT: You omitted required keys: "
                        + ", ".join(missing)
                        + ". Return ONLY JSON with ALL required keys."
                    )
                    if use_openai and concurrency > 1:
                        client = _get_openai_client()
                        resp = client.responses.create(
                            model=config.model,
                            input=retry_prompt,
                            temperature=0.0,
                        )
                        raw_text_retry = resp.output_text
                    elif use_google and concurrency > 1:
                        client = _get_google_client()
                        try:
                            from google.genai import types as gtypes
                            config_obj = gtypes.GenerateContentConfig(
                                temperature=0.0,
                                response_mime_type="application/json",
                            )
                            resp = client.models.generate_content(
                                model=google_model,
                                contents=retry_prompt,
                                config=config_obj,
                            )
                        except Exception:
                            resp = client.models.generate_content(
                                model=google_model,
                                contents=retry_prompt,
                            )
                        raw_text_retry = _google_output_text(resp)
                    else:
                        resp = trl_llm.respond_sync(
                            llm_priority=[model_choice],
                            query=retry_prompt,
                            response_format=str,
                            temperature=0.0,
                        )
                        raw_text_retry = str(resp.response)
                    _accumulate_usage(retry_prompt, raw_text_retry)
                    raw_text = raw_text + "\n\n--- RETRY ---\n\n" + raw_text_retry
                    try:
                        parsed_single = json.loads(raw_text_retry)
                    except Exception:
                        parsed_single = None
        return batch_idx, batch_start, batch_recs, parsed_items, parsed_single, raw_text, batch_failures

    def _rows_with_nan() -> list[int]:
        nan_rows: list[int] = []
        numeric_keys = [k for k in feature_keys if k not in text_keys]
        for i in range(total_records):
            for key in numeric_keys:
                val = features[key][i]
                if isinstance(val, float) and np.isnan(val):
                    nan_rows.append(i)
                    break
        return nan_rows

    numeric_keys = [k for k in feature_keys if k not in text_keys]

    def _batch_has_nan(batch_start: int, batch_len: int) -> bool:
        if not numeric_keys:
            return False
        end = batch_start + batch_len
        for i in range(batch_start, end):
            for key in numeric_keys:
                val = features[key][i]
                if isinstance(val, float) and np.isnan(val):
                    return True
        return False

    repair_only = existing_df_local is not None and (config.repair_existing or config.target_batch_indices)
    initial_nan_rows: list[int] = []
    batches: list[tuple[int, int, list[dict[str, Any]], bool]] = []
    target_batches = (
        sorted({int(b) for b in (config.target_batch_indices or []) if int(b) >= 0})
        if config.target_batch_indices
        else None
    )
    if target_batches:
        for batch_idx in target_batches:
            batch_start = batch_idx * batch_size
            if batch_start >= total_records:
                continue
            batch_end = min(total_records, batch_start + batch_size)
            batch_recs = selected_records[batch_start:batch_end]
            batches.append((batch_idx, batch_start, batch_recs, True))
            if progress_log is not None and (batch_idx % config.log_every == 0):
                _append_jsonl(
                    progress_log,
                    {
                        "index": batch_idx,
                        "stage": "repair_start",
                        "experiment": exp_label,
                        "processed": processed,
                        "total": total_records,
                        "batch_index": batch_idx,
                        "batch_start": batch_start,
                        "batch_end": batch_end - 1,
                        "batch_size": len(batch_recs),
                        "strict_retry": True,
                        "timestamp": time.time(),
                    },
                )
    else:
        if repair_only and config.repair_nan:
            initial_nan_rows = _rows_with_nan()
        if repair_only and config.repair_nan and initial_nan_rows:
            repair_batch_ids = sorted({i // batch_size for i in initial_nan_rows})
            for batch_idx in repair_batch_ids:
                batch_start = batch_idx * batch_size
                batch_end = min(total_records, batch_start + batch_size)
                batch_recs = selected_records[batch_start:batch_end]
                batches.append((batch_idx, batch_start, batch_recs, False))
                if progress_log is not None and (batch_idx % config.log_every == 0):
                    _append_jsonl(
                        progress_log,
                        {
                            "index": batch_idx,
                            "stage": "repair_start",
                            "experiment": exp_label,
                            "processed": processed,
                            "total": total_records,
                            "batch_index": batch_idx,
                            "batch_start": batch_start,
                            "batch_end": batch_end - 1,
                            "batch_size": len(batch_recs),
                            "timestamp": time.time(),
                        },
                    )
        else:
            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(total_records, batch_start + batch_size)
                batch_recs = selected_records[batch_start:batch_end]
                batches.append((batch_idx, batch_start, batch_recs, False))
                if progress_log is not None and (batch_idx % config.log_every == 0):
                    _append_jsonl(
                        progress_log,
                        {
                            "index": batch_idx,
                            "stage": "start_call",
                            "experiment": exp_label,
                            "processed": processed,
                            "total": total_records,
                            "batch_index": batch_idx,
                            "batch_start": batch_start,
                            "batch_end": batch_end - 1,
                            "batch_size": len(batch_recs),
                            "timestamp": time.time(),
                        },
                    )

    if config.max_batches and int(config.max_batches) > 0:
        batches = batches[: int(config.max_batches)]

    def _apply_result(
        batch_idx: int,
        batch_start: int,
        batch_recs: list[dict[str, Any]],
        parsed_items: dict[int, dict[str, Any]],
        parsed_single: dict[str, Any] | None,
    ) -> None:
        nonlocal failures
        for i, rec in enumerate(batch_recs):
            record_idx = batch_start + i
            item = parsed_items.get(i) if use_batch else parsed_single
            if isinstance(item, dict):
                _assert_no_label_fields(item, "LLM output")
                for key in feature_keys:
                    short_key = key_alias[key]
                    if key in text_keys:
                        if short_key in item:
                            features[key][record_idx] = _sanitize(str(item.get(short_key, "")))
                    else:
                        if short_key in item:
                            try:
                                val = float(item.get(short_key))
                            except Exception:
                                failures += 1
                                continue
                            min_v, max_v = numeric_ranges.get(key, (1.0, 5.0))
                            if val < min_v or val > max_v:
                                failures += 1
                                continue
                            features[key][record_idx] = val

    if concurrency > 1 and not config.dry_run:
        queue = list(batches)
        window_index = 0
        fallback_windows_remaining = 0
        fallback_level = -1
        while queue:
            if fallback_level >= 0:
                current_concurrency = min(concurrency, fallback_sequence[fallback_level])
            elif fallback_windows_remaining > 0:
                current_concurrency = min(concurrency, fallback_concurrency)
            else:
                current_concurrency = concurrency
            current_concurrency = max(1, int(current_concurrency))
            window = [queue.pop(0) for _ in range(min(current_concurrency, len(queue)))]
            batch_ids = [b[0] for b in window]
            if progress_log is not None:
                _append_jsonl(
                    progress_log,
                    {
                        "index": window_index,
                        "stage": "window_start",
                        "experiment": exp_label,
                        "window": window_index,
                        "concurrency": current_concurrency,
                        "batch_ids": batch_ids,
                        "timestamp": time.time(),
                    },
                )
            window_index += 1
            failed_batches: list[tuple[int, int, list[dict[str, Any]], bool]] = []
            rate_limited = False
            max_retry_delay = 0.0
            with ThreadPoolExecutor(max_workers=current_concurrency) as executor:
                future_map = {
                    executor.submit(_call_batch, batch_idx, batch_start, batch_recs, strict_retry): (batch_idx, batch_start, batch_recs, strict_retry)
                    for batch_idx, batch_start, batch_recs, strict_retry in window
                }
                for future in as_completed(future_map):
                    batch_idx, batch_start, batch_recs, strict_retry = future_map[future]
                    try:
                        (
                            _batch_idx,
                            _batch_start,
                            _batch_recs,
                            parsed_items,
                            parsed_single,
                            raw_text,
                            batch_failures,
                        ) = future.result()
                    except Exception as exc:
                        if _is_rate_limit_error(exc):
                            rate_limited = True
                            delay = _extract_retry_delay_seconds(str(exc))
                            if delay is not None:
                                max_retry_delay = max(max_retry_delay, delay)
                            rate_limit_error_count += 1
                            attempts = rate_limit_attempts.get(batch_idx, 0) + 1
                            rate_limit_attempts[batch_idx] = attempts
                            if max_rate_limit_errors and rate_limit_error_count >= max_rate_limit_errors:
                                raise RuntimeError(
                                    f"Max rate limit errors reached ({rate_limit_error_count})."
                                ) from exc
                            if max_rate_limit_retries and attempts > max_rate_limit_retries:
                                _append_jsonl(
                                    error_log,
                                    {
                                        "index": batch_idx,
                                        "error": str(exc),
                                        "stage": "rate_limit_exceeded",
                                        "attempt": attempts,
                                        "timestamp": time.time(),
                                    },
                                )
                                continue
                            failed_batches.append((batch_idx, batch_start, batch_recs, strict_retry))
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": str(exc),
                                    "stage": "rate_limit",
                                    "attempt": attempts,
                                    "timestamp": time.time(),
                                },
                            )
                            continue
                        if _is_retryable_error(exc):
                            rate_limited = True
                            delay = _extract_retry_delay_seconds(str(exc))
                            if delay is not None:
                                max_retry_delay = max(max_retry_delay, delay)
                            attempts = rate_limit_attempts.get(batch_idx, 0) + 1
                            rate_limit_attempts[batch_idx] = attempts
                            if max_rate_limit_retries and attempts > max_rate_limit_retries:
                                requeues = retryable_requeues.get(batch_idx, 0)
                                if requeues < max_retryable_requeues:
                                    retryable_requeues[batch_idx] = requeues + 1
                                    failed_batches.append((batch_idx, batch_start, batch_recs, True))
                                    _append_jsonl(
                                        error_log,
                                        {
                                            "index": batch_idx,
                                            "error": str(exc),
                                            "stage": "retryable_exceeded_requeue",
                                            "attempt": attempts,
                                            "requeue": requeues + 1,
                                            "timestamp": time.time(),
                                        },
                                    )
                                    continue
                                _append_jsonl(
                                    error_log,
                                    {
                                        "index": batch_idx,
                                        "error": str(exc),
                                        "stage": "retryable_exceeded_drop",
                                        "attempt": attempts,
                                        "requeue": requeues,
                                        "timestamp": time.time(),
                                    },
                                )
                                continue
                            failed_batches.append((batch_idx, batch_start, batch_recs, strict_retry))
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": str(exc),
                                    "stage": "retryable_error",
                                    "attempt": attempts,
                                    "timestamp": time.time(),
                                },
                            )
                            continue
                        _append_jsonl(
                            error_log,
                            {
                                "index": batch_idx,
                                "error": str(exc),
                                "stage": "batch_error",
                                "timestamp": time.time(),
                            },
                        )
                        continue
                    failures += batch_failures
                    _apply_result(batch_idx, batch_start, batch_recs, parsed_items, parsed_single)
                    if raw_text:
                        raw_responses.append((batch_idx, raw_text))
                    if config.inline_repair and _batch_has_nan(batch_start, len(batch_recs)):
                        attempts = inline_attempts.get(batch_idx, 0)
                        if attempts < max(1, int(config.inline_repair_max_attempts)):
                            inline_attempts[batch_idx] = attempts + 1
                            inline_repair_retries += 1
                            queue.insert(0, (batch_idx, batch_start, batch_recs, True))
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": "Inline repair requeue (NaNs detected)",
                                    "stage": "inline_repair",
                                    "attempt": attempts + 1,
                                    "strict_retry": True,
                                    "timestamp": time.time(),
                                },
                            )
                            continue
                        else:
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": "Inline repair max attempts reached",
                                    "stage": "inline_repair_failed",
                                    "attempt": attempts,
                                    "timestamp": time.time(),
                                },
                            )
                    completed_batches += 1
                    processed += len(batch_recs)
                    if progress_log is not None and (
                        (completed_batches % config.log_every == 0) or (processed == total_records)
                    ):
                        _append_jsonl(
                            progress_log,
                            {
                                "index": batch_idx,
                                "stage": "repair_done" if repair_only else "done",
                                "experiment": exp_label,
                                "processed": processed,
                                "total": total_records,
                                "batch_index": batch_idx,
                                "batch_start": batch_start,
                                "batch_end": batch_start + len(batch_recs) - 1,
                                "batch_size": len(batch_recs),
                                "timestamp": time.time(),
                            },
                        )
                    if not config.dry_run_fast:
                        time.sleep(0.05)
            if rate_limited:
                rate_limit_fallbacks += 1
                if fallback_level < len(fallback_sequence) - 1:
                    fallback_level += 1
                if fallback_windows > 0:
                    fallback_windows_remaining = fallback_windows
                sleep_seconds = 0.0
                if max_retry_delay > 0:
                    sleep_seconds = max(rate_limit_sleep_min, max_retry_delay)
                else:
                    sleep_seconds = max(rate_limit_sleep_min, 2 ** min(rate_limit_fallbacks, 5))
                if sleep_seconds > 0:
                    sleep_seconds = min(rate_limit_sleep_max, sleep_seconds)
                    if progress_log is not None:
                        _append_jsonl(
                            progress_log,
                            {
                                "index": window_index - 1,
                                "stage": "rate_limit_sleep",
                                "experiment": exp_label,
                                "sleep_seconds": sleep_seconds,
                                "timestamp": time.time(),
                            },
                        )
                    time.sleep(sleep_seconds)
                queue = failed_batches + queue
                if progress_log is not None:
                    _append_jsonl(
                        progress_log,
                        {
                            "index": window_index - 1,
                            "stage": "rate_limit_fallback",
                            "experiment": exp_label,
                            "concurrency_next": min(concurrency, fallback_sequence[fallback_level])
                            if fallback_level >= 0
                            else min(concurrency, fallback_concurrency),
                            "failed_batch_ids": [b[0] for b in failed_batches],
                            "timestamp": time.time(),
                        },
                    )
            else:
                if fallback_windows_remaining > 0:
                    fallback_windows_remaining -= 1
                if fallback_level >= 0:
                    fallback_level = -1
    else:
        queue = list(batches)
        while queue:
            batch_idx, batch_start, batch_recs, strict_retry = queue.pop(0)
            skip_batch = False
            attempts = 0
            while True:
                try:
                    (
                        _batch_idx,
                        _batch_start,
                        _batch_recs,
                        parsed_items,
                        parsed_single,
                        raw_text,
                        batch_failures,
                    ) = _call_batch(batch_idx, batch_start, batch_recs, strict_retry)
                    break
                except Exception as exc:
                    if _is_rate_limit_error(exc):
                        attempts += 1
                        rate_limit_error_count += 1
                        delay = _extract_retry_delay_seconds(str(exc))
                        if max_rate_limit_errors and rate_limit_error_count >= max_rate_limit_errors:
                            raise RuntimeError(
                                f"Max rate limit errors reached ({rate_limit_error_count})."
                            ) from exc
                        if max_rate_limit_retries and attempts > max_rate_limit_retries:
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": str(exc),
                                    "stage": "rate_limit_exceeded",
                                    "attempt": attempts,
                                    "timestamp": time.time(),
                                },
                            )
                            skip_batch = True
                            break
                        sleep_seconds = 0.0
                        if delay is not None and delay > 0:
                            sleep_seconds = max(rate_limit_sleep_min, delay)
                        else:
                            sleep_seconds = max(rate_limit_sleep_min, 2 ** min(attempts, 5))
                        if sleep_seconds > 0:
                            sleep_seconds = min(rate_limit_sleep_max, sleep_seconds)
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": str(exc),
                                    "stage": "rate_limit_sleep",
                                    "sleep_seconds": sleep_seconds,
                                    "timestamp": time.time(),
                                },
                            )
                            time.sleep(sleep_seconds)
                        continue
                    if _is_retryable_error(exc):
                        attempts += 1
                        delay = _extract_retry_delay_seconds(str(exc))
                        if max_rate_limit_retries and attempts > max_rate_limit_retries:
                            requeues = retryable_requeues.get(batch_idx, 0)
                            if requeues < max_retryable_requeues:
                                retryable_requeues[batch_idx] = requeues + 1
                                _append_jsonl(
                                    error_log,
                                    {
                                        "index": batch_idx,
                                        "error": str(exc),
                                        "stage": "retryable_exceeded_requeue",
                                        "attempt": attempts,
                                        "requeue": requeues + 1,
                                        "timestamp": time.time(),
                                    },
                                )
                                queue.append((batch_idx, batch_start, batch_recs, True))
                                skip_batch = True
                                break
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": str(exc),
                                    "stage": "retryable_exceeded_drop",
                                    "attempt": attempts,
                                    "requeue": requeues,
                                    "timestamp": time.time(),
                                },
                            )
                            skip_batch = True
                            break
                        sleep_seconds = 0.0
                        if delay is not None and delay > 0:
                            sleep_seconds = max(rate_limit_sleep_min, delay)
                        else:
                            sleep_seconds = max(rate_limit_sleep_min, 2 ** min(attempts, 5))
                        if sleep_seconds > 0:
                            sleep_seconds = min(rate_limit_sleep_max, sleep_seconds)
                            _append_jsonl(
                                error_log,
                                {
                                    "index": batch_idx,
                                    "error": str(exc),
                                    "stage": "retryable_sleep",
                                    "sleep_seconds": sleep_seconds,
                                    "timestamp": time.time(),
                                },
                            )
                            time.sleep(sleep_seconds)
                        continue
                    _append_jsonl(
                        error_log,
                        {
                            "index": batch_idx,
                            "error": str(exc),
                            "stage": "batch_error",
                            "timestamp": time.time(),
                        },
                    )
                    skip_batch = True
                    break
            if skip_batch:
                continue
            failures += batch_failures
            _apply_result(batch_idx, batch_start, batch_recs, parsed_items, parsed_single)
            if raw_text:
                raw_responses.append((batch_idx, raw_text))
            if config.inline_repair and _batch_has_nan(batch_start, len(batch_recs)):
                attempts = inline_attempts.get(batch_idx, 0)
                if attempts < max(1, int(config.inline_repair_max_attempts)):
                    inline_attempts[batch_idx] = attempts + 1
                    inline_repair_retries += 1
                    _append_jsonl(
                        error_log,
                        {
                            "index": batch_idx,
                            "error": "Inline repair requeue (NaNs detected)",
                            "stage": "inline_repair",
                            "attempt": attempts + 1,
                            "strict_retry": True,
                            "timestamp": time.time(),
                        },
                    )
                    # Re-run this batch immediately in sequential mode.
                    queue.insert(0, (batch_idx, batch_start, batch_recs, True))
                    continue
                else:
                    _append_jsonl(
                        error_log,
                        {
                            "index": batch_idx,
                            "error": "Inline repair max attempts reached",
                            "stage": "inline_repair_failed",
                            "attempt": attempts,
                            "timestamp": time.time(),
                        },
                    )
            completed_batches += 1
            processed += len(batch_recs)
            if progress_log is not None and (
                (completed_batches % config.log_every == 0) or (processed == total_records)
            ):
                _append_jsonl(
                    progress_log,
                    {
                        "index": batch_idx,
                        "stage": "repair_done" if repair_only else "done",
                        "experiment": exp_label,
                        "processed": processed,
                        "total": total_records,
                        "batch_index": batch_idx,
                        "batch_start": batch_start,
                        "batch_end": batch_start + len(batch_recs) - 1,
                        "batch_size": len(batch_recs),
                        "timestamp": time.time(),
                    },
                )
            if not config.dry_run_fast:
                time.sleep(0.05)

    df = pd.DataFrame(features)
    df.insert(0, "founder_uuid", [r.get("founder_uuid") for r in selected_records])
    df.insert(1, "success", selected_labels)
    # Replicate global evidence_support_rating per experiment for clarity.
    if "evidence_support_rating" in df.columns:
        for exp_id in exp_to_keys.keys():
            col_name = f"{exp_id}_evidence_support_rating"
            df[col_name] = df["evidence_support_rating"]
            if col_name not in exp_to_keys[exp_id]:
                exp_to_keys[exp_id].append(col_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"llm_reasoning_{dataset_size}.parquet"
    df.to_parquet(out_path, index=False)

    nan_rows_after = _rows_with_nan() if config.repair_nan else []

    # Per-experiment outputs
    write_per_experiment_parquets(
        df,
        exp_to_keys,
        output_dir,
        dataset_size,
        overwrite=True,
    )

    exp_root = output_dir / "experiments"
    metadata = {
        "model": config.model,
        "dataset_size": dataset_size,
        "random_state": config.random_state,
        "core_prompt_path": str(config.core_prompt_path),
        "experiments_path": str(config.experiments_path),
        "experiments": [e.get("id") for e in experiments],
        "output_parquet": str(out_path),
        "experiment_parquets_root": str(exp_root),
        "n_records": len(selected_records),
        "batch_size": batch_size,
        "n_batches": int(np.ceil(len(selected_records) / batch_size)),
        "concurrency": concurrency,
        "rate_limit_fallback_concurrency": fallback_concurrency,
        "rate_limit_fallback_windows": fallback_windows,
        "rate_limit_fallbacks": rate_limit_fallbacks,
        "rate_limit_error_count": rate_limit_error_count,
        "inline_repair": config.inline_repair,
        "inline_repair_max_attempts": config.inline_repair_max_attempts,
        "inline_repair_retries": inline_repair_retries,
        "repair_only": bool(config.target_batch_indices),
        "target_batch_count": len(config.target_batch_indices or []),
        "validation_failures": failures,
        "repair_existing": config.repair_existing,
        "repair_nan": config.repair_nan,
        "nan_rows_before": len(initial_nan_rows),
        "nan_rows_after": len(nan_rows_after),
        "dry_run": config.dry_run,
        "dry_run_fast": config.dry_run_fast,
        "estimated_prompt_tokens": usage["prompt_tokens_est"],
        "estimated_completion_tokens": usage["completion_tokens_est"],
        "estimated_prompt_chars": usage["prompt_chars"],
        "estimated_completion_chars": usage["completion_chars"],
        "estimated_requests": usage["requests"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest = {
        "dataset_size": dataset_size,
        "random_state": config.random_state,
        "model": config.model,
        "providers": config.providers,
        "google_model": config.google_model,
        "experiments": [e.get("id") for e in experiments],
        "core_prompt_hash": _hash_text(core_prompt),
        "experiments_hash": _hash_text(json.dumps(experiments, sort_keys=True)),
        "global_numeric_keys": GLOBAL_NUMERIC_KEYS,
        "global_text_keys": GLOBAL_TEXT_KEYS,
        "output_parquet": str(out_path),
        "experiment_parquets_root": str(exp_root),
    }
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        pass

    if raw_responses:
        raw_path = metadata_path.with_name(metadata_path.stem + "_raw_responses.txt")
        raw_responses_sorted = [t[1] for t in sorted(raw_responses, key=lambda x: x[0])]
        raw_path.write_text("\n\n---\n\n".join(raw_responses_sorted), encoding="utf-8")

    if summary_path is not None:
        summary = {
            "status": "completed",
            "experiment": exp_label,
            "total_records": total_records,
            "processed": processed,
            "validation_failures": failures,
            "dataset_size": dataset_size,
            "nan_rows_before": len(initial_nan_rows),
            "nan_rows_after": len(nan_rows_after),
            "rate_limit_fallbacks": rate_limit_fallbacks,
            "rate_limit_error_count": rate_limit_error_count,
            "inline_repair_retries": inline_repair_retries,
            "repair_only": bool(config.target_batch_indices),
            "estimated_prompt_tokens": usage["prompt_tokens_est"],
            "estimated_completion_tokens": usage["completion_tokens_est"],
            "estimated_prompt_chars": usage["prompt_chars"],
            "estimated_completion_chars": usage["completion_chars"],
            "estimated_requests": usage["requests"],
            "timestamp": time.time(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return df, numeric_feature_keys
