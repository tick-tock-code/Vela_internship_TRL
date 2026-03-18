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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

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
        if os.getenv("OPENAI_API_KEY"):
            trl_config.settings.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        if os.getenv("GOOGLE_AI_API_KEY"):
            trl_config.settings.GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
        if os.getenv("ANTHROPIC_API_KEY"):
            trl_config.settings.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        if os.getenv("XAI_API_KEY"):
            trl_config.settings.XAI_API_KEY = os.getenv("XAI_API_KEY", "")
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
        client = OpenAI(api_key=api_key)
        _THREAD_LOCAL.openai_client = client
    return client


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


def _write_per_experiment_parquets(
    df: pd.DataFrame,
    exp_to_keys: dict[str, list[str]],
    output_dir: Path,
    dataset_size: str,
    overwrite: bool,
) -> None:
    exp_root = output_dir / "experiments"
    for exp_id, exp_keys in exp_to_keys.items():
        exp_cols = ["founder_uuid", "success"] + GLOBAL_NUMERIC_KEYS + GLOBAL_TEXT_KEYS + exp_keys
        exp_df = df[exp_cols].copy()
        exp_dir = exp_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_path = exp_dir / f"llm_reasoning_{dataset_size}.parquet"
        if exp_path.exists() and not overwrite:
            continue
        exp_df.to_parquet(exp_path, index=False)


def generate_reasoning_features(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    config: ReasoningConfig,
    output_dir: Path,
    metadata_path: Path,
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
    existing_df: pd.DataFrame | None = None
    if not config.dry_run and not config.dry_run_fast and not in_preview_dir and combined_path.exists():
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
                    _write_per_experiment_parquets(
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
                existing_df = df.copy()

    features: dict[str, list[Any]] = {}
    if existing_df is not None and len(existing_df) == total_records:
        for key in feature_keys:
            if key in existing_df.columns:
                features[key] = existing_df[key].tolist()
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

    required_short_keys = [_short_key(k) for k in feature_keys]
    text_short_keys = {_short_key(k) for k in text_keys}
    experiment_instructions = "\n\n".join([str(exp.get("instructions", "")) for exp in experiments])
    use_batch = batch_size > 1
    concurrency = max(1, int(config.concurrency))
    if use_google and concurrency > 1:
        # Google client is managed by TRL and not thread-safe in this pipeline.
        concurrency = 1

    def _append_jsonl(path: Path | None, entry: dict[str, Any]) -> None:
        if path is None:
            return
        with log_lock:
            path.open("a", encoding="utf-8").write(json.dumps(entry) + "\n")

    def _call_batch(
        batch_idx: int,
        batch_start: int,
        batch_recs: list[dict[str, Any]],
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

        prompt = core_prompt.replace("{{EXPERIMENT_INSTRUCTIONS}}", experiment_instructions)
        prompt = prompt.replace("{{FOUNDER}}", founder_text)
        if batch_prefix:
            prompt = batch_prefix + prompt

        parsed_items: dict[int, dict[str, Any]] = {}
        parsed_single: dict[str, Any] | None = None
        raw_text = ""
        batch_failures = 0

        if config.dry_run:
            if use_batch:
                mock_items: list[dict[str, Any]] = []
                for i in range(batch_len):
                    mock = {"index": i}
                    for key in feature_keys:
                        short_key = _short_key(key)
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
                    short_key = _short_key(key)
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
                else:
                    resp = trl_llm.respond_sync(
                        llm_priority=[model_choice],
                        query=retry_prompt,
                        response_format=str,
                        temperature=0.0,
                    )
                    raw_text_retry = str(resp.response)
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
                    retry_prompt_2 = (
                        "You MUST return ONLY valid JSON. Use this exact structure (fill values for each index):\n"
                        + skeleton
                    )
                    if use_openai and concurrency > 1:
                        client = _get_openai_client()
                        resp = client.responses.create(
                            model=config.model,
                            input=retry_prompt_2,
                            temperature=0.0,
                        )
                        raw_text_retry2 = resp.output_text
                    else:
                        resp = trl_llm.respond_sync(
                            llm_priority=[model_choice],
                            query=retry_prompt_2,
                            response_format=str,
                            temperature=0.0,
                        )
                        raw_text_retry2 = str(resp.response)
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
                else:
                    resp = trl_llm.respond_sync(
                        llm_priority=[model_choice],
                        query=prompt,
                        response_format=str,
                        temperature=0.0,
                    )
                    raw_text_retry = str(resp.response)
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
                    else:
                        resp = trl_llm.respond_sync(
                            llm_priority=[model_choice],
                            query=retry_prompt,
                            response_format=str,
                            temperature=0.0,
                        )
                        raw_text_retry = str(resp.response)
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

    repair_only = existing_df is not None and config.repair_existing
    initial_nan_rows: list[int] = _rows_with_nan() if repair_only else []
    batches: list[tuple[int, int, list[dict[str, Any]]]] = []
    if repair_only and config.repair_nan and initial_nan_rows:
        repair_batch_ids = sorted({i // batch_size for i in initial_nan_rows})
        for batch_idx in repair_batch_ids:
            batch_start = batch_idx * batch_size
            batch_end = min(total_records, batch_start + batch_size)
            batch_recs = selected_records[batch_start:batch_end]
            batches.append((batch_idx, batch_start, batch_recs))
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
            batches.append((batch_idx, batch_start, batch_recs))
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
                    short_key = _short_key(key)
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
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(_call_batch, batch_idx, batch_start, batch_recs): (batch_idx, batch_start, batch_recs)
                for batch_idx, batch_start, batch_recs in batches
            }
            for future in as_completed(future_map):
                batch_idx, batch_start, batch_recs = future_map[future]
                (
                    _batch_idx,
                    _batch_start,
                    _batch_recs,
                    parsed_items,
                    parsed_single,
                    raw_text,
                    batch_failures,
                ) = future.result()
                failures += batch_failures
                _apply_result(batch_idx, batch_start, batch_recs, parsed_items, parsed_single)
                if raw_text:
                    raw_responses.append((batch_idx, raw_text))
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
    else:
        for batch_idx, batch_start, batch_recs in batches:
            (
                _batch_idx,
                _batch_start,
                _batch_recs,
                parsed_items,
                parsed_single,
                raw_text,
                batch_failures,
            ) = _call_batch(batch_idx, batch_start, batch_recs)
            failures += batch_failures
            _apply_result(batch_idx, batch_start, batch_recs, parsed_items, parsed_single)
            if raw_text:
                raw_responses.append((batch_idx, raw_text))
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

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"llm_reasoning_{dataset_size}.parquet"
    df.to_parquet(out_path, index=False)

    nan_rows_after = _rows_with_nan() if config.repair_nan else []

    # Per-experiment outputs
    _write_per_experiment_parquets(
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
        "validation_failures": failures,
        "repair_existing": config.repair_existing,
        "repair_nan": config.repair_nan,
        "nan_rows_before": len(initial_nan_rows),
        "nan_rows_after": len(nan_rows_after),
        "dry_run": config.dry_run,
        "dry_run_fast": config.dry_run_fast,
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
            "timestamp": time.time(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return df, numeric_feature_keys
