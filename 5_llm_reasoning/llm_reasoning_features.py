"""LLM reasoning feature generation for VCBench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice
from think_reason_learn.core.llms import llm as trl_llm


@dataclass
class ReasoningConfig:
    model: str
    dataset_size: str  # "full", "200", "400", "1000"
    random_state: int
    prompts_path: Path
    providers: dict[str, bool]
    google_model: str | None = None
    batch_size: int = 20


def _load_prompts(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Prompts JSON must be a non-empty dict.")
    return {str(k): str(v) for k, v in data.items()}


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


def _expected_range(prompt_key: str) -> tuple[float, float]:
    if prompt_key.endswith("_rating"):
        return (1.0, 5.0)
    return (0.0, 1.0)


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


def generate_reasoning_features(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    config: ReasoningConfig,
    output_dir: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    prompts = _load_prompts(config.prompts_path)
    selected_records, selected_labels = _select_records(
        records, labels, config.dataset_size, config.random_state
    )

    features = {k: [] for k in prompts.keys()}
    failures = 0
    if config.providers.get("openai", False):
        model_choice = OpenAIChoice(model=config.model)
    elif config.providers.get("google", False):
        model_choice = GoogleChoice(model=config.google_model or "gemini-2.0-flash")
    else:
        raise RuntimeError("No LLM providers enabled for reasoning features.")

    batch_size = max(1, int(config.batch_size))
    rng = np.random.RandomState(config.random_state)
    order = rng.permutation(len(selected_records))
    ordered_records = [selected_records[i] for i in order]
    ordered_labels = selected_labels[order]

    for start in range(0, len(ordered_records), batch_size):
        batch = ordered_records[start : start + batch_size]
        batch_texts = "\n".join(
            [f"Index {i}: {_format_record(rec)}" for i, rec in enumerate(batch)]
        )
        for key, prompt in prompts.items():
            min_v, max_v = _expected_range(key)
            full_prompt = (
                f"{prompt}\n"
                f"Batch size: {len(batch)}. Return JSON list with entries for indices 0..{len(batch)-1}.\n"
                f"Founder batch:\n{batch_texts}"
            )
            response = trl_llm.respond(
                full_prompt,
                choice=model_choice,
                temperature=0.0,
            )
            parsed = _parse_json_scores(_sanitize(str(response)))
            if len(parsed) != len(batch):
                failures += 1
                # One retry with sanitized prompt
                response = trl_llm.respond(
                    full_prompt,
                    choice=model_choice,
                    temperature=0.0,
                )
                parsed = _parse_json_scores(_sanitize(str(response)))
                if len(parsed) != len(batch):
                    parsed = []
            scores = [float('nan')] * len(batch)
            for item in parsed:
                idx = item["index"]
                score = item["score"]
                if 0 <= idx < len(batch):
                    if score < min_v or score > max_v:
                        failures += 1
                        continue
                    scores[idx] = score
            if any(np.isnan(scores)):
                failures += 1
            features[key].extend(scores)
        time.sleep(0.05)

    # Restore original ordering
    df = pd.DataFrame(features)
    df.insert(0, "founder_uuid", [r.get("founder_uuid") for r in ordered_records])
    df.insert(1, "success", ordered_labels)
    df = df.iloc[np.argsort(order)].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"llm_reasoning_{config.dataset_size}.parquet"
    df.to_parquet(out_path, index=False)

    metadata = {
        "model": config.model,
        "dataset_size": config.dataset_size,
        "random_state": config.random_state,
        "prompts_path": str(config.prompts_path),
        "prompt_keys": list(prompts.keys()),
        "output_parquet": str(out_path),
        "n_records": len(selected_records),
        "batch_size": batch_size,
        "n_batches": int(np.ceil(len(selected_records) / batch_size)),
        "validation_failures": failures,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return df, list(prompts.keys())
