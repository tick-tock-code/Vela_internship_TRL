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

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.core.llms import llm as trl_llm


@dataclass
class ReasoningConfig:
    model: str
    dataset_size: str  # "full", "200", "400", "1000"
    random_state: int
    prompts_path: Path


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
    model_choice = OpenAIChoice(model=config.model)

    for rec in selected_records:
        rec_text = _format_record(rec)
        for key, prompt in prompts.items():
            min_v, max_v = _expected_range(key)
            full_prompt = (
                f"{prompt}\n"
                f"Return a single numeric value between {min_v} and {max_v}.\n"
                f"Founder record:\n{rec_text}"
            )
            response = trl_llm.respond(
                full_prompt,
                choice=model_choice,
                temperature=0.0,
            )
            value = _parse_numeric(_sanitize(str(response)))
            features[key].append(value)
        time.sleep(0.05)

    df = pd.DataFrame(features)
    df.insert(0, "founder_uuid", [r.get("founder_uuid") for r in selected_records])
    df.insert(1, "success", selected_labels)

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
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return df, list(prompts.keys())
