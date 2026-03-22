from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Schema and dry-run tests for LLM reasoning (no API calls)."""


import json
from pathlib import Path
import shutil

import numpy as np

from lib.llm_reasoning_features import (
    ReasoningConfig,
    _load_core_prompt,
    _load_experiments,
    _validate_experiments,
    generate_reasoning_features,
)
from lib.paths import BASE_DIR


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _tmp_dir(name: str) -> Path:
    base = BASE_DIR / ".tmp_tests"
    base.mkdir(parents=True, exist_ok=True)
    d = base / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_core_prompt_placeholder_validation() -> None:
    td = _tmp_dir("core_prompt")
    p = td / "core.txt"
    _write(p, "missing placeholders")
    try:
        _load_core_prompt(p)
        raise AssertionError("Expected ValueError for missing placeholders.")
    except ValueError:
        pass
    shutil.rmtree(td, ignore_errors=True)


def test_experiments_validation() -> None:
    td = _tmp_dir("experiments")
    p = td / "exps.json"
    _write(p, "[]")
    try:
        _load_experiments(p)
        raise AssertionError("Expected ValueError for empty experiments.")
    except ValueError:
        pass

    _write(
        p,
        json.dumps(
            [{"id": "A", "instructions": "x", "numeric_keys": [], "text_keys": []}]
        ),
    )
    exps = _load_experiments(p)
    try:
        _validate_experiments(exps)
        raise AssertionError("Expected ValueError for missing keys.")
    except ValueError:
        pass
    shutil.rmtree(td, ignore_errors=True)


def test_dry_run_generation() -> None:
    td = _tmp_dir("dry_run")
    core = td / "core.txt"
    exps = td / "exps.json"
    _write(core, "Core {{EXPERIMENT_INSTRUCTIONS}}\nFounder: {{FOUNDER}}\nOutput ONLY valid JSON.")
    _write(
        exps,
        json.dumps(
            [
                {
                    "id": "A",
                    "instructions": "Score A",
                    "numeric_keys": ["trajectory_strength"],
                    "text_keys": ["trajectory_strength_justification"],
                }
            ]
        ),
    )
    records = [{"founder_uuid": "f1", "foo": "bar"}]
    labels = np.array([1])
    cfg = ReasoningConfig(
        model="gpt-4.1-nano",
        dataset_size="full",
        random_state=42,
        core_prompt_path=core,
        experiments_path=exps,
        providers={"openai": False, "google": False},
        dry_run=True,
    )
    df, numeric_keys = generate_reasoning_features(
        records=records,
        labels=labels,
        config=cfg,
        output_dir=td,
        metadata_path=td / "meta.json",
    )
    assert len(df) == 1
    assert "evidence_support_rating" in df.columns
    assert "A_trajectory_strength" in df.columns
    assert "A_trajectory_strength_justification" in df.columns
    assert numeric_keys == ["evidence_support_rating", "A_trajectory_strength"]
    per_exp = td / "experiments" / "A" / "llm_reasoning_full.parquet"
    assert per_exp.exists()
    shutil.rmtree(td, ignore_errors=True)


def test_dry_run_fast_limits_rows() -> None:
    td = _tmp_dir("dry_run_fast")
    core = td / "core.txt"
    exps = td / "exps.json"
    _write(core, "Core {{EXPERIMENT_INSTRUCTIONS}}\nFounder: {{FOUNDER}}\nOutput ONLY valid JSON.")
    _write(
        exps,
        json.dumps(
            [
                {
                    "id": "A",
                    "instructions": "Score A",
                    "numeric_keys": ["trajectory_strength"],
                    "text_keys": [],
                }
            ]
        ),
    )
    records = [{"founder_uuid": f"f{i}", "foo": "bar"} for i in range(60)]
    labels = np.array([1] * 60)
    cfg = ReasoningConfig(
        model="gpt-4.1-nano",
        dataset_size="full",
        random_state=42,
        core_prompt_path=core,
        experiments_path=exps,
        providers={"openai": False, "google": False},
        dry_run=True,
        dry_run_fast=True,
    )
    df, _ = generate_reasoning_features(
        records=records,
        labels=labels,
        config=cfg,
        output_dir=td,
        metadata_path=td / "meta.json",
    )
    assert len(df) <= 50
    shutil.rmtree(td, ignore_errors=True)


def test_label_field_rejection() -> None:
    from lib.llm_reasoning_features import _assert_no_label_fields

    try:
        _assert_no_label_fields({"success": 1}, "record")
        raise AssertionError("Expected RuntimeError for success field.")
    except RuntimeError:
        pass

    try:
        _assert_no_label_fields({"label": 0}, "output")
        raise AssertionError("Expected RuntimeError for label field.")
    except RuntimeError:
        pass


def test_record_with_success_raises() -> None:
    td = _tmp_dir("label_in_record")
    core = td / "core.txt"
    exps = td / "exps.json"
    _write(core, "Core {{EXPERIMENT_INSTRUCTIONS}}\nFounder: {{FOUNDER}}\nOutput ONLY valid JSON.")
    _write(
        exps,
        json.dumps(
            [
                {
                    "id": "A",
                    "instructions": "Score A",
                    "numeric_keys": ["trajectory_strength"],
                    "text_keys": [],
                }
            ]
        ),
    )
    records = [{"founder_uuid": "f1", "success": 1}]
    labels = np.array([1])
    cfg = ReasoningConfig(
        model="gpt-4.1-nano",
        dataset_size="full",
        random_state=42,
        core_prompt_path=core,
        experiments_path=exps,
        providers={"openai": False, "google": False},
        dry_run=True,
    )
    try:
        generate_reasoning_features(
            records=records,
            labels=labels,
            config=cfg,
            output_dir=td,
            metadata_path=td / "meta.json",
        )
        raise AssertionError("Expected RuntimeError for success in record.")
    except RuntimeError:
        pass
    shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    test_core_prompt_placeholder_validation()
    test_experiments_validation()
    test_dry_run_generation()
    test_dry_run_fast_limits_rows()
    test_label_field_rejection()
    test_record_with_success_raises()
    print("Reasoning schema tests passed.")
