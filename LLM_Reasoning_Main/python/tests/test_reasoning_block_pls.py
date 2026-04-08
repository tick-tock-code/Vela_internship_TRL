from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.reasoning_block_pls import (
    fit_reasoning_block_pls,
    load_reasoning_block_pls_config,
    run_reasoning_block_pls_study,
)
from pipelines.instability_control import reasoning_block_pls as reasoning_block_pls_pipeline


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_temp_dir() -> Path:
    tmp_path = Path(__file__).resolve().parents[2] / ".tmp_tests" / f"reasoning_block_pls_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _build_fixture_config(tmp_path: Path) -> tuple[Path, Path]:
    baseline_path = tmp_path / "baseline.csv"
    reasoning_a_path = tmp_path / "reasoning_a.csv"
    reasoning_f_path = tmp_path / "reasoning_f.csv"
    folds_path = tmp_path / "folds.json"
    families_path = tmp_path / "families.json"
    pls_path = tmp_path / "reasoning_block_pls.json"

    rows = []
    for idx in range(48):
        success = 1 if idx % 4 in {1, 2} else 0
        rows.append(
            {
                "founder_uuid": f"id_{idx}",
                "success": success,
                "exit_count": 0,
                "base_signal": 0.85 if success else 0.15,
                "base_noise": ((idx * 3) % 17) / 17.0,
            }
        )
    baseline_df = pd.DataFrame(rows)
    reasoning_a_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "A_one": [0.80 if value == 1 else 0.20 for value in baseline_df["success"]],
            "A_two": [0.72 if value == 1 else 0.28 for value in baseline_df["success"]],
            "A_three": [((idx * 5) % 19) / 19.0 for idx in range(len(baseline_df))],
        }
    )
    reasoning_f_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "F_one": [0.92 if value == 1 else 0.08 for value in baseline_df["success"]],
            "F_two": [0.88 if value == 1 else 0.12 for value in baseline_df["success"]],
            "F_three": [0.84 if value == 1 else 0.16 for value in baseline_df["success"]],
        }
    )
    baseline_df.to_csv(baseline_path, index=False)
    reasoning_a_df.to_csv(reasoning_a_path, index=False)
    reasoning_f_df.to_csv(reasoning_f_path, index=False)
    _write_json(
        folds_path,
        {
            "fold_map": {f"id_{idx}": idx % 3 for idx in range(len(baseline_df))},
            "k": 3,
            "random_state": 42,
        },
    )
    _write_json(
        families_path,
        {
            "dataset": {
                "id_column": "founder_uuid",
                "label_column": "success",
                "folds_path": str(folds_path),
            },
            "baseline": {
                "id": "HQ_anchor_xgb1_unpruned",
                "label": "HQ",
                "path": str(baseline_path),
            },
            "families": [
                {
                    "id": "reasoning_A",
                    "label": "A",
                    "path": str(reasoning_a_path),
                    "feature_prefixes": ["A_"],
                },
                {
                    "id": "reasoning_F",
                    "label": "F",
                    "path": str(reasoning_f_path),
                    "feature_prefixes": ["F_"],
                },
            ],
        },
    )
    _write_json(
        pls_path,
        {
            "family_ids": ["reasoning_A", "reasoning_F"],
            "outer_cv": {"n_splits": 3, "n_repeats": 2, "random_state": 42},
            "evaluation": {
                "models": ["logistic"],
                "apply_rule_override": True,
                "threshold_grid": "default",
                "inner_threshold_cv_splits": 2,
                "random_state": 42,
            },
            "family_components": {
                "reasoning_A": [2],
                "reasoning_F": [2],
            },
            "docs_output": {
                "protocol_markdown": str(tmp_path / "step_3_PLS_reasoning" / "10_protocol.md"),
                "summary_markdown": str(tmp_path / "step_3_PLS_reasoning" / "11_summary.md"),
                "details_markdown": str(tmp_path / "step_3_PLS_reasoning" / "12_details.md"),
                "audit_markdown": str(tmp_path / "step_3_PLS_reasoning" / "13_audit.md"),
                "summary_csv": str(tmp_path / "step_3_PLS_reasoning" / "summary.csv"),
                "fold_metrics_csv": str(tmp_path / "step_3_PLS_reasoning" / "fold_metrics.csv"),
                "component_summary_csv": str(tmp_path / "step_3_PLS_reasoning" / "component_summary.csv"),
            },
        },
    )
    return families_path, pls_path


def test_reasoning_block_pls_config_has_expected_grids() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "instability_control" / "reasoning_block_pls.json"
    config = load_reasoning_block_pls_config(config_path)
    assert config["outer_cv"]["n_splits"] == 3
    assert config["outer_cv"]["n_repeats"] == 16
    assert config["family_components"]["reasoning_A"] == [2, 3]
    assert config["family_components"]["reasoning_E"] == [1]
    assert config["evaluation"]["models"] == ["logistic", "xgb1"]


def test_fit_reasoning_block_pls_uses_train_only_statistics_and_expected_width() -> None:
    baseline_train = pd.DataFrame({"hq_a": [1.0, 2.0, 3.0], "hq_b": [0.2, 0.3, 0.4]})
    baseline_test = pd.DataFrame({"hq_a": [10.0, 20.0], "hq_b": [0.8, 0.9]})
    reasoning_train = pd.DataFrame({"A_1": [1.0, 2.0, 3.0], "A_2": [4.0, 5.0, 6.0]})
    reasoning_test = pd.DataFrame({"A_1": [100.0, 200.0], "A_2": [400.0, 500.0]})
    y_train = np.array([0, 1, 1], dtype=int)

    result = fit_reasoning_block_pls(
        baseline_train,
        baseline_test,
        reasoning_train,
        reasoning_test,
        y_train,
        family_label="A",
        requested_n_components=1,
    )

    assert result.actual_n_components == 1
    assert result.component_names == ["A_pls_1"]
    assert result.train_df.shape[1] == baseline_train.shape[1] + 1
    assert result.test_df.shape[1] == baseline_test.shape[1] + 1
    assert float(result.reasoning_train_means["A_1"]) == float(reasoning_train["A_1"].mean())
    assert float(result.reasoning_train_means["A_1"]) != float(pd.concat([reasoning_train, reasoning_test])["A_1"].mean())


def test_reasoning_block_pls_study_and_pipeline_write_expected_outputs(monkeypatch) -> None:
    tmp_path = _base_temp_dir()
    families_path, pls_path = _build_fixture_config(tmp_path)
    registry = load_family_registry(families_path)
    data = load_aligned_family_data(registry)
    config = load_reasoning_block_pls_config(pls_path)

    outputs = run_reasoning_block_pls_study(data, ["reasoning_A", "reasoning_F"], config)
    summary = outputs["summary"]
    all_metrics = outputs["all_metrics"]
    component_summary = outputs["component_summary"]

    assert {"HQ", "reasoning_A", "reasoning_F"} == set(summary["family_id"].unique().tolist())
    assert set(all_metrics["route_group"].unique().tolist()) == {"hq_benchmark", "raw_family", "block_pls"}
    assert set(all_metrics[all_metrics["family_id"] == "reasoning_A"]["outer_split_id"].tolist()) == set(
        all_metrics[all_metrics["family_id"] == "reasoning_F"]["outer_split_id"].tolist()
    )
    assert not component_summary.empty
    assert "weight_mean" in component_summary.columns
    assert "loading_sign_flip_rate" in component_summary.columns

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reasoning_block_pls.py",
            "--families-config",
            str(families_path),
            "--pls-config",
            str(pls_path),
        ],
    )
    reasoning_block_pls_pipeline.main()
    protocol = (tmp_path / "step_3_PLS_reasoning" / "10_protocol.md").read_text(encoding="utf-8")
    summary_md = (tmp_path / "step_3_PLS_reasoning" / "11_summary.md").read_text(encoding="utf-8")
    details = (tmp_path / "step_3_PLS_reasoning" / "12_details.md").read_text(encoding="utf-8")
    audit = (tmp_path / "step_3_PLS_reasoning" / "13_audit.md").read_text(encoding="utf-8")
    fold_metrics_csv = (tmp_path / "step_3_PLS_reasoning" / "fold_metrics.csv").read_text(encoding="utf-8")

    assert "reasoning block only" in protocol
    assert "HQ + A" in summary_md
    assert "Best blockwise `PLS` k by LR CV" in details
    assert "Private-test audit is deferred" in audit
    assert "route_group" in fold_metrics_csv
