from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.supervised_grouping import (
    fit_supervised_grouped_scope,
    load_supervised_grouping_config,
    run_supervised_grouping_study,
)
from pipelines.instability_control import supervised_grouping as supervised_grouping_pipeline


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_temp_dir() -> Path:
    tmp_path = Path(__file__).resolve().parents[2] / ".tmp_tests" / f"supervised_grouping_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _build_fixture_config(tmp_path: Path) -> tuple[Path, Path]:
    baseline_path = tmp_path / "baseline.csv"
    reasoning_a_path = tmp_path / "reasoning_a.csv"
    reasoning_f_path = tmp_path / "reasoning_f.csv"
    folds_path = tmp_path / "folds.json"
    families_path = tmp_path / "families.json"
    grouping_path = tmp_path / "supervised_grouping.json"

    rows = []
    for idx in range(48):
        success = 1 if idx % 4 in {1, 2} else 0
        rows.append(
            {
                "founder_uuid": f"id_{idx}",
                "success": success,
                "exit_count": 1 if idx % 11 == 0 else 0,
                "hq_signal": 0.85 if success else 0.15,
                "hq_noise": ((idx * 3) % 17) / 17.0,
                "hq_aux": ((idx * 5) % 13) / 13.0,
            }
        )
    baseline_df = pd.DataFrame(rows)
    reasoning_a_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "A_one": [0.82 if value == 1 else 0.18 for value in baseline_df["success"]],
            "A_two": [0.75 if value == 1 else 0.25 for value in baseline_df["success"]],
            "A_three": [((idx * 7) % 19) / 19.0 for idx in range(len(baseline_df))],
        }
    )
    reasoning_f_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "F_one": [0.90 if value == 1 else 0.10 for value in baseline_df["success"]],
            "F_two": [0.86 if value == 1 else 0.14 for value in baseline_df["success"]],
            "F_three": [0.83 if value == 1 else 0.17 for value in baseline_df["success"]],
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
        grouping_path,
        {
            "outer_cv": {"n_splits": 3, "n_repeats": 2, "random_state": 42},
            "evaluation": {
                "models": ["logistic"],
                "apply_rule_override": True,
                "threshold_grid": "default",
                "inner_threshold_cv_splits": 2,
                "random_state": 42,
            },
            "augmentation": {
                "groups_per_family": [2],
                "feature_sets": [
                    {
                        "id": "hq_plus_a",
                        "label": "HQ + A",
                        "family_ids": ["reasoning_A"],
                    }
                ],
            },
            "competition": {
                "total_groups": 2,
                "feature_sets": [
                    {
                        "id": "hq_only",
                        "label": "HQ",
                        "family_ids": [],
                    },
                    {
                        "id": "hq_plus_a",
                        "label": "HQ + A",
                        "family_ids": ["reasoning_A"],
                    }
                ],
            },
            "docs_output": {
                "protocol_markdown": str(tmp_path / "step_4_supervised_grouping" / "10_protocol.md"),
                "summary_markdown": str(tmp_path / "step_4_supervised_grouping" / "11_summary.md"),
                "details_markdown": str(tmp_path / "step_4_supervised_grouping" / "12_details.md"),
                "audit_markdown": str(tmp_path / "step_4_supervised_grouping" / "13_audit.md"),
                "summary_csv": str(tmp_path / "step_4_supervised_grouping" / "summary.csv"),
                "fold_metrics_csv": str(tmp_path / "step_4_supervised_grouping" / "fold_metrics.csv"),
                "cluster_assignments_csv": str(tmp_path / "step_4_supervised_grouping" / "cluster_assignments.csv"),
                "component_summary_csv": str(tmp_path / "step_4_supervised_grouping" / "component_summary.csv"),
                "cocluster_summary_csv": str(tmp_path / "step_4_supervised_grouping" / "cocluster_summary.csv"),
            },
        },
    )
    return families_path, grouping_path


def test_supervised_grouping_config_has_expected_routes() -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "instability_control" / "supervised_grouping.json"
    config = load_supervised_grouping_config(config_path)
    assert config["outer_cv"]["n_splits"] == 3
    assert config["outer_cv"]["n_repeats"] == 16
    assert config["augmentation"]["groups_per_family"] == [2, 3]
    assert config["competition"]["total_groups"] == 6
    assert [row["id"] for row in config["augmentation"]["feature_sets"]] == [
        "hq_plus_a",
        "hq_plus_f",
        "hq_plus_a_b_c_d_e",
        "hq_plus_b_c_d_e_f",
    ]


def test_fit_supervised_grouped_scope_uses_train_only_statistics_and_singleton_passthrough() -> None:
    train_df = pd.DataFrame({"A_one": [1.0, 2.0, 3.0], "A_two": [4.0, 5.0, 6.0]})
    test_df = pd.DataFrame({"A_one": [100.0, 200.0], "A_two": [400.0, 500.0]})
    y_train = np.array([0, 1, 1], dtype=int)
    source_map = {
        "A_one": {"source_family_id": "reasoning_A", "source_family_label": "A", "feature_group": "reasoning"},
        "A_two": {"source_family_id": "reasoning_A", "source_family_label": "A", "feature_group": "reasoning"},
    }

    result = fit_supervised_grouped_scope(
        train_df,
        test_df,
        y_train,
        requested_group_count=4,
        track="augmentation",
        feature_set_id="hq_plus_a",
        feature_set_label="HQ + A",
        grouping_spec_id="per_family_4",
        grouping_spec_label="4 groups per family",
        grouping_scope_id="reasoning_A",
        grouping_scope_label="A",
        component_prefix="A",
        source_family_map=source_map,
    )

    assert result.actual_group_count == 2
    assert result.train_df.shape[1] == 2
    assert set(result.train_df.columns.tolist()) == {"A_grp_1", "A_grp_2"}
    assert set(result.component_rows["x_weight"].tolist()) == {1.0}
    observed_train_mean = float(result.component_rows[result.component_rows["original_feature"] == "A_one"]["train_mean"].iloc[0])
    combined_mean = float(pd.concat([train_df, test_df])["A_one"].mean())
    assert observed_train_mean == float(train_df["A_one"].mean())
    assert observed_train_mean != combined_mean


def test_supervised_grouping_study_and_pipeline_write_expected_outputs(monkeypatch) -> None:
    tmp_path = _base_temp_dir()
    families_path, grouping_path = _build_fixture_config(tmp_path)
    registry = load_family_registry(families_path)
    data = load_aligned_family_data(registry)
    config = load_supervised_grouping_config(grouping_path)

    outputs = run_supervised_grouping_study(data, config)
    summary = outputs["summary"]
    cluster_assignments = outputs["cluster_assignments"]
    component_summary = outputs["component_summary"]
    cocluster_summary = outputs["cocluster_summary"]

    assert {"benchmark", "augmentation", "competition"} == set(summary["track"].unique().tolist())
    assert {"hq_benchmark", "raw_comparison", "grouped_route"} == set(summary["route_group"].unique().tolist())
    assert set(cluster_assignments[cluster_assignments["track"] == "augmentation"]["grouping_scope_id"].unique().tolist()) == {"reasoning_A"}
    assert set(cluster_assignments[cluster_assignments["track"] == "competition"]["grouping_scope_id"].unique().tolist()) == {"full_route"}
    assert not component_summary.empty
    assert not cocluster_summary.empty

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supervised_grouping.py",
            "--families-config",
            str(families_path),
            "--grouping-config",
            str(grouping_path),
        ],
    )
    supervised_grouping_pipeline.main()
    protocol = (tmp_path / "step_4_supervised_grouping" / "10_protocol.md").read_text(encoding="utf-8")
    summary_md = (tmp_path / "step_4_supervised_grouping" / "11_summary.md").read_text(encoding="utf-8")
    details = (tmp_path / "step_4_supervised_grouping" / "12_details.md").read_text(encoding="utf-8")
    audit = (tmp_path / "step_4_supervised_grouping" / "13_audit.md").read_text(encoding="utf-8")
    cluster_csv = (tmp_path / "step_4_supervised_grouping" / "cluster_assignments.csv").read_text(encoding="utf-8")

    assert "1`-component `PLS`" in protocol or "1-component `PLS`" in protocol
    assert "Augmentation Track" in summary_md
    assert "Representative cluster layout" in details
    assert "deferred" in audit.lower()
    assert "grouping_scope_id" in cluster_csv
