from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.stability_selection import (
    build_row_subsamples,
    load_stability_selection_config,
    run_family_stability_study,
)
from pipelines.instability_control import stability_selection as stability_selection_pipeline


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_temp_dir() -> Path:
    tmp_path = Path(__file__).resolve().parents[2] / ".tmp_tests" / f"stability_selection_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _build_fixture_config(tmp_path: Path) -> tuple[Path, Path, int]:
    baseline_path = tmp_path / "baseline.csv"
    reasoning_a_path = tmp_path / "reasoning_a.csv"
    reasoning_f_path = tmp_path / "reasoning_f.csv"
    folds_path = tmp_path / "folds.json"
    families_path = tmp_path / "families.json"
    stability_path = tmp_path / "stability_selection.json"

    rows = []
    for idx in range(60):
        success = 1 if idx % 4 in {1, 2} else 0
        rows.append(
            {
                "founder_uuid": f"id_{idx}",
                "success": success,
                "exit_count": 0,
                "base_signal": 0.75 if success else 0.25,
                "base_noise_1": (idx % 7) / 7.0,
                "base_noise_2": ((idx * 3) % 11) / 11.0,
            }
        )
    baseline_df = pd.DataFrame(rows)
    reasoning_a_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "A_signal": [0.62 if value == 1 else 0.38 for value in baseline_df["success"]],
            "A_noise": [((idx * 5) % 13) / 13.0 for idx in range(len(baseline_df))],
        }
    )
    reasoning_f_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "F_signal_1": [0.92 if value == 1 else 0.08 for value in baseline_df["success"]],
            "F_signal_2": [0.88 if value == 1 else 0.12 for value in baseline_df["success"]],
        }
    )
    baseline_df.to_csv(baseline_path, index=False)
    reasoning_a_df.to_csv(reasoning_a_path, index=False)
    reasoning_f_df.to_csv(reasoning_f_path, index=False)
    _write_json(
        folds_path,
        {
            "fold_map": {f"id_{idx}": idx % 2 for idx in range(len(baseline_df))},
            "k": 2,
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
        stability_path,
        {
            "family_ids": ["reasoning_A", "reasoning_F"],
            "outer_cv": {"n_splits": 2, "n_repeats": 2, "random_state": 42},
            "subsampling": {"fraction": 0.5, "n_subsamples": 6, "random_state": 42, "stratified": True},
            "selector": {"c_grid": [0.1, 1.0], "max_iter": 2000, "selection_epsilon": 1e-8},
            "stability": {
                "primary_threshold": 0.8,
                "sign_consistency_threshold": 0.9,
                "report_thresholds": [0.6, 0.8, 0.9],
            },
            "evaluation": {
                "models": ["logistic"],
                "apply_rule_override": True,
                "threshold_grid": "default",
                "inner_threshold_cv_splits": 2,
                "random_state": 42,
            },
            "docs_output": {
                "protocol_markdown": str(tmp_path / "step_2_stability_analysis" / "10_protocol.md"),
                "summary_markdown": str(tmp_path / "step_2_stability_analysis" / "11_summary.md"),
                "details_markdown": str(tmp_path / "step_2_stability_analysis" / "12_details.md"),
                "audit_markdown": str(tmp_path / "step_2_stability_analysis" / "13_audit.md"),
                "summary_csv": str(tmp_path / "step_2_stability_analysis" / "summary.csv"),
                "feature_frequencies_csv": str(tmp_path / "step_2_stability_analysis" / "feature_frequencies.csv"),
                "coefficient_summary_csv": str(tmp_path / "step_2_stability_analysis" / "coefficient_summary.csv"),
            },
        },
    )
    baseline_feature_count = len([col for col in baseline_df.columns if col not in {"founder_uuid", "success"}])
    return families_path, stability_path, baseline_feature_count


def test_build_row_subsamples_samples_rows_not_features() -> None:
    y = pd.Series([0, 1] * 10).to_numpy(dtype=int)
    subsamples = build_row_subsamples(y, fraction=0.5, n_subsamples=4, random_state=42, stratified=True)
    assert len(subsamples) == 4
    assert all(len(sample) == 10 for sample in subsamples)
    assert all(int(sample.max()) < len(y) for sample in subsamples)


def test_stability_selection_study_runs_per_family() -> None:
    tmp_path = _base_temp_dir()
    families_path, stability_path, baseline_feature_count = _build_fixture_config(tmp_path)
    registry = load_family_registry(families_path)
    data = load_aligned_family_data(registry)
    config = load_stability_selection_config(stability_path)

    outputs = run_family_stability_study(data, ["reasoning_A", "reasoning_F"], config)
    summary = outputs["summary"]
    feature_freq = outputs["feature_frequencies"]
    benchmark = outputs["benchmark_metrics"]
    family_metrics = outputs["family_metrics"]
    coefficient_summary = outputs["coefficient_summary"]

    assert {"HQ", "reasoning_A", "reasoning_F"} == set(summary["family_id"].unique().tolist())
    assert set(family_metrics["route_group"].unique().tolist()) == {
        "raw_family",
        "competition_track",
        "augmentation_track",
    }
    assert set(feature_freq["family_id"].unique().tolist()) == {"reasoning_A", "reasoning_F"}
    assert set(feature_freq["subsample_count"].unique().tolist()) == {6}
    assert set(feature_freq["c_grid_size"].unique().tolist()) == {2}
    assert set(benchmark["outer_split_id"].unique().tolist()) == set(family_metrics["outer_split_id"].unique().tolist())
    assert "sign_consistency" in feature_freq.columns
    assert "positive_selection_frequency" in feature_freq.columns
    assert "negative_selection_frequency" in feature_freq.columns
    assert "mean_reasoning_sign_consistency" in summary.columns
    assert "sign_stable_reasoning_count_mean" in summary.columns
    assert not coefficient_summary.empty
    assert "coefficient_mean" in coefficient_summary.columns
    assert "sign_flip_rate" in coefficient_summary.columns

    augmentation_rows = summary[
        (summary["route_group"] == "augmentation_track") & (summary["evaluator_model"] == "logistic")
    ]
    assert not augmentation_rows.empty
    assert set(augmentation_rows["selected_hq_feature_count_mean"].round(6).tolist()) == {float(baseline_feature_count)}

    competition_rows = summary[
        (summary["route_group"] == "competition_track") & (summary["evaluator_model"] == "logistic")
    ]
    assert not competition_rows.empty
    assert float(competition_rows["selected_hq_feature_count_mean"].min()) < float(baseline_feature_count)


def test_stability_selection_pipeline_writes_docs() -> None:
    tmp_path = _base_temp_dir()
    families_path, stability_path, _ = _build_fixture_config(tmp_path)
    sys.argv = [
        "stability_selection.py",
        "--families-config",
        str(families_path),
        "--stability-config",
        str(stability_path),
    ]
    stability_selection_pipeline.main()
    protocol = (tmp_path / "step_2_stability_analysis" / "10_protocol.md").read_text(encoding="utf-8")
    summary = (tmp_path / "step_2_stability_analysis" / "11_summary.md").read_text(encoding="utf-8")
    details = (tmp_path / "step_2_stability_analysis" / "12_details.md").read_text(encoding="utf-8")
    audit = (tmp_path / "step_2_stability_analysis" / "13_audit.md").read_text(encoding="utf-8")
    coefficient_csv = (tmp_path / "step_2_stability_analysis" / "coefficient_summary.csv").read_text(encoding="utf-8")

    assert "row-subsampled stability selection" in protocol
    assert "Sign-consistency threshold" in protocol
    assert "HQ + A" in summary
    assert "HQ + F" in summary
    assert "Top reasoning features" in details
    assert "Mean Sign Consistency" in details
    assert "Reasoning-feature LR coefficient stability" in details
    assert "No scored private-test audit" in audit
    assert "coefficient_mean" in coefficient_csv
