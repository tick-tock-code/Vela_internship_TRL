from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.shared.folds import load_fold_ids
from lib.stability.combo_catalog import build_evaluation_units, load_combo_catalog
from lib.stability.evidence import run_step1_evidence_map
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.routes import evaluate_route_cv
from pipelines.instability_control import evidence_map as evidence_map_pipeline


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_temp_dir() -> Path:
    tmp_path = Path(__file__).resolve().parents[2] / ".tmp_tests" / f"stability_workflow_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def _build_fixture_config(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    baseline_path = tmp_path / "baseline.csv"
    reasoning_a_path = tmp_path / "reasoning_a.csv"
    reasoning_f_path = tmp_path / "reasoning_f.csv"
    folds_path = tmp_path / "folds.json"
    families_path = tmp_path / "families.json"
    combos_path = tmp_path / "combo_catalog.json"
    controls_path = tmp_path / "controls.json"
    reporting_path = tmp_path / "reporting.json"

    baseline_df = pd.DataFrame(
        {
            "founder_uuid": [f"id_{idx}" for idx in range(12)],
            "success": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "exit_count": [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
            "base_signal": [0.1, 0.9, 0.2, 0.82, 0.11, 0.83, 0.19, 0.85, 0.12, 0.9, 0.18, 0.87],
        }
    )
    reasoning_a_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "A_reason": [0.12, 0.83, 0.25, 0.76, 0.20, 0.70, 0.22, 0.75, 0.24, 0.80, 0.26, 0.79],
        }
    )
    reasoning_f_df = pd.DataFrame(
        {
            "founder_uuid": baseline_df["founder_uuid"],
            "success": baseline_df["success"],
            "F_reason": [0.05, 0.95, 0.08, 0.92, 0.07, 0.89, 0.09, 0.91, 0.08, 0.93, 0.10, 0.94],
        }
    )
    baseline_df.to_csv(baseline_path, index=False)
    reasoning_a_df.to_csv(reasoning_a_path, index=False)
    reasoning_f_df.to_csv(reasoning_f_path, index=False)
    _write_json(
        folds_path,
        {
            "fold_map": {
                "id_0": 0,
                "id_1": 0,
                "id_2": 1,
                "id_3": 1,
                "id_4": 2,
                "id_5": 2,
                "id_6": 0,
                "id_7": 0,
                "id_8": 1,
                "id_9": 1,
                "id_10": 2,
                "id_11": 2
            },
            "k": 3,
            "random_state": 42
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
                "legacy_labels": {
                    "raw_lr_base": "LR_BASE_HQ"
                }
            },
            "families": [
                {
                    "id": "reasoning_A",
                    "label": "A",
                    "path": str(reasoning_a_path),
                    "feature_prefixes": ["A_"],
                    "legacy_labels": {
                        "raw_lr_base": "LR_BASE_A",
                        "pls_lr_n6": "LR_PLS_A"
                    }
                },
                {
                    "id": "reasoning_F",
                    "label": "F",
                    "path": str(reasoning_f_path),
                    "feature_prefixes": ["F_"],
                    "legacy_labels": {
                        "pls_lr_n6": "LR_PLS_F"
                    }
                }
            ],
        },
    )
    _write_json(
        combos_path,
        {
            "combos": [
                {
                    "id": "combo_A_F",
                    "label": "A+F",
                    "family_ids": ["reasoning_A", "reasoning_F"]
                }
            ]
        },
    )
    _write_json(
        controls_path,
        {
            "routes": [
                {
                    "id": "raw_lr_base",
                    "label": "Raw LR (BASE)",
                    "model": "logistic",
                    "transform": "BASE",
                    "threshold_grid": "default",
                    "apply_rule_override": True,
                    "drop_baseline_features": [],
                    "classification_role": "raw"
                },
                {
                    "id": "pls_lr_n6",
                    "label": "PLS LR (n=6)",
                    "model": "logistic",
                    "transform": "PLS",
                    "pls_components": 2,
                    "threshold_grid": "default",
                    "apply_rule_override": True,
                    "drop_baseline_features": [],
                    "classification_role": "transformed"
                }
            ],
            "classification": {
                "primary_raw_route_id": "raw_lr_base",
                "secondary_raw_route_id": "raw_lr_base",
                "primary_transformed_route_id": "pls_lr_n6",
                "compression_priority_limit": 5,
                "stability_priority_limit": 5,
                "grouped_priority_limit": 5
            }
        },
    )
    _write_json(
        reporting_path,
        {
            "families_config": str(families_path),
            "combo_catalog_config": str(combos_path),
            "controls_config": str(controls_path),
            "methods_config": str(tmp_path / "methods.json"),
            "docs_output": {
                "results_markdown": str(tmp_path / "step1" / "03_results.md"),
                "legacy_alignment_markdown": str(tmp_path / "step1" / "07_alignment.md"),
                "snapshot_csv": str(tmp_path / "step1" / "step1_snapshot.csv"),
                "method_benchmark_markdown": str(tmp_path / "step_2_stability_analysis" / "08_method.md"),
                "status_report_markdown": str(tmp_path / "current_status.md")
            }
        },
    )
    _write_json(tmp_path / "methods.json", {"active_methods": [], "available_methods": [{"id": "spls", "notes": "placeholder"}]})
    return families_path, combos_path, controls_path, reporting_path


def test_step1_evidence_map_runs_on_synthetic_data() -> None:
    tmp_path = _base_temp_dir()
    families_path, combos_path, controls_path, _ = _build_fixture_config(tmp_path)
    registry = load_family_registry(families_path)
    data = load_aligned_family_data(registry)
    combo_specs = load_combo_catalog(combos_path)
    units = build_evaluation_units(data, combo_specs)
    frame = data.baseline.copy()
    frame["founder_uuid"] = data.labels.index
    frame["row_index"] = range(len(frame))
    fold_ids = load_fold_ids(frame, registry.dataset.folds_path)

    payload = run_step1_evidence_map(data, units, fold_ids, json.loads(controls_path.read_text(encoding="utf-8")))
    assert not payload["route_metrics"].empty
    assert not payload["overlap_metrics"].empty
    assert "classification" in payload
    assert "## Unit Classification" in payload["synthesis_markdown"]
    assert "compression_priority" in json.dumps(payload["priorities"])


def test_route_evaluators_share_primary_metric_keys() -> None:
    tmp_path = _base_temp_dir()
    families_path, _, _, _ = _build_fixture_config(tmp_path)
    registry = load_family_registry(families_path)
    data = load_aligned_family_data(registry)
    frame = data.baseline.copy()
    frame["founder_uuid"] = data.labels.index
    frame["row_index"] = range(len(frame))
    fold_ids = load_fold_ids(frame, registry.dataset.folds_path)
    extra = data.candidate_frames["reasoning_A"]

    raw = evaluate_route_cv(
        data.baseline,
        extra,
        data.labels,
        fold_ids,
        {"id": "raw_lr_base", "model": "logistic", "transform": "BASE", "apply_rule_override": True},
        rule_mask=data.baseline["exit_count"].astype(float).to_numpy() > 0,
    )
    pls = evaluate_route_cv(
        data.baseline,
        extra,
        data.labels,
        fold_ids,
        {"id": "pls_lr_n6", "model": "logistic", "transform": "PLS", "pls_components": 2, "apply_rule_override": True},
        rule_mask=data.baseline["exit_count"].astype(float).to_numpy() > 0,
    )

    for result in (raw, pls):
        assert "f0_5_mean" in result["summary"]
        assert "pr_auc_mean" in result["summary"]
        assert "precision_at_10_mean" in result["summary"]
        assert result["per_fold"]


def test_evidence_map_pipeline_writes_expected_artifacts(monkeypatch) -> None:
    tmp_path = _base_temp_dir()
    families_path, combos_path, controls_path, reporting_path = _build_fixture_config(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evidence_map.py",
            "--families-config",
            str(families_path),
            "--combo-catalog-config",
            str(combos_path),
            "--controls-config",
            str(controls_path),
            "--reporting-config",
            str(reporting_path),
        ],
    )
    evidence_map_pipeline.main()
    assert (tmp_path / "step1" / "03_results.md").exists()
    assert (tmp_path / "step1" / "07_alignment.md").exists()
    assert (tmp_path / "step1" / "step1_snapshot.csv").exists()


def test_instability_docs_use_evidence_first_language() -> None:
    docs_dir = Path(__file__).resolve().parents[2] / "docs" / "instability_control"
    protocol = (docs_dir / "step1" / "02_protocol.md").read_text(encoding="utf-8")
    note_05 = (docs_dir / "step1" / "05_hq_anchor_and_route_protocol.md").read_text(encoding="utf-8")
    note_06 = (docs_dir / "step1" / "06_step1_family_diagnostics_and_admission.md").read_text(encoding="utf-8")

    assert "Step 1 is evidence mapping" in protocol
    assert "Step 1 uses raw controls plus transformed controls" in note_05
    assert "transform-sensitive" in note_06.lower()
    assert "not a keep/discard gate based on raw gain" in note_06
