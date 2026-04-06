from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.shared.folds import load_fold_ids
from lib.stability.admission import run_sequential_admission
from lib.stability.diagnostics import run_family_diagnostics
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.residuals import summarize_residual_gain
from lib.stability.routes import evaluate_route_cv


def test_stability_modules_run_on_synthetic_data() -> None:
    tmp_path = Path(__file__).resolve().parents[2] / ".tmp_tests" / f"stability_workflow_{uuid.uuid4().hex}"
    tmp_path.mkdir(parents=True, exist_ok=True)

    baseline_path = tmp_path / "baseline.csv"
    family_path = tmp_path / "family.csv"
    folds_path = tmp_path / "folds.json"
    config_path = tmp_path / "families.json"

    baseline_df = pd.DataFrame(
        {
            "founder_uuid": [f"id_{idx}" for idx in range(8)],
            "success": [0, 1, 0, 1, 0, 1, 0, 1],
            "base_signal": [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.25, 0.75],
        }
    )
    family_df = pd.DataFrame(
        {
            "founder_uuid": [f"id_{idx}" for idx in range(8)],
            "success": [0, 1, 0, 1, 0, 1, 0, 1],
            "family_signal": [0.05, 0.95, 0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        }
    )
    baseline_df.to_csv(baseline_path, index=False)
    family_df.to_csv(family_path, index=False)
    folds_path.write_text(
        json.dumps(
                {
                    "fold_map": {
                        "id_0": 0,
                        "id_1": 0,
                        "id_2": 1,
                        "id_3": 1,
                        "id_4": 0,
                        "id_5": 0,
                        "id_6": 1,
                        "id_7": 1
                    },
                "k": 2,
                "random_state": 42,
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "id_column": "founder_uuid",
                    "label_column": "success",
                    "folds_path": str(folds_path),
                },
                "baseline": {
                    "id": "hq_baseline",
                    "label": "HQ",
                    "path": str(baseline_path),
                },
                "families": [
                    {
                        "id": "reasoning_A",
                        "label": "A",
                        "path": str(family_path),
                        "feature_prefixes": ["family_"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = load_family_registry(config_path)
    data = load_aligned_family_data(registry)
    frame = data.baseline.copy()
    frame["founder_uuid"] = data.labels.index
    frame["row_index"] = range(len(frame))
    fold_ids = load_fold_ids(frame, folds_path)

    diagnostics = run_family_diagnostics(data, fold_ids)
    assert "reasoning_A" in diagnostics

    admission = run_sequential_admission(
        data,
        fold_ids,
        family_order=["reasoning_A"],
        min_delta_f0_5=-1.0,
        max_delta_f0_5_std=1.0,
        min_delta_precision_at_10=-1.0,
    )
    assert admission["decisions"]

    combined = pd.concat([data.baseline, data.candidate_frames["reasoning_A"]], axis=1)
    route = evaluate_route_cv(combined, data.labels, fold_ids, {"id": "baseline_logistic", "type": "logistic"})
    assert route["summary"]["f0_5_mean"] >= 0.0

    residuals = summarize_residual_gain(data.baseline, combined, data.labels, fold_ids)
    assert "recovered_false_negatives" in residuals
