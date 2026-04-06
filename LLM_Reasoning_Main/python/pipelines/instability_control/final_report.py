from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import read_json, timestamped_run_dir, write_json, write_markdown
from lib.shared.folds import load_fold_ids
from lib.stability.admission import run_sequential_admission
from lib.stability.diagnostics import run_family_diagnostics
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.reporting import final_report_markdown
from lib.stability.residuals import summarize_residual_gain
from lib.stability.routes import evaluate_route_cv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the instability-control final report.")
    parser.add_argument("--config", default=str(INSTABILITY_CONFIG_DIR / "reporting.json"))
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main() -> None:
    args = _parse_args()
    config = read_json(_resolve_path(str(args.config)))
    families_config = _resolve_path(str(config["families_config"]))
    admission_config = read_json(_resolve_path(str(config["admission_config"])))
    routes_config = read_json(_resolve_path(str(config["routes_config"])))

    registry = load_family_registry(families_config)
    data = load_aligned_family_data(registry)
    baseline_frame = data.baseline.copy()
    baseline_frame[registry.dataset.id_column] = data.labels.index
    baseline_frame["row_index"] = range(len(baseline_frame))
    fold_ids = load_fold_ids(baseline_frame, registry.dataset.folds_path, id_column=registry.dataset.id_column)

    diagnostics = run_family_diagnostics(data, fold_ids)
    admission = run_sequential_admission(
        data,
        fold_ids,
        family_order=list(admission_config.get("family_order", [spec.id for spec in registry.families])),
        min_delta_f0_5=float(admission_config.get("min_delta_f0_5", 0.0)),
        max_delta_f0_5_std=float(admission_config.get("max_delta_f0_5_std", 1.0)),
        min_delta_precision_at_10=float(admission_config.get("min_delta_precision_at_10", 0.0)),
    )

    combined = data.baseline.copy()
    for family_id in admission["admitted_families"]:
        combined = pd.concat([combined, data.candidate_frames[family_id]], axis=1)
    routes = [evaluate_route_cv(combined, data.labels, fold_ids, route) for route in routes_config.get("routes", [])]
    residuals = summarize_residual_gain(data.baseline, combined, data.labels, fold_ids)

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "final_report", "final_report")
    payload = {
        "diagnostics": diagnostics,
        "admission": admission,
        "routes": routes,
        "residuals": residuals,
    }
    write_json(run_dir / "final_report.json", payload)
    write_markdown(
        INSTABILITY_CONTROL_DOCS_DIR / "04_final_report.md",
        final_report_markdown(diagnostics, admission, routes, residuals),
    )
    print(f"Wrote final report to {run_dir}")


if __name__ == "__main__":
    main()
