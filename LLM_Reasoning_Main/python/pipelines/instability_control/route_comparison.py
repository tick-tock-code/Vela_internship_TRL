from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import read_json, timestamped_run_dir, write_json, write_markdown
from lib.shared.folds import load_fold_ids
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.reporting import routes_markdown
from lib.stability.residuals import summarize_residual_gain
from lib.stability.routes import evaluate_route_cv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare route options for admitted families.")
    parser.add_argument("--config", default=str(INSTABILITY_CONFIG_DIR / "routes.json"))
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main() -> None:
    args = _parse_args()
    config = read_json(_resolve_path(str(args.config)))
    registry = load_family_registry(_resolve_path(str(config["families_config"])))
    data = load_aligned_family_data(registry)

    baseline_frame = data.baseline.copy()
    baseline_frame[registry.dataset.id_column] = data.labels.index
    baseline_frame["row_index"] = range(len(baseline_frame))
    fold_ids = load_fold_ids(baseline_frame, registry.dataset.folds_path, id_column=registry.dataset.id_column)

    selected_families = list(config.get("selected_families", []))
    if not selected_families and config.get("admission_results"):
        selected_families = list(read_json(_resolve_path(str(config["admission_results"]))).get("admitted_families", []))

    combined = data.baseline.copy()
    for family_id in selected_families:
        combined = pd.concat([combined, data.candidate_frames[family_id]], axis=1)

    route_results = [
        evaluate_route_cv(combined, data.labels, fold_ids, route_spec)
        for route_spec in config.get("routes", [])
    ]
    residuals = summarize_residual_gain(data.baseline, combined, data.labels, fold_ids)

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "route_comparison", "routes")
    write_json(run_dir / "routes.json", route_results)
    write_json(run_dir / "residuals.json", residuals)
    write_markdown(INSTABILITY_CONTROL_DOCS_DIR / "03_route_comparison.md", routes_markdown(route_results))
    print(f"Wrote route comparison to {run_dir}")


if __name__ == "__main__":
    main()
