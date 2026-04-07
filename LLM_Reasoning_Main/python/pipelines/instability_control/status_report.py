from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import read_json, timestamped_run_dir, write_json, write_markdown
from lib.shared.folds import load_fold_ids
from lib.stability.combo_catalog import build_evaluation_units, load_combo_catalog
from lib.stability.evidence import run_step1_evidence_map
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.methods import build_method_scaffold_summary, load_methods_config
from lib.stability.reporting import status_report_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the current instability-control status report.")
    parser.add_argument("--config", default=str(INSTABILITY_CONFIG_DIR / "reporting.json"))
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main() -> None:
    args = _parse_args()
    reporting_payload = read_json(_resolve_path(str(args.config)))
    families_config = _resolve_path(str(reporting_payload["families_config"]))
    combo_catalog_config = _resolve_path(str(reporting_payload["combo_catalog_config"]))
    controls_config = _resolve_path(str(reporting_payload["controls_config"]))
    methods_config = _resolve_path(str(reporting_payload["methods_config"]))

    registry = load_family_registry(families_config)
    data = load_aligned_family_data(registry)
    combo_specs = load_combo_catalog(combo_catalog_config)
    units = build_evaluation_units(data, combo_specs)
    baseline_frame = data.baseline.copy()
    baseline_frame[registry.dataset.id_column] = data.labels.index
    baseline_frame["row_index"] = range(len(baseline_frame))
    fold_ids = load_fold_ids(baseline_frame, registry.dataset.folds_path, id_column=registry.dataset.id_column)

    controls_payload = read_json(controls_config)
    step1 = run_step1_evidence_map(data, units, fold_ids, controls_payload)
    method_summary = build_method_scaffold_summary(load_methods_config(methods_config), step1["priorities"])
    markdown = status_report_markdown(step1["route_metrics"], step1["classification"], step1["priorities"], method_summary)

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "status_report", "status_report")
    payload = {
        "route_metrics": step1["route_metrics"].to_dict(orient="records"),
        "classification": step1["classification"].to_dict(orient="records"),
        "priorities": step1["priorities"],
        "method_summary": method_summary,
    }
    write_json(run_dir / "status_report.json", payload)
    write_markdown(run_dir / "status_report.md", markdown)

    docs_output = reporting_payload.get("docs_output", {})
    docs_path = _resolve_path(str(docs_output.get("status_report_markdown", INSTABILITY_CONTROL_DOCS_DIR / "step3" / "09_status_report.md")))
    write_markdown(docs_path, markdown)
    print(f"Wrote status report to {run_dir}")


if __name__ == "__main__":
    main()
