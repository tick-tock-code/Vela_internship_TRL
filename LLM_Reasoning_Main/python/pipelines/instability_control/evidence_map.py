from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import read_json, timestamped_run_dir, write_csv, write_json, write_markdown
from lib.shared.folds import load_fold_ids
from lib.stability.combo_catalog import build_evaluation_units, load_combo_catalog
from lib.stability.evidence import build_route_snapshot, run_step1_evidence_map
from lib.stability.family_registry import load_aligned_family_data, load_family_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 1 evidence mapping for instability-control work.")
    parser.add_argument("--families-config", default=str(INSTABILITY_CONFIG_DIR / "families.json"))
    parser.add_argument("--combo-catalog-config", default=str(INSTABILITY_CONFIG_DIR / "combo_catalog.json"))
    parser.add_argument("--controls-config", default=str(INSTABILITY_CONFIG_DIR / "controls.json"))
    parser.add_argument("--reporting-config", default=str(INSTABILITY_CONFIG_DIR / "reporting.json"))
    parser.add_argument("--config", default="", help="Deprecated alias for --families-config.")
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main() -> None:
    args = _parse_args()
    families_config = _resolve_path(args.config or str(args.families_config))
    combo_config = _resolve_path(str(args.combo_catalog_config))
    controls_config = _resolve_path(str(args.controls_config))
    reporting_config = _resolve_path(str(args.reporting_config))

    registry = load_family_registry(families_config)
    data = load_aligned_family_data(registry)
    combo_specs = load_combo_catalog(combo_config)
    units = build_evaluation_units(data, combo_specs)

    baseline_frame = data.baseline.copy()
    baseline_frame[registry.dataset.id_column] = data.labels.index
    baseline_frame["row_index"] = range(len(baseline_frame))
    fold_ids = load_fold_ids(baseline_frame, registry.dataset.folds_path, id_column=registry.dataset.id_column)

    controls_payload = read_json(controls_config)
    reporting_payload = read_json(reporting_config)
    results = run_step1_evidence_map(data, units, fold_ids, controls_payload)

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "evidence_map", "step1")
    route_metrics = results["route_metrics"]
    overlap_metrics = results["overlap_metrics"]
    classification = results["classification"]
    priorities = results["priorities"]
    snapshot = build_route_snapshot(route_metrics, classification)

    write_csv(run_dir / "step1_route_metrics.csv", route_metrics)
    write_json(run_dir / "step1_route_metrics.json", route_metrics.to_dict(orient="records"))
    write_csv(run_dir / "step1_overlap_metrics.csv", overlap_metrics)
    write_json(run_dir / "step1_overlap_metrics.json", overlap_metrics.to_dict(orient="records"))
    write_csv(run_dir / "step1_snapshot.csv", snapshot)
    write_json(run_dir / "step1_snapshot.json", snapshot.to_dict(orient="records"))
    write_json(run_dir / "step1_classification.json", classification.to_dict(orient="records"))
    write_json(run_dir / "step1_priorities.json", priorities)
    write_markdown(run_dir / "step1_synthesis.md", results["synthesis_markdown"])
    write_markdown(run_dir / "step1_legacy_alignment.md", results["legacy_alignment_markdown"])

    docs_output = reporting_payload.get("docs_output", {})
    results_path = _resolve_path(str(docs_output.get("results_markdown", INSTABILITY_CONTROL_DOCS_DIR / "03_results.md")))
    legacy_alignment_path = _resolve_path(
        str(docs_output.get("legacy_alignment_markdown", INSTABILITY_CONTROL_DOCS_DIR / "07_step1_legacy_alignment.md"))
    )
    snapshot_path = _resolve_path(
        str(docs_output.get("snapshot_csv", INSTABILITY_CONTROL_DOCS_DIR / "step1" / "step1_snapshot.csv"))
    )
    write_markdown(results_path, results["synthesis_markdown"])
    write_markdown(legacy_alignment_path, results["legacy_alignment_markdown"])
    write_csv(snapshot_path, snapshot)
    print(f"Wrote evidence map outputs to {run_dir}")


if __name__ == "__main__":
    main()
