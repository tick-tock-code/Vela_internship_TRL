from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import read_json, timestamped_run_dir, write_json, write_markdown
from lib.shared.folds import load_fold_ids
from lib.stability.admission import run_sequential_admission
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.reporting import admission_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential family admission.")
    parser.add_argument("--config", default=str(INSTABILITY_CONFIG_DIR / "admission.json"))
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

    family_order = list(config.get("family_order", [spec.id for spec in registry.families]))
    results = run_sequential_admission(
        data,
        fold_ids,
        family_order=family_order,
        min_delta_f0_5=float(config.get("min_delta_f0_5", 0.0)),
        max_delta_f0_5_std=float(config.get("max_delta_f0_5_std", 1.0)),
        min_delta_precision_at_10=float(config.get("min_delta_precision_at_10", 0.0)),
    )

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "family_admission", "admission")
    write_json(run_dir / "admission.json", results)
    write_markdown(INSTABILITY_CONTROL_DOCS_DIR / "02_family_admission.md", admission_markdown(results))
    print(f"Wrote admission results to {run_dir}")


if __name__ == "__main__":
    main()
