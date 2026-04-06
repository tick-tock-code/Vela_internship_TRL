from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import timestamped_run_dir, write_json, write_markdown
from lib.shared.folds import load_fold_ids
from lib.stability.diagnostics import run_family_diagnostics
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.reporting import diagnostics_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run family diagnostics for instability-control work.")
    parser.add_argument("--config", default=str(INSTABILITY_CONFIG_DIR / "families.json"))
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main() -> None:
    args = _parse_args()
    registry = load_family_registry(_resolve_path(str(args.config)))
    data = load_aligned_family_data(registry)
    baseline_frame = data.baseline.copy()
    baseline_frame[registry.dataset.id_column] = data.labels.index
    baseline_frame["row_index"] = range(len(baseline_frame))
    fold_ids = load_fold_ids(baseline_frame, registry.dataset.folds_path, id_column=registry.dataset.id_column)
    diagnostics = run_family_diagnostics(data, fold_ids)

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "family_diagnostics", "diagnostics")
    write_json(run_dir / "diagnostics.json", diagnostics)
    write_markdown(INSTABILITY_CONTROL_DOCS_DIR / "01_family_diagnostics.md", diagnostics_markdown(diagnostics))
    print(f"Wrote diagnostics to {run_dir}")


if __name__ == "__main__":
    main()
