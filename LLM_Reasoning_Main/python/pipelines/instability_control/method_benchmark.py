from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_DOCS_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import read_json, timestamped_run_dir, write_json, write_markdown
from lib.stability.methods import build_method_scaffold_summary, load_methods_config, method_scaffold_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the method-benchmark scaffold for instability-control work.")
    parser.add_argument("--methods-config", default=str(INSTABILITY_CONFIG_DIR / "methods.json"))
    parser.add_argument("--reporting-config", default=str(INSTABILITY_CONFIG_DIR / "reporting.json"))
    parser.add_argument("--priorities-json", default="")
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def main() -> None:
    args = _parse_args()
    methods_config = load_methods_config(_resolve_path(str(args.methods_config)))
    priorities: dict[str, list[str]] | None = None
    if args.priorities_json:
        priorities_payload = read_json(_resolve_path(str(args.priorities_json)))
        if isinstance(priorities_payload, dict):
            priorities = {str(key): [str(item) for item in value] for key, value in priorities_payload.items() if isinstance(value, list)}

    summary = build_method_scaffold_summary(methods_config, priorities)
    markdown = method_scaffold_markdown(summary)

    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "method_benchmark", "scaffold")
    write_json(run_dir / "method_benchmark.json", summary)
    write_markdown(run_dir / "method_benchmark.md", markdown)

    reporting_payload = read_json(_resolve_path(str(args.reporting_config)))
    docs_output = reporting_payload.get("docs_output", {})
    docs_path = _resolve_path(str(docs_output.get("method_benchmark_markdown", INSTABILITY_CONTROL_DOCS_DIR / "step_2_stability_analysis" / "08_method_benchmark.md")))
    write_markdown(docs_path, markdown)
    print(f"Wrote method scaffold outputs to {run_dir}")


if __name__ == "__main__":
    main()
