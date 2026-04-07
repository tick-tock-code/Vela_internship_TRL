from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.paths import BASE_DIR, INSTABILITY_CONFIG_DIR, INSTABILITY_CONTROL_RUNS_DIR
from lib.shared.artifact_io import ensure_dir, timestamped_run_dir, write_csv, write_json, write_markdown
from lib.stability.family_registry import load_aligned_family_data, load_family_registry
from lib.stability.stability_selection import (
    load_stability_selection_config,
    run_family_stability_study,
    stability_audit_markdown,
    stability_details_markdown,
    stability_protocol_markdown,
    stability_summary_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run row-subsampled stability selection for HQ + A-F routes.")
    parser.add_argument("--families-config", default=str(INSTABILITY_CONFIG_DIR / "families.json"))
    parser.add_argument("--stability-config", default=str(INSTABILITY_CONFIG_DIR / "stability_selection.json"))
    parser.add_argument("--family-ids", default="", help="Optional comma-separated family id override.")
    parser.add_argument("--outer-repeats", type=int, default=0, help="Optional override for outer CV repeats.")
    parser.add_argument("--outer-splits", type=int, default=0, help="Optional override for outer CV split count.")
    parser.add_argument("--n-subsamples", type=int, default=0, help="Optional override for row subsamples per outer split.")
    parser.add_argument("--run-label", default="", help="Optional suffix for the run directory label.")
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _docs_path(config: dict[str, object], key: str) -> Path:
    docs_output = config.get("docs_output", {})
    if not isinstance(docs_output, dict) or key not in docs_output:
        raise RuntimeError(f"Missing docs_output.{key} in stability-selection config.")
    return _resolve_path(str(docs_output[key]))


def _write_dataframe_bundle(run_dir: Path, stem: str, frame) -> None:
    write_csv(run_dir / f"{stem}.csv", frame)
    write_json(run_dir / f"{stem}.json", json.loads(frame.to_json(orient="records")))


def _serializable_splits(splits: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in splits:
        row = dict(split)
        for key in ["train_idx", "test_idx"]:
            value = row.get(key)
            if hasattr(value, "tolist"):
                row[key] = value.tolist()
        rows.append(row)
    return rows


def _apply_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    updated = json.loads(json.dumps(config))
    outer_cv = updated.setdefault("outer_cv", {})
    subsampling = updated.setdefault("subsampling", {})
    if args.outer_repeats > 0:
        outer_cv["n_repeats"] = int(args.outer_repeats)
    if args.outer_splits > 0:
        outer_cv["n_splits"] = int(args.outer_splits)
    if args.n_subsamples > 0:
        subsampling["n_subsamples"] = int(args.n_subsamples)
    if args.family_ids.strip():
        updated["family_ids"] = [value.strip() for value in str(args.family_ids).split(",") if value.strip()]
    return updated


def _append_progress(log_path: Path, payload: dict[str, object]) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = str(payload.get("event", "progress"))
    outer_split = str(payload.get("outer_split_id", ""))
    family = str(payload.get("family_label", payload.get("family_id", "")))
    completed = int(payload.get("completed_units", 0))
    total = int(payload.get("total_units", 0))
    line = f"[{stamp}] {event} split={outer_split} family={family} progress={completed}/{total}"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def main() -> None:
    args = _parse_args()
    registry = load_family_registry(_resolve_path(str(args.families_config)))
    data = load_aligned_family_data(registry)
    config = _apply_overrides(load_stability_selection_config(_resolve_path(str(args.stability_config))), args)
    family_ids = [str(value) for value in config.get("family_ids", [])]
    family_ids = [family_id for family_id in family_ids if family_id in data.family_specs]
    if not family_ids:
        raise RuntimeError("No valid reasoning families selected for stability selection.")

    label_parts = ["study"]
    if args.run_label.strip():
        label_parts.append(str(args.run_label).strip())
    elif family_ids:
        label_parts.append("pilot_" + "_".join(family_id.replace("reasoning_", "").lower() for family_id in family_ids[:3]))
    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "stability_selection", "_".join(label_parts))
    progress_log_path = run_dir / "progress.log"
    write_json(run_dir / "effective_config.json", config)
    write_markdown(run_dir / "progress.log", "")

    def _progress_callback(payload: dict[str, object]) -> None:
        _append_progress(progress_log_path, payload)
        write_json(run_dir / "progress.json", payload)

    outputs = run_family_stability_study(data, family_ids, config, progress_callback=_progress_callback)
    family_labels = [str(data.family_specs[family_id].label) for family_id in family_ids]

    protocol_markdown = stability_protocol_markdown(config, family_labels)
    summary_markdown = stability_summary_markdown(
        outputs["summary"],
        outputs["family_selection_summary"],
        outputs["benchmark_metrics"],
    )
    details_markdown = stability_details_markdown(
        outputs["feature_frequencies"],
        outputs["family_selection_summary"],
        outputs["summary"],
    )
    audit_markdown = stability_audit_markdown(outputs["audit"])

    write_json(run_dir / "outer_splits.json", _serializable_splits(outputs["outer_splits"]))
    _write_dataframe_bundle(run_dir, "benchmark_metrics", outputs["benchmark_metrics"])
    _write_dataframe_bundle(run_dir, "family_metrics", outputs["family_metrics"])
    _write_dataframe_bundle(run_dir, "feature_frequencies", outputs["feature_frequencies"])
    _write_dataframe_bundle(run_dir, "selection_summary", outputs["selection_summary"])
    _write_dataframe_bundle(run_dir, "stability_selection_summary", outputs["summary"])
    _write_dataframe_bundle(run_dir, "family_selection_summary", outputs["family_selection_summary"])
    _write_dataframe_bundle(run_dir, "final_audit", outputs["audit"])
    write_markdown(run_dir / "10_stability_selection_protocol.md", protocol_markdown)
    write_markdown(run_dir / "11_stability_selection_summary.md", summary_markdown)
    write_markdown(run_dir / "12_stability_selection_family_details.md", details_markdown)
    write_markdown(run_dir / "13_stability_selection_final_audit.md", audit_markdown)

    protocol_path = _docs_path(config, "protocol_markdown")
    summary_path = _docs_path(config, "summary_markdown")
    details_path = _docs_path(config, "details_markdown")
    audit_path = _docs_path(config, "audit_markdown")
    summary_csv_path = _docs_path(config, "summary_csv")
    freq_csv_path = _docs_path(config, "feature_frequencies_csv")

    write_markdown(protocol_path, protocol_markdown)
    write_markdown(summary_path, summary_markdown)
    write_markdown(details_path, details_markdown)
    write_markdown(audit_path, audit_markdown)
    write_csv(summary_csv_path, outputs["summary"])
    write_csv(freq_csv_path, outputs["feature_frequencies"])

    for docs_path in [protocol_path, summary_path, details_path, audit_path, summary_csv_path, freq_csv_path]:
        ensure_dir(docs_path.parent)

    print(f"Wrote stability-selection outputs to {run_dir}")


if __name__ == "__main__":
    main()
