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
from lib.stability.reasoning_block_pls import (
    block_pls_audit_markdown,
    block_pls_details_markdown,
    block_pls_protocol_markdown,
    block_pls_summary_markdown,
    load_reasoning_block_pls_config,
    run_reasoning_block_pls_study,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blockwise reasoning-only PLS before HQ concatenation.")
    parser.add_argument("--families-config", default=str(INSTABILITY_CONFIG_DIR / "families.json"))
    parser.add_argument("--pls-config", default=str(INSTABILITY_CONFIG_DIR / "reasoning_block_pls.json"))
    parser.add_argument("--family-ids", default="", help="Optional comma-separated family id override.")
    parser.add_argument("--outer-repeats", type=int, default=0, help="Optional override for outer CV repeats.")
    parser.add_argument("--outer-splits", type=int, default=0, help="Optional override for outer CV split count.")
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
        raise RuntimeError(f"Missing docs_output.{key} in reasoning-block-pls config.")
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
    if args.outer_repeats > 0:
        outer_cv["n_repeats"] = int(args.outer_repeats)
    if args.outer_splits > 0:
        outer_cv["n_splits"] = int(args.outer_splits)
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
    config = _apply_overrides(load_reasoning_block_pls_config(_resolve_path(str(args.pls_config))), args)
    family_ids = [str(value) for value in config.get("family_ids", [])]
    family_ids = [family_id for family_id in family_ids if family_id in data.family_specs]
    if not family_ids:
        raise RuntimeError("No valid reasoning families selected for reasoning-block PLS.")

    label_parts = ["study"]
    if args.run_label.strip():
        label_parts.append(str(args.run_label).strip())
    else:
        label_parts.append("all_families")
    run_dir = timestamped_run_dir(INSTABILITY_CONTROL_RUNS_DIR / "reasoning_block_pls", "_".join(label_parts))
    progress_log_path = run_dir / "progress.log"
    write_json(run_dir / "effective_config.json", config)
    write_markdown(progress_log_path, "")

    def _progress_callback(payload: dict[str, object]) -> None:
        _append_progress(progress_log_path, payload)
        write_json(run_dir / "progress.json", payload)

    outputs = run_reasoning_block_pls_study(data, family_ids, config, progress_callback=_progress_callback)
    family_labels = [str(data.family_specs[family_id].label) for family_id in family_ids]

    protocol_markdown = block_pls_protocol_markdown(config, family_labels)
    summary_markdown = block_pls_summary_markdown(outputs["summary"], outputs["benchmark_metrics"])
    details_markdown = block_pls_details_markdown(outputs["summary"], outputs["component_summary"])
    audit_markdown = block_pls_audit_markdown(outputs["audit"])

    write_json(run_dir / "outer_splits.json", _serializable_splits(outputs["outer_splits"]))
    _write_dataframe_bundle(run_dir, "benchmark_metrics", outputs["benchmark_metrics"])
    _write_dataframe_bundle(run_dir, "family_metrics", outputs["family_metrics"])
    _write_dataframe_bundle(run_dir, "all_metrics", outputs["all_metrics"])
    _write_dataframe_bundle(run_dir, "component_weights", outputs["component_weights"])
    _write_dataframe_bundle(run_dir, "component_summary", outputs["component_summary"])
    _write_dataframe_bundle(run_dir, "reasoning_block_pls_summary", outputs["summary"])
    _write_dataframe_bundle(run_dir, "final_audit", outputs["audit"])
    write_markdown(run_dir / "10_reasoning_block_pls_protocol.md", protocol_markdown)
    write_markdown(run_dir / "11_reasoning_block_pls_summary.md", summary_markdown)
    write_markdown(run_dir / "12_reasoning_block_pls_family_details.md", details_markdown)
    write_markdown(run_dir / "13_reasoning_block_pls_final_audit.md", audit_markdown)

    protocol_path = _docs_path(config, "protocol_markdown")
    summary_path = _docs_path(config, "summary_markdown")
    details_path = _docs_path(config, "details_markdown")
    audit_path = _docs_path(config, "audit_markdown")
    summary_csv_path = _docs_path(config, "summary_csv")
    fold_metrics_csv_path = _docs_path(config, "fold_metrics_csv")
    component_summary_csv_path = _docs_path(config, "component_summary_csv")

    for docs_path in [
        protocol_path,
        summary_path,
        details_path,
        audit_path,
        summary_csv_path,
        fold_metrics_csv_path,
        component_summary_csv_path,
    ]:
        ensure_dir(docs_path.parent)

    write_markdown(protocol_path, protocol_markdown)
    write_markdown(summary_path, summary_markdown)
    write_markdown(details_path, details_markdown)
    write_markdown(audit_path, audit_markdown)
    write_csv(summary_csv_path, outputs["summary"])
    write_csv(fold_metrics_csv_path, outputs["all_metrics"])
    write_csv(component_summary_csv_path, outputs["component_summary"])

    print(f"Wrote reasoning-block-pls outputs to {run_dir}")


if __name__ == "__main__":
    main()
