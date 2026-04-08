from __future__ import annotations

from typing import Any

import pandas as pd

from lib.stability.evidence import build_route_snapshot


def _fmt_num(value: Any, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    return f"{float(value):.{places}f}"


def status_report_markdown(
    route_metrics: pd.DataFrame,
    classification: pd.DataFrame,
    priorities: dict[str, list[str]],
    method_summary: dict[str, Any],
) -> str:
    counts = (
        classification[classification["classification"] != "anchor_reference"]["classification"]
        .value_counts()
        .to_dict()
    )
    snapshot = build_route_snapshot(route_metrics, classification)
    lines = [
        "# Instability-Control Status Report",
        "",
        "This is a current-state synthesis for the active study path. It is not a final experimental report.",
        "",
        "## Step 1 Outcome",
        "",
        f"- Transform-sensitive units: {int(counts.get('transform_sensitive', 0))}",
        f"- Raw-fail units: {int(counts.get('raw_fail', 0))}",
        f"- No reproducible evidence: {int(counts.get('no_reproducible_evidence', 0))}",
        "",
        "## Whole-Data Snapshot",
        "",
        "| Unit | Class | Raw LR Mean | Raw LR Std | Raw XGB1 Mean | Raw XGB1 Std | PLS LR Mean | PLS LR Std | Legacy Raw Test | Legacy PLS Test |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in snapshot.to_dict(orient="records"):
        lines.append(
            "| {unit} | {label} | {raw_lr_mean} | {raw_lr_std} | {raw_xgb_mean} | {raw_xgb_std} | {pls_mean} | {pls_std} | {raw_test} | {pls_test} |".format(
                unit=row["unit_display_label"],
                label=row["classification"],
                raw_lr_mean=_fmt_num(row.get("raw_lr_cv_mean")),
                raw_lr_std=_fmt_num(row.get("raw_lr_cv_std")),
                raw_xgb_mean=_fmt_num(row.get("raw_xgb1_cv_mean")),
                raw_xgb_std=_fmt_num(row.get("raw_xgb1_cv_std")),
                pls_mean=_fmt_num(row.get("pls_lr_cv_mean")),
                pls_std=_fmt_num(row.get("pls_lr_cv_std")),
                raw_test=_fmt_num(row.get("legacy_raw_test_f0_5")),
                pls_test=_fmt_num(row.get("legacy_pls_test_f0_5")),
            )
        )

    lines += [
        "",
        "## Next Method Targets",
        "",
        f"- Compression: {', '.join(priorities.get('compression_priority', [])) or '(none)'}",
        f"- Stability selection: {', '.join(priorities.get('stability_selection_priority', [])) or '(none)'}",
        f"- Grouped penalty: {', '.join(priorities.get('grouped_penalty_priority', [])) or '(none)'}",
        "",
        "## Scaffold Status",
        "",
        f"- Active mathematical methods in this pass: {int(method_summary.get('active_method_count', 0))}",
        "- The active next method stage is `supervised_grouping`, with Step 2 stability selection and Step 3 blockwise reasoning-PLS preserved as frozen references.",
    ]
    return "\n".join(lines)
