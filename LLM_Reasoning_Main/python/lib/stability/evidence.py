from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.paths import MODEL_TESTING_DOCS_DIR, PAPER_STATS_DIR
from lib.stability.combo_catalog import EvaluationUnit
from lib.stability.family_registry import AlignedFamilyData
from lib.stability.routes import evaluate_route_cv


def _parse_markdown_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        if len(values) != len(header):
            continue
        rows.append(dict(zip(header, values)))
    return rows


def load_legacy_model_testing_metrics() -> dict[str, dict[str, float | str]]:
    rows = _parse_markdown_table(MODEL_TESTING_DOCS_DIR / "summary_final_table.md")
    metrics: dict[str, dict[str, float | str]] = {}
    for row in rows:
        model_label = row.get("Model", "").strip()
        if not model_label:
            continue
        cv_text = row.get("CV F0.5 (mean+/-std)", "--")
        cv_mean = None
        cv_std = None
        if "+/-" in cv_text:
            mean_text, std_text = cv_text.split("+/-", 1)
            cv_mean = float(mean_text.strip())
            cv_std = float(std_text.strip())
        metrics[model_label] = {
            "cv_f0_5_mean": cv_mean,
            "cv_f0_5_std": cv_std,
            "test_precision": float(row["Test P"]),
            "test_recall": float(row["Test R"]),
            "test_f0_5": float(row["Test F0.5"]),
            "fold_1_f0_5": float(row["Fold 1"]),
            "fold_2_f0_5": float(row["Fold 2"]),
            "fold_3_f0_5": float(row["Fold 3"]),
            "source": "docs/model_testing/summary_final_table.md",
        }
    return metrics


def load_legacy_paper_top_picks() -> list[dict[str, str]]:
    path = PAPER_STATS_DIR / "experiment_top_picks.md"
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        values = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(values) != 5 or values[0] in {"Experiment", "---"}:
            continue
        rows.append(
            {
                "experiment": values[0],
                "logistic_hq_no_rule": values[1],
                "logistic_mirror_rule": values[2],
                "xgboost_hq_no_rule": values[3],
                "xgboost_mirror_rule": values[4],
            }
        )
    return rows


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    fill_values = numeric.mean(numeric_only=True)
    return numeric.fillna(fill_values).fillna(0.0)


def _corr_stats(frame: pd.DataFrame) -> dict[str, float]:
    if frame.shape[1] < 2:
        return {
            "max_abs_corr": 0.0,
            "mean_abs_corr": 0.0,
        }
    corr = _numeric_frame(frame).corr().abs()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    values = corr.where(mask).stack().to_numpy()
    if values.size == 0:
        return {"max_abs_corr": 0.0, "mean_abs_corr": 0.0}
    return {
        "max_abs_corr": float(values.max()),
        "mean_abs_corr": float(values.mean()),
    }


def _cross_corr_stats(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float]:
    if left.empty or right.empty:
        return {
            "max_abs_cross_corr": 0.0,
            "mean_abs_cross_corr": 0.0,
        }
    left_use = _numeric_frame(left)
    right_use = _numeric_frame(right)
    matrix = pd.concat({"left": left_use, "right": right_use}, axis=1).corr().abs()
    cross = matrix.loc["left", "right"]
    values = cross.to_numpy().reshape(-1)
    if values.size == 0:
        return {"max_abs_cross_corr": 0.0, "mean_abs_cross_corr": 0.0}
    return {
        "max_abs_cross_corr": float(values.max()),
        "mean_abs_cross_corr": float(values.mean()),
    }


def _condition_number(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return 0.0
    numeric = _numeric_frame(frame)
    if numeric.shape[1] == 0:
        return 0.0
    try:
        value = float(np.linalg.cond(numeric.to_numpy(dtype=float)))
    except np.linalg.LinAlgError:
        return None
    if math.isinf(value) or math.isnan(value):
        return None
    return value


def _max_vif(frame: pd.DataFrame) -> float | None:
    if frame.shape[1] < 2 or frame.shape[1] > 40:
        return None
    numeric = _numeric_frame(frame)
    std = numeric.std(numeric_only=True)
    keep = std[std > 0.0].index.tolist()
    if len(keep) < 2:
        return None
    numeric = numeric[keep].copy()
    std = numeric.std(numeric_only=True).replace(0.0, 1.0)
    scaled = ((numeric - numeric.mean(numeric_only=True)) / std).fillna(0.0)
    corr = np.corrcoef(scaled.to_numpy(dtype=float), rowvar=False)
    try:
        inv = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return None
    values = np.diag(inv)
    if values.size == 0:
        return None
    return float(np.max(values))


def compute_overlap_metrics(
    baseline_df: pd.DataFrame,
    units: list[EvaluationUnit],
    primary_route_spec: dict[str, Any],
) -> pd.DataFrame:
    drop_features = set(str(value) for value in primary_route_spec.get("drop_baseline_features", []))
    baseline_use = baseline_df.drop(columns=[c for c in drop_features if c in baseline_df.columns]).copy()
    rows: list[dict[str, Any]] = []
    for unit in units:
        combined = pd.concat([baseline_use, unit.feature_frame], axis=1)
        combined = combined.loc[:, ~combined.columns.duplicated()].copy()
        family_corr = _corr_stats(unit.feature_frame)
        cross_corr = _cross_corr_stats(baseline_use, unit.feature_frame)
        rows.append(
            {
                "unit_id": unit.id,
                "unit_label": unit.label,
                "unit_kind": unit.kind,
                "family_ids": ",".join(unit.family_ids),
                "added_feature_count": int(unit.feature_count),
                "baseline_feature_count": int(baseline_use.shape[1]),
                "combined_feature_count": int(combined.shape[1]),
                "raw_max_abs_corr": float(family_corr["max_abs_corr"]),
                "raw_mean_abs_corr": float(family_corr["mean_abs_corr"]),
                "max_abs_cross_corr_vs_hq": float(cross_corr["max_abs_cross_corr"]),
                "mean_abs_cross_corr_vs_hq": float(cross_corr["mean_abs_cross_corr"]),
                "raw_condition_number": _condition_number(unit.feature_frame),
                "combined_condition_number": _condition_number(combined),
                "raw_max_vif": _max_vif(unit.feature_frame),
            }
        )
    return pd.DataFrame(rows)


def evaluate_units_across_routes(
    data: AlignedFamilyData,
    units: list[EvaluationUnit],
    fold_ids: np.ndarray,
    routes: list[dict[str, Any]],
) -> pd.DataFrame:
    rule_mask = data.baseline["exit_count"].fillna(0.0).astype(float).to_numpy() > 0 if "exit_count" in data.baseline.columns else None
    rows: list[dict[str, Any]] = []
    for unit in units:
        for route_spec in routes:
            result = evaluate_route_cv(
                data.baseline,
                unit.feature_frame,
                data.labels,
                fold_ids,
                route_spec,
                rule_mask=rule_mask,
            )
            summary = result["summary"]
            row: dict[str, Any] = {
                "unit_id": unit.id,
                "unit_label": unit.label,
                "unit_kind": unit.kind,
                "family_ids": ",".join(unit.family_ids),
                "route_id": result["route_id"],
                "route_label": result["route_label"],
                "route_model": result["model"],
                "route_transform": result["transform"],
                "route_role": str(route_spec.get("classification_role", "")),
                "exclude_from_classification": bool(route_spec.get("exclude_from_classification", False)),
                "added_feature_count": int(unit.feature_count),
                "oof_threshold": float(summary["threshold_oof"]),
                "cv_f0_5_mean": float(summary["f0_5_mean"]),
                "cv_f0_5_std": float(summary["f0_5_std"]),
                "cv_precision_mean": float(summary["precision_mean"]),
                "cv_recall_mean": float(summary["recall_mean"]),
                "cv_pr_auc_mean": float(summary["pr_auc_mean"]),
                "cv_precision_at_10_mean": float(summary["precision_at_10_mean"]),
                "full_train_f0_5": float(summary["full_train_f0_5"]),
                "full_train_precision": float(summary["full_train_precision"]),
                "full_train_recall": float(summary["full_train_recall"]),
                "full_train_pr_auc": float(summary["full_train_pr_auc"]),
                "legacy_model_label": None if unit.legacy_labels is None else unit.legacy_labels.get(result["route_id"]),
            }
            for index, fold in enumerate(result["per_fold"], start=1):
                row[f"cv_fold_{index}_f0_5"] = float(fold["f0_5"])
            rows.append(row)
    return pd.DataFrame(rows)


def attach_legacy_metrics(route_metrics: pd.DataFrame, legacy_metrics: dict[str, dict[str, float | str]]) -> pd.DataFrame:
    frame = route_metrics.copy()
    legacy_test_precision: list[float | None] = []
    legacy_test_recall: list[float | None] = []
    legacy_test_f0_5: list[float | None] = []
    legacy_cv_f0_5_mean: list[float | None] = []
    legacy_cv_f0_5_std: list[float | None] = []
    legacy_source: list[str | None] = []
    for label in frame["legacy_model_label"].tolist():
        if not label or label not in legacy_metrics:
            legacy_test_precision.append(None)
            legacy_test_recall.append(None)
            legacy_test_f0_5.append(None)
            legacy_cv_f0_5_mean.append(None)
            legacy_cv_f0_5_std.append(None)
            legacy_source.append(None)
            continue
        payload = legacy_metrics[str(label)]
        legacy_test_precision.append(float(payload["test_precision"]))
        legacy_test_recall.append(float(payload["test_recall"]))
        legacy_test_f0_5.append(float(payload["test_f0_5"]))
        legacy_cv_f0_5_mean.append(None if payload["cv_f0_5_mean"] is None else float(payload["cv_f0_5_mean"]))
        legacy_cv_f0_5_std.append(None if payload["cv_f0_5_std"] is None else float(payload["cv_f0_5_std"]))
        legacy_source.append(str(payload["source"]))
    frame["legacy_test_precision"] = legacy_test_precision
    frame["legacy_test_recall"] = legacy_test_recall
    frame["legacy_test_f0_5"] = legacy_test_f0_5
    frame["legacy_cv_f0_5_mean"] = legacy_cv_f0_5_mean
    frame["legacy_cv_f0_5_std"] = legacy_cv_f0_5_std
    frame["legacy_source"] = legacy_source
    return frame


def _find_metric(frame: pd.DataFrame, unit_id: str, route_id: str, column: str) -> float | None:
    subset = frame[(frame["unit_id"] == unit_id) & (frame["route_id"] == route_id)]
    if subset.empty:
        return None
    value = subset.iloc[0][column]
    return None if pd.isna(value) else float(value)


def classify_units(
    route_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    controls: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    hq_raw_lr_cv = _find_metric(route_metrics, "HQ", str(controls["primary_raw_route_id"]), "cv_f0_5_mean")
    hq_raw_xgb_cv = _find_metric(route_metrics, "HQ", str(controls["secondary_raw_route_id"]), "cv_f0_5_mean")
    hq_legacy_raw_test = _find_metric(route_metrics, "HQ", str(controls["primary_raw_route_id"]), "legacy_test_f0_5")
    overlap_lookup = overlap_metrics.set_index("unit_id").to_dict(orient="index")

    rows: list[dict[str, Any]] = []
    for unit_id in sorted(route_metrics["unit_id"].unique().tolist()):
        if unit_id == "HQ":
            rows.append(
                {
                    "unit_id": "HQ",
                    "classification": "anchor_reference",
                    "rationale": "Frozen route reference.",
                }
            )
            continue
        raw_lr_cv = _find_metric(route_metrics, unit_id, str(controls["primary_raw_route_id"]), "cv_f0_5_mean")
        raw_xgb_cv = _find_metric(route_metrics, unit_id, str(controls["secondary_raw_route_id"]), "cv_f0_5_mean")
        raw_legacy_test = _find_metric(route_metrics, unit_id, str(controls["primary_raw_route_id"]), "legacy_test_f0_5")
        transformed_legacy_test = _find_metric(
            route_metrics,
            unit_id,
            str(controls["primary_transformed_route_id"]),
            "legacy_test_f0_5",
        )
        transformed_cv = _find_metric(route_metrics, unit_id, str(controls["primary_transformed_route_id"]), "cv_f0_5_mean")
        raw_signal = bool(
            (raw_lr_cv is not None and hq_raw_lr_cv is not None and raw_lr_cv > hq_raw_lr_cv)
            or (raw_xgb_cv is not None and hq_raw_xgb_cv is not None and raw_xgb_cv > hq_raw_xgb_cv)
        )
        raw_test_beats_hq = bool(
            raw_legacy_test is not None
            and hq_legacy_raw_test is not None
            and raw_legacy_test > hq_legacy_raw_test
        )
        transformed_beats_hq = bool(
            transformed_legacy_test is not None
            and hq_legacy_raw_test is not None
            and transformed_legacy_test >= hq_legacy_raw_test
        )
        if transformed_beats_hq and not raw_test_beats_hq:
            label = "transform_sensitive"
            rationale = "Raw route does not clear HQ on legacy test, but PLS route does."
        elif raw_signal and not raw_test_beats_hq:
            label = "raw_fail"
            rationale = "Raw route lifts CV or full-train signal without a report-linked HQ test win."
        else:
            label = "no_reproducible_evidence"
            rationale = "Neither raw nor transformed evidence clears the current benchmark story."
        overlap_row = overlap_lookup.get(unit_id, {})
        rows.append(
            {
                "unit_id": unit_id,
                "classification": label,
                "rationale": rationale,
                "raw_lr_cv_f0_5": raw_lr_cv,
                "raw_xgb_cv_f0_5": raw_xgb_cv,
                "pls_lr_cv_f0_5": transformed_cv,
                "legacy_raw_test_f0_5": raw_legacy_test,
                "legacy_pls_test_f0_5": transformed_legacy_test,
                "max_abs_cross_corr_vs_hq": overlap_row.get("max_abs_cross_corr_vs_hq"),
                "raw_condition_number": overlap_row.get("raw_condition_number"),
            }
        )

    classification_df = pd.DataFrame(rows)
    compression_priority = classification_df[classification_df["classification"] == "transform_sensitive"].copy()
    compression_priority = compression_priority.sort_values(
        by=["legacy_pls_test_f0_5", "pls_lr_cv_f0_5"],
        ascending=False,
        na_position="last",
    )

    stability_priority = classification_df[classification_df["classification"] == "raw_fail"].copy()
    stability_priority["raw_best_cv"] = stability_priority[["raw_lr_cv_f0_5", "raw_xgb_cv_f0_5"]].max(axis=1)
    stability_priority = stability_priority.sort_values(by=["raw_best_cv"], ascending=False, na_position="last")

    grouped_priority = classification_df[classification_df["classification"].isin(["transform_sensitive", "raw_fail"])].copy()
    grouped_priority = grouped_priority.sort_values(
        by=["max_abs_cross_corr_vs_hq", "raw_condition_number"],
        ascending=False,
        na_position="last",
    )
    priorities = {
        "compression_priority": compression_priority["unit_id"].head(int(controls.get("compression_priority_limit", 5))).tolist(),
        "stability_selection_priority": stability_priority["unit_id"].head(int(controls.get("stability_priority_limit", 5))).tolist(),
        "grouped_penalty_priority": grouped_priority["unit_id"].head(int(controls.get("grouped_priority_limit", 5))).tolist(),
    }
    return classification_df, priorities


def _fmt_num(value: Any, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{float(value):.{places}f}"


def build_route_snapshot(route_metrics: pd.DataFrame, classification_df: pd.DataFrame) -> pd.DataFrame:
    class_lookup = classification_df.set_index("unit_id")["classification"].to_dict()
    units = []
    for unit_id in route_metrics["unit_id"].unique().tolist():
        subset = route_metrics[route_metrics["unit_id"] == unit_id]
        label = str(subset.iloc[0]["unit_label"])
        unit_kind = str(subset.iloc[0]["unit_kind"])
        display_label = "HQ" if unit_id == "HQ" else f"HQ + {label}"
        row: dict[str, Any] = {
            "unit_id": unit_id,
            "unit_label": label,
            "unit_display_label": display_label,
            "unit_kind": unit_kind,
            "classification": class_lookup.get(unit_id, "anchor_reference" if unit_id == "HQ" else "--"),
        }
        for route_id, prefix in (
            ("raw_lr_base", "raw_lr"),
            ("raw_xgb1", "raw_xgb1"),
            ("pls_lr_n6", "pls_lr"),
        ):
            route_row = subset[subset["route_id"] == route_id]
            if route_row.empty:
                row[f"{prefix}_cv_mean"] = None
                row[f"{prefix}_cv_std"] = None
                continue
            route_payload = route_row.iloc[0]
            row[f"{prefix}_cv_mean"] = route_payload["cv_f0_5_mean"]
            row[f"{prefix}_cv_std"] = route_payload["cv_f0_5_std"]
            if prefix == "raw_lr":
                row["legacy_raw_test_f0_5"] = route_payload.get("legacy_test_f0_5")
            if prefix == "pls_lr":
                row["legacy_pls_test_f0_5"] = route_payload.get("legacy_test_f0_5")
        units.append(row)
    snapshot = pd.DataFrame(units)
    order_map = {"anchor_reference": 0, "transform_sensitive": 1, "raw_fail": 2, "no_reproducible_evidence": 3}
    snapshot["sort_key"] = snapshot["classification"].map(order_map).fillna(9)
    snapshot = snapshot.sort_values(by=["sort_key", "unit_id"]).drop(columns=["sort_key"])
    return snapshot


def render_step1_synthesis(
    route_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    classification_df: pd.DataFrame,
    priorities: dict[str, list[str]],
) -> str:
    display = classification_df[classification_df["unit_id"] != "HQ"].copy()
    display = display.sort_values(by=["classification", "unit_id"])
    label_lookup = route_metrics.groupby("unit_id", dropna=False)["unit_label"].first().to_dict()
    snapshot = build_route_snapshot(route_metrics, classification_df)
    lines = [
        "# Step 1: Evidence Map",
        "",
        "This Step 1 pass is a clean reproduction layer. It does not admit families or reject them permanently.",
        "",
        "## Current Read",
        "",
        "- The raw reasoning routes remain unstable relative to the HQ benchmark story.",
        "- The clearest report-linked wins still come from transformed `PLS` routes, especially `F` and `D+E+F`.",
        "- This step therefore exists to map raw failure patterns and transform-sensitive signal before new mathematical methods are added.",
        "",
        "## Unit Classification",
        "",
        "| Unit | Classification | Raw LR CV | Raw XGB1 CV | PLS LR CV | Legacy Raw Test F0.5 | Legacy PLS Test F0.5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in display.to_dict(orient="records"):
        raw_label = str(label_lookup.get(row["unit_id"], row["unit_id"]))
        unit_label = "HQ" if row["unit_id"] == "HQ" else f"HQ + {raw_label}"
        lines.append(
            "| {unit} | {label} | {raw_lr} | {raw_xgb} | {pls_lr} | {raw_test} | {pls_test} |".format(
                unit=unit_label,
                label=row["classification"],
                raw_lr=_fmt_num(row.get("raw_lr_cv_f0_5")),
                raw_xgb=_fmt_num(row.get("raw_xgb_cv_f0_5")),
                pls_lr=_fmt_num(row.get("pls_lr_cv_f0_5")),
                raw_test=_fmt_num(row.get("legacy_raw_test_f0_5")),
                pls_test=_fmt_num(row.get("legacy_pls_test_f0_5")),
            )
        )

    overlap_lookup = overlap_metrics.set_index("unit_id").to_dict(orient="index")
    lines += [
        "",
        "## Whole-Data Route Snapshot",
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
        "## Priority Lists",
        "",
        f"- Compression priority: {', '.join(priorities['compression_priority']) if priorities['compression_priority'] else '(none)'}",
        f"- Stability-selection priority: {', '.join(priorities['stability_selection_priority']) if priorities['stability_selection_priority'] else '(none)'}",
        f"- Grouped-penalty priority: {', '.join(priorities['grouped_penalty_priority']) if priorities['grouped_penalty_priority'] else '(none)'}",
        "",
        "## Redundancy Snapshot",
        "",
        "| Unit | Max Cross Corr vs HQ | Raw Cond. No. | Raw Max VIF |",
        "|---|---:|---:|---:|",
    ]
    for unit_id in display["unit_id"].tolist():
        overlap_row = overlap_lookup.get(unit_id, {})
        lines.append(
            "| {unit} | {cross} | {cond} | {vif} |".format(
                unit=unit_id,
                cross=_fmt_num(overlap_row.get("max_abs_cross_corr_vs_hq")),
                cond=_fmt_num(overlap_row.get("raw_condition_number")),
                vif=_fmt_num(overlap_row.get("raw_max_vif")),
            )
        )
    return "\n".join(lines)


def render_legacy_alignment(
    route_metrics: pd.DataFrame,
    legacy_metrics: dict[str, dict[str, float | str]],
    paper_top_picks: list[dict[str, str]],
) -> str:
    checks: list[tuple[str, bool]] = []
    hq_test = _find_metric(route_metrics, "HQ", "raw_lr_base", "legacy_test_f0_5")
    a_test = _find_metric(route_metrics, "reasoning_A", "raw_lr_base", "legacy_test_f0_5")
    f_pls_test = _find_metric(route_metrics, "reasoning_F", "pls_lr_n6", "legacy_test_f0_5")
    def_pls_test = _find_metric(route_metrics, "combo_D_E_F", "pls_lr_n6", "legacy_test_f0_5")
    a_pls_test = _find_metric(route_metrics, "reasoning_A", "pls_lr_n6", "legacy_test_f0_5")
    if hq_test is not None and a_test is not None:
        checks.append(("`LR_BASE_HQ` remains ahead of raw `A` on legacy test.", hq_test > a_test))
    if hq_test is not None and f_pls_test is not None:
        checks.append(("`LR_PLS_F` remains ahead of `LR_BASE_HQ` on legacy test.", f_pls_test > hq_test))
    if hq_test is not None and def_pls_test is not None:
        checks.append(("`LR_PLS_DEF` remains ahead of `LR_BASE_HQ` on legacy test.", def_pls_test > hq_test))
    if hq_test is not None and a_pls_test is not None:
        checks.append(("`LR_PLS_A` does not overtake `LR_BASE_HQ` by default.", a_pls_test <= hq_test))

    lines = [
        "# Step 1 Legacy Alignment",
        "",
        "This note compares the new evidence-map rerun against the curated legacy evidence surfaces.",
        "",
        "## Model-Testing Alignment",
        "",
        f"- Parsed legacy summary rows: {len(legacy_metrics)} from `docs/model_testing/summary_final_table.md`.",
        "",
    ]
    for text, passed in checks:
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {text}")

    lines += [
        "",
        "## Paper-Pipeline Alignment",
        "",
        f"- Parsed paper top-pick rows: {len(paper_top_picks)} from `docs/paper_stats/experiment_top_picks.md`.",
        "- The paper-stage raw CV tables still show large apparent gains for richer reasoning combinations.",
        "- That earlier raw-CV optimism is consistent with the current Step 1 framing: raw reasoning signal exists, but does not hold up cleanly without stronger extraction or control methods.",
    ]
    return "\n".join(lines)


def run_step1_evidence_map(
    data: AlignedFamilyData,
    units: list[EvaluationUnit],
    fold_ids: np.ndarray,
    controls_payload: dict[str, Any],
) -> dict[str, Any]:
    routes = list(controls_payload.get("routes", []))
    if not routes:
        raise RuntimeError("No routes defined in controls config.")
    route_metrics = evaluate_units_across_routes(data, units, fold_ids, routes)
    primary_raw_route = next((route for route in routes if route.get("id") == controls_payload["classification"]["primary_raw_route_id"]), routes[0])
    overlap_metrics = compute_overlap_metrics(data.baseline, units, primary_raw_route)
    legacy_metrics = load_legacy_model_testing_metrics()
    paper_top_picks = load_legacy_paper_top_picks()
    route_metrics = attach_legacy_metrics(route_metrics, legacy_metrics)
    classification_df, priorities = classify_units(route_metrics, overlap_metrics, controls_payload["classification"])
    synthesis_markdown = render_step1_synthesis(route_metrics, overlap_metrics, classification_df, priorities)
    legacy_alignment_markdown = render_legacy_alignment(route_metrics, legacy_metrics, paper_top_picks)
    return {
        "route_metrics": route_metrics,
        "overlap_metrics": overlap_metrics,
        "classification": classification_df,
        "priorities": priorities,
        "synthesis_markdown": synthesis_markdown,
        "legacy_alignment_markdown": legacy_alignment_markdown,
    }
