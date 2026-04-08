from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

from lib.shared.artifact_io import read_json
from lib.stability.family_registry import AlignedFamilyData
from lib.stability.routes import _fill_missing
from lib.stability.stability_selection import build_outer_splits, evaluate_selected_route, family_display_label


def load_reasoning_block_pls_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid reasoning-block-pls config: {path}")
    return payload


@dataclass(frozen=True)
class BlockPLSTransformResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    requested_n_components: int
    actual_n_components: int
    component_names: list[str]
    component_weights: pd.DataFrame
    reasoning_train_means: pd.Series
    reasoning_train_stds: pd.Series


def _safe_fmt(value: Any, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "--"
    return f"{float(value):.{places}f}"


def _family_component_grid(config: dict[str, Any], family_id: str) -> list[int]:
    family_components = config.get("family_components", {})
    if not isinstance(family_components, dict):
        raise RuntimeError("reasoning_block_pls config must include family_components.")
    values = family_components.get(family_id, [])
    grid = [int(value) for value in values]
    if not grid:
        raise RuntimeError(f"No n_components configured for family {family_id}.")
    return grid


def _sign_flip_rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    positive = float(np.mean(values > 0.0))
    negative = float(np.mean(values < 0.0))
    nonzero = positive + negative
    if nonzero == 0.0:
        return 0.0
    return float(min(positive, negative) / nonzero)


def fit_reasoning_block_pls(
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    reasoning_train: pd.DataFrame,
    reasoning_test: pd.DataFrame,
    y_train: np.ndarray,
    *,
    family_label: str,
    requested_n_components: int,
) -> BlockPLSTransformResult:
    reasoning_train_filled, reasoning_test_filled = _fill_missing(
        reasoning_train.copy(),
        reasoning_test.copy(),
        model="logistic",
    )
    means = reasoning_train_filled.mean(numeric_only=True)
    stds = reasoning_train_filled.std(numeric_only=True).replace(0.0, 1.0)
    reasoning_train_scaled = ((reasoning_train_filled - means) / stds).fillna(0.0)
    reasoning_test_scaled = ((reasoning_test_filled - means) / stds).fillna(0.0)

    actual_n_components = min(
        int(requested_n_components),
        reasoning_train_scaled.shape[1],
        max(1, len(y_train) - 1),
    )
    pls = PLSRegression(n_components=actual_n_components)
    train_scores = pls.fit_transform(reasoning_train_scaled.to_numpy(dtype=float), y_train)[0]
    test_scores = pls.transform(reasoning_test_scaled.to_numpy(dtype=float))

    component_names = [f"{family_label}_pls_{index + 1}" for index in range(actual_n_components)]
    component_train_df = pd.DataFrame(train_scores, index=baseline_train.index, columns=component_names)
    component_test_df = pd.DataFrame(test_scores, index=baseline_test.index, columns=component_names)
    train_df = pd.concat([baseline_train.copy(), component_train_df], axis=1)
    test_df = pd.concat([baseline_test.copy(), component_test_df], axis=1)

    rows: list[dict[str, Any]] = []
    for component_index, component_name in enumerate(component_names):
        for feature_index, feature_name in enumerate(reasoning_train.columns.tolist()):
            rows.append(
                {
                    "component_index": component_index + 1,
                    "component_name": component_name,
                    "original_feature": feature_name,
                    "x_weight": float(pls.x_weights_[feature_index, component_index]),
                    "x_loading": float(pls.x_loadings_[feature_index, component_index]),
                }
            )
    component_weights = pd.DataFrame(rows)
    return BlockPLSTransformResult(
        train_df=train_df,
        test_df=test_df,
        requested_n_components=int(requested_n_components),
        actual_n_components=int(actual_n_components),
        component_names=component_names,
        component_weights=component_weights,
        reasoning_train_means=means,
        reasoning_train_stds=stds,
    )


def run_reasoning_block_pls_study(
    data: AlignedFamilyData,
    family_ids: list[str],
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    outer_cfg = config.get("outer_cv", {})
    outer_splits = build_outer_splits(
        data.labels,
        n_splits=int(outer_cfg.get("n_splits", 3)),
        n_repeats=int(outer_cfg.get("n_repeats", 16)),
        random_state=int(outer_cfg.get("random_state", 42)),
    )
    evaluation_cfg = config.get("evaluation", {})
    models = [str(value) for value in evaluation_cfg.get("models", ["logistic", "xgb1"])]

    baseline = data.baseline.copy()
    labels = data.labels.to_numpy(dtype=int)
    rule_mask = baseline["exit_count"].fillna(0.0).astype(float).to_numpy() > 0 if "exit_count" in baseline.columns else None
    baseline_columns = baseline.columns.tolist()

    benchmark_rows: list[dict[str, Any]] = []
    family_metric_rows: list[dict[str, Any]] = []
    component_weight_rows: list[dict[str, Any]] = []

    total_units = max(1, len(outer_splits) * max(1, len(family_ids)))
    completed_units = 0

    for split_index, split in enumerate(outer_splits):
        outer_split_id = str(split["outer_split_id"])
        train_idx = np.asarray(split["train_idx"], dtype=int)
        test_idx = np.asarray(split["test_idx"], dtype=int)
        baseline_train = baseline.iloc[train_idx].copy()
        baseline_test = baseline.iloc[test_idx].copy()
        y_train = labels[train_idx]
        y_test = labels[test_idx]
        rule_mask_train = None if rule_mask is None else rule_mask[train_idx]
        rule_mask_test = None if rule_mask is None else rule_mask[test_idx]

        if progress_callback is not None:
            progress_callback(
                {
                    "event": "outer_split_start",
                    "outer_split_id": outer_split_id,
                    "split_index": split_index + 1,
                    "total_splits": len(outer_splits),
                    "completed_units": completed_units,
                    "total_units": total_units,
                }
            )

        for model_type in models:
            metrics = evaluate_selected_route(
                baseline_train[baseline_columns],
                baseline_test[baseline_columns],
                y_train,
                y_test,
                model_type=model_type,
                threshold_grid=str(evaluation_cfg.get("threshold_grid", "default")),
                apply_rule_override=bool(evaluation_cfg.get("apply_rule_override", True)),
                rule_mask_train=rule_mask_train,
                rule_mask_test=rule_mask_test,
                random_state=int(evaluation_cfg.get("random_state", 42)) + split_index,
                inner_threshold_cv_splits=int(evaluation_cfg.get("inner_threshold_cv_splits", 3)),
            )
            benchmark_rows.append(
                {
                    "family_id": "HQ",
                    "family_label": "HQ",
                    "family_display_label": "HQ",
                    "route_group": "hq_benchmark",
                    "route_label": "HQ",
                    "evaluator_model": model_type,
                    "requested_n_components": None,
                    "actual_n_components": 0,
                    "original_reasoning_feature_count": 0,
                    "latent_feature_count": 0,
                    "outer_split_id": outer_split_id,
                    "repeat_index": int(split["repeat_index"]),
                    "fold_index": int(split["fold_index"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "f0_5": float(metrics["f0_5"]),
                    "threshold": float(metrics["threshold"]),
                }
            )

        for family_id in family_ids:
            family_spec = data.family_specs[family_id]
            family_label = str(family_spec.label)
            display_label = family_display_label(family_label)
            reasoning_train = data.candidate_frames[family_id].iloc[train_idx].copy()
            reasoning_test = data.candidate_frames[family_id].iloc[test_idx].copy()
            original_reasoning_count = int(reasoning_train.shape[1])

            raw_train = pd.concat([baseline_train, reasoning_train], axis=1)
            raw_test = pd.concat([baseline_test, reasoning_test], axis=1)
            raw_train = raw_train.loc[:, ~raw_train.columns.duplicated()].copy()
            raw_test = raw_test.loc[:, raw_train.columns].copy()

            for model_type in models:
                raw_metrics = evaluate_selected_route(
                    raw_train,
                    raw_test,
                    y_train,
                    y_test,
                    model_type=model_type,
                    threshold_grid=str(evaluation_cfg.get("threshold_grid", "default")),
                    apply_rule_override=bool(evaluation_cfg.get("apply_rule_override", True)),
                    rule_mask_train=rule_mask_train,
                    rule_mask_test=rule_mask_test,
                    random_state=int(evaluation_cfg.get("random_state", 42)) + split_index,
                    inner_threshold_cv_splits=int(evaluation_cfg.get("inner_threshold_cv_splits", 3)),
                )
                family_metric_rows.append(
                    {
                        "family_id": family_id,
                        "family_label": family_label,
                        "family_display_label": display_label,
                        "route_group": "raw_family",
                        "route_label": f"{display_label} (raw)",
                        "evaluator_model": model_type,
                        "requested_n_components": None,
                        "actual_n_components": 0,
                        "original_reasoning_feature_count": original_reasoning_count,
                        "latent_feature_count": original_reasoning_count,
                        "outer_split_id": outer_split_id,
                        "repeat_index": int(split["repeat_index"]),
                        "fold_index": int(split["fold_index"]),
                        "precision": float(raw_metrics["precision"]),
                        "recall": float(raw_metrics["recall"]),
                        "pr_auc": float(raw_metrics["pr_auc"]),
                        "f0_5": float(raw_metrics["f0_5"]),
                        "threshold": float(raw_metrics["threshold"]),
                    }
                )

            for requested_n_components in _family_component_grid(config, family_id):
                block_pls = fit_reasoning_block_pls(
                    baseline_train,
                    baseline_test,
                    reasoning_train,
                    reasoning_test,
                    y_train,
                    family_label=family_label,
                    requested_n_components=requested_n_components,
                )
                component_weights = block_pls.component_weights.copy()
                component_weights["family_id"] = family_id
                component_weights["family_label"] = family_label
                component_weights["family_display_label"] = display_label
                component_weights["outer_split_id"] = outer_split_id
                component_weights["repeat_index"] = int(split["repeat_index"])
                component_weights["fold_index"] = int(split["fold_index"])
                component_weights["requested_n_components"] = int(requested_n_components)
                component_weights["actual_n_components"] = int(block_pls.actual_n_components)
                component_weight_rows.extend(component_weights.to_dict(orient="records"))

                for model_type in models:
                    block_metrics = evaluate_selected_route(
                        block_pls.train_df,
                        block_pls.test_df,
                        y_train,
                        y_test,
                        model_type=model_type,
                        threshold_grid=str(evaluation_cfg.get("threshold_grid", "default")),
                        apply_rule_override=bool(evaluation_cfg.get("apply_rule_override", True)),
                        rule_mask_train=rule_mask_train,
                        rule_mask_test=rule_mask_test,
                        random_state=int(evaluation_cfg.get("random_state", 42)) + split_index,
                        inner_threshold_cv_splits=int(evaluation_cfg.get("inner_threshold_cv_splits", 3)),
                    )
                    family_metric_rows.append(
                        {
                            "family_id": family_id,
                            "family_label": family_label,
                            "family_display_label": display_label,
                            "route_group": "block_pls",
                            "route_label": f"{display_label} PLS-block (k={requested_n_components})",
                            "evaluator_model": model_type,
                            "requested_n_components": int(requested_n_components),
                            "actual_n_components": int(block_pls.actual_n_components),
                            "original_reasoning_feature_count": original_reasoning_count,
                            "latent_feature_count": int(block_pls.actual_n_components),
                            "outer_split_id": outer_split_id,
                            "repeat_index": int(split["repeat_index"]),
                            "fold_index": int(split["fold_index"]),
                            "precision": float(block_metrics["precision"]),
                            "recall": float(block_metrics["recall"]),
                            "pr_auc": float(block_metrics["pr_auc"]),
                            "f0_5": float(block_metrics["f0_5"]),
                            "threshold": float(block_metrics["threshold"]),
                        }
                    )

            completed_units += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "family_complete",
                        "outer_split_id": outer_split_id,
                        "family_id": family_id,
                        "family_label": display_label,
                        "completed_units": completed_units,
                        "total_units": total_units,
                    }
                )

    benchmark_df = pd.DataFrame(benchmark_rows)
    family_metrics_df = pd.DataFrame(family_metric_rows)
    component_weights_df = pd.DataFrame(component_weight_rows)
    all_metrics_df = pd.DataFrame(benchmark_rows + family_metric_rows)

    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "family_id",
        "family_label",
        "family_display_label",
        "route_group",
        "route_label",
        "evaluator_model",
        "requested_n_components",
        "actual_n_components",
        "original_reasoning_feature_count",
        "latent_feature_count",
    ]
    if not all_metrics_df.empty:
        for keys, subset in all_metrics_df.groupby(group_columns, dropna=False):
            (
                family_id,
                family_label,
                family_display_label_text,
                route_group,
                route_label,
                evaluator_model,
                requested_n_components,
                actual_n_components,
                original_reasoning_feature_count,
                latent_feature_count,
            ) = keys
            summary_rows.append(
                {
                    "family_id": family_id,
                    "family_label": family_label,
                    "family_display_label": family_display_label_text,
                    "route_group": route_group,
                    "route_label": route_label,
                    "evaluator_model": evaluator_model,
                    "requested_n_components": None if pd.isna(requested_n_components) else int(requested_n_components),
                    "actual_n_components": int(actual_n_components),
                    "original_reasoning_feature_count": int(original_reasoning_feature_count),
                    "latent_feature_count": int(latent_feature_count),
                    "outer_split_count": int(subset["outer_split_id"].nunique()),
                    "outer_f0_5_mean": float(subset["f0_5"].mean()),
                    "outer_f0_5_std": float(subset["f0_5"].std(ddof=0)),
                    "precision_mean": float(subset["precision"].mean()),
                    "precision_std": float(subset["precision"].std(ddof=0)),
                    "recall_mean": float(subset["recall"].mean()),
                    "recall_std": float(subset["recall"].std(ddof=0)),
                    "pr_auc_mean": float(subset["pr_auc"].mean()),
                    "pr_auc_std": float(subset["pr_auc"].std(ddof=0)),
                    "threshold_mean": float(subset["threshold"].mean()),
                    "threshold_std": float(subset["threshold"].std(ddof=0)),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    component_summary_rows: list[dict[str, Any]] = []
    if not component_weights_df.empty:
        for keys, subset in component_weights_df.groupby(
            [
                "family_id",
                "family_label",
                "family_display_label",
                "requested_n_components",
                "actual_n_components",
                "component_index",
                "component_name",
                "original_feature",
            ],
            dropna=False,
        ):
            (
                family_id,
                family_label,
                family_display_label_text,
                requested_n_components,
                actual_n_components,
                component_index,
                component_name,
                original_feature,
            ) = keys
            x_weight_values = subset["x_weight"].to_numpy(dtype=float)
            x_loading_values = subset["x_loading"].to_numpy(dtype=float)
            component_summary_rows.append(
                {
                    "family_id": family_id,
                    "family_label": family_label,
                    "family_display_label": family_display_label_text,
                    "requested_n_components": int(requested_n_components),
                    "actual_n_components": int(actual_n_components),
                    "component_index": int(component_index),
                    "component_name": component_name,
                    "original_feature": original_feature,
                    "weight_mean": float(np.mean(x_weight_values)),
                    "weight_std": float(np.std(x_weight_values, ddof=0)),
                    "abs_weight_mean": float(np.mean(np.abs(x_weight_values))),
                    "weight_sign_flip_rate": _sign_flip_rate(x_weight_values),
                    "loading_mean": float(np.mean(x_loading_values)),
                    "loading_std": float(np.std(x_loading_values, ddof=0)),
                    "abs_loading_mean": float(np.mean(np.abs(x_loading_values))),
                    "loading_sign_flip_rate": _sign_flip_rate(x_loading_values),
                }
            )
    component_summary_df = pd.DataFrame(component_summary_rows)

    audit_df = pd.DataFrame(
        [
            {
                "candidate": "",
                "prediction_path": "",
                "external_test_f0_5": "",
                "status": "deferred_until_route_lock",
            }
        ]
    )
    return {
        "outer_splits": outer_splits,
        "benchmark_metrics": benchmark_df,
        "family_metrics": family_metrics_df,
        "all_metrics": all_metrics_df,
        "summary": summary_df,
        "component_weights": component_weights_df,
        "component_summary": component_summary_df,
        "audit": audit_df,
    }


def block_pls_protocol_markdown(config: dict[str, Any], family_labels: list[str]) -> str:
    outer_cfg = config.get("outer_cv", {})
    family_components = config.get("family_components", {})
    lines = [
        "# Reasoning-Block PLS Protocol",
        "",
        "This Step 3 method applies `PLS` to the reasoning block only, inside each outer fold, before concatenating the latent components with raw `HQ`.",
        "",
        "## Study Setup",
        "",
        f"- Families in scope: {', '.join(family_labels)}",
        f"- Outer CV: {int(outer_cfg.get('n_splits', 3))}-fold repeated {int(outer_cfg.get('n_repeats', 16))} times",
        "- Models: `LR`, `XGB1`",
        "- No row subsampling is used in this pass.",
        "- Holdout/test is deferred until a candidate route is locked.",
        "",
        "## Component Grids",
        "",
    ]
    for family_id, grid in family_components.items():
        short_label = family_id.replace("reasoning_", "")
        grid_text = ", ".join(str(int(value)) for value in grid)
        lines.append(f"- `{short_label}`: {grid_text}")
    lines += [
        "",
        "## Leakage Control",
        "",
        "- The reasoning block is filled, standardised, and fit with `PLSRegression` using outer-train rows only.",
        "- Validation rows are transformed using the train-fit scaler and train-fit `PLS` model only.",
        "- Raw `HQ` is concatenated after the reasoning block has been transformed.",
        "- Thresholds are selected by train-only inner CV inside each outer split.",
    ]
    return "\n".join(lines)


def block_pls_summary_markdown(summary_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> str:
    benchmark_summary = (
        benchmark_df.groupby("evaluator_model", dropna=False)
        .agg(outer_f0_5_mean=("f0_5", "mean"), outer_f0_5_std=("f0_5", "std"))
        .reset_index()
        if not benchmark_df.empty
        else pd.DataFrame()
    )
    lines = [
        "# Reasoning-Block PLS Summary",
        "",
        "This note compares raw `HQ + family` routes against leakage-safe blockwise `PLS` routes.",
        "",
        "## HQ Benchmarks",
        "",
        "| Evaluator | Outer CV F0.5 Mean | Outer CV F0.5 Std |",
        "|---|---:|---:|",
    ]
    for row in benchmark_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['evaluator_model']} | {_safe_fmt(row.get('outer_f0_5_mean'))} | {_safe_fmt(row.get('outer_f0_5_std'))} |"
        )
    if benchmark_summary.empty:
        lines.append("| -- | -- | -- |")

    lines += [
        "",
        "## Best Blockwise PLS By Family",
        "",
        "| Family | Model | Raw F0.5 | Best Block k | Best Block F0.5 | Delta | Threshold Mean | Threshold Std |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    family_ids = [value for value in sorted(set(summary_df["family_id"].tolist())) if value != "HQ"] if not summary_df.empty else []
    for family_id in family_ids:
        family_subset = summary_df[summary_df["family_id"] == family_id].copy()
        display = str(family_subset["family_display_label"].iloc[0])
        for evaluator_model in ["logistic", "xgb1"]:
            raw_subset = family_subset[
                (family_subset["route_group"] == "raw_family")
                & (family_subset["evaluator_model"] == evaluator_model)
            ]
            block_subset = family_subset[
                (family_subset["route_group"] == "block_pls")
                & (family_subset["evaluator_model"] == evaluator_model)
            ].sort_values(by=["outer_f0_5_mean", "actual_n_components"], ascending=[False, True])
            raw_mean = raw_subset["outer_f0_5_mean"].iloc[0] if not raw_subset.empty else None
            best_block = block_subset.iloc[0] if not block_subset.empty else None
            best_block_mean = None if best_block is None else float(best_block["outer_f0_5_mean"])
            delta = None if raw_mean is None or best_block_mean is None else float(best_block_mean - float(raw_mean))
            lines.append(
                "| {family} | {model} | {raw} | {best_k} | {best_block} | {delta} | {th_mean} | {th_std} |".format(
                    family=display,
                    model=evaluator_model,
                    raw=_safe_fmt(raw_mean),
                    best_k="--" if best_block is None else str(int(best_block["actual_n_components"])),
                    best_block=_safe_fmt(best_block_mean),
                    delta=_safe_fmt(delta),
                    th_mean="--" if best_block is None else _safe_fmt(best_block["threshold_mean"]),
                    th_std="--" if best_block is None else _safe_fmt(best_block["threshold_std"]),
                )
            )
    if not family_ids:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | -- |")
    return "\n".join(lines)


def block_pls_details_markdown(summary_df: pd.DataFrame, component_summary_df: pd.DataFrame) -> str:
    lines = [
        "# Reasoning-Block PLS Family Details",
        "",
        "This note shows raw versus blockwise-PLS route performance and the component diagnostics for each reasoning family.",
    ]
    family_ids = [value for value in sorted(set(summary_df["family_id"].tolist())) if value != "HQ"] if not summary_df.empty else []
    for family_id in family_ids:
        family_subset = summary_df[summary_df["family_id"] == family_id].copy()
        display = str(family_subset["family_display_label"].iloc[0])
        original_feature_count = int(family_subset["original_reasoning_feature_count"].max())
        lr_block = family_subset[
            (family_subset["route_group"] == "block_pls")
            & (family_subset["evaluator_model"] == "logistic")
        ].sort_values(by=["outer_f0_5_mean", "actual_n_components"], ascending=[False, True])
        xgb_block = family_subset[
            (family_subset["route_group"] == "block_pls")
            & (family_subset["evaluator_model"] == "xgb1")
        ].sort_values(by=["outer_f0_5_mean", "actual_n_components"], ascending=[False, True])
        lines += [
            "",
            f"## {display}",
            "",
            f"- Original reasoning feature count: {original_feature_count}",
            f"- Best blockwise `PLS` k by LR CV: {'--' if lr_block.empty else int(lr_block.iloc[0]['actual_n_components'])}",
            f"- Best blockwise `PLS` k by XGB1 CV: {'--' if xgb_block.empty else int(xgb_block.iloc[0]['actual_n_components'])}",
            "",
            "| Route | k | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | PR-AUC Mean | Threshold Mean | Threshold Std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        raw_lr = family_subset[
            (family_subset["route_group"] == "raw_family")
            & (family_subset["evaluator_model"] == "logistic")
        ]
        raw_xgb = family_subset[
            (family_subset["route_group"] == "raw_family")
            & (family_subset["evaluator_model"] == "xgb1")
        ]
        lines.append(
            "| Raw | -- | {lr_mean} | {lr_std} | {xgb_mean} | {xgb_std} | {pr_auc} | {th_mean} | {th_std} |".format(
                lr_mean="--" if raw_lr.empty else _safe_fmt(raw_lr.iloc[0]["outer_f0_5_mean"]),
                lr_std="--" if raw_lr.empty else _safe_fmt(raw_lr.iloc[0]["outer_f0_5_std"]),
                xgb_mean="--" if raw_xgb.empty else _safe_fmt(raw_xgb.iloc[0]["outer_f0_5_mean"]),
                xgb_std="--" if raw_xgb.empty else _safe_fmt(raw_xgb.iloc[0]["outer_f0_5_std"]),
                pr_auc="--" if raw_lr.empty else _safe_fmt(raw_lr.iloc[0]["pr_auc_mean"]),
                th_mean="--" if raw_lr.empty else _safe_fmt(raw_lr.iloc[0]["threshold_mean"]),
                th_std="--" if raw_lr.empty else _safe_fmt(raw_lr.iloc[0]["threshold_std"]),
            )
        )
        for k_value in sorted(
            int(value)
            for value in family_subset[family_subset["route_group"] == "block_pls"]["actual_n_components"].dropna().unique().tolist()
        ):
            lr_row = family_subset[
                (family_subset["route_group"] == "block_pls")
                & (family_subset["evaluator_model"] == "logistic")
                & (family_subset["actual_n_components"] == k_value)
            ]
            xgb_row = family_subset[
                (family_subset["route_group"] == "block_pls")
                & (family_subset["evaluator_model"] == "xgb1")
                & (family_subset["actual_n_components"] == k_value)
            ]
            lines.append(
                "| Block PLS | {k} | {lr_mean} | {lr_std} | {xgb_mean} | {xgb_std} | {pr_auc} | {th_mean} | {th_std} |".format(
                    k=k_value,
                    lr_mean="--" if lr_row.empty else _safe_fmt(lr_row.iloc[0]["outer_f0_5_mean"]),
                    lr_std="--" if lr_row.empty else _safe_fmt(lr_row.iloc[0]["outer_f0_5_std"]),
                    xgb_mean="--" if xgb_row.empty else _safe_fmt(xgb_row.iloc[0]["outer_f0_5_mean"]),
                    xgb_std="--" if xgb_row.empty else _safe_fmt(xgb_row.iloc[0]["outer_f0_5_std"]),
                    pr_auc="--" if lr_row.empty else _safe_fmt(lr_row.iloc[0]["pr_auc_mean"]),
                    th_mean="--" if lr_row.empty else _safe_fmt(lr_row.iloc[0]["threshold_mean"]),
                    th_std="--" if lr_row.empty else _safe_fmt(lr_row.iloc[0]["threshold_std"]),
                )
            )

        component_subset = component_summary_df[component_summary_df["family_id"] == family_id].copy()
        if component_subset.empty:
            continue
        for k_value in sorted(int(value) for value in component_subset["actual_n_components"].unique().tolist()):
            lines += [
                "",
                f"Component diagnostics for k = {k_value}:",
                "",
                "| Component | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean | Loading Std | Loading Flip |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
            k_subset = component_subset[component_subset["actual_n_components"] == k_value].copy()
            for component_name, component_frame in k_subset.groupby("component_name", dropna=False):
                top_rows = component_frame.sort_values(by=["abs_weight_mean", "original_feature"], ascending=[False, True]).head(5)
                for row in top_rows.to_dict(orient="records"):
                    lines.append(
                        "| {component} | {feature} | {weight_mean} | {weight_std} | {weight_flip} | {loading_mean} | {loading_std} | {loading_flip} |".format(
                            component=component_name,
                            feature=row["original_feature"],
                            weight_mean=_safe_fmt(row["weight_mean"]),
                            weight_std=_safe_fmt(row["weight_std"]),
                            weight_flip=_safe_fmt(row["weight_sign_flip_rate"]),
                            loading_mean=_safe_fmt(row["loading_mean"]),
                            loading_std=_safe_fmt(row["loading_std"]),
                            loading_flip=_safe_fmt(row["loading_sign_flip_rate"]),
                        )
                    )
    return "\n".join(lines)


def block_pls_audit_markdown(audit_df: pd.DataFrame) -> str:
    lines = [
        "# Reasoning-Block PLS Final Audit",
        "",
        "Private-test audit is deferred in this pass.",
        "",
        "This table will be populated only after a candidate blockwise-PLS route is locked from CV.",
        "",
        "| Candidate | Prediction Path | External Test F0.5 | Status |",
        "|---|---|---:|---|",
    ]
    for row in audit_df.to_dict(orient="records"):
        lines.append(
            "| {candidate} | {path} | {score} | {status} |".format(
                candidate=row.get("candidate", ""),
                path=row.get("prediction_path", ""),
                score=row.get("external_test_f0_5", ""),
                status=row.get("status", ""),
            )
        )
    return "\n".join(lines)
