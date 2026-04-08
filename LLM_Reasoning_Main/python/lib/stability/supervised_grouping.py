from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cross_decomposition import PLSRegression

from lib.shared.artifact_io import read_json
from lib.stability.family_registry import AlignedFamilyData
from lib.stability.routes import _fill_missing
from lib.stability.stability_selection import build_outer_splits, evaluate_selected_route


def load_supervised_grouping_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid supervised-grouping config: {path}")
    return payload


@dataclass(frozen=True)
class GroupedScopeResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cluster_assignments: pd.DataFrame
    component_rows: pd.DataFrame
    requested_group_count: int
    actual_group_count: int
    original_feature_count: int


@dataclass(frozen=True)
class GroupedRouteResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cluster_assignments: pd.DataFrame
    component_rows: pd.DataFrame
    original_grouping_input_feature_count: int
    grouped_feature_count: int
    final_model_feature_count: int


def _safe_fmt(value: Any, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "--"
    return f"{float(value):.{places}f}"


def _md_text(value: Any) -> str:
    return str(value).replace("|", "/")


def _sign_flip_rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    positive = float(np.mean(values > 0.0))
    negative = float(np.mean(values < 0.0))
    nonzero = positive + negative
    if nonzero == 0.0:
        return 0.0
    return float(min(positive, negative) / nonzero)


def _resolve_feature_set(config_section: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [item for item in config_section.get("feature_sets", []) if isinstance(item, dict)]
    if not rows:
        raise RuntimeError("Supervised-grouping config requires feature_sets for each track.")
    return rows


def _route_feature_set_label(route_spec: dict[str, Any]) -> str:
    return str(route_spec.get("label", route_spec.get("id", "")))


def _family_short_label(data: AlignedFamilyData, family_id: str) -> str:
    return str(data.family_specs[family_id].label)


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, axis=1)
    return combined.loc[:, ~combined.columns.duplicated()].copy()


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_use = np.asarray(left, dtype=float)
    right_use = np.asarray(right, dtype=float)
    if left_use.size == 0 or right_use.size == 0:
        return 0.0
    if np.std(left_use) == 0.0 or np.std(right_use) == 0.0:
        return 0.0
    value = float(np.corrcoef(left_use, right_use)[0, 1])
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(np.clip(value, -1.0, 1.0))


def _train_only_zscore(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_filled, test_filled = _fill_missing(train_df.copy(), test_df.copy(), model="logistic")
    means = train_filled.mean(numeric_only=True)
    stds = train_filled.std(numeric_only=True).replace(0.0, 1.0)
    train_scaled = ((train_filled - means) / stds).fillna(0.0)
    test_scaled = ((test_filled - means) / stds).fillna(0.0)
    return train_scaled, test_scaled, means, stds


def _supervised_distance_matrix(train_scaled: pd.DataFrame, y_train: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    columns = train_scaled.columns.tolist()
    if not columns:
        return np.zeros((0, 0), dtype=float), {}
    if len(columns) == 1:
        single = columns[0]
        return np.zeros((1, 1), dtype=float), {single: _safe_corr(train_scaled[single].to_numpy(dtype=float), y_train)}

    corr_matrix = train_scaled.corr(method="pearson").fillna(0.0).clip(-1.0, 1.0)
    target_corr = {
        column: _safe_corr(train_scaled[column].to_numpy(dtype=float), y_train)
        for column in columns
    }
    distance = np.zeros((len(columns), len(columns)), dtype=float)
    for left_idx, left_column in enumerate(columns):
        for right_idx in range(left_idx + 1, len(columns)):
            right_column = columns[right_idx]
            feature_similarity = abs(float(corr_matrix.iloc[left_idx, right_idx]))
            target_delta = abs(target_corr[left_column] - target_corr[right_column]) / 2.0
            value = 0.7 * (1.0 - feature_similarity) + 0.3 * target_delta
            clipped = float(np.clip(value, 0.0, 1.0))
            distance[left_idx, right_idx] = clipped
            distance[right_idx, left_idx] = clipped
    return distance, target_corr


def _cluster_features(distance_matrix: np.ndarray, requested_group_count: int) -> np.ndarray:
    feature_count = int(distance_matrix.shape[0])
    actual_group_count = min(max(1, int(requested_group_count)), max(1, feature_count))
    if feature_count <= 1 or actual_group_count >= feature_count:
        return np.arange(feature_count, dtype=int)
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=actual_group_count,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=actual_group_count,
            affinity="precomputed",
            linkage="average",
        )
    return clusterer.fit_predict(distance_matrix)


def fit_supervised_grouped_scope(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    requested_group_count: int,
    track: str,
    feature_set_id: str,
    feature_set_label: str,
    grouping_spec_id: str,
    grouping_spec_label: str,
    grouping_scope_id: str,
    grouping_scope_label: str,
    component_prefix: str,
    source_family_map: dict[str, dict[str, str]],
) -> GroupedScopeResult:
    train_scaled, test_scaled, means, stds = _train_only_zscore(train_df, test_df)
    columns = train_scaled.columns.tolist()
    feature_count = len(columns)
    if feature_count == 0:
        return GroupedScopeResult(
            train_df=pd.DataFrame(index=train_df.index),
            test_df=pd.DataFrame(index=test_df.index),
            cluster_assignments=pd.DataFrame(),
            component_rows=pd.DataFrame(),
            requested_group_count=int(requested_group_count),
            actual_group_count=0,
            original_feature_count=0,
        )

    distance_matrix, target_corr = _supervised_distance_matrix(train_scaled, y_train)
    cluster_labels = _cluster_features(distance_matrix, int(requested_group_count))
    actual_group_count = int(len(set(int(value) for value in cluster_labels.tolist())))

    grouped_train_parts: list[pd.Series] = []
    grouped_test_parts: list[pd.Series] = []
    assignment_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for cluster_rank, cluster_id in enumerate(sorted(set(int(value) for value in cluster_labels.tolist())), start=1):
        member_indices = [idx for idx, value in enumerate(cluster_labels.tolist()) if int(value) == cluster_id]
        cluster_columns = [columns[idx] for idx in member_indices]
        cluster_signature = " | ".join(sorted(cluster_columns))
        cluster_name = f"{component_prefix}_grp_{cluster_rank}"
        cluster_train = train_scaled[cluster_columns].copy()
        cluster_test = test_scaled[cluster_columns].copy()

        if len(cluster_columns) == 1:
            train_scores = cluster_train.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
            test_scores = cluster_test.iloc[:, 0].to_numpy(dtype=float).reshape(-1, 1)
            component_rows.append(
                {
                    "track": track,
                    "feature_set_id": feature_set_id,
                    "feature_set_label": feature_set_label,
                    "grouping_spec_id": grouping_spec_id,
                    "grouping_spec_label": grouping_spec_label,
                    "grouping_scope_id": grouping_scope_id,
                    "grouping_scope_label": grouping_scope_label,
                    "cluster_id": int(cluster_rank),
                    "cluster_name": cluster_name,
                    "cluster_signature": cluster_signature,
                    "original_feature": cluster_columns[0],
                    "source_family_id": source_family_map[cluster_columns[0]]["source_family_id"],
                    "source_family_label": source_family_map[cluster_columns[0]]["source_family_label"],
                    "feature_group": source_family_map[cluster_columns[0]]["feature_group"],
                    "x_weight": 1.0,
                    "x_loading": 1.0,
                    "target_correlation": float(target_corr.get(cluster_columns[0], 0.0)),
                    "train_mean": float(means.get(cluster_columns[0], 0.0)),
                    "train_std": float(stds.get(cluster_columns[0], 1.0)),
                }
            )
        else:
            pls = PLSRegression(n_components=1)
            train_scores = pls.fit_transform(cluster_train.to_numpy(dtype=float), y_train)[0]
            test_scores = pls.transform(cluster_test.to_numpy(dtype=float))
            for feature_index, feature_name in enumerate(cluster_columns):
                component_rows.append(
                    {
                        "track": track,
                        "feature_set_id": feature_set_id,
                        "feature_set_label": feature_set_label,
                        "grouping_spec_id": grouping_spec_id,
                        "grouping_spec_label": grouping_spec_label,
                        "grouping_scope_id": grouping_scope_id,
                        "grouping_scope_label": grouping_scope_label,
                        "cluster_id": int(cluster_rank),
                        "cluster_name": cluster_name,
                        "cluster_signature": cluster_signature,
                        "original_feature": feature_name,
                        "source_family_id": source_family_map[feature_name]["source_family_id"],
                        "source_family_label": source_family_map[feature_name]["source_family_label"],
                        "feature_group": source_family_map[feature_name]["feature_group"],
                        "x_weight": float(pls.x_weights_[feature_index, 0]),
                        "x_loading": float(pls.x_loadings_[feature_index, 0]),
                        "target_correlation": float(target_corr.get(feature_name, 0.0)),
                        "train_mean": float(means.get(feature_name, 0.0)),
                        "train_std": float(stds.get(feature_name, 1.0)),
                    }
                )

        grouped_train_parts.append(pd.Series(train_scores.reshape(-1), index=train_df.index, name=cluster_name))
        grouped_test_parts.append(pd.Series(test_scores.reshape(-1), index=test_df.index, name=cluster_name))

        for feature_name in cluster_columns:
            assignment_rows.append(
                {
                    "track": track,
                    "feature_set_id": feature_set_id,
                    "feature_set_label": feature_set_label,
                    "grouping_spec_id": grouping_spec_id,
                    "grouping_spec_label": grouping_spec_label,
                    "grouping_scope_id": grouping_scope_id,
                    "grouping_scope_label": grouping_scope_label,
                    "cluster_id": int(cluster_rank),
                    "cluster_name": cluster_name,
                    "cluster_signature": cluster_signature,
                    "original_feature": feature_name,
                    "source_family_id": source_family_map[feature_name]["source_family_id"],
                    "source_family_label": source_family_map[feature_name]["source_family_label"],
                    "feature_group": source_family_map[feature_name]["feature_group"],
                    "target_correlation": float(target_corr.get(feature_name, 0.0)),
                }
            )

    grouped_train = pd.concat(grouped_train_parts, axis=1) if grouped_train_parts else pd.DataFrame(index=train_df.index)
    grouped_test = pd.concat(grouped_test_parts, axis=1) if grouped_test_parts else pd.DataFrame(index=test_df.index)
    return GroupedScopeResult(
        train_df=grouped_train,
        test_df=grouped_test,
        cluster_assignments=pd.DataFrame(assignment_rows),
        component_rows=pd.DataFrame(component_rows),
        requested_group_count=int(requested_group_count),
        actual_group_count=int(actual_group_count),
        original_feature_count=int(feature_count),
    )


def _feature_source_map_for_route(
    baseline_columns: list[str],
    family_columns: dict[str, list[str]],
    family_labels: dict[str, str],
) -> dict[str, dict[str, str]]:
    source_map: dict[str, dict[str, str]] = {}
    for column in baseline_columns:
        source_map[column] = {
            "source_family_id": "HQ",
            "source_family_label": "HQ",
            "feature_group": "hq",
        }
    for family_id, columns in family_columns.items():
        for column in columns:
            source_map[column] = {
                "source_family_id": family_id,
                "source_family_label": family_labels[family_id],
                "feature_group": "reasoning",
            }
    return source_map


def _build_raw_feature_set(
    data: AlignedFamilyData,
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    family_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    family_frames_train: list[pd.DataFrame] = []
    family_frames_test: list[pd.DataFrame] = []
    family_columns: dict[str, list[str]] = {}
    family_labels: dict[str, str] = {}
    for family_id in family_ids:
        family_train = data.candidate_frames[family_id].iloc[train_idx].copy()
        family_test = data.candidate_frames[family_id].iloc[test_idx].copy()
        family_frames_train.append(family_train)
        family_frames_test.append(family_test)
        family_columns[family_id] = family_train.columns.tolist()
        family_labels[family_id] = _family_short_label(data, family_id)
    raw_train = _combine_frames([baseline_train] + family_frames_train)
    raw_test = _combine_frames([baseline_test] + family_frames_test)
    raw_test = raw_test.loc[:, raw_train.columns].copy()
    return raw_train, raw_test, family_columns, family_labels


def _build_augmentation_route(
    data: AlignedFamilyData,
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    route_spec: dict[str, Any],
    requested_groups_per_family: int,
) -> GroupedRouteResult:
    family_ids = [str(value) for value in route_spec.get("family_ids", [])]
    feature_set_id = str(route_spec["id"])
    feature_set_label = _route_feature_set_label(route_spec)
    grouping_spec_id = f"per_family_{int(requested_groups_per_family)}"
    grouping_spec_label = f"{int(requested_groups_per_family)} groups per family"

    grouped_train_parts: list[pd.DataFrame] = []
    grouped_test_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    component_parts: list[pd.DataFrame] = []
    total_original = 0
    total_grouped = 0

    y_train = data.labels.to_numpy(dtype=int)[train_idx]
    for family_id in family_ids:
        family_train = data.candidate_frames[family_id].iloc[train_idx].copy()
        family_test = data.candidate_frames[family_id].iloc[test_idx].copy()
        family_label = _family_short_label(data, family_id)
        source_map = _feature_source_map_for_route(
            baseline_columns=[],
            family_columns={family_id: family_train.columns.tolist()},
            family_labels={family_id: family_label},
        )
        scope_result = fit_supervised_grouped_scope(
            family_train,
            family_test,
            y_train,
            requested_group_count=int(requested_groups_per_family),
            track="augmentation",
            feature_set_id=feature_set_id,
            feature_set_label=feature_set_label,
            grouping_spec_id=grouping_spec_id,
            grouping_spec_label=grouping_spec_label,
            grouping_scope_id=family_id,
            grouping_scope_label=family_label,
            component_prefix=family_label,
            source_family_map=source_map,
        )
        grouped_train_parts.append(scope_result.train_df)
        grouped_test_parts.append(scope_result.test_df)
        if not scope_result.cluster_assignments.empty:
            assignment_parts.append(scope_result.cluster_assignments)
        if not scope_result.component_rows.empty:
            component_parts.append(scope_result.component_rows)
        total_original += int(scope_result.original_feature_count)
        total_grouped += int(scope_result.actual_group_count)

    grouped_train = _combine_frames(grouped_train_parts)
    grouped_test = _combine_frames(grouped_test_parts)
    final_train = _combine_frames([baseline_train, grouped_train])
    final_test = _combine_frames([baseline_test, grouped_test]).loc[:, final_train.columns].copy()
    return GroupedRouteResult(
        train_df=final_train,
        test_df=final_test,
        cluster_assignments=pd.concat(assignment_parts, ignore_index=True) if assignment_parts else pd.DataFrame(),
        component_rows=pd.concat(component_parts, ignore_index=True) if component_parts else pd.DataFrame(),
        original_grouping_input_feature_count=int(total_original),
        grouped_feature_count=int(total_grouped),
        final_model_feature_count=int(final_train.shape[1]),
    )


def _build_competition_route(
    data: AlignedFamilyData,
    baseline_train: pd.DataFrame,
    baseline_test: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    route_spec: dict[str, Any],
    requested_total_groups: int,
) -> GroupedRouteResult:
    feature_set_id = str(route_spec["id"])
    feature_set_label = _route_feature_set_label(route_spec)
    family_ids = [str(value) for value in route_spec.get("family_ids", [])]
    raw_train, raw_test, family_columns, family_labels = _build_raw_feature_set(
        data,
        baseline_train,
        baseline_test,
        train_idx,
        test_idx,
        family_ids,
    )
    source_map = _feature_source_map_for_route(
        baseline_columns=baseline_train.columns.tolist(),
        family_columns=family_columns,
        family_labels=family_labels,
    )
    scope_result = fit_supervised_grouped_scope(
        raw_train,
        raw_test,
        data.labels.to_numpy(dtype=int)[train_idx],
        requested_group_count=int(requested_total_groups),
        track="competition",
        feature_set_id=feature_set_id,
        feature_set_label=feature_set_label,
        grouping_spec_id=f"total_groups_{int(requested_total_groups)}",
        grouping_spec_label=f"{int(requested_total_groups)} total groups",
        grouping_scope_id="full_route",
        grouping_scope_label=feature_set_label,
        component_prefix="competition",
        source_family_map=source_map,
    )
    return GroupedRouteResult(
        train_df=scope_result.train_df,
        test_df=scope_result.test_df,
        cluster_assignments=scope_result.cluster_assignments,
        component_rows=scope_result.component_rows,
        original_grouping_input_feature_count=int(scope_result.original_feature_count),
        grouped_feature_count=int(scope_result.actual_group_count),
        final_model_feature_count=int(scope_result.train_df.shape[1]),
    )


def _append_metric_rows(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    track: str,
    feature_set_id: str,
    feature_set_label: str,
    route_group: str,
    grouping_spec_id: str,
    grouping_spec_label: str,
    evaluator_model: str,
    outer_split_id: str,
    repeat_index: int,
    fold_index: int,
    original_feature_count: int,
    grouped_feature_count: int,
    final_model_feature_count: int,
) -> None:
    rows.append(
        {
            "track": track,
            "feature_set_id": feature_set_id,
            "feature_set_label": feature_set_label,
            "route_group": route_group,
            "grouping_spec_id": grouping_spec_id,
            "grouping_spec_label": grouping_spec_label,
            "evaluator_model": evaluator_model,
            "outer_split_id": outer_split_id,
            "repeat_index": int(repeat_index),
            "fold_index": int(fold_index),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "pr_auc": float(metrics["pr_auc"]),
            "f0_5": float(metrics["f0_5"]),
            "threshold": float(metrics["threshold"]),
            "original_feature_count": int(original_feature_count),
            "grouped_feature_count": int(grouped_feature_count),
            "final_model_feature_count": int(final_model_feature_count),
        }
    )


def _cocluster_frequency(cluster_assignments: pd.DataFrame, expected_split_count: int) -> pd.DataFrame:
    if cluster_assignments.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    split_group_cols = [
        "track",
        "feature_set_id",
        "feature_set_label",
        "grouping_spec_id",
        "grouping_spec_label",
        "grouping_scope_id",
        "grouping_scope_label",
        "outer_split_id",
    ]
    aggregate_cols = split_group_cols[:-1]
    split_rows: list[dict[str, Any]] = []
    for keys, subset in cluster_assignments.groupby(split_group_cols, dropna=False):
        feature_rows = subset[["original_feature", "cluster_signature"]].drop_duplicates()
        cluster_map = {
            str(row["original_feature"]): str(row["cluster_signature"])
            for row in feature_rows.to_dict(orient="records")
        }
        features = sorted(cluster_map.keys())
        for left_feature, right_feature in itertools.combinations(features, 2):
            split_rows.append(
                {
                    aggregate_cols[0]: keys[0],
                    aggregate_cols[1]: keys[1],
                    aggregate_cols[2]: keys[2],
                    aggregate_cols[3]: keys[3],
                    aggregate_cols[4]: keys[4],
                    aggregate_cols[5]: keys[5],
                    aggregate_cols[6]: keys[6],
                    "feature_a": left_feature,
                    "feature_b": right_feature,
                    "same_cluster": float(cluster_map[left_feature] == cluster_map[right_feature]),
                }
            )
    if not split_rows:
        return pd.DataFrame()
    split_df = pd.DataFrame(split_rows)
    for keys, subset in split_df.groupby(aggregate_cols + ["feature_a", "feature_b"], dropna=False):
        (
            track,
            feature_set_id,
            feature_set_label,
            grouping_spec_id,
            grouping_spec_label,
            grouping_scope_id,
            grouping_scope_label,
            feature_a,
            feature_b,
        ) = keys
        rows.append(
            {
                "track": track,
                "feature_set_id": feature_set_id,
                "feature_set_label": feature_set_label,
                "grouping_spec_id": grouping_spec_id,
                "grouping_spec_label": grouping_spec_label,
                "grouping_scope_id": grouping_scope_id,
                "grouping_scope_label": grouping_scope_label,
                "feature_a": feature_a,
                "feature_b": feature_b,
                "same_cluster_frequency": float(subset["same_cluster"].mean()),
                "split_count": int(len(subset)),
                "expected_split_count": int(expected_split_count),
            }
        )
    return pd.DataFrame(rows)


def run_supervised_grouping_study(
    data: AlignedFamilyData,
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
    baseline_feature_count = len(baseline_columns)

    augmentation_cfg = config.get("augmentation", {})
    competition_cfg = config.get("competition", {})
    augmentation_routes = _resolve_feature_set(augmentation_cfg)
    competition_routes = _resolve_feature_set(competition_cfg)
    augmentation_group_counts = [int(value) for value in augmentation_cfg.get("groups_per_family", [2, 3])]
    competition_total_groups = int(competition_cfg.get("total_groups", 6))

    benchmark_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    cluster_assignment_frames: list[pd.DataFrame] = []
    component_frames: list[pd.DataFrame] = []

    total_units = max(
        1,
        len(outer_splits) * (len(augmentation_routes) * len(augmentation_group_counts) + len(competition_routes)),
    )
    completed_units = 0

    distinct_feature_set_ids = {
        str(route["id"]): [str(value) for value in route.get("family_ids", [])]
        for route in augmentation_routes + competition_routes
    }

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
            benchmark_metrics = evaluate_selected_route(
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
            _append_metric_rows(
                benchmark_rows,
                benchmark_metrics,
                track="benchmark",
                feature_set_id="hq_benchmark",
                feature_set_label="HQ",
                route_group="hq_benchmark",
                grouping_spec_id="frozen_hq",
                grouping_spec_label="Frozen HQ benchmark",
                evaluator_model=model_type,
                outer_split_id=outer_split_id,
                repeat_index=int(split["repeat_index"]),
                fold_index=int(split["fold_index"]),
                original_feature_count=baseline_feature_count,
                grouped_feature_count=baseline_feature_count,
                final_model_feature_count=baseline_feature_count,
            )

        raw_feature_set_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame, int]] = {}
        raw_metric_cache: dict[tuple[str, str], dict[str, Any]] = {}
        for feature_set_id, family_ids in distinct_feature_set_ids.items():
            raw_train, raw_test, _, _ = _build_raw_feature_set(
                data,
                baseline_train,
                baseline_test,
                train_idx,
                test_idx,
                family_ids,
            )
            raw_feature_set_cache[feature_set_id] = (raw_train, raw_test, int(raw_train.shape[1]))
            for model_type in models:
                raw_metric_cache[(feature_set_id, model_type)] = evaluate_selected_route(
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

        for route_spec in augmentation_routes:
            feature_set_id = str(route_spec["id"])
            feature_set_label = _route_feature_set_label(route_spec)
            raw_train, raw_test, raw_feature_count = raw_feature_set_cache[feature_set_id]
            reasoning_input_count = int(raw_feature_count - baseline_feature_count)
            for requested_group_count in augmentation_group_counts:
                grouping_spec_id = f"per_family_{int(requested_group_count)}"
                grouping_spec_label = f"{int(requested_group_count)} groups per family"
                for model_type in models:
                    raw_metrics = raw_metric_cache[(feature_set_id, model_type)]
                    _append_metric_rows(
                        route_rows,
                        raw_metrics,
                        track="augmentation",
                        feature_set_id=feature_set_id,
                        feature_set_label=feature_set_label,
                        route_group="raw_comparison",
                        grouping_spec_id=grouping_spec_id,
                        grouping_spec_label=grouping_spec_label,
                        evaluator_model=model_type,
                        outer_split_id=outer_split_id,
                        repeat_index=int(split["repeat_index"]),
                        fold_index=int(split["fold_index"]),
                        original_feature_count=reasoning_input_count,
                        grouped_feature_count=reasoning_input_count,
                        final_model_feature_count=raw_feature_count,
                    )

                grouped_route = _build_augmentation_route(
                    data,
                    baseline_train,
                    baseline_test,
                    train_idx,
                    test_idx,
                    route_spec,
                    requested_group_count,
                )
                if not grouped_route.cluster_assignments.empty:
                    cluster_frame = grouped_route.cluster_assignments.copy()
                    cluster_frame["outer_split_id"] = outer_split_id
                    cluster_frame["repeat_index"] = int(split["repeat_index"])
                    cluster_frame["fold_index"] = int(split["fold_index"])
                    cluster_assignment_frames.append(cluster_frame)
                if not grouped_route.component_rows.empty:
                    component_frame = grouped_route.component_rows.copy()
                    component_frame["outer_split_id"] = outer_split_id
                    component_frame["repeat_index"] = int(split["repeat_index"])
                    component_frame["fold_index"] = int(split["fold_index"])
                    component_frames.append(component_frame)
                for model_type in models:
                    grouped_metrics = evaluate_selected_route(
                        grouped_route.train_df,
                        grouped_route.test_df,
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
                    _append_metric_rows(
                        route_rows,
                        grouped_metrics,
                        track="augmentation",
                        feature_set_id=feature_set_id,
                        feature_set_label=feature_set_label,
                        route_group="grouped_route",
                        grouping_spec_id=grouping_spec_id,
                        grouping_spec_label=grouping_spec_label,
                        evaluator_model=model_type,
                        outer_split_id=outer_split_id,
                        repeat_index=int(split["repeat_index"]),
                        fold_index=int(split["fold_index"]),
                        original_feature_count=grouped_route.original_grouping_input_feature_count,
                        grouped_feature_count=grouped_route.grouped_feature_count,
                        final_model_feature_count=grouped_route.final_model_feature_count,
                    )
                completed_units += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "augmentation_route_complete",
                            "outer_split_id": outer_split_id,
                            "feature_set_label": feature_set_label,
                            "grouping_spec_label": grouping_spec_label,
                            "completed_units": completed_units,
                            "total_units": total_units,
                        }
                    )

        for route_spec in competition_routes:
            feature_set_id = str(route_spec["id"])
            feature_set_label = _route_feature_set_label(route_spec)
            raw_train, raw_test, raw_feature_count = raw_feature_set_cache.get(
                feature_set_id,
                (baseline_train[baseline_columns], baseline_test[baseline_columns], baseline_feature_count),
            )
            grouping_spec_id = f"total_groups_{competition_total_groups}"
            grouping_spec_label = f"{competition_total_groups} total groups"
            for model_type in models:
                raw_metrics = raw_metric_cache.get((feature_set_id, model_type))
                if raw_metrics is None:
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
                _append_metric_rows(
                    route_rows,
                    raw_metrics,
                    track="competition",
                    feature_set_id=feature_set_id,
                    feature_set_label=feature_set_label,
                    route_group="raw_comparison",
                    grouping_spec_id=grouping_spec_id,
                    grouping_spec_label=grouping_spec_label,
                    evaluator_model=model_type,
                    outer_split_id=outer_split_id,
                    repeat_index=int(split["repeat_index"]),
                    fold_index=int(split["fold_index"]),
                    original_feature_count=raw_feature_count,
                    grouped_feature_count=raw_feature_count,
                    final_model_feature_count=raw_feature_count,
                )

            grouped_route = _build_competition_route(
                data,
                baseline_train,
                baseline_test,
                train_idx,
                test_idx,
                route_spec,
                competition_total_groups,
            )
            if not grouped_route.cluster_assignments.empty:
                cluster_frame = grouped_route.cluster_assignments.copy()
                cluster_frame["outer_split_id"] = outer_split_id
                cluster_frame["repeat_index"] = int(split["repeat_index"])
                cluster_frame["fold_index"] = int(split["fold_index"])
                cluster_assignment_frames.append(cluster_frame)
            if not grouped_route.component_rows.empty:
                component_frame = grouped_route.component_rows.copy()
                component_frame["outer_split_id"] = outer_split_id
                component_frame["repeat_index"] = int(split["repeat_index"])
                component_frame["fold_index"] = int(split["fold_index"])
                component_frames.append(component_frame)
            for model_type in models:
                grouped_metrics = evaluate_selected_route(
                    grouped_route.train_df,
                    grouped_route.test_df,
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
                _append_metric_rows(
                    route_rows,
                    grouped_metrics,
                    track="competition",
                    feature_set_id=feature_set_id,
                    feature_set_label=feature_set_label,
                    route_group="grouped_route",
                    grouping_spec_id=grouping_spec_id,
                    grouping_spec_label=grouping_spec_label,
                    evaluator_model=model_type,
                    outer_split_id=outer_split_id,
                    repeat_index=int(split["repeat_index"]),
                    fold_index=int(split["fold_index"]),
                    original_feature_count=grouped_route.original_grouping_input_feature_count,
                    grouped_feature_count=grouped_route.grouped_feature_count,
                    final_model_feature_count=grouped_route.final_model_feature_count,
                )
            completed_units += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "competition_route_complete",
                        "outer_split_id": outer_split_id,
                        "feature_set_label": feature_set_label,
                        "grouping_spec_label": grouping_spec_label,
                        "completed_units": completed_units,
                        "total_units": total_units,
                    }
                )

    benchmark_df = pd.DataFrame(benchmark_rows)
    route_metrics_df = pd.DataFrame(route_rows)
    cluster_assignments_df = pd.concat(cluster_assignment_frames, ignore_index=True) if cluster_assignment_frames else pd.DataFrame()
    component_rows_df = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    all_metrics_df = pd.concat([benchmark_df, route_metrics_df], ignore_index=True) if not route_metrics_df.empty else benchmark_df.copy()

    summary_rows: list[dict[str, Any]] = []
    group_columns = [
        "track",
        "feature_set_id",
        "feature_set_label",
        "route_group",
        "grouping_spec_id",
        "grouping_spec_label",
        "evaluator_model",
        "original_feature_count",
        "grouped_feature_count",
        "final_model_feature_count",
    ]
    if not all_metrics_df.empty:
        for keys, subset in all_metrics_df.groupby(group_columns, dropna=False):
            (
                track,
                feature_set_id,
                feature_set_label,
                route_group,
                grouping_spec_id,
                grouping_spec_label,
                evaluator_model,
                original_feature_count,
                grouped_feature_count,
                final_model_feature_count,
            ) = keys
            summary_rows.append(
                {
                    "track": track,
                    "feature_set_id": feature_set_id,
                    "feature_set_label": feature_set_label,
                    "route_group": route_group,
                    "grouping_spec_id": grouping_spec_id,
                    "grouping_spec_label": grouping_spec_label,
                    "evaluator_model": evaluator_model,
                    "original_feature_count": int(original_feature_count),
                    "grouped_feature_count": int(grouped_feature_count),
                    "final_model_feature_count": int(final_model_feature_count),
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
    if not component_rows_df.empty:
        for keys, subset in component_rows_df.groupby(
            [
                "track",
                "feature_set_id",
                "feature_set_label",
                "grouping_spec_id",
                "grouping_spec_label",
                "grouping_scope_id",
                "grouping_scope_label",
                "cluster_signature",
                "original_feature",
                "source_family_id",
                "source_family_label",
                "feature_group",
            ],
            dropna=False,
        ):
            (
                track,
                feature_set_id,
                feature_set_label,
                grouping_spec_id,
                grouping_spec_label,
                grouping_scope_id,
                grouping_scope_label,
                cluster_signature,
                original_feature,
                source_family_id,
                source_family_label,
                feature_group,
            ) = keys
            weights = subset["x_weight"].to_numpy(dtype=float)
            loadings = subset["x_loading"].to_numpy(dtype=float)
            component_summary_rows.append(
                {
                    "track": track,
                    "feature_set_id": feature_set_id,
                    "feature_set_label": feature_set_label,
                    "grouping_spec_id": grouping_spec_id,
                    "grouping_spec_label": grouping_spec_label,
                    "grouping_scope_id": grouping_scope_id,
                    "grouping_scope_label": grouping_scope_label,
                    "cluster_signature": cluster_signature,
                    "representative_cluster_name": str(subset["cluster_name"].iloc[0]),
                    "original_feature": original_feature,
                    "source_family_id": source_family_id,
                    "source_family_label": source_family_label,
                    "feature_group": feature_group,
                    "weight_mean": float(np.mean(weights)),
                    "weight_std": float(np.std(weights, ddof=0)),
                    "abs_weight_mean": float(np.mean(np.abs(weights))),
                    "weight_sign_flip_rate": _sign_flip_rate(weights),
                    "loading_mean": float(np.mean(loadings)),
                    "loading_std": float(np.std(loadings, ddof=0)),
                    "abs_loading_mean": float(np.mean(np.abs(loadings))),
                    "loading_sign_flip_rate": _sign_flip_rate(loadings),
                    "split_count": int(subset["outer_split_id"].nunique()),
                }
            )
    component_summary_df = pd.DataFrame(component_summary_rows)
    cocluster_summary_df = _cocluster_frequency(cluster_assignments_df, expected_split_count=len(outer_splits))
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
        "route_metrics": route_metrics_df,
        "all_metrics": all_metrics_df,
        "summary": summary_df,
        "cluster_assignments": cluster_assignments_df,
        "component_rows": component_rows_df,
        "component_summary": component_summary_df,
        "cocluster_summary": cocluster_summary_df,
        "audit": audit_df,
    }


def supervised_grouping_protocol_markdown(config: dict[str, Any]) -> str:
    outer_cfg = config.get("outer_cv", {})
    augmentation_cfg = config.get("augmentation", {})
    competition_cfg = config.get("competition", {})
    aug_routes = _resolve_feature_set(augmentation_cfg)
    comp_routes = _resolve_feature_set(competition_cfg)
    aug_counts = ", ".join(str(int(value)) for value in augmentation_cfg.get("groups_per_family", [2, 3]))
    lines = [
        "# Supervised Grouping Protocol",
        "",
        "This Step 4 method groups correlated features inside each outer training fold, then collapses each cluster to one latent feature with `1`-component `PLS`.",
        "",
        "## Study Setup",
        "",
        f"- Outer CV: {int(outer_cfg.get('n_splits', 3))}-fold repeated {int(outer_cfg.get('n_repeats', 16))} times",
        "- Models: `LR`, `XGB1`",
        "- Holdout/test is deferred in this first pass.",
        "- Frozen `HQ` is always rerun on the exact same outer splits.",
        "",
        "## Tracks",
        "",
        "- `augmentation`: group reasoning families separately, then append grouped reasoning features to raw `HQ`.",
        "- `competition`: cluster the full active feature set together, including `HQ`, and evaluate only the grouped features.",
        "",
        "## Augmentation Feature Sets",
        "",
    ]
    for route in aug_routes:
        lines.append(f"- `{_route_feature_set_label(route)}`")
    lines += [
        "",
        f"- Groups per family explored: {aug_counts}",
        "",
        "## Competition Feature Sets",
        "",
    ]
    for route in comp_routes:
        lines.append(f"- `{_route_feature_set_label(route)}`")
    lines += [
        "",
        f"- Total groups explored: {int(competition_cfg.get('total_groups', 6))}",
        "",
        "## Leakage Control",
        "",
        "- Train-only Z-scoring is fit inside each outer fold.",
        "- Supervised distances are computed from train-only correlations.",
        "- Agglomerative clustering is fit on the train-only distance matrix.",
        "- `PLSRegression(n_components=1)` is fit on each train-only cluster.",
        "- Threshold selection uses train-only inner CV after grouping is complete.",
    ]
    return "\n".join(lines)


def supervised_grouping_summary_markdown(summary_df: pd.DataFrame) -> str:
    benchmark_subset = summary_df[summary_df["route_group"] == "hq_benchmark"].copy()
    lines = [
        "# Supervised Grouping Summary",
        "",
        "This note compares raw routes against supervised grouping routes on the same `3 x 16` outer CV.",
        "",
        "## HQ Benchmarks",
        "",
        "| Evaluator | Outer CV F0.5 Mean | Outer CV F0.5 Std | Threshold Mean | Threshold Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in benchmark_subset.sort_values(by=["evaluator_model"]).to_dict(orient="records"):
        lines.append(
            "| {model} | {f_mean} | {f_std} | {t_mean} | {t_std} |".format(
                model=row["evaluator_model"],
                f_mean=_safe_fmt(row["outer_f0_5_mean"]),
                f_std=_safe_fmt(row["outer_f0_5_std"]),
                t_mean=_safe_fmt(row["threshold_mean"]),
                t_std=_safe_fmt(row["threshold_std"]),
            )
        )
    if benchmark_subset.empty:
        lines.append("| -- | -- | -- | -- | -- |")

    for track in ["augmentation", "competition"]:
        track_subset = summary_df[summary_df["track"] == track].copy()
        lines += [
            "",
            f"## {track.title()} Track",
            "",
            "| Feature Set | Grouping | Model | Raw F0.5 | Grouped F0.5 | Delta | Raw Std | Grouped Std | Original Count | Grouped Count | Final Count |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        route_keys = (
            track_subset[["feature_set_id", "feature_set_label", "grouping_spec_id", "grouping_spec_label"]]
            .drop_duplicates()
            .sort_values(by=["feature_set_label", "grouping_spec_label"])
        )
        has_rows = False
        for route_row in route_keys.to_dict(orient="records"):
            for evaluator_model in ["logistic", "xgb1"]:
                raw_row = track_subset[
                    (track_subset["feature_set_id"] == route_row["feature_set_id"])
                    & (track_subset["grouping_spec_id"] == route_row["grouping_spec_id"])
                    & (track_subset["route_group"] == "raw_comparison")
                    & (track_subset["evaluator_model"] == evaluator_model)
                ]
                grouped_row = track_subset[
                    (track_subset["feature_set_id"] == route_row["feature_set_id"])
                    & (track_subset["grouping_spec_id"] == route_row["grouping_spec_id"])
                    & (track_subset["route_group"] == "grouped_route")
                    & (track_subset["evaluator_model"] == evaluator_model)
                ]
                if raw_row.empty or grouped_row.empty:
                    continue
                has_rows = True
                raw_mean = float(raw_row.iloc[0]["outer_f0_5_mean"])
                grouped_mean = float(grouped_row.iloc[0]["outer_f0_5_mean"])
                lines.append(
                    "| {feature_set} | {grouping} | {model} | {raw_mean} | {grouped_mean} | {delta} | {raw_std} | {grouped_std} | {original} | {grouped} | {final_count} |".format(
                        feature_set=route_row["feature_set_label"],
                        grouping=route_row["grouping_spec_label"],
                        model=evaluator_model,
                        raw_mean=_safe_fmt(raw_mean),
                        grouped_mean=_safe_fmt(grouped_mean),
                        delta=_safe_fmt(grouped_mean - raw_mean),
                        raw_std=_safe_fmt(raw_row.iloc[0]["outer_f0_5_std"]),
                        grouped_std=_safe_fmt(grouped_row.iloc[0]["outer_f0_5_std"]),
                        original=_safe_fmt(grouped_row.iloc[0]["original_feature_count"], places=0),
                        grouped=_safe_fmt(grouped_row.iloc[0]["grouped_feature_count"], places=0),
                        final_count=_safe_fmt(grouped_row.iloc[0]["final_model_feature_count"], places=0),
                    )
                )
        if not has_rows:
            lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    return "\n".join(lines)


def supervised_grouping_details_markdown(
    summary_df: pd.DataFrame,
    cluster_assignments_df: pd.DataFrame,
    component_summary_df: pd.DataFrame,
    cocluster_summary_df: pd.DataFrame,
) -> str:
    lines = [
        "# Supervised Grouping Details",
        "",
        "This note records the grouped-route deltas, representative cluster layouts, and stable co-clustering patterns.",
    ]
    route_keys = (
        summary_df[summary_df["route_group"] == "grouped_route"][
            ["track", "feature_set_id", "feature_set_label", "grouping_spec_id", "grouping_spec_label"]
        ]
        .drop_duplicates()
        .sort_values(by=["track", "feature_set_label", "grouping_spec_label"])
    )
    for route_row in route_keys.to_dict(orient="records"):
        track = str(route_row["track"])
        feature_set_id = str(route_row["feature_set_id"])
        feature_set_label = str(route_row["feature_set_label"])
        grouping_spec_id = str(route_row["grouping_spec_id"])
        grouping_spec_label = str(route_row["grouping_spec_label"])
        lines += [
            "",
            f"## {track.title()} | {feature_set_label} | {grouping_spec_label}",
            "",
            "| Model | Raw F0.5 | Grouped F0.5 | Delta | Raw PR-AUC | Grouped PR-AUC | Raw Threshold | Grouped Threshold |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        route_subset = summary_df[
            (summary_df["track"] == track)
            & (summary_df["feature_set_id"] == feature_set_id)
            & (summary_df["grouping_spec_id"] == grouping_spec_id)
        ]
        for evaluator_model in ["logistic", "xgb1"]:
            raw_row = route_subset[
                (route_subset["route_group"] == "raw_comparison")
                & (route_subset["evaluator_model"] == evaluator_model)
            ]
            grouped_row = route_subset[
                (route_subset["route_group"] == "grouped_route")
                & (route_subset["evaluator_model"] == evaluator_model)
            ]
            if raw_row.empty or grouped_row.empty:
                continue
            raw_mean = float(raw_row.iloc[0]["outer_f0_5_mean"])
            grouped_mean = float(grouped_row.iloc[0]["outer_f0_5_mean"])
            lines.append(
                "| {model} | {raw_mean} | {grouped_mean} | {delta} | {raw_pr} | {grouped_pr} | {raw_threshold} | {grouped_threshold} |".format(
                    model=evaluator_model,
                    raw_mean=_safe_fmt(raw_mean),
                    grouped_mean=_safe_fmt(grouped_mean),
                    delta=_safe_fmt(grouped_mean - raw_mean),
                    raw_pr=_safe_fmt(raw_row.iloc[0]["pr_auc_mean"]),
                    grouped_pr=_safe_fmt(grouped_row.iloc[0]["pr_auc_mean"]),
                    raw_threshold=_safe_fmt(raw_row.iloc[0]["threshold_mean"]),
                    grouped_threshold=_safe_fmt(grouped_row.iloc[0]["threshold_mean"]),
                )
            )

        representative = cluster_assignments_df[
            (cluster_assignments_df["track"] == track)
            & (cluster_assignments_df["feature_set_id"] == feature_set_id)
            & (cluster_assignments_df["grouping_spec_id"] == grouping_spec_id)
            & (cluster_assignments_df["outer_split_id"] == "repeat_01_fold_01")
        ].copy()
        lines += [
            "",
            "Representative cluster layout from `repeat_01_fold_01`:",
            "",
        ]
        if representative.empty:
            lines.append("- No representative cluster layout available.")
        else:
            for scope_name, scope_subset in representative.groupby("grouping_scope_label", dropna=False):
                lines.append(f"- Scope `{scope_name}`:")
                for cluster_name, cluster_subset in scope_subset.groupby("cluster_name", dropna=False):
                    features = ", ".join(sorted(cluster_subset["original_feature"].tolist()))
                    lines.append(f"  {cluster_name}: {features}")

        top_pairs = cocluster_summary_df[
            (cocluster_summary_df["track"] == track)
            & (cocluster_summary_df["feature_set_id"] == feature_set_id)
            & (cocluster_summary_df["grouping_spec_id"] == grouping_spec_id)
        ].copy()
        lines += [
            "",
            "Top co-clustering pairs:",
            "",
            "| Scope | Feature A | Feature B | Same-Cluster Frequency |",
            "|---|---|---|---:|",
        ]
        if top_pairs.empty:
            lines.append("| -- | -- | -- | -- |")
        else:
            for row in top_pairs.sort_values(
                by=["same_cluster_frequency", "grouping_scope_label", "feature_a", "feature_b"],
                ascending=[False, True, True, True],
            ).head(8).to_dict(orient="records"):
                lines.append(
                    "| {scope} | {left} | {right} | {freq} |".format(
                        scope=_md_text(row["grouping_scope_label"]),
                        left=_md_text(row["feature_a"]),
                        right=_md_text(row["feature_b"]),
                        freq=_safe_fmt(row["same_cluster_frequency"]),
                    )
                )

        top_components = component_summary_df[
            (component_summary_df["track"] == track)
            & (component_summary_df["feature_set_id"] == feature_set_id)
            & (component_summary_df["grouping_spec_id"] == grouping_spec_id)
        ].copy()
        lines += [
            "",
            "Top component weights:",
            "",
            "| Scope | Cluster Signature | Feature | Weight Mean | Weight Std | Weight Flip | Loading Mean |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        if top_components.empty:
            lines.append("| -- | -- | -- | -- | -- | -- | -- |")
        else:
            for row in top_components.sort_values(
                by=["abs_weight_mean", "grouping_scope_label", "cluster_signature", "original_feature"],
                ascending=[False, True, True, True],
            ).head(10).to_dict(orient="records"):
                lines.append(
                    "| {scope} | {signature} | {feature} | {weight_mean} | {weight_std} | {weight_flip} | {loading_mean} |".format(
                        scope=_md_text(row["grouping_scope_label"]),
                        signature=_md_text(row["cluster_signature"]),
                        feature=_md_text(row["original_feature"]),
                        weight_mean=_safe_fmt(row["weight_mean"]),
                        weight_std=_safe_fmt(row["weight_std"]),
                        weight_flip=_safe_fmt(row["weight_sign_flip_rate"]),
                        loading_mean=_safe_fmt(row["loading_mean"]),
                    )
                )
    return "\n".join(lines)


def supervised_grouping_audit_markdown(audit_df: pd.DataFrame) -> str:
    lines = [
        "# Supervised Grouping Final Audit",
        "",
        "Private-test audit is deferred in this first grouped-routing pass.",
        "",
        "This table will be populated only after a grouped route is locked from CV.",
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
