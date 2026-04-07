from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from lib.shared.artifact_io import read_json
from lib.stability.family_registry import AlignedFamilyData
from lib.stability.routes import (
    _apply_rule_override,
    _build_model,
    _fill_missing,
    _metric_row,
    _select_threshold,
)


def load_stability_selection_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid stability-selection config: {path}")
    return payload


def family_display_label(label: str) -> str:
    return "HQ" if label == "HQ" else f"HQ + {label}"


def build_outer_splits(
    labels: pd.Series,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[dict[str, Any]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    y = labels.to_numpy(dtype=int)
    splits: list[dict[str, Any]] = []
    for split_index, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        repeat_index = split_index // n_splits
        fold_index = split_index % n_splits
        splits.append(
            {
                "outer_split_id": f"repeat_{repeat_index + 1:02d}_fold_{fold_index + 1:02d}",
                "repeat_index": repeat_index + 1,
                "fold_index": fold_index + 1,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )
    return splits


def build_row_subsamples(
    y: np.ndarray,
    *,
    fraction: float,
    n_subsamples: int,
    random_state: int,
    stratified: bool = True,
) -> list[np.ndarray]:
    if len(y) == 0:
        return []
    rng = np.random.default_rng(random_state)
    subsamples: list[np.ndarray] = []
    if stratified and len(np.unique(y)) > 1:
        class_to_indices = {
            int(label): np.flatnonzero(y == label)
            for label in sorted(set(int(value) for value in y.tolist()))
        }
        for _ in range(n_subsamples):
            sampled_parts: list[np.ndarray] = []
            for indices in class_to_indices.values():
                take = max(1, int(math.floor(len(indices) * fraction)))
                take = min(take, len(indices))
                sampled_parts.append(np.sort(rng.choice(indices, size=take, replace=False)))
            subsamples.append(np.sort(np.concatenate(sampled_parts)))
        return subsamples

    sample_size = max(2, int(math.floor(len(y) * fraction)))
    sample_size = min(sample_size, len(y))
    population = np.arange(len(y))
    for _ in range(n_subsamples):
        subsamples.append(np.sort(rng.choice(population, size=sample_size, replace=False)))
    return subsamples


def _prepare_model_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_use = train_df.copy()
    test_use = test_df.copy()
    train_use, test_use = _fill_missing(train_use, test_use, model=model_type)
    if model_type == "logistic":
        means = train_use.mean(numeric_only=True)
        stds = train_use.std(numeric_only=True).replace(0.0, 1.0)
        train_use = ((train_use - means) / stds).fillna(0.0)
        test_use = ((test_use - means) / stds).fillna(0.0)
    return train_use.to_numpy(dtype=float), test_use.to_numpy(dtype=float)


def _mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    if not sets:
        return 0.0
    if len(sets) == 1:
        return 1.0
    values: list[float] = []
    for left, right in itertools.combinations(sets, 2):
        if not left and not right:
            values.append(1.0)
        elif not left or not right:
            values.append(0.0)
        else:
            values.append(len(left & right) / len(left | right))
    return float(np.mean(values)) if values else 0.0


def _select_with_l1_coefficients(
    X_df: pd.DataFrame,
    y: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    selection_epsilon: float,
    random_state: int,
) -> np.ndarray:
    train_arr, _ = _prepare_model_frames(X_df, X_df, model_type="logistic")
    selector = LogisticRegression(
        solver="saga",
        penalty="l1",
        C=float(c_value),
        max_iter=max_iter,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        selector.fit(train_arr, y)
    return selector.coef_.reshape(-1)


def compute_selection_frequencies(
    combined_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    reasoning_columns: set[str],
    subsampling: dict[str, Any],
    selector: dict[str, Any],
    stability: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subsamples = build_row_subsamples(
        y_train,
        fraction=float(subsampling.get("fraction", 0.5)),
        n_subsamples=int(subsampling.get("n_subsamples", 100)),
        random_state=seed,
        stratified=bool(subsampling.get("stratified", True)),
    )
    feature_names = combined_train.columns.tolist()
    c_grid = [float(value) for value in selector.get("c_grid", [1.0])]
    counts_by_c = {c_value: np.zeros(len(feature_names), dtype=float) for c_value in c_grid}
    positive_counts_by_c = {c_value: np.zeros(len(feature_names), dtype=float) for c_value in c_grid}
    negative_counts_by_c = {c_value: np.zeros(len(feature_names), dtype=float) for c_value in c_grid}
    subsample_union_sets: list[set[str]] = []
    sign_consistency_threshold = float(stability.get("sign_consistency_threshold", 0.9))

    for subsample_index, subsample_rows in enumerate(subsamples):
        union_mask = np.zeros(len(feature_names), dtype=bool)
        X_sub = combined_train.iloc[subsample_rows].copy()
        y_sub = y_train[subsample_rows]
        for c_index, c_value in enumerate(c_grid):
            coefficients = _select_with_l1_coefficients(
                X_sub,
                y_sub,
                c_value=c_value,
                max_iter=int(selector.get("max_iter", 5000)),
                selection_epsilon=float(selector.get("selection_epsilon", 1e-8)),
                random_state=seed + subsample_index * 97 + c_index,
            )
            epsilon = float(selector.get("selection_epsilon", 1e-8))
            mask = np.abs(coefficients) > epsilon
            positive_mask = coefficients > epsilon
            negative_mask = coefficients < -epsilon
            counts_by_c[c_value] += mask.astype(float)
            positive_counts_by_c[c_value] += positive_mask.astype(float)
            negative_counts_by_c[c_value] += negative_mask.astype(float)
            union_mask |= mask
        subsample_union_sets.append(
            {
                feature_names[idx]
                for idx, selected in enumerate(union_mask.tolist())
                if selected and feature_names[idx] in reasoning_columns
            }
        )

    thresholds = [float(value) for value in stability.get("report_thresholds", [0.6, 0.8, 0.9])]
    rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(feature_names):
        per_c: dict[str, Any] = {}
        best_c_value = None
        best_total_freq = -1.0
        best_positive_freq = 0.0
        best_negative_freq = 0.0
        for c_value in c_grid:
            suffix = str(c_value).replace(".", "p")
            selection_freq = float(counts_by_c[c_value][feature_index] / max(1, len(subsamples)))
            positive_freq = float(positive_counts_by_c[c_value][feature_index] / max(1, len(subsamples)))
            negative_freq = float(negative_counts_by_c[c_value][feature_index] / max(1, len(subsamples)))
            per_c[f"selection_freq_c_{suffix}"] = selection_freq
            per_c[f"positive_selection_freq_c_{suffix}"] = positive_freq
            per_c[f"negative_selection_freq_c_{suffix}"] = negative_freq
            if selection_freq > best_total_freq:
                best_total_freq = selection_freq
                best_c_value = c_value
                best_positive_freq = positive_freq
                best_negative_freq = negative_freq
        max_freq = max(best_total_freq, 0.0)
        sign_consistency = (
            max(best_positive_freq, best_negative_freq) / max_freq if max_freq > 0.0 else 1.0
        )
        dominant_sign = "none"
        if max_freq > 0.0:
            dominant_sign = "positive" if best_positive_freq >= best_negative_freq else "negative"
        row = {
            "feature_name": feature_name,
            "feature_group": "reasoning" if feature_name in reasoning_columns else "hq",
            "selection_frequency": float(max_freq),
            "positive_selection_frequency": float(best_positive_freq),
            "negative_selection_frequency": float(best_negative_freq),
            "sign_consistency": float(sign_consistency),
            "sign_stable": bool(max_freq > 0.0 and sign_consistency >= sign_consistency_threshold),
            "dominant_sign": dominant_sign,
            "best_c_value": float(best_c_value) if best_c_value is not None else None,
            "subsample_count": int(len(subsamples)),
            "c_grid_size": int(len(c_grid)),
        }
        for threshold in thresholds:
            row[f"stable_at_{str(threshold).replace('.', 'p')}"] = bool(max_freq >= threshold)
        row.update(per_c)
        rows.append(row)

    feature_freq_df = pd.DataFrame(rows).sort_values(
        by=["feature_group", "selection_frequency", "feature_name"],
        ascending=[True, False, True],
    )
    reasoning_freqs = feature_freq_df[feature_freq_df["feature_group"] == "reasoning"][
        "selection_frequency"
    ].to_numpy(dtype=float)
    reasoning_sign_consistency = feature_freq_df[feature_freq_df["feature_group"] == "reasoning"][
        "sign_consistency"
    ].to_numpy(dtype=float)
    sign_stable_reasoning = feature_freq_df[
        (feature_freq_df["feature_group"] == "reasoning") & (feature_freq_df["sign_stable"])
    ]
    threshold_counts = {}
    for threshold in thresholds:
        threshold_counts[str(threshold)] = int(np.sum(reasoning_freqs >= threshold))
    selection_meta = {
        "subsample_fraction": float(subsampling.get("fraction", 0.5)),
        "n_subsamples": int(subsampling.get("n_subsamples", 100)),
        "c_grid": c_grid,
        "jaccard_mean": _mean_pairwise_jaccard(subsample_union_sets),
        "mean_selected_feature_count": float(np.mean([len(feature_set) for feature_set in subsample_union_sets]))
        if subsample_union_sets
        else 0.0,
        "std_selected_feature_count": float(np.std([len(feature_set) for feature_set in subsample_union_sets]))
        if subsample_union_sets
        else 0.0,
        "max_reasoning_frequency": float(np.max(reasoning_freqs)) if reasoning_freqs.size else 0.0,
        "mean_reasoning_frequency": float(np.mean(reasoning_freqs)) if reasoning_freqs.size else 0.0,
        "mean_reasoning_sign_consistency": float(np.mean(reasoning_sign_consistency))
        if reasoning_sign_consistency.size
        else 0.0,
        "min_reasoning_sign_consistency": float(np.min(reasoning_sign_consistency))
        if reasoning_sign_consistency.size
        else 0.0,
        "sign_stable_reasoning_count": int(len(sign_stable_reasoning)),
        "reasoning_threshold_counts": threshold_counts,
    }
    return feature_freq_df, selection_meta


def _inner_threshold(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    model_type: str,
    threshold_grid: str,
    rule_mask: np.ndarray | None,
    random_state: int,
    n_splits: int,
) -> float:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    use_splits = min(int(n_splits), positives, negatives)
    if use_splits < 2:
        train_arr, _ = _prepare_model_frames(train_df, train_df, model_type=model_type)
        model = _build_model({"model": model_type}, random_state=random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(train_arr, y_train)
        scores = model.predict_proba(train_arr)[:, 1]
        if rule_mask is not None:
            scores = _apply_rule_override(scores, rule_mask)
        return _select_threshold(y_train, scores, mode=threshold_grid)

    splitter = StratifiedKFold(n_splits=use_splits, shuffle=True, random_state=random_state)
    oof_scores = np.zeros(len(y_train), dtype=float)
    seen = np.zeros(len(y_train), dtype=bool)
    for inner_train_idx, inner_val_idx in splitter.split(np.zeros(len(y_train)), y_train):
        X_fit, X_val = _prepare_model_frames(
            train_df.iloc[inner_train_idx].copy(),
            train_df.iloc[inner_val_idx].copy(),
            model_type=model_type,
        )
        model = _build_model({"model": model_type}, random_state=random_state)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_fit, y_train[inner_train_idx])
        scores = model.predict_proba(X_val)[:, 1]
        if rule_mask is not None:
            scores = _apply_rule_override(scores, rule_mask[inner_val_idx])
        oof_scores[inner_val_idx] = scores
        seen[inner_val_idx] = True

    if not np.any(seen):
        return 0.5
    return _select_threshold(y_train[seen], oof_scores[seen], mode=threshold_grid)


def evaluate_selected_route(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    model_type: str,
    threshold_grid: str,
    apply_rule_override: bool,
    rule_mask_train: np.ndarray | None,
    rule_mask_test: np.ndarray | None,
    random_state: int,
    inner_threshold_cv_splits: int,
) -> dict[str, Any]:
    threshold = _inner_threshold(
        train_df,
        y_train,
        model_type=model_type,
        threshold_grid=threshold_grid,
        rule_mask=rule_mask_train if apply_rule_override else None,
        random_state=random_state,
        n_splits=inner_threshold_cv_splits,
    )
    X_train, X_test = _prepare_model_frames(train_df, test_df, model_type=model_type)
    model = _build_model({"model": model_type}, random_state=random_state)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    if apply_rule_override:
        scores = _apply_rule_override(scores, rule_mask_test)
    metrics = _metric_row(y_test, scores, threshold)
    metrics["threshold"] = float(threshold)
    return metrics


def _summary_value(frame: pd.DataFrame, mask: pd.Series, column: str) -> float | None:
    subset = frame.loc[mask, column]
    subset = subset.dropna()
    if subset.empty:
        return None
    return float(subset.iloc[0])


def _safe_fmt(value: Any, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "--"
    return f"{float(value):.{places}f}"


def run_family_stability_study(
    data: AlignedFamilyData,
    family_ids: list[str],
    config: dict[str, Any],
    *,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    family_specs = {family_id: data.family_specs[family_id] for family_id in family_ids}
    outer_cfg = config.get("outer_cv", {})
    outer_splits = build_outer_splits(
        data.labels,
        n_splits=int(outer_cfg.get("n_splits", 3)),
        n_repeats=int(outer_cfg.get("n_repeats", 10)),
        random_state=int(outer_cfg.get("random_state", 42)),
    )
    evaluation_cfg = config.get("evaluation", {})
    stability_cfg = config.get("stability", {})
    primary_threshold = float(stability_cfg.get("primary_threshold", 0.8))
    report_thresholds = [float(value) for value in stability_cfg.get("report_thresholds", [0.6, 0.8, 0.9])]
    sign_consistency_threshold = float(stability_cfg.get("sign_consistency_threshold", 0.9))
    models = [str(value) for value in evaluation_cfg.get("models", ["logistic", "xgb1"])]

    baseline = data.baseline.copy()
    labels = data.labels.to_numpy(dtype=int)
    rule_mask = baseline["exit_count"].fillna(0.0).astype(float).to_numpy() > 0 if "exit_count" in baseline.columns else None
    baseline_columns = baseline.columns.tolist()

    benchmark_rows: list[dict[str, Any]] = []
    feature_frequency_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    family_metric_rows: list[dict[str, Any]] = []

    total_splits = len(outer_splits)
    total_units = max(1, total_splits * max(1, len(family_ids)))
    completed_units = 0

    for split_offset, split in enumerate(outer_splits):
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
                    "split_index": split_offset + 1,
                    "total_splits": total_splits,
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
                random_state=int(evaluation_cfg.get("random_state", 42)) + split_offset,
                inner_threshold_cv_splits=int(evaluation_cfg.get("inner_threshold_cv_splits", 3)),
            )
            benchmark_rows.append(
                {
                    "family_id": "HQ",
                    "family_label": "HQ",
                    "family_display_label": "HQ",
                    "outer_split_id": outer_split_id,
                    "repeat_index": int(split["repeat_index"]),
                    "fold_index": int(split["fold_index"]),
                    "route_group": "hq_only",
                    "evaluator_model": model_type,
                    "selected_feature_count": int(len(baseline_columns)),
                    "selected_hq_feature_count": int(len(baseline_columns)),
                    "selected_reasoning_feature_count": 0,
                    "selection_fallback": "none",
                    "mean_reasoning_frequency": 0.0,
                    "max_reasoning_frequency": 0.0,
                    "reasoning_features_above_60": 0,
                    "reasoning_features_above_80": 0,
                    "reasoning_features_above_90": 0,
                    "reasoning_jaccard_mean": 0.0,
                    "mean_reasoning_sign_consistency": 0.0,
                    "min_reasoning_sign_consistency": 0.0,
                    "sign_stable_reasoning_count": 0,
                    "threshold": float(metrics["threshold"]),
                    "f0_5": float(metrics["f0_5"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "precision_at_10": float(metrics["precision_at_10"]),
                    "brier": float(metrics["brier"]),
                }
            )

        for family_id in family_ids:
            family_spec = family_specs[family_id]
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "family_start",
                        "outer_split_id": outer_split_id,
                        "family_id": family_id,
                        "family_label": family_spec.label,
                        "completed_units": completed_units,
                        "total_units": total_units,
                    }
                )
            family_frame = data.candidate_frames[family_id].copy()
            family_train = family_frame.iloc[train_idx].copy()
            family_test = family_frame.iloc[test_idx].copy()
            family_columns = family_train.columns.tolist()
            combined_train = pd.concat([baseline_train, family_train], axis=1)
            combined_test = pd.concat([baseline_test, family_test], axis=1)
            combined_train = combined_train.loc[:, ~combined_train.columns.duplicated()].copy()
            combined_test = combined_test.loc[:, combined_train.columns].copy()

            feature_freq_df, selection_meta = compute_selection_frequencies(
                combined_train,
                y_train,
                reasoning_columns=set(family_columns),
                subsampling=dict(config.get("subsampling", {})),
                selector=dict(config.get("selector", {})),
                stability=stability_cfg,
                seed=int(config.get("subsampling", {}).get("random_state", 42)) + split_offset * 101 + len(feature_frequency_rows),
            )
            feature_freq_df["family_id"] = family_id
            feature_freq_df["family_label"] = family_spec.label
            feature_freq_df["family_display_label"] = family_display_label(family_spec.label)
            feature_freq_df["outer_split_id"] = outer_split_id
            feature_freq_df["repeat_index"] = int(split["repeat_index"])
            feature_freq_df["fold_index"] = int(split["fold_index"])
            feature_frequency_rows.extend(feature_freq_df.to_dict(orient="records"))

            stable_reasoning_features = feature_freq_df[
                (feature_freq_df["feature_group"] == "reasoning")
                & (feature_freq_df["selection_frequency"] >= primary_threshold)
                & (feature_freq_df["sign_consistency"] >= sign_consistency_threshold)
            ]["feature_name"].tolist()
            stable_competition_features = feature_freq_df[
                (
                    (feature_freq_df["feature_group"] == "hq")
                    & (feature_freq_df["selection_frequency"] >= primary_threshold)
                )
                | (
                    (feature_freq_df["feature_group"] == "reasoning")
                    & (feature_freq_df["selection_frequency"] >= primary_threshold)
                    & (feature_freq_df["sign_consistency"] >= sign_consistency_threshold)
                )
            ]["feature_name"].tolist()
            if not stable_competition_features:
                stable_competition_features = list(baseline_columns)
                competition_fallback = "hq_only_safe_fallback"
            else:
                competition_fallback = "none"
            augmentation_features = list(dict.fromkeys(list(baseline_columns) + stable_reasoning_features))
            augmentation_fallback = "none" if stable_reasoning_features else "hq_only_safe_fallback"

            threshold_count_lookup = {
                f"reasoning_features_above_{str(threshold).replace('.', 'p')}": int(
                    np.sum(
                        feature_freq_df[feature_freq_df["feature_group"] == "reasoning"]["selection_frequency"].to_numpy(dtype=float) >= threshold
                    )
                )
                for threshold in report_thresholds
            }

            selection_rows.append(
                {
                    "family_id": family_id,
                    "family_label": family_spec.label,
                    "family_display_label": family_display_label(family_spec.label),
                    "outer_split_id": outer_split_id,
                    "repeat_index": int(split["repeat_index"]),
                    "fold_index": int(split["fold_index"]),
                    "stable_competition_feature_count": int(len(stable_competition_features)),
                    "stable_competition_hq_feature_count": int(sum(feature in baseline_columns for feature in stable_competition_features)),
                    "stable_competition_reasoning_feature_count": int(sum(feature in family_columns for feature in stable_competition_features)),
                    "stable_augmentation_hq_feature_count": int(len(baseline_columns)),
                    "stable_augmentation_reasoning_feature_count": int(len(stable_reasoning_features)),
                    "competition_fallback": competition_fallback,
                    "augmentation_fallback": augmentation_fallback,
                    "mean_reasoning_frequency": float(selection_meta["mean_reasoning_frequency"]),
                    "max_reasoning_frequency": float(selection_meta["max_reasoning_frequency"]),
                    "reasoning_jaccard_mean": float(selection_meta["jaccard_mean"]),
                    "subsample_reasoning_count_mean": float(selection_meta["mean_selected_feature_count"]),
                    "subsample_reasoning_count_std": float(selection_meta["std_selected_feature_count"]),
                    "mean_reasoning_sign_consistency": float(selection_meta["mean_reasoning_sign_consistency"]),
                    "min_reasoning_sign_consistency": float(selection_meta["min_reasoning_sign_consistency"]),
                    "sign_stable_reasoning_count": int(selection_meta["sign_stable_reasoning_count"]),
                    **threshold_count_lookup,
                }
            )

            raw_features = list(dict.fromkeys(list(baseline_columns) + list(family_columns)))
            route_feature_sets = [
                ("raw_family", raw_features, "none"),
                ("competition_track", stable_competition_features, competition_fallback),
                ("augmentation_track", augmentation_features, augmentation_fallback),
            ]
            for route_group, selected_features, fallback_label in route_feature_sets:
                train_selected = combined_train[selected_features].copy()
                test_selected = combined_test[selected_features].copy()
                selected_hq_count = int(sum(feature in baseline_columns for feature in selected_features))
                selected_reasoning_count = int(sum(feature in family_columns for feature in selected_features))
                for model_type in models:
                    metrics = evaluate_selected_route(
                        train_selected,
                        test_selected,
                        y_train,
                        y_test,
                        model_type=model_type,
                        threshold_grid=str(evaluation_cfg.get("threshold_grid", "default")),
                        apply_rule_override=bool(evaluation_cfg.get("apply_rule_override", True)),
                        rule_mask_train=rule_mask_train,
                        rule_mask_test=rule_mask_test,
                        random_state=int(evaluation_cfg.get("random_state", 42)) + split_offset,
                        inner_threshold_cv_splits=int(evaluation_cfg.get("inner_threshold_cv_splits", 3)),
                    )
                    family_metric_rows.append(
                        {
                            "family_id": family_id,
                            "family_label": family_spec.label,
                            "family_display_label": family_display_label(family_spec.label),
                            "outer_split_id": outer_split_id,
                            "repeat_index": int(split["repeat_index"]),
                            "fold_index": int(split["fold_index"]),
                            "route_group": route_group,
                            "evaluator_model": model_type,
                            "selected_feature_count": int(len(selected_features)),
                            "selected_hq_feature_count": selected_hq_count,
                            "selected_reasoning_feature_count": selected_reasoning_count,
                            "selection_fallback": fallback_label,
                            "mean_reasoning_frequency": float(selection_meta["mean_reasoning_frequency"]),
                            "max_reasoning_frequency": float(selection_meta["max_reasoning_frequency"]),
                            "reasoning_features_above_60": int(threshold_count_lookup.get("reasoning_features_above_0p6", 0)),
                            "reasoning_features_above_80": int(threshold_count_lookup.get("reasoning_features_above_0p8", 0)),
                            "reasoning_features_above_90": int(threshold_count_lookup.get("reasoning_features_above_0p9", 0)),
                            "reasoning_jaccard_mean": float(selection_meta["jaccard_mean"]),
                            "mean_reasoning_sign_consistency": float(selection_meta["mean_reasoning_sign_consistency"]),
                            "min_reasoning_sign_consistency": float(selection_meta["min_reasoning_sign_consistency"]),
                            "sign_stable_reasoning_count": int(selection_meta["sign_stable_reasoning_count"]),
                            "threshold": float(metrics["threshold"]),
                            "f0_5": float(metrics["f0_5"]),
                            "precision": float(metrics["precision"]),
                            "recall": float(metrics["recall"]),
                            "pr_auc": float(metrics["pr_auc"]),
                            "precision_at_10": float(metrics["precision_at_10"]),
                            "brier": float(metrics["brier"]),
                        }
                    )
            completed_units += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "family_done",
                        "outer_split_id": outer_split_id,
                        "family_id": family_id,
                        "family_label": family_spec.label,
                        "completed_units": completed_units,
                        "total_units": total_units,
                    }
                )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "outer_split_done",
                    "outer_split_id": outer_split_id,
                    "split_index": split_offset + 1,
                    "total_splits": total_splits,
                    "completed_units": completed_units,
                    "total_units": total_units,
                }
            )

    benchmark_df = pd.DataFrame(benchmark_rows)
    family_metrics_df = pd.DataFrame(family_metric_rows)
    feature_frequency_df = pd.DataFrame(feature_frequency_rows)
    selection_df = pd.DataFrame(selection_rows)

    metric_frames = [frame for frame in [benchmark_df, family_metrics_df] if not frame.empty]
    combined_metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    if not combined_metrics_df.empty:
        for (family_id, family_label, family_display_label_text, route_group, evaluator_model), subset in combined_metrics_df.groupby(
            ["family_id", "family_label", "family_display_label", "route_group", "evaluator_model"],
            dropna=False,
        ):
            summary_rows.append(
                {
                    "family_id": family_id,
                    "family_label": family_label,
                    "family_display_label": family_display_label_text,
                    "route_group": route_group,
                    "evaluator_model": evaluator_model,
                    "outer_f0_5_mean": float(subset["f0_5"].mean()),
                    "outer_f0_5_std": float(subset["f0_5"].std(ddof=0)),
                    "outer_precision_mean": float(subset["precision"].mean()),
                    "outer_recall_mean": float(subset["recall"].mean()),
                    "outer_pr_auc_mean": float(subset["pr_auc"].mean()),
                    "outer_precision_at_10_mean": float(subset["precision_at_10"].mean()),
                    "threshold_mean": float(subset["threshold"].mean()),
                    "threshold_std": float(subset["threshold"].std(ddof=0)),
                    "selected_feature_count_mean": float(subset["selected_feature_count"].mean()),
                    "selected_feature_count_std": float(subset["selected_feature_count"].std(ddof=0)),
                    "selected_hq_feature_count_mean": float(subset["selected_hq_feature_count"].mean()),
                    "selected_reasoning_feature_count_mean": float(subset["selected_reasoning_feature_count"].mean()),
                    "mean_reasoning_frequency": float(subset["mean_reasoning_frequency"].mean()),
                    "max_reasoning_frequency": float(subset["max_reasoning_frequency"].mean()),
                    "reasoning_jaccard_mean": float(subset["reasoning_jaccard_mean"].mean()),
                    "mean_reasoning_sign_consistency": float(subset["mean_reasoning_sign_consistency"].mean()),
                    "min_reasoning_sign_consistency": float(subset["min_reasoning_sign_consistency"].mean()),
                    "sign_stable_reasoning_count_mean": float(subset["sign_stable_reasoning_count"].mean()),
                    "reasoning_features_above_60_mean": float(subset["reasoning_features_above_60"].mean()),
                    "reasoning_features_above_80_mean": float(subset["reasoning_features_above_80"].mean()),
                    "reasoning_features_above_90_mean": float(subset["reasoning_features_above_90"].mean()),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    family_selection_summary = pd.DataFrame()
    if not selection_df.empty:
        family_selection_summary = (
            selection_df.groupby(["family_id", "family_label", "family_display_label"], dropna=False)
            .agg(
                competition_selected_count_mean=("stable_competition_feature_count", "mean"),
                competition_selected_count_std=("stable_competition_feature_count", "std"),
                competition_hq_selected_mean=("stable_competition_hq_feature_count", "mean"),
                competition_reasoning_selected_mean=("stable_competition_reasoning_feature_count", "mean"),
                augmentation_reasoning_selected_mean=("stable_augmentation_reasoning_feature_count", "mean"),
                augmentation_reasoning_selected_std=("stable_augmentation_reasoning_feature_count", "std"),
                mean_reasoning_frequency=("mean_reasoning_frequency", "mean"),
                max_reasoning_frequency=("max_reasoning_frequency", "mean"),
                reasoning_jaccard_mean=("reasoning_jaccard_mean", "mean"),
                mean_reasoning_sign_consistency=("mean_reasoning_sign_consistency", "mean"),
                min_reasoning_sign_consistency=("min_reasoning_sign_consistency", "mean"),
                sign_stable_reasoning_count_mean=("sign_stable_reasoning_count", "mean"),
                reasoning_features_above_60_mean=("reasoning_features_above_0p6", "mean"),
                reasoning_features_above_80_mean=("reasoning_features_above_0p8", "mean"),
                reasoning_features_above_90_mean=("reasoning_features_above_0p9", "mean"),
            )
            .reset_index()
            .fillna(0.0)
        )

    audit_df = pd.DataFrame(
        [
            {
                "candidate": "",
                "prediction_path": "",
                "external_test_f0_5": "",
                "status": "pending_external_audit",
            }
        ]
    )
    return {
        "outer_splits": outer_splits,
        "benchmark_metrics": benchmark_df,
        "family_metrics": family_metrics_df,
        "feature_frequencies": feature_frequency_df,
        "selection_summary": selection_df,
        "summary": summary_df,
        "family_selection_summary": family_selection_summary,
        "audit": audit_df,
    }


def stability_protocol_markdown(config: dict[str, Any], family_labels: list[str]) -> str:
    outer_cfg = config.get("outer_cv", {})
    subsampling = config.get("subsampling", {})
    selector = config.get("selector", {})
    stability = config.get("stability", {})
    thresholds = ", ".join(str(value) for value in stability.get("report_thresholds", [0.6, 0.8, 0.9]))
    c_grid = ", ".join(str(value) for value in selector.get("c_grid", []))
    lines = [
        "# Stability Selection Protocol",
        "",
        "This Step 2 method pass tests row-subsampled stability selection before any new `PLS` work.",
        "",
        "## Study Setup",
        "",
        f"- Families in scope: {', '.join(family_labels)}",
        "- Units are evaluated as `HQ + family`, not family-only models.",
        "- Selector: `LogisticRegression(solver=\"saga\", penalty=\"l1\")`",
        f"- C grid: {c_grid}",
        f"- Outer CV: {int(outer_cfg.get('n_splits', 3))}-fold repeated {int(outer_cfg.get('n_repeats', 10))} times",
        f"- Row subsamples per outer-train fold: {int(subsampling.get('n_subsamples', 100))}",
        f"- Subsample fraction: {float(subsampling.get('fraction', 0.5)):.2f}",
        f"- Primary stability threshold: {float(stability.get('primary_threshold', 0.8)):.2f}",
        f"- Sign-consistency threshold for reasoning features: {float(stability.get('sign_consistency_threshold', 0.9)):.2f}",
        f"- Report thresholds: {thresholds}",
        "",
        "## Tracks",
        "",
        "- `competition_track`: HQ features survive on selection frequency alone; reasoning features must pass both the frequency and sign-consistency thresholds.",
        "- `augmentation_track`: HQ is always retained in the final refit, while reasoning features are admitted only if they pass both thresholds.",
        "",
        "## Evaluation",
        "",
        "- Downstream evaluators: `LR` and `XGB1`",
        "- Thresholds are chosen using train-only inner CV inside each outer fold.",
        "- Private-test scoring is not available locally in this pass; the final audit table is reserved for locked candidates and external submission results.",
    ]
    return "\n".join(lines)


def stability_summary_markdown(
    summary_df: pd.DataFrame,
    family_selection_summary: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> str:
    benchmark_summary = (
        benchmark_df.groupby("evaluator_model", dropna=False)
        .agg(outer_f0_5_mean=("f0_5", "mean"), outer_f0_5_std=("f0_5", "std"))
        .reset_index()
        if not benchmark_df.empty
        else pd.DataFrame()
    )
    lines = [
        "# Stability Selection Summary",
        "",
        "This note separates route performance from selector behavior for each `HQ + family` pilot.",
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
        "## Route Comparison",
        "",
        "| Family | Raw LR | Comp LR | Aug LR | Raw XGB1 | Comp XGB1 | Aug XGB1 | Comp HQ Kept | Comp Reason Kept | Aug Reason Kept | Mean Reason Freq | Mean Sign Consistency | Mean Sign-Consistent Reason | Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    family_rows = [row for row in family_selection_summary.to_dict(orient="records") if row.get("family_id") != "HQ"]
    for family_row in family_rows:
        family_id = str(family_row["family_id"])
        display = str(family_row["family_display_label"])

        def lookup(route_group: str, evaluator: str) -> str:
            mask = (
                (summary_df["family_id"] == family_id)
                & (summary_df["route_group"] == route_group)
                & (summary_df["evaluator_model"] == evaluator)
            )
            mean_value = _summary_value(summary_df, mask, "outer_f0_5_mean")
            std_value = _summary_value(summary_df, mask, "outer_f0_5_std")
            if mean_value is None:
                return "--"
            return f"{_safe_fmt(mean_value)} +/- {_safe_fmt(std_value)}"

        lines.append(
            "| {family} | {raw_lr} | {comp_lr} | {aug_lr} | {raw_xgb} | {comp_xgb} | {aug_xgb} | {comp_hq} | {comp_reason} | {aug_reason} | {mean_freq} | {mean_sign} | {sign_stable} | {jaccard} |".format(
                family=display,
                raw_lr=lookup("raw_family", "logistic"),
                comp_lr=lookup("competition_track", "logistic"),
                aug_lr=lookup("augmentation_track", "logistic"),
                raw_xgb=lookup("raw_family", "xgb1"),
                comp_xgb=lookup("competition_track", "xgb1"),
                aug_xgb=lookup("augmentation_track", "xgb1"),
                comp_hq=_safe_fmt(family_row.get("competition_hq_selected_mean")),
                comp_reason=_safe_fmt(family_row.get("competition_reasoning_selected_mean")),
                aug_reason=_safe_fmt(family_row.get("augmentation_reasoning_selected_mean")),
                mean_freq=_safe_fmt(family_row.get("mean_reasoning_frequency")),
                mean_sign=_safe_fmt(family_row.get("mean_reasoning_sign_consistency")),
                sign_stable=_safe_fmt(family_row.get("sign_stable_reasoning_count_mean")),
                jaccard=_safe_fmt(family_row.get("reasoning_jaccard_mean")),
            )
        )
    if not family_rows:
        lines.append("| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |")
    return "\n".join(lines)


def stability_details_markdown(
    feature_freq_df: pd.DataFrame,
    family_selection_summary: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> str:
    lines = [
        "# Stability Selection Family Details",
        "",
        "This note shows, for each family, how route performance changed and which reasoning features survived the sign-aware selector.",
    ]
    family_rows = family_selection_summary.to_dict(orient="records")
    for family_row in family_rows:
        family_id = str(family_row["family_id"])
        if family_id == "HQ":
            continue
        display = str(family_row["family_display_label"])
        lines += [
            "",
            f"## {display}",
            "",
            f"- Mean reasoning selection frequency: {_safe_fmt(family_row.get('mean_reasoning_frequency'))}",
            f"- Mean reasoning sign consistency: {_safe_fmt(family_row.get('mean_reasoning_sign_consistency'))}",
            f"- Minimum reasoning sign consistency: {_safe_fmt(family_row.get('min_reasoning_sign_consistency'))}",
            f"- Mean sign-consistent reasoning features across outer splits: {_safe_fmt(family_row.get('sign_stable_reasoning_count_mean'))}",
            f"- Mean admitted reasoning features after both thresholds: {_safe_fmt(family_row.get('augmentation_reasoning_selected_mean'))}",
            f"- Mean subsample Jaccard: {_safe_fmt(family_row.get('reasoning_jaccard_mean'))}",
            "",
            "| Route | LR F0.5 | LR Std | XGB1 F0.5 | XGB1 Std | Mean Selected Features | Mean HQ Features | Mean Reasoning Features |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for route_group, label in [
            ("raw_family", "Raw"),
            ("competition_track", "Competition"),
            ("augmentation_track", "Augmentation"),
        ]:
            lr_mask = (
                (summary_df["family_id"] == family_id)
                & (summary_df["route_group"] == route_group)
                & (summary_df["evaluator_model"] == "logistic")
            )
            xgb_mask = (
                (summary_df["family_id"] == family_id)
                & (summary_df["route_group"] == route_group)
                & (summary_df["evaluator_model"] == "xgb1")
            )
            lines.append(
                "| {label} | {lr} | {lr_std} | {xgb} | {xgb_std} | {selected} | {hq} | {reasoning} |".format(
                    label=label,
                    lr=_safe_fmt(_summary_value(summary_df, lr_mask, "outer_f0_5_mean")),
                    lr_std=_safe_fmt(_summary_value(summary_df, lr_mask, "outer_f0_5_std")),
                    xgb=_safe_fmt(_summary_value(summary_df, xgb_mask, "outer_f0_5_mean")),
                    xgb_std=_safe_fmt(_summary_value(summary_df, xgb_mask, "outer_f0_5_std")),
                    selected=_safe_fmt(_summary_value(summary_df, lr_mask, "selected_feature_count_mean")),
                    hq=_safe_fmt(_summary_value(summary_df, lr_mask, "selected_hq_feature_count_mean")),
                    reasoning=_safe_fmt(_summary_value(summary_df, lr_mask, "selected_reasoning_feature_count_mean")),
                )
            )
        family_features = feature_freq_df[feature_freq_df["family_id"] == family_id].copy()
        if family_features.empty:
            continue
        grouped = (
            family_features.groupby(["feature_name", "feature_group"], dropna=False)
            .agg(
                selection_frequency_mean=("selection_frequency", "mean"),
                positive_selection_frequency_mean=("positive_selection_frequency", "mean"),
                negative_selection_frequency_mean=("negative_selection_frequency", "mean"),
                sign_consistency_mean=("sign_consistency", "mean"),
                sign_stable_rate=("sign_stable", "mean"),
                dominant_sign_mode=("dominant_sign", lambda values: pd.Series(values).mode().iloc[0] if not pd.Series(values).mode().empty else "--"),
            )
            .reset_index()
            .sort_values(by=["feature_group", "selection_frequency_mean", "feature_name"], ascending=[True, False, True])
        )
        top_reasoning = grouped[grouped["feature_group"] == "reasoning"].head(8)
        top_hq = grouped[grouped["feature_group"] == "hq"].head(8)
        lines += [
            "",
            "Top reasoning features:",
            "",
            "| Feature | Mean Selection Frequency | Mean Positive Freq | Mean Negative Freq | Mean Sign Consistency | Share Sign-Consistent | Dominant Sign |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in top_reasoning.to_dict(orient="records"):
            lines.append(
                "| {feature} | {selection} | {positive} | {negative} | {sign_consistency} | {share_sign_stable} | {dominant_sign} |".format(
                    feature=row["feature_name"],
                    selection=_safe_fmt(row["selection_frequency_mean"]),
                    positive=_safe_fmt(row.get("positive_selection_frequency_mean")),
                    negative=_safe_fmt(row.get("negative_selection_frequency_mean")),
                    sign_consistency=_safe_fmt(row.get("sign_consistency_mean")),
                    share_sign_stable=_safe_fmt(row.get("sign_stable_rate")),
                    dominant_sign=row.get("dominant_sign_mode", "--"),
                )
            )
        if top_reasoning.empty:
            lines.append("| -- | -- | -- | -- | -- | -- | -- |")
        lines += [
            "",
            "Top HQ features in the competition-track selector view:",
            "",
            "| Feature | Mean Selection Frequency |",
            "|---|---:|",
        ]
        for row in top_hq.to_dict(orient="records"):
            lines.append(f"| {row['feature_name']} | {_safe_fmt(row['selection_frequency_mean'])} |")
        if top_hq.empty:
            lines.append("| -- | -- |")
    return "\n".join(lines)


def stability_audit_markdown(audit_df: pd.DataFrame) -> str:
    lines = [
        "# Stability Selection Final Audit",
        "",
        "No scored private-test audit is available locally in this pass.",
        "",
        "This table is reserved for locked candidates and external submission results.",
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
