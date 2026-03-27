from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.neural_network import MLPClassifier

from think_reason_learn.datasets import load_vcbench
from think_reason_learn.features import FeatureEvaluator
from think_reason_learn.features._types import Rule
from think_reason_learn.datasets._vcbench import VCBENCH_HELPERS

from lib.cv_folds import load_or_create_folds
from pipelines.vcbench_pipeline import (
    HQ_FEATURES_BASE,
    _apply_rule_override,
    _build_high_quality_features,
    _metrics_from_scores,
    _select_threshold,
    _standardize_continuous,
)
from lib.paths import BASE_DIR

OUTPUT_DIR = BASE_DIR / "docs" / "model_testing"
ENGINEERED_SET_ID_DEFAULT = "set_05"
PERM_REPEATS = 3
SHAP_VAL_SAMPLE = 200
SHAP_BG_SAMPLE = 200
INTERP_COMBOS = {"HQ", "D", "A+B+C+D+E+F"}
PCA_VARIANCE_DEFAULT = 0.999
PCA_VARIANCE = PCA_VARIANCE_DEFAULT
PCA_SWEEP_VALUES = [0.9, 0.95, 0.99, 0.999, 0.9999]
PLS_COMPONENTS_DEFAULT = 10
SFT_K_DEFAULT = 30
PLS_SWEEP_VALUES = [2, 4, 6, 8, 10]
SFT_SWEEP_VALUES = [5, 10, 15, 20, 25, 30]


@dataclass
class ModelRun:
    name: str
    model_type: str
    feature_names: list[str]
    rule_mask: np.ndarray | None = None
    family: str = ""
    reasoning_combo: str = ""


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _pick_latest_family_id(archives_dir: Path) -> str:
    families = [d for d in archives_dir.iterdir() if d.is_dir() and d.name.startswith("family_")]
    if not families:
        raise RuntimeError(f"No engineered families found in {archives_dir}.")
    latest = max(families, key=lambda p: p.stat().st_mtime)
    return latest.name.replace("family_", "")


def _load_engineered_rules(family_id: str, set_id: str) -> list[Rule]:
    rules_path = (
        BASE_DIR
        / "features_storage"
        / "llm_engineered"
        / "archives"
        / f"family_{family_id}"
        / set_id
        / "current"
        / "llm_rules.json"
    )
    if not rules_path.exists():
        raise RuntimeError(f"Missing engineered rules: {rules_path}")
    data = json.loads(rules_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Invalid rule format in {rules_path}")
    rules: list[Rule] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        expression = str(item.get("expression", "")).strip()
        if not name or not expression:
            continue
        try:
            compile(expression, "<rule>", "eval")
        except SyntaxError:
            print(f"[model_testing] Skipping rule with invalid syntax: {name}")
            continue
        rules.append(Rule(name=name, description=description, expression=expression))
    if not rules:
        raise RuntimeError(f"No usable rules found in {rules_path}")
    return rules


def _evaluate_engineered_rules(records: list[dict[str, Any]], rules: list[Rule]) -> pd.DataFrame:
    evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)
    df = evaluator.evaluate_df(records)
    # Preserve rule order
    ordered = [r.name for r in rules if r.name in df.columns]
    if ordered:
        df = df[ordered].copy()
    return df


def _reindex_reasoning(
    reasoning_df: pd.DataFrame,
    founder_ids: list[str],
) -> pd.DataFrame:
    df = reasoning_df.copy()
    if "founder_uuid" in df.columns:
        df = df.set_index("founder_uuid").reindex(founder_ids)
        df = df.reset_index(drop=True)
    elif "row_index" in df.columns:
        df = df.sort_values("row_index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def _build_reasoning_combos_numeric(
    reasoning_df: pd.DataFrame,
    experiment_ids: list[str],
) -> dict[str, list[str]]:
    numeric_cols = {
        c
        for c in reasoning_df.columns
        if c not in ("founder_uuid", "success", "row_index")
        and pd.api.types.is_numeric_dtype(reasoning_df[c])
    }
    exp_to_cols: dict[str, list[str]] = {}
    for exp_id in experiment_ids:
        if not exp_id:
            continue
        cols = [c for c in reasoning_df.columns if c.startswith(f"{exp_id}_") and c in numeric_cols]
        if cols:
            exp_to_cols[exp_id] = cols
    combos: dict[str, list[str]] = {}
    exp_keys = list(exp_to_cols.keys())
    for r in range(1, len(exp_keys) + 1):
        for subset in combinations(exp_keys, r):
            combo = "+".join(subset)
            cols: list[str] = []
            for exp_id in subset:
                cols.extend(exp_to_cols.get(exp_id, []))
            if cols:
                combos[combo] = sorted(cols)
    return combos


def _train_xgboost_depth(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray, Any]:
    try:
        import xgboost as xgb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "xgboost is required for model_type=xgb. Install with: pip install xgboost"
        ) from exc
    params = {
        "n_estimators": 227,
        "max_depth": int(max_depth),
        "learning_rate": 0.0674,
        "subsample": 0.949,
        "colsample_bytree": 0.413,
        "scale_pos_weight": 10,
        "min_child_weight": 14,
        "gamma": 4.19,
        "reg_alpha": 0.73,
        "reg_lambda": 15.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": 1,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    return train_scores, test_scores, model


def _train_model_local(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    random_state: int,
    logistic_penalty: str = "elasticnet",
    logistic_c: float = 1.0,
    logistic_l1_ratio: float | None = 0.5,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if model_type in {"logistic", "elasticnet"}:
        penalty = logistic_penalty
        solver = "lbfgs"
        l1_ratio = None
        if penalty in {"l1", "elasticnet"}:
            solver = "saga"
        if penalty == "elasticnet":
            l1_ratio = 0.5 if logistic_l1_ratio is None else float(logistic_l1_ratio)
        clf = LogisticRegression(
            max_iter=3000,
            random_state=random_state,
            penalty=penalty,
            C=float(logistic_c),
            solver=solver,
            l1_ratio=l1_ratio,
        )
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        return train_scores, test_scores, clf
    if model_type == "xgb1":
        return _train_xgboost_depth(X_train, y_train, X_test, random_state, max_depth=1)
    if model_type == "xgb3":
        return _train_xgboost_depth(X_train, y_train, X_test, random_state, max_depth=3)
    if model_type == "mlp32":
        clf = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            random_state=random_state,
            max_iter=2000,
        )
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        return train_scores, test_scores, clf
    if model_type == "mlp4":
        clf = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="relu",
            random_state=random_state,
            max_iter=2000,
        )
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        return train_scores, test_scores, clf
    if model_type == "mlp2":
        clf = MLPClassifier(
            hidden_layer_sizes=(2,),
            activation="relu",
            random_state=random_state,
            max_iter=2000,
        )
        clf.fit(X_train, y_train)
        train_scores = clf.predict_proba(X_train)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]
        return train_scores, test_scores, clf
    raise ValueError(f"Unknown model_type={model_type}")


def _preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    model_type: str,
    transform: str,
    y_train: np.ndarray | None,
    pca_variance: float,
    pls_components: int,
    sft_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Any | None]:
    X_train = train_df.copy().apply(pd.to_numeric, errors="coerce")
    X_test = test_df.copy().apply(pd.to_numeric, errors="coerce")
    if X_train.isna().any().any() or X_test.isna().any().any():
        if model_type in ("xgb1", "xgb3"):
            X_train = X_train.fillna(0.0)
            X_test = X_test.fillna(0.0)
        else:
            fill_values = X_train.mean(numeric_only=True)
            X_train = X_train.fillna(fill_values)
            X_test = X_test.fillna(fill_values)
            X_train = X_train.fillna(0.0)
            X_test = X_test.fillna(0.0)
    transform = transform.upper()
    needs_scale = transform in {"PCA", "PLS"} or model_type in ("logistic", "elasticnet", "mlp32", "mlp4", "mlp2")
    if needs_scale:
        X_train, X_test = _standardize_continuous(X_train, X_test, feature_names)

    if transform == "PCA":
        pca = PCA(n_components=pca_variance, svd_solver="full", random_state=42)
        X_train_arr = pca.fit_transform(X_train.values.astype(float))
        X_test_arr = pca.transform(X_test.values.astype(float))
        n_components = X_train_arr.shape[1]
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        X_train = pd.DataFrame(X_train_arr, columns=pca_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_arr, columns=pca_cols, index=X_test.index)
        return X_train, X_test, pca_cols, pca

    if transform == "PLS":
        if y_train is None:
            raise RuntimeError("PLS requires y_train to be provided.")
        n_components = min(pls_components, X_train.shape[1], max(1, len(y_train) - 1))
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train.values.astype(float), y_train)
        X_train_arr = pls.transform(X_train.values.astype(float))
        X_test_arr = pls.transform(X_test.values.astype(float))
        pls_cols = [f"PLS{i+1}" for i in range(X_train_arr.shape[1])]
        X_train = pd.DataFrame(X_train_arr, columns=pls_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_arr, columns=pls_cols, index=X_test.index)
        return X_train, X_test, pls_cols, pls

    if transform == "SFT":
        if y_train is None:
            raise RuntimeError("SFT requires y_train to be provided.")
        k = min(max(1, sft_k), X_train.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(X_train.values.astype(float), y_train)
        X_train_arr = selector.transform(X_train.values.astype(float))
        X_test_arr = selector.transform(X_test.values.astype(float))
        support = selector.get_support()
        selected = [name for name, keep in zip(feature_names, support) if keep]
        X_train = pd.DataFrame(X_train_arr, columns=selected, index=X_train.index)
        X_test = pd.DataFrame(X_test_arr, columns=selected, index=X_test.index)
        return X_train, X_test, selected, selector

    return X_train, X_test, feature_names, None


def _oof_cv_metrics(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    rule_mask: np.ndarray | None = None,
    transform: str = "BASE",
    y_train: np.ndarray | None = None,
    pca_variance: float = PCA_VARIANCE_DEFAULT,
    pls_components: int = PLS_COMPONENTS_DEFAULT,
    sft_k: int = SFT_K_DEFAULT,
    logistic_penalty: str = "elasticnet",
    logistic_c: float = 1.0,
    logistic_l1_ratio: float | None = 0.5,
) -> tuple[dict[str, float], dict[str, float], float, list[dict[str, Any]]]:
    oof_scores: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []
    fold_artifacts: list[dict[str, Any]] = []

    for train_idx, test_idx in splits:
        X_train = df.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()
        X_train, X_test, feature_names_out, transformer = _preprocess_features(
            X_train,
            X_test,
            feature_names,
            model_type,
            transform,
            y[train_idx] if y_train is None else y_train,
            pca_variance,
            pls_components,
            sft_k,
        )

        train_scores, test_scores, model = _train_model_local(
            model_type,
            X_train.values.astype(float),
            y[train_idx],
            X_test.values.astype(float),
            42,
            logistic_penalty=logistic_penalty,
            logistic_c=logistic_c,
            logistic_l1_ratio=logistic_l1_ratio,
        )
        if rule_mask is not None:
            train_scores = _apply_rule_override(train_scores, rule_mask[train_idx])
            test_scores = _apply_rule_override(test_scores, rule_mask[test_idx])

        oof_scores.append(test_scores)
        oof_labels.append(y[test_idx])
        fold_artifacts.append(
            {
                "model": model,
                "X_train": X_train.values.astype(float),
                "X_val": X_test.values.astype(float),
                "y_val": y[test_idx],
                "rule_mask_val": None if rule_mask is None else rule_mask[test_idx],
                "feature_names": feature_names_out,
                "transformer": transformer,
            }
        )

    all_scores = np.concatenate(oof_scores)
    all_labels = np.concatenate(oof_labels)
    oof_threshold, _ = _select_threshold(all_labels, all_scores)

    for test_idx, scores in zip([s[1] for s in splits], oof_scores):
        metrics = _metrics_from_scores(y[test_idx], scores, oof_threshold)
        metrics["accuracy"] = float(np.mean((scores >= metrics["threshold"]).astype(int) == y[test_idx]))
        fold_metrics.append(metrics)

    keys = ["roc_auc", "pr_auc", "precision", "recall", "f0.5", "accuracy"]
    means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    stds = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return means, stds, float(oof_threshold), fold_artifacts


def _full_train_metrics(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str,
    threshold: float,
    rule_mask: np.ndarray | None = None,
    transform: str = "BASE",
    y_train: np.ndarray | None = None,
    pca_variance: float = PCA_VARIANCE_DEFAULT,
    pls_components: int = PLS_COMPONENTS_DEFAULT,
    sft_k: int = SFT_K_DEFAULT,
    logistic_penalty: str = "elasticnet",
    logistic_c: float = 1.0,
    logistic_l1_ratio: float | None = 0.5,
    return_model: bool = False,
) -> dict[str, float] | tuple[dict[str, float], Any, Any | None, list[str]]:
    X_train, _, feature_names_out, transformer = _preprocess_features(
        df,
        df.copy(),
        feature_names,
        model_type,
        transform,
        y if y_train is None else y_train,
        pca_variance,
        pls_components,
        sft_k,
    )
    scores, _, model = _train_model_local(
        model_type,
        X_train.values.astype(float),
        y,
        X_train.values.astype(float),
        42,
        logistic_penalty=logistic_penalty,
        logistic_c=logistic_c,
        logistic_l1_ratio=logistic_l1_ratio,
    )
    if rule_mask is not None:
        scores = _apply_rule_override(scores, rule_mask)
    metrics = _metrics_from_scores(y, scores, threshold)
    metrics["accuracy"] = float(np.mean((scores >= metrics["threshold"]).astype(int) == y))
    if return_model:
        return metrics, model, transformer, feature_names_out
    return metrics


def _slugify(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("|", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


def _resolve_model_dir(transform_upper: str, sweep_mode: bool, output_suffix: str) -> Path:
    base_dir = OUTPUT_DIR
    if sweep_mode:
        if transform_upper == "PCA":
            base_dir = OUTPUT_DIR / "PCA_Sweep_reports"
        elif transform_upper == "PLS":
            base_dir = OUTPUT_DIR / "PLS_Sweep_reports"
        elif transform_upper == "SFT":
            base_dir = OUTPUT_DIR / "SFT_Sweep_reports"
    return base_dir / f"model_testing_models{output_suffix}"


def _resolve_report_dir(
    transform_upper: str,
    sweep_mode: bool,
    base_sweep_enabled: bool,
) -> Path:
    report_dir = OUTPUT_DIR
    if transform_upper == "BASE" and base_sweep_enabled:
        report_dir = OUTPUT_DIR / "base_sweep"
    elif transform_upper == "PCA" and sweep_mode:
        report_dir = OUTPUT_DIR / "PCA_Sweep_reports"
    elif transform_upper == "PLS" and sweep_mode:
        report_dir = OUTPUT_DIR / "PLS_Sweep_reports"
    elif transform_upper == "SFT" and sweep_mode:
        report_dir = OUTPUT_DIR / "SFT_Sweep_reports"
    _ensure_dir(report_dir)
    return report_dir


def _resolve_collinearity_dir(report_dir: Path, output_suffix: str) -> Path:
    return report_dir / f"collinearity_reports{output_suffix}"


def _build_collinearity_section(
    summary_rows: list[dict[str, Any]],
    corr_threshold: float,
) -> list[str]:
    if not summary_rows:
        return []
    def _fmt(val: float | None) -> str:
        if val is None:
            return "--"
        if not np.isfinite(val):
            return "inf"
        return f"{val:.3f}"

    # Pick a single top-risk row per combo to keep the table compact.
    picked: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        combo = row.get("reasoning_combo", "HQ")
        current = picked.get(combo)
        score = row.get("max_vif") if not row.get("vif_skipped") else row.get("max_abs_corr")
        if score is None:
            score = 0.0
        if current is None:
            picked[combo] = row
            continue
        cur_score = current.get("max_vif") if not current.get("vif_skipped") else current.get("max_abs_corr")
        if cur_score is None:
            cur_score = 0.0
        if float(score) > float(cur_score):
            picked[combo] = row

    lines = [
        "",
        "## Collinearity Diagnostics (Logistic only)",
        "_Model-input stats use transformed features; raw stats use pre-transform standardized features._",
        f"_Correlation threshold: {corr_threshold}_",
        "",
        "| Combo | Family | Transform | Sweep | max_vif | max_abs_corr | cond_num | avg_sign_flip | raw_max_vif | raw_max_abs_corr | raw_cond_num |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for combo, row in sorted(picked.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    combo,
                    str(row.get("family", "")),
                    str(row.get("transform", "")),
                    str(row.get("sweep_param", "")),
                    _fmt(row.get("max_vif")),
                    _fmt(row.get("max_abs_corr")),
                    _fmt(row.get("cond_number")),
                    _fmt(row.get("avg_sign_flip_rate")),
                    _fmt(row.get("raw_max_vif")),
                    _fmt(row.get("raw_max_abs_corr")),
                    _fmt(row.get("raw_cond_number")),
                ]
            )
            + " |"
        )
    return lines


def _build_collinearity_full_table(
    summary_rows: list[dict[str, Any]],
    corr_threshold: float,
) -> list[str]:
    if not summary_rows:
        return []
    def _fmt(val: float | None) -> str:
        if val is None:
            return "--"
        if not np.isfinite(val):
            return "inf"
        return f"{val:.3f}"

    lines = [
        "# Collinearity Diagnostics (Logistic only)",
        f"_Correlation threshold: {corr_threshold}_",
        "",
        "| Family | Combo | Transform | Sweep | max_vif | max_abs_corr | cond_num | avg_sign_flip | raw_max_vif | raw_max_abs_corr | raw_cond_num | vif_skipped | raw_vif_skipped |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(
        summary_rows,
        key=lambda r: (str(r.get("family", "")), str(r.get("reasoning_combo", "")), str(r.get("sweep_param", ""))),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("family", "")),
                    str(row.get("reasoning_combo", "")),
                    str(row.get("transform", "")),
                    str(row.get("sweep_param", "")),
                    _fmt(row.get("max_vif")),
                    _fmt(row.get("max_abs_corr")),
                    _fmt(row.get("cond_number")),
                    _fmt(row.get("avg_sign_flip_rate")),
                    _fmt(row.get("raw_max_vif")),
                    _fmt(row.get("raw_max_abs_corr")),
                    _fmt(row.get("raw_cond_number")),
                    str(bool(row.get("vif_skipped"))),
                    str(bool(row.get("raw_vif_skipped"))),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _build_elasticnet_sweep_tables(
    sweep_df: pd.DataFrame,
    family_key: str,
    transform_upper: str,
    sweep_param: str,
    allowed_combos: list[str],
    header_title: str,
) -> list[str]:
    subset = sweep_df[
        (sweep_df["family"] == family_key)
        & (sweep_df["model_type"] == "elasticnet")
        & (sweep_df["transform"] == transform_upper)
        & (sweep_df["sweep_param"] == sweep_param)
    ]
    if subset.empty:
        return []

    c_vals = sorted({float(v) for v in subset["logistic_C"].dropna().unique()})
    l1_vals = sorted({float(v) for v in subset["logistic_l1_ratio"].dropna().unique()})
    if not c_vals or not l1_vals:
        return []

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f}+/-{std:.3f}"

    def _md_separator(cols: int) -> str:
        return "|" + "|".join(["---"] + ["---:" for _ in range(cols - 1)]) + "|"

    lines = ["", header_title]
    if sweep_param:
        lines.append(f"_Sweep param: {sweep_param}_")
    for combo in allowed_combos:
        combo_df = subset[subset["reasoning_combo"] == combo]
        if combo_df.empty:
            continue
        lines += ["", f"### {combo}"]
        header = ["l1_ratio \\ C"] + [str(c).replace(".", "p") for c in c_vals]
        lines.append("| " + " | ".join(header) + " |")
        lines.append(_md_separator(len(header)))
        for l1 in l1_vals:
            row = [str(l1).replace(".", "p")]
            for c in c_vals:
                match = combo_df[(combo_df["logistic_C"] == c) & (combo_df["logistic_l1_ratio"] == l1)]
                if match.empty:
                    row.append("--")
                else:
                    r = match.iloc[0]
                    row.append(_fmt(float(r["f0.5_mean"]), float(r["f0.5_std"])))
            lines.append("| " + " | ".join(row) + " |")
    return lines


def _build_logistic_c_sweep_tables(
    sweep_df: pd.DataFrame,
    family_key: str,
    transform_upper: str,
    sweep_param: str,
    allowed_combos: list[str],
    header_title: str,
) -> list[str]:
    subset = sweep_df[
        (sweep_df["family"] == family_key)
        & (sweep_df["model_type"] == "logistic")
        & (sweep_df["transform"] == transform_upper)
        & (sweep_df["sweep_param"] == sweep_param)
    ]
    if subset.empty:
        return []

    c_vals = sorted({float(v) for v in subset["logistic_C"].dropna().unique()})
    if not c_vals:
        return []

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f}+/-{std:.3f}"

    def _md_separator(cols: int) -> str:
        return "|" + "|".join(["---"] + ["---:" for _ in range(cols - 1)]) + "|"

    lines = ["", header_title]
    if sweep_param:
        lines.append(f"_Sweep param: {sweep_param}_")
    header = ["Combo"] + [str(c).replace(".", "p") for c in c_vals]
    lines.append("| " + " | ".join(header) + " |")
    lines.append(_md_separator(len(header)))
    for combo in allowed_combos:
        combo_df = subset[subset["reasoning_combo"] == combo]
        row = [combo]
        for c in c_vals:
            match = combo_df[(combo_df["logistic_C"] == c)]
            if match.empty:
                row.append("--")
            else:
                r = match.iloc[0]
                row.append(_fmt(float(r["f0.5_mean"]), float(r["f0.5_std"])))
        lines.append("| " + " | ".join(row) + " |")
    return lines

def _save_model_bundle(
    out_dir: Path,
    run: ModelRun,
    combo: str,
    transform: str,
    sweep_param: str | None,
    model: Any,
    transformer: Any | None,
    feature_names_in: list[str],
    feature_names_out: list[str],
    threshold_oof: float,
    full_metrics: dict[str, float],
    logistic_penalty: str,
    logistic_c: float | None,
    logistic_l1_ratio: float | None,
) -> None:
    subdir = (
        out_dir
        / _slugify(run.family)
        / _slugify(combo or "HQ")
        / _slugify(run.model_type)
        / _slugify(transform)
        / _slugify(str(sweep_param or "base"))
    )
    _ensure_dir(subdir)
    model_path = subdir / "model.joblib"
    joblib.dump(model, model_path)
    transformer_path = None
    if transformer is not None:
        transformer_path = subdir / "transform.joblib"
        joblib.dump(transformer, transformer_path)
    meta = {
        "family": run.family,
        "model_name": run.name,
        "model_type": run.model_type,
        "reasoning_combo": combo,
        "transform": transform,
        "sweep_param": sweep_param,
        "feature_names_in": feature_names_in,
        "feature_names_out": feature_names_out,
        "threshold_oof": threshold_oof,
        "full_metrics": full_metrics,
        "logistic_penalty": logistic_penalty if run.model_type == "logistic" else None,
        "logistic_C": logistic_c if run.model_type == "logistic" else None,
        "logistic_l1_ratio": logistic_l1_ratio if run.model_type == "logistic" else None,
        "model_path": str(model_path),
        "transformer_path": str(transformer_path) if transformer_path else None,
    }
    meta_path = subdir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _compute_f05(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    preds = (scores >= threshold).astype(int)
    return float(fbeta_score(y_true, preds, beta=0.5, zero_division=0))


def _zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    means = num.mean(axis=0)
    stds = num.std(axis=0).replace(0, 1.0)
    return (num - means) / stds


def _corr_stats(
    X: np.ndarray,
    feature_names: list[str],
    top_k: int,
    threshold: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    n_features = X.shape[1]
    if n_features < 2:
        stats = {"max_abs_corr": 0.0, "mean_abs_corr": 0.0, "count_ge_threshold": 0.0}
        return stats, pd.DataFrame(columns=["feature_a", "feature_b", "corr"])
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices(n_features, k=1)
    vals = np.abs(corr[iu])
    max_abs = float(vals.max()) if vals.size else 0.0
    mean_abs = float(vals.mean()) if vals.size else 0.0
    count_ge = float(np.sum(vals >= threshold)) if vals.size else 0.0
    stats = {"max_abs_corr": max_abs, "mean_abs_corr": mean_abs, "count_ge_threshold": count_ge}

    if vals.size:
        top_k = max(1, int(top_k))
        top_idx = np.argsort(vals)[::-1][:top_k]
        rows = []
        for idx in top_idx:
            i = int(iu[0][idx])
            j = int(iu[1][idx])
            rows.append(
                {
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "corr": float(corr[i, j]),
                }
            )
        pairs_df = pd.DataFrame(rows)
    else:
        pairs_df = pd.DataFrame(columns=["feature_a", "feature_b", "corr"])
    return stats, pairs_df


def _condition_number(X: np.ndarray) -> float:
    n_features = X.shape[1]
    if n_features < 2:
        return 0.0
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return float(np.linalg.cond(corr))
    except Exception:
        return float("inf")


def _vif_stats(
    X: np.ndarray,
    feature_names: list[str],
    max_features: int,
) -> tuple[pd.DataFrame, bool]:
    n_features = X.shape[1]
    if n_features < 2:
        return pd.DataFrame(columns=["feature", "vif"]), False
    if n_features > max_features:
        return pd.DataFrame(columns=["feature", "vif"]), True

    vifs: list[dict[str, float]] = []
    for j in range(n_features):
        y = X[:, j]
        X_other = np.delete(X, j, axis=1)
        if X_other.shape[1] == 0:
            vifs.append({"feature": feature_names[j], "vif": float("inf")})
            continue
        try:
            coef, _, _, _ = np.linalg.lstsq(X_other, y, rcond=None)
            y_hat = X_other @ coef
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            if ss_tot <= 0:
                r2 = 1.0
            else:
                r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
            vif = float("inf") if r2 >= 0.999999 else float(1.0 / (1.0 - r2))
        except Exception:
            vif = float("inf")
        vifs.append({"feature": feature_names[j], "vif": vif})
    return pd.DataFrame(vifs).sort_values("vif", ascending=False), False


def _coef_stability(fold_artifacts: list[dict[str, Any]]) -> tuple[pd.DataFrame, float]:
    coefs = [fa["model"].coef_[0] for fa in fold_artifacts]
    coef_mat = np.vstack(coefs)
    mean = coef_mat.mean(axis=0)
    std = coef_mat.std(axis=0)
    signs = np.sign(coef_mat)
    sign_ref = np.sign(mean)
    flip_rates = []
    for j in range(coef_mat.shape[1]):
        ref = sign_ref[j]
        if ref == 0:
            flips = np.mean(signs[:, j] != 0)
        else:
            flips = np.mean(signs[:, j] != ref)
        flip_rates.append(float(flips))
    coef_cv = np.where(np.abs(mean) > 0, std / np.abs(mean), np.inf)
    features = fold_artifacts[0].get("feature_names", [])
    df = pd.DataFrame(
        {
            "feature": features,
            "coef_mean": mean,
            "coef_std": std,
            "sign_flip_rate": flip_rates,
            "coef_cv": coef_cv,
        }
    ).sort_values("coef_std", ascending=False)
    avg_flip = float(np.mean(flip_rates)) if flip_rates else 0.0
    return df, avg_flip

def _compute_perm_importance(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float,
    feature_names: list[str],
    rule_mask_val: np.ndarray | None,
    repeats: int = PERM_REPEATS,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    baseline_scores = model.predict_proba(X_val)[:, 1]
    if rule_mask_val is not None:
        baseline_scores = _apply_rule_override(baseline_scores, rule_mask_val)
    baseline_f = _compute_f05(y_val, baseline_scores, threshold)
    importances = np.zeros(X_val.shape[1], dtype=float)
    for j in range(X_val.shape[1]):
        drops: list[float] = []
        for _ in range(repeats):
            X_perm = X_val.copy()
            rng.shuffle(X_perm[:, j])
            scores = model.predict_proba(X_perm)[:, 1]
            if rule_mask_val is not None:
                scores = _apply_rule_override(scores, rule_mask_val)
            drops.append(baseline_f - _compute_f05(y_val, scores, threshold))
        importances[j] = float(np.mean(drops))
    return importances


def _compute_shap_importance(
    model: Any,
    model_type: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    try:
        import shap  # type: ignore
    except Exception as exc:
        raise RuntimeError("shap is required for SHAP importance. Install with: pip install shap") from exc
    rng = np.random.default_rng(seed)
    if X_val.shape[0] > SHAP_VAL_SAMPLE:
        idx = rng.choice(X_val.shape[0], SHAP_VAL_SAMPLE, replace=False)
        X_val_sample = X_val[idx]
    else:
        X_val_sample = X_val
    if X_train.shape[0] > SHAP_BG_SAMPLE:
        bg_idx = rng.choice(X_train.shape[0], SHAP_BG_SAMPLE, replace=False)
        X_bg = X_train[bg_idx]
    else:
        X_bg = X_train

    if model_type == "logistic":
        explainer = shap.LinearExplainer(model, X_bg)
        shap_vals = explainer.shap_values(X_val_sample)
    elif model_type in ("xgb1", "xgb3"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_val_sample)
    elif model_type in ("mlp32", "mlp4", "mlp2"):
        explainer = shap.KernelExplainer(model.predict_proba, X_bg)
        shap_vals = explainer.shap_values(X_val_sample)
    else:
        raise ValueError(f"Unknown model_type for SHAP: {model_type}")

    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    else:
        shap_arr = shap_vals

    arr = np.array(shap_arr)
    n_features = X_val.shape[1]
    if arr.ndim == 3:
        # Common cases: (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if arr.shape[-1] == 2 and arr.shape[1] == n_features:
            arr = arr[:, :, 1]
        elif arr.shape[0] == 2 and arr.shape[2] == n_features:
            arr = arr[1]
        elif arr.shape[0] == 2 and arr.shape[1] == n_features:
            arr = arr[1]
        else:
            raise RuntimeError(f"Unexpected SHAP shape: {arr.shape} (n_features={n_features})")
    if arr.ndim == 2:
        if arr.shape[1] == n_features:
            mean_abs = np.mean(np.abs(arr), axis=0)
        elif arr.shape[0] == n_features:
            mean_abs = np.mean(np.abs(arr), axis=1)
        else:
            raise RuntimeError(f"Unexpected SHAP 2D shape: {arr.shape} (n_features={n_features})")
    elif arr.ndim == 1:
        if arr.shape[0] != n_features:
            raise RuntimeError(f"Unexpected SHAP 1D length: {arr.shape[0]} (n_features={n_features})")
        mean_abs = np.abs(arr)
    else:
        raise RuntimeError(f"Unexpected SHAP ndim: {arr.ndim} (shape={arr.shape})")
    return mean_abs.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model testing pipeline (Part 2 only).")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--engineered_family_id", type=str, default=None)
    parser.add_argument("--engineered_set_id", type=str, default=ENGINEERED_SET_ID_DEFAULT)
    parser.add_argument("--exp_scope", type=str, default="full_exps")
    parser.add_argument("--run_interpretability", type=str, default="true")
    parser.add_argument("--use_pca", type=str, default="false")
    parser.add_argument("--model_complexity", type=str, default="complex")
    parser.add_argument("--pca_var_sweep", type=str, default="false")
    parser.add_argument("--transform_sweep", type=str, default="false")
    parser.add_argument("--feature_transforms", type=str, default="Base")
    parser.add_argument("--pls_components", type=int, default=PLS_COMPONENTS_DEFAULT)
    parser.add_argument("--sft_k", type=int, default=SFT_K_DEFAULT)
    parser.add_argument("--logistic_penalty", type=str, default="elasticnet")
    parser.add_argument("--logistic_c_grid", type=str, default="0.1,1,10")
    parser.add_argument("--logistic_l1_ratio_grid", type=str, default="0.2,0.8")
    parser.add_argument("--logistic_tuning_mode", type=str, default="per_model")
    parser.add_argument("--logistic_c", type=float, default=0.3)
    parser.add_argument("--logistic_l1_ratio", type=float, default=0.5)
    parser.add_argument("--save_models", type=str, default="true")
    parser.add_argument("--collinearity_report", type=str, default="true")
    parser.add_argument("--collinearity_corr_topk", type=int, default=20)
    parser.add_argument("--collinearity_corr_threshold", type=float, default=0.9)
    parser.add_argument("--vif_max_features", type=int, default=200)
    args = parser.parse_args()

    _ensure_dir(OUTPUT_DIR)
    exp_scope = str(args.exp_scope).strip().lower()
    if exp_scope not in {"reduced_exps", "no_llm_exps", "full_exps"}:
        raise RuntimeError("--exp_scope must be one of: reduced_exps, no_llm_exps, full_exps")
    include_llm_engineered = exp_scope == "full_exps"
    reduced_exps = exp_scope == "reduced_exps"
    run_interpretability_default = str(args.run_interpretability).strip().lower() not in {"0", "false", "no"}
    use_pca = str(args.use_pca).strip().lower() in {"1", "true", "yes"}
    model_complexity = str(args.model_complexity).strip().lower()
    pca_var_sweep = str(args.pca_var_sweep).strip().lower() in {"1", "true", "yes"}
    transform_sweep = str(args.transform_sweep).strip().lower() in {"1", "true", "yes"}
    save_models = str(args.save_models).strip().lower() in {"1", "true", "yes"}
    collinearity_report = str(args.collinearity_report).strip().lower() in {"1", "true", "yes"}
    collinearity_corr_topk = int(args.collinearity_corr_topk)
    collinearity_corr_threshold = float(args.collinearity_corr_threshold)
    vif_max_features = int(args.vif_max_features)
    pls_components = int(args.pls_components)
    sft_k = int(args.sft_k)
    logistic_penalty = str(args.logistic_penalty).strip().lower()
    logistic_tuning_mode = str(args.logistic_tuning_mode).strip().lower()
    logistic_c = float(args.logistic_c)
    logistic_l1_ratio = float(args.logistic_l1_ratio)
    try:
        logistic_c_grid = [float(v) for v in str(args.logistic_c_grid).split(",") if v.strip()]
    except ValueError as exc:
        raise RuntimeError("--logistic_c_grid must be a comma-separated list of floats") from exc
    try:
        logistic_l1_ratio_grid = [float(v) for v in str(args.logistic_l1_ratio_grid).split(",") if v.strip()]
    except ValueError as exc:
        raise RuntimeError("--logistic_l1_ratio_grid must be a comma-separated list of floats") from exc

    if logistic_penalty not in {"l1", "l2", "elasticnet"}:
        raise RuntimeError("--logistic_penalty must be one of: l1, l2, elasticnet")
    if logistic_tuning_mode not in {"per_model", "fixed"}:
        raise RuntimeError("--logistic_tuning_mode must be one of: per_model, fixed")

    if model_complexity not in {"simple", "complex"}:
        raise RuntimeError("--model_complexity must be 'simple' or 'complex'")

    transforms_raw = [t.strip().upper() for t in str(args.feature_transforms).split(",") if t.strip()]
    if not transforms_raw:
        transforms_raw = ["BASE"]
    for t in transforms_raw:
        if t not in {"BASE", "PCA", "PLS", "SFT"}:
            raise RuntimeError(f"Unknown feature transform: {t}")
    if use_pca and "PCA" not in transforms_raw:
        transforms_raw.append("PCA")
    if pca_var_sweep and "PCA" not in transforms_raw:
        transforms_raw.append("PCA")

    # Load public data (full 4,500 founders)
    input_csv = BASE_DIR.parent / "VCBench-Starter-Kit" / "vcbench_final_public.csv"
    records, labels = load_vcbench(input_csv, "success", 0, 42)
    full_founder_ids = [r.get("founder_uuid") for r in records]

    # Reasoning features (full_current)
    reasoning_full = pd.read_parquet(
        BASE_DIR / "features_storage" / "llm_reasoning" / "full_current" / "llm_reasoning_full.parquet"
    )
    full_reasoning_df = _reindex_reasoning(reasoning_full, full_founder_ids)
    for col in full_reasoning_df.columns:
        if col not in ("founder_uuid", "success", "row_index"):
            full_reasoning_df[col] = pd.to_numeric(full_reasoning_df[col], errors="ignore")

    if reduced_exps:
        allowed_combos = ["HQ", "A", "B", "C", "D", "E", "F", "A+B+C+D+E+F"]
    else:
        allowed_combos = ["HQ", "A", "B", "C", "D", "E", "F", "A+B+C+D+E+F"]
    combos = {"HQ": []}
    combos.update(_build_reasoning_combos_numeric(full_reasoning_df, ["A", "B", "C", "D", "E", "F"]))
    combos = {k: v for k, v in combos.items() if k in allowed_combos}
    missing = [c for c in allowed_combos if c not in combos]
    if missing:
        raise RuntimeError(f"Missing required Full Mirror combos: {missing}. Check full_current reasoning columns.")

    # HQ features + rule mask
    hq_script = BASE_DIR.parent / "High_Quality_human_features" / "features" / "extract_structured.py"
    hq_df_full = _build_high_quality_features(records, hq_script)
    hq_full_no_gap = hq_df_full[HQ_FEATURES_BASE].copy()
    rule_mask_full = hq_df_full["exit_count"].fillna(0.0).astype(float).values > 0

    engineered_df: pd.DataFrame | None = None
    engineered_set_id = args.engineered_set_id
    if include_llm_engineered:
        archives_dir = BASE_DIR / "features_storage" / "llm_engineered" / "archives"
        engineered_family_id = args.engineered_family_id or _pick_latest_family_id(archives_dir)
        rules = _load_engineered_rules(engineered_family_id, engineered_set_id)
        engineered_df = _evaluate_engineered_rules(records, rules)

    # CV folds (full dataset)
    full_folds_path = BASE_DIR / "features_storage" / "cv_folds" / f"folds_k{args.cv_folds}_seed42_full.json"
    full_splits, _, _ = load_or_create_folds(
        founder_ids=full_founder_ids,
        labels=labels,
        cv_folds=args.cv_folds,
        random_state=42,
        folds_path=full_folds_path,
        dataset_label=f"{input_csv}_full",
        use_fixed=True,
    )

    if model_complexity == "simple":
        model_types = ["logistic", "elasticnet"]
    else:
        model_types = ["logistic", "elasticnet", "mlp32", "mlp4", "mlp2"]

    model_runs: list[ModelRun] = []
    for combo, combo_cols in combos.items():
        for model_type in model_types:
            model_runs.append(
                ModelRun(
                    name=f"Mirror HQ {combo} ({model_type})",
                    model_type=model_type,
                    feature_names=HQ_FEATURES_BASE + combo_cols,
                    rule_mask=rule_mask_full,
                    family="hq_mirror",
                    reasoning_combo=combo,
                )
            )
            if include_llm_engineered and engineered_df is not None:
                model_runs.append(
                    ModelRun(
                        name=f"Engineered {engineered_set_id} {combo} ({model_type})",
                        model_type=model_type,
                        feature_names=list(engineered_df.columns) + combo_cols,
                        rule_mask=None,
                        family=f"engineered_{engineered_set_id}",
                        reasoning_combo=combo,
                    )
                )

    def _dedupe(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in seq:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    transforms = _dedupe(transforms_raw)
    print(f"Feature transforms: {', '.join(transforms)}")

    def _build_suffix(
        transform: str,
        pca_variance: float | None,
        interp_on: bool,
        sweep_token: str | None = None,
    ) -> str:
        parts: list[str] = []
        parts.append(f"exp_{exp_scope}")
        parts.append("simple_models" if model_complexity == "simple" else "complex_models")
        parts.append("interp_on" if interp_on else "interp_off")
        transform_upper = transform.upper()
        if transform_upper == "PCA":
            if sweep_token:
                pca_token = sweep_token
            else:
                pca_token = "on" if pca_variance is None else str(pca_variance).replace(".", "p")
            parts.append(f"pca_{pca_token}")
        elif transform_upper == "PLS":
            parts.append("pls")
        elif transform_upper == "SFT":
            parts.append("sft")
        else:
            parts.append("base")
        return "" if not parts else "_" + "_".join(parts)

    def _run_for_transform(transform: str) -> None:
        transform_upper = transform.upper()
        interp_on = run_interpretability_default and transform_upper not in {"PCA", "PLS"}
        if transform_upper == "PCA":
            if pca_var_sweep or transform_sweep:
                variance_list = PCA_SWEEP_VALUES
            else:
                variance_list = [PCA_VARIANCE_DEFAULT]
        elif transform_upper == "PLS":
            variance_list = PLS_SWEEP_VALUES if transform_sweep else [pls_components]
        elif transform_upper == "SFT":
            variance_list = SFT_SWEEP_VALUES if transform_sweep else [sft_k]
        else:
            variance_list = [None]

        sweep_mode = (transform_upper == "PCA" and (pca_var_sweep or transform_sweep)) or (
            transform_upper in {"PLS", "SFT"} and transform_sweep
        )

        sweep_token = "sweep" if (sweep_mode and transform_upper == "PCA") else None
        base_pca = None if sweep_mode else (PCA_VARIANCE_DEFAULT if transform_upper == "PCA" else None)
        output_suffix = _build_suffix(transform_upper, base_pca, interp_on, sweep_token=sweep_token)
        report_dir = _resolve_report_dir(
            transform_upper,
            sweep_mode,
            base_sweep_enabled=(transform_upper == "BASE" and logistic_tuning_mode == "per_model"),
        )
        interp_dir = OUTPUT_DIR / f"interpretability{output_suffix}"
        if interp_on:
            _ensure_dir(interp_dir)
        model_dir = _resolve_model_dir(transform_upper, sweep_mode, output_suffix) if save_models else None
        col_dir = _resolve_collinearity_dir(report_dir, output_suffix) if collinearity_report else None

        results_rows: list[dict[str, Any]] = []
        sweep_rows: list[dict[str, Any]] = []
        perm_results: dict[tuple[str, str], pd.DataFrame] = {}
        shap_results: dict[tuple[str, str], pd.DataFrame] = {}
        col_summary_rows: list[dict[str, Any]] = []

        for sweep_value in variance_list:
            pca_variance_value = PCA_VARIANCE_DEFAULT
            cur_pls_components = pls_components
            cur_sft_k = sft_k
            if transform_upper == "PCA":
                pca_variance_value = float(sweep_value)
            elif transform_upper == "PLS":
                cur_pls_components = int(sweep_value)
            elif transform_upper == "SFT":
                cur_sft_k = int(sweep_value)
            for run in model_runs:
                combo = run.reasoning_combo or "HQ"
                combo_cols = combos.get(combo, [])
                if run.family == "hq_mirror":
                    base_df = hq_full_no_gap
                else:
                    if engineered_df is None:
                        raise RuntimeError("Engineered features requested but include_llm_engineered is false.")
                    base_df = engineered_df
                train_df = (
                    pd.concat([base_df, full_reasoning_df[combo_cols]], axis=1)
                    if combo_cols
                    else base_df.copy()
                )

                if run.model_type in {"elasticnet", "logistic"} and logistic_tuning_mode == "per_model":
                    best_mean = -1.0
                    best_std = 1e9
                    best_c = logistic_c
                    best_l1 = logistic_l1_ratio
                    best_metrics = None
                    best_threshold = None
                    best_folds = None
                    c_grid = logistic_c_grid
                    if run.model_type == "elasticnet":
                        l1_grid = logistic_l1_ratio_grid
                        penalty = "elasticnet"
                    else:
                        l1_grid = [None]
                        penalty = "l2"
                    for c_val in c_grid:
                        for l1_val in l1_grid:
                            means, stds, oof_threshold, fold_artifacts = _oof_cv_metrics(
                                train_df,
                                labels,
                                run.feature_names,
                                run.model_type,
                                full_splits,
                                rule_mask=run.rule_mask,
                                transform=transform_upper,
                                y_train=None,
                                pca_variance=pca_variance_value,
                                pls_components=cur_pls_components,
                                sft_k=cur_sft_k,
                                logistic_penalty=penalty,
                                logistic_c=c_val,
                                logistic_l1_ratio=l1_val,
                            )
                            sweep_rows.append(
                                {
                                    "family": run.family,
                                    "model_name": run.name,
                                    "model_type": run.model_type,
                                    "reasoning_combo": combo,
                                    "transform": transform_upper,
                                    "sweep_param": str(sweep_value),
                                    "logistic_penalty": penalty,
                                    "logistic_C": c_val,
                                    "logistic_l1_ratio": l1_val,
                                    "f0.5_mean": means["f0.5"],
                                    "f0.5_std": stds["f0.5"],
                                    "roc_auc_mean": means["roc_auc"],
                                    "roc_auc_std": stds["roc_auc"],
                                    "pr_auc_mean": means["pr_auc"],
                                    "pr_auc_std": stds["pr_auc"],
                                    "precision_mean": means["precision"],
                                    "precision_std": stds["precision"],
                                    "recall_mean": means["recall"],
                                    "recall_std": stds["recall"],
                                    "acc_mean": means["accuracy"],
                                    "acc_std": stds["accuracy"],
                                    "threshold_oof": oof_threshold,
                                }
                            )
                            mean_f = means["f0.5"]
                            std_f = stds["f0.5"]
                            if (mean_f > best_mean + 1e-12) or (
                                abs(mean_f - best_mean) <= 1e-12 and std_f < best_std
                            ):
                                best_mean = mean_f
                                best_std = std_f
                                best_c = c_val
                                best_l1 = l1_val
                                best_metrics = (means, stds)
                                best_threshold = oof_threshold
                                best_folds = fold_artifacts
                    if best_metrics is None:
                        raise RuntimeError("Logistic sweep failed to produce metrics.")
                    means, stds = best_metrics
                    oof_threshold = float(best_threshold)
                    fold_artifacts = best_folds
                    chosen_c = best_c
                    chosen_l1 = best_l1
                else:
                    penalty = "elasticnet" if run.model_type == "elasticnet" else "l2"
                    means, stds, oof_threshold, fold_artifacts = _oof_cv_metrics(
                        train_df,
                        labels,
                        run.feature_names,
                        run.model_type,
                        full_splits,
                        rule_mask=run.rule_mask,
                        transform=transform_upper,
                        y_train=None,
                        pca_variance=pca_variance_value,
                        pls_components=cur_pls_components,
                        sft_k=cur_sft_k,
                        logistic_penalty=penalty,
                        logistic_c=logistic_c,
                        logistic_l1_ratio=logistic_l1_ratio,
                    )
                    chosen_c = logistic_c
                    chosen_l1 = logistic_l1_ratio

                if interp_on and run.family == "hq_mirror" and combo in INTERP_COMBOS:
                    perm_fold_vals: list[np.ndarray] = []
                    shap_fold_vals: list[np.ndarray] = []
                    for fold in fold_artifacts:
                        perm_vals = _compute_perm_importance(
                            model=fold["model"],
                            X_val=fold["X_val"],
                            y_val=fold["y_val"],
                            threshold=oof_threshold,
                            feature_names=fold.get("feature_names", run.feature_names),
                            rule_mask_val=fold["rule_mask_val"],
                        )
                        perm_fold_vals.append(perm_vals)
                        shap_vals = _compute_shap_importance(
                            model=fold["model"],
                            model_type=run.model_type,
                            X_train=fold["X_train"],
                            X_val=fold["X_val"],
                        )
                        shap_fold_vals.append(shap_vals)

                    perm_arr = np.vstack(perm_fold_vals)
                    shap_arr = np.vstack(shap_fold_vals)
                    perm_df = pd.DataFrame(
                        {
                            "feature": fold_artifacts[0].get("feature_names", run.feature_names),
                            "mean": perm_arr.mean(axis=0),
                            "std": perm_arr.std(axis=0),
                        }
                    ).sort_values("mean", ascending=False)
                    shap_df = pd.DataFrame(
                        {
                            "feature": fold_artifacts[0].get("feature_names", run.feature_names),
                            "mean": shap_arr.mean(axis=0),
                            "std": shap_arr.std(axis=0),
                        }
                    ).sort_values("mean", ascending=False)
                    perm_results[(combo, run.model_type)] = perm_df
                    shap_results[(combo, run.model_type)] = shap_df
                    perm_path = interp_dir / f"perm_hq_{combo}_{run.model_type}.csv"
                    shap_path = interp_dir / f"shap_hq_{combo}_{run.model_type}.csv"
                    perm_df.to_csv(perm_path, index=False)
                    shap_df.to_csv(shap_path, index=False)

                if collinearity_report and run.model_type == "logistic":
                    X_model_df, _, model_feature_names, _ = _preprocess_features(
                        train_df,
                        train_df.copy(),
                        run.feature_names,
                        run.model_type,
                        transform_upper,
                        y_train=labels,
                        pca_variance=pca_variance_value,
                        pls_components=cur_pls_components,
                        sft_k=cur_sft_k,
                    )
                    X_model_std = _zscore_df(X_model_df)
                    X_model = X_model_std.values.astype(float)
                    cond_num = _condition_number(X_model)
                    corr_stats, corr_pairs = _corr_stats(
                        X_model,
                        model_feature_names,
                        collinearity_corr_topk,
                        collinearity_corr_threshold,
                    )
                    vif_df, vif_skipped = _vif_stats(
                        X_model,
                        model_feature_names,
                        vif_max_features,
                    )
                    max_vif = float(vif_df["vif"].max()) if not vif_df.empty else None

                    coef_df, avg_flip = _coef_stability(fold_artifacts)

                    raw_df_std = _zscore_df(train_df[run.feature_names].copy())
                    X_raw = raw_df_std.values.astype(float)
                    raw_cond_num = _condition_number(X_raw)
                    raw_corr_stats, raw_corr_pairs = _corr_stats(
                        X_raw,
                        run.feature_names,
                        collinearity_corr_topk,
                        collinearity_corr_threshold,
                    )
                    raw_vif_df, raw_vif_skipped = _vif_stats(
                        X_raw,
                        run.feature_names,
                        vif_max_features,
                    )
                    raw_max_vif = float(raw_vif_df["vif"].max()) if not raw_vif_df.empty else None

                    if col_dir is not None:
                        subdir = (
                            col_dir
                            / _slugify(run.family)
                            / _slugify(combo)
                            / _slugify(run.model_type)
                            / _slugify(transform_upper)
                            / _slugify(str(sweep_value))
                        )
                        _ensure_dir(subdir)
                        coef_df.to_csv(subdir / "coef_stats.csv", index=False)
                        corr_pairs.to_csv(subdir / "corr_pairs.csv", index=False)
                        raw_corr_pairs.to_csv(subdir / "raw_corr_pairs.csv", index=False)
                        vif_df.to_csv(subdir / "vif.csv", index=False)
                        raw_vif_df.to_csv(subdir / "raw_vif.csv", index=False)
                        summary = {
                            "family": run.family,
                            "model_name": run.name,
                            "model_type": run.model_type,
                            "reasoning_combo": combo,
                            "transform": transform_upper,
                            "sweep_param": str(sweep_value),
                            "n_features_model": int(X_model.shape[1]),
                            "n_features_raw": int(X_raw.shape[1]),
                            "cond_number": cond_num,
                            "max_abs_corr": corr_stats["max_abs_corr"],
                            "mean_abs_corr": corr_stats["mean_abs_corr"],
                            "corr_count_ge_threshold": corr_stats["count_ge_threshold"],
                            "max_vif": max_vif,
                            "vif_skipped": bool(vif_skipped),
                            "avg_sign_flip_rate": avg_flip,
                            "raw_cond_number": raw_cond_num,
                            "raw_max_abs_corr": raw_corr_stats["max_abs_corr"],
                            "raw_mean_abs_corr": raw_corr_stats["mean_abs_corr"],
                            "raw_corr_count_ge_threshold": raw_corr_stats["count_ge_threshold"],
                            "raw_max_vif": raw_max_vif,
                            "raw_vif_skipped": bool(raw_vif_skipped),
                        }
                        (subdir / "summary.json").write_text(
                            json.dumps(summary, indent=2),
                            encoding="utf-8",
                        )

                    col_summary_rows.append(
                        {
                            "family": run.family,
                            "model_name": run.name,
                            "model_type": run.model_type,
                            "reasoning_combo": combo,
                            "transform": transform_upper,
                            "sweep_param": str(sweep_value),
                            "cond_number": cond_num,
                            "max_abs_corr": corr_stats["max_abs_corr"],
                            "mean_abs_corr": corr_stats["mean_abs_corr"],
                            "corr_count_ge_threshold": corr_stats["count_ge_threshold"],
                            "max_vif": max_vif,
                            "vif_skipped": bool(vif_skipped),
                            "avg_sign_flip_rate": avg_flip,
                            "raw_cond_number": raw_cond_num,
                            "raw_max_abs_corr": raw_corr_stats["max_abs_corr"],
                            "raw_mean_abs_corr": raw_corr_stats["mean_abs_corr"],
                            "raw_corr_count_ge_threshold": raw_corr_stats["count_ge_threshold"],
                            "raw_max_vif": raw_max_vif,
                            "raw_vif_skipped": bool(raw_vif_skipped),
                        }
                    )

                if save_models:
                    full_metrics, full_model, full_transformer, feature_names_out = _full_train_metrics(
                        train_df,
                        labels,
                        run.feature_names,
                        run.model_type,
                        oof_threshold,
                        rule_mask=run.rule_mask,
                        transform=transform_upper,
                        y_train=None,
                        pca_variance=pca_variance_value,
                        pls_components=cur_pls_components,
                        sft_k=cur_sft_k,
                        logistic_penalty=penalty,
                        logistic_c=chosen_c,
                        logistic_l1_ratio=chosen_l1,
                        return_model=True,
                    )
                    if model_dir is not None:
                        _save_model_bundle(
                            out_dir=model_dir,
                            run=run,
                            combo=combo,
                            transform=transform_upper,
                            sweep_param=str(sweep_value),
                            model=full_model,
                            transformer=full_transformer,
                            feature_names_in=run.feature_names,
                            feature_names_out=feature_names_out,
                            threshold_oof=oof_threshold,
                            full_metrics=full_metrics,
                            logistic_penalty=penalty,
                            logistic_c=chosen_c,
                            logistic_l1_ratio=chosen_l1,
                        )
                else:
                    full_metrics = _full_train_metrics(
                        train_df,
                        labels,
                        run.feature_names,
                        run.model_type,
                        oof_threshold,
                        rule_mask=run.rule_mask,
                        transform=transform_upper,
                        y_train=None,
                        pca_variance=pca_variance_value,
                        pls_components=cur_pls_components,
                        sft_k=cur_sft_k,
                        logistic_penalty=penalty,
                        logistic_c=chosen_c,
                        logistic_l1_ratio=chosen_l1,
                    )
                results_rows.append(
                    {
                        "family": run.family,
                        "model_name": run.name,
                        "model_type": run.model_type,
                        "reasoning_combo": combo,
                        "transform": transform_upper,
                        "sweep_param": str(sweep_value),
                        "logistic_penalty": penalty if run.model_type in {"logistic", "elasticnet"} else "",
                        "logistic_C": chosen_c if run.model_type in {"logistic", "elasticnet"} else None,
                        "logistic_l1_ratio": chosen_l1 if run.model_type == "elasticnet" else None,
                        "f0.5_mean": means["f0.5"],
                        "f0.5_std": stds["f0.5"],
                        "roc_auc_mean": means["roc_auc"],
                        "roc_auc_std": stds["roc_auc"],
                        "pr_auc_mean": means["pr_auc"],
                        "pr_auc_std": stds["pr_auc"],
                        "precision_mean": means["precision"],
                        "precision_std": stds["precision"],
                        "recall_mean": means["recall"],
                        "recall_std": stds["recall"],
                        "acc_mean": means["accuracy"],
                        "acc_std": stds["accuracy"],
                        "threshold_oof": oof_threshold,
                        "full_train_f0.5": full_metrics["f0.5"],
                    }
                )
        results_df = pd.DataFrame(results_rows)
        sweep_df = pd.DataFrame(sweep_rows) if sweep_rows else None
        results_path = OUTPUT_DIR / f"model_testing_results{output_suffix}.csv"
        results_df.to_csv(results_path, index=False)
        if sweep_df is not None:
            if transform_upper == "BASE":
                sweep_dir = OUTPUT_DIR / "base_sweep"
            else:
                sweep_dir = OUTPUT_DIR / "model_param_sweep"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{output_suffix}.csv"
            sweep_df.to_csv(sweep_path, index=False)
        if transform_upper == "PCA" and (pca_var_sweep or transform_sweep):
            sweep_dir = OUTPUT_DIR / "PCA_Sweep_reports"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{output_suffix}.csv"
            (sweep_df if sweep_df is not None else results_df).to_csv(sweep_path, index=False)
        if transform_upper == "PLS" and transform_sweep:
            sweep_dir = OUTPUT_DIR / "PLS_Sweep_reports"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{output_suffix}.csv"
            (sweep_df if sweep_df is not None else results_df).to_csv(sweep_path, index=False)
        if transform_upper == "SFT" and transform_sweep:
            sweep_dir = OUTPUT_DIR / "SFT_Sweep_reports"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{output_suffix}.csv"
            (sweep_df if sweep_df is not None else results_df).to_csv(sweep_path, index=False)

        def _fmt(mean: float, std: float) -> str:
            return f"{mean:.3f}+/-{std:.3f}"

        model_order = model_types

        def _md_separator(cols: int) -> str:
            return "|" + "|".join(["---"] + ["---:" for _ in range(cols - 1)]) + "|"

        def _build_matrix_table(family_key: str, title: str) -> list[str]:
            header = ["Combo"] + [m.upper() for m in model_order]
            table = [title, "| " + " | ".join(header) + " |", _md_separator(len(header))]
            for combo in allowed_combos:
                row_cells = []
                for model in model_order:
                    match = next(
                        (
                            r
                            for r in results_rows
                            if r["family"] == family_key
                            and r["reasoning_combo"] == combo
                            and r["model_type"] == model
                        ),
                        None,
                    )
                    if match is None:
                        row_cells.append("--")
                    else:
                        row_cells.append(_fmt(match["f0.5_mean"], match["f0.5_std"]))
                table.append("| " + " | ".join([combo] + row_cells) + " |")
            return table

        def _build_cv_full_table(family_key: str, title: str) -> list[str]:
            header_cols = []
            for model in model_order:
                header_cols.append(f"{model.upper()} CV")
                header_cols.append(f"{model.upper()} Full")
            header = ["Combo"] + header_cols
            table = [title, "| " + " | ".join(header) + " |", _md_separator(len(header))]
            for combo in allowed_combos:
                row_cells = []
                for model in model_order:
                    match = next(
                        (
                            r
                            for r in results_rows
                            if r["family"] == family_key
                            and r["reasoning_combo"] == combo
                            and r["model_type"] == model
                        ),
                        None,
                    )
                    if match is None:
                        row_cells.extend(["--", "--"])
                    else:
                        row_cells.append(_fmt(match["f0.5_mean"], match["f0.5_std"]))
                        row_cells.append(f"{match['full_train_f0.5']:.3f}")
                table.append("| " + " | ".join([combo] + row_cells) + " |")
            return table

        model_variants_line = (
            "Model variants: logistic (l2), elasticnet."
            if model_complexity == "simple"
            else "Model variants: logistic (l2), elasticnet, mlp32/mlp4/mlp2 (1 hidden layer)."
        )
        lines = [
            "# Model Testing Report",
            f"Generated: {pd.Timestamp.utcnow().isoformat()}Z",
            "",
            model_variants_line,
            f"Model set: {model_complexity}.",
            f"Transform: {transform_upper}.",
            "",
        ]
        if transform_upper in {"PCA", "PLS", "SFT"} and (transform_sweep or (transform_upper == "PCA" and pca_var_sweep)):
            lines += ["## Sweep Results"]
            for model in model_order:
                lines += ["", f"### {model.upper()}"]
                header = ["Combo"] + [str(v).replace(".", "p") for v in variance_list]
                lines.append("| " + " | ".join(header) + " |")
                lines.append(_md_separator(len(header)))
                for combo in allowed_combos:
                    row = [combo]
                    for v in variance_list:
                        match = next(
                            (
                                r
                                for r in results_rows
                                if r["family"] == "hq_mirror"
                                and r["reasoning_combo"] == combo
                                and r["model_type"] == model
                                and r.get("sweep_param") == str(v)
                            ),
                            None,
                        )
                        row.append(_fmt(match["f0.5_mean"], match["f0.5_std"]) if match else "--")
                    lines.append("| " + " | ".join(row) + " |")
            if include_llm_engineered:
                lines += ["", "## Engineered Sweep Results"]
                for model in model_order:
                    lines += ["", f"### {model.upper()}"]
                    header = ["Combo"] + [str(v).replace(".", "p") for v in variance_list]
                    lines.append("| " + " | ".join(header) + " |")
                    lines.append(_md_separator(len(header)))
                    for combo in allowed_combos:
                        row = [combo]
                        for v in variance_list:
                            match = next(
                                (
                                    r
                                    for r in results_rows
                                    if r["family"] == f"engineered_{engineered_set_id}"
                                    and r["reasoning_combo"] == combo
                                    and r["model_type"] == model
                                    and r.get("sweep_param") == str(v)
                                ),
                                None,
                            )
                            row.append(_fmt(match["f0.5_mean"], match["f0.5_std"]) if match else "--")
                        lines.append("| " + " | ".join(row) + " |")
            lines += ["", "## Selected Logistic Hyperparameters (HQ Mirror)"]
            header = ["Combo"] + [str(v).replace(".", "p") for v in variance_list]
            lines.append("| " + " | ".join(header) + " |")
            lines.append(_md_separator(len(header)))
            for combo in allowed_combos:
                row = [combo]
                for v in variance_list:
                    match = next(
                        (
                            r
                            for r in results_rows
                            if r["family"] == "hq_mirror"
                            and r["reasoning_combo"] == combo
                            and r["model_type"] == "logistic"
                            and r.get("sweep_param") == str(v)
                        ),
                        None,
                    )
                    if match is None:
                        row.append("--")
                    else:
                        row.append(
                            f"C={match.get('logistic_C')},l1={match.get('logistic_l1_ratio')}"
                        )
                lines.append("| " + " | ".join(row) + " |")
        else:
            lines += _build_matrix_table("hq_mirror", "## HQ Mirror + Reasoning (rule layer)")
            if include_llm_engineered:
                lines += [""]
                lines += _build_matrix_table(
                    f"engineered_{engineered_set_id}",
                    f"## Engineered {engineered_set_id} + Reasoning (no rule layer)",
                )

            lines += ["", "## CV vs Full-Train (per combo, same model)"]
            lines += _build_cv_full_table("hq_mirror", "### HQ Mirror + Reasoning")
            if include_llm_engineered:
                lines += [""]
                lines += _build_cv_full_table(
                    f"engineered_{engineered_set_id}",
                    f"### Engineered {engineered_set_id} + Reasoning",
                )
            lines += ["", "## Selected Logistic Hyperparameters (HQ Mirror)"]
            lines.append("| Combo | Penalty | C | l1_ratio |")
            lines.append("|---|---|---:|---:|")
            for combo in allowed_combos:
                match = next(
                    (
                        r
                        for r in results_rows
                        if r["family"] == "hq_mirror"
                        and r["reasoning_combo"] == combo
                        and r["model_type"] == "logistic"
                    ),
                    None,
                )
                if match is None:
                    lines.append(f"| {combo} | -- | -- | -- |")
                else:
                    lines.append(
                        f"| {combo} | {match.get('logistic_penalty','')} | {match.get('logistic_C')} | {match.get('logistic_l1_ratio')} |"
                    )

        if sweep_df is not None and not sweep_df.empty:
            lines += ["", "## ElasticNet Hyperparameter Sweep (C × l1_ratio)"]
            sweep_params = sorted({str(v) for v in sweep_df["sweep_param"].unique()})
            for sweep_param in sweep_params:
                label = "base" if sweep_param in {"", "None"} else sweep_param
                lines += _build_elasticnet_sweep_tables(
                    sweep_df,
                    "hq_mirror",
                    transform_upper,
                    sweep_param,
                    allowed_combos,
                    f"### HQ Mirror (sweep={label})",
                )
                if include_llm_engineered:
                    lines += _build_elasticnet_sweep_tables(
                        sweep_df,
                        f"engineered_{engineered_set_id}",
                        transform_upper,
                        sweep_param,
                        allowed_combos,
                        f"### Engineered {engineered_set_id} (sweep={label})",
                    )
            lines += ["", "## Logistic C Sweep"]
            for sweep_param in sweep_params:
                label = "base" if sweep_param in {"", "None"} else sweep_param
                lines += _build_logistic_c_sweep_tables(
                    sweep_df,
                    "hq_mirror",
                    transform_upper,
                    sweep_param,
                    allowed_combos,
                    f"### HQ Mirror (sweep={label})",
                )
                if include_llm_engineered:
                    lines += _build_logistic_c_sweep_tables(
                        sweep_df,
                        f"engineered_{engineered_set_id}",
                        transform_upper,
                        sweep_param,
                        allowed_combos,
                        f"### Engineered {engineered_set_id} (sweep={label})",
                    )

        lines += ["", "## Interpretability (HQ Mirror only)"]
        if not interp_on:
            lines += ["Interpretability disabled for this run."]
        else:
            for combo in INTERP_COMBOS:
                if combo not in allowed_combos:
                    continue
                lines += ["", f"### {combo}"]
                for model in model_order:
                    perm_df = perm_results.get((combo, model))
                    shap_df = shap_results.get((combo, model))
                    lines += [f"**{model.upper()} â€” Permutation Importance (Top 10)**"]
                    if perm_df is None or perm_df.empty:
                        lines.append("_No permutation results._")
                    else:
                        lines.append("| Feature | Mean drop | Std |")
                        lines.append("|---|---:|---:|")
                        for _, row in perm_df.head(10).iterrows():
                            lines.append(f"| {row['feature']} | {row['mean']:.4f} | {row['std']:.4f} |")
                    lines += [f"**{model.upper()} â€” SHAP (Mean |value|, Top 10)**"]
                    if shap_df is None or shap_df.empty:
                        lines.append("_No permutation results._")
                    else:
                        lines.append("| Feature | Mean | Std |")
                        lines.append("|---|---:|---:|")
                        for _, row in shap_df.head(10).iterrows():
                            lines.append(f"| {row['feature']} | {row['mean']:.4f} | {row['std']:.4f} |")

        report_dir = report_dir
        if collinearity_report and col_summary_rows:
            col_path = report_dir / f"collinearity_summary{output_suffix}.csv"
            pd.DataFrame(col_summary_rows).to_csv(col_path, index=False)
            col_md_path = report_dir / f"collinearity_report{output_suffix}.md"
            col_md_lines = _build_collinearity_full_table(col_summary_rows, collinearity_corr_threshold)
            col_md_path.write_text("\n".join(col_md_lines), encoding="utf-8")
        report_path = report_dir / f"model_testing_report{output_suffix}.md"
        if collinearity_report and col_summary_rows:
            lines += _build_collinearity_section(col_summary_rows, collinearity_corr_threshold)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        notes_path = OUTPUT_DIR / "model_testing_notes.md"
        notes_path.write_text(
            "# Model Testing Notes\n\n"
            "(2) Reduce collinearity: feature de-duplication, weighted quality scores.\n\n"
            "(3) Richer prompt outputs: add orthogonal signals beyond evidence/support ratings.\n\n"
            "(4) Calibration/thresholding: compare OOF tuning, Platt scaling, and cost-aware thresholds.\n",
            encoding="utf-8",
        )

        print(f"Report saved: {report_path}")
        print(f"Results saved: {results_path}")
    for transform in transforms:
        print(f"Running transform: {transform}")
        _run_for_transform(transform)


if __name__ == "__main__":
    main()

