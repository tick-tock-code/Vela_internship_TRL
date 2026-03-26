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
) -> tuple[np.ndarray, np.ndarray, Any]:
    if model_type == "logistic":
        clf = LogisticRegression(max_iter=3000, random_state=random_state)
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
    raise ValueError(f"Unknown model_type={model_type}")


def _preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if model_type in ("logistic", "mlp32"):
        X_train, X_test = _standardize_continuous(X_train, X_test, feature_names)
    return X_train, X_test


def _oof_cv_metrics(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    rule_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    oof_scores: list[np.ndarray] = []
    oof_labels: list[np.ndarray] = []
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in splits:
        X_train = df.iloc[train_idx].copy()
        X_test = df.iloc[test_idx].copy()
        X_train, X_test = _preprocess_features(X_train, X_test, feature_names, model_type)

        train_scores, test_scores, _ = _train_model_local(
            model_type,
            X_train.values.astype(float),
            y[train_idx],
            X_test.values.astype(float),
            42,
        )
        if rule_mask is not None:
            train_scores = _apply_rule_override(train_scores, rule_mask[train_idx])
            test_scores = _apply_rule_override(test_scores, rule_mask[test_idx])

        oof_scores.append(test_scores)
        oof_labels.append(y[test_idx])

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
    return means, stds, float(oof_threshold)


def _full_train_metrics(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str,
    threshold: float,
    rule_mask: np.ndarray | None = None,
) -> dict[str, float]:
    X_train, _ = _preprocess_features(df, df.copy(), feature_names, model_type)
    scores, _, _ = _train_model_local(
        model_type,
        X_train.values.astype(float),
        y,
        X_train.values.astype(float),
        42,
    )
    if rule_mask is not None:
        scores = _apply_rule_override(scores, rule_mask)
    metrics = _metrics_from_scores(y, scores, threshold)
    metrics["accuracy"] = float(np.mean((scores >= metrics["threshold"]).astype(int) == y))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Model testing pipeline (Part 2 only).")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--engineered_family_id", type=str, default=None)
    parser.add_argument("--engineered_set_id", type=str, default=ENGINEERED_SET_ID_DEFAULT)
    args = parser.parse_args()

    _ensure_dir(OUTPUT_DIR)

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

    allowed_combos = [
        "HQ",
        "A",
        "B",
        "D",
        "A+E",
        "A+F",
        "A+D+E+F",
        "A+B+C+D+E+F",
    ]
    combos = {"HQ": []}
    combos.update(_build_reasoning_combos_numeric(full_reasoning_df, ["A", "B", "C", "D", "E", "F"]))
    combos = {k: v for k, v in combos.items() if k in allowed_combos}
    missing = [c for c in allowed_combos if c not in combos]
    if missing:
        raise RuntimeError(
            f"Missing required Full Mirror combos: {missing}. Check full_current reasoning columns."
        )

    # HQ features + rule mask
    hq_script = BASE_DIR.parent / "High_Quality_human_features" / "features" / "extract_structured.py"
    hq_df_full = _build_high_quality_features(records, hq_script)
    hq_full_no_gap = hq_df_full[HQ_FEATURES_BASE].copy()
    rule_mask_full = hq_df_full["exit_count"].fillna(0.0).astype(float).values > 0

    # Engineered set_05 (rules applied to full data)
    archives_dir = BASE_DIR / "features_storage" / "llm_engineered" / "archives"
    engineered_family_id = args.engineered_family_id or _pick_latest_family_id(archives_dir)
    engineered_set_id = args.engineered_set_id
    rules = _load_engineered_rules(engineered_family_id, engineered_set_id)
    engineered_df = _evaluate_engineered_rules(records, rules)

    # CV folds (full dataset)
    full_folds_path = (
        BASE_DIR / "features_storage" / "cv_folds" / f"folds_k{args.cv_folds}_seed42_full.json"
    )
    full_splits, _, _ = load_or_create_folds(
        founder_ids=full_founder_ids,
        labels=labels,
        cv_folds=args.cv_folds,
        random_state=42,
        folds_path=full_folds_path,
        dataset_label=f"{input_csv}_full",
        use_fixed=True,
    )

    model_runs: list[ModelRun] = []
    model_types = ["logistic", "xgb1", "xgb3", "mlp32"]
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

    results_rows: list[dict[str, Any]] = []
    for run in model_runs:
        combo = run.reasoning_combo or "HQ"
        combo_cols = combos.get(combo, [])
        if run.family == "hq_mirror":
            base_df = hq_full_no_gap
        else:
            base_df = engineered_df
        train_df = pd.concat([base_df, full_reasoning_df[combo_cols]], axis=1) if combo_cols else base_df.copy()

        means, stds, oof_threshold = _oof_cv_metrics(
            train_df,
            labels,
            run.feature_names,
            run.model_type,
            full_splits,
            rule_mask=run.rule_mask,
        )
        full_metrics = _full_train_metrics(
            train_df,
            labels,
            run.feature_names,
            run.model_type,
            oof_threshold,
            rule_mask=run.rule_mask,
        )
        results_rows.append(
            {
                "family": run.family,
                "model_name": run.name,
                "model_type": run.model_type,
                "reasoning_combo": combo,
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

    results_path = OUTPUT_DIR / "model_testing_results.csv"
    pd.DataFrame(results_rows).to_csv(results_path, index=False)

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f}+/-{std:.3f}"

    model_order = ["logistic", "xgb1", "xgb3", "mlp32"]
    def _build_matrix_table(family_key: str, title: str) -> list[str]:
        table = [title, "| Combo | " + " | ".join(m.upper() for m in model_order) + " |", "|---|---:|---:|---:|---:|"]
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
            table.append(f"| {combo} | " + " | ".join(row_cells) + " |")
        return table

    def _build_cv_full_table(family_key: str, title: str) -> list[str]:
        header_cols = []
        for model in model_order:
            header_cols.append(f"{model.upper()} CV")
            header_cols.append(f"{model.upper()} Full")
        table = [title, "| Combo | " + " | ".join(header_cols) + " |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
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
            table.append(f"| {combo} | " + " | ".join(row_cells) + " |")
        return table

    lines = [
        "# Model Testing Report",
        f"Generated: {pd.Timestamp.utcnow().isoformat()}Z",
        "",
        "Model variants: logistic, xgb1 (stump), xgb3 (depth=3), mlp32 (1 hidden layer).",
        "",
    ]
    lines += _build_matrix_table("hq_mirror", "## HQ Mirror + Reasoning (rule layer)")
    lines += ["",]
    lines += _build_matrix_table(
        f"engineered_{engineered_set_id}",
        f"## Engineered {engineered_set_id} + Reasoning (no rule layer)",
    )

    lines += ["", "## CV vs Full-Train (per combo, same model)"]
    lines += _build_cv_full_table("hq_mirror", "### HQ Mirror + Reasoning")
    lines += ["",]
    lines += _build_cv_full_table(f"engineered_{engineered_set_id}", f"### Engineered {engineered_set_id} + Reasoning")

    report_path = OUTPUT_DIR / "model_testing_report.md"
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


if __name__ == "__main__":
    main()
