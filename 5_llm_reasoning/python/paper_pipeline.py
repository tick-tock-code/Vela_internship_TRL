from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
import asyncio
import os
import hashlib
import importlib.util
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import joblib

from think_reason_learn.datasets import load_vcbench
from think_reason_learn.features import FeatureEvaluator, FeatureGenerator
from think_reason_learn.features._types import Rule
from think_reason_learn.features._types import Rule
from think_reason_learn.datasets._vcbench import (
    VCBENCH_SCHEMA,
    VCBENCH_HELPERS,
    _safe_json_parse,
)
from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice

from feature_registry import FEATURE_REGISTRY
from llm_reasoning_features import (
    ReasoningConfig,
    build_experiment_key_map,
    generate_reasoning_features,
    _refresh_llm_from_env,
)

from cv_folds import load_or_create_folds

from vcbench_pipeline import (
    HQ_FEATURES_BASE,
    LEGACY_HUMAN_FEATURE_SETS,
    _apply_rule_override,
    _build_high_quality_features,
    _custom_feature_df,
    _load_base_feature_extractor,
    _load_or_create_seed,
    _metrics_from_scores,
    _select_threshold,
    _standardize_continuous,
    _train_model,
)

from paths import BASE_DIR, PROJECT_ROOT, CONFIG_DIR, PROMPT_DIR

PAPER_DIR = BASE_DIR / "docs" / "paper_stats"
TEST_DIR = BASE_DIR / "test_dataset"
DEFAULT_TEST_CSV = TEST_DIR / "vcbench_final_private (success column removed) - vcbench_final_private.csv"
DEFAULT_TEST_REASONING = TEST_DIR / "llm_reasoning_private.parquet"
ENGINEERED_SET_IDS_DEFAULT = ["set_01", "set_04", "set_05"]
TEST_PARSE_VERSION = "vcbench_safe_json_parse_v1"


def _load_env_if_present() -> None:
    # Load .env from parent of Project_folder if present (one level above this file's parent).
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from think_reason_learn.core import _config as trl_config
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            val = val.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = val
            if key in ("OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY"):
                if hasattr(trl_config, "settings") and not getattr(trl_config.settings, key, ""):
                    setattr(trl_config.settings, key, val)
        # Reset LLM singleton so it re-reads updated settings
        from think_reason_learn.core.llms._ask import LLM
        from think_reason_learn.core._singleton import SingletonMeta
        SingletonMeta._instances.pop(LLM, None)
        import think_reason_learn.core.llms as trl_llms
        trl_llms.llm = trl_llms.LLM()
    except Exception:
        # Best-effort only; do not crash if .env is malformed.
        return


@dataclass
class ModelRun:
    name: str
    model_type: str
    feature_names: list[str]
    rule_mask: np.ndarray | None = None
    part: str = ""
    family: str = ""
    reasoning_combo: str = ""
    set_id: str = ""


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Do not overwrite existing artifacts
        return
    if src.is_dir():
        import shutil
        shutil.copytree(src, dst)
    else:
        import shutil
        shutil.copy2(src, dst)


def _pick_latest_family_id(archives_dir: Path) -> str:
    families = [d for d in archives_dir.iterdir() if d.is_dir() and d.name.startswith("family_")]
    if not families:
        raise RuntimeError(f"No engineered families found in {archives_dir}.")
    latest = max(families, key=lambda p: p.stat().st_mtime)
    return latest.name.replace("family_", "")


def _load_top3_engineered_sets() -> list[str] | None:
    path = BASE_DIR / "docs" / "paper_stats" / "engineered_top3_xgb.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ids = data.get("set_ids")
    if isinstance(ids, list) and all(isinstance(v, str) for v in ids):
        return [v for v in ids if v]
    return None


def _parse_set_ids(value: str | None) -> list[str]:
    if not value or value.strip().lower() in {"auto", "auto_xgb", "auto-xgb"}:
        auto_ids = _load_top3_engineered_sets()
        if auto_ids:
            return auto_ids
        return ENGINEERED_SET_IDS_DEFAULT.copy()
    ids = [v.strip() for v in value.split(",") if v.strip()]
    return ids or ENGINEERED_SET_IDS_DEFAULT.copy()


def _load_engineered_family_sets(
    family_id: str,
    set_ids: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    archives_dir = BASE_DIR / "features_storage" / "llm_engineered" / "archives"
    base_dir = archives_dir / f"family_{family_id}"
    if not base_dir.exists():
        raise RuntimeError(f"Engineered family not found: {base_dir}")
    pool_frames: dict[str, pd.DataFrame] = {}
    feature_names: dict[str, list[str]] = {}
    for set_id in set_ids:
        set_dir = base_dir / set_id / "current"
        parquet_path = set_dir / "llm_features.parquet"
        meta_path = set_dir / "llm_features_meta.json"
        if not parquet_path.exists():
            raise RuntimeError(f"Missing engineered features: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        feats = list(df.columns)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_feats = meta.get("feature_names") or []
            if meta_feats:
                feats = list(meta_feats)
        # Ensure consistent column order
        if feats and all(c in df.columns for c in feats):
            df = df[feats].copy()
        pool_frames[set_id] = df.copy()
        feature_names[set_id] = feats
    return pool_frames, feature_names


def _load_engineered_rules(
    family_id: str,
    set_id: str,
) -> list[Rule] | None:
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
        return None
    try:
        data = json.loads(rules_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    rules: list[Rule] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        expression = str(item.get("expression", "")).strip()
        if not name or not expression:
            continue
        rules.append(Rule(name=name, description=description, expression=expression))
    return rules or None


def _build_engineered_test_from_rules(
    family_id: str,
    set_id: str,
    feature_names: list[str],
    test_records: list[dict[str, Any]],
    test_ids: list[str],
    target_path: Path,
) -> pd.DataFrame | None:
    rules = _load_engineered_rules(family_id, set_id)
    if not rules:
        return None
    evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)
    test_df = evaluator.evaluate_df(test_records)
    missing = [c for c in feature_names if c not in test_df.columns]
    if missing:
        return None
    test_df = test_df[feature_names].copy()
    test_df.index = test_ids
    target_path.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_parquet(target_path, index=False)
    return test_df

def _make_unique_ids(records: list[dict[str, Any]], prefix: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for i, rec in enumerate(records):
        raw = rec.get("founder_uuid")
        cand: str | None = None
        if raw is not None:
            cand = str(raw).strip()
            if cand.lower() in {"", "none", "nan"}:
                cand = None
        if cand is None or cand in seen:
            cand = f"{prefix}_{i}"
        seen.add(cand)
        ids.append(cand)
    return ids


def _load_test_records(test_csv: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Load test CSV and parse JSON fields using vcbench-safe parser (mirror-aligned)."""
    df = pd.read_csv(test_csv)
    has_prose = "anonymised_prose" in df.columns
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {
            "founder_uuid": row.get("founder_uuid", None),
            "industry": row.get("industry", "") or "",
            "educations": _safe_json_parse(row.get("educations_json", "")),
            "jobs": _safe_json_parse(row.get("jobs_json", "")),
            "ipos": _safe_json_parse(row.get("ipos", "")),
            "acquisitions": _safe_json_parse(row.get("acquisitions", "")),
        }
        if has_prose:
            rec["anonymised_prose"] = row.get("anonymised_prose", "") or ""
        records.append(rec)
    return records, df


def _records_hash(records: list[dict[str, Any]]) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    payload = json.dumps(records, sort_keys=True, default=_default, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_hq_extractor(script_path: Path):
    spec = importlib.util.spec_from_file_location("hq_extract_structured", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load HQ feature script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    if not hasattr(module, "extract_features"):
        raise AttributeError(f"HQ script missing extract_features: {script_path}")
    return module.extract_features  # type: ignore[attr-defined]


def _extract_hq_from_raw_df(raw_df: pd.DataFrame, script_path: Path) -> pd.DataFrame:
    extractor = _load_hq_extractor(script_path)
    hq_df = extractor(raw_df)
    missing = [f for f in HQ_FEATURES_BASE if f not in hq_df.columns]
    if missing:
        raise RuntimeError("Missing HQ features: " + ", ".join(missing))
    return hq_df


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

        # Coerce any non-numeric columns to numeric (NaN on failure),
        # so imputation can fully resolve missing values for LR.
        X_train = X_train.apply(pd.to_numeric, errors="coerce")
        X_test = X_test.apply(pd.to_numeric, errors="coerce")

        if X_train.isna().any().any() or X_test.isna().any().any():
            if model_type == "xgboost":
                X_train = X_train.fillna(0.0)
                X_test = X_test.fillna(0.0)
            else:
                fill_values = X_train.mean(numeric_only=True)
                X_train = X_train.fillna(fill_values)
                X_test = X_test.fillna(fill_values)
                # If a column was entirely NaN, mean will be NaN; fill remaining with 0.
                X_train = X_train.fillna(0.0)
                X_test = X_test.fillna(0.0)

        if model_type == "logistic":
            X_train, X_test = _standardize_continuous(X_train, X_test, feature_names)

        train_scores, test_scores, _ = _train_model(
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

    for (train_idx, test_idx), scores in zip(splits, oof_scores):
        metrics = _metrics_from_scores(y[test_idx], scores, oof_threshold)
        metrics["accuracy"] = float(np.mean((scores >= metrics["threshold"]).astype(int) == y[test_idx]))
        fold_metrics.append(metrics)

    keys = ["roc_auc", "pr_auc", "precision", "recall", "f0.5", "accuracy"]
    means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    stds = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return means, stds, float(oof_threshold)


def _preprocess_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    model_type: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = train_df.copy()
    X_test = test_df.copy()
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    if X_train.isna().any().any() or X_test.isna().any().any():
        if model_type == "xgboost":
            X_train = X_train.fillna(0.0)
            X_test = X_test.fillna(0.0)
        else:
            fill_values = X_train.mean(numeric_only=True)
            X_train = X_train.fillna(fill_values)
            X_test = X_test.fillna(fill_values)
            X_train = X_train.fillna(0.0)
            X_test = X_test.fillna(0.0)

    if model_type == "logistic":
        X_train, X_test = _standardize_continuous(X_train, X_test, feature_names)
    return X_train, X_test


def _full_train_metrics(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_names: list[str],
    model_type: str,
    threshold: float,
    rule_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], Any, np.ndarray]:
    X_train, _ = _preprocess_features(df, df.copy(), feature_names, model_type)

    scores, _, model = _train_model(
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
    return metrics, model, X_train.values.astype(float)


def _predict_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_names: list[str],
    model_type: str,
    threshold: float,
    model: Any,
    rule_mask_train: np.ndarray | None = None,
    rule_mask_test: np.ndarray | None = None,
) -> np.ndarray:
    X_train, X_test = _preprocess_features(df_train, df_test, feature_names, model_type)

    # Use the already-trained model for predictions to avoid hidden retraining drift.
    test_scores = model.predict_proba(X_test.values.astype(float))[:, 1]
    if rule_mask_train is not None:
        # Train scores are not used for test prediction, but keep this for parity if needed.
        pass
    if rule_mask_test is not None:
        test_scores = _apply_rule_override(test_scores, rule_mask_test)
    return (test_scores >= threshold).astype(int)


def _load_reasoning_cache(df: pd.DataFrame, exp_ids: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    _, mapping = build_experiment_key_map(CONFIG_DIR / "experiments.json")
    needed = {exp: mapping.get(exp, []) for exp in exp_ids}
    id_col = "founder_uuid" if "founder_uuid" in df.columns else "row_index"
    def _is_numeric(col: str) -> bool:
        return (not col.endswith("_justification")) and (not col.endswith("_underrated_aspects"))
    cols = [id_col] + [
        c
        for exp in exp_ids
        for c in needed.get(exp, [])
        if _is_numeric(c) and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Reasoning cache missing columns: {missing}")
    return df[cols].copy(), needed


def _reindex_reasoning(
    df: pd.DataFrame,
    ids: list[str],
    row_indices: np.ndarray,
) -> pd.DataFrame:
    if "founder_uuid" in df.columns:
        if df["founder_uuid"].notna().any() and not df["founder_uuid"].duplicated().any():
            return df.set_index("founder_uuid").reindex(ids)
        # founder_uuid missing or duplicated (full_current); preserve row order
        return df.reset_index(drop=True)
    if "row_index" in df.columns:
        return df.set_index("row_index").reindex(row_indices)
    raise RuntimeError("Reasoning cache missing founder_uuid and row_index columns.")


def _build_reasoning_combos_numeric(
    reasoning_df: pd.DataFrame,
    experiment_ids: list[str],
) -> dict[str, list[str]]:
    if reasoning_df is None or reasoning_df.empty:
        return {}
    numeric_cols = {
        c
        for c in reasoning_df.columns
        if c not in ("founder_uuid", "success")
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


def _hash_frame(df: pd.DataFrame) -> str:
    arr = df.copy()
    arr = arr.fillna(0.0)
    arr = arr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    hashed = pd.util.hash_pandas_object(arr, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hash_feature_names(feature_names: list[str]) -> str:
    payload = json.dumps(feature_names, ensure_ascii=True, sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_model(model: Any) -> str:
    import io
    buff = io.BytesIO()
    joblib.dump(model, buff)
    return hashlib.sha256(buff.getvalue()).hexdigest()


def _build_model_manifest(
    model: Any,
    model_type: str,
    seed: int,
    threshold: float,
    feature_names: list[str],
    train_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    return {
        "model_type": model_type,
        "seed": seed,
        "threshold": float(threshold),
        "feature_names_hash": _hash_feature_names(feature_names),
        "train_matrix_hash": _hash_array(train_matrix),
        "label_hash": _hash_array(labels.astype(int)),
        "model_hash": _hash_model(model),
    }


def _verify_model_manifest(
    manifest: dict[str, Any],
    model: Any,
    model_type: str,
    seed: int,
    threshold: float,
    feature_names: list[str],
    train_matrix: np.ndarray,
    labels: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("model_type") != model_type:
        errors.append("model_type mismatch")
    if int(manifest.get("seed", -1)) != int(seed):
        errors.append("seed mismatch")
    if not math.isclose(float(manifest.get("threshold", -1)), float(threshold), rel_tol=1e-9, abs_tol=1e-9):
        errors.append("threshold mismatch")
    if manifest.get("feature_names_hash") != _hash_feature_names(feature_names):
        errors.append("feature_names hash mismatch")
    if manifest.get("train_matrix_hash") != _hash_array(train_matrix):
        errors.append("train_matrix hash mismatch")
    if manifest.get("label_hash") != _hash_array(labels.astype(int)):
        errors.append("label hash mismatch")
    if manifest.get("model_hash") != _hash_model(model):
        errors.append("model hash mismatch")
    return errors


def _assert_frames_equal(
    label: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    if left.shape != right.shape:
        raise RuntimeError(
            f"[parity] {label}: shape mismatch {left.shape} vs {right.shape}"
        )
    if list(left.columns) != list(right.columns):
        raise RuntimeError(
            f"[parity] {label}: column mismatch (order-sensitive)"
        )
    if any(left.dtypes != right.dtypes):
        raise RuntimeError(
            f"[parity] {label}: dtype mismatch"
        )
    if _hash_frame(left) != _hash_frame(right):
        raise RuntimeError(f"[parity] {label}: content hash mismatch")


def _parse_full_mirror_report(report_path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    if not report_path.exists():
        return {}
    lines = report_path.read_text(encoding="utf-8").splitlines()
    in_mirror = False
    model_type: str | None = None
    results: dict[tuple[str, str], tuple[float, float]] = {}
    for line in lines:
        if line.startswith("### Full Mirror + Reasoning (rule layer)"):
            in_mirror = True
            model_type = None
            continue
        if in_mirror and line.startswith("### ") and not line.startswith("### Full Mirror"):
            break
        if not in_mirror:
            continue
        if line.startswith("#### Logistic"):
            model_type = "logistic"
            continue
        if line.startswith("#### XGBoost"):
            model_type = "xgboost"
            continue
        if model_type and line.startswith("|") and not line.startswith("|---"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 3:
                continue
            combo = parts[1]
            f05 = parts[2]
            if "+/-" in f05:
                mean_str, std_str = f05.split("+/-", 1)
                try:
                    mean = float(mean_str)
                    std = float(std_str)
                except ValueError:
                    continue
            else:
                try:
                    mean = float(f05)
                    std = float("nan")
                except ValueError:
                    continue
            results[(model_type, combo)] = (mean, std)
    return results


def _ensure_test_reasoning(
    records: list[dict[str, Any]],
    test_reasoning_path: Path,
    core_prompt_path: Path,
    experiments_path: Path,
    exp_ids: list[str],
    log_dir: Path,
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
    records_hash: str,
    parse_version: str,
) -> pd.DataFrame:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "paper_pipeline_reasoning.log"

    def _log(msg: str) -> None:
        ts = datetime.now().isoformat()
        line = f"[{ts}] {msg}"
        print(line)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    meta_path = test_reasoning_path.parent / "llm_reasoning_private_meta.json"
    regen_all = False
    force_meta_write = False
    if test_reasoning_path.exists():
        if not meta_path.exists():
            force_meta_write = True
            regen_all = False
            _log(
                "Test reasoning meta missing; will reuse cached reasoning and rebuild meta. "
                f"parse_version={parse_version}"
            )
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
            if (
                meta.get("parse_version") != parse_version
                or meta.get("records_hash") != records_hash
            ):
                regen_all = True

    if regen_all and test_reasoning_path.exists():
        _log(
            "Regenerating test reasoning: metadata mismatch or missing. "
            f"parse_version={parse_version}"
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = test_reasoning_path.parent / f"{test_reasoning_path.stem}.bak_{ts}{test_reasoning_path.suffix}"
        test_reasoning_path.replace(backup_path)
        if meta_path.exists():
            meta_backup = meta_path.parent / f"{meta_path.stem}.bak_{ts}{meta_path.suffix}"
            meta_path.replace(meta_backup)
        # Archive per-experiment cached outputs to force regeneration.
        stale_root = test_reasoning_path.parent / f"_stale_{ts}"
        stale_root.mkdir(parents=True, exist_ok=True)
        for exp_id in exp_ids:
            exp_dir = test_reasoning_path.parent / f"exp_{exp_id}"
            if exp_dir.exists():
                dest = stale_root / exp_dir.name
                _log(f"Archiving stale test reasoning dir: {exp_dir} -> {dest}")
                try:
                    shutil.move(str(exp_dir), str(dest))
                except Exception as exc:
                    _log(f"WARNING: failed to archive {exp_dir}: {exc}")

    if test_reasoning_path.exists() and not regen_all:
        df = pd.read_parquet(test_reasoning_path)
        # Check for missing experiment columns before validating.
        missing_exps = [exp for exp in exp_ids if not any(c.startswith(f"{exp}_") for c in df.columns)]
        if not missing_exps:
            _load_reasoning_cache(df, exp_ids)
        numeric_cols = [c for c in df.columns if c not in ("founder_uuid", "row_index", "success")]
        # Compute NaN presence per experiment (avoid repairing clean experiments).
        _, exp_key_map = build_experiment_key_map(CONFIG_DIR / "experiments.json")
        exp_nan_flags: dict[str, bool] = {}
        for exp_id in exp_ids:
            exp_cols = [
                c
                for c in exp_key_map.get(exp_id, [])
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            exp_nan_flags[exp_id] = bool(exp_cols and df[exp_cols].isna().any().any())
        any_nan = any(exp_nan_flags.values())

        if not missing_exps and not any_nan:
            _log("Using cached test reasoning (clean, no missing experiments).")
            meta_payload = {
                "parse_version": parse_version,
                "records_hash": records_hash,
                "n_records": len(records),
                "experiments": exp_ids,
                "saved_at": datetime.now().isoformat(),
            }
            if force_meta_write or not meta_path.exists():
                meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
            return df
        # Targeted repair or missing-exp generation (do not delete the file)
        _log(
            "Cached reasoning needs repair/missing experiments. "
            f"missing_exps={missing_exps} nan_flags={exp_nan_flags}"
        )
        batch_size = 20
        labels = np.zeros(len(records), dtype=int)
        output_dir = test_reasoning_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_df: pd.DataFrame | None = None
        id_col: str | None = None
        exp_list = missing_exps if missing_exps else exp_ids
        # If we're only filling missing experiments, start from existing df.
        if missing_exps:
            merged_df = df.copy()
            id_col = "founder_uuid" if "founder_uuid" in df.columns else "row_index"
        for exp_id in exp_list:
            exp_missing = exp_id in missing_exps
            exp_cols = [
                c
                for c in exp_key_map.get(exp_id, [])
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            exp_nan_rows: list[int] = []
            if not exp_missing and exp_cols:
                exp_nan_rows = (
                    df[exp_cols].isna().any(axis=1).to_numpy().nonzero()[0].tolist()
                )
            if not exp_missing and not exp_nan_rows:
                _log(f"Skipping exp {exp_id}: no NaNs detected.")
                continue
            batch_ids = (
                sorted({int(idx // batch_size) for idx in exp_nan_rows})
                if exp_nan_rows
                else None
            )
            _log(
                f"Repairing/adding test reasoning for exp {exp_id} "
                f"(target_batches={len(batch_ids) if batch_ids else 'all'})"
            )
            exp_dir = output_dir / f"exp_{exp_id}"
            exp_dir.mkdir(parents=True, exist_ok=True)
            meta_path = exp_dir / "llm_reasoning_private_manifest.json"
            cfg_repair = ReasoningConfig(
                model=model,
                dataset_size="full",
                random_state=42,
                core_prompt_path=core_prompt_path,
                experiments_path=experiments_path,
                providers=providers,
                google_model=google_model,
                batch_size=batch_size,
                concurrency=10,
                experiments=[exp_id],
                dry_run=False,
                log_dir=log_dir,
                log_every=10,
                repair_nan=True,
                inline_repair=True,
                rate_limit_fallback_sequence=[8, 6, 4, 2, 1],
                repair_existing=bool(batch_ids),
                target_batch_indices=batch_ids,
            )
            try:
                df_exp, _ = generate_reasoning_features(
                    records,
                    labels,
                    cfg_repair,
                    output_dir=exp_dir,
                    metadata_path=meta_path,
                    existing_df=df if batch_ids else None,
                )
            except Exception as exc:
                _log(f"ERROR: exp {exp_id} repair failed: {exc}")
                raise
            df_exp, _ = _load_reasoning_cache(df_exp, [exp_id])
            if id_col is None:
                id_col = "founder_uuid" if "founder_uuid" in df_exp.columns else "row_index"
            exp_df = df_exp[[id_col] + [c for c in df_exp.columns if c != id_col]].copy()
            if merged_df is None:
                merged_df = exp_df
            else:
                # Drop overlapping experiment columns to allow refreshed values to merge cleanly.
                overlap = [c for c in exp_df.columns if c != id_col and c in merged_df.columns]
                if overlap:
                    merged_df = merged_df.drop(columns=overlap, errors="ignore")
                merged_df = merged_df.merge(exp_df, on=id_col, how="left")
        if merged_df is not None:
            merged_df.to_parquet(test_reasoning_path, index=False)
            _log("Repaired/merged test reasoning written.")
            meta_payload = {
                "parse_version": parse_version,
                "records_hash": records_hash,
                "n_records": len(records),
                "experiments": exp_ids,
                "saved_at": datetime.now().isoformat(),
            }
            meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
            return merged_df

    labels = np.zeros(len(records), dtype=int)
    output_dir = test_reasoning_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_df: pd.DataFrame | None = None
    id_col: str | None = None

    for exp_id in exp_ids:
        _log(f"Generating test reasoning for exp {exp_id} (full run)")
        cfg = ReasoningConfig(
            model=model,
            dataset_size="full",
            random_state=42,
            core_prompt_path=core_prompt_path,
            experiments_path=experiments_path,
            providers=providers,
            google_model=google_model,
            batch_size=20,
            concurrency=10,
            experiments=[exp_id],
            dry_run=False,
            log_dir=log_dir,
            log_every=10,
            repair_nan=True,
            inline_repair=True,
            rate_limit_fallback_sequence=[8, 6, 4, 2, 1],
        )
        exp_dir = output_dir / f"exp_{exp_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        meta_path = exp_dir / "llm_reasoning_private_manifest.json"
        try:
            df, _ = generate_reasoning_features(
                records,
                labels,
                cfg,
                output_dir=exp_dir,
                metadata_path=meta_path,
            )
        except Exception as exc:
            _log(f"ERROR: exp {exp_id} generation failed: {exc}")
            raise
        # Targeted repair for NaNs if any remain
        numeric_cols = [c for c in df.columns if c not in ("founder_uuid", "row_index", "success")]
        if numeric_cols and df[numeric_cols].isna().any().any():
            nan_rows = df[numeric_cols].isna().any(axis=1).to_numpy().nonzero()[0].tolist()
            if nan_rows:
                batch_size = max(1, int(cfg.batch_size))
                batch_ids = sorted({int(idx // batch_size) for idx in nan_rows})
                cfg_repair = ReasoningConfig(
                    model=model,
                    dataset_size="full",
                    random_state=42,
                    core_prompt_path=core_prompt_path,
                    experiments_path=experiments_path,
                    providers=providers,
                    google_model=google_model,
                    batch_size=cfg.batch_size,
                    concurrency=cfg.concurrency,
                    experiments=[exp_id],
                    dry_run=False,
                    log_dir=log_dir,
                    log_every=10,
                    repair_nan=True,
                    inline_repair=True,
                    rate_limit_fallback_sequence=[8, 6, 4, 2, 1],
                    repair_existing=True,
                    target_batch_indices=batch_ids,
                )
                try:
                    df, _ = generate_reasoning_features(
                        records,
                        labels,
                        cfg_repair,
                        output_dir=exp_dir,
                        metadata_path=meta_path,
                        existing_df=df,
                    )
                except Exception as exc:
                    _log(f"ERROR: exp {exp_id} repair failed: {exc}")
                    raise
        df, _ = _load_reasoning_cache(df, [exp_id])
        if id_col is None:
            id_col = "founder_uuid" if "founder_uuid" in df.columns else "row_index"
        exp_df = df[[id_col] + [c for c in df.columns if c != id_col]].copy()
        if merged_df is None:
            merged_df = exp_df
        else:
            merged_df = merged_df.merge(exp_df, on=id_col, how="left")

    if merged_df is None:
        raise RuntimeError("Failed to generate test reasoning features.")
    merged_df.to_parquet(test_reasoning_path, index=False)
    _log("Test reasoning generation complete (merged parquet written).")
    meta_payload = {
        "parse_version": parse_version,
        "records_hash": records_hash,
        "n_records": len(records),
        "experiments": exp_ids,
        "saved_at": datetime.now().isoformat(),
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
    return merged_df


def _generate_engineered_sets(
    seed_records: list[dict[str, Any]],
    seed_labels: np.ndarray,
    pool_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    n_sets: int,
    n_rules: int,
    model: str,
    providers: dict[str, bool],
    google_model: str | None,
    out_dir: Path,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[list[str]]]:
    _load_env_if_present()
    _refresh_llm_from_env()
    try:
        import think_reason_learn.core.llms as trl_llms
        import think_reason_learn.features._generator as trl_gen
        trl_gen.llm = trl_llms.llm
    except Exception:
        pass
    _ensure_dir(out_dir)
    rules_path = out_dir / "engineered_rules.json"
    if rules_path.exists():
        data = json.loads(rules_path.read_text(encoding="utf-8"))
        rules_sets = data.get("rules", [])
    else:
        rules_sets = []
        llm_priority = []
        if providers.get("openai", True):
            llm_priority.append(OpenAIChoice(model=model))
        if providers.get("google", False):
            llm_priority.append(GoogleChoice(model=google_model or "gemini-2.0-flash"))
        generator = FeatureGenerator(
            schema=VCBENCH_SCHEMA,
            helpers=VCBENCH_HELPERS,
            llm_priority=llm_priority,
            temperature=0.7,
        )
        for idx in range(n_sets):
            rules = asyncio.run(
                generator.generate(
                    seed_records,
                    seed_labels,
                    n_rules=n_rules,
                    n_samples=min(60, len(seed_records)),
                )
            )
            rules_sets.append(
                [
                    {
                        "name": r.name,
                        "description": r.description,
                        "expression": r.expression,
                    }
                    for r in rules
                ]
            )
        rules_path.write_text(json.dumps({"rules": rules_sets}, indent=2), encoding="utf-8")

    pool_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    feature_names: list[list[str]] = []
    for idx, rules in enumerate(rules_sets[:n_sets], 1):
        rule_objs = [Rule(name=r["name"], description=r["description"], expression=r["expression"]) for r in rules]
        evaluator = FeatureEvaluator(rule_objs, helpers=VCBENCH_HELPERS)
        pool_df = evaluator.evaluate_df(pool_records)
        test_df = evaluator.evaluate_df(test_records)
        pool_frames.append(pool_df)
        test_frames.append(test_df)
        feature_names.append(list(pool_df.columns))
        set_dir = out_dir / f"set_{idx:02d}"
        _ensure_dir(set_dir)
        pool_df.to_parquet(set_dir / "engineered_pool.parquet", index=False)
        test_df.to_parquet(set_dir / "engineered_test.parquet", index=False)
    return pool_frames, test_frames, feature_names


def main() -> None:
    _load_env_if_present()
    _refresh_llm_from_env()
    parser = argparse.ArgumentParser(description="Paper pipeline (3x3 + 2x5 models).")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--test_reasoning_parquet", type=str, default=str(DEFAULT_TEST_REASONING))
    parser.add_argument("--include_abcdef_test_preds", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--google_model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--engineered_family_id", type=str, default=None)
    parser.add_argument("--engineered_set_ids", type=str, default="auto")
    parser.add_argument("--n_engineered_sets", type=int, default=3)
    parser.add_argument("--engineered_n_rules", type=int, default=18)
    parser.add_argument("--pt2_pred_combos", type=str, default="")
    parser.add_argument("--pt2_pred_models", type=str, default="")
    args = parser.parse_args()

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    features_dir = PAPER_DIR / "features"
    human_feat_dir = features_dir / "human"
    eng_feat_dir = features_dir / "llm_engineered"
    reason_feat_dir = features_dir / "llm_reasoned"
    for d in (human_feat_dir, eng_feat_dir, reason_feat_dir):
        _ensure_dir(d)
    pt1_dir = PAPER_DIR / "pt1"
    pt2_dir = PAPER_DIR / "pt2"
    for sub in ["human", "llm_engineered", "llm_engineered_plus_reasoning"]:
        _ensure_dir(pt1_dir / sub)
    for sub in ["HQ", "A", "AE", "AF", "ADEF", "ABCDEF"]:
        _ensure_dir(pt2_dir / sub)

    test_csv = Path(args.test_csv)
    test_reasoning_path = Path(args.test_reasoning_parquet)
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    _copy_if_exists(test_csv, human_feat_dir / "test_raw.csv")

    # Load public data
    input_csv = BASE_DIR.parent / "VCBench-Starter-Kit" / "vcbench_final_public.csv"
    records, labels = load_vcbench(input_csv, "success", 0, 42)
    founder_ids = _make_unique_ids(records, "full")
    full_founder_ids = [r.get("founder_uuid") for r in records]

    # Seed + pool
    seed_path = BASE_DIR / "features_storage" / "llm_engineered" / "seed_100.json"
    seed_idx, _, _ = _load_or_create_seed(records, labels, founder_ids, 100, 42, seed_path)
    pool_idx = np.array([i for i in range(len(records)) if i not in set(seed_idx)], dtype=int)
    pool_records = [records[i] for i in pool_idx]
    pool_labels = labels[pool_idx]
    seed_records = [records[i] for i in seed_idx]
    seed_labels = labels[seed_idx]
    pool_ids = _make_unique_ids(pool_records, "pool")
    pool_founder_ids = [r.get("founder_uuid") for r in pool_records]

    # Base human features for legacy sets
    base_extractor = _load_base_feature_extractor(
        BASE_DIR.parent / "2_Human_features_running_example_script" / "vcbench_lambda_features_minimal.py"
    )
    base_rows = [base_extractor(r) for r in pool_records]
    base_df = pd.DataFrame(base_rows, index=pool_ids)
    legacy_union = set()
    for legacy in LEGACY_HUMAN_FEATURE_SETS:
        legacy_union.update(legacy["features"])
    legacy_custom = [f for f in legacy_union if f not in base_df.columns]
    legacy_custom_df = _custom_feature_df(pool_records, legacy_custom)
    legacy_custom_df.index = pool_ids
    legacy_full = pd.concat([base_df, legacy_custom_df], axis=1)
    legacy_full.to_parquet(human_feat_dir / "features_pool.parquet", index=False)

    # Legacy features for test set
    test_records, raw_test_df = _load_test_records(test_csv)
    test_records_hash = _records_hash(test_records)
    test_ids = _make_unique_ids(test_records, "test")
    base_rows_test = [base_extractor(r) for r in test_records]
    base_df_test = pd.DataFrame(base_rows_test, index=test_ids)
    legacy_custom_df_test = _custom_feature_df(test_records, legacy_custom)
    legacy_custom_df_test.index = test_ids
    legacy_full_test = pd.concat([base_df_test, legacy_custom_df_test], axis=1)
    legacy_full_test.to_parquet(human_feat_dir / "features_test.parquet", index=False)

    # Load reasoning caches for pool (A/E) and full (A/D/E/F)
    reasoning_pool = pd.read_parquet(
        BASE_DIR / "features_storage" / "llm_reasoning" / "currently_in_use" / "llm_reasoning_full.parquet"
    )
    reasoning_full = pd.read_parquet(
        BASE_DIR / "features_storage" / "llm_reasoning" / "full_current" / "llm_reasoning_full.parquet"
    )
    pool_reasoning_df, pool_map = _load_reasoning_cache(reasoning_pool, ["A", "B", "E"])
    # Full reasoning uses numeric columns directly (match vcbench_pipeline behavior)
    full_reasoning_df = reasoning_full.copy()
    has_b = any(c.startswith("B_") for c in full_reasoning_df.columns)
    has_c = any(c.startswith("C_") for c in full_reasoning_df.columns)
    has_abcdef = has_b and has_c
    # Normalize reasoning columns to numeric where possible (match vcbench_pipeline)
    for col in pool_reasoning_df.columns:
        if col not in ("founder_uuid", "success", "row_index"):
            pool_reasoning_df[col] = pd.to_numeric(pool_reasoning_df[col], errors="ignore")
    for col in full_reasoning_df.columns:
        if col not in ("founder_uuid", "success", "row_index"):
            full_reasoning_df[col] = pd.to_numeric(full_reasoning_df[col], errors="ignore")

    # Normalize indices to align with pool/test ordering.
    if "row_index" in pool_reasoning_df.columns:
        pool_reasoning_df = pool_reasoning_df.sort_values("row_index").reset_index(drop=True)
        pool_reasoning_df = pool_reasoning_df.drop(columns=["row_index"])
    pool_reasoning_df.index = pool_ids
    full_reasoning_df = _reindex_reasoning(full_reasoning_df, full_founder_ids, np.arange(len(records)))
    full_reasoning_df = full_reasoning_df.reset_index(drop=True)
    _copy_if_exists(
        BASE_DIR / "features_storage" / "llm_reasoning" / "currently_in_use" / "llm_reasoning_full.parquet",
        reason_feat_dir / "reasoning_pool.parquet",
    )

    # Test reasoning (A/D/E/F) via API
    test_reasoning_df = _ensure_test_reasoning(
        test_records,
        test_reasoning_path,
        PROMPT_DIR / "core_prompt.txt",
        CONFIG_DIR / "experiments.json",
        ["A", "D", "E", "F"],
        PAPER_DIR / "test_reasoning_logs",
        args.llm_model,
        {"openai": True, "google": False},
        args.google_model,
        test_records_hash,
        TEST_PARSE_VERSION,
    )
    test_reasoning_df, _ = _load_reasoning_cache(test_reasoning_df, ["A", "D", "E", "F"])
    if "founder_uuid" not in test_reasoning_df.columns and "row_index" in test_reasoning_df.columns:
        test_map = {i: test_ids[i] for i in range(len(test_ids))}
        test_reasoning_df["founder_uuid"] = test_reasoning_df["row_index"].map(test_map)
    if "founder_uuid" in test_reasoning_df.columns and test_reasoning_df["founder_uuid"].duplicated().any():
        test_reasoning_df = test_reasoning_df.drop_duplicates(subset=["founder_uuid"], keep="first")
    test_reasoning_df = test_reasoning_df.set_index("founder_uuid").reindex(test_ids)
    test_reasoning_df.to_parquet(reason_feat_dir / "reasoning_test.parquet", index=False)
    # copy per-experiment outputs if present
    for exp in ["A", "D", "E", "F"]:
        _copy_if_exists(BASE_DIR / "test_dataset" / f"exp_{exp}", reason_feat_dir / f"exp_{exp}")
    if args.include_abcdef_test_preds and not has_abcdef:
        raise RuntimeError("ABCDEF test predictions requested but full_current lacks B/C reasoning columns.")

    # LLM-engineered sets (load cached family)
    archives_dir = BASE_DIR / "features_storage" / "llm_engineered" / "archives"
    engineered_family_id = args.engineered_family_id or _pick_latest_family_id(archives_dir)
    engineered_set_ids = _parse_set_ids(args.engineered_set_ids)
    eng_pool_frames_by_id, eng_feature_names_by_id = _load_engineered_family_sets(
        engineered_family_id,
        engineered_set_ids,
    )
    eng_test_frames_by_id: dict[str, pd.DataFrame | None] = {}
    # Determine canonical indices for pool/test to avoid concat duplication
    pool_index = pool_ids
    test_index = test_ids

    # Align legacy human features to pool/test indices
    legacy_full.index = pool_index
    legacy_full_test.index = test_index
    # reasoning dfs already indexed by founder_uuid strings

    # Align engineered frames to pool indices and snapshot to paper_stats
    family_meta = BASE_DIR / "features_storage" / "llm_engineered" / "families" / f"family_{engineered_family_id}_meta.json"
    _copy_if_exists(family_meta, eng_feat_dir / f"family_{engineered_family_id}_meta.json")
    for set_id, pool_df in eng_pool_frames_by_id.items():
        pool_df = pool_df.copy()
        pool_df.index = pool_index
        eng_pool_frames_by_id[set_id] = pool_df
        _ensure_dir(eng_feat_dir / set_id)
        pool_df.to_parquet(eng_feat_dir / set_id / "engineered_pool.parquet", index=False)
        meta_path = archives_dir / f"family_{engineered_family_id}" / set_id / "current" / "llm_features_meta.json"
        _copy_if_exists(meta_path, eng_feat_dir / set_id / "engineered_meta.json")
        # Load test features if they exist and match columns; otherwise rebuild from rules.
        test_path = eng_feat_dir / set_id / "engineered_test.parquet"
        test_df: pd.DataFrame | None = None
        if test_path.exists():
            test_df = pd.read_parquet(test_path)
            if list(test_df.columns) != list(pool_df.columns):
                test_df = None
        if test_df is None:
            rebuilt = _build_engineered_test_from_rules(
                engineered_family_id,
                set_id,
                list(pool_df.columns),
                test_records,
                test_ids,
                test_path,
            )
            if rebuilt is not None:
                test_df = rebuilt
                print(f"[paper_pipeline] Rebuilt test engineered features for {set_id} from rules.")
            else:
                print(
                    f"[paper_pipeline] Test features for {set_id} missing or mismatched; "
                    "unable to rebuild from rules."
                )
        if test_df is not None:
            test_df.index = test_index
            eng_test_frames_by_id[set_id] = test_df
        else:
            eng_test_frames_by_id[set_id] = None

    # Build model runs
    splits = []
    pool_folds_path = (
        BASE_DIR / "features_storage" / "cv_folds" / f"folds_k{args.cv_folds}_seed42.json"
    )
    pool_splits, pool_fold_ids, _ = load_or_create_folds(
        founder_ids=pool_founder_ids,
        labels=pool_labels,
        cv_folds=args.cv_folds,
        random_state=42,
        folds_path=pool_folds_path,
        dataset_label=str(input_csv),
        use_fixed=True,
    )
    # Full splits must align with vcbench_pipeline fold cache
    full_folds_path = (
        BASE_DIR / "features_storage" / "cv_folds" / f"folds_k{args.cv_folds}_seed42_full.json"
    )
    full_splits, full_fold_ids, _ = load_or_create_folds(
        founder_ids=full_founder_ids,
        labels=labels,
        cv_folds=args.cv_folds,
        random_state=42,
        folds_path=full_folds_path,
        dataset_label=f"{input_csv}_full",
        use_fixed=True,
    )

    model_runs: list[ModelRun] = []
    # Part 1: human legacy (baseline comparisons)
    for legacy in LEGACY_HUMAN_FEATURE_SETS:
        model_runs.append(
            ModelRun(
                name=legacy["name"],
                model_type="xgboost",
                feature_names=legacy["features"],
                part="pt1",
                family="human",
            )
        )

    # Part 1: engineered only (cached family sets)
    for set_id in engineered_set_ids:
        feats = eng_feature_names_by_id[set_id]
        model_runs.append(
            ModelRun(
                name=f"Engineered {set_id}",
                model_type="xgboost",
                feature_names=feats,
                part="pt1",
                family="llm_engineered",
                set_id=set_id,
            )
        )
    # Part 1: engineered + reasoning (A+E only)
    pool_combos = _build_reasoning_combos_numeric(pool_reasoning_df, ["A", "E"])
    desired_pool_combos = ["A+E"]
    missing_pool_combos = [c for c in desired_pool_combos if c not in pool_combos]
    if missing_pool_combos:
        raise RuntimeError(
            f"Missing required pool reasoning combos: {missing_pool_combos}. "
            "Check currently_in_use reasoning columns."
        )
    for set_id in engineered_set_ids:
        feats = eng_feature_names_by_id[set_id]
        for combo in desired_pool_combos:
            combo_cols = pool_combos[combo]
            model_runs.append(
                ModelRun(
                    name=f"Engineered {set_id} + {combo}",
                    model_type="xgboost",
                    feature_names=feats + combo_cols,
                    part="pt1",
                    family="llm_engineered_plus_reasoning",
                    reasoning_combo=combo,
                    set_id=set_id,
                )
            )

    # Part 2: HQ + reasoning combos (mirror rule layer)
    hq_script = BASE_DIR.parent / "High_Quality_human_features" / "features" / "extract_structured.py"
    hq_df_full = _build_high_quality_features(records, hq_script)
    # Keep row order alignment with full_current (positional index)
    hq_full_no_gap = hq_df_full[HQ_FEATURES_BASE].copy()
    rule_mask_full = hq_df_full["exit_count"].fillna(0.0).astype(float).values > 0
    hq_df_test = _extract_hq_from_raw_df(raw_test_df, hq_script)
    hq_df_test.index = test_ids
    hq_test_no_gap = hq_df_test[HQ_FEATURES_BASE].copy()
    rule_mask_test = hq_df_test["exit_count"].fillna(0.0).astype(float).values > 0

    # --- Fail-fast parity checks for full mirror inputs ---
    full_reasoning_cols = [
        c
        for c in full_reasoning_df.columns
        if c not in ("founder_uuid", "success", "row_index")
        and pd.api.types.is_numeric_dtype(full_reasoning_df[c])
    ]
    full_reasoning_numeric = full_reasoning_df[full_reasoning_cols].copy()
    # Save feature snapshots to paper_stats for audit
    full_reasoning_numeric.to_parquet(reason_feat_dir / "full_current_numeric.parquet", index=False)
    hq_full_no_gap.to_parquet(human_feat_dir / "hq_full_no_gap.parquet", index=False)

    # Compare against full_current source (vcbench pipeline uses this)
    ref_full_numeric = reasoning_full[full_reasoning_cols].copy()
    for col in ref_full_numeric.columns:
        ref_full_numeric[col] = pd.to_numeric(ref_full_numeric[col], errors="ignore")
    _assert_frames_equal(
        "full_current_numeric",
        full_reasoning_numeric.reset_index(drop=True),
        ref_full_numeric.reset_index(drop=True),
    )
    # Compare against paper_stats snapshots
    _assert_frames_equal(
        "paper_stats/full_current_numeric",
        full_reasoning_numeric.reset_index(drop=True),
        pd.read_parquet(reason_feat_dir / "full_current_numeric.parquet").reset_index(drop=True),
    )
    _assert_frames_equal(
        "paper_stats/hq_full_no_gap",
        hq_full_no_gap.reset_index(drop=True),
        pd.read_parquet(human_feat_dir / "hq_full_no_gap.parquet").reset_index(drop=True),
    )
    combos = {"HQ": []}
    exp_list_full = ["A", "B", "C", "D", "E", "F"]
    combos.update(_build_reasoning_combos_numeric(full_reasoning_df, exp_list_full))
    # Restrict to requested experiment top picks
    allowed_combos = {
        "HQ",
        "A",
        "A+E",
        "A+F",
        "A+D+E+F",
        "A+B+C+D+E+F",
    }
    combos = {k: v for k, v in combos.items() if k in allowed_combos}
    missing_combos = sorted(allowed_combos.difference(combos.keys()))
    if missing_combos:
        raise RuntimeError(
            f"Missing required Full Mirror combos: {missing_combos}. "
            "Check full_current reasoning columns."
        )
    for combo, cols in combos.items():
        for model_type in ("logistic", "xgboost"):
            model_runs.append(
                ModelRun(
                    name=f"Mirror HQ {combo} ({model_type.upper()})",
                    model_type=model_type,
                    feature_names=HQ_FEATURES_BASE + cols,
                    rule_mask=rule_mask_full,
                    part="pt2",
                    family=combo,
                    reasoning_combo=combo,
                )
            )

    results_rows: list[dict[str, Any]] = []
    preds_pt1: dict[str, list[int]] = {"founder_uuid": test_ids}
    preds_pt2: dict[str, list[int]] = {"founder_uuid": test_ids}
    verification_rows: list[dict[str, Any]] = []
    pred_combo_filter = [c.strip() for c in args.pt2_pred_combos.split(",") if c.strip()]
    pred_model_filter = [m.strip().lower() for m in args.pt2_pred_models.split(",") if m.strip()]
    # Default Part-2 prediction filter: HQ, A+E, A+D+E+F (both models).
    if not pred_combo_filter:
        pred_combo_filter = ["HQ", "A+E", "A+D+E+F"]
    if not pred_model_filter:
        pred_model_filter = ["logistic", "xgboost"]

    def _safe_col(name: str) -> str:
        return name.replace(" ", "_").replace("+", "_")

    for run in model_runs:
        if run.part == "pt1":
            y_train = pool_labels
            splits = pool_splits
            if run.family == "human":
                train_df = legacy_full[run.feature_names].copy()
                test_df = legacy_full_test[run.feature_names].copy()
            elif run.family == "llm_engineered":
                set_id = run.set_id
                train_df = eng_pool_frames_by_id[set_id].copy()
                test_df = eng_test_frames_by_id.get(set_id)
            else:
                set_id = run.set_id
                combo = run.reasoning_combo or ""
                combo_cols = pool_combos.get(combo, [])
                train_df = pd.concat([eng_pool_frames_by_id[set_id], pool_reasoning_df[combo_cols]], axis=1)
                test_base = eng_test_frames_by_id.get(set_id)
                if test_base is None:
                    test_df = None
                else:
                    missing_test_cols = [c for c in combo_cols if c not in test_reasoning_df.columns]
                    if missing_test_cols:
                        print(
                            f"[paper_pipeline] Missing test reasoning columns for combo {combo}: "
                            f"{missing_test_cols}. Skipping test preds."
                        )
                        test_df = None
                    else:
                        test_df = pd.concat([test_base, test_reasoning_df[combo_cols]], axis=1)
            rule_mask_train = None
            rule_mask_pred = None
        else:
            y_train = labels
            splits = full_splits
            combo = run.reasoning_combo or "HQ"
            combo_cols = combos.get(combo, [])
            train_df = pd.concat([hq_full_no_gap, full_reasoning_df[combo_cols]], axis=1) if combo_cols else hq_full_no_gap.copy()
            missing_test_cols = [c for c in combo_cols if c not in test_reasoning_df.columns]
            if missing_test_cols:
                test_df = None
                if args.include_abcdef_test_preds:
                    raise RuntimeError(
                        f"Missing test reasoning columns for combo {combo}: {missing_test_cols}"
                    )
                print(f"[paper_pipeline] Skipping test preds for combo {combo} (missing {len(missing_test_cols)} cols).")
            else:
                test_df = pd.concat([hq_test_no_gap, test_reasoning_df[combo_cols]], axis=1) if combo_cols else hq_test_no_gap.copy()
            rule_mask_train = run.rule_mask
            rule_mask_pred = rule_mask_test

        means, stds, oof_threshold = _oof_cv_metrics(
            train_df,
            y_train,
            run.feature_names,
            run.model_type,
            splits,
            rule_mask=rule_mask_train,
        )
        full_metrics, model, train_matrix = _full_train_metrics(
            train_df,
            y_train,
            run.feature_names,
            run.model_type,
            oof_threshold,
            rule_mask=rule_mask_train,
        )

        folder = (pt1_dir if run.part == "pt1" else pt2_dir) / run.family / _safe_col(run.name)
        _ensure_dir(folder)
        (folder / "metrics_cv.json").write_text(json.dumps({"means": means, "stds": stds}, indent=2), encoding="utf-8")
        (folder / "threshold_oof.json").write_text(json.dumps({"threshold": oof_threshold}, indent=2), encoding="utf-8")
        (folder / "metrics_full_train.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
        joblib.dump(model, folder / "model.joblib")
        manifest = _build_model_manifest(
            model=model,
            model_type=run.model_type,
            seed=42,
            threshold=oof_threshold,
            feature_names=run.feature_names,
            train_matrix=train_matrix,
            labels=y_train,
        )
        (folder / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        results_rows.append(
            {
                "part": run.part,
                "family": run.family,
                "model_name": run.name,
                "model_type": run.model_type,
                "reasoning_combo": run.reasoning_combo or "n/a",
                "set_id": run.set_id or "",
                "f0.5_mean": means["f0.5"],
                "f0.5_std": stds["f0.5"],
                "threshold_oof": oof_threshold,
                "full_train_f0.5": full_metrics["f0.5"],
            }
        )

        # test predictions
        if run.part == "pt2" and run.reasoning_combo == "ABCDEF" and not args.include_abcdef_test_preds:
            continue
        if run.part == "pt2" and pred_combo_filter and (run.reasoning_combo or "HQ") not in pred_combo_filter:
            continue
        if run.part == "pt2" and pred_model_filter and run.model_type.lower() not in pred_model_filter:
            continue
        if test_df is None:
            preds = np.full(len(test_ids), np.nan)
            verification_rows.append(
                {
                    "model_name": run.name,
                    "status": "SKIP",
                    "details": "Missing test features; no predictions generated.",
                }
            )
        else:
            train_prepped, _ = _preprocess_features(train_df, train_df.copy(), run.feature_names, run.model_type)
            verify_errors = _verify_model_manifest(
                manifest=manifest,
                model=model,
                model_type=run.model_type,
                seed=42,
                threshold=oof_threshold,
                feature_names=run.feature_names,
                train_matrix=train_prepped.values.astype(float),
                labels=y_train,
            )
            if verify_errors:
                verification_rows.append(
                    {
                        "model_name": run.name,
                        "status": "FAIL",
                        "details": "; ".join(verify_errors),
                    }
                )
                raise RuntimeError(f"Prediction verification failed for {run.name}: {verify_errors}")
            verification_rows.append(
                {
                    "model_name": run.name,
                    "status": "PASS",
                    "details": "manifest hashes matched",
                }
            )
            preds = _predict_test(
                train_df,
                test_df,
                run.feature_names,
                run.model_type,
                oof_threshold,
                model=model,
                rule_mask_train=rule_mask_train,
                rule_mask_test=rule_mask_pred,
            )
        if run.part == "pt1":
            preds_pt1[_safe_col(run.name)] = preds.tolist()
        else:
            preds_pt2[_safe_col(run.name)] = preds.tolist()

    # Write results CSV
    results_path = PAPER_DIR / "paper_pipeline_results.csv"
    pd.DataFrame(results_rows).to_csv(results_path, index=False)

    # Write prediction CSVs
    pd.DataFrame(preds_pt1).to_csv(PAPER_DIR / "paper_pipeline_test_preds_pt1.csv", index=False)
    pd.DataFrame(preds_pt2).to_csv(PAPER_DIR / "paper_pipeline_test_preds_pt2.csv", index=False)
    (PAPER_DIR / "prediction_verification_report.json").write_text(
        json.dumps(verification_rows, indent=2), encoding="utf-8"
    )

    # Report
    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f}+/-{std:.3f}"

    part1_rows = [r for r in results_rows if r["part"] == "pt1"]
    part2_rows = [r for r in results_rows if r["part"] == "pt2"]

    lines = [
        "# Paper Pipeline Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Part 1 (Pool 4400, XGB only — engineered sets 01/04/05 and A+E only)",
        "| Set ID | Regression | Reasoning Combo | F0.5 (mean+/-std) |",
        "|---|---|---|---:|",
    ]
    for row in part1_rows:
        lines.append(
            f"| {row.get('set_id', '') or '--'} | {row['family']} | {row['reasoning_combo']} | {_fmt(row['f0.5_mean'], row['f0.5_std'])} |"
        )
    # Family mean/std across replicates
    if part1_rows:
        lines += ["", "**Part 1 family mean +/- std (F0.5):**"]
        for fam in sorted({r["family"] for r in part1_rows}):
            vals = [r["f0.5_mean"] for r in part1_rows if r["family"] == fam]
            if vals:
                lines.append(f"- {fam}: {np.mean(vals):.3f}+/-{np.std(vals):.3f}")

    lines += [
        "",
        "## Part 2 (Full 4500, Full Mirror + Reasoning; rule layer — top picks only)",
        "| Model | Combo | Type | F0.5 (mean±std) |",
        "|---|---|---|---:|",
    ]
    for row in part2_rows:
        lines.append(f"| {row['model_name']} | {row['reasoning_combo']} | {row['model_type']} | {_fmt(row['f0.5_mean'], row['f0.5_std'])} |")
    if pred_combo_filter or pred_model_filter:
        combo_note = ",".join(pred_combo_filter) if pred_combo_filter else "ALL"
        model_note = ",".join(pred_model_filter) if pred_model_filter else "ALL"
        lines.append("")
        lines.append(
            f"**Test predictions filtered to combos:** {combo_note}; **models:** {model_note}"
        )

    # Mirror summary table (LR vs XGB) in the same order as the Part 2 table above
    combos_order = [
        "HQ",
        "A",
        "A+E",
        "A+F",
        "A+D+E+F",
        "A+B+C+D+E+F",
    ]
    lines += ["", "### Mirror LR/XGB summary", "| Combo | LR F0.5 | XGB F0.5 |", "|---|---:|---:|"]
    for combo in combos_order:
        lr = next((r for r in part2_rows if r["reasoning_combo"] == combo and r["model_type"] == "logistic"), None)
        xgb = next((r for r in part2_rows if r["reasoning_combo"] == combo and r["model_type"] == "xgboost"), None)
        lr_val = _fmt(lr["f0.5_mean"], lr["f0.5_std"]) if lr else "—"
        xgb_val = _fmt(xgb["f0.5_mean"], xgb["f0.5_std"]) if xgb else "—"
        lines.append(f"| {combo} | {lr_val} | {xgb_val} |")

    # Full Mirror parity check against llm_regression_report
    mirror_ref = _parse_full_mirror_report(BASE_DIR / "docs" / "llm_regression_report.md")
    if mirror_ref:
        lines += ["", "### Full Mirror parity check (vs llm_regression_report)", "| Combo | Model | Paper F0.5 | Report F0.5 | Δ | Status |", "|---|---|---:|---:|---:|---|"]
        for row in part2_rows:
            combo_key = row["reasoning_combo"] or "HQ"
            report_key = "n/a" if combo_key == "HQ" else combo_key
            ref = mirror_ref.get((row["model_type"], report_key))
            if ref is None:
                lines.append(f"| {combo_key} | {row['model_type']} | {_fmt(row['f0.5_mean'], row['f0.5_std'])} | — | — | MISSING |")
                continue
            ref_mean, ref_std = ref
            delta = row["f0.5_mean"] - ref_mean
            status = "PASS" if abs(delta) <= 0.002 else "FAIL"
            lines.append(
                f"| {combo_key} | {row['model_type']} | {_fmt(row['f0.5_mean'], row['f0.5_std'])} | {_fmt(ref_mean, ref_std)} | {delta:+.3f} | {status} |"
            )

    lines += [
        "",
        "## Max F0.5 on full train (biased upper bound)",
        "| Model | Full-train F0.5 |",
        "|---|---:|",
    ]
    for row in results_rows:
        lines.append(f"| {row['model_name']} | {row['full_train_f0.5']:.3f} |")

    report_path = PAPER_DIR / "paper_pipeline_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Write features README
    readme = features_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# paper_stats/features\n\n"
            "This folder contains feature artifacts duplicated for paper runs.\n\n"
            "- human/\n"
            "  - features_pool.parquet: pooled training features used by paper pipeline\n"
            "  - features_test.parquet: derived test features (from private test CSV)\n"
            "  - hq_full_no_gap.parquet: HQ features for full-mirror alignment\n"
            "  - test_raw.csv: private test input CSV (no success labels)\n\n"
            "- llm_engineered/\n"
            "  - set_01/engineered_pool.parquet\n"
            "  - set_04/engineered_pool.parquet\n"
            "  - set_05/engineered_pool.parquet\n"
            "  - engineered_meta.json (per set)\n"
            "  - engineered_test.parquet (if present; optional)\n\n"
            "- llm_reasoned/\n"
            "  - reasoning_pool.parquet: pooled reasoning features (currently_in_use)\n"
            "  - reasoning_test.parquet: merged private test reasoning (A/D/E/F)\n"
            "  - full_current_numeric.parquet: numeric full_current features for mirror parity\n"
            "  - exp_A/, exp_D/, exp_E/, exp_F/: per-experiment test outputs\n",
            encoding="utf-8",
        )

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
