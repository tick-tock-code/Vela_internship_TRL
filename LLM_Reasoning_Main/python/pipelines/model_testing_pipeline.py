from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import json
import math
import io
import os
import shutil
import hashlib
import warnings
from datetime import datetime
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.neural_network import MLPClassifier

from think_reason_learn.datasets import load_vcbench
from think_reason_learn.features import FeatureEvaluator
from think_reason_learn.features._types import Rule
from think_reason_learn.datasets._vcbench import VCBENCH_HELPERS, _safe_json_parse
from lib.llm_reasoning_features import (
    ReasoningConfig,
    build_experiment_key_map,
    generate_reasoning_features,
    _refresh_llm_from_env,
)

from lib.cv_folds import load_or_create_folds
from pipelines.vcbench_pipeline import (
    HQ_FEATURES_BASE,
    _apply_rule_override,
    _build_high_quality_features,
    _metrics_from_scores,
    _select_threshold,
    _standardize_continuous,
)
from lib.paths import BASE_DIR, PROJECT_ROOT, CONFIG_DIR, PROMPT_DIR

OUTPUT_DIR = BASE_DIR / "docs" / "model_testing"
TEST_DIR = BASE_DIR / "test_dataset"
DEFAULT_TEST_CSV = TEST_DIR / "vcbench_final_private (success column removed) - vcbench_final_private.csv"
DEFAULT_TEST_REASONING = TEST_DIR / "llm_reasoning_private.parquet"
TEST_PARSE_VERSION = "vcbench_safe_json_parse_v1"
LOGGER = logging.getLogger("model_testing_pipeline")
ENGINEERED_SET_ID_DEFAULT = "set_05"
PERM_REPEATS = 3
SHAP_VAL_SAMPLE = 200
SHAP_BG_SAMPLE = 200
INTERP_COMBOS = {"HQ", "D", "A+B+C+D+E+F"}
PCA_VARIANCE_DEFAULT = 0.999
PCA_VARIANCE = PCA_VARIANCE_DEFAULT
PCA_SWEEP_VALUES = [0.9, 0.95, 0.99, 0.999, 0.9999]
PLS_COMPONENTS_DEFAULT = 6
SFT_K_DEFAULT = 30
PLS_SWEEP_VALUES = [2, 4, 6, 8, 10]
SFT_SWEEP_VALUES = [5, 10, 15, 20, 25, 30]
FINAL_BASE_COMBOS = ["HQ", "A", "A+C"]
FINAL_PLS_COMBOS = ["A", "F", "D+E+F", "A+B+C+D+E+F", "C+D+E+F"]
FINAL_PRED_COLUMNS = {
    ("logistic", "BASE", "HQ"): "LR_BASE_HQ",
    ("logistic", "BASE", "A"): "LR_BASE_A",
    ("logistic", "BASE", "A+C"): "LR_BASE_A_C",
    ("logistic", "PLS", "F"): "LR_PLS_F",
    ("logistic", "PLS", "A"): "LR_PLS_A",
    ("logistic", "PLS", "D+E+F"): "LR_PLS_DEF",
    ("logistic", "PLS", "A+B+C+D+E+F"): "LR_PLS_ABCDEF",
    ("mlp4", "PLS", "C+D+E+F"): "MLP4_PLS_CDEF",
    ("mlp4", "PLS", "A+B+C+D+E+F"): "MLP4_PLS_ABCDEF",
}


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


def _load_env_if_present() -> None:
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
        from think_reason_learn.core.llms._ask import LLM
        from think_reason_learn.core._singleton import SingletonMeta
        SingletonMeta._instances.pop(LLM, None)
        import think_reason_learn.core.llms as trl_llms
        trl_llms.llm = trl_llms.LLM()
    except Exception:
        return


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


def _records_hash(records: list[dict[str, Any]]) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return str(obj)

    payload = json.dumps(records, sort_keys=True, default=_default, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_test_records(test_csv: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
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


def _load_reasoning_cache(
    df: pd.DataFrame,
    exp_ids: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
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
    log_path = log_dir / "model_testing_reasoning.log"

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
        missing_exps = [exp for exp in exp_ids if not any(c.startswith(f"{exp}_") for c in df.columns)]
        if not missing_exps:
            _load_reasoning_cache(df, exp_ids)
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
            exp_meta = exp_dir / "llm_reasoning_private_manifest.json"
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
                    metadata_path=exp_meta,
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
        exp_meta = exp_dir / "llm_reasoning_private_manifest.json"
        try:
            df_exp, _ = generate_reasoning_features(
                records,
                labels,
                cfg,
                output_dir=exp_dir,
                metadata_path=exp_meta,
            )
        except Exception as exc:
            _log(f"ERROR: exp {exp_id} generation failed: {exc}")
            raise
        numeric_cols = [c for c in df_exp.columns if c not in ("founder_uuid", "row_index", "success")]
        if numeric_cols and df_exp[numeric_cols].isna().any().any():
            nan_rows = df_exp[numeric_cols].isna().any(axis=1).to_numpy().nonzero()[0].tolist()
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
                    df_exp, _ = generate_reasoning_features(
                        records,
                        labels,
                        cfg_repair,
                        output_dir=exp_dir,
                        metadata_path=exp_meta,
                        existing_df=df_exp,
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


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hash_feature_names(feature_names: list[str]) -> str:
    payload = json.dumps(feature_names, ensure_ascii=True, sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_model(model: Any) -> str:
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


def _resolve_run_root_dir(
    exp_scope: str,
    finalising_experiments: bool,
    group_by_evidence: bool,
) -> tuple[Path, bool]:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    scope_map = {
        "full_exps": "full_run",
        "no_llm_exps": "no_llm_exps",
        "minimal_exps": "minimal_exps",
        "vif50_exps": "vif50_exps",
    }
    desc = scope_map.get(exp_scope, exp_scope or "run")
    if finalising_experiments:
        desc = f"finalising_{desc}"
    if group_by_evidence:
        desc = f"no_evidence_rating_{desc}"
    root_dir = OUTPUT_DIR / f"{ts}_{desc}"
    _ensure_dir(root_dir)
    full_run_root = finalising_experiments or exp_scope == "full_exps"
    return root_dir, full_run_root


def _resolve_run_output_dir(
    evidence_tag: str | None,
    group_by_evidence: bool,
    run_root: Path,
    full_run_root: bool,
) -> Path:
    if not group_by_evidence:
        return run_root
    base_dir = run_root
    sub_dir = base_dir / ("no_evidence_rating" if evidence_tag else "normal")
    _ensure_dir(sub_dir)
    return sub_dir


def _init_temp_logger(log_path: Path) -> None:
    logger = logging.getLogger("model_testing_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


def _install_exception_logger() -> None:
    logger = logging.getLogger("model_testing_pipeline")

    def _excepthook(exc_type, exc_value, exc_tb):
        logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook


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
    exclude_evidence_support_rating: bool = False,
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
        cols = [
            c
            for c in reasoning_df.columns
            if c.startswith(f"{exp_id}_")
            and c in numeric_cols
            and (
                not exclude_evidence_support_rating
                or "evidence_support_rating" not in c.lower()
            )
        ]
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
    mlp_alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if model_type in {"logistic", "elasticnet"}:
        penalty = logistic_penalty
        solver = "lbfgs"
        params: dict[str, Any] = {
            "max_iter": 3000,
            "random_state": random_state,
            "C": float(logistic_c),
        }
        if penalty in {"l1", "elasticnet"}:
            solver = "saga"
            params["penalty"] = penalty
            if penalty == "elasticnet":
                l1_ratio = 0.5 if logistic_l1_ratio is None else float(logistic_l1_ratio)
                params["l1_ratio"] = l1_ratio
        params["solver"] = solver
        clf = LogisticRegression(**params)
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
            alpha=float(mlp_alpha),
            early_stopping=False,
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
            alpha=float(mlp_alpha),
            early_stopping=False,
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
            alpha=float(mlp_alpha),
            early_stopping=False,
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
    missing_train = [c for c in feature_names if c not in train_df.columns]
    missing_test = [c for c in feature_names if c not in test_df.columns]
    if missing_train or missing_test:
        raise RuntimeError(
            f"Missing features in train/test: train={missing_train}, test={missing_test}"
        )
    X_train = train_df[feature_names].copy().apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_names].copy().apply(pd.to_numeric, errors="coerce")
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


def _predict_test(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_names: list[str],
    model_type: str,
    transform: str,
    threshold: float,
    y_train: np.ndarray,
    pca_variance: float,
    pls_components: int,
    sft_k: int,
    model: Any,
    rule_mask_train: np.ndarray | None = None,
    rule_mask_test: np.ndarray | None = None,
) -> np.ndarray:
    X_train, X_test, feature_names_out, _ = _preprocess_features(
        df_train,
        df_test,
        feature_names,
        model_type,
        transform,
        y_train,
        pca_variance,
        pls_components,
        sft_k,
    )
    test_scores = model.predict_proba(X_test.values.astype(float))[:, 1]
    if rule_mask_train is not None:
        pass
    if rule_mask_test is not None:
        test_scores = _apply_rule_override(test_scores, rule_mask_test)
    return (test_scores >= threshold).astype(int)


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
    mlp_alpha: float = 0.01,
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
            mlp_alpha=mlp_alpha,
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
    mlp_alpha: float = 0.01,
    return_model: bool = False,
) -> dict[str, float] | tuple[dict[str, float], Any, Any | None, list[str], np.ndarray]:
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
        mlp_alpha=mlp_alpha,
    )
    if rule_mask is not None:
        scores = _apply_rule_override(scores, rule_mask)
    metrics = _metrics_from_scores(y, scores, threshold)
    metrics["accuracy"] = float(np.mean((scores >= metrics["threshold"]).astype(int) == y))
    if return_model:
        return metrics, model, transformer, feature_names_out, X_train.values.astype(float)
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


def _short_hash(text: str, length: int = 8) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def _safe_output_suffix(base_dir: Path, output_suffix: str) -> str:
    probe = base_dir / f"model_testing_report{output_suffix}.md"
    if len(str(probe)) > 240:
        return "_" + _short_hash(output_suffix)
    return output_suffix


def _resolve_model_dir(
    transform_upper: str,
    sweep_mode: bool,
    output_suffix: str,
    base_output_dir: Path,
    pruning_sweep: bool = False,
) -> Path:
    base_dir = base_output_dir / "pruning_sweep" if pruning_sweep else base_output_dir
    if sweep_mode and not pruning_sweep:
        if transform_upper == "PCA":
            base_dir = base_output_dir / "PCA_Sweep_reports"
        elif transform_upper == "PLS":
            base_dir = base_output_dir / "PLS_Sweep_reports"
        elif transform_upper == "SFT":
            base_dir = base_output_dir / "SFT_Sweep_reports"
    base = base_dir / f"model_testing_models{output_suffix}"
    if len(str(base)) > 200:
        digest = _short_hash(output_suffix)
        base = base_dir / f"model_testing_models_{digest}"
    return base


def _resolve_report_dir(
    transform_upper: str,
    sweep_mode: bool,
    base_sweep_enabled: bool,
    prune_tag: str,
    base_output_dir: Path,
    full_run_root: bool = False,
    pruning_sweep: bool = False,
) -> Path:
    if pruning_sweep:
        report_dir = base_output_dir / "pruning_sweep"
    else:
        if full_run_root:
            base_dir = base_output_dir
        else:
            base_dir = base_output_dir / f"reports_prune_{prune_tag}"
            _ensure_dir(base_dir)
        report_dir = base_dir
        if transform_upper == "PCA" and sweep_mode:
            report_dir = base_dir / "PCA_Sweep_reports"
        elif transform_upper == "PLS" and sweep_mode:
            report_dir = base_dir / "PLS_Sweep_reports"
        elif transform_upper == "SFT" and sweep_mode:
            report_dir = base_dir / "SFT_Sweep_reports"
    _ensure_dir(report_dir)
    return report_dir


def _resolve_collinearity_dir(report_dir: Path, output_suffix: str) -> Path:
    base = report_dir / f"collinearity_reports{output_suffix}"
    if len(str(base)) > 200:
        digest = hashlib.md5(output_suffix.encode("utf-8")).hexdigest()[:8]
        base = report_dir / f"collinearity_reports_{digest}"
    return base


def _build_collinearity_section(
    summary_rows: list[dict[str, Any]],
    corr_threshold: float,
    all_models: bool = False,
) -> list[str]:
    if not summary_rows:
        return []
    def _fmt(val: float | None) -> str:
        if val is None:
            return "--"
        if not np.isfinite(val):
            return "inf"
        return f"{val:.3f}"

    if all_models:
        rows = sorted(
            summary_rows,
            key=lambda r: (
                str(r.get("family", "")),
                str(r.get("reasoning_combo", "")),
                str(r.get("model_type", "")),
                str(r.get("transform", "")),
                str(r.get("sweep_param", "")),
            ),
        )
    else:
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
        rows = [picked[k] for k in sorted(picked.keys())]

    title = "## Collinearity Diagnostics (All models)" if all_models else "## Collinearity Diagnostics (Logistic only)"
    lines = [
        "",
        title,
        "_Model-input stats use transformed features; raw stats use pre-transform standardized features._",
        f"_Correlation threshold: {corr_threshold}_",
        "",
        "| Combo | Family | Model | Transform | Sweep | max_vif | max_abs_corr | cond_num | avg_sign_flip | raw_max_vif | raw_max_abs_corr | raw_cond_num |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        combo = row.get("reasoning_combo", "HQ")
        lines.append(
            "| "
            + " | ".join(
                [
                    combo,
                    str(row.get("family", "")),
                    str(row.get("model_type", "")),
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
    all_models: bool = False,
) -> list[str]:
    if not summary_rows:
        return []
    def _fmt(val: float | None) -> str:
        if val is None:
            return "--"
        if not np.isfinite(val):
            return "inf"
        return f"{val:.3f}"

    title = "# Collinearity Diagnostics (All models)" if all_models else "# Collinearity Diagnostics (Logistic only)"
    lines = [
        title,
        f"_Correlation threshold: {corr_threshold}_",
        "",
        "| Family | Combo | Model | Transform | Sweep | max_vif | max_abs_corr | cond_num | avg_sign_flip | raw_max_vif | raw_max_abs_corr | raw_cond_num | vif_skipped | raw_vif_skipped |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(
        summary_rows,
        key=lambda r: (
            str(r.get("family", "")),
            str(r.get("reasoning_combo", "")),
            str(r.get("model_type", "")),
            str(r.get("transform", "")),
            str(r.get("sweep_param", "")),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("family", "")),
                    str(row.get("reasoning_combo", "")),
                    str(row.get("model_type", "")),
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

    def _display_combo(combo: str) -> str:
        if family_key.startswith("engineered_") and combo == "HQ":
            return "LLM-eng"
        return combo

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
        lines += ["", f"### {_display_combo(combo)}"]
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

    def _display_combo(combo: str) -> str:
        if family_key.startswith("engineered_") and combo == "HQ":
            return "LLM-eng"
        return combo

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
        row = [_display_combo(combo)]
        for c in c_vals:
            match = combo_df[(combo_df["logistic_C"] == c)]
            if match.empty:
                row.append("--")
            else:
                r = match.iloc[0]
                row.append(_fmt(float(r["f0.5_mean"]), float(r["f0.5_std"])))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _build_mlp_sweep_tables(
    sweep_df: pd.DataFrame,
    family_key: str,
    transform_upper: str,
    sweep_param: str,
    allowed_combos: list[str],
    model_type: str,
    header_title: str,
) -> list[str]:
    subset = sweep_df[
        (sweep_df["family"] == family_key)
        & (sweep_df["model_type"] == model_type)
        & (sweep_df["transform"] == transform_upper)
        & (sweep_df["sweep_param"] == sweep_param)
    ]
    if subset.empty:
        return []

    def _display_combo(combo: str) -> str:
        if family_key.startswith("engineered_") and combo == "HQ":
            return "LLM-eng"
        return combo

    alpha_vals = sorted({float(v) for v in subset["mlp_alpha"].dropna().unique()})
    if not alpha_vals:
        return []

    def _fmt(mean: float, std: float) -> str:
        return f"{mean:.3f}+/-{std:.3f}"

    def _md_separator(cols: int) -> str:
        return "|" + "|".join(["---"] + ["---:" for _ in range(cols - 1)]) + "|"

    lines = ["", header_title]
    if sweep_param:
        lines.append(f"_Sweep param: {sweep_param}_")
    header = ["Combo"] + [str(a).replace(".", "p") for a in alpha_vals]
    lines.append("| " + " | ".join(header) + " |")
    lines.append(_md_separator(len(header)))
    for combo in allowed_combos:
        row = [_display_combo(combo)]
        combo_df = subset[subset["reasoning_combo"] == combo]
        for a in alpha_vals:
            match = combo_df[combo_df["mlp_alpha"] == a]
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
    model_manifest: dict[str, Any] | None = None,
) -> None:
    family_slug = _slugify(run.family)
    combo_slug = _slugify(combo or "HQ")
    model_slug = _slugify(run.model_type)
    transform_slug = _slugify(transform)
    sweep_slug = _slugify(str(sweep_param or "base"))
    subdir = out_dir / family_slug / combo_slug / model_slug / transform_slug / sweep_slug
    if len(str(subdir)) > 240:
        combo_slug = f"combo_{_short_hash(combo_slug)}"
        subdir = out_dir / family_slug / combo_slug / model_slug / transform_slug / sweep_slug
    if len(str(subdir)) > 240:
        sweep_slug = f"sweep_{_short_hash(sweep_slug)}"
        subdir = out_dir / family_slug / combo_slug / model_slug / transform_slug / sweep_slug
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
    if model_manifest is not None:
        manifest_path = subdir / "model_manifest.json"
        manifest_path.write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")


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
    coefs: list[np.ndarray] = []
    lengths: list[int] = []
    for fa in fold_artifacts:
        coef = np.asarray(fa["model"].coef_)
        if coef.ndim == 2:
            coef = coef[0]
        coefs.append(coef.astype(float))
        lengths.append(int(coef.shape[0]))
    if not coefs:
        return pd.DataFrame(columns=["feature", "coef_mean", "coef_std", "sign_flip_rate", "coef_cv"]), 0.0
    min_len = min(lengths) if lengths else 0
    if min_len <= 0:
        return pd.DataFrame(columns=["feature", "coef_mean", "coef_std", "sign_flip_rate", "coef_cv"]), 0.0
    if len(set(lengths)) > 1:
        coefs = [coef[:min_len] for coef in coefs]
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
    if features:
        features = features[:coef_mat.shape[1]]
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
    if model_type == "elasticnet":
        model_type = "logistic"
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
    parser.add_argument("--model_complexity", type=str, default="complex")
    parser.add_argument("--transform_sweep", type=str, default="false")
    parser.add_argument("--feature_transforms", type=str, default="Base,PLS")
    parser.add_argument("--pls_components", type=int, default=PLS_COMPONENTS_DEFAULT)
    parser.add_argument("--sft_k", type=int, default=SFT_K_DEFAULT)
    parser.add_argument("--logistic_penalty", type=str, default="l2")
    parser.add_argument("--logistic_c_grid", type=str, default="0.1,1,10")
    parser.add_argument("--logistic_l1_ratio_grid", type=str, default="0.2,0.8")
    parser.add_argument("--regularization_sweep", type=str, default="none")
    parser.add_argument("--logistic_c", type=float, default=0.1)
    parser.add_argument("--logistic_l1_ratio", type=float, default=0.5)
    parser.add_argument("--mlp_alpha_grid", type=str, default="0.001,0.01,0.1")
    parser.add_argument("--mlp_alpha", type=float, default=0.01)
    parser.add_argument("--finalising_experiments", type=str, default="false")
    parser.add_argument("--generating_test_predictions", type=str, default="true")
    parser.add_argument("--test_csv", type=str, default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--test_reasoning_parquet", type=str, default=str(DEFAULT_TEST_REASONING))
    parser.add_argument("--exclude_evidence_support_rating", type=str, default="false")
    parser.add_argument("--resume", type=str, default="false")
    parser.add_argument("--save_models", type=str, default="true")
    parser.add_argument("--collinearity_report", type=str, default="true")
    parser.add_argument("--feature_pruning", type=str, default="aggressive_pruning")
    parser.add_argument("--pruning_sweep", type=str, default="false")
    parser.add_argument("--combo_filter", type=str, default="")
    parser.add_argument("--collinearity_corr_topk", type=int, default=20)
    parser.add_argument("--collinearity_corr_threshold", type=float, default=0.9)
    parser.add_argument("--vif_max_features", type=int, default=200)
    args = parser.parse_args()
    warnings.filterwarnings(
        "ignore",
        message=".*penalty.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Inconsistent values: penalty.*",
        category=UserWarning,
    )
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    _ensure_dir(OUTPUT_DIR)
    _init_temp_logger(OUTPUT_DIR / "temp_log")
    _install_exception_logger()
    LOGGER.info("Starting model testing pipeline")
    LOGGER.info("Args: %s", vars(args))
    finalising_experiments = (
        str(args.finalising_experiments).strip().lower() in {"1", "true", "yes"}
    )
    generating_test_predictions = (
        str(args.generating_test_predictions).strip().lower() in {"1", "true", "yes"}
    )
    exp_scope = str(args.exp_scope).strip().lower()
    if exp_scope not in {"no_llm_exps", "full_exps", "minimal_exps", "vif50_exps"}:
        raise RuntimeError("--exp_scope must be one of: no_llm_exps, full_exps, minimal_exps, vif50_exps")
    if finalising_experiments and exp_scope == "minimal_exps":
        LOGGER.info("finalising_experiments=true: upgrading exp_scope from minimal_exps to no_llm_exps")
        exp_scope = "no_llm_exps"
    include_llm_engineered = exp_scope == "full_exps"
    minimal_exps = exp_scope == "minimal_exps"
    vif50_exps = exp_scope == "vif50_exps"
    run_interpretability_default = str(args.run_interpretability).strip().lower() not in {"0", "false", "no"}
    model_complexity = str(args.model_complexity).strip().lower()
    transform_sweep = str(args.transform_sweep).strip().lower() in {"1", "true", "yes"}
    save_models = str(args.save_models).strip().lower() in {"1", "true", "yes"}
    collinearity_report = str(args.collinearity_report).strip().lower() in {"1", "true", "yes"}
    collinearity_corr_topk = int(args.collinearity_corr_topk)
    collinearity_corr_threshold = float(args.collinearity_corr_threshold)
    vif_max_features = int(args.vif_max_features)
    pls_components = int(args.pls_components)
    sft_k = int(args.sft_k)
    logistic_penalty = str(args.logistic_penalty).strip().lower()
    logistic_c = float(args.logistic_c)
    logistic_l1_ratio = float(args.logistic_l1_ratio)
    regularization_sweep = str(args.regularization_sweep).strip().lower()
    exclude_evidence_support_rating = (
        str(args.exclude_evidence_support_rating).strip().lower() in {"1", "true", "yes"}
    )
    resume = str(args.resume).strip().lower() in {"1", "true", "yes"}
    feature_pruning = str(args.feature_pruning).strip().lower()
    pruning_sweep = str(args.pruning_sweep).strip().lower() in {"1", "true", "yes"}
    combo_filter_raw = str(args.combo_filter).strip()
    if generating_test_predictions:
        finalising_experiments = True
        include_llm_engineered = False
        exclude_evidence_support_rating = False
        combo_filter_raw = ""
        regularization_sweep = "none"
    collinearity_all_models = finalising_experiments
    try:
        logistic_c_grid = [float(v) for v in str(args.logistic_c_grid).split(",") if v.strip()]
    except ValueError as exc:
        raise RuntimeError("--logistic_c_grid must be a comma-separated list of floats") from exc
    try:
        logistic_l1_ratio_grid = [float(v) for v in str(args.logistic_l1_ratio_grid).split(",") if v.strip()]
    except ValueError as exc:
        raise RuntimeError("--logistic_l1_ratio_grid must be a comma-separated list of floats") from exc
    try:
        mlp_alpha_grid = [float(v) for v in str(args.mlp_alpha_grid).split(",") if v.strip()]
    except ValueError as exc:
        raise RuntimeError("--mlp_alpha_grid must be a comma-separated list of floats") from exc
    mlp_alpha = float(args.mlp_alpha)

    if logistic_penalty not in {"l1", "l2", "elasticnet"}:
        raise RuntimeError("--logistic_penalty must be one of: l1, l2, elasticnet")
    if regularization_sweep not in {"none", "elasticnet_only", "all", "mlp"}:
        raise RuntimeError("--regularization_sweep must be one of: none, elasticnet_only, all, mlp")
    if feature_pruning not in {"none", "mild_pruning", "aggressive_pruning"}:
        raise RuntimeError("--feature_pruning must be one of: none, mild_pruning, aggressive_pruning")

    run_root_dir, full_run_root = _resolve_run_root_dir(
        exp_scope,
        finalising_experiments,
        exclude_evidence_support_rating,
    )
    preds_output: dict[str, list[int]] | None = None

    prune_token_map = {
        "none": "none",
        "mild_pruning": "mild",
        "aggressive_pruning": "aggressive",
    }
    prune_tag = "pruning_sweep" if pruning_sweep else prune_token_map[feature_pruning]
    feature_pruning_label = (
        "sweep (none, mild, aggressive)" if pruning_sweep else feature_pruning
    )

    if model_complexity not in {"simple", "complex", "mlp_only"}:
        raise RuntimeError("--model_complexity must be 'simple', 'complex', or 'mlp_only'")

    transforms_raw = [t.strip().upper() for t in str(args.feature_transforms).split(",") if t.strip()]
    if finalising_experiments:
        transforms_raw = ["BASE", "PLS"]
        transform_sweep = False
    if not transforms_raw:
        transforms_raw = ["BASE"]
    for t in transforms_raw:
        if t not in {"BASE", "PCA", "PLS", "SFT"}:
            raise RuntimeError(f"Unknown feature transform: {t}")
    if pruning_sweep:
        transforms_raw = ["BASE"]

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
            try:
                full_reasoning_df[col] = pd.to_numeric(full_reasoning_df[col])
            except (TypeError, ValueError):
                pass

    if generating_test_predictions:
        allowed_combos = FINAL_BASE_COMBOS.copy()
    elif minimal_exps:
        allowed_combos = ["HQ", "A", "D", "F"]
    elif vif50_exps:
        allowed_combos = [
            "HQ",
            "A",
            "B",
            "C",
            "D",
            "E",
            "A+B",
            "A+C",
            "A+D",
            "A+E",
            "A+D+E",
            "A+C+D+E",
        ]
    elif exp_scope == "full_exps":
        allowed_combos = [
            "HQ",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "A+C",
            "A+B",
            "A+E",
            "A+D",
            "A+D+E",
            "A+C+D+E",
            "C+F",
            "D+F",
            "D+E+F",
            "C+D+E+F",
            "A+B+C+D+E+F",
        ]
    else:
        allowed_combos = [
            "HQ",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "A+C",
            "A+B",
            "A+E",
            "A+D",
            "A+D+E",
            "A+C+D+E",
            "C+F",
            "D+F",
            "D+E+F",
            "C+D+E+F",
            "A+B+C+D+E+F",
        ]
    if combo_filter_raw:
        requested: list[str] = []
        for item in combo_filter_raw.split(","):
            token = item.strip().upper()
            if not token:
                continue
            if token.startswith("HQ+"):
                token = token[3:]
            requested.append("HQ" if token == "HQ" else token)
        seen: set[str] = set()
        allowed_combos = []
        for item in requested:
            if item not in seen:
                allowed_combos.append(item)
                seen.add(item)
    finalising_pls_combos = (
        FINAL_PLS_COMBOS.copy()
        if generating_test_predictions
        else [
            "F",
            "D+F",
            "C+F",
            "D+E+F",
            "C+D+E+F",
            "A+B+C+D+E+F",
        ]
    )
    finalising_pls_combos_filtered = list(finalising_pls_combos)

    def _build_combos_for_run(
        exclude_evidence: bool,
        allowed_list: list[str],
    ) -> dict[str, list[str]]:
        combos_local = {"HQ": []}
        combos_local.update(
            _build_reasoning_combos_numeric(
                full_reasoning_df,
                ["A", "B", "C", "D", "E", "F"],
                exclude_evidence_support_rating=exclude_evidence,
            )
        )
        combos_local = {k: v for k, v in combos_local.items() if k in allowed_list}
        missing = [c for c in allowed_list if c not in combos_local]
        if missing:
            raise RuntimeError(
                f"Missing required Full Mirror combos: {missing}. Check full_current reasoning columns."
            )
        return combos_local

    # HQ features + rule mask
    hq_script = BASE_DIR.parent / "High_Quality_human_features" / "features" / "extract_structured.py"
    hq_df_full = _build_high_quality_features(records, hq_script)
    mild_prune = {
        "best_degree_prestige",
        "comfort_index",
        "longest_founding_tenure",
        "persistence_score",
    }
    aggressive_prune = mild_prune | {
        "has_prior_ipo",
        "is_serial_founder",
        "founding_role_count",
        "edu_prestige_tier",
    }
    prune_set_map = {
        "none": set(),
        "mild_pruning": mild_prune,
        "aggressive_pruning": aggressive_prune,
    }
    pruning_levels = ["none", "mild_pruning", "aggressive_pruning"] if pruning_sweep else [feature_pruning]
    rule_mask_full = hq_df_full["exit_count"].fillna(0.0).astype(float).values > 0

    test_ids: list[str] = []
    hq_test_no_gap: pd.DataFrame | None = None
    rule_mask_test: np.ndarray | None = None
    test_reasoning_df: pd.DataFrame | None = None
    if generating_test_predictions:
        _load_env_if_present()
        _refresh_llm_from_env()
        test_csv = Path(args.test_csv)
        test_reasoning_path = Path(args.test_reasoning_parquet)
        if not test_csv.exists():
            raise FileNotFoundError(f"Test CSV not found: {test_csv}")
        test_records, _ = _load_test_records(test_csv)
        records_hash = _records_hash(test_records)
        test_ids = _make_unique_ids(test_records, "test")
        hq_df_test = _build_high_quality_features(test_records, hq_script)
        hq_df_test.index = test_ids
        hq_test_no_gap = hq_df_test[HQ_FEATURES_BASE].copy()
        rule_mask_test = hq_df_test["exit_count"].fillna(0.0).astype(float).values > 0
        exp_ids = ["A", "B", "C", "D", "E", "F"]
        providers = {"openai": True, "google": False}
        test_reasoning_df = _ensure_test_reasoning(
            test_records,
            test_reasoning_path,
            PROMPT_DIR / "core_prompt.txt",
            CONFIG_DIR / "experiments.json",
            exp_ids,
            run_root_dir / "test_reasoning_logs",
            "gpt-4.1-nano",
            providers,
            "gemini-2.0-flash",
            records_hash,
            TEST_PARSE_VERSION,
        )
        test_reasoning_df, _ = _load_reasoning_cache(test_reasoning_df, exp_ids)
        if "founder_uuid" not in test_reasoning_df.columns and "row_index" in test_reasoning_df.columns:
            test_map = {i: test_ids[i] for i in range(len(test_ids))}
            test_reasoning_df["founder_uuid"] = test_reasoning_df["row_index"].map(test_map)
        if "founder_uuid" in test_reasoning_df.columns and test_reasoning_df["founder_uuid"].duplicated().any():
            test_reasoning_df = test_reasoning_df.drop_duplicates(subset=["founder_uuid"], keep="first")
        if "founder_uuid" not in test_reasoning_df.columns:
            raise RuntimeError("Test reasoning parquet missing founder_uuid (or row_index).")
        test_reasoning_df = test_reasoning_df.set_index("founder_uuid").reindex(test_ids)
        preds_output = {"founder_uuid": test_ids}

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

    if generating_test_predictions:
        model_types = ["logistic", "mlp4"]
    elif finalising_experiments:
        model_types = ["logistic", "xgb1", "mlp4"]
    elif model_complexity == "simple":
        model_types = ["logistic", "xgb1"]
    elif model_complexity == "mlp_only":
        model_types = ["mlp32", "mlp4", "mlp2"]
    else:
        model_types = ["logistic", "xgb1", "mlp4", "mlp2"]

    def _build_model_runs(
        hq_feature_list: list[str],
        combos_local: dict[str, list[str]],
    ) -> list[ModelRun]:
        runs: list[ModelRun] = []
        for combo, combo_cols in combos_local.items():
            for model_type in model_types:
                runs.append(
                    ModelRun(
                        name=f"Mirror HQ {combo} ({model_type})",
                        model_type=model_type,
                        feature_names=hq_feature_list + combo_cols,
                        rule_mask=rule_mask_full,
                        family="hq_mirror",
                        reasoning_combo=combo,
                    )
                )
                if include_llm_engineered and engineered_df is not None:
                    runs.append(
                        ModelRun(
                            name=f"Engineered {engineered_set_id} {combo} ({model_type})",
                            model_type=model_type,
                            feature_names=list(engineered_df.columns) + combo_cols,
                            rule_mask=None,
                            family=f"engineered_{engineered_set_id}",
                            reasoning_combo=combo,
                        )
                    )
        return runs

    def _dedupe(seq: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in seq:
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    transforms = _dedupe(transforms_raw)
    multi_transform = len(transforms) > 1
    print(f"Feature transforms: {', '.join(transforms)}")
    LOGGER.info("Feature transforms: %s", ", ".join(transforms))

    def _build_suffix(
        transform: str,
        pca_variance: float | None,
        interp_on: bool,
        prune_tag: str,
        sweep_token: str | None = None,
        evidence_tag: str | None = None,
    ) -> str:
        parts: list[str] = []
        parts.append(f"exp_{exp_scope}")
        if model_complexity == "simple":
            parts.append("simple_models")
        elif model_complexity == "mlp_only":
            parts.append("mlp_only")
        else:
            parts.append("complex_models")
        parts.append("interp_on" if interp_on else "interp_off")
        parts.append(f"prune_{prune_tag}")
        if evidence_tag:
            parts.append(evidence_tag)
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

    def _write_finalising_report(
        results_by_transform: dict[str, list[dict[str, Any]]],
        col_summary_rows: list[dict[str, Any]],
        evidence_tag: str | None,
        prune_tag_local: str,
        include_engineered: bool,
        engineered_set_id_local: str,
        lr_only: bool,
    ) -> None:
        def _fmt(mean: float | None, std: float | None) -> str:
            if mean is None or std is None:
                return "--"
            return f"{mean:.3f}+/-{std:.3f}"

        def _display_combo_label(family_key: str, combo: str) -> str:
            if family_key.startswith("engineered_") and combo == "HQ":
                return "LLM-eng"
            return combo

        run_output_dir = _resolve_run_output_dir(
            evidence_tag,
            group_by_evidence=exclude_evidence_support_rating,
            run_root=run_root_dir,
            full_run_root=full_run_root,
        )
        report_dir = run_output_dir if full_run_root else run_output_dir / f"reports_prune_{prune_tag_local}"
        _ensure_dir(report_dir)

        combined_rows: list[dict[str, Any]] = []
        for rows in results_by_transform.values():
            combined_rows.extend(rows)
        lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in combined_rows:
            key = (
                str(row.get("family", "")),
                str(row.get("reasoning_combo", "")),
                str(row.get("model_type", "")),
                str(row.get("transform", "")),
            )
            lookup[key] = row

        model_line = "Models: LR (BASE), LR (PLS), MLP4 (PLS)." if lr_only else "Models: LR (BASE), XGB1 (BASE), LR (PLS), MLP4 (PLS)."
        lines = [
            "# Model Testing Report (Finalising Experiments)",
            f"Generated: {pd.Timestamp.utcnow().isoformat()}Z",
            "",
            model_line,
            f"Feature pruning: {feature_pruning_label}.",
            "",
        ]

        lines += ["## HQ Mirror + Reasoning", "### BASE (LR)" if lr_only else "### BASE (LR, XGB1)"]
        header = ["Combo", "LR (BASE) CV", "LR (BASE) Full"]
        if not lr_only:
            header += ["XGB1 (BASE) CV", "XGB1 (BASE) Full"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|---:|---:|" + ("---:|---:|" if not lr_only else ""))
        for combo in allowed_combos:
            lr_row = lookup.get(("hq_mirror", combo, "logistic", "BASE"))
            xgb_row = lookup.get(("hq_mirror", combo, "xgb1", "BASE"))
            row_cells = [
                combo,
                _fmt(
                    lr_row.get("f0.5_mean") if lr_row else None,
                    lr_row.get("f0.5_std") if lr_row else None,
                ),
                f"{lr_row['full_train_f0.5']:.3f}" if lr_row else "--",
            ]
            if not lr_only:
                row_cells += [
                    _fmt(
                        xgb_row.get("f0.5_mean") if xgb_row else None,
                        xgb_row.get("f0.5_std") if xgb_row else None,
                    ),
                    f"{xgb_row['full_train_f0.5']:.3f}" if xgb_row else "--",
                ]
            lines.append("| " + " | ".join(row_cells) + " |")
        lines += ["", "### PLS (LR, MLP4)"]
        header = [
            "Combo",
            "LR (PLS) CV",
            "LR (PLS) Full",
            "MLP4 (PLS) CV",
            "MLP4 (PLS) Full",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---|---:|---:|---:|---:|")
        combos_pls_report = finalising_pls_combos_filtered or allowed_combos
        for combo in combos_pls_report:
            lr_row = lookup.get(("hq_mirror", combo, "logistic", "PLS"))
            mlp_row = lookup.get(("hq_mirror", combo, "mlp4", "PLS"))
            lines.append(
                "| "
                + " | ".join(
                    [
                        combo,
                        _fmt(
                            lr_row.get("f0.5_mean") if lr_row else None,
                            lr_row.get("f0.5_std") if lr_row else None,
                        ),
                        f"{lr_row['full_train_f0.5']:.3f}" if lr_row else "--",
                        _fmt(
                            mlp_row.get("f0.5_mean") if mlp_row else None,
                            mlp_row.get("f0.5_std") if mlp_row else None,
                        ),
                        f"{mlp_row['full_train_f0.5']:.3f}" if mlp_row else "--",
                    ]
                )
                + " |"
            )

        if include_engineered:
            family_key = f"engineered_{engineered_set_id_local}"
            lines += ["", "## LLM-Engineered + Reasoning", "### BASE (LR)" if lr_only else "### BASE (LR, XGB1)"]
            header = ["Combo", "LR (BASE) CV", "LR (BASE) Full"]
            if not lr_only:
                header += ["XGB1 (BASE) CV", "XGB1 (BASE) Full"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|---|---:|---:|" + ("---:|---:|" if not lr_only else ""))
            for combo in allowed_combos:
                lr_base = lookup.get((family_key, combo, "logistic", "BASE"))
                xgb_base = lookup.get((family_key, combo, "xgb1", "BASE"))
                row_cells = [
                    _display_combo_label(family_key, combo),
                    _fmt(
                        lr_base.get("f0.5_mean") if lr_base else None,
                        lr_base.get("f0.5_std") if lr_base else None,
                    ),
                    f"{lr_base['full_train_f0.5']:.3f}" if lr_base else "--",
                ]
                if not lr_only:
                    row_cells += [
                        _fmt(
                            xgb_base.get("f0.5_mean") if xgb_base else None,
                            xgb_base.get("f0.5_std") if xgb_base else None,
                        ),
                        f"{xgb_base['full_train_f0.5']:.3f}" if xgb_base else "--",
                    ]
                lines.append("| " + " | ".join(row_cells) + " |")
            lines += ["", "### PLS (LR, MLP4)"]
            header = [
                "Combo",
                "LR (PLS) CV",
                "LR (PLS) Full",
                "MLP4 (PLS) CV",
                "MLP4 (PLS) Full",
            ]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|---|---:|---:|---:|---:|")
            combos_pls_report = finalising_pls_combos_filtered or allowed_combos
            for combo in combos_pls_report:
                lr_pls = lookup.get((family_key, combo, "logistic", "PLS"))
                mlp_pls = lookup.get((family_key, combo, "mlp4", "PLS"))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _display_combo_label(family_key, combo),
                            _fmt(
                                lr_pls.get("f0.5_mean") if lr_pls else None,
                                lr_pls.get("f0.5_std") if lr_pls else None,
                            ),
                            f"{lr_pls['full_train_f0.5']:.3f}" if lr_pls else "--",
                            _fmt(
                                mlp_pls.get("f0.5_mean") if mlp_pls else None,
                                mlp_pls.get("f0.5_std") if mlp_pls else None,
                            ),
                            f"{mlp_pls['full_train_f0.5']:.3f}" if mlp_pls else "--",
                        ]
                    )
                    + " |"
                )

        if collinearity_report and col_summary_rows:
            lines += _build_collinearity_section(
                col_summary_rows,
                collinearity_corr_threshold,
                all_models=True,
            )

        report_path = report_dir / "model_testing_report_finalising_experiments.md"
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOGGER.info("Finalising report saved: %s", report_path)

    def _run_for_transform(
        transform: str,
        combos: dict[str, list[str]],
        evidence_tag: str | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        transform_upper = transform.upper()
        interp_on = run_interpretability_default and transform_upper not in {"PCA", "PLS"}
        LOGGER.info("Run transform start: %s (interp_on=%s)", transform_upper, interp_on)
        if transform_upper == "PCA":
            variance_list = PCA_SWEEP_VALUES if transform_sweep else [PCA_VARIANCE_DEFAULT]
        elif transform_upper == "PLS":
            variance_list = PLS_SWEEP_VALUES if transform_sweep else [pls_components]
        elif transform_upper == "SFT":
            variance_list = SFT_SWEEP_VALUES if transform_sweep else [sft_k]
        else:
            variance_list = [None]

        sweep_mode = (transform_upper in {"PCA", "PLS", "SFT"} and transform_sweep)
        pruning_tokens = [prune_token_map[level] for level in pruning_levels]

        sweep_token = "sweep" if (sweep_mode and transform_upper == "PCA") else None
        base_pca = None if sweep_mode else (PCA_VARIANCE_DEFAULT if transform_upper == "PCA" else None)
        output_suffix = _build_suffix(
            transform_upper,
            base_pca,
            interp_on,
            prune_tag,
            sweep_token=sweep_token,
            evidence_tag=evidence_tag,
        )
        run_output_dir = _resolve_run_output_dir(
            evidence_tag,
            group_by_evidence=exclude_evidence_support_rating,
            run_root=run_root_dir,
            full_run_root=full_run_root,
        )
        report_dir = _resolve_report_dir(
            transform_upper,
            sweep_mode,
            base_sweep_enabled=(
                transform_upper == "BASE"
                and regularization_sweep != "none"
                and not pruning_sweep
            ),
            prune_tag=prune_tag,
            base_output_dir=run_output_dir,
            full_run_root=full_run_root,
            pruning_sweep=pruning_sweep,
        )
        file_suffix_report = _safe_output_suffix(report_dir, output_suffix)
        file_suffix_run = _safe_output_suffix(run_output_dir, output_suffix)
        interp_dir = run_output_dir / f"interpretability{output_suffix}"
        if interp_on:
            _ensure_dir(interp_dir)
        model_dir = (
            _resolve_model_dir(
                transform_upper,
                sweep_mode,
                output_suffix,
                base_output_dir=run_output_dir,
                pruning_sweep=pruning_sweep,
            )
            if save_models
            else None
        )
        col_dir = _resolve_collinearity_dir(report_dir, output_suffix) if collinearity_report else None

        checkpoint_results_path = report_dir / f"checkpoint_results{file_suffix_report}.csv"
        checkpoint_sweep_path = report_dir / f"checkpoint_sweep{file_suffix_report}.csv"
        checkpoint_col_path = report_dir / f"checkpoint_collinearity{file_suffix_report}.csv"

        results_rows: list[dict[str, Any]] = []
        sweep_rows: list[dict[str, Any]] = []
        perm_results: dict[tuple[str, str], pd.DataFrame] = {}
        shap_results: dict[tuple[str, str], pd.DataFrame] = {}
        col_summary_rows: list[dict[str, Any]] = []
        completed_run_keys: set[tuple[str, str, str, str, str]] = set()
        completed_col_keys: set[tuple[str, ...]] = set()
        feature_col_cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}

        if resume:
            if checkpoint_results_path.exists():
                checkpoint_df = pd.read_csv(checkpoint_results_path)
                results_rows = checkpoint_df.to_dict("records")
                completed_run_keys = {
                    (
                        str(r.get("family", "")),
                        str(r.get("reasoning_combo", "")),
                        str(r.get("model_type", "")),
                        str(r.get("transform", "")),
                        str(r.get("sweep_param", "")),
                    )
                    for r in results_rows
                }
            if checkpoint_sweep_path.exists():
                sweep_rows = pd.read_csv(checkpoint_sweep_path).to_dict("records")
            if checkpoint_col_path.exists():
                col_summary_rows = pd.read_csv(checkpoint_col_path).to_dict("records")
                if collinearity_all_models:
                    completed_col_keys = {
                        (
                            str(r.get("family", "")),
                            str(r.get("reasoning_combo", "")),
                            str(r.get("model_type", "")),
                            str(r.get("transform", "")),
                            str(r.get("sweep_param", "")),
                        )
                        for r in col_summary_rows
                    }
                else:
                    completed_col_keys = {
                        (
                            str(r.get("family", "")),
                            str(r.get("reasoning_combo", "")),
                            str(r.get("transform", "")),
                            str(r.get("sweep_param", "")),
                        )
                        for r in col_summary_rows
                    }
            LOGGER.info(
                "Resume enabled: %s runs, %s sweep rows, %s collinearity rows loaded.",
                len(completed_run_keys),
                len(sweep_rows),
                len(col_summary_rows),
            )

        for prune_level in pruning_levels:
            prune_token = prune_token_map[prune_level]
            LOGGER.info("Pruning level: %s", prune_level)
            prune_set = prune_set_map[prune_level]
            hq_features_pruned = [f for f in HQ_FEATURES_BASE if f not in prune_set]
            hq_full_no_gap = hq_df_full[hq_features_pruned].copy()
            model_runs = _build_model_runs(hq_features_pruned, combos)
            for sweep_value in variance_list:
                LOGGER.info("Transform=%s sweep_value=%s", transform_upper, sweep_value)
                pca_variance_value = PCA_VARIANCE_DEFAULT
                cur_pls_components = pls_components
                cur_sft_k = sft_k
                if transform_upper == "PCA":
                    pca_variance_value = float(sweep_value)
                elif transform_upper == "PLS":
                    cur_pls_components = int(sweep_value)
                elif transform_upper == "SFT":
                    cur_sft_k = int(sweep_value)
                sweep_param_value = prune_token if pruning_sweep else str(sweep_value)
                for run in model_runs:
                    if multi_transform and transform_upper != "BASE" and run.model_type == "elasticnet":
                        continue
                    if generating_test_predictions:
                        if transform_upper == "BASE" and run.model_type != "logistic":
                            continue
                        if transform_upper == "PLS" and run.model_type not in {"logistic", "mlp4"}:
                            continue
                    combo = run.reasoning_combo or "HQ"
                    LOGGER.info(
                        "Run start family=%s combo=%s model=%s transform=%s sweep=%s reg_sweep=%s",
                        run.family,
                        combo,
                        run.model_type,
                        transform_upper,
                        sweep_param_value,
                        regularization_sweep,
                    )
                    run_key = (
                        run.family,
                        combo,
                        run.model_type,
                        transform_upper,
                        str(sweep_param_value),
                    )
                    if resume and run_key in completed_run_keys:
                        LOGGER.info("Skipping completed run: %s", run_key)
                        continue
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

                    penalty = "elasticnet" if run.model_type == "elasticnet" else "l2"
                    chosen_c = logistic_c
                    chosen_l1 = logistic_l1_ratio
                    chosen_mlp_alpha = mlp_alpha
                    if (
                        run.model_type in {"elasticnet", "logistic"}
                        and transform_upper == "BASE"
                        and not pruning_sweep
                        and (
                            regularization_sweep == "all"
                            or (
                                regularization_sweep == "elasticnet_only"
                                and run.model_type == "elasticnet"
                            )
                        )
                    ):
                            LOGGER.info(
                                "Regularization sweep: model=%s penalty=%s C_grid=%s l1_grid=%s",
                                run.model_type,
                                "elasticnet" if run.model_type == "elasticnet" else "l2",
                                logistic_c_grid,
                                logistic_l1_ratio_grid if run.model_type == "elasticnet" else [None],
                            )
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
                                    LOGGER.info(
                                        "Sweep step start: model=%s C=%s l1_ratio=%s",
                                        run.model_type,
                                        c_val,
                                        l1_val,
                                    )
                                    try:
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
                                            mlp_alpha=mlp_alpha,
                                        )
                                    except Exception:
                                        LOGGER.exception(
                                            "Sweep step failed: model=%s C=%s l1_ratio=%s",
                                            run.model_type,
                                            c_val,
                                            l1_val,
                                        )
                                        sweep_rows.append(
                                            {
                                                "family": run.family,
                                                "model_name": run.name,
                                                "model_type": run.model_type,
                                                "reasoning_combo": combo,
                                                "transform": transform_upper,
                                                "sweep_param": sweep_param_value,
                                                "logistic_penalty": penalty,
                                                "logistic_C": c_val,
                                                "logistic_l1_ratio": l1_val,
                                                "mlp_alpha": None,
                                                "f0.5_mean": None,
                                                "f0.5_std": None,
                                                "roc_auc_mean": None,
                                                "roc_auc_std": None,
                                                "pr_auc_mean": None,
                                                "pr_auc_std": None,
                                                "precision_mean": None,
                                                "precision_std": None,
                                                "recall_mean": None,
                                                "recall_std": None,
                                                "acc_mean": None,
                                                "acc_std": None,
                                                "threshold_oof": None,
                                                "error": "exception",
                                            }
                                        )
                                        pd.DataFrame(sweep_rows).to_csv(checkpoint_sweep_path, index=False)
                                        continue
                                        sweep_rows.append(
                                            {
                                                "family": run.family,
                                                "model_name": run.name,
                                                "model_type": run.model_type,
                                                "reasoning_combo": combo,
                                                "transform": transform_upper,
                                                "sweep_param": sweep_param_value,
                                                "logistic_penalty": penalty,
                                                "logistic_C": c_val,
                                                "logistic_l1_ratio": l1_val,
                                                "mlp_alpha": None,
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
                                            "error": "",
                                        }
                                    )
                                    pd.DataFrame(sweep_rows).to_csv(checkpoint_sweep_path, index=False)
                                    LOGGER.info(
                                        "Sweep step done: model=%s C=%s l1_ratio=%s f0.5=%.4f",
                                        run.model_type,
                                        c_val,
                                        l1_val,
                                        means["f0.5"],
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
                    elif (
                        run.model_type in {"mlp32", "mlp4", "mlp2"}
                        and transform_upper in {"BASE", "PLS"}
                        and not pruning_sweep
                        and regularization_sweep == "mlp"
                    ):
                            LOGGER.info(
                                "MLP alpha sweep: model=%s alpha_grid=%s",
                                run.model_type,
                                mlp_alpha_grid,
                            )
                            best_mean = -1.0
                            best_std = 1e9
                            best_alpha = mlp_alpha
                            best_metrics = None
                            best_threshold = None
                            best_folds = None
                            for alpha_val in mlp_alpha_grid:
                                LOGGER.info(
                                    "Sweep step start: model=%s alpha=%s",
                                    run.model_type,
                                    alpha_val,
                                )
                                try:
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
                                        logistic_penalty="l2",
                                        logistic_c=logistic_c,
                                        logistic_l1_ratio=logistic_l1_ratio,
                                        mlp_alpha=alpha_val,
                                    )
                                except Exception:
                                    LOGGER.exception(
                                        "MLP sweep step failed: model=%s alpha=%s",
                                        run.model_type,
                                        alpha_val,
                                    )
                                    sweep_rows.append(
                                        {
                                            "family": run.family,
                                            "model_name": run.name,
                                            "model_type": run.model_type,
                                            "reasoning_combo": combo,
                                            "transform": transform_upper,
                                            "sweep_param": sweep_param_value,
                                            "logistic_penalty": None,
                                            "logistic_C": None,
                                            "logistic_l1_ratio": None,
                                            "mlp_alpha": alpha_val,
                                            "f0.5_mean": None,
                                            "f0.5_std": None,
                                            "roc_auc_mean": None,
                                            "roc_auc_std": None,
                                            "pr_auc_mean": None,
                                            "pr_auc_std": None,
                                            "precision_mean": None,
                                            "precision_std": None,
                                            "recall_mean": None,
                                            "recall_std": None,
                                            "acc_mean": None,
                                            "acc_std": None,
                                            "threshold_oof": None,
                                            "error": "exception",
                                        }
                                    )
                                    pd.DataFrame(sweep_rows).to_csv(checkpoint_sweep_path, index=False)
                                    continue
                                sweep_rows.append(
                                    {
                                        "family": run.family,
                                        "model_name": run.name,
                                        "model_type": run.model_type,
                                        "reasoning_combo": combo,
                                        "transform": transform_upper,
                                        "sweep_param": sweep_param_value,
                                        "logistic_penalty": None,
                                        "logistic_C": None,
                                        "logistic_l1_ratio": None,
                                        "mlp_alpha": alpha_val,
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
                                        "error": "",
                                    }
                                )
                                pd.DataFrame(sweep_rows).to_csv(checkpoint_sweep_path, index=False)
                                LOGGER.info(
                                    "Sweep step done: model=%s alpha=%s f0.5=%.4f",
                                    run.model_type,
                                    alpha_val,
                                    means["f0.5"],
                                )
                                mean_f = means["f0.5"]
                                std_f = stds["f0.5"]
                                if (mean_f > best_mean + 1e-12) or (
                                    abs(mean_f - best_mean) <= 1e-12 and std_f < best_std
                                ):
                                    best_mean = mean_f
                                    best_std = std_f
                                    best_alpha = alpha_val
                                    best_metrics = (means, stds)
                                    best_threshold = oof_threshold
                                    best_folds = fold_artifacts
                            if best_metrics is None:
                                raise RuntimeError("MLP sweep failed to produce metrics.")
                            means, stds = best_metrics
                            oof_threshold = float(best_threshold)
                            fold_artifacts = best_folds
                            chosen_mlp_alpha = best_alpha
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
                            mlp_alpha=mlp_alpha,
                        )
                        chosen_c = logistic_c
                        chosen_l1 = logistic_l1_ratio
                        chosen_mlp_alpha = mlp_alpha
    
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
    
                    if collinearity_report and (collinearity_all_models or run.model_type == "logistic"):
                        if collinearity_all_models:
                            col_key = (
                                run.family,
                                combo,
                                run.model_type,
                                transform_upper,
                                str(sweep_param_value),
                            )
                        else:
                            col_key = (run.family, combo, transform_upper, str(sweep_param_value))
                        if resume and col_key in completed_col_keys:
                            LOGGER.info("Skipping collinearity (already computed): %s", col_key)
                        else:
                            preproc_group = "xgb" if run.model_type in {"xgb1", "xgb3"} else "scaled"
                            cache_key = (
                                run.family,
                                combo,
                                transform_upper,
                                str(sweep_param_value),
                                preproc_group,
                            )
                            cache = feature_col_cache.get(cache_key)
                            if cache is None:
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

                                cache = {
                                    "model_feature_names": model_feature_names,
                                    "n_features_model": int(X_model.shape[1]),
                                    "cond_num": cond_num,
                                    "corr_stats": corr_stats,
                                    "corr_pairs": corr_pairs,
                                    "vif_df": vif_df,
                                    "vif_skipped": bool(vif_skipped),
                                    "max_vif": max_vif,
                                    "n_features_raw": int(X_raw.shape[1]),
                                    "raw_cond_num": raw_cond_num,
                                    "raw_corr_stats": raw_corr_stats,
                                    "raw_corr_pairs": raw_corr_pairs,
                                    "raw_vif_df": raw_vif_df,
                                    "raw_vif_skipped": bool(raw_vif_skipped),
                                    "raw_max_vif": raw_max_vif,
                                }
                                feature_col_cache[cache_key] = cache

                            if run.model_type == "logistic":
                                coef_df, avg_flip = _coef_stability(fold_artifacts)
                            else:
                                coef_df = pd.DataFrame(
                                    columns=["feature", "coef_mean", "coef_std", "sign_flip_rate", "coef_cv"]
                                )
                                avg_flip = None

                            if col_dir is not None:
                                family_slug = _slugify(run.family)
                                combo_slug = _slugify(combo)
                                model_slug = _slugify(run.model_type)
                                transform_slug = _slugify(transform_upper)
                                sweep_slug = _slugify(str(sweep_param_value))
                                subdir = col_dir / family_slug / combo_slug / model_slug / transform_slug / sweep_slug
                                if len(str(subdir)) > 240:
                                    combo_slug = f"combo_{_short_hash(combo_slug)}"
                                    subdir = col_dir / family_slug / combo_slug / model_slug / transform_slug / sweep_slug
                                if len(str(subdir)) > 240:
                                    sweep_slug = f"sweep_{_short_hash(sweep_slug)}"
                                    subdir = col_dir / family_slug / combo_slug / model_slug / transform_slug / sweep_slug
                                _ensure_dir(subdir)
                                coef_df.to_csv(subdir / "coef_stats.csv", index=False)
                                cache["corr_pairs"].to_csv(subdir / "corr_pairs.csv", index=False)
                                cache["raw_corr_pairs"].to_csv(subdir / "raw_corr_pairs.csv", index=False)
                                cache["vif_df"].to_csv(subdir / "vif.csv", index=False)
                                cache["raw_vif_df"].to_csv(subdir / "raw_vif.csv", index=False)
                                summary = {
                                    "family": run.family,
                                    "model_name": run.name,
                                    "model_type": run.model_type,
                                    "reasoning_combo": combo,
                                    "transform": transform_upper,
                                    "sweep_param": sweep_param_value,
                                    "n_features_model": cache["n_features_model"],
                                    "n_features_raw": cache["n_features_raw"],
                                    "cond_number": cache["cond_num"],
                                    "max_abs_corr": cache["corr_stats"]["max_abs_corr"],
                                    "mean_abs_corr": cache["corr_stats"]["mean_abs_corr"],
                                    "corr_count_ge_threshold": cache["corr_stats"]["count_ge_threshold"],
                                    "max_vif": cache["max_vif"],
                                    "vif_skipped": cache["vif_skipped"],
                                    "avg_sign_flip_rate": avg_flip,
                                    "raw_cond_number": cache["raw_cond_num"],
                                    "raw_max_abs_corr": cache["raw_corr_stats"]["max_abs_corr"],
                                    "raw_mean_abs_corr": cache["raw_corr_stats"]["mean_abs_corr"],
                                    "raw_corr_count_ge_threshold": cache["raw_corr_stats"]["count_ge_threshold"],
                                    "raw_max_vif": cache["raw_max_vif"],
                                    "raw_vif_skipped": cache["raw_vif_skipped"],
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
                                    "sweep_param": sweep_param_value,
                                    "cond_number": cache["cond_num"],
                                    "max_abs_corr": cache["corr_stats"]["max_abs_corr"],
                                    "mean_abs_corr": cache["corr_stats"]["mean_abs_corr"],
                                    "corr_count_ge_threshold": cache["corr_stats"]["count_ge_threshold"],
                                    "max_vif": cache["max_vif"],
                                    "vif_skipped": cache["vif_skipped"],
                                    "avg_sign_flip_rate": avg_flip,
                                    "raw_cond_number": cache["raw_cond_num"],
                                    "raw_max_abs_corr": cache["raw_corr_stats"]["max_abs_corr"],
                                    "raw_mean_abs_corr": cache["raw_corr_stats"]["mean_abs_corr"],
                                    "raw_corr_count_ge_threshold": cache["raw_corr_stats"]["count_ge_threshold"],
                                    "raw_max_vif": cache["raw_max_vif"],
                                    "raw_vif_skipped": cache["raw_vif_skipped"],
                                }
                            )

                            completed_col_keys.add(col_key)
                    need_model = save_models or generating_test_predictions
                    full_model = None
                    full_transformer = None
                    feature_names_out = []
                    train_matrix = None
                    model_manifest = None
                    if need_model:
                        full_metrics, full_model, full_transformer, feature_names_out, train_matrix = _full_train_metrics(
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
                            mlp_alpha=chosen_mlp_alpha,
                            return_model=True,
                        )
                        if train_matrix is not None and full_model is not None:
                            model_manifest = _build_model_manifest(
                                model=full_model,
                                model_type=run.model_type,
                                seed=42,
                                threshold=oof_threshold,
                                feature_names=feature_names_out,
                                train_matrix=train_matrix,
                                labels=labels,
                            )
                        if save_models and model_dir is not None and full_model is not None:
                            _save_model_bundle(
                                out_dir=model_dir,
                                run=run,
                                combo=combo,
                                transform=transform_upper,
                                sweep_param=str(sweep_param_value),
                                model=full_model,
                                transformer=full_transformer,
                                feature_names_in=run.feature_names,
                                feature_names_out=feature_names_out,
                                threshold_oof=oof_threshold,
                                full_metrics=full_metrics,
                                logistic_penalty=penalty,
                                logistic_c=chosen_c,
                                logistic_l1_ratio=chosen_l1,
                                model_manifest=model_manifest,
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
                            mlp_alpha=chosen_mlp_alpha,
                        )
                    if generating_test_predictions and preds_output is not None:
                        out_col = FINAL_PRED_COLUMNS.get((run.model_type, transform_upper, combo))
                        if out_col:
                            if hq_test_no_gap is None or test_reasoning_df is None:
                                raise RuntimeError("Test features are required for generating predictions.")
                            combo_cols = combos.get(combo, [])
                            if combo_cols:
                                missing_test_cols = [c for c in combo_cols if c not in test_reasoning_df.columns]
                                if missing_test_cols:
                                    raise RuntimeError(
                                        f"Missing test reasoning columns for combo {combo}: {missing_test_cols}"
                                    )
                                test_df = pd.concat([hq_test_no_gap, test_reasoning_df[combo_cols]], axis=1)
                            else:
                                test_df = hq_test_no_gap.copy()
                            if full_model is None or model_manifest is None:
                                raise RuntimeError("Missing trained model or manifest for test prediction.")
                            train_prepped, _, feature_names_out_check, _ = _preprocess_features(
                                train_df,
                                train_df.copy(),
                                run.feature_names,
                                run.model_type,
                                transform_upper,
                                labels,
                                pca_variance_value,
                                cur_pls_components,
                                cur_sft_k,
                            )
                            verify_errors = _verify_model_manifest(
                                manifest=model_manifest,
                                model=full_model,
                                model_type=run.model_type,
                                seed=42,
                                threshold=oof_threshold,
                                feature_names=feature_names_out_check,
                                train_matrix=train_prepped.values.astype(float),
                                labels=labels,
                            )
                            if verify_errors:
                                raise RuntimeError(
                                    f"Prediction verification failed for {run.name}: {verify_errors}"
                                )
                            preds = _predict_test(
                                train_df,
                                test_df,
                                run.feature_names,
                                run.model_type,
                                transform_upper,
                                oof_threshold,
                                labels,
                                pca_variance_value,
                                cur_pls_components,
                                cur_sft_k,
                                full_model,
                                rule_mask_train=run.rule_mask,
                                rule_mask_test=rule_mask_test,
                            )
                            preds_output[out_col] = preds.tolist()
                    results_rows.append(
                        {
                            "family": run.family,
                            "model_name": run.name,
                            "model_type": run.model_type,
                            "reasoning_combo": combo,
                            "transform": transform_upper,
                            "sweep_param": sweep_param_value,
                            "logistic_penalty": penalty if run.model_type in {"logistic", "elasticnet"} else "",
                            "logistic_C": chosen_c if run.model_type in {"logistic", "elasticnet"} else None,
                            "logistic_l1_ratio": chosen_l1 if run.model_type == "elasticnet" else None,
                            "mlp_alpha": chosen_mlp_alpha if run.model_type in {"mlp32", "mlp4", "mlp2"} else None,
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
                    completed_run_keys.add(run_key)
                    pd.DataFrame(results_rows).to_csv(checkpoint_results_path, index=False)
                    if sweep_rows:
                        pd.DataFrame(sweep_rows).to_csv(checkpoint_sweep_path, index=False)
                    if col_summary_rows:
                        pd.DataFrame(col_summary_rows).to_csv(checkpoint_col_path, index=False)
        results_df = pd.DataFrame(results_rows)
        sweep_df = pd.DataFrame(sweep_rows) if sweep_rows else None
        results_path = (
            report_dir / f"model_testing_results{file_suffix_report}.csv"
            if pruning_sweep
            else run_output_dir / f"model_testing_results{file_suffix_run}.csv"
        )
        results_df.to_csv(results_path, index=False)
        if pruning_sweep:
            pruning_path = report_dir / f"model_testing_pruning_sweep{file_suffix_report}.csv"
            results_df.to_csv(pruning_path, index=False)
        if sweep_df is not None:
            if transform_upper == "BASE":
                sweep_dir = run_output_dir / "base_sweep"
            else:
                sweep_dir = run_output_dir / "model_param_sweep"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{file_suffix_run}.csv"
            sweep_df.to_csv(sweep_path, index=False)
        if transform_upper == "PCA" and transform_sweep:
            sweep_dir = run_output_dir / "PCA_Sweep_reports"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{file_suffix_run}.csv"
            (sweep_df if sweep_df is not None else results_df).to_csv(sweep_path, index=False)
        if transform_upper == "PLS" and transform_sweep:
            sweep_dir = run_output_dir / "PLS_Sweep_reports"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{file_suffix_run}.csv"
            (sweep_df if sweep_df is not None else results_df).to_csv(sweep_path, index=False)
        if transform_upper == "SFT" and transform_sweep:
            sweep_dir = run_output_dir / "SFT_Sweep_reports"
            _ensure_dir(sweep_dir)
            sweep_path = sweep_dir / f"model_testing_sweep{file_suffix_run}.csv"
            (sweep_df if sweep_df is not None else results_df).to_csv(sweep_path, index=False)

        def _fmt(mean: float, std: float) -> str:
            return f"{mean:.3f}+/-{std:.3f}"

        model_order = (
            [m for m in model_types if m != "elasticnet"]
            if multi_transform and transform_upper != "BASE"
            else model_types
        )
        has_logistic = "logistic" in model_order
        has_mlp = any(m in model_order for m in ("mlp32", "mlp4", "mlp2"))

        def _md_separator(cols: int) -> str:
            return "|" + "|".join(["---"] + ["---:" for _ in range(cols - 1)]) + "|"

        def _display_combo_label(family_key: str, combo: str) -> str:
            if family_key.startswith("engineered_") and combo == "HQ":
                return "LLM-eng"
            return combo

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
                table.append("| " + " | ".join([_display_combo_label(family_key, combo)] + row_cells) + " |")
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
                table.append("| " + " | ".join([_display_combo_label(family_key, combo)] + row_cells) + " |")
            return table

        if model_complexity == "simple":
            model_variants_line = "Model variants: logistic (l2), xgb1."
        elif model_complexity == "mlp_only":
            model_variants_line = "Model variants: mlp32/mlp4/mlp2 (1 hidden layer)."
        else:
            model_variants_line = "Model variants: logistic (l2), xgb1, mlp4/mlp2 (1 hidden layer)."
        lines = [
            "# Model Testing Report",
            f"Generated: {pd.Timestamp.utcnow().isoformat()}Z",
            "",
            model_variants_line,
            f"Model set: {model_complexity}.",
            f"Transform: {transform_upper}.",
            f"Feature pruning: {feature_pruning_label}.",
            "",
        ]
        if pruning_sweep:
            lines += ["## Pruning Sweep Results"]
            for model in model_order:
                lines += ["", f"### {model.upper()}"]
                header = ["Combo"] + pruning_tokens
                lines.append("| " + " | ".join(header) + " |")
                lines.append(_md_separator(len(header)))
                for combo in allowed_combos:
                    row = [combo]
                    for token in pruning_tokens:
                        match = next(
                            (
                                r
                                for r in results_rows
                                if r["family"] == "hq_mirror"
                                and r["reasoning_combo"] == combo
                                and r["model_type"] == model
                                and r.get("sweep_param") == token
                            ),
                            None,
                        )
                        row.append(_fmt(match["f0.5_mean"], match["f0.5_std"]) if match else "--")
                    lines.append("| " + " | ".join(row) + " |")
            if include_llm_engineered:
                lines += ["", "## Engineered Pruning Sweep Results"]
                for model in model_order:
                    lines += ["", f"### {model.upper()}"]
                    header = ["Combo"] + pruning_tokens
                    lines.append("| " + " | ".join(header) + " |")
                    lines.append(_md_separator(len(header)))
                    for combo in allowed_combos:
                        row = [combo]
                        for token in pruning_tokens:
                            match = next(
                                (
                                    r
                                    for r in results_rows
                                    if r["family"] == f"engineered_{engineered_set_id}"
                                    and r["reasoning_combo"] == combo
                                    and r["model_type"] == model
                                    and r.get("sweep_param") == token
                                ),
                                None,
                            )
                            row.append(_fmt(match["f0.5_mean"], match["f0.5_std"]) if match else "--")
                        lines.append("| " + " | ".join(row) + " |")

            lines += ["", "## Pruning Sweep CV vs Full-Train (per combo, same model)"]
            for model in model_order:
                header_cols = []
                for token in pruning_tokens:
                    header_cols.append(f"{token} CV")
                    header_cols.append(f"{token} Full")
                header = ["Combo"] + header_cols
                lines += ["", f"### {model.upper()} - HQ Mirror + Reasoning"]
                lines.append("| " + " | ".join(header) + " |")
                lines.append(_md_separator(len(header)))
                for combo in allowed_combos:
                    row_cells: list[str] = []
                    for token in pruning_tokens:
                        match = next(
                            (
                                r
                                for r in results_rows
                                if r["family"] == "hq_mirror"
                                and r["reasoning_combo"] == combo
                                and r["model_type"] == model
                                and r.get("sweep_param") == token
                            ),
                            None,
                        )
                        if match is None:
                            row_cells.extend(["--", "--"])
                        else:
                            row_cells.append(_fmt(match["f0.5_mean"], match["f0.5_std"]))
                            row_cells.append(f"{match['full_train_f0.5']:.3f}")
                    lines.append("| " + " | ".join([combo] + row_cells) + " |")
                if include_llm_engineered:
                    lines += ["", f"### {model.upper()} - Engineered {engineered_set_id} + Reasoning"]
                    lines.append("| " + " | ".join(header) + " |")
                    lines.append(_md_separator(len(header)))
                    for combo in allowed_combos:
                        row_cells = []
                        for token in pruning_tokens:
                            match = next(
                                (
                                    r
                                    for r in results_rows
                                    if r["family"] == f"engineered_{engineered_set_id}"
                                    and r["reasoning_combo"] == combo
                                    and r["model_type"] == model
                                    and r.get("sweep_param") == token
                                ),
                                None,
                            )
                            if match is None:
                                row_cells.extend(["--", "--"])
                            else:
                                row_cells.append(_fmt(match["f0.5_mean"], match["f0.5_std"]))
                                row_cells.append(f"{match['full_train_f0.5']:.3f}")
                        lines.append("| " + " | ".join([combo] + row_cells) + " |")

            if has_logistic:
                lines += ["", "## Selected Logistic Hyperparameters (HQ Mirror)"]
                header = ["Combo"] + pruning_tokens
                lines.append("| " + " | ".join(header) + " |")
                lines.append(_md_separator(len(header)))
                for combo in allowed_combos:
                    row = [combo]
                    for token in pruning_tokens:
                        match = next(
                            (
                                r
                                for r in results_rows
                                if r["family"] == "hq_mirror"
                                and r["reasoning_combo"] == combo
                                and r["model_type"] == "logistic"
                                and r.get("sweep_param") == token
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
        elif transform_upper in {"PCA", "PLS", "SFT"} and transform_sweep:
            lines += ["## Sweep Results (CV / Full)"]
            for model in model_order:
                lines += ["", f"### {model.upper()}"]
                header = ["Combo"]
                for v in variance_list:
                    token = str(v).replace(".", "p")
                    header.append(f"{token} CV")
                    header.append(f"{token} Full")
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
                        if match is None:
                            row.extend(["--", "--"])
                        else:
                            row.append(_fmt(match["f0.5_mean"], match["f0.5_std"]))
                            row.append(f"{match['full_train_f0.5']:.3f}")
                    lines.append("| " + " | ".join(row) + " |")
            if include_llm_engineered:
                lines += ["", "## Engineered Sweep Results (CV / Full)"]
                for model in model_order:
                    lines += ["", f"### {model.upper()}"]
                    header = ["Combo"]
                    for v in variance_list:
                        token = str(v).replace(".", "p")
                        header.append(f"{token} CV")
                        header.append(f"{token} Full")
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
                            if match is None:
                                row.extend(["--", "--"])
                            else:
                                row.append(_fmt(match["f0.5_mean"], match["f0.5_std"]))
                                row.append(f"{match['full_train_f0.5']:.3f}")
                        lines.append("| " + " | ".join(row) + " |")
            if has_logistic:
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
            if has_logistic:
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
            if has_mlp:
                mlp_models = [m for m in model_order if m in {"mlp32", "mlp4", "mlp2"}]
                lines += ["", "## Selected MLP Alpha (HQ Mirror)"]
                header = ["Combo"] + [m.upper() for m in mlp_models]
                lines.append("| " + " | ".join(header) + " |")
                lines.append(_md_separator(len(header)))
                for combo in allowed_combos:
                    row = [combo]
                    for model in mlp_models:
                        match = next(
                            (
                                r
                                for r in results_rows
                                if r["family"] == "hq_mirror"
                                and r["reasoning_combo"] == combo
                                and r["model_type"] == model
                            ),
                            None,
                        )
                        row.append(str(match.get("mlp_alpha")) if match else "--")
                    lines.append("| " + " | ".join(row) + " |")

        if sweep_df is not None and not sweep_df.empty:
            sweep_params = sorted({str(v) for v in sweep_df["sweep_param"].unique()})
            has_elasticnet = not sweep_df[
                (sweep_df["model_type"] == "elasticnet") & (sweep_df["transform"] == transform_upper)
            ].empty
            has_mlp_sweep = not sweep_df[
                (sweep_df["model_type"].isin(["mlp32", "mlp4", "mlp2"]))
                & (sweep_df["transform"] == transform_upper)
            ].empty
            if has_elasticnet:
                lines += ["", "## ElasticNet Hyperparameter Sweep (C × l1_ratio)"]
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
            if regularization_sweep == "mlp" and has_mlp_sweep:
                lines += ["", "## MLP Alpha Sweep"]
                mlp_models = [m for m in model_order if m in {"mlp32", "mlp4", "mlp2"}]
                for sweep_param in sweep_params:
                    label = "base" if sweep_param in {"", "None"} else sweep_param
                    for model in mlp_models:
                        lines += _build_mlp_sweep_tables(
                            sweep_df,
                            "hq_mirror",
                            transform_upper,
                            sweep_param,
                            allowed_combos,
                            model,
                            f"### HQ Mirror {model.upper()} (sweep={label})",
                        )
                        if include_llm_engineered:
                            lines += _build_mlp_sweep_tables(
                                sweep_df,
                                f"engineered_{engineered_set_id}",
                                transform_upper,
                                sweep_param,
                                allowed_combos,
                                model,
                                f"### Engineered {engineered_set_id} {model.upper()} (sweep={label})",
                            )
            if regularization_sweep != "all":
                lines += ["", "## Logistic C Sweep", "_Disabled (regularization_sweep != all)._"]
            else:
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
        if not finalising_experiments:
            if collinearity_report and col_summary_rows:
                col_path = report_dir / f"collinearity_summary{file_suffix_report}.csv"
                pd.DataFrame(col_summary_rows).to_csv(col_path, index=False)
                col_md_path = report_dir / f"collinearity_report{file_suffix_report}.md"
                col_md_lines = _build_collinearity_full_table(
                    col_summary_rows,
                    collinearity_corr_threshold,
                    all_models=collinearity_all_models,
                )
                col_md_path.write_text("\n".join(col_md_lines), encoding="utf-8")
            report_path = report_dir / f"model_testing_report{file_suffix_report}.md"
            if collinearity_report and col_summary_rows:
                lines += _build_collinearity_section(
                    col_summary_rows,
                    collinearity_corr_threshold,
                    all_models=collinearity_all_models,
                )
            report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            LOGGER.info("Report saved: %s", report_path)
            LOGGER.info("Results saved: %s", results_path)
            if collinearity_report and col_summary_rows:
                LOGGER.info("Collinearity summary saved: %s", col_path)
        else:
            LOGGER.info("Finalising mode: standard report skipped.")
            LOGGER.info("Results saved: %s", results_path)

        notes_path = OUTPUT_DIR / "model_testing_notes.md"
        notes_path.write_text(
            "# Model Testing Notes\n\n"
            "(2) Reduce collinearity: feature de-duplication, weighted quality scores.\n\n"
            "(3) Richer prompt outputs: add orthogonal signals beyond evidence/support ratings.\n\n"
            "(4) Calibration/thresholding: compare OOF tuning, Platt scaling, and cost-aware thresholds.\n",
            encoding="utf-8",
        )

        if not finalising_experiments:
            print(f"Report saved: {report_path}")
        print(f"Results saved: {results_path}")
        return results_rows, col_summary_rows
    exclude_modes = [False, True] if exclude_evidence_support_rating else [False]
    for exclude_evidence in exclude_modes:
        evidence_tag = "no_evidence_rating" if exclude_evidence else None
        base_combos = _build_combos_for_run(exclude_evidence, allowed_combos)
        pls_combos = (
            _build_combos_for_run(exclude_evidence, finalising_pls_combos_filtered)
            if finalising_experiments and finalising_pls_combos_filtered
            else base_combos
        )
        if exclude_evidence:
            LOGGER.info("Running pipeline with evidence_support_rating excluded from reasoning features.")
        finalising_results: dict[str, list[dict[str, Any]]] = {}
        finalising_col_rows: list[dict[str, Any]] = []
        for transform in transforms:
            print(f"Running transform: {transform}")
            combos = pls_combos if (finalising_experiments and transform.upper() == "PLS") else base_combos
            run_rows, run_col_rows = _run_for_transform(transform, combos, evidence_tag)
            if finalising_experiments:
                finalising_results[transform.upper()] = run_rows
                finalising_col_rows.extend(run_col_rows)
        if finalising_experiments:
            _write_finalising_report(
                finalising_results,
                finalising_col_rows,
                evidence_tag,
                prune_tag,
                include_llm_engineered,
                engineered_set_id,
                lr_only=generating_test_predictions,
            )

    if generating_test_predictions and preds_output is not None:
        missing_cols = [c for c in FINAL_PRED_COLUMNS.values() if c not in preds_output]
        if missing_cols:
            raise RuntimeError(f"Missing test predictions for columns: {missing_cols}")
        preds_path = run_root_dir / "model_testing_test_predictions.csv"
        pd.DataFrame(preds_output).to_csv(preds_path, index=False)
        LOGGER.info("Test predictions saved: %s", preds_path)


if __name__ == "__main__":
    main()

