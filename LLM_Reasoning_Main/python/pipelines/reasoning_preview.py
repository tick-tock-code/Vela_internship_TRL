from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Preview LLM reasoning outputs on first N train founders."""


import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from think_reason_learn.datasets import load_vcbench

from lib.llm_reasoning_features import ReasoningConfig, generate_reasoning_features
from lib.paths import BASE_DIR, PROJECT_ROOT, CONFIG_DIR, PROMPT_DIR
from lib.cv_folds import load_or_create_folds, resolve_folds_path


def _resolve_input_csv(dataset: str, override: str) -> str:
    if override:
        return override
    base = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit"
    if dataset == "sample":
        return str(base / "vcbench_final_public_sample100.csv")
    return str(base / "vcbench_final_public.csv")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview LLM reasoning outputs for first N train founders.")
    p.add_argument("--dataset", choices=["sample", "full"], default="full")
    p.add_argument("--input_csv", default="", help="Optional override CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--n_founders", type=int, default=5)
    p.add_argument(
        "--per_class",
        type=int,
        default=0,
        help="If >0, select this many founders per class (0/1) from the pool.",
    )
    p.add_argument(
        "--feature_config",
        default=str(CONFIG_DIR / "features.json"),
        help="Path to features.json (experiments + prompt paths).",
    )
    p.add_argument("--llm_model", default="gpt-4.1-nano")
    p.add_argument(
        "--core_prompt",
        default=str(PROMPT_DIR / "core_prompt.txt"),
        help="Override core prompt path.",
    )
    p.add_argument(
        "--experiments",
        default=str(CONFIG_DIR / "experiments.json"),
        help="Override experiments JSON path.",
    )
    p.add_argument(
        "--use_api",
        action="store_true",
        help="Use live API calls (default is dry-run).",
    )
    return p.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_env_if_present() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from think_reason_learn.core import _config as trl_config
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
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
        return


def _generate_by_fold(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    config: ReasoningConfig,
    output_dir: Path,
    meta_path: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    numeric_keys: list[str] | None = None
    for fold_id in sorted(set(int(x) for x in fold_ids)):
        idx = np.where(fold_ids == fold_id)[0]
        if idx.size == 0:
            continue
        fold_records = [records[i] for i in idx]
        fold_labels = labels[idx]
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_meta = meta_path.with_name(meta_path.stem + f"_fold{fold_id}.json")
        fold_config = ReasoningConfig(
            model=config.model,
            dataset_size=config.dataset_size,
            random_state=config.random_state,
            core_prompt_path=config.core_prompt_path,
            experiments_path=config.experiments_path,
            providers=config.providers,
            google_model=config.google_model,
            batch_size=config.batch_size,
            concurrency=config.concurrency,
            experiments=config.experiments,
            dry_run=config.dry_run,
            dry_run_fast=config.dry_run_fast,
        )
        fold_df, fold_numeric = generate_reasoning_features(
            records=fold_records,
            labels=fold_labels,
            config=fold_config,
            output_dir=fold_dir,
            metadata_path=fold_meta,
        )
        if numeric_keys is None:
            numeric_keys = list(fold_numeric)
        elif set(fold_numeric) != set(numeric_keys):
            raise RuntimeError("Fold numeric keys mismatch in preview.")
        fold_df.insert(0, "__row_index__", idx)
        frames.append(fold_df)

    if not frames:
        raise RuntimeError("No preview outputs generated.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("__row_index__").drop(columns=["__row_index__"])
    return combined


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    env_path = PROJECT_ROOT / ".env"
    print(f".env path: {env_path} (exists={env_path.exists()})")
    print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
    input_csv = _resolve_input_csv(args.dataset, args.input_csv)

    cfg = _load_config(Path(args.feature_config))
    experiments = cfg.get("llm_reasoning_experiments", None)
    core_prompt_path = cfg.get("llm_reasoning_core_prompt_path", args.core_prompt)
    experiments_path = cfg.get("llm_reasoning_experiments_path", args.experiments)
    llm_providers = cfg.get("llm_providers", {"openai": True, "google": False})
    llm_google_model = cfg.get("llm_google_model", None)
    llm_reasoning_batch_size = cfg.get("llm_reasoning_batch_size", 20)
    llm_reasoning_concurrency = cfg.get("llm_reasoning_concurrency", 1)
    cv_folds = int(cfg.get("cv_folds", 10))
    cv_use_fixed_folds = bool(cfg.get("cv_use_fixed_folds", True))
    cv_folds_path = str(cfg.get("cv_folds_path", "") or "")
    seed_size = int(cfg.get("llm_engineered_seed_size", 100))

    if not Path(core_prompt_path).is_absolute():
        core_prompt_path = str(PROMPT_DIR / core_prompt_path)
    if not Path(experiments_path).is_absolute():
        experiments_path = str(CONFIG_DIR / experiments_path)

    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        args.random_state,
    )
    founder_ids = [r.get("founder_uuid") for r in records]
    seed_path = (
        BASE_DIR
        / "features_storage"
        / "llm_engineered"
        / f"seed_{seed_size}.json"
    )
    if seed_path.exists():
        try:
            seed_payload = _load_config(seed_path)
            seed_uuids = set(seed_payload.get("uuids", []))
        except Exception:
            seed_uuids = set()
        if seed_uuids:
            keep_mask = [fid not in seed_uuids for fid in founder_ids]
            records = [r for r, keep in zip(records, keep_mask) if keep]
            labels = labels[np.array(keep_mask)]
            founder_ids = [fid for fid, keep in zip(founder_ids, keep_mask) if keep]

    folds_path = resolve_folds_path(BASE_DIR, cv_folds, args.random_state, cv_folds_path)
    _, fold_ids, _ = load_or_create_folds(
        founder_ids=founder_ids,
        labels=labels,
        cv_folds=cv_folds,
        random_state=args.random_state,
        folds_path=folds_path,
        dataset_label=input_csv,
        use_fixed=cv_use_fixed_folds,
    )

    if args.per_class and args.per_class > 0:
        rng = np.random.RandomState(args.random_state)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        pos_idx = [i for i in idx if labels[i] == 1][: args.per_class]
        neg_idx = [i for i in idx if labels[i] == 0][: args.per_class]
        if len(pos_idx) < args.per_class or len(neg_idx) < args.per_class:
            raise RuntimeError(
                f"Not enough samples per class in pool: "
                f"pos={len(pos_idx)}, neg={len(neg_idx)}"
            )
        selected = neg_idx + pos_idx
        subset_recs = [records[i] for i in selected]
        subset_labels = labels[selected]
        subset_fold_ids = fold_ids[selected]
    else:
        n = max(1, min(args.n_founders, len(records)))
        subset_recs = records[:n]
        subset_labels = labels[:n]
        subset_fold_ids = fold_ids[:n]

    config = ReasoningConfig(
        model=args.llm_model,
        dataset_size="full",
        random_state=args.random_state,
        core_prompt_path=Path(core_prompt_path),
        experiments_path=Path(experiments_path),
        providers=llm_providers,
        google_model=llm_google_model,
        batch_size=llm_reasoning_batch_size,
        concurrency=llm_reasoning_concurrency,
        experiments=experiments,
        dry_run=not args.use_api,
        dry_run_fast=not args.use_api,
    )

    output_dir = BASE_DIR / "features_storage" / "llm_reasoning" / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"preview_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    start_time = time.time()
    df = _generate_by_fold(
        records=subset_recs,
        labels=subset_labels,
        fold_ids=subset_fold_ids,
        config=config,
        output_dir=output_dir,
        meta_path=meta_path,
    )
    elapsed = time.time() - start_time

    # Build preview output
    out_lines: list[str] = []
    out_lines.append(f"Total runtime (seconds): {elapsed:.2f}")
    out_lines.append("")
    for i, rec in enumerate(subset_recs):
        out_lines.append(f"Founder {i + 1}")
        out_lines.append("Founder info:")
        out_lines.append(json.dumps(rec, ensure_ascii=False, indent=2))
        out_lines.append("Outputs:")
        row = df.iloc[i].to_dict()
        out_lines.append(json.dumps(row, ensure_ascii=False, indent=2))
        out_lines.append("")

    log_dir = BASE_DIR / "training_logs" / "previews"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"reasoning_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Preview saved to: {out_path}")


if __name__ == "__main__":
    main()
