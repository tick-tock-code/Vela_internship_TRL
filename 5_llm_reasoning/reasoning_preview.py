"""Preview LLM reasoning outputs on first N train founders."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from think_reason_learn.datasets import load_vcbench
from sklearn.model_selection import train_test_split

from llm_reasoning_features import ReasoningConfig, generate_reasoning_features


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
        help="If >0, select this many founders per class (0/1) from the train split.",
    )
    p.add_argument(
        "--feature_config",
        default=str(Path(__file__).parent / "features.json"),
        help="Path to features.json (experiments + prompt paths).",
    )
    p.add_argument("--llm_model", default="gpt-4.1-nano")
    p.add_argument(
        "--core_prompt",
        default=str(Path(__file__).parent / "core_prompt.txt"),
        help="Override core prompt path.",
    )
    p.add_argument(
        "--experiments",
        default=str(Path(__file__).parent / "experiments.json"),
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
    env_path = Path(__file__).resolve().parents[1].parent / ".env"
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


def main() -> None:
    _load_env_if_present()
    args = _parse_args()
    env_path = Path(__file__).resolve().parents[1].parent / ".env"
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

    if not Path(core_prompt_path).is_absolute():
        core_prompt_path = str(Path(__file__).parent / core_prompt_path)
    if not Path(experiments_path).is_absolute():
        experiments_path = str(Path(__file__).parent / experiments_path)

    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        args.random_state,
    )

    idx = np.arange(len(records))
    train_idx, _ = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=labels,
        random_state=args.random_state,
    )
    train_recs = [records[i] for i in train_idx]
    train_labels = labels[train_idx]

    if args.per_class and args.per_class > 0:
        rng = np.random.RandomState(args.random_state)
        idx = np.arange(len(train_labels))
        rng.shuffle(idx)
        pos_idx = [i for i in idx if train_labels[i] == 1][: args.per_class]
        neg_idx = [i for i in idx if train_labels[i] == 0][: args.per_class]
        if len(pos_idx) < args.per_class or len(neg_idx) < args.per_class:
            raise RuntimeError(
                f"Not enough samples per class in train split: "
                f"pos={len(pos_idx)}, neg={len(neg_idx)}"
            )
        selected = neg_idx + pos_idx
        subset_recs = [train_recs[i] for i in selected]
        subset_labels = train_labels[selected]
    else:
        n = max(1, min(args.n_founders, len(train_recs)))
        subset_recs = train_recs[:n]
        subset_labels = train_labels[:n]

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

    output_dir = Path(__file__).parent / "features_storage" / "llm_reasoning" / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / f"preview_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    start_time = time.time()
    df, _ = generate_reasoning_features(
        records=subset_recs,
        labels=subset_labels,
        config=config,
        output_dir=output_dir,
        metadata_path=meta_path,
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

    log_dir = Path(__file__).parent / "training_logs" / "previews"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"reasoning_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Preview saved to: {out_path}")


if __name__ == "__main__":
    main()
