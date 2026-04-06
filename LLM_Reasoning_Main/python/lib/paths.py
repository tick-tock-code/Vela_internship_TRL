from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR.parent
CONFIG_DIR = BASE_DIR / "configs"
PROMPT_DIR = BASE_DIR / "prompts"
DOCS_DIR = BASE_DIR / "docs"
DATA_DIR = BASE_DIR / "data"
TMP_DIR = BASE_DIR / ".tmp"
LEGACY_PIPELINES_DIR = BASE_DIR / "python" / "pipelines" / "legacy"

PRESETS_DIR = CONFIG_DIR / "presets"
INSTABILITY_CONFIG_DIR = CONFIG_DIR / "instability_control"

RUNS_DIR = TMP_DIR / "runs"
VCBENCH_RUNS_DIR = RUNS_DIR / "vcbench"
PAPER_RUNS_DIR = RUNS_DIR / "paper"
MODEL_TESTING_RUNS_DIR = RUNS_DIR / "model_testing"
INSTABILITY_CONTROL_RUNS_DIR = RUNS_DIR / "instability_control"

VCBENCH_DATA_DIR = DATA_DIR / "vcbench"
VCBENCH_RAW_DIR = VCBENCH_DATA_DIR / "raw"
VCBENCH_FOLDS_DIR = VCBENCH_DATA_DIR / "folds"
VCBENCH_FEATURES_DIR = VCBENCH_DATA_DIR / "features"
VCBENCH_MANIFESTS_DIR = VCBENCH_DATA_DIR / "manifests"

VCBENCH_HUMAN_FEATURES_DIR = VCBENCH_FEATURES_DIR / "human"
VCBENCH_HQ_FEATURES_DIR = VCBENCH_FEATURES_DIR / "human_high_quality"
VCBENCH_COMBINED_FEATURES_DIR = VCBENCH_FEATURES_DIR / "combined"
VCBENCH_LLM_ENGINEERED_DIR = VCBENCH_FEATURES_DIR / "llm_engineered"
VCBENCH_LLM_ENGINEERED_ARCHIVES_DIR = VCBENCH_LLM_ENGINEERED_DIR / "archives"
VCBENCH_LLM_ENGINEERED_FAMILIES_DIR = VCBENCH_LLM_ENGINEERED_DIR / "families"
VCBENCH_LLM_REASONING_DIR = VCBENCH_FEATURES_DIR / "llm_reasoning"
VCBENCH_LLM_REASONING_TRAIN_DIR = VCBENCH_LLM_REASONING_DIR / "train"
VCBENCH_LLM_REASONING_TEST_DIR = VCBENCH_LLM_REASONING_DIR / "test"
VCBENCH_LLM_REASONING_TRAIN_COMBINED_DIR = VCBENCH_LLM_REASONING_TRAIN_DIR / "combined"
VCBENCH_LLM_REASONING_TEST_COMBINED_DIR = VCBENCH_LLM_REASONING_TEST_DIR / "combined"
VCBENCH_LLM_REASONING_RUNS_DIR = VCBENCH_LLM_REASONING_TRAIN_DIR / "runs"
VCBENCH_LLM_REASONING_CURRENT_DIR = VCBENCH_LLM_REASONING_TRAIN_COMBINED_DIR / "current"
VCBENCH_LLM_REASONING_CURRENTLY_IN_USE_DIR = (
    VCBENCH_LLM_REASONING_TRAIN_COMBINED_DIR / "currently_in_use"
)
VCBENCH_LLM_REASONING_FULL_CURRENT_DIR = (
    VCBENCH_LLM_REASONING_TRAIN_COMBINED_DIR / "full_current"
)

PAPER_STATS_DIR = DOCS_DIR / "paper_stats"
MODEL_TESTING_DOCS_DIR = DOCS_DIR / "model_testing"
INSTABILITY_CONTROL_DOCS_DIR = DOCS_DIR / "instability_control"
ARCHIVE_DOCS_DIR = DOCS_DIR / "archive"
VCBENCH_LOGGING_DIR = VCBENCH_RUNS_DIR / "logging"
VCBENCH_PREVIEWS_DIR = VCBENCH_RUNS_DIR / "previews"

DEPRECATED_FEATURES_STORAGE_DIR = BASE_DIR / "features_storage"
DEPRECATED_TEST_DATASET_DIR = BASE_DIR / "test_dataset"
DEPRECATED_TRAINING_LOGS_DIR = BASE_DIR / "training_logs"
DEPRECATED_LOGGING_DIR = BASE_DIR / "logging"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing(*paths: Path) -> Path:
    if not paths:
        raise ValueError("resolve_existing requires at least one path.")
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def raw_private_csv_path() -> Path:
    name = "vcbench_final_private (success column removed) - vcbench_final_private.csv"
    return resolve_existing(
        VCBENCH_RAW_DIR / name,
        DEPRECATED_TEST_DATASET_DIR / name,
    )


def raw_public_csv_path() -> Path:
    return resolve_existing(
        VCBENCH_RAW_DIR / "vcbench_final_public.csv",
        PROJECT_ROOT / "VCBench-Starter-Kit" / "vcbench_final_public.csv",
    )


def raw_public_sample_csv_path() -> Path:
    return resolve_existing(
        VCBENCH_RAW_DIR / "vcbench_final_public_sample100.csv",
        PROJECT_ROOT / "VCBench-Starter-Kit" / "vcbench_final_public_sample100.csv",
    )


def fold_cache_path(name: str) -> Path:
    return resolve_existing(
        VCBENCH_FOLDS_DIR / name,
        DEPRECATED_FEATURES_STORAGE_DIR / "cv_folds" / name,
    )


def engineered_seed_path(seed_size: int) -> Path:
    name = f"seed_{seed_size}.json"
    return resolve_existing(
        VCBENCH_LLM_ENGINEERED_DIR / name,
        DEPRECATED_FEATURES_STORAGE_DIR / "llm_engineered" / name,
    )
