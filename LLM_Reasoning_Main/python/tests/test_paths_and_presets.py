from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.paths import DATA_DIR, DOCS_DIR, INSTABILITY_CONFIG_DIR, PRESETS_DIR, TMP_DIR
from lib.shared.presets import build_legacy_argv, legacy_wrapper_help_text


def test_core_paths_exist() -> None:
    assert DATA_DIR.exists()
    assert DOCS_DIR.exists()
    assert TMP_DIR.exists()
    assert PRESETS_DIR.exists()
    assert INSTABILITY_CONFIG_DIR.exists()


def test_model_testing_canonical_preset_builds_args() -> None:
    preset_name, argv = build_legacy_argv("model_testing", [])
    assert preset_name == "canonical"
    assert "--engineered_set_id" in argv
    assert "--test_csv" in argv
    assert "--test_reasoning_parquet" in argv


def test_legacy_wrapper_help_mentions_default_preset() -> None:
    help_text = legacy_wrapper_help_text("vcbench")
    assert "--preset PRESET" in help_text
    assert "vcbench_canonical.json" in help_text
