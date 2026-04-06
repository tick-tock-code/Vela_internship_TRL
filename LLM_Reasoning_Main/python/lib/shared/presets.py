from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from lib.paths import BASE_DIR, PRESETS_DIR

_PATH_KEYS = {
    "feature_config",
    "test_csv",
    "test_reasoning_parquet",
}


def _preset_path(pipeline_name: str, preset_name: str) -> Path:
    return PRESETS_DIR / f"{pipeline_name}_{preset_name}.json"


def _load_preset_args(pipeline_name: str, preset_name: str) -> dict[str, Any]:
    path = _preset_path(pipeline_name, preset_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing preset for {pipeline_name}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise RuntimeError(f"Invalid preset format: {path}")
    return args


def _coerce_cli_value(key: str, value: Any) -> list[str]:
    flag = f"--{key}"
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, (list, tuple)):
        return [flag, ",".join(str(item) for item in value)]
    text = str(value)
    if key in _PATH_KEYS:
        path = Path(text)
        if not path.is_absolute():
            text = str((BASE_DIR / path).resolve())
    return [flag, text]


def _extract_preset_name(argv: list[str]) -> tuple[str, list[str]]:
    preset_name = "canonical"
    cleaned: list[str] = []
    skip = False
    for idx, token in enumerate(argv):
        if skip:
            skip = False
            continue
        if token == "--preset":
            if idx + 1 >= len(argv):
                raise RuntimeError("--preset requires a value.")
            preset_name = argv[idx + 1]
            skip = True
            continue
        if token.startswith("--preset="):
            preset_name = token.split("=", 1)[1]
            continue
        cleaned.append(token)
    return preset_name, cleaned


def build_legacy_argv(pipeline_name: str, argv: list[str]) -> tuple[str, list[str]]:
    preset_name, cleaned = _extract_preset_name(argv)
    preset_args: list[str] = []
    for key, value in _load_preset_args(pipeline_name, preset_name).items():
        preset_args.extend(_coerce_cli_value(key, value))
    return preset_name, preset_args + cleaned


def legacy_wrapper_help_text(pipeline_name: str, preset_name: str = "canonical") -> str:
    preset_path = _preset_path(pipeline_name, preset_name)
    return (
        f"usage: {pipeline_name}_pipeline.py [--preset PRESET] [pipeline args...]\n\n"
        f"Legacy pipeline wrapper for '{pipeline_name}'.\n"
        f"Default preset: {preset_name} ({preset_path})\n"
        "Behavior:\n"
        "  - running with no args loads the canonical preset\n"
        "  - explicit CLI flags override preset values\n"
        "  - remaining args are forwarded to the frozen legacy implementation\n"
    )


def run_legacy_pipeline(pipeline_name: str, module_name: str) -> None:
    if any(token in {"-h", "--help"} for token in sys.argv[1:]):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            print(legacy_wrapper_help_text(pipeline_name))
            print(
                f"[legacy pipeline] detailed parser help unavailable because dependency "
                f"'{exc.name}' is not installed in this Python environment."
            )
            return
        sys.argv = [sys.argv[0], "--help"]
        module.main()
        return

    preset_name, argv = build_legacy_argv(pipeline_name, sys.argv[1:])
    print(f"[legacy pipeline] {pipeline_name} preset={preset_name}")
    sys.argv = [sys.argv[0]] + argv
    module = importlib.import_module(module_name)
    module.main()
