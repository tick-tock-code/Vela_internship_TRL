from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BASE_DIR.parent
CONFIG_DIR = BASE_DIR / "configs"
PROMPT_DIR = BASE_DIR / "prompts"
