from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.shared.presets import run_legacy_pipeline


if __name__ == "__main__":
    run_legacy_pipeline("reasoning_preview", "pipelines.legacy.reasoning_preview")
