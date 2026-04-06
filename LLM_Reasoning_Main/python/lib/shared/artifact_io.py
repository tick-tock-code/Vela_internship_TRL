from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_markdown(path: Path, content: str) -> Path:
    ensure_dir(path.parent)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")
    return path


def timestamped_run_dir(root: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return ensure_dir(root / f"{stamp}_{label}")
