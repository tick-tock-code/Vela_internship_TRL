from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.shared.artifact_io import read_json


def load_methods_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid methods config: {path}")
    return payload


def build_method_scaffold_summary(payload: dict[str, Any], priorities: dict[str, list[str]] | None = None) -> dict[str, Any]:
    active_methods = [item for item in payload.get("active_methods", []) if isinstance(item, dict)]
    available_methods = [item for item in payload.get("available_methods", []) if isinstance(item, dict)]
    return {
        "active_method_count": len(active_methods),
        "active_methods": active_methods,
        "available_methods": available_methods,
        "priorities": priorities or {},
    }


def method_scaffold_markdown(summary: dict[str, Any]) -> str:
    active_methods = summary.get("active_methods", [])
    available_methods = summary.get("available_methods", [])
    priorities = summary.get("priorities", {})
    lines = [
        "# Method Benchmark Scaffold",
        "",
        "This file records the active Step 2 method order and the current priority inputs.",
        "",
        f"- Active methods configured: {len(active_methods)}",
        "",
        "## Active Methods",
        "",
    ]
    for item in active_methods:
        lines.append(f"- `{item.get('id', 'unknown')}`: {item.get('notes', '')}".rstrip())
    lines += [
        "",
        "## Planned Order",
        "",
    ]
    for item in available_methods:
        lines.append(f"- `{item.get('id', 'unknown')}`: {item.get('notes', '')}".rstrip())
    lines += [
        "",
        "## Step 1 Priority Inputs",
        "",
        f"- Compression priority: {', '.join(priorities.get('compression_priority', [])) or '(none)'}",
        f"- Stability-selection priority: {', '.join(priorities.get('stability_selection_priority', [])) or '(none)'}",
        f"- Grouped-penalty priority: {', '.join(priorities.get('grouped_penalty_priority', [])) or '(none)'}",
    ]
    return "\n".join(lines)
