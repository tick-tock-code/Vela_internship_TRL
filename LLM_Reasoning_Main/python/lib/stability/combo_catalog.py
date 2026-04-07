from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lib.shared.artifact_io import read_json
from lib.stability.family_registry import AlignedFamilyData


@dataclass(frozen=True)
class ComboSpec:
    id: str
    label: str
    family_ids: list[str]
    notes: str = ""
    legacy_labels: dict[str, str] | None = None


@dataclass(frozen=True)
class EvaluationUnit:
    id: str
    label: str
    kind: str
    family_ids: list[str]
    feature_frame: pd.DataFrame
    feature_count: int
    notes: str = ""
    legacy_labels: dict[str, str] | None = None


def load_combo_catalog(config_path: Path) -> list[ComboSpec]:
    payload = read_json(config_path)
    combos = payload.get("combos", [])
    if not isinstance(combos, list):
        raise RuntimeError(f"Invalid combo catalog config: {config_path}")
    combo_specs: list[ComboSpec] = []
    for item in combos:
        if not isinstance(item, dict):
            continue
        legacy_payload = item.get("legacy_labels", {})
        legacy_labels = None
        if isinstance(legacy_payload, dict):
            legacy_labels = {str(key): str(value) for key, value in legacy_payload.items()}
        combo_specs.append(
            ComboSpec(
                id=str(item["id"]),
                label=str(item.get("label", item["id"])),
                family_ids=[str(value) for value in item.get("family_ids", [])],
                notes=str(item.get("notes", "")),
                legacy_labels=legacy_labels,
            )
        )
    return combo_specs


def _dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.duplicated()].copy()


def build_evaluation_units(data: AlignedFamilyData, combo_specs: list[ComboSpec]) -> list[EvaluationUnit]:
    units: list[EvaluationUnit] = [
        EvaluationUnit(
            id="HQ",
            label="HQ",
            kind="baseline",
            family_ids=[],
            feature_frame=pd.DataFrame(index=data.baseline.index),
            feature_count=0,
            notes="HQ-only route unit.",
            legacy_labels=data.baseline_spec.legacy_labels,
        )
    ]

    for family_id, frame in data.candidate_frames.items():
        spec = data.family_specs[family_id]
        units.append(
            EvaluationUnit(
                id=family_id,
                label=spec.label,
                kind="atomic_family",
                family_ids=[family_id],
                feature_frame=_dedupe_columns(frame),
                feature_count=int(frame.shape[1]),
                notes=spec.notes,
                legacy_labels=spec.legacy_labels,
            )
        )

    seen = {unit.id for unit in units}
    for combo in combo_specs:
        if combo.id in seen:
            raise RuntimeError(f"Duplicate evaluation unit id: {combo.id}")
        missing = [family_id for family_id in combo.family_ids if family_id not in data.candidate_frames]
        if missing:
            raise RuntimeError(f"Combo {combo.id} references unknown families: {missing}")
        combo_frame = pd.concat(
            [data.candidate_frames[family_id] for family_id in combo.family_ids],
            axis=1,
        )
        combo_frame = _dedupe_columns(combo_frame)
        units.append(
            EvaluationUnit(
                id=combo.id,
                label=combo.label,
                kind="legacy_combo",
                family_ids=list(combo.family_ids),
                feature_frame=combo_frame,
                feature_count=int(combo_frame.shape[1]),
                notes=combo.notes,
                legacy_labels=combo.legacy_labels,
            )
        )
        seen.add(combo.id)

    return units
