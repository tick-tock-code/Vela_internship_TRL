from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lib.paths import BASE_DIR
from lib.shared.artifact_io import read_json
from lib.shared.data_loading import LoadedFrame, align_on_common_rows, load_feature_frame


@dataclass(frozen=True)
class DatasetSpec:
    id_column: str
    label_column: str
    folds_path: Path


@dataclass(frozen=True)
class FamilySpec:
    id: str
    label: str
    path: Path
    feature_columns: list[str] | None = None
    feature_prefixes: list[str] | None = None
    exclude_columns: list[str] | None = None
    kind: str = "candidate"
    notes: str = ""


@dataclass(frozen=True)
class FamilyRegistry:
    dataset: DatasetSpec
    baseline: FamilySpec
    families: list[FamilySpec]


@dataclass(frozen=True)
class AlignedFamilyData:
    join_key: str
    labels: pd.Series
    baseline: pd.DataFrame
    candidate_frames: dict[str, pd.DataFrame]
    family_specs: dict[str, FamilySpec]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _family_from_payload(payload: dict[str, object]) -> FamilySpec:
    return FamilySpec(
        id=str(payload["id"]),
        label=str(payload.get("label", payload["id"])),
        path=_resolve_path(str(payload["path"])),
        feature_columns=list(payload.get("feature_columns", []) or []) or None,
        feature_prefixes=list(payload.get("feature_prefixes", []) or []) or None,
        exclude_columns=list(payload.get("exclude_columns", []) or []) or None,
        kind=str(payload.get("kind", "candidate")),
        notes=str(payload.get("notes", "")),
    )


def load_family_registry(config_path: Path) -> FamilyRegistry:
    payload = read_json(config_path)
    dataset = payload.get("dataset", {})
    baseline = payload.get("baseline")
    families = payload.get("families", [])
    if not isinstance(dataset, dict) or not isinstance(baseline, dict) or not isinstance(families, list):
        raise RuntimeError(f"Invalid family registry config: {config_path}")
    dataset_spec = DatasetSpec(
        id_column=str(dataset.get("id_column", "founder_uuid")),
        label_column=str(dataset.get("label_column", "success")),
        folds_path=_resolve_path(str(dataset["folds_path"])),
    )
    baseline_spec = _family_from_payload(baseline)
    family_specs = [_family_from_payload(item) for item in families if isinstance(item, dict)]
    return FamilyRegistry(dataset=dataset_spec, baseline=baseline_spec, families=family_specs)


def _load_spec(spec: FamilySpec, dataset: DatasetSpec) -> LoadedFrame:
    return load_feature_frame(
        spec.path,
        id_column=dataset.id_column,
        label_column=dataset.label_column,
        feature_columns=spec.feature_columns,
        feature_prefixes=spec.feature_prefixes,
        exclude_columns=spec.exclude_columns,
    )


def load_aligned_family_data(registry: FamilyRegistry) -> AlignedFamilyData:
    loaded_specs: dict[str, LoadedFrame] = {
        registry.baseline.id: _load_spec(registry.baseline, registry.dataset)
    }
    for spec in registry.families:
        loaded_specs[spec.id] = _load_spec(spec, registry.dataset)

    join_key, aligned_frames = align_on_common_rows(
        loaded_specs,
        id_column=registry.dataset.id_column,
        label_column=registry.dataset.label_column,
    )
    baseline_frame = aligned_frames[registry.baseline.id]
    labels = baseline_frame[registry.dataset.label_column].astype(int)
    baseline_features = baseline_frame[loaded_specs[registry.baseline.id].feature_names].copy()

    candidate_frames: dict[str, pd.DataFrame] = {}
    family_specs = {spec.id: spec for spec in registry.families}
    for spec in registry.families:
        candidate_frames[spec.id] = aligned_frames[spec.id][loaded_specs[spec.id].feature_names].copy()

    return AlignedFamilyData(
        join_key=join_key,
        labels=labels,
        baseline=baseline_features,
        candidate_frames=candidate_frames,
        family_specs=family_specs,
    )
