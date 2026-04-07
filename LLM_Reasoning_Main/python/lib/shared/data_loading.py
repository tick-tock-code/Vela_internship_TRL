from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


NON_FEATURE_COLUMNS = {"founder_uuid", "success", "row_index", "set_id"}


@dataclass(frozen=True)
class LoadedFrame:
    path: Path
    frame: pd.DataFrame
    feature_names: list[str]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def prepare_frame(
    frame: pd.DataFrame,
    *,
    id_column: str = "founder_uuid",
    label_column: str = "success",
    allow_missing_label: bool = False,
) -> pd.DataFrame:
    prepared = frame.copy()
    if "row_index" not in prepared.columns:
        prepared = prepared.reset_index(drop=True)
        prepared["row_index"] = prepared.index
    if id_column not in prepared.columns:
        prepared[id_column] = prepared["row_index"].astype(str)
    if label_column not in prepared.columns:
        if not allow_missing_label:
            raise KeyError(f"Missing label column '{label_column}'.")
        prepared[label_column] = 0
    return prepared


def select_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
    feature_prefixes: Iterable[str] | None = None,
    exclude_columns: Iterable[str] | None = None,
) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.update(exclude_columns or [])
    if feature_columns:
        return [col for col in feature_columns if col in frame.columns and col not in excluded]
    if feature_prefixes:
        prefixes = tuple(str(prefix) for prefix in feature_prefixes)
        cols = [
            col
            for col in frame.columns
            if col.startswith(prefixes)
            and col not in excluded
            and pd.api.types.is_numeric_dtype(frame[col])
        ]
        return cols
    return [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]


def load_feature_frame(
    path: Path,
    *,
    id_column: str = "founder_uuid",
    label_column: str = "success",
    allow_missing_label: bool = False,
    feature_columns: Iterable[str] | None = None,
    feature_prefixes: Iterable[str] | None = None,
    exclude_columns: Iterable[str] | None = None,
) -> LoadedFrame:
    frame = prepare_frame(
        read_table(path),
        id_column=id_column,
        label_column=label_column,
        allow_missing_label=allow_missing_label,
    )
    features = select_feature_columns(
        frame,
        feature_columns=feature_columns,
        feature_prefixes=feature_prefixes,
        exclude_columns=exclude_columns,
    )
    keep_columns = ["row_index", id_column, label_column] + features
    deduped = frame[keep_columns].copy()
    if id_column in deduped.columns:
        deduped = deduped.drop_duplicates(subset=[id_column, "row_index"])
    return LoadedFrame(path=path, frame=deduped, feature_names=features)


def align_on_common_rows(
    loaded_frames: dict[str, LoadedFrame],
    *,
    id_column: str = "founder_uuid",
    label_column: str = "success",
) -> tuple[str, dict[str, pd.DataFrame]]:
    if not loaded_frames:
        raise ValueError("No frames provided for alignment.")

    use_id_column = all(
        id_column in loaded.frame.columns
        and loaded.frame[id_column].notna().all()
        and loaded.frame[id_column].is_unique
        for loaded in loaded_frames.values()
    )
    join_key = id_column if use_id_column else "row_index"

    aligned: dict[str, pd.DataFrame] = {}
    common_index = None
    for family_id, loaded in loaded_frames.items():
        frame = loaded.frame.copy().set_index(join_key)
        if common_index is None:
            common_index = frame.index
        else:
            common_index = common_index.intersection(frame.index)
        aligned[family_id] = frame

    if common_index is None:
        raise RuntimeError("Failed to establish common alignment index.")

    for family_id, frame in aligned.items():
        family_frame = frame.loc[common_index].sort_index().copy()
        if label_column in family_frame.columns:
            family_frame[label_column] = family_frame[label_column].astype(int)
        aligned[family_id] = family_frame
    return join_key, aligned
