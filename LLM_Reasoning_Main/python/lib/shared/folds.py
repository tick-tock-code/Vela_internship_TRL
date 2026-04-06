from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_fold_ids(frame: pd.DataFrame, folds_path: Path, *, id_column: str = "founder_uuid") -> np.ndarray:
    payload = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_map = payload.get("fold_map", {})
    if not isinstance(fold_map, dict):
        raise RuntimeError(f"Invalid fold cache format: {folds_path}")

    ids = frame[id_column].astype(str).tolist() if id_column in frame.columns else []
    has_id_mapping = bool(ids) and all(identifier in fold_map for identifier in ids)
    if has_id_mapping:
        return np.array([int(fold_map[identifier]) for identifier in ids], dtype=int)

    row_keys = [f"row_{int(idx)}" for idx in frame["row_index"].tolist()]
    if not all(key in fold_map for key in row_keys):
        raise RuntimeError(f"Fold cache missing row_index entries: {folds_path}")
    return np.array([int(fold_map[key]) for key in row_keys], dtype=int)
