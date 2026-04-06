from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from lib.paths import VCBENCH_FOLDS_DIR


def resolve_folds_path(
    base_dir: Path, cv_folds: int, random_state: int, override_path: str | None
) -> Path:
    if override_path:
        path = Path(override_path)
        if not path.is_absolute():
            path = base_dir / path
        return path
    return VCBENCH_FOLDS_DIR / f"folds_k{cv_folds}_seed{random_state}.json"


def build_splits_from_fold_ids(
    fold_ids: np.ndarray, n_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_id in range(n_folds):
        test_idx = np.where(fold_ids == fold_id)[0]
        if test_idx.size == 0:
            continue
        train_idx = np.where(fold_ids != fold_id)[0]
        splits.append((train_idx, test_idx))
    return splits


def load_or_create_folds(
    founder_ids: list[str | None],
    labels: np.ndarray,
    cv_folds: int,
    random_state: int,
    folds_path: Path,
    dataset_label: str,
    use_fixed: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, Path]:
    if len(founder_ids) != len(labels):
        raise RuntimeError("Founder IDs and labels length mismatch.")

    str_ids: list[str] = []
    missing_count = 0
    for idx, fid in enumerate(founder_ids):
        if fid is None or str(fid).strip() == "":
            str_ids.append(f"row_{idx}")
            missing_count += 1
        else:
            str_ids.append(str(fid))

    if len(set(str_ids)) != len(str_ids):
        raise RuntimeError("Duplicate founder_uuid values detected; cannot build fixed folds.")

    if use_fixed and folds_path.exists():
        payload = json.loads(folds_path.read_text(encoding="utf-8"))
        if int(payload.get("k", -1)) != cv_folds:
            raise RuntimeError("Fold cache k mismatch; delete cache or update settings.")
        if int(payload.get("random_state", -1)) != random_state:
            raise RuntimeError("Fold cache random_state mismatch; delete cache or update settings.")
        fold_map = payload.get("fold_map", {})
        if not isinstance(fold_map, dict):
            raise RuntimeError("Fold cache has invalid format.")
        try:
            fold_ids = np.array([int(fold_map[str(fid)]) for fid in str_ids])
        except KeyError as exc:
            raise RuntimeError(
                "Fold cache missing founder_uuid entries; delete cache to rebuild."
            ) from exc
        splits = build_splits_from_fold_ids(fold_ids, cv_folds)
        return splits, fold_ids, folds_path

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_ids = np.full(len(labels), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        fold_ids[test_idx] = fold_idx
    if (fold_ids < 0).any():
        raise RuntimeError("Failed to assign fold IDs for all records.")

    if use_fixed:
        folds_path.parent.mkdir(parents=True, exist_ok=True)
        id_source = "founder_uuid" if missing_count == 0 else "row_index_fallback"
        payload: dict[str, Any] = {
            "k": cv_folds,
            "random_state": random_state,
            "dataset": dataset_label,
            "timestamp": datetime.now().isoformat(),
            "id_source": id_source,
            "missing_uuid_count": missing_count,
            "fold_map": {fid: int(fold_id) for fid, fold_id in zip(str_ids, fold_ids)},
        }
        folds_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    splits = build_splits_from_fold_ids(fold_ids, cv_folds)
    return splits, fold_ids, folds_path
