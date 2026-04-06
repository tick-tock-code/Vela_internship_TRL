from __future__ import annotations

import numpy as np


def threshold_scores(scores, threshold: float = 0.5) -> np.ndarray:
    """Convert probability-like scores into a binary prediction vector."""
    arr = np.asarray(scores, dtype=float)
    return (arr >= threshold).astype(int)


def top_k_mask(scores, k: int) -> np.ndarray:
    """Return a binary mask for the top-k scores, preserving input order."""
    if k <= 0:
        raise ValueError("k must be positive")
    arr = np.asarray(scores, dtype=float)
    if k >= arr.size:
        return np.ones(arr.size, dtype=int)
    mask = np.zeros(arr.size, dtype=int)
    top_idx = np.argpartition(arr, -k)[-k:]
    mask[top_idx] = 1
    return mask
