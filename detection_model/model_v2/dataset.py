"""Build chunk-level feature matrices from either file format."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .features import chunk_feature_vector, feature_names_for
from .schema import Chunk, load_chunks


def build_feature_matrix(
    path: str | Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[str], List[int]]:
    """Return (X, y_or_None, feature_names, chunk_ids).

    ``y`` is None when the file carries no labels (evaluation set). Column order
    is the fixed :func:`feature_names_for` order so train and serve always align.
    """
    chunks: List[Chunk] = load_chunks(path)
    names = feature_names_for()
    rows: List[List[float]] = []
    labels: List[Optional[int]] = []
    ids: List[int] = []

    for ch in chunks:
        feats = chunk_feature_vector(ch.hands)
        rows.append([float(feats.get(name, 0.0)) for name in names])
        labels.append(ch.label)
        ids.append(ch.chunk_id)

    x = np.asarray(rows, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    has_labels = all(v is not None for v in labels) and len(labels) > 0
    y = np.asarray([int(v) for v in labels], dtype=np.int64) if has_labels else None
    return x, y, names, ids


def chunk_split(
    n: int, val_ratio: float = 0.2, seed: int = 44
) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic chunk-level train/val index split (never split by hand)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(val_ratio * n)))
    return order[n_val:], order[:n_val]
