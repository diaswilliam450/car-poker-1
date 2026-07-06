"""Schema normalizer — turn both file formats into a common ``Chunk``.

Two on-disk shapes exist and must map to one internal type:

* **Benchmark (labeled)**: ``[{"is_bot": bool, "hands": [hand, ...], ...}, ...]``
* **Evaluation (unlabeled)**: ``[[hand, hand, ...], ...]`` — each chunk is a bare
  list of hand dicts (the canonical miner payload).

We never trust hand order across a chunk, but *within* a hand we sort actions by
integer ``action_id`` (the one ordering the design deems reliable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: int
    label: Optional[int]                 # 1=bot, 0=human, None=unlabeled
    hands: List[Dict[str, Any]] = field(default_factory=list)
    source_date: Optional[str] = None


def _sort_actions(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Return the hand with its actions sorted by integer action_id (stable)."""
    actions = hand.get("actions") or []
    if actions:
        def _key(a: Dict[str, Any]) -> int:
            try:
                return int(a.get("action_id", 0))
            except (TypeError, ValueError):
                return 0
        hand = dict(hand)
        hand["actions"] = sorted(actions, key=_key)
    return hand


def _coerce_hands(raw_hands: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_hands, list):
        return []
    return [_sort_actions(h) for h in raw_hands if isinstance(h, dict)]


def load_chunks(path: str | Path) -> List[Chunk]:
    """Load either file format into a list of ``Chunk`` (labels only if present)."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Top-level JSON must be a list, got {type(items).__name__}")

    chunks: List[Chunk] = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):                       # benchmark labeled form
            label = item.get("is_bot")
            chunks.append(
                Chunk(
                    chunk_id=idx,
                    label=int(bool(label)) if label is not None else None,
                    hands=_coerce_hands(item.get("hands")),
                    source_date=item.get("source_date"),
                )
            )
        elif isinstance(item, list):                     # evaluation bare-list form
            chunks.append(Chunk(chunk_id=idx, label=None, hands=_coerce_hands(item)))
        # anything else is skipped (defensive)
    return chunks


def is_labeled(chunks: List[Chunk]) -> bool:
    return any(c.label is not None for c in chunks)
