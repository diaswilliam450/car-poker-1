"""Pull newly-released public benchmark chunks from the Poker44 platform API.

Docs (docs/training-benchmark.md): base ``https://api.poker44.net/api/v1``, no
auth. Endpoints used:
    GET /benchmark                      -> {latestSourceDate, autoRelease, ...}
    GET /benchmark/releases?limit=N     -> [{sourceDate, chunkCount, ...}, ...]
    GET /benchmark/chunks?sourceDate=YYYY-MM-DD&split=train|validation
        &limit=&cursor=                 -> paginated chunk payloads

Each new release is written as a labeled-list JSON
(``benchmark_<sourceDate>.json`` = ``[{chunk_id,is_bot,split,hands}, ...]``) into
``--out-dir`` (the retrain's ``--extra-dir``). Chunks already cached (by
``chunkHash``) are skipped, so this is idempotent and cheap to run daily.

This is BEST-EFFORT: if the API is unreachable or the schema differs, it logs
and exits 0 so the daily retrain proceeds on existing data. No secrets, no
side effects beyond writing under ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_BASE = os.getenv("POKER44_BENCHMARK_API", "https://api.poker44.net/api/v1")
_TIMEOUT = float(os.getenv("POKER44_BENCHMARK_TIMEOUT", "20"))


def _get(base: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = base.rstrip("/") + "/" + path.lstrip("/")
    resp = requests.get(url, params=params or {}, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _norm_chunk(raw: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
    hands = raw.get("hands") or raw.get("chunk") or raw.get("payload")
    if not isinstance(hands, list) or not hands:
        return None
    is_bot = raw.get("is_bot")
    if is_bot is None:
        is_bot = raw.get("label") in (1, "1", "bot", True)
    return {
        "chunk_id": raw.get("chunkId") or raw.get("chunk_id") or raw.get("chunkHash"),
        "chunk_hash": raw.get("chunkHash") or raw.get("chunk_hash"),
        "is_bot": bool(is_bot),
        "split": raw.get("split", split),
        "hands": hands,
    }


def _pull_release(base: str, source_date: str, seen: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for split in ("train", "validation"):
        cursor = None
        for _ in range(1000):  # hard page cap
            params = {"sourceDate": source_date, "split": split, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            page = _get(base, "benchmark/chunks", params)
            items = page.get("items", page) if isinstance(page, dict) else page
            if not items:
                break
            for raw in items:
                nc = _norm_chunk(raw, split)
                if not nc:
                    continue
                h = nc.get("chunk_hash") or nc.get("chunk_id")
                if h and h in seen:
                    continue
                if h:
                    seen.add(h)
                out.append(nc)
            cursor = page.get("cursor") or page.get("nextCursor") if isinstance(page, dict) else None
            if not cursor:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--max-releases", type=int, default=7, help="newest N releases to ensure locally")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_path = out_dir / "_seen_chunk_hashes.json"
    seen = set(json.loads(seen_path.read_text())) if seen_path.exists() else set()

    try:
        try:
            releases = _get(args.base, "benchmark/releases", {"limit": args.max_releases})
            dates = [r.get("sourceDate") for r in releases if r.get("sourceDate")]
        except Exception:
            root = _get(args.base, "benchmark")
            latest = root.get("latestSourceDate")
            dates = [latest] if latest else []
        dates = [d for d in dates if d][: args.max_releases]
        if not dates:
            print("[get_dump] no releases reported by API; nothing to pull.")
            return 0
        pulled_total = 0
        for d in dates:
            target = out_dir / f"benchmark_{d}.json"
            if target.exists():
                continue  # already have this release
            chunks = _pull_release(args.base, d, seen)
            if chunks:
                target.write_text(json.dumps(chunks), encoding="utf-8")
                pulled_total += len(chunks)
                print(f"[get_dump] release {d}: wrote {len(chunks)} chunks -> {target}")
        seen_path.write_text(json.dumps(sorted(seen)), encoding="utf-8")
        print(f"[get_dump] done: {pulled_total} new chunks across {len(dates)} releases.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[get_dump] API unavailable/failed ({type(exc).__name__}: {exc}); "
              f"retrain will proceed on existing data.")
        return 0


if __name__ == "__main__":
    time.sleep(0)  # keep import of time meaningful; no-op
    sys.exit(main())
