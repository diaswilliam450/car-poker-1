"""Build a large, balanced labeled training set for the v2 chunk classifier.

The shipped public benchmark is only 40 chunks — too small to train a
non-degenerate model on its own only because of *hyperparameters*; with proper
regularization those 40 real chunks generalize (5-fold CV reward ~1.0). This
builder grows that real signal into a robust training set:

  * BENCHMARK (real): the 40 public benchmark chunks (real reference bots +
    humans) are the PRIMARY, ground-truth signal and are always included.
  * BOOTSTRAP augmentation: extra chunks resampled (with replacement) from the
    benchmark bot-hand pool / human-hand pool — same distribution, more volume,
    so the model sees the real behaviour at many chunk sizes.
  * CORPUS humans: chunks sampled from the real 32k-hand human corpus — adds
    human diversity and false-positive safety (neutral-to-helpful in CV).
  * SYNTHETIC bots (small, optional): rigid-policy bot chunks
    (``tools/bot_synthesizer.py``) as *robustness insurance* against bot
    behaviours the public benchmark doesn't cover. Kept small because a large
    synthetic bot mass hurts held-out benchmark reward (structural mismatch).
  * DAILY pulls: any real labeled chunks fetched by ``tools/get_dump.py`` into
    ``--extra-dir`` (new platform releases) are folded in as first-class data.

Output is the plain-JSON list of ``{chunk_id, split, is_bot, hands}`` that
``model_v2.train`` consumes. Date-seedable so the daily retrain produces fresh
data even when the platform API has no new release. The FROZEN n-gram vocab is
built once (or on ``--build-vocab``); feature stability across retrains depends
on it staying fixed.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve()
_REPO_DM = _HERE.parent.parent           # detection_model/
sys.path.insert(0, str(_REPO_DM.parent))  # repo root so detection_model.* imports work

from detection_model.tools.bot_synthesizer import ARCHETYPES, generate_bot_chunk  # noqa: E402
from detection_model.model_v2 import ngram_features as ng  # noqa: E402


def _open_maybe_gz(path: str) -> Any:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_archetypes(spec: str) -> List[tuple]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, w = part.partition(":")
        if name.strip() in ARCHETYPES:
            out.append((name.strip(), float(w) if w else 1.0))
    return out or [(a, 1.0) for a in ARCHETYPES]


def _weighted_pick(rng: random.Random, weighted: List[tuple]) -> str:
    total = sum(w for _, w in weighted)
    r = rng.random() * total
    acc = 0.0
    for name, w in weighted:
        acc += w
        if r <= acc:
            return name
    return weighted[-1][0]


def _bootstrap(rng: random.Random, pool: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return [rng.choice(pool) for _ in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output labeled-list JSON")
    ap.add_argument("--human-corpus", default=str(
        _REPO_DM.parent / "hands_generator" / "human_hands" / "poker_hands_combined.json.gz"))
    ap.add_argument("--benchmark", default=str(_REPO_DM / "public_miner_benchmark.json.gz"))
    ap.add_argument("--extra-dir", default="", help="dir of extra labeled-list JSONs (daily pulls)")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--bench-bot-aug", type=int, default=120, help="bootstrap bot chunks from benchmark pool")
    ap.add_argument("--bench-hum-aug", type=int, default=120, help="bootstrap human chunks from benchmark pool")
    ap.add_argument("--n-corpus-human", type=int, default=100, help="human chunks sampled from the 32k corpus")
    ap.add_argument("--n-synth-bot", type=int, default=0, help="synthetic bot chunks (robustness insurance)")
    ap.add_argument("--synth-archetypes", default="nit:1,aggro:1,station:1,cbot:1")
    ap.add_argument("--min-hands", type=int, default=60)
    ap.add_argument("--max-hands", type=int, default=120)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--build-vocab", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks: List[Dict[str, Any]] = []
    bot_pool: List[Dict[str, Any]] = []
    hum_pool: List[Dict[str, Any]] = []

    # --- benchmark real chunks (primary signal) ------------------------------
    try:
        bench = _open_maybe_gz(args.benchmark)
        for c in bench.get("labeled_chunks", []):
            is_bot = bool(c.get("is_bot"))
            hands = c.get("hands") or []
            chunks.append({"chunk_id": "bench_" + str(c.get("chunk_id")), "is_bot": is_bot, "hands": hands})
            (bot_pool if is_bot else hum_pool).extend(h for h in hands if isinstance(h, dict))
    except Exception as exc:  # noqa: BLE001
        print(f"[build_dataset] benchmark load skipped: {exc}")

    # --- daily-pulled real chunks (also feed the bootstrap pools) ------------
    if args.extra_dir and os.path.isdir(args.extra_dir):
        for fp in sorted(glob.glob(os.path.join(args.extra_dir, "*.json"))):
            try:
                for c in _open_maybe_gz(fp):
                    if isinstance(c, dict) and "hands" in c:
                        is_bot = bool(c.get("is_bot"))
                        hands = c.get("hands") or []
                        chunks.append({"chunk_id": "extra_" + str(c.get("chunk_id", Path(fp).stem)),
                                       "is_bot": is_bot, "hands": hands})
                        (bot_pool if is_bot else hum_pool).extend(h for h in hands if isinstance(h, dict))
            except Exception as exc:  # noqa: BLE001
                print(f"[build_dataset] extra file {fp} skipped: {exc}")

    # --- bootstrap augmentation from the real pools --------------------------
    if bot_pool:
        for i in range(args.bench_bot_aug):
            chunks.append({"chunk_id": f"botaug_{i}", "is_bot": True,
                           "hands": _bootstrap(rng, bot_pool, rng.randint(args.min_hands, args.max_hands))})
    if hum_pool:
        for i in range(args.bench_hum_aug):
            chunks.append({"chunk_id": f"humaug_{i}", "is_bot": False,
                           "hands": _bootstrap(rng, hum_pool, rng.randint(args.min_hands, args.max_hands))})

    # --- corpus human chunks (diversity + FPR safety) ------------------------
    if args.n_corpus_human:
        corpus = _open_maybe_gz(args.human_corpus)
        human_hands = [h for h in corpus if isinstance(h, dict) and h.get("actions")]
        for i in range(args.n_corpus_human):
            chunks.append({"chunk_id": f"corpus_{i}", "is_bot": False,
                           "hands": [rng.choice(human_hands) for _ in range(rng.randint(args.min_hands, args.max_hands))]})

    # --- synthetic bot robustness component (small) --------------------------
    if args.n_synth_bot:
        weighted = _parse_archetypes(args.synth_archetypes)
        for i in range(args.n_synth_bot):
            arche = _weighted_pick(rng, weighted)
            chunks.append({"chunk_id": f"synth_{arche}_{i}", "is_bot": True,
                           "hands": generate_bot_chunk(rng, arche, rng.randint(args.min_hands, args.max_hands))})

    # --- deterministic stratified chunk-level split --------------------------
    rng.shuffle(chunks)
    bots = [c for c in chunks if c["is_bot"]]
    humans = [c for c in chunks if not c["is_bot"]]

    def assign(group: List[Dict[str, Any]]):
        n_val = int(round(args.val_ratio * len(group)))
        for j, c in enumerate(group):
            c["split"] = "validation" if j < n_val else "train"

    assign(bots)
    assign(humans)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks), encoding="utf-8")
    print(f"[build_dataset] wrote {len(chunks)} chunks -> {out_path} "
          f"(bot={len(bots)} human={len(humans)}, "
          f"val={sum(1 for c in chunks if c['split']=='validation')}) "
          f"[bench_pool bot={len(bot_pool)} hum={len(hum_pool)} hands]")

    # --- freeze the n-gram vocab (once) --------------------------------------
    vocab_path = Path(ng.__file__).with_name(ng._VOCAB_FILENAME)
    if args.build_vocab or not vocab_path.exists():
        size = ng.write_vocab([c["hands"] for c in chunks])
        print(f"[build_dataset] froze n-gram vocab: {size} patterns -> {vocab_path}")
    else:
        print(f"[build_dataset] keeping existing frozen vocab ({len(ng.load_vocab())} patterns)")


if __name__ == "__main__":
    main()
