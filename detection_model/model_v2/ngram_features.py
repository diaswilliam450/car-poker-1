"""Behavioral n-gram features for the v2 chunk classifier.

Each poker action is turned into a short "word" token that encodes *how* the
player acted, then the per-hand token stream is expanded into 1/2/3-grams
(short ordered patterns).  A chunk's feature is the average per-hand count of
each pattern in a FROZEN vocabulary, so the same behavioural fingerprint is
comparable across chunks of different sizes.

Token layout  (street + action + size-bucket), e.g. ``fBm`` = "medium bet on
the flop":

    street  : p=preflop f=flop t=turn r=river s=showdown x=other
    action  : F=fold C=call R=raise K=check B=bet  b=blind  X=other
    bucket  : 0=no amount  s=small  m=medium  p=pot-sized  o=overbet  (pot-relative)

Optional position tokens ``posN<act>`` capture "seat N did <act>" independent
of street.  These sequence fingerprints separate rigid bots (they repeat the
same lines every hand) from humans (who mix lines), and generalise to live
tables far better than raw table statistics.

The vocabulary is generated ONCE from the training corpus (``build_vocab`` →
``ngram_vocab.json``) and then frozen — retraining never rebuilds it, so the
served ``feature_names`` stay byte-stable across model versions.  If the vocab
file is absent the module contributes zero features (base pipeline unchanged).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Per-miner configuration.  DIFFERENTIATION LEVER: each of the 4 miners ships a
# different NGRAM_CONFIG so their behavioural feature spaces (and therefore
# their models + artifacts) genuinely differ.
# --------------------------------------------------------------------------- #
NGRAM_CONFIG: Dict[str, Any] = {
    "orders": (1, 2, 3),       # which n-gram orders to emit
    "include_position": True,  # add posN<act> tokens
    "top_k": {1: 80, 2: 160, 3: 90},  # max patterns kept per order in the vocab
}

_VOCAB_FILENAME = "ngram_vocab.json"

_ACTION_CODES = {
    "fold": "F",
    "call": "C",
    "raise": "R",
    "check": "K",
    "bet": "B",
    "small_blind": "b",
    "big_blind": "b",
}

_STREET_CODES = {
    "preflop": "p",
    "flop": "f",
    "turn": "t",
    "river": "r",
    "showdown": "s",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _size_bucket(action: Dict[str, Any]) -> str:
    """Pot-relative bet-size bucket: 0/s/m/p/o."""
    amount = _num(action.get("amount"), 0.0)
    if amount <= 0.0:
        return "0"
    pot = _num(action.get("pot_before"), 0.0)
    if pot <= 0.0:
        # fall back to a big-blind-relative view when pot is unknown
        bb = _num(action.get("normalized_amount_bb"), 0.0)
        if bb <= 0:
            return "0"
        if bb < 2.0:
            return "s"
        if bb < 5.0:
            return "m"
        if bb < 10.0:
            return "p"
        return "o"
    ratio = amount / pot
    if ratio < 0.4:
        return "s"
    if ratio < 0.75:
        return "m"
    if ratio <= 1.15:
        return "p"
    return "o"


def _action_token(action: Dict[str, Any]) -> str:
    street = _STREET_CODES.get(str(action.get("street") or "").lower(), "x")
    act = _ACTION_CODES.get(str(action.get("action_type") or "").lower(), "X")
    bucket = _size_bucket(action)
    return street + act + bucket


def _hand_tokens(hand: Dict[str, Any], include_position: bool) -> List[str]:
    actions = hand.get("actions") or []
    tokens: List[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        act_code = _ACTION_CODES.get(str(action.get("action_type") or "").lower(), "X")
        if act_code == "b":  # forced blinds carry no behavioural signal
            continue
        tokens.append(_action_token(action))
        if include_position:
            seat = action.get("actor_seat")
            if seat is not None:
                tokens.append(f"pos{seat}{act_code}")
    return tokens


def _hand_ngram_doc(hand: Dict[str, Any], orders: Tuple[int, ...], include_position: bool) -> Counter:
    """Count every n-gram (for the requested orders) in one hand."""
    tokens = _hand_tokens(hand, include_position)
    grams: Counter = Counter()
    n = len(tokens)
    for order in orders:
        if order == 1:
            grams.update(tokens)
            continue
        for i in range(n - order + 1):
            grams["|".join(tokens[i : i + order])] += 1
    return grams


# --------------------------------------------------------------------------- #
# Vocabulary handling (frozen at build time).
# --------------------------------------------------------------------------- #
_VOCAB_CACHE: Optional[List[str]] = None


def _vocab_path() -> Path:
    return Path(__file__).with_name(_VOCAB_FILENAME)


def _sanitize(token: str) -> str:
    return token.replace("|", "__")


def load_vocab() -> List[str]:
    """Load the frozen vocabulary (cached).  Empty list if not built yet."""
    global _VOCAB_CACHE
    if _VOCAB_CACHE is not None:
        return _VOCAB_CACHE
    path = _vocab_path()
    if not path.exists():
        _VOCAB_CACHE = []
        return _VOCAB_CACHE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _VOCAB_CACHE = list(data.get("vocab", []))
    except Exception:
        _VOCAB_CACHE = []
    return _VOCAB_CACHE


def ngram_feature_names() -> List[str]:
    """Column names contributed by the n-gram block (order == vocab order)."""
    return ["schema_ngram_" + _sanitize(tok) for tok in load_vocab()]


def ngram_chunk_features(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average per-hand count of each vocabulary n-gram across the chunk."""
    vocab = load_vocab()
    if not vocab:
        return {}
    orders = tuple(NGRAM_CONFIG["orders"])
    include_position = bool(NGRAM_CONFIG["include_position"])
    totals: Counter = Counter()
    n = max(1, len(hands))
    for hand in hands:
        if isinstance(hand, dict):
            totals.update(_hand_ngram_doc(hand, orders, include_position))
    out: Dict[str, float] = {}
    for tok in vocab:
        out["schema_ngram_" + _sanitize(tok)] = float(totals.get(tok, 0.0)) / n
    return out


def build_vocab(chunks: List[List[Dict[str, Any]]]) -> List[str]:
    """Scan a corpus of chunks and return the frozen top-k n-gram vocabulary.

    ``chunks`` is a list of hand-lists (labels irrelevant here).  The most
    frequent grams per order are kept, capped by ``NGRAM_CONFIG['top_k']``.
    """
    orders = tuple(NGRAM_CONFIG["orders"])
    include_position = bool(NGRAM_CONFIG["include_position"])
    top_k = NGRAM_CONFIG["top_k"]
    per_order: Dict[int, Counter] = {o: Counter() for o in orders}
    for hands in chunks:
        for hand in hands:
            if not isinstance(hand, dict):
                continue
            doc = _hand_ngram_doc(hand, orders, include_position)
            for tok, cnt in doc.items():
                order = 1 + tok.count("|")
                if order in per_order:
                    per_order[order][tok] += cnt
    vocab: List[str] = []
    for order in orders:
        k = int(top_k.get(order, top_k.get(str(order), 100)))
        vocab.extend(tok for tok, _ in per_order[order].most_common(k))
    return vocab


def write_vocab(chunks: List[List[Dict[str, Any]]]) -> int:
    """Build and persist the frozen vocab next to this module. Returns its size."""
    vocab = build_vocab(chunks)
    payload = {"config": NGRAM_CONFIG, "vocab": vocab}
    _vocab_path().write_text(json.dumps(payload, indent=0), encoding="utf-8")
    global _VOCAB_CACHE
    _VOCAB_CACHE = vocab
    return len(vocab)
