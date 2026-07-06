"""Order-invariant, scale-robust feature extraction (action -> hand -> chunk).

Design rules baked in:

* **Hand order is never used.** Chunk features are permutation-invariant
  aggregates over hands (mean/std/min/max/quantiles + set-level signatures).
* **Action order inside a hand is used** (actions are pre-sorted by ``action_id``
  in the schema layer) for local anomaly features only.
* **Scale robustness.** No raw stacks/amounts as-is: only ratios, entropies,
  bucket shares, and ``normalized_amount_bb`` quantiles — so the train (stack~4.5)
  vs eval (stack~2.0) shift does not move the features.
* **Anomalies are features, not filters.** Invalid poker patterns (street jumps,
  acting-after-fold, pot mismatches) are counted and kept, since the noisier eval
  set carries signal there.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 4}
ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
# Coarse BB buckets on normalized_amount_bb (scale-invariant across datasets).
_AMOUNT_EDGES = (0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
_STAT_SUFFIXES = ("mean", "std", "min", "max", "q25", "q50", "q75")


# ----------------------------------------------------------------- small utils

def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _entropy(labels: List[Any]) -> float:
    """Normalized Shannon entropy in [0, 1] (0 = single symbol, 1 = uniform)."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    if len(counts) <= 1:
        return 0.0
    total = float(len(labels))
    ent = -sum((c / total) * math.log(c / total + 1e-12) for c in counts.values())
    return ent / math.log(len(counts))


def _amount_bucket(bb: float) -> int:
    if bb <= 0.0:
        return 0
    for i, edge in enumerate(_AMOUNT_EDGES[1:], start=1):
        if bb <= edge:
            return i
    return len(_AMOUNT_EDGES)


# ------------------------------------------------------------- hand-level feats

def extract_hand_features(hand: Dict[str, Any]) -> Dict[str, float]:
    metadata = hand.get("metadata") or {}
    actions = hand.get("actions") or []
    players = hand.get("players") or []

    max_seats = max(1, _i(metadata.get("max_seats"), 6))
    hero_seat = _i(metadata.get("hero_seat"), 0)
    n_actions = len(actions)

    action_counts = {a: 0 for a in ACTION_TYPES}
    street_counts = {s: 0 for s in STREET_ORDER}
    amounts: List[float] = []
    actor_seats: List[int] = []
    action_type_seq: List[str] = []
    street_seq: List[str] = []

    folded: set = set()
    acted_after_fold = same_actor_run = 0
    street_regr = street_jump = max_jump = 0
    pot_mismatch = pot_decrease = pot_negative = 0
    amount_negative = raise_to_missing = call_to_missing = zero_amt_betraise = 0
    hero_actions = 0

    prev_actor = None
    prev_rank = None

    for i, a in enumerate(actions):
        at = str(a.get("action_type") or "").lower().strip()
        st = str(a.get("street") or "").lower().strip()
        actor = _i(a.get("actor_seat"), -1)
        amt = _f(a.get("normalized_amount_bb"), 0.0)

        if at in action_counts:
            action_counts[at] += 1
        if st in street_counts:
            street_counts[st] += 1
        action_type_seq.append(at)
        street_seq.append(st)
        if actor >= 0:
            actor_seats.append(actor)
        amounts.append(amt)

        if hero_seat and actor == hero_seat:
            hero_actions += 1
        if amt < 0:
            amount_negative += 1
        if at == "raise" and a.get("raise_to") is None:
            raise_to_missing += 1
        if at == "call" and a.get("call_to") is None:
            call_to_missing += 1
        if at in ("bet", "raise") and amt <= 0.0:
            zero_amt_betraise += 1

        if actor in folded:
            acted_after_fold += 1
        if at == "fold":
            folded.add(actor)
        if prev_actor is not None and actor == prev_actor:
            same_actor_run += 1

        rank = STREET_ORDER.get(st)
        if prev_rank is not None and rank is not None:
            diff = rank - prev_rank
            if diff < 0:
                street_regr += 1
            if diff > 1:
                street_jump += 1
                max_jump = max(max_jump, diff)

        if i + 1 < len(actions):
            pa = a.get("pot_after")
            nb = actions[i + 1].get("pot_before")
            if pa is not None and nb is not None:
                if abs(_f(pa) - _f(nb)) > 0.01:
                    pot_mismatch += 1
                if _f(nb) < _f(pa) - 0.01:
                    pot_decrease += 1
        if _f(a.get("pot_after"), 0.0) < 0:
            pot_negative += 1

        prev_actor = actor
        prev_rank = rank

    reached = {s: (street_counts[s] > 0) for s in STREET_ORDER}
    last_street_rank = max((STREET_ORDER[s] for s in STREET_ORDER if street_counts[s] > 0), default=0)
    denom_a = max(1, n_actions)
    aggressive = action_counts["bet"] + action_counts["raise"]
    passive = action_counts["check"] + action_counts["call"]

    feats: Dict[str, float] = {
        "num_players": float(len(players)),
        "seat_utilization": _ratio(len(players), max_seats),
        "hero_seat_norm": _ratio(hero_seat, max_seats),
        "num_actions": float(n_actions),
        "num_streets_observed": float(sum(1 for s in STREET_ORDER if street_counts[s] > 0)),
        "last_street_rank": float(last_street_rank),
        "ended_preflop": float(last_street_rank == 0),
        "reached_flop": float(reached["flop"]),
        "reached_turn": float(reached["turn"]),
        "reached_river": float(reached["river"]),
        "reached_showdown": float(reached["showdown"]),
        "aggression_ratio": _ratio(aggressive, max(1, passive)),
        "passive_ratio": _ratio(passive, denom_a),
        "hero_action_ratio": _ratio(hero_actions, denom_a),
        # entropy / repetition
        "action_type_entropy": _entropy(action_type_seq),
        "actor_seat_entropy": _entropy(actor_seats),
        "street_entropy": _entropy(street_seq),
        # anomaly counts + ratios
        "acted_after_fold_ratio": _ratio(acted_after_fold, denom_a),
        "has_actor_after_fold": float(acted_after_fold > 0),
        "same_actor_consecutive_ratio": _ratio(same_actor_run, denom_a),
        "street_regression_ratio": _ratio(street_regr, denom_a),
        "street_jump_ratio": _ratio(street_jump, denom_a),
        "max_street_jump": float(max_jump),
        "pot_mismatch_ratio": _ratio(pot_mismatch, max(1, n_actions - 1)),
        "pot_decrease_ratio": _ratio(pot_decrease, max(1, n_actions - 1)),
        "pot_negative_ratio": _ratio(pot_negative, denom_a),
        "amount_negative_ratio": _ratio(amount_negative, denom_a),
        "raise_to_missing_ratio": _ratio(raise_to_missing, denom_a),
        "call_to_missing_ratio": _ratio(call_to_missing, denom_a),
        "zero_amount_betraise_ratio": _ratio(zero_amt_betraise, denom_a),
    }

    for t in ACTION_TYPES:
        feats[f"{t}_ratio"] = _ratio(action_counts[t], denom_a)
    for s in STREET_ORDER:
        feats[f"{s}_action_ratio"] = _ratio(street_counts[s], denom_a)

    pos = [a for a in amounts if a > 0]
    arr = np.asarray(pos, dtype=float) if pos else np.zeros(1)
    feats["amount_mean"] = float(arr.mean())
    feats["amount_std"] = float(arr.std())
    feats["amount_max"] = float(arr.max())
    feats["amount_q75"] = float(np.quantile(arr, 0.75))
    feats["amount_log_mean"] = float(np.log1p(arr).mean())
    feats["nonzero_amount_ratio"] = _ratio(len(pos), denom_a)
    return feats


# canonical hand-feature key order (empty hand emits the full, fixed key set)
HAND_FEATURE_KEYS: Tuple[str, ...] = tuple(sorted(extract_hand_features({}).keys()))


# ------------------------------------------------------------ chunk-level feats

def _hand_signature(hand: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str((a or {}).get("action_type") or "").lower() for a in (hand.get("actions") or []))


def _amount_bucket_signature(hand: Dict[str, Any]) -> Tuple[int, ...]:
    return tuple(_amount_bucket(_f((a or {}).get("normalized_amount_bb"), 0.0)) for a in (hand.get("actions") or []))


def _agg(prefix: str, values: np.ndarray, out: Dict[str, float]) -> None:
    out[f"{prefix}_mean"] = float(values.mean())
    out[f"{prefix}_std"] = float(values.std())
    out[f"{prefix}_min"] = float(values.min())
    out[f"{prefix}_max"] = float(values.max())
    out[f"{prefix}_q25"] = float(np.quantile(values, 0.25))
    out[f"{prefix}_q50"] = float(np.quantile(values, 0.50))
    out[f"{prefix}_q75"] = float(np.quantile(values, 0.75))


def chunk_feature_vector(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """One row of features for a chunk (list of hand dicts). Order-invariant."""
    out: Dict[str, float] = {"num_hands": float(len(hands))}
    if not hands:
        for key in HAND_FEATURE_KEYS:
            for suf in _STAT_SUFFIXES:
                out[f"{key}_{suf}"] = 0.0
        for k in _CHUNK_ONLY_KEYS:
            out[k] = 0.0
        return out

    per_hand = [extract_hand_features(h) for h in hands]
    for key in HAND_FEATURE_KEYS:
        series = np.asarray([hf.get(key, 0.0) for hf in per_hand], dtype=float)
        _agg(key, series, out)

    n = float(len(hands))
    act_sigs = [_hand_signature(h) for h in hands]
    amt_sigs = [_amount_bucket_signature(h) for h in hands]
    out["unique_action_pattern_ratio"] = _ratio(len(set(act_sigs)), n)
    out["top_action_pattern_share"] = _ratio(max(Counter(act_sigs).values()), n)
    out["unique_amount_pattern_ratio"] = _ratio(len(set(amt_sigs)), n)
    out["top_amount_pattern_share"] = _ratio(max(Counter(amt_sigs).values()), n)
    # cross-hand behavioral consistency (low = repetitive = bot-like)
    out["chunk_aggression_dispersion"] = float(np.std([hf["aggression_ratio"] for hf in per_hand]))
    out["chunk_action_entropy_mean"] = float(np.mean([hf["action_type_entropy"] for hf in per_hand]))
    out["chunk_anomaly_load"] = float(np.mean([
        hf["street_jump_ratio"] + hf["street_regression_ratio"]
        + hf["acted_after_fold_ratio"] + hf["pot_mismatch_ratio"]
        for hf in per_hand
    ]))
    return out


_CHUNK_ONLY_KEYS: Tuple[str, ...] = (
    "unique_action_pattern_ratio", "top_action_pattern_share",
    "unique_amount_pattern_ratio", "top_amount_pattern_share",
    "chunk_aggression_dispersion", "chunk_action_entropy_mean", "chunk_anomaly_load",
)


def feature_names_for() -> List[str]:
    """Deterministic, fixed feature-column order (train == serve)."""
    names = ["num_hands"]
    for key in HAND_FEATURE_KEYS:
        names.extend(f"{key}_{suf}" for suf in _STAT_SUFFIXES)
    names.extend(_CHUNK_ONLY_KEYS)
    return names
