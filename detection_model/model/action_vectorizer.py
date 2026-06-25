"""Action tokenizer (schema v2) for the hierarchical Poker44 detector.

Each poker action becomes:

* **6 categorical channels** ``[street, action_type, seat, amount_bucket,
  pot_flow, first_in_street]``. The first three are learned vocabularies fit
  from data; the last three are fixed, hand-engineered buckets that give the
  model an explicit, low-variance view of *sizing* and *betting rhythm* — the
  signals bots leak most. (This is the expressive-token idea adapted from the
  reference stacked model.)
* **numeric channels** with money/pot/stack context (unchanged, already rich).

Each hand additionally produces:

* **hand_meta** (8 dims): stack depth, distinct actors, streets dealt, per-street
  action counts, hero engagement — derived only from miner-visible fields.
* **hand_end**: the deepest street the hand reached (a strong "did this hand go
  the distance" signal).

``encode_chunk`` returns a dict so the dataset/collate and inference paths share
one contract. The categorical layout is fixed by :data:`CAT_CHANNELS`.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = 3

# Fixed categorical layout. Index positions are load-bearing: the model embeds
# each channel by position, so do not reorder without retraining.
#
# v3 adds two channels ported from the reference stacked model:
#   * actor_role     — the acting seat's position relative to the button
#                      (button/SB/BB/early/middle/late). Bots often act
#                      position-agnostically; humans are strongly positional.
#   * street_position — this action's ordinal index within its street
#                      (1st/2nd/3rd/4th+). Captures betting-rhythm/order tells.
CAT_CHANNELS = (
    "street",
    "action_type",
    "seat",
    "amount_bucket",
    "pot_flow",
    "first_in_street",
    "actor_role",
    "street_position",
)
CAT_DIM = len(CAT_CHANNELS)

# Amount buckets in BB (mirrors the canonicalization live payloads apply). Bucket
# id is index+1; 0 is the pad/empty id.
AMOUNT_BUCKETS_BB = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 36.0, 56.0, 84.0, 126.0)
AMOUNT_BUCKET_VOCAB_SIZE = len(AMOUNT_BUCKETS_BB) + 1

# Pot-flow buckets: how much the pot grew on this action.
POT_FLOW_VOCAB_SIZE = 5  # pad, flat, small_up, medium_up, large_up

# Per-action "first action on its street" indicator: pad / continuation / first.
FIRST_IN_STREET_VOCAB_SIZE = 3

# Actor role relative to the button: pad, unknown, button, small_blind,
# big_blind, early, middle, late.
ACTOR_ROLE_VOCAB_SIZE = 8

# Ordinal action index within its street: pad, 1st, 2nd, 3rd, 4th+.
STREET_POSITION_VOCAB_SIZE = 5

# Deepest street the hand reached (per-hand token).
HAND_END_TO_ID = {"<pad>": 0, "preflop": 1, "flop": 2, "turn": 3, "river": 4}
HAND_END_VOCAB_SIZE = len(HAND_END_TO_ID)

# Per-hand continuous context. Order is consumed by the model's hand_meta_proj;
# do not reorder without retraining.
HAND_META_DIM = 8
#   0 log1p(hero_starting_stack_bb)   4 actions_per_street_flop / cap
#   1 distinct_actors / 10            5 actions_per_street_turn / cap
#   2 streets_dealt / 4               6 actions_per_street_river / cap
#   3 actions_per_street_preflop/cap  7 hero_action_share


class ActionVectorizer:
    PAD = "<PAD>"
    UNK = "<UNK>"

    FORCED_ACTIONS = {"small_blind", "big_blind", "ante", "straddle", "bring_in"}
    MONEY_ACTIONS = FORCED_ACTIONS | {"call", "bet", "raise", "all_in", "allin"}
    AGGRESSIVE_ACTIONS = {"bet", "raise", "all_in", "allin"}
    PASSIVE_ACTIONS = {"check", "call"}
    _STREET_BUCKET = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

    def __init__(self, max_actions_per_hand: int = 64, max_table_seats: int = 9):
        self.max_actions_per_hand = max(1, int(max_actions_per_hand))
        self.max_table_seats = max(1, int(max_table_seats))

        self.street_to_id = {self.PAD: 0, self.UNK: 1}
        self.action_type_to_id = {self.PAD: 0, self.UNK: 1}
        self.seat_to_id = {self.PAD: 0, self.UNK: 1}

        self.numeric_feature_names = [
            "action_order_norm",
            "amount_bb_scaled",
            "raise_to_bb_scaled",
            "call_to_bb_scaled",
            "pot_before_bb_scaled",
            "pot_after_bb_scaled",
            "pot_delta_bb_signed",
            "amount_to_pot",
            "raise_to_pot",
            "call_to_pot",
            "has_amount",
            "has_raise_to",
            "has_call_to",
            "is_forced_action",
            "is_money_action",
            "is_aggressive_action",
            "is_passive_action",
            "is_all_in",
            "actor_seat_norm",
            "actor_stack_bb_scaled",
            "amount_to_actor_stack",
            "street_progress",
        ]

    # ---- fixed-vocab sizes the model needs to build embeddings -------------

    cat_dim = CAT_DIM
    amount_bucket_vocab_size = AMOUNT_BUCKET_VOCAB_SIZE
    pot_flow_vocab_size = POT_FLOW_VOCAB_SIZE
    first_in_street_vocab_size = FIRST_IN_STREET_VOCAB_SIZE
    actor_role_vocab_size = ACTOR_ROLE_VOCAB_SIZE
    street_position_vocab_size = STREET_POSITION_VOCAB_SIZE
    hand_end_vocab_size = HAND_END_VOCAB_SIZE
    hand_meta_dim = HAND_META_DIM

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric_feature_names)

    @property
    def street_vocab_size(self) -> int:
        return len(self.street_to_id)

    @property
    def action_type_vocab_size(self) -> int:
        return len(self.action_type_to_id)

    @property
    def seat_vocab_size(self) -> int:
        return len(self.seat_to_id)

    # ---- small helpers ------------------------------------------------------

    def normalize_text(self, value: Any, default: str = "unknown") -> str:
        if value is None:
            return default
        text = str(value).strip().lower()
        if not text:
            return default
        return text.replace("-", "_").replace(" ", "_").replace("/", "_")

    def safe_float(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("€", "").replace("$", "").replace("£", "").strip()
            out = float(value)
            return out if math.isfinite(out) else default
        except Exception:
            return default

    def safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return default

    def is_present(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan"}:
            return False
        return True

    def capped_ratio(self, numerator: float, denominator: float, cap: float = 10.0) -> float:
        if denominator <= 0:
            return 0.0
        return max(0.0, min(float(numerator) / float(denominator), cap)) / cap

    def scaled_bb(self, value: float, cap: float = 200.0) -> float:
        value = max(0.0, float(value))
        if cap <= 0:
            return 0.0
        return min(math.log1p(value) / math.log1p(cap), 1.0)

    def parse_seat_from_uid(self, value: Any) -> int:
        if value is None:
            return 0
        text = str(value).strip().lower()
        if text.startswith("seat_"):
            return self.safe_int(text.replace("seat_", ""), default=0)
        return 0

    @staticmethod
    def _amount_bucket_id(amount_bb: float) -> int:
        value = max(0.0, float(amount_bb))
        if value <= 0.0:
            return 1
        nearest = min(AMOUNT_BUCKETS_BB, key=lambda edge: abs(edge - value))
        return AMOUNT_BUCKETS_BB.index(nearest) + 1

    @staticmethod
    def _actor_role_id(actor_seat: int, button_seat: int, max_seats: int) -> int:
        """Position of the acting seat relative to the button.

        Returns: 1 unknown, 2 button, 3 small_blind, 4 big_blind,
        5 early, 6 middle, 7 late. 0 is reserved for padding.
        """
        if actor_seat <= 0 or button_seat <= 0 or max_seats <= 1:
            return 1  # unknown
        offset = (actor_seat - button_seat) % max_seats
        if offset == 0:
            return 2  # button
        if offset == 1:
            return 3  # small blind
        if offset == 2:
            return 4  # big blind
        # Split the remaining seats into early / middle / late thirds.
        remaining = max_seats - 3
        if remaining <= 0:
            return 6  # middle (heads-up / very small tables)
        rank = offset - 3  # 0-based position among the non-blind seats
        third = rank / max(1, remaining)
        if third < 1.0 / 3.0:
            return 5  # early
        if third < 2.0 / 3.0:
            return 6  # middle
        return 7  # late

    @staticmethod
    def _street_position_id(position_within_street: int) -> int:
        """Ordinal index of an action within its street, capped at 4+."""
        if position_within_street <= 0:
            return 0
        return min(position_within_street, 4)

    @staticmethod
    def _pot_flow_id(pot_before_bb: float, pot_after_bb: float) -> int:
        delta = max(0.0, float(pot_after_bb) - float(pot_before_bb))
        if delta <= 1e-6:
            return 1  # flat
        if delta <= 1.0:
            return 2  # small_up
        if delta <= 4.0:
            return 3  # medium_up
        return 4  # large_up

    # ---- table / seat inference (unchanged) --------------------------------

    def _infer_hand_max_seats_raw(self, hand: Dict[str, Any]) -> int:
        if not isinstance(hand, dict):
            return self.max_table_seats
        metadata = hand.get("metadata") or {}
        explicit = self.safe_int(metadata.get("max_seats"), default=0)
        if explicit > 0:
            return explicit
        seats: List[int] = []
        for key in ("hero_seat", "button_seat"):
            seat = self.safe_int(metadata.get(key), default=0)
            if seat > 0:
                seats.append(seat)
        for player in hand.get("players") or []:
            if not isinstance(player, dict):
                continue
            seat = self.safe_int(player.get("seat"), default=0) or self.parse_seat_from_uid(player.get("player_uid"))
            if seat > 0:
                seats.append(seat)
        for action in hand.get("actions") or []:
            if not isinstance(action, dict):
                continue
            seat = self.safe_int(action.get("actor_seat"), default=0)
            if seat > 0:
                seats.append(seat)
        return max(seats) if seats else self.max_table_seats

    def get_hand_max_seats(self, hand: Dict[str, Any]) -> int:
        return max(1, min(int(self._infer_hand_max_seats_raw(hand)), self.max_table_seats))

    def get_players_by_seat(self, hand: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        players_by_seat: Dict[int, Dict[str, Any]] = {}
        if not isinstance(hand, dict):
            return players_by_seat
        max_seats = self.get_hand_max_seats(hand)
        for player in hand.get("players") or []:
            if not isinstance(player, dict):
                continue
            seat = self.safe_int(player.get("seat"), default=0) or self.parse_seat_from_uid(player.get("player_uid"))
            if 1 <= seat <= max_seats:
                players_by_seat[seat] = player
        return players_by_seat

    # ---- fitting ------------------------------------------------------------

    def fit(self, chunks: List[List[Dict[str, Any]]], min_freq: int = 1) -> "ActionVectorizer":
        street_counter: Counter[str] = Counter()
        action_counter: Counter[str] = Counter()
        seat_counter: Counter[str] = Counter()
        observed_max_seats = self.max_table_seats

        for chunk in chunks:
            if not isinstance(chunk, list):
                continue
            for hand in chunk:
                if not isinstance(hand, dict):
                    continue
                observed_max_seats = max(observed_max_seats, self._infer_hand_max_seats_raw(hand))
                for action in hand.get("actions") or []:
                    if not isinstance(action, dict):
                        continue
                    street_counter[self.normalize_text(action.get("street"))] += 1
                    action_counter[self.normalize_text(action.get("action_type"))] += 1
                    seat = self.safe_int(action.get("actor_seat"), default=0)
                    seat_counter[f"seat_{seat}" if seat > 0 else "seat_unknown"] += 1

        self.max_table_seats = max(1, int(observed_max_seats))
        for value, count in street_counter.items():
            if count >= min_freq and value not in self.street_to_id:
                self.street_to_id[value] = len(self.street_to_id)
        for value, count in action_counter.items():
            if count >= min_freq and value not in self.action_type_to_id:
                self.action_type_to_id[value] = len(self.action_type_to_id)
        for value, count in seat_counter.items():
            if count >= min_freq and value not in self.seat_to_id:
                self.seat_to_id[value] = len(self.seat_to_id)
        return self

    # ---- encoding -----------------------------------------------------------

    def _encode_action(
        self,
        action: Dict[str, Any],
        action_index: int,
        total_actions: int,
        *,
        bb: float,
        max_seats: int,
        actor_stack: float,
    ) -> Tuple[List[int], List[float], str]:
        """Return (cat_ids[5], numeric[numeric_dim], normalized_street).

        first_in_street (the 6th categorical) is added by ``encode_hand`` since it
        needs the previous action's street.
        """
        street = self.normalize_text(action.get("street"))
        action_type = self.normalize_text(action.get("action_type"))
        seat = self.safe_int(action.get("actor_seat"), default=0)
        seat_key = f"seat_{seat}" if seat > 0 else "seat_unknown"

        street_id = self.street_to_id.get(street, self.street_to_id[self.UNK])
        action_type_id = self.action_type_to_id.get(action_type, self.action_type_to_id[self.UNK])
        seat_id = self.seat_to_id.get(seat_key, self.seat_to_id[self.UNK])

        amount = self.safe_float(action.get("amount"), default=0.0)
        raise_to = self.safe_float(action.get("raise_to"), default=0.0)
        call_to = self.safe_float(action.get("call_to"), default=0.0)
        normalized_amount_bb = self.safe_float(action.get("normalized_amount_bb"), default=0.0)
        pot_before = self.safe_float(action.get("pot_before"), default=0.0)
        pot_after = self.safe_float(action.get("pot_after"), default=0.0)
        pot_delta = pot_after - pot_before

        amount_bb = normalized_amount_bb if normalized_amount_bb > 0 else (amount / bb if amount > 0 and bb > 0 else 0.0)
        raise_to_bb = raise_to / bb if raise_to > 0 and bb > 0 else 0.0
        call_to_bb = call_to / bb if call_to > 0 and bb > 0 else 0.0
        pot_before_bb = pot_before / bb if pot_before > 0 and bb > 0 else 0.0
        pot_after_bb = pot_after / bb if pot_after > 0 and bb > 0 else 0.0
        pot_delta_bb = pot_delta / bb if bb > 0 else 0.0
        actor_stack_bb = actor_stack / bb if actor_stack > 0 and bb > 0 else 0.0

        amount_bucket_id = self._amount_bucket_id(amount_bb)
        pot_flow_id = self._pot_flow_id(pot_before_bb, pot_after_bb)

        action_order_norm = 0.0 if total_actions <= 1 else action_index / max(1, total_actions - 1)
        street_progress = {"preflop": 0.25, "flop": 0.50, "turn": 0.75, "river": 1.00}.get(street, 0.0)

        feature_map: Dict[str, float] = {
            "action_order_norm": float(action_order_norm),
            "amount_bb_scaled": self.scaled_bb(amount_bb),
            "raise_to_bb_scaled": self.scaled_bb(raise_to_bb),
            "call_to_bb_scaled": self.scaled_bb(call_to_bb),
            "pot_before_bb_scaled": self.scaled_bb(pot_before_bb),
            "pot_after_bb_scaled": self.scaled_bb(pot_after_bb),
            "pot_delta_bb_signed": max(-1.0, min(1.0, pot_delta_bb / 200.0)),
            "amount_to_pot": self.capped_ratio(amount, pot_before),
            "raise_to_pot": self.capped_ratio(raise_to, pot_before),
            "call_to_pot": self.capped_ratio(call_to, pot_before),
            "has_amount": 1.0 if amount > 0 else 0.0,
            "has_raise_to": 1.0 if self.is_present(action.get("raise_to")) else 0.0,
            "has_call_to": 1.0 if self.is_present(action.get("call_to")) else 0.0,
            "is_forced_action": 1.0 if action_type in self.FORCED_ACTIONS else 0.0,
            "is_money_action": 1.0 if action_type in self.MONEY_ACTIONS else 0.0,
            "is_aggressive_action": 1.0 if action_type in self.AGGRESSIVE_ACTIONS else 0.0,
            "is_passive_action": 1.0 if action_type in self.PASSIVE_ACTIONS else 0.0,
            "is_all_in": 1.0 if action_type in {"all_in", "allin"} else 0.0,
            "actor_seat_norm": seat / max(1, max_seats) if seat > 0 else 0.0,
            "actor_stack_bb_scaled": self.scaled_bb(actor_stack_bb),
            "amount_to_actor_stack": self.capped_ratio(amount, actor_stack, cap=1.0),
            "street_progress": street_progress,
        }

        cat_ids = [street_id, action_type_id, seat_id, amount_bucket_id, pot_flow_id]
        numeric = [float(feature_map.get(name, 0.0)) for name in self.numeric_feature_names]
        return cat_ids, numeric, street

    def _hand_meta(self, hand: Dict[str, Any], *, bb: float, hero_seat: int) -> Tuple[List[float], int]:
        """Per-hand continuous context + deepest-street-reached token."""
        actions = hand.get("actions") or []
        streets_list = hand.get("streets") or []
        players_by_seat = self.get_players_by_seat(hand)

        hero_stack_bb = 0.0
        if hero_seat > 0:
            hero = players_by_seat.get(hero_seat) or {}
            stack = self.safe_float(hero.get("starting_stack"), default=0.0)
            hero_stack_bb = stack / bb if stack > 0 and bb > 0 else 0.0

        per_street = [0, 0, 0, 0]
        distinct_actors: set[int] = set()
        hero_actions = 0
        deepest = 0
        for action in actions:
            if not isinstance(action, dict):
                continue
            street = self.normalize_text(action.get("street"))
            bucket = self._STREET_BUCKET.get(street)
            if bucket is not None:
                per_street[bucket] += 1
                deepest = max(deepest, bucket + 1)
            seat = self.safe_int(action.get("actor_seat"), default=0)
            if seat > 0:
                distinct_actors.add(seat)
                if hero_seat and seat == hero_seat:
                    hero_actions += 1

        if deepest == 0:
            for entry in reversed(streets_list):
                if isinstance(entry, dict):
                    candidate = HAND_END_TO_ID.get(self.normalize_text(entry.get("street")), 0)
                    if candidate > 0:
                        deepest = candidate
                        break

        streets_dealt = sum(
            1 for entry in streets_list
            if isinstance(entry, dict) and str(entry.get("street", "")).strip()
        )
        cap = float(max(self.max_actions_per_hand, 1))
        total_actions = sum(per_street)
        meta = [
            math.log1p(max(hero_stack_bb, 0.0)),
            min(len(distinct_actors), 10) / 10.0,
            min(streets_dealt, 4) / 4.0,
            min(per_street[0], cap) / cap,
            min(per_street[1], cap) / cap,
            min(per_street[2], cap) / cap,
            min(per_street[3], cap) / cap,
            (hero_actions / total_actions) if total_actions > 0 else 0.0,
        ]
        return meta, int(deepest)

    def encode_hand(
        self, hand: Dict[str, Any]
    ) -> Tuple[List[List[int]], List[List[float]], List[float], int]:
        actions = hand.get("actions") or []
        metadata = hand.get("metadata") or {}
        bb = self.safe_float(metadata.get("bb"), default=0.0)
        hero_seat = self.safe_int(metadata.get("hero_seat"), default=0)
        button_seat = self.safe_int(metadata.get("button_seat"), default=0)
        max_seats = self.get_hand_max_seats(hand)
        players_by_seat = self.get_players_by_seat(hand)

        cat_rows: List[List[int]] = []
        num_rows: List[List[float]] = []
        prev_street: str | None = None
        street_position_counts: Dict[str, int] = {}

        for idx, action in enumerate(actions[: self.max_actions_per_hand]):
            if not isinstance(action, dict):
                continue
            actor_seat = self.safe_int(action.get("actor_seat"), default=0)
            actor = players_by_seat.get(actor_seat) or {}
            actor_stack = self.safe_float(actor.get("starting_stack"), default=0.0)

            cat_ids, numeric, street = self._encode_action(
                action=action,
                action_index=idx,
                total_actions=len(actions),
                bb=bb,
                max_seats=max_seats,
                actor_stack=actor_stack,
            )
            first_in_street = 2 if street != prev_street else 1
            prev_street = street
            street_position_counts[street] = street_position_counts.get(street, 0) + 1
            actor_role_id = self._actor_role_id(actor_seat, button_seat, max_seats)
            street_position_id = self._street_position_id(street_position_counts[street])
            cat_rows.append(cat_ids + [first_in_street, actor_role_id, street_position_id])
            num_rows.append(numeric)

        if not cat_rows:
            cat_rows.append([0] * CAT_DIM)
            num_rows.append([0.0] * self.numeric_dim)

        hand_meta, hand_end = self._hand_meta(hand, bb=bb, hero_seat=hero_seat)
        return cat_rows, num_rows, hand_meta, hand_end

    def encode_chunk(self, chunk: List[Dict[str, Any]], max_hands: int) -> Dict[str, Any]:
        chunk_cat: List[List[List[int]]] = []
        chunk_num: List[List[List[float]]] = []
        chunk_hand_meta: List[List[float]] = []
        chunk_hand_end: List[int] = []

        hands = chunk[:max_hands] if isinstance(chunk, list) else []
        for hand in hands:
            if not isinstance(hand, dict):
                continue
            cat_rows, num_rows, hand_meta, hand_end = self.encode_hand(hand)
            chunk_cat.append(cat_rows)
            chunk_num.append(num_rows)
            chunk_hand_meta.append(hand_meta)
            chunk_hand_end.append(hand_end)

        if not chunk_cat:
            chunk_cat.append([[0] * CAT_DIM])
            chunk_num.append([[0.0] * self.numeric_dim])
            chunk_hand_meta.append([0.0] * HAND_META_DIM)
            chunk_hand_end.append(0)

        return {
            "cat": chunk_cat,
            "num": chunk_num,
            "hand_meta": chunk_hand_meta,
            "hand_end": chunk_hand_end,
        }

    # ---- (de)serialize ------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "max_actions_per_hand": self.max_actions_per_hand,
            "max_table_seats": self.max_table_seats,
            "street_to_id": dict(self.street_to_id),
            "action_type_to_id": dict(self.action_type_to_id),
            "seat_to_id": dict(self.seat_to_id),
            "numeric_feature_names": list(self.numeric_feature_names),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != SCHEMA_VERSION:
            raise ValueError(
                f"ActionVectorizer schema mismatch: artifact is v{state.get('schema_version', 1)}, "
                f"code is v{SCHEMA_VERSION}. Retrain with the current model."
            )
        self.max_actions_per_hand = int(state.get("max_actions_per_hand", self.max_actions_per_hand))
        self.max_table_seats = int(state.get("max_table_seats", self.max_table_seats))
        self.street_to_id = {str(k): int(v) for k, v in state["street_to_id"].items()}
        self.action_type_to_id = {str(k): int(v) for k, v in state["action_type_to_id"].items()}
        self.seat_to_id = {str(k): int(v) for k, v in state["seat_to_id"].items()}
        self.numeric_feature_names = list(state.get("numeric_feature_names", self.numeric_feature_names))

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "ActionVectorizer":
        obj = cls(
            max_actions_per_hand=int(state.get("max_actions_per_hand", 64)),
            max_table_seats=int(state.get("max_table_seats", 9)),
        )
        obj.load_state_dict(state)
        return obj
