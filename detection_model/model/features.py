from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np


class FeatureVectorizer:
    """
    Compact chunk-level feature engineering for the final XGBoost head.

    These features deliberately avoid wide per-seat expansion. They keep the
    signals that usually matter for predictability: volume, action mix, street
    coverage, money/pot scale, stack shape, and basic metadata consistency.
    """

    DEFAULT_FEATURE_NAMES: List[str] = [
        # chunk/table shape
        "num_hands_scaled",
        "avg_actions_per_hand_scaled",
        "std_actions_per_hand_scaled",
        "avg_players_norm",
        "avg_occupied_ratio",
        "showdown_rate",
        # action mix
        "fold_rate",
        "check_rate",
        "call_rate",
        "bet_raise_rate",
        "all_in_rate",
        "blind_rate",
        "money_action_rate",
        "zero_amount_rate",
        "aggression_to_passive",
        # street and actor distribution
        "preflop_action_rate",
        "postflop_action_rate",
        "street_coverage",
        "action_entropy_norm",
        "street_entropy_norm",
        "actor_entropy_norm",
        # amounts and pots in BB units
        "mean_amount_bb_scaled",
        "std_amount_bb_scaled",
        "max_amount_bb_scaled",
        "mean_pot_bb_scaled",
        "max_pot_bb_scaled",
        "pot_increase_rate",
        "mean_amount_to_pot",
        # stack/player metadata
        "mean_stack_bb_scaled",
        "min_stack_bb_scaled",
        "max_stack_bb_scaled",
        "stack_spread_bb_scaled",
        "short_stack_rate",
        "deep_stack_rate",
        "hero_present_rate",
        "button_present_rate",
        "visible_hole_card_rate",
        "showed_hand_rate",
        # cross-hand consistency / repetition (bot tells): bots replay near
        # identical action and sizing sequences across hands, so a high "top
        # signature share" / low "unique share" is suspicious.
        "action_signature_top_share",
        "action_signature_unique_share",
        "amount_bucket_signature_top_share",
        "amount_bucket_signature_unique_share",
        "low_action_entropy_hand_rate",
        "high_aggression_hand_rate",
        # Bet-sizing consistency: bots vary their sizing far less than humans.
        # Only this one survived the temporal-holdout check (train<=06-08,
        # test 06-09..06-13); the other sizing/determinism/CV candidates
        # degraded held-out AP on this small dataset and were dropped.
        "amount_cv",
    ]

    ACTIONS = {"fold", "check", "call", "bet", "raise", "all_in", "allin"}
    BLINDS = {"small_blind", "big_blind", "ante", "straddle", "bring_in"}
    MONEY_ACTIONS = BLINDS | {"call", "bet", "raise", "all_in", "allin"}

    def __init__(self):
        self.feature_names: List[str] = list(self.DEFAULT_FEATURE_NAMES)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            if isinstance(value, str):
                value = (
                    value.replace(",", "")
                    .replace("$", "")
                    .replace("€", "")
                    .replace("£", "")
                    .strip()
                )
            out = float(value)
            return out if math.isfinite(out) else default
        except Exception:
            return default

    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            try:
                return int(float(value))
            except Exception:
                return default

    @staticmethod
    def norm_text(value: Any, default: str = "unknown") -> str:
        if value is None:
            return default
        text = str(value).strip().lower()
        if not text:
            return default
        return text.replace("-", "_").replace(" ", "_").replace("/", "_")

    @staticmethod
    def mean(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def std(values: Iterable[float]) -> float:
        values = list(values)
        return float(np.std(values)) if values else 0.0

    @staticmethod
    def scaled_log(value: float, cap: float = 200.0) -> float:
        value = max(0.0, float(value))
        if cap <= 0:
            return 0.0
        return min(math.log1p(value) / math.log1p(cap), 1.0)

    @staticmethod
    def capped_ratio(numerator: float, denominator: float, cap: float = 10.0) -> float:
        if denominator <= 0:
            return 0.0
        return max(0.0, min(float(numerator) / float(denominator), cap)) / cap

    @staticmethod
    def cv(values: List[float], cap: float = 3.0) -> float:
        """Coefficient of variation (std/mean), normalized to [0, 1].

        Low CV across a chunk = mechanically consistent behavior = bot-like.
        Returns 0 for fewer than 2 samples or a ~zero mean.
        """
        vals = [float(v) for v in values]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        if m <= 1e-9:
            return 0.0
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        return min(math.sqrt(max(0.0, var)) / m, cap) / cap

    @staticmethod
    def amount_bucket_label(amount_bb: float) -> str:
        """Coarse BB-size bucket for cross-hand sizing-signature comparison."""
        value = max(0.0, float(amount_bb))
        for edge, label in ((0.0, "z"), (1.0, "xs"), (2.0, "s"), (4.0, "m"), (8.0, "l"), (20.0, "xl")):
            if value <= edge:
                return label
        return "xxl"

    @staticmethod
    def entropy_norm(counter: Counter, max_categories: int) -> float:
        total = sum(counter.values())
        if total <= 0 or max_categories <= 1:
            return 0.0
        probs = np.asarray([v / total for v in counter.values() if v > 0], dtype=np.float32)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        return min(entropy / math.log(max_categories), 1.0)

    def _hand_max_seats(self, hand: Dict[str, Any]) -> int:
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
            if isinstance(player, dict):
                seat = self.safe_int(player.get("seat"), default=0)
                if seat > 0:
                    seats.append(seat)

        for action in hand.get("actions") or []:
            if isinstance(action, dict):
                seat = self.safe_int(action.get("actor_seat"), default=0)
                if seat > 0:
                    seats.append(seat)

        return max(seats) if seats else 1

    def transform_one_raw(self, chunk: List[Dict[str, Any]]) -> np.ndarray:
        if not isinstance(chunk, list) or not chunk:
            return np.zeros(self.feature_dim, dtype=np.float32)

        action_counts: List[float] = []
        player_counts: List[float] = []
        occupied_ratios: List[float] = []
        showdown_flags: List[float] = []
        street_coverage_values: List[float] = []

        amount_bbs: List[float] = []
        pot_bbs: List[float] = []
        amount_to_pot: List[float] = []
        stack_bbs: List[float] = []

        total_players = 0
        short_stack_count = 0
        deep_stack_count = 0
        hero_present_count = 0
        button_present_count = 0
        visible_hole_count = 0
        showed_hand_count = 0
        zero_amount_count = 0
        pot_increase_count = 0

        action_counter: Counter[str] = Counter()
        street_counter: Counter[str] = Counter()
        actor_counter: Counter[str] = Counter()

        action_signatures: List[tuple] = []
        amount_bucket_signatures: List[tuple] = []
        low_entropy_hands = 0
        high_aggression_hands = 0

        for hand in chunk:
            if not isinstance(hand, dict):
                continue

            this_hand_actions: List[str] = []
            this_hand_buckets: List[str] = []
            this_hand_counter: Counter[str] = Counter()

            metadata = hand.get("metadata") or {}
            actions = hand.get("actions") or []
            players = hand.get("players") or []
            streets = hand.get("streets") or []
            outcome = hand.get("outcome") or {}

            bb = self.safe_float(metadata.get("bb"), default=0.0)
            max_seats = max(1, self._hand_max_seats(hand))
            hero_seat = self.safe_int(metadata.get("hero_seat"), default=0)
            button_seat = self.safe_int(metadata.get("button_seat"), default=0)

            player_seats = set()
            for player in players:
                if not isinstance(player, dict):
                    continue
                seat = self.safe_int(player.get("seat"), default=0)
                if seat > 0:
                    player_seats.add(seat)

                stack = self.safe_float(player.get("starting_stack"), default=0.0)
                stack_bb = stack / bb if stack > 0 and bb > 0 else 0.0
                if stack_bb > 0:
                    stack_bbs.append(stack_bb)
                    if stack_bb <= 20:
                        short_stack_count += 1
                    if stack_bb >= 100:
                        deep_stack_count += 1

                if player.get("hole_cards"):
                    visible_hole_count += 1
                if bool(player.get("showed_hand")):
                    showed_hand_count += 1

            if hero_seat in player_seats:
                hero_present_count += 1
            if button_seat in player_seats:
                button_present_count += 1

            total_players += len(players)
            action_counts.append(float(len(actions)))
            player_counts.append(float(len(player_seats) if player_seats else len(players)))
            occupied_ratios.append(float(len(player_seats) / max_seats))
            showdown_flags.append(1.0 if outcome.get("showdown") else 0.0)

            street_names = set()
            for street in streets:
                if isinstance(street, dict):
                    name = self.norm_text(street.get("street"))
                    if name != "unknown":
                        street_names.add(name)

            for action in actions:
                if not isinstance(action, dict):
                    continue
                action_type = self.norm_text(action.get("action_type"))
                street = self.norm_text(action.get("street"), default="preflop")
                actor = self.safe_int(action.get("actor_seat"), default=0)
                street_names.add(street)

                action_counter[action_type] += 1
                street_counter[street] += 1
                actor_counter[f"seat_{actor}" if actor > 0 else "seat_unknown"] += 1

                amount = self.safe_float(action.get("amount"), default=0.0)
                normalized_amount_bb = self.safe_float(action.get("normalized_amount_bb"), default=0.0)
                amount_bb = normalized_amount_bb if normalized_amount_bb > 0 else (amount / bb if amount > 0 and bb > 0 else 0.0)
                if amount_bb > 0:
                    amount_bbs.append(amount_bb)
                else:
                    zero_amount_count += 1

                this_hand_actions.append(action_type)
                this_hand_buckets.append(self.amount_bucket_label(amount_bb))
                this_hand_counter[action_type] += 1

                pot_before = self.safe_float(action.get("pot_before"), default=0.0)
                pot_after = self.safe_float(action.get("pot_after"), default=0.0)
                if bb > 0:
                    if pot_before > 0:
                        pot_bbs.append(pot_before / bb)
                    if pot_after > 0:
                        pot_bbs.append(pot_after / bb)

                if pot_after - pot_before > 1e-9:
                    pot_increase_count += 1
                if amount > 0 and pot_before > 0:
                    amount_to_pot.append(min(amount / pot_before, 10.0) / 10.0)

            meaningful_streets = {s for s in street_names if s in {"preflop", "flop", "turn", "river"}}
            street_coverage_values.append(len(meaningful_streets) / 4.0)

            action_signatures.append(tuple(this_hand_actions))
            amount_bucket_signatures.append(tuple(this_hand_buckets))
            if this_hand_actions:
                if self.entropy_norm(this_hand_counter, max_categories=12) <= 0.35:
                    low_entropy_hands += 1
                aggressive = sum(
                    this_hand_counter.get(name, 0) for name in ("bet", "raise", "all_in", "allin")
                )
                if aggressive / max(1, len(this_hand_actions)) >= 0.35:
                    high_aggression_hands += 1

        num_hands = max(1, len(chunk))
        total_actions = max(1, sum(action_counter.values()))
        playable_actions = max(1, sum(action_counter.get(name, 0) for name in self.ACTIONS))

        bet_raise_count = (
            action_counter.get("bet", 0)
            + action_counter.get("raise", 0)
            + action_counter.get("all_in", 0)
            + action_counter.get("allin", 0)
        )
        passive_count = action_counter.get("check", 0) + action_counter.get("call", 0)
        blind_count = sum(action_counter.get(name, 0) for name in self.BLINDS)
        money_count = sum(action_counter.get(name, 0) for name in self.MONEY_ACTIONS)

        mean_stack = self.mean(stack_bbs)
        min_stack = min(stack_bbs) if stack_bbs else 0.0
        max_stack = max(stack_bbs) if stack_bbs else 0.0

        feature_map: Dict[str, float] = {
            "num_hands_scaled": self.scaled_log(num_hands, cap=32.0),
            "avg_actions_per_hand_scaled": self.scaled_log(self.mean(action_counts), cap=64.0),
            "std_actions_per_hand_scaled": self.scaled_log(self.std(action_counts), cap=32.0),
            "avg_players_norm": min(self.mean(player_counts) / 9.0, 1.0),
            "avg_occupied_ratio": self.mean(occupied_ratios),
            "showdown_rate": self.mean(showdown_flags),
            "fold_rate": action_counter.get("fold", 0) / playable_actions,
            "check_rate": action_counter.get("check", 0) / playable_actions,
            "call_rate": action_counter.get("call", 0) / playable_actions,
            "bet_raise_rate": bet_raise_count / playable_actions,
            "all_in_rate": (action_counter.get("all_in", 0) + action_counter.get("allin", 0)) / playable_actions,
            "blind_rate": blind_count / total_actions,
            "money_action_rate": money_count / total_actions,
            "zero_amount_rate": zero_amount_count / total_actions,
            "aggression_to_passive": self.capped_ratio(bet_raise_count, passive_count, cap=5.0),
            "preflop_action_rate": street_counter.get("preflop", 0) / total_actions,
            "postflop_action_rate": (
                street_counter.get("flop", 0)
                + street_counter.get("turn", 0)
                + street_counter.get("river", 0)
            ) / total_actions,
            "street_coverage": self.mean(street_coverage_values),
            "action_entropy_norm": self.entropy_norm(action_counter, max_categories=12),
            "street_entropy_norm": self.entropy_norm(street_counter, max_categories=4),
            "actor_entropy_norm": self.entropy_norm(actor_counter, max_categories=9),
            "mean_amount_bb_scaled": self.scaled_log(self.mean(amount_bbs)),
            "std_amount_bb_scaled": self.scaled_log(self.std(amount_bbs)),
            "max_amount_bb_scaled": self.scaled_log(max(amount_bbs) if amount_bbs else 0.0),
            "mean_pot_bb_scaled": self.scaled_log(self.mean(pot_bbs)),
            "max_pot_bb_scaled": self.scaled_log(max(pot_bbs) if pot_bbs else 0.0),
            "pot_increase_rate": pot_increase_count / total_actions,
            "mean_amount_to_pot": self.mean(amount_to_pot),
            "mean_stack_bb_scaled": self.scaled_log(mean_stack),
            "min_stack_bb_scaled": self.scaled_log(min_stack),
            "max_stack_bb_scaled": self.scaled_log(max_stack),
            "stack_spread_bb_scaled": self.scaled_log(max_stack - min_stack),
            "short_stack_rate": short_stack_count / max(1, total_players),
            "deep_stack_rate": deep_stack_count / max(1, total_players),
            "hero_present_rate": hero_present_count / num_hands,
            "button_present_rate": button_present_count / num_hands,
            "visible_hole_card_rate": visible_hole_count / max(1, total_players),
            "showed_hand_rate": showed_hand_count / max(1, total_players),
            "action_signature_top_share": (
                max(Counter(action_signatures).values()) / num_hands if action_signatures else 0.0
            ),
            "action_signature_unique_share": (
                len(set(action_signatures)) / num_hands if action_signatures else 0.0
            ),
            "amount_bucket_signature_top_share": (
                max(Counter(amount_bucket_signatures).values()) / num_hands
                if amount_bucket_signatures else 0.0
            ),
            "amount_bucket_signature_unique_share": (
                len(set(amount_bucket_signatures)) / num_hands if amount_bucket_signatures else 0.0
            ),
            "low_action_entropy_hand_rate": low_entropy_hands / num_hands,
            "high_aggression_hand_rate": high_aggression_hands / num_hands,
            # Bet-sizing consistency (low CV = bot-like). Survived the
            # temporal-holdout check; the other consistency candidates hurt
            # held-out AP on this small dataset and were dropped.
            "amount_cv": self.cv(amount_bbs),
        }

        values = np.asarray([float(feature_map.get(name, 0.0)) for name in self.feature_names], dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, chunks: List[List[Dict[str, Any]]]) -> "FeatureVectorizer":
        raw = np.vstack([self.transform_one_raw(chunk) for chunk in chunks]).astype(np.float32)
        self.mean_ = raw.mean(axis=0)
        self.std_ = raw.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, chunks: List[List[Dict[str, Any]]]) -> np.ndarray:
        raw = np.vstack([self.transform_one_raw(chunk) for chunk in chunks]).astype(np.float32)
        if self.mean_ is None or self.std_ is None:
            return raw
        return ((raw - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.feature_names = list(state.get("feature_names") or self.feature_names)
        mean = state.get("mean")
        std = state.get("std")
        self.mean_ = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std_ = None if std is None else np.asarray(std, dtype=np.float32)

        if self.mean_ is not None and len(self.mean_) != len(self.feature_names):
            self.mean_ = None
        if self.std_ is not None and len(self.std_) != len(self.feature_names):
            self.std_ = None

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "FeatureVectorizer":
        obj = cls()
        obj.load_state_dict(state)
        return obj
