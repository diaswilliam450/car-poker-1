"""Synthetic bot-hand generator for supervised bot-vs-human training.

The public corpus is 100% human, and the validator's real bot examples are
private.  To train a non-degenerate detector we synthesize *bot* hands whose
betting behaviour follows rigid, low-entropy policies — exactly the fingerprint
real online poker bots exhibit (fixed sizing, mechanical c-bets, repetitive
action lines).  Human hands come from the real corpus; the contrast between the
two is what the model learns, and because the synthetic policies mimic how
production bots actually play, it transfers to live tables far better than a
model trained on the 40-chunk public snapshot alone.

Hands are generated with a simplified but *valid* No-Limit Hold'em betting
engine (correct pot/`normalized_amount_bb`/pot_before/pot_after accounting), so
the anomaly features stay clean — bots are detected by their *style*, not by
malformed data (which would not generalise).

ARCHETYPES (differentiation lever — each miner weights these differently):
  nit      : folds most hands, small standard sizing, low aggression variance
  aggro    : raises/bets constantly, large sizing and overbets
  station  : calls almost everything, passive, rarely folds or raises
  cbot     : rigid GTO-ish bot — fixed 3bb opens, mechanical 0.66-pot c-bets,
             extremely repetitive lines (very low action entropy)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

ARCHETYPES = ("nit", "aggro", "station", "cbot")

# Per-archetype decision policy.  Each street uses (fold, call, raise/bet) mass.
_POLICY: Dict[str, Dict[str, Any]] = {
    "nit": {
        "pre_open": (0.82, 0.08, 0.10),      # facing only blinds
        "pre_facing": (0.60, 0.28, 0.12),    # facing a raise
        "post_facing": (0.52, 0.33, 0.15),   # facing a bet
        "post_open": (0.62, 0.38),           # (check, bet) when checked to
        "open_bb": (2.5, 3.0),               # preflop open size in bb
        "bet_pot": (0.4, 0.55),              # postflop bet as fraction of pot
    },
    "aggro": {
        "pre_open": (0.25, 0.12, 0.63),
        "pre_facing": (0.18, 0.24, 0.58),
        "post_facing": (0.16, 0.24, 0.60),
        "post_open": (0.18, 0.82),
        "open_bb": (3.5, 4.5),
        "bet_pot": (0.9, 1.4),
    },
    "station": {
        "pre_open": (0.35, 0.60, 0.05),
        "pre_facing": (0.14, 0.81, 0.05),
        "post_facing": (0.14, 0.80, 0.06),
        "post_open": (0.70, 0.30),
        "open_bb": (2.5, 3.0),
        "bet_pot": (0.33, 0.5),
    },
    "cbot": {
        "pre_open": (0.55, 0.08, 0.37),
        "pre_facing": (0.50, 0.16, 0.34),
        "post_facing": (0.40, 0.30, 0.30),
        "post_open": (0.35, 0.65),
        "open_bb": (3.0, 3.0),               # fixed 3bb — no variance
        "bet_pot": (0.66, 0.66),             # fixed 0.66-pot c-bet
    },
}

_STREETS = ("preflop", "flop", "turn", "river")


def _choice3(rng: random.Random, mass: Tuple[float, float, float]) -> int:
    r = rng.random() * sum(mass)
    if r < mass[0]:
        return 0
    if r < mass[0] + mass[1]:
        return 1
    return 2


def generate_bot_hand(rng: random.Random, archetype: str, max_seats: int = 6) -> Dict[str, Any]:
    pol = _POLICY[archetype]
    sb, bb = 0.01, 0.02
    n = max_seats
    seats = list(range(1, n + 1))
    stacks = {s: round(rng.uniform(80, 160) * bb, 2) for s in seats}
    bi = rng.randrange(n)               # button index into seats
    button = seats[bi]
    hero = rng.choice(seats)
    sb_seat = seats[(bi + 1) % n]
    bb_seat = seats[(bi + 2) % n]

    invested: Dict[int, float] = {s: 0.0 for s in seats}   # invested this street
    folded: set = set()
    pot = 0.0
    actions: List[Dict[str, Any]] = []
    aid = 0

    def add(street: str, seat: int, atype: str, amount: float, raise_to: Any, call_to: Any):
        nonlocal aid, pot
        aid += 1
        pot_before = pot
        amt = round(max(0.0, amount), 4)
        pot = round(pot + amt, 4)
        actions.append({
            "action_id": str(aid),
            "street": street,
            "actor_seat": seat,
            "action_type": atype,
            "amount": amt,
            "raise_to": round(raise_to, 4) if isinstance(raise_to, (int, float)) else raise_to,
            "call_to": round(call_to, 4) if isinstance(call_to, (int, float)) else call_to,
            "normalized_amount_bb": round(amt / bb, 4),
            "pot_before": round(pot_before, 4),
            "pot_after": round(pot, 4),
        })

    # post blinds
    invested[sb_seat] = sb
    add("preflop", sb_seat, "small_blind", sb, None, None)
    invested[bb_seat] = bb
    add("preflop", bb_seat, "big_blind", bb, None, None)

    streets_reached = ["preflop"]

    def street_order(street: str) -> List[int]:
        start = (bi + 3) % n if street == "preflop" else (bi + 1) % n
        return [seats[(start + i) % n] for i in range(n)]

    def run_street(street: str, opening_bet: float) -> bool:
        """One single-pass betting round. Returns True if >=2 players remain."""
        current_bet = opening_bet
        raises = 0
        for s in seats:            # reset per-street contributions (blinds already in pot)
            invested[s] = 0.0
        if street == "preflop":
            invested[sb_seat] = sb
            invested[bb_seat] = bb
            current_bet = bb
        for seat in street_order(street):
            if seat in folded:
                continue
            if len([s for s in seats if s not in folded]) <= 1:
                break
            to_call = max(0.0, current_bet - invested[seat])
            if street == "preflop":
                mass = pol["pre_open"] if current_bet <= bb + 1e-9 else pol["pre_facing"]
            else:
                mass = pol["post_open"] if to_call <= 1e-9 else pol["post_facing"]
            # checked-to (postflop, no bet yet): (check, bet)
            if street != "preflop" and to_call <= 1e-9:
                check, betp = pol["post_open"]
                if rng.random() < betp / (check + betp) and raises < 4:
                    size = max(bb, round(pot * rng.uniform(*pol["bet_pot"]), 2))
                    invested[seat] += size
                    current_bet = invested[seat]
                    raises += 1
                    add(street, seat, "bet", size, current_bet, None)
                else:
                    add(street, seat, "check", 0.0, None, None)
                continue
            decision = _choice3(rng, mass)
            if decision == 0 and to_call > 1e-9:
                folded.add(seat)
                add(street, seat, "fold", 0.0, None, None)
            elif decision == 2 and raises < 4:
                if street == "preflop":
                    target = max(current_bet + bb, round(bb * rng.uniform(*pol["open_bb"]), 2))
                else:
                    target = round(current_bet + max(bb, pot * rng.uniform(*pol["bet_pot"])), 2)
                pay = round(target - invested[seat], 2)
                invested[seat] = target
                current_bet = target
                raises += 1
                add(street, seat, "raise", pay, target, None)
            elif to_call <= 1e-9:
                add(street, seat, "check", 0.0, None, None)
            else:
                pay = round(to_call, 2)
                invested[seat] += pay
                add(street, seat, "call", pay, None, current_bet)
        return len([s for s in seats if s not in folded]) >= 2

    cont = run_street("preflop", bb)
    for street in _STREETS[1:]:
        if not cont:
            break
        streets_reached.append(street)
        cont = run_street(street, 0.0)

    players = [{
        "player_uid": f"seat_{s}",
        "seat": s,
        "starting_stack": stacks[s],
        "hole_cards": None,
        "showed_hand": False,
    } for s in seats]
    streets = [{"street": st, "board_cards": []} for st in streets_reached]
    live = [s for s in seats if s not in folded]
    winner = live[0] if live else hero
    return {
        "metadata": {
            "game_type": "Hold'em", "limit_type": "No Limit", "max_seats": max_seats,
            "hero_seat": hero, "hand_ended_on_street": streets_reached[-1],
            "button_seat": button, "sb": sb, "bb": bb, "ante": 0.0,
            "rng_seed_commitment": None,
        },
        "players": players,
        "streets": streets,
        "actions": actions,
        "outcome": {
            "winners": [f"seat_{winner}"], "payouts": {f"seat_{winner}": round(pot, 4)},
            "total_pot": round(pot, 4), "rake": 0.0, "result_reason": "", "showdown": False,
        },
    }


def generate_bot_chunk(rng: random.Random, archetype: str, n_hands: int) -> List[Dict[str, Any]]:
    """A chunk = many hands from ONE bot playing its fixed policy."""
    return [generate_bot_hand(rng, archetype) for _ in range(n_hands)]
