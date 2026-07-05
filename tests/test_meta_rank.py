"""Tests for the meta/KOL ranking boost (scanner.meta_rank).

Proves the boost is a RANKING multiplier, never a trigger and never a penalty:
buyer breadth and a KOL hold raise it, an absent/failed signal floors it at 1.0
(neutral — never below), and it is bounded by the ceiling.
"""

from __future__ import annotations

import asyncio

from data.onchain import KolSignal, OnchainFlow
from scanner.meta_rank import (
    MAX_META_BOOST,
    NEUTRAL_BOOST,
    compute_meta_boost,
    rank_boost_for,
)


def run(coro):
    return asyncio.run(coro)


def flow(*, buyers: int, available: bool = True) -> OnchainFlow:
    return OnchainFlow(buyer_wallets=buyers, available=available)


def kol(*, hold: bool, available: bool = True) -> KolSignal:
    return KolSignal(any_hold=hold, holding_wallets=("K",) if hold else (), available=available)


# --- Fail-closed neutrality (never a penalty) -------------------------------

def test_absent_signals_are_neutral_never_below_one() -> None:
    """Unavailable flow + KOL -> exactly 1.0 (neutral), never below."""
    boost = compute_meta_boost(
        OnchainFlow(available=False), KolSignal(available=False)
    )
    assert boost.boost == 1.0
    assert boost.is_neutral
    assert boost.available is False


def test_none_signals_are_neutral() -> None:
    """None inputs (nothing fetched) -> neutral 1.0, never raises."""
    boost = compute_meta_boost(None, None)
    assert boost.boost == 1.0
    assert boost.buyer_breadth == 0 and boost.kol_hit is False


def test_zero_buyers_available_is_still_neutral() -> None:
    """A token that is safe/moving but shows no breadth is not penalised."""
    boost = compute_meta_boost(flow(buyers=0), kol(hold=False))
    assert boost.boost == 1.0  # floored at neutral, never below


# --- Buyer breadth raises the boost -----------------------------------------

def test_broad_buyers_boost_higher_than_few() -> None:
    """More distinct buyers -> a higher (but bounded) boost."""
    few = compute_meta_boost(flow(buyers=2), kol(hold=False))
    broad = compute_meta_boost(flow(buyers=20), kol(hold=False))
    assert broad.boost > few.boost > 1.0
    assert broad.buyer_breadth == 20


def test_buyer_breadth_saturates_at_target() -> None:
    """Beyond the breadth target the buyer bonus is capped (no runaway)."""
    at_target = compute_meta_boost(flow(buyers=20), kol(hold=False))
    way_over = compute_meta_boost(flow(buyers=200), kol(hold=False))
    assert at_target.boost == way_over.boost


# --- KOL hold adds a small flat bonus ---------------------------------------

def test_kol_hold_adds_bonus() -> None:
    """A tracked wallet holding the token raises the boost above breadth alone."""
    no_kol = compute_meta_boost(flow(buyers=5), kol(hold=False))
    with_kol = compute_meta_boost(flow(buyers=5), kol(hold=True))
    assert with_kol.boost > no_kol.boost
    assert with_kol.kol_hit is True


def test_kol_unavailable_adds_nothing() -> None:
    """An unavailable KOL signal contributes no bonus (fail-closed)."""
    boost = compute_meta_boost(flow(buyers=5), KolSignal(available=False))
    assert boost.kol_hit is False


# --- Bounded --------------------------------------------------------------

def test_boost_is_bounded_by_ceiling() -> None:
    """Boost stays within [1.0, ceiling]; the ceiling clamp binds when it must."""
    # Default weights top out at 1 + 0.25 + 0.15 = 1.40 (headroom under the cap).
    maxed = compute_meta_boost(flow(buyers=1000), kol(hold=True))
    assert 1.0 <= maxed.boost <= MAX_META_BOOST
    assert abs(maxed.boost - 1.40) < 1e-9
    # With weights that would overshoot, the ceiling actually clamps.
    over = compute_meta_boost(
        flow(buyers=1000), kol(hold=True), buyer_weight=1.0, kol_weight=1.0
    )
    assert over.boost == MAX_META_BOOST


# --- rank_boost_for is fail-closed ------------------------------------------

def test_rank_boost_for_no_endpoint_is_neutral() -> None:
    """No Helius endpoint / no curated wallets -> a neutral boost, never raises."""
    # rpc_url="" makes the on-chain signals unavailable without any network call.
    boost = run(rank_boost_for("MINT", 1.0, rpc_url="", kol_wallets=[]))
    assert boost.boost == 1.0
    assert boost.is_neutral
