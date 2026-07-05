"""Tests for the entry-strategy dispatcher (main.evaluate_entry_strategy).

Covers the per-strategy routing and, for ``"both"``, the two guarantees:

    * the first strategy to return BUY wins (dip is evaluated first);
    * when neither BUYs, the MORE INFORMATIVE decision is surfaced (a WAIT
      near-miss over a SKIP rejection), tie-breaking to the dip (primary) so the
      reported reason is deterministic.

No network: ``EntryMarketData`` is constructed directly and ``now`` pinned.
"""

from __future__ import annotations

import asyncio
import time

import main
from strategies.entry import EntryAction, EntryDecision, EntryMarketData
from strategies.entry_momentum import MomentumDecision

NOW = time.time()
HOUR = 3600.0
MINT = "DISPATCHMINT"


def run(coro):
    return asyncio.run(coro)


def dispatch(market: EntryMarketData, strategy: str):
    return run(
        main.evaluate_entry_strategy(MINT, market, strategy=strategy, dry_run=True)
    )


def test_momentum_only_routes_to_momentum() -> None:
    """strategy='momentum' returns a MomentumDecision (BUY on a clean breakout)."""
    breakout = EntryMarketData(
        current_price=1.18, ath_price=1.18, ath_timestamp=NOW,
        price_history=[1.00, 1.15, 0.95, 1.00, 1.05, 1.18],
        volume_history=[100.0, 110.0, 90.0, 300.0, 320.0, 310.0],
        current_liquidity=50_000.0, pre_dip_liquidity=50_000.0,
        sample_interval_minutes=5.0, market_cap_usd=1_000_000.0, now=NOW,
    )
    decision = dispatch(breakout, "momentum")
    assert isinstance(decision, MomentumDecision)
    assert decision.action is EntryAction.BUY


def test_both_first_buy_wins_momentum() -> None:
    """A breakout the dip declines is bought by momentum under 'both'."""
    breakout = EntryMarketData(
        current_price=1.18, ath_price=1.18, ath_timestamp=NOW,
        price_history=[1.00, 1.15, 0.95, 1.00, 1.05, 1.18],
        volume_history=[100.0, 110.0, 90.0, 300.0, 320.0, 310.0],
        current_liquidity=50_000.0, pre_dip_liquidity=50_000.0,
        sample_interval_minutes=5.0, market_cap_usd=1_000_000.0, now=NOW,
    )
    decision = dispatch(breakout, "both")
    assert isinstance(decision, MomentumDecision)
    assert decision.action is EntryAction.BUY


def test_both_surfaces_momentum_wait_over_dip_skip() -> None:
    """dip SKIP + momentum WAIT -> the WAIT (more informative) is surfaced."""
    # Flat/weak move: momentum is WAIT (too weak); the dip is SKIP (no volume
    # spike -> not a proven runner).
    weak = EntryMarketData(
        current_price=1.04, ath_price=1.04, ath_timestamp=NOW - 5 * HOUR,
        price_history=[1.00, 1.02, 1.01, 1.03, 1.02, 1.04],
        volume_history=[100.0, 110.0, 105.0, 100.0, 120.0],  # ~1.2x -> no spike
        current_liquidity=50_000.0, pre_dip_liquidity=60_000.0,
        sample_interval_minutes=5.0, market_cap_usd=1_000_000.0, now=NOW,
    )
    # Sanity: the two strategies disagree in action as intended.
    assert dispatch(weak, "dip").action is EntryAction.SKIP
    assert dispatch(weak, "momentum").action is EntryAction.WAIT

    decision = dispatch(weak, "both")
    assert isinstance(decision, MomentumDecision)  # the WAIT came from momentum
    assert decision.action is EntryAction.WAIT
    assert "momentum" in decision.reason


def test_both_surfaces_dip_wait_over_momentum_skip() -> None:
    """dip WAIT + momentum SKIP -> the WAIT (more informative) is surfaced."""
    # Strong run toward the ATH from a low base: momentum is SKIP (overextended,
    # run-up > cap); the dip is WAIT (proven runner, pullback too shallow yet).
    running = EntryMarketData(
        current_price=1.00, ath_price=1.00, ath_timestamp=NOW - 5 * HOUR,
        price_history=[0.40, 0.50, 0.70, 0.85, 0.95, 1.00],
        volume_history=[100.0, 100.0, 100.0, 100.0, 1000.0],  # 10x -> proven runner
        current_liquidity=50_000.0, pre_dip_liquidity=60_000.0,
        sample_interval_minutes=5.0, market_cap_usd=1_000_000.0, now=NOW,
    )
    assert dispatch(running, "dip").action is EntryAction.WAIT
    assert dispatch(running, "momentum").action is EntryAction.SKIP

    decision = dispatch(running, "both")
    assert isinstance(decision, EntryDecision)  # the WAIT came from the dip
    assert decision.action is EntryAction.WAIT
    assert "too early" in decision.reason


def test_both_ties_break_to_dip() -> None:
    """When both SKIP (equal rank), the dip (primary) decision is surfaced."""
    # Empty history -> both strategies fail closed to SKIP.
    empty = EntryMarketData(
        current_price=1.00, ath_price=1.00, ath_timestamp=NOW,
        price_history=[], volume_history=[],
        current_liquidity=50_000.0, pre_dip_liquidity=50_000.0,
        sample_interval_minutes=5.0, market_cap_usd=1_000_000.0, now=NOW,
    )
    assert dispatch(empty, "dip").action is EntryAction.SKIP
    assert dispatch(empty, "momentum").action is EntryAction.SKIP

    decision = dispatch(empty, "both")
    assert isinstance(decision, EntryDecision)  # tie-break to the dip (primary)
    assert decision.action is EntryAction.SKIP
