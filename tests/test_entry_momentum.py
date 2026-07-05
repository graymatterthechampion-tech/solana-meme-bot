"""Tests for the momentum/breakout entry decision (strategies.entry_momentum).

Covers the required scenarios: a clean volume-confirmed breakout (BUY), a move
that is too weak yet (WAIT), the two critical overextension rejections — RSI
overbought and a run past the run-up cap (SKIP) — and a single unsustained
volume spike (WAIT). Plus a liquidity-floor SKIP and a fail-closed bad-price
SKIP for parity with the dip-buy module.

All cases are pure (no network): ``EntryMarketData`` is constructed directly.
The sample interval is pinned to 5min so the 15min momentum window spans the
last 3 samples, keeping the hand-crafted series short and readable.
"""

from __future__ import annotations

import asyncio

from strategies.entry import EntryAction, EntryMarketData
from strategies.entry_momentum import evaluate_entry_momentum

NOW = 1_000_000.0
MINT = "MOMENTUMMINT"


def run(coro):
    return asyncio.run(coro)


def make_market(**overrides) -> EntryMarketData:
    """A baseline BUY-eligible breakout market.

    Defaults: +24% over the last 15min (3 x 5min samples), run-up ~24% off the
    recent base (< 100% cap), RSI ~66 (< 75), and sustained elevated volume
    across the move (no single candle dominating). ``ath_*`` are unused by the
    momentum strategy but required by the shared container.
    """
    base = dict(
        current_price=1.18,
        ath_price=1.18,
        ath_timestamp=NOW,
        # Up into the breakout with a real dip in the middle (keeps RSI < 75).
        price_history=[1.00, 1.15, 0.95, 1.00, 1.05, 1.18],
        # Low baseline (100) then a sustained elevated cluster (~300+).
        volume_history=[100.0, 110.0, 90.0, 300.0, 320.0, 310.0],
        current_liquidity=50_000.0,
        pre_dip_liquidity=50_000.0,
        sample_interval_minutes=5.0,
        market_cap_usd=1_000_000.0,
        now=NOW,
    )
    base.update(overrides)
    return EntryMarketData(**base)


def test_clean_momentum_buy() -> None:
    """Strong, volume-confirmed, not-overextended breakout -> BUY, sized."""
    decision = run(evaluate_entry_momentum(MINT, make_market()))

    assert decision.action is EntryAction.BUY
    assert decision.momentum_pct >= 0.15
    assert decision.run_up_pct < 1.0
    assert decision.rsi is not None and decision.rsi <= 75.0
    assert decision.overextended is False
    assert decision.volume_confirmed is True
    assert decision.position is not None
    # 1% of the default $10k portfolio = $100 notional, under the pool-frac cap.
    assert decision.notional_usd == 100.0
    assert decision.position.original_size == 100.0 / 1.18
    # Advisory breakout risk notes present (tighter stop than the dip-buy).
    assert decision.stop_loss_pct == 0.15
    assert decision.stop_loss_price == 1.18 * (1.0 - 0.15)
    assert decision.time_stop_minutes == 120.0


def test_weak_momentum_waits() -> None:
    """Price barely moved over the window -> WAIT (too weak, no trade)."""
    weak = make_market(
        current_price=1.04,
        price_history=[1.00, 1.02, 1.01, 1.03, 1.02, 1.04],  # ~3% over 15min
    )
    decision = run(evaluate_entry_momentum(MINT, weak))

    assert decision.action is EntryAction.WAIT
    assert "momentum" in decision.reason
    assert decision.momentum_pct < 0.15
    assert decision.position is None


def test_overextended_rsi_skips() -> None:
    """Momentum is strong and under the run-up cap, but RSI is overbought -> SKIP.

    A steady, pullback-free climb keeps the run-up modest (~45%) while driving
    RSI to 100 — the "do not buy the top" guard must fire on RSI alone.
    """
    overbought = make_market(
        current_price=1.45,
        price_history=[1.00, 1.10, 1.12, 1.25, 1.30, 1.45],  # monotonic, no dips
    )
    decision = run(evaluate_entry_momentum(MINT, overbought))

    assert decision.action is EntryAction.SKIP
    assert "RSI" in decision.reason
    assert decision.overextended is True
    assert decision.rsi is not None and decision.rsi > 75.0
    assert decision.run_up_pct <= 1.0  # run-up did NOT trip; RSI did
    assert decision.position is None


def test_overextended_runup_skips() -> None:
    """Already run more than the run-up cap off the recent base -> SKIP."""
    parabolic = make_market(
        current_price=1.60,
        price_history=[0.50, 0.70, 0.90, 1.10, 1.30, 1.60],  # +220% off the base
    )
    decision = run(evaluate_entry_momentum(MINT, parabolic))

    assert decision.action is EntryAction.SKIP
    assert "overextended" in decision.reason
    assert decision.overextended is True
    assert decision.run_up_pct > 1.0
    assert decision.position is None


def test_single_volume_spike_not_sustained_waits() -> None:
    """Good price momentum but the volume move is one lone candle -> WAIT.

    A single dominating spike (not sustained across the window) fails volume
    confirmation, so the breakout is not yet tradable.
    """
    spike = make_market(
        volume_history=[100.0, 110.0, 90.0, 100.0, 900.0, 100.0],  # one huge candle
    )
    decision = run(evaluate_entry_momentum(MINT, spike))

    assert decision.action is EntryAction.WAIT
    assert "spike" in decision.reason
    assert decision.volume_confirmed is False
    assert decision.peak_volume_share > 0.6
    assert decision.position is None


def test_liquidity_below_floor_skips() -> None:
    """A confirmed breakout on a pool below the liquidity floor -> SKIP."""
    thin = make_market(current_liquidity=5_000.0)  # below the $10k floor
    decision = run(evaluate_entry_momentum(MINT, thin))

    assert decision.action is EntryAction.SKIP
    assert "liquidity" in decision.reason
    assert decision.position is None


def test_invalid_price_skips() -> None:
    """Non-positive price -> fail-closed SKIP."""
    bad = make_market(current_price=0.0)
    decision = run(evaluate_entry_momentum(MINT, bad))

    assert decision.action is EntryAction.SKIP
    assert decision.position is None


def test_insufficient_history_skips() -> None:
    """Too little price/volume history -> fail-closed SKIP (never guess)."""
    thin_history = make_market(price_history=[1.18], volume_history=[300.0])
    decision = run(evaluate_entry_momentum(MINT, thin_history))

    assert decision.action is EntryAction.SKIP
    assert "insufficient" in decision.reason
    assert decision.position is None
