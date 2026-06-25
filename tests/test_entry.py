"""Tests for the post-pump dip-buy entry decision (strategies.entry).

Covers the required scenarios: a clean stabilised dip (BUY), a still-falling
knife (WAIT), a retrace past the band (SKIP), and rejections for an
out-of-band market cap / no-recent-ATH (SKIP). Plus the proven-runner gate and
the too-early/too-hot WAIT branches.

All cases are pure (no network): ``EntryMarketData`` is constructed directly
and ``now`` is pinned so ATH age is deterministic.
"""

from __future__ import annotations

import asyncio

from strategies.entry import (
    EntryAction,
    EntryMarketData,
    evaluate_entry,
)

NOW = 1_000_000.0
HOUR = 3600.0
MINT = "TESTMINT"


def run(coro):
    return asyncio.run(coro)


def make_market(**overrides) -> EntryMarketData:
    """A baseline BUY-eligible market: proven runner, 40% dip, stabilised.

    Defaults: ATH 5h ago (inside 2-12h), 10x volume spike, pullback 40% (in
    30-50%), last samples within ~3% band, healthy liquidity not drained.
    """
    base = dict(
        current_price=0.60,
        ath_price=1.00,
        ath_timestamp=NOW - 5 * HOUR,
        # Decline into a tight consolidation around 0.60 (last 5 within ~3%).
        price_history=[1.00, 0.90, 0.75, 0.62, 0.61, 0.60, 0.605, 0.60],
        # Baseline ~100 with a 1000 spike -> 10x.
        volume_history=[100.0, 100.0, 100.0, 100.0, 1000.0],
        current_liquidity=50_000.0,
        pre_dip_liquidity=60_000.0,
        sample_interval_minutes=1.0,
        market_cap_usd=1_000_000.0,
        now=NOW,
    )
    base.update(overrides)
    return EntryMarketData(**base)


def test_clean_dip_buy() -> None:
    """Stabilised 40% dip on a proven runner -> BUY, sized with stops."""
    decision = run(evaluate_entry(MINT, make_market()))

    assert decision.action is EntryAction.BUY
    assert decision.proven_runner is True
    assert decision.consolidated is True
    assert decision.position is not None
    # 1% of the default $10k portfolio = $100 notional, capped under pool frac.
    assert decision.notional_usd == 100.0
    assert decision.position.original_size == 100.0 / 0.60
    # Advisory risk notes present.
    assert decision.stop_loss_pct == 0.20
    assert decision.stop_loss_price == 0.60 * 0.80
    assert decision.time_stop_minutes == 240.0


def test_still_falling_knife_waits() -> None:
    """Pullback is in-band but price is still dropping vertically -> WAIT."""
    # Last 5 samples span a wide range (~29%): not stabilised.
    knife = make_market(
        price_history=[1.00, 0.95, 0.85, 0.78, 0.70, 0.64, 0.60],
    )
    decision = run(evaluate_entry(MINT, knife))

    assert decision.action is EntryAction.WAIT
    assert decision.proven_runner is True
    assert decision.consolidated is False
    assert "falling knife" in decision.reason


def test_retraced_too_far_skips() -> None:
    """Retrace beyond the max band -> SKIP (likely dying)."""
    too_far = make_market(current_price=0.40)  # 60% off ATH (> 50%)
    decision = run(evaluate_entry(MINT, too_far))

    assert decision.action is EntryAction.SKIP
    assert "dropped too far" in decision.reason


def test_no_recent_ath_skips() -> None:
    """ATH older than the window -> SKIP (no recent ATH)."""
    stale = make_market(ath_timestamp=NOW - 20 * HOUR)
    decision = run(evaluate_entry(MINT, stale))

    assert decision.action is EntryAction.SKIP
    assert "no recent ATH" in decision.reason


def test_out_of_mcap_skips() -> None:
    """Market cap below the band -> SKIP, before any runner checks."""
    tiny = make_market(market_cap_usd=10_000.0)
    decision = run(evaluate_entry(MINT, tiny))

    assert decision.action is EntryAction.SKIP
    assert "market cap" in decision.reason


def test_not_proven_runner_low_volume_skips() -> None:
    """No volume spike -> fails the proven-runner gate -> SKIP."""
    flat = make_market(volume_history=[100.0, 110.0, 105.0, 100.0, 120.0])
    decision = run(evaluate_entry(MINT, flat))

    assert decision.action is EntryAction.SKIP
    assert "not a proven runner" in decision.reason
    assert decision.volume_spike_ratio < 5.0


def test_too_early_shallow_pullback_waits() -> None:
    """Proven runner but pullback too shallow -> WAIT (too early)."""
    shallow = make_market(
        current_price=0.90,  # only 10% off ATH (< 30%)
        price_history=[1.00, 0.95, 0.92, 0.91, 0.90, 0.905, 0.90],
    )
    decision = run(evaluate_entry(MINT, shallow))

    assert decision.action is EntryAction.WAIT
    assert "too early" in decision.reason


def test_too_hot_recent_ath_waits() -> None:
    """ATH within the min-age window -> WAIT (too hot)."""
    hot = make_market(ath_timestamp=NOW - 1 * HOUR)  # 1h < 2h min
    decision = run(evaluate_entry(MINT, hot))

    assert decision.action is EntryAction.WAIT
    assert "too hot" in decision.reason


def test_liquidity_drained_skips() -> None:
    """Liquidity drained beyond the limit vs pre-dip -> SKIP."""
    drained = make_market(current_liquidity=30_000.0, pre_dip_liquidity=100_000.0)
    decision = run(evaluate_entry(MINT, drained))

    assert decision.action is EntryAction.SKIP
    assert "drained" in decision.reason


def test_invalid_price_skips() -> None:
    """Non-positive price -> fail-closed SKIP."""
    bad = make_market(current_price=0.0)
    decision = run(evaluate_entry(MINT, bad))

    assert decision.action is EntryAction.SKIP
    assert decision.position is None
