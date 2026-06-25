"""Tests for the per-loop strategy orchestration in main.process_position.

The central guarantee from CLAUDE.md: hard-exit checks run BEFORE profit-taking,
and when a hard exit fires, profit-taking is skipped ENTIRELY for that position.
"""

from __future__ import annotations

import asyncio

import pytest

import main
from strategies import profit_taking
from strategies.hard_exit import MarketData
from strategies.profit_taking import Position

ENTRY_PRICE = 0.001
ORIGINAL_SIZE = 1000.0


def make_position() -> Position:
    return Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)


def healthy_market(current_price: float) -> MarketData:
    """Market snapshot with no hard-exit trigger; price set by caller."""
    return MarketData(
        current_price=current_price,
        rolling_volume_15m=50_000.0,
        peak_volume_15m=50_000.0,
        current_liquidity=100_000.0,
        entry_liquidity=100_000.0,
        candle_1m_open=current_price,
        candle_1m_close=current_price,
    )


def test_hard_exit_skips_profit_taking(monkeypatch) -> None:
    """When a hard-exit trigger fires, evaluate_take_profit must NOT be called."""
    calls: list = []

    async def spy_take_profit(position, current_price, dry_run=True):
        calls.append((position, current_price))
        return []

    # Patch the module attribute that process_position resolves at call time.
    monkeypatch.setattr(profit_taking, "evaluate_take_profit", spy_take_profit)

    pos = make_position()
    # Price 50% below entry -> hard_stop_loss fires.
    market = healthy_market(current_price=ENTRY_PRICE * 0.5)

    outcome = asyncio.run(main.process_position(pos, market))

    assert outcome.path == "hard_exit"
    assert outcome.exit_decision is not None
    assert outcome.exit_decision.trigger == "hard_stop_loss"
    # The core assertion: profit-taking was never invoked for this position.
    assert calls == []


def test_no_hard_exit_runs_profit_taking(monkeypatch) -> None:
    """With no hard-exit trigger, evaluate_take_profit IS called exactly once."""
    calls: list = []

    async def spy_take_profit(position, current_price, dry_run=True):
        calls.append((position, current_price))
        return []

    monkeypatch.setattr(profit_taking, "evaluate_take_profit", spy_take_profit)

    pos = make_position()
    market = healthy_market(current_price=ENTRY_PRICE * 3)  # healthy, 3x

    outcome = asyncio.run(main.process_position(pos, market))

    assert outcome.path == "hold"  # spy returns [] -> no tiers fired
    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(ENTRY_PRICE * 3)


def test_no_hard_exit_real_profit_tier_fires() -> None:
    """End-to-end (no patching): a clean 2x produces the profit_taking path."""
    pos = make_position()
    market = healthy_market(current_price=ENTRY_PRICE * 2)  # exactly 2x

    outcome = asyncio.run(main.process_position(pos, market))

    assert outcome.path == "profit_taking"
    assert [a.tier for a in outcome.sell_actions] == ["2x"]
