"""Tests that the fill simulator is wired into the management loop.

Proves process_position attaches realistic FillResults to profit-taking and
hard-exit sells (none on hold), and that realised_pnl_usd aggregates net
proceeds minus cost basis across a session's outcomes.
"""

from __future__ import annotations

import asyncio

import pytest

import main
from strategies.hard_exit import MarketData
from strategies.profit_taking import Position


def run(coro):
    return asyncio.run(coro)


def test_profit_take_attaches_a_simulated_fill() -> None:
    pos = Position(entry_price=1.0, original_size=1000.0)
    md = MarketData(current_price=2.0, current_liquidity=100_000.0,
                    entry_liquidity=100_000.0)  # 2x -> sell 50% (500 tokens)

    outcome = run(main.process_position(pos, md))

    assert outcome.path == "profit_taking"
    assert len(outcome.fills) == 1
    fill = outcome.fills[0]
    assert fill.requested_tokens == pytest.approx(500.0)
    assert fill.filled is True
    assert 0.0 < fill.net_proceeds_usd < fill.notional_usd  # friction applied

    pnl = main.realised_pnl_usd([outcome], pos.entry_price)
    assert pnl == pytest.approx(fill.net_proceeds_usd - 500.0 * 1.0)


def test_hard_exit_attaches_full_exit_fill() -> None:
    pos = Position(entry_price=1.0, original_size=1000.0)
    md = MarketData(current_price=1.0, current_liquidity=50_000.0,
                    entry_liquidity=100_000.0)  # 50% < 70% floor -> liquidity_drop

    outcome = run(main.process_position(pos, md))

    assert outcome.path == "hard_exit"
    assert outcome.exit_decision is not None
    assert outcome.exit_decision.trigger == "liquidity_drop"
    assert len(outcome.fills) == 1
    assert outcome.fills[0].requested_tokens == pytest.approx(1000.0)  # full position


def test_hold_has_no_fills() -> None:
    pos = Position(entry_price=1.0, original_size=1000.0)
    md = MarketData(current_price=1.5, current_liquidity=100_000.0,
                    entry_liquidity=100_000.0)  # 1.5x, healthy -> hold

    outcome = run(main.process_position(pos, md))

    assert outcome.path == "hold"
    assert outcome.fills == []
    assert main.realised_pnl_usd([outcome, None], pos.entry_price) == 0.0
