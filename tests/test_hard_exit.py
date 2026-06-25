"""Tests for the hard-exit strategy.

Each trigger is exercised in isolation by starting from a healthy market
snapshot and perturbing a single signal. The async evaluator is driven
synchronously via ``asyncio.run`` (no pytest-asyncio plugin required).
"""

from __future__ import annotations

import asyncio

import pytest

from strategies.hard_exit import MarketData, evaluate_hard_exit
from strategies.profit_taking import Position

ENTRY_PRICE = 0.001
ORIGINAL_SIZE = 1000.0
MOONBAG = 100.0  # remaining position under test (the 10% moonbag)


def make_position() -> Position:
    """A position already reduced to its moonbag, to prove a FULL exit."""
    return Position(
        entry_price=ENTRY_PRICE,
        original_size=ORIGINAL_SIZE,
        remaining=MOONBAG,
        sold_2x=True,
        sold_5x=True,
        sold_10x=True,
    )


def healthy_market() -> MarketData:
    """A market snapshot where no trigger should fire."""
    return MarketData(
        current_price=ENTRY_PRICE * 1.5,   # well above stop-loss
        rolling_volume_15m=50_000.0,
        peak_volume_15m=50_000.0,
        current_liquidity=100_000.0,
        entry_liquidity=100_000.0,
        largest_single_sell=0.0,
        top_wallet_sells=0,
        candle_1m_open=ENTRY_PRICE * 1.5,
        candle_1m_close=ENTRY_PRICE * 1.5,
    )


def evaluate(position: Position, market: MarketData):
    return asyncio.run(evaluate_hard_exit(position, market))


def test_no_trigger_holds_position() -> None:
    decision = evaluate(make_position(), healthy_market())
    assert decision.should_exit is False
    assert decision.trigger is None
    assert decision.tokens_to_sell == 0.0


def test_hard_stop_loss_fires() -> None:
    market = healthy_market()
    market.current_price = ENTRY_PRICE * 0.79  # 21% below entry
    decision = evaluate(make_position(), market)
    assert decision.should_exit is True
    assert decision.trigger == "hard_stop_loss"
    # Full remaining position, moonbag included.
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)


def test_volume_collapse_fires() -> None:
    market = healthy_market()
    market.rolling_volume_15m = market.peak_volume_15m * 0.29  # 71% drop
    decision = evaluate(make_position(), market)
    assert decision.should_exit is True
    assert decision.trigger == "volume_collapse"
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)


def test_liquidity_drop_fires() -> None:
    market = healthy_market()
    market.current_liquidity = market.entry_liquidity * 0.69  # below 70% floor
    decision = evaluate(make_position(), market)
    assert decision.should_exit is True
    assert decision.trigger == "liquidity_drop"
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)


def test_coordinated_dump_single_large_sell_fires() -> None:
    market = healthy_market()
    market.largest_single_sell = market.current_liquidity * 0.06  # > 5% of pool
    decision = evaluate(make_position(), market)
    assert decision.should_exit is True
    assert decision.trigger == "coordinated_dump"
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)


def test_coordinated_dump_multiple_wallets_fires() -> None:
    market = healthy_market()
    market.top_wallet_sells = 3  # several top wallets dumping together
    decision = evaluate(make_position(), market)
    assert decision.should_exit is True
    assert decision.trigger == "coordinated_dump"
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)


def test_flash_crash_fires() -> None:
    market = healthy_market()
    market.candle_1m_open = ENTRY_PRICE * 1.5
    market.candle_1m_close = market.candle_1m_open * 0.20  # 80% intra-candle drop
    decision = evaluate(make_position(), market)
    assert decision.should_exit is True
    assert decision.trigger == "flash_crash"
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)


def test_full_exit_sells_entire_remaining_including_moonbag() -> None:
    """tokens_to_sell must equal the whole remaining balance, not part of it."""
    pos = make_position()
    market = healthy_market()
    market.current_price = ENTRY_PRICE * 0.5  # deep stop-loss
    decision = evaluate(pos, market)
    assert decision.should_exit is True
    assert decision.tokens_to_sell == pytest.approx(pos.remaining)
    assert decision.tokens_to_sell == pytest.approx(MOONBAG)
