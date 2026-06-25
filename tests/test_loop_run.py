"""Tests for the bounded async trading loop (main.run_loop).

Drives the loop against injected snapshot providers: the mock source, a
fail-closed (None) source, and a source that triggers a hard exit.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from data.market_feed import mock_market_snapshot
import main
from strategies.hard_exit import MarketData
from strategies.profit_taking import Position

ENTRY_PRICE = 0.001
ENTRY_LIQUIDITY = 100_000.0


def make_position() -> Position:
    return Position(entry_price=ENTRY_PRICE, original_size=1000.0)


def run(coro):
    return asyncio.run(coro)


def test_loop_runs_bounded_iterations_against_mock() -> None:
    """A few iterations against the mock complete without error."""
    pos = make_position()
    outcomes = run(
        main.run_loop(
            pos,
            token_address="MINT",
            entry_liquidity=ENTRY_LIQUIDITY,
            max_iterations=3,
            snapshot_provider=mock_market_snapshot,
        )
    )

    assert len(outcomes) == 3
    # Mock price is 1.5x entry: no hard exit, no profit tier -> all "hold".
    assert all(o is not None for o in outcomes)
    assert [o.path for o in outcomes] == ["hold", "hold", "hold"]


def test_loop_skips_on_fail_closed_none() -> None:
    """When the provider returns None, the iteration is skipped, not traded."""

    async def none_provider(token_address: str, entry_liquidity: float) -> Optional[MarketData]:
        return None

    pos = make_position()
    outcomes = run(
        main.run_loop(
            pos,
            token_address="MINT",
            entry_liquidity=ENTRY_LIQUIDITY,
            max_iterations=4,
            snapshot_provider=none_provider,
        )
    )

    assert len(outcomes) == 4
    assert outcomes == [None, None, None, None]


def test_loop_stops_early_on_hard_exit() -> None:
    """A hard-exit trigger closes the position and halts the loop early."""

    async def crash_provider(token_address: str, entry_liquidity: float) -> Optional[MarketData]:
        # Price 50% below entry -> hard_stop_loss fires.
        return MarketData(
            current_price=ENTRY_PRICE * 0.5,
            rolling_volume_15m=50_000.0,
            peak_volume_15m=50_000.0,
            current_liquidity=100_000.0,
            entry_liquidity=entry_liquidity,
            candle_1m_open=ENTRY_PRICE * 0.5,
            candle_1m_close=ENTRY_PRICE * 0.5,
        )

    pos = make_position()
    outcomes = run(
        main.run_loop(
            pos,
            token_address="MINT",
            entry_liquidity=ENTRY_LIQUIDITY,
            max_iterations=10,
            snapshot_provider=crash_provider,
        )
    )

    # Stops on the first iteration despite max_iterations=10.
    assert len(outcomes) == 1
    assert outcomes[0] is not None
    assert outcomes[0].path == "hard_exit"
