"""Tests for scenario mode: each scripted market path drives the loop to its
expected decision path (pump -> profit_taking tiers; rug/dump -> hard_exit).
"""

from __future__ import annotations

import asyncio

import main
from data.market_feed import make_scenario_provider
from strategies.profit_taking import Position

ENTRY_PRICE = 0.001
ENTRY_LIQUIDITY = 100_000.0


def make_position() -> Position:
    return Position(entry_price=ENTRY_PRICE, original_size=1000.0)


def run_scenario(name: str, max_iterations: int):
    provider = make_scenario_provider(
        name, entry_price=ENTRY_PRICE, entry_liquidity=ENTRY_LIQUIDITY
    )
    return asyncio.run(
        main.run_loop(
            make_position(),
            token_address="MINT",
            entry_liquidity=ENTRY_LIQUIDITY,
            max_iterations=max_iterations,
            snapshot_provider=provider,
        )
    )


def test_flat_scenario_holds() -> None:
    outcomes = run_scenario("flat", max_iterations=3)
    assert [o.path for o in outcomes] == ["hold", "hold", "hold"]


def test_pump_scenario_fires_all_profit_tiers() -> None:
    outcomes = run_scenario("pump", max_iterations=4)

    # First three iterations fire the 2x, 5x, 10x tiers in order.
    assert outcomes[0].path == "profit_taking"
    assert outcomes[1].path == "profit_taking"
    assert outcomes[2].path == "profit_taking"
    assert [a.tier for a in outcomes[0].sell_actions] == ["2x"]
    assert [a.tier for a in outcomes[1].sell_actions] == ["5x"]
    assert [a.tier for a in outcomes[2].sell_actions] == ["10x"]
    # Fourth iteration: price holds at 10x, all tiers already sold -> HOLD.
    assert outcomes[3].path == "hold"


def test_rug_scenario_fires_liquidity_hard_exit() -> None:
    outcomes = run_scenario("rug", max_iterations=5)

    last = outcomes[-1]
    assert last.path == "hard_exit"
    assert last.exit_decision is not None
    assert last.exit_decision.trigger == "liquidity_drop"
    # Loop stopped early on the exit (sequence is 2 steps, not 5).
    assert len(outcomes) == 2


def test_dump_scenario_fires_flash_crash_hard_exit() -> None:
    outcomes = run_scenario("dump", max_iterations=5)

    last = outcomes[-1]
    assert last.path == "hard_exit"
    assert last.exit_decision is not None
    assert last.exit_decision.trigger == "flash_crash"
    assert len(outcomes) == 2
