"""Tests for the tiered profit-taking strategy.

The strategy function is async; rather than depend on the pytest-asyncio
plugin, each test drives the coroutine synchronously via ``asyncio.run`` in
the ``evaluate`` helper below.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from strategies.profit_taking import (
    MOONBAG_FRACTION,
    Position,
    SellAction,
    evaluate_take_profit,
)

ENTRY_PRICE = 0.001
ORIGINAL_SIZE = 1000.0


def evaluate(position: Position, current_price: float) -> List[SellAction]:
    """Run the async evaluator to completion and return its actions."""
    return asyncio.run(evaluate_take_profit(position, current_price))


def tiers_fired(actions: List[SellAction]) -> List[str]:
    return [a.tier for a in actions]


def test_gradual_climb_fires_each_tier_in_order() -> None:
    """2x, 5x, 10x crossed on separate readings fire one tier at a time."""
    pos = Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)

    # Reading 1: 2x -> sell 50% of original (500), 500 remaining.
    actions = evaluate(pos, ENTRY_PRICE * 2)
    assert tiers_fired(actions) == ["2x"]
    assert actions[0].tokens_to_sell == pytest.approx(500.0)
    assert pos.remaining == pytest.approx(500.0)

    # Reading 2: 5x -> sell 25% of original (250), 250 remaining.
    actions = evaluate(pos, ENTRY_PRICE * 5)
    assert tiers_fired(actions) == ["5x"]
    assert actions[0].tokens_to_sell == pytest.approx(250.0)
    assert pos.remaining == pytest.approx(250.0)

    # Reading 3: 10x -> sell 15% of original (150), 100 (moonbag) remaining.
    actions = evaluate(pos, ENTRY_PRICE * 10)
    assert tiers_fired(actions) == ["10x"]
    assert actions[0].tokens_to_sell == pytest.approx(150.0)
    assert pos.remaining == pytest.approx(100.0)

    assert (pos.sold_2x, pos.sold_5x, pos.sold_10x) == (True, True, True)


def test_gap_fires_all_eligible_tiers_in_one_call() -> None:
    """A jump straight from 1x to 10x fires 2x + 5x + 10x in a single call."""
    pos = Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)

    actions = evaluate(pos, ENTRY_PRICE * 10)

    assert tiers_fired(actions) == ["2x", "5x", "10x"]
    # 50% + 25% + 15% = 90% of original sold in this one call.
    total_sold = sum(a.tokens_to_sell for a in actions)
    assert total_sold == pytest.approx(900.0)
    assert pos.remaining == pytest.approx(100.0)
    assert (pos.sold_2x, pos.sold_5x, pos.sold_10x) == (True, True, True)


def test_once_only_guard_blocks_refire_on_later_readings() -> None:
    """Once a tier has fired, staying above its threshold never re-sells."""
    pos = Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)

    first = evaluate(pos, ENTRY_PRICE * 10)
    assert tiers_fired(first) == ["2x", "5x", "10x"]
    remaining_after_first = pos.remaining

    # Subsequent readings still well above 10x must fire nothing.
    for multiple in (10, 12, 25, 100):
        again = evaluate(pos, ENTRY_PRICE * multiple)
        assert again == []
        assert pos.remaining == pytest.approx(remaining_after_first)


def test_partial_then_remaining_tiers_do_not_refire_lower_tier() -> None:
    """After 2x fires gradually, a later gap to 10x fires only 5x and 10x."""
    pos = Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)

    assert tiers_fired(evaluate(pos, ENTRY_PRICE * 2)) == ["2x"]
    # Now gap up to 10x: only the still-unsold 5x and 10x tiers fire.
    assert tiers_fired(evaluate(pos, ENTRY_PRICE * 10)) == ["5x", "10x"]
    assert pos.remaining == pytest.approx(100.0)


def test_moonbag_invariant_never_sold() -> None:
    """The final 10% moonbag is never auto-sold, however high the price runs."""
    pos = Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)
    expected_moonbag = ORIGINAL_SIZE * MOONBAG_FRACTION  # 100 tokens

    # Run the full ladder plus extreme follow-up readings.
    evaluate(pos, ENTRY_PRICE * 10)
    for multiple in (50, 500, 5000):
        evaluate(pos, ENTRY_PRICE * multiple)

    assert pos.remaining == pytest.approx(expected_moonbag)
    assert pos.remaining == pytest.approx(pos.moonbag_tokens)
    assert pos.remaining > 0  # moonbag is retained, never zeroed


def test_below_first_threshold_fires_nothing() -> None:
    """Under 2x, no tier fires and the position is untouched."""
    pos = Position(entry_price=ENTRY_PRICE, original_size=ORIGINAL_SIZE)

    actions = evaluate(pos, ENTRY_PRICE * 1.99)

    assert actions == []
    assert pos.remaining == pytest.approx(ORIGINAL_SIZE)
    assert (pos.sold_2x, pos.sold_5x, pos.sold_10x) == (False, False, False)
