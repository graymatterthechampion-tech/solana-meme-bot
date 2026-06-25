"""Tests for the full candidate pipeline in main.evaluate_and_trade.

The chain: safety gate FIRST -> entry decision -> (only on BUY) open a position
and hand it to the managed loop. Proves two end-to-end guarantees:

    * a safety-fail token never reaches entry and opens no position;
    * a safety-pass + entry-BUY token opens a position and enters the loop.

No network: the safety checker and entry market are injected/constructed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import main
from data.market_feed import mock_market_snapshot
from safety.rug_check import SafetyReport, _fail_closed_report
from strategies.entry import EntryAction, EntryMarketData

NOW = time.time()
HOUR = 3600.0
MINT = "CHAINMINT"
ENTRY_PRICE = 0.001
ENTRY_LIQUIDITY = 100_000.0


def run(coro):
    return asyncio.run(coro)


async def safe_checker(mint: str) -> SafetyReport:
    """A passing safety report (token is clean)."""
    return SafetyReport(
        mint_address=mint,
        lp_locked_or_burned=True,
        lp_lock_detail="LP burned",
        tax_pct=0.0,
        top10_holder_pct=5.0,
        holder_concentration_pass=True,
        funding_source_clustered=False,
        farmed_volume_flag=False,
        volume_to_mcap_ratio=0.1,
        passed=True,
        reasons=[],
    )


async def unsafe_checker(mint: str) -> SafetyReport:
    """A failing safety report (fail-closed / unsafe token)."""
    return _fail_closed_report(mint, "LP not locked; tax too high")


def buy_market() -> EntryMarketData:
    """A BUY-eligible candidate: proven runner, stabilised ~40% dip."""
    return EntryMarketData(
        current_price=ENTRY_PRICE * 0.60,
        ath_price=ENTRY_PRICE,
        ath_timestamp=NOW - 5 * HOUR,
        price_history=[
            ENTRY_PRICE * m for m in (1.0, 0.9, 0.75, 0.62, 0.61, 0.60, 0.605, 0.60)
        ],
        volume_history=[100.0, 100.0, 100.0, 100.0, 1000.0],
        current_liquidity=ENTRY_LIQUIDITY,
        pre_dip_liquidity=ENTRY_LIQUIDITY * 1.1,
        now=NOW,
    )


def test_safety_fail_never_reaches_entry(monkeypatch) -> None:
    """A safety-fail token is rejected before entry; no position opened."""
    # Spy: evaluate_entry must NOT be called when safety fails.
    called = {"entry": False}

    async def spy_entry(*args, **kwargs):  # pragma: no cover - must not run
        called["entry"] = True
        raise AssertionError("evaluate_entry called despite safety failure")

    monkeypatch.setattr(main, "evaluate_entry", spy_entry)

    session = run(
        main.evaluate_and_trade(
            MINT,
            buy_market(),
            max_iterations=3,
            safety_checker=unsafe_checker,
        )
    )

    assert called["entry"] is False
    assert session.status == "rejected_safety"
    assert session.position is None
    assert session.entry_decision is None
    assert session.loop_outcomes == []
    assert session.safety_report is not None and not session.safety_report.passed


def test_safety_pass_buy_opens_position_and_enters_loop() -> None:
    """Safety passes + entry BUYs -> a position is opened and the loop runs."""
    session = run(
        main.evaluate_and_trade(
            MINT,
            buy_market(),
            max_iterations=3,
            safety_checker=safe_checker,
            snapshot_provider=mock_market_snapshot,
        )
    )

    assert session.status == "entered"
    assert session.entry_decision is not None
    assert session.entry_decision.action is EntryAction.BUY
    # A real sized position was opened from the BUY decision.
    assert session.position is not None
    assert session.position.original_size > 0
    assert session.position.entry_price == ENTRY_PRICE * 0.60
    # The loop actually ran the bounded iterations against the mock feed
    # (the position was handed off and managed end-to-end).
    assert len(session.loop_outcomes) == 3
    assert all(o is not None for o in session.loop_outcomes)
    assert {o.path for o in session.loop_outcomes} <= {
        "hold", "profit_taking", "hard_exit"
    }


def test_safety_pass_but_entry_wait_opens_nothing() -> None:
    """Safety passes but entry is WAIT (too shallow) -> no position, no loop."""
    shallow = buy_market()
    shallow.current_price = ENTRY_PRICE * 0.90  # only 10% off ATH -> WAIT
    shallow.price_history = [
        ENTRY_PRICE * m for m in (1.0, 0.95, 0.92, 0.91, 0.90, 0.90)
    ]

    session = run(
        main.evaluate_and_trade(
            MINT, shallow, max_iterations=3, safety_checker=safe_checker
        )
    )

    assert session.status == "no_entry"
    assert session.entry_decision is not None
    assert session.entry_decision.action is EntryAction.WAIT
    assert session.position is None
    assert session.loop_outcomes == []
