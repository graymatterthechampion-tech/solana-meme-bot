"""Tests for the clean per-trade reporting layer (reporting.trade_report).

Proves the report answers the two questions the cumulative INFO stream buries:
*why did the position exit* and *what was the per-trade PnL*. Everything is
built from an already-completed ``TradeSession`` — no network, no trading.
"""

from __future__ import annotations

import main
from execution.fill_simulator import simulate_fill
from main import LoopOutcome, TradeSession, realised_pnl_usd
from reporting.trade_report import (
    build_trade_report,
    format_trade_report,
)
from strategies.entry import EntryAction, EntryDecision
from strategies.entry_momentum import MomentumDecision
from strategies.hard_exit import ExitDecision
from strategies.profit_taking import Position, SellAction

ENTRY_PRICE = 0.001
ENTRY_LIQUIDITY = 100_000.0


def _dip_decision() -> EntryDecision:
    return EntryDecision(
        action=EntryAction.BUY,
        mint_address="MINT",
        reason="stabilised dip on a proven runner",
        entry_liquidity=ENTRY_LIQUIDITY,
        entry_price=ENTRY_PRICE,
    )


def _profit_outcome() -> LoopOutcome:
    """A profit-taking tick: the 2x tier sells 50% of the original size."""
    fill = simulate_fill(500.0, ENTRY_PRICE * 2, ENTRY_LIQUIDITY)
    action = SellAction(
        tier="2x",
        price_multiple=2.0,
        fraction_of_original=0.50,
        tokens_to_sell=500.0,
        tokens_remaining_after=500.0,
    )
    return LoopOutcome(path="profit_taking", sell_actions=[action], fills=[fill])


def _hard_exit_outcome(tokens: float) -> LoopOutcome:
    """A hard-exit tick: the full remaining position is dumped at a loss."""
    fill = simulate_fill(tokens, ENTRY_PRICE * 0.5, ENTRY_LIQUIDITY)
    decision = ExitDecision(
        should_exit=True,
        trigger="hard_stop_loss",
        reason="price 0.00050000 is 50.0% below entry 0.00100000",
        tokens_to_sell=tokens,
    )
    return LoopOutcome(path="hard_exit", exit_decision=decision, fills=[fill])


def _entered_session(outcomes, decision=None, remaining=1000.0) -> TradeSession:
    position = Position(
        entry_price=ENTRY_PRICE, original_size=1000.0, remaining=remaining
    )
    return TradeSession(
        mint_address="MINT",
        status="entered",
        entry_decision=decision or _dip_decision(),
        position=position,
        loop_outcomes=outcomes,
        symbol="PEPE",
    )


# --- No trade to report ------------------------------------------------------

def test_non_entered_session_has_no_report() -> None:
    """Rejected / no-entry / no-market-data sessions opened no position."""
    for status in ("rejected_safety", "no_entry", "no_market_data"):
        session = TradeSession(mint_address="M", status=status, symbol="X")
        assert build_trade_report(session) is None


# --- Hard exit: why it exited + full-exit PnL --------------------------------

def test_hard_exit_report_explains_exit_and_totals_loss() -> None:
    session = _entered_session([_hard_exit_outcome(1000.0)])
    report = build_trade_report(session)
    assert report is not None

    # The exit is attributed to the firing trigger and its reason is carried.
    assert report.exit_trigger == "hard_stop_loss"
    assert "HARD EXIT [hard_stop_loss]" in report.exit_reason
    assert "below entry" in report.exit_reason
    assert report.fully_exited is True
    assert report.tokens_held == 0.0

    # One full-exit leg, labelled, at a realised loss (sold at 0.5x entry).
    assert len(report.legs) == 1
    leg = report.legs[0]
    assert leg.label == "HARD_EXIT/hard_stop_loss"
    assert leg.kind == "hard_exit"
    assert leg.tokens == 1000.0
    assert leg.pnl_usd < 0.0
    assert report.realised_pnl_usd == leg.pnl_usd
    assert report.fill_count == 1


# --- Profit taking: per-sell PnL, moonbag retained ---------------------------

def test_profit_taking_report_captures_tier_and_moonbag() -> None:
    session = _entered_session([_profit_outcome()], remaining=500.0)
    report = build_trade_report(session)
    assert report is not None

    assert len(report.legs) == 1
    leg = report.legs[0]
    assert leg.label == "PROFIT/2x"
    assert leg.kind == "profit"
    assert leg.pnl_usd > 0.0  # a 2x sell is profitable net of friction

    # No hard exit: the loop ended with the moonbag/remaining still held.
    assert report.exit_trigger is None
    assert report.fully_exited is False
    assert "profit tiers 2x sold" in report.exit_reason
    assert report.tokens_held == 500.0


# --- Hold-only: nothing sold, explained rather than looking abandoned --------

def test_hold_only_report_has_no_sells_and_explains_hold() -> None:
    session = _entered_session([LoopOutcome(path="hold"), LoopOutcome(path="hold")])
    report = build_trade_report(session)
    assert report is not None

    assert report.legs == []
    assert report.fill_count == 0
    assert report.realised_pnl_usd == 0.0
    assert report.fully_exited is False
    assert "no exit trigger fired over 2 tick(s)" in report.exit_reason
    assert report.tokens_held == 1000.0


# --- Mixed lifecycle: profits then a hard exit -------------------------------

def test_mixed_lifecycle_pnl_matches_main_realised_pnl() -> None:
    """The report's realised PnL equals main.realised_pnl_usd over all fills."""
    outcomes = [
        LoopOutcome(path="hold"),
        _profit_outcome(),
        _hard_exit_outcome(500.0),
    ]
    session = _entered_session(outcomes, remaining=0.0)
    report = build_trade_report(session)
    assert report is not None

    # A hard exit anywhere in the lifecycle is THE terminal reason.
    assert report.exit_trigger == "hard_stop_loss"
    assert report.fully_exited is True
    # Two sell legs: the profit tier and the full hard exit.
    assert [leg.label for leg in report.legs] == [
        "PROFIT/2x",
        "HARD_EXIT/hard_stop_loss",
    ]
    assert report.realised_pnl_usd == realised_pnl_usd(outcomes, ENTRY_PRICE)


# --- Strategy attribution ----------------------------------------------------

def test_strategy_name_distinguishes_dip_and_momentum() -> None:
    dip = build_trade_report(_entered_session([LoopOutcome(path="hold")]))
    assert dip is not None and dip.strategy == "dip"

    momentum_decision = MomentumDecision(
        action=EntryAction.BUY,
        mint_address="MINT",
        reason="breakout with sustained volume",
        entry_liquidity=ENTRY_LIQUIDITY,
        entry_price=ENTRY_PRICE,
    )
    mom = build_trade_report(
        _entered_session([LoopOutcome(path="hold")], decision=momentum_decision)
    )
    assert mom is not None and mom.strategy == "momentum"
    assert mom.entry_reason == "breakout with sustained volume"


# --- Rendering ---------------------------------------------------------------

def test_format_is_ascii_and_shows_key_facts() -> None:
    session = _entered_session([_profit_outcome(), _hard_exit_outcome(500.0)])
    report = build_trade_report(session)
    assert report is not None

    text = format_trade_report(report)
    assert text.isascii()  # Windows-console safe, no em dashes / box glyphs
    assert "TRADE PEPE [dip]" in text
    assert "PROFIT/2x" in text
    assert "HARD_EXIT/hard_stop_loss" in text
    assert "realised PnL" in text
    # Signed PnL rendering: the hard-exit loss shows a minus sign.
    assert "-$" in text


def test_log_trade_report_emits_only_for_entered(caplog) -> None:
    """log_trade_report logs one INFO block for an entered session, else nothing."""
    import logging

    with caplog.at_level(logging.INFO, logger="reporting.trade_report"):
        assert main is not None  # module import sanity
        from reporting.trade_report import log_trade_report

        entered = _entered_session([_hard_exit_outcome(1000.0)])
        report = log_trade_report(entered)
        assert report is not None
        assert any("TRADE PEPE" in rec.getMessage() for rec in caplog.records)

        caplog.clear()
        skipped = TradeSession(mint_address="M", status="no_entry", symbol="X")
        assert log_trade_report(skipped) is None
        assert not caplog.records
