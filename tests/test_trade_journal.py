"""Tests for append-only trade-journal persistence (reporting.trade_journal).

Proves one completed TradeReport round-trips to a JSON Lines file and back with
every summary-relevant field intact, that writing is a pure append (history is
preserved), that a hard-exit vs profit-taking trade is captured distinctly, and
that journalling is best-effort — a write failure never propagates into the loop.
"""

from __future__ import annotations

import os

from reporting.trade_journal import (
    JOURNAL_SCHEMA,
    append_record,
    append_trade,
    load_records,
    record_from_report,
)
from reporting.trade_report import TradeLeg, TradeReport


def _profit_report() -> TradeReport:
    """A profit-taking trade: 2x and 5x tiers fired, moonbag held."""
    legs = [
        TradeLeg("PROFIT/2x", "profit", 500.0, 0.002, 1.0, 0.5, False),
        TradeLeg("PROFIT/5x", "profit", 250.0, 0.005, 1.25, 1.0, False),
    ]
    return TradeReport(
        mint_address="MintProfit",
        symbol="WIN",
        strategy="momentum",
        entry_reason="breakout",
        entry_price=0.001,
        entry_size=1000.0,
        entry_liquidity=100_000.0,
        legs=legs,
        exit_trigger=None,
        exit_reason="profit tiers 2x,5x sold, moonbag held",
        fully_exited=False,
        tokens_held=100.0,
        realised_pnl_usd=1.5,
        fill_count=2,
    )


def _hard_exit_report() -> TradeReport:
    """A hard-exit trade: flash_crash fired, full exit at a loss."""
    legs = [TradeLeg("HARD_EXIT/flash_crash", "hard_exit", 1000.0, 0.0003, 0.3, -0.7, False)]
    return TradeReport(
        mint_address="MintRug",
        symbol="RUG",
        strategy="dip",
        entry_reason="post-pump dip",
        entry_price=0.001,
        entry_size=1000.0,
        entry_liquidity=80_000.0,
        legs=legs,
        exit_trigger="flash_crash",
        exit_reason="HARD EXIT [flash_crash] - 1m candle dropped 70%",
        fully_exited=True,
        tokens_held=0.0,
        realised_pnl_usd=-0.7,
        fill_count=1,
    )


def test_record_from_report_extracts_profit_tiers() -> None:
    record = record_from_report(_profit_report(), dry_run=True)
    assert record.schema == JOURNAL_SCHEMA
    assert record.profit_tiers == ["2x", "5x"]
    assert record.exit_trigger is None
    assert record.realised_pnl_usd == 1.5
    assert record.fully_exited is False
    assert len(record.legs) == 2


def test_record_from_report_hard_exit_has_trigger_no_tiers() -> None:
    record = record_from_report(_hard_exit_report())
    assert record.exit_trigger == "flash_crash"
    assert record.profit_tiers == []
    assert record.fully_exited is True
    assert record.realised_pnl_usd == -0.7


def test_hold_seconds_computed_from_timestamps() -> None:
    record = record_from_report(
        _profit_report(),
        entry_time="2026-07-18T00:00:00+00:00",
        exit_time="2026-07-18T00:05:00+00:00",
        hold_ticks=5,
    )
    assert record.hold_ticks == 5
    assert record.hold_seconds == 300.0


def test_hold_seconds_zero_when_untimed() -> None:
    record = record_from_report(_profit_report())
    assert record.hold_seconds == 0.0
    assert record.entry_time is None


def test_append_is_pure_append_and_roundtrips(tmp_path) -> None:
    path = os.path.join(str(tmp_path), "sub", "trades.jsonl")  # parent auto-created

    r1 = append_trade(_profit_report(), path, hold_ticks=5)
    r2 = append_trade(_hard_exit_report(), path, hold_ticks=2)
    assert r1 is not None and r2 is not None

    records = load_records(path)
    assert len(records) == 2  # second write appended, did not overwrite
    assert records[0].symbol == "WIN"
    assert records[1].symbol == "RUG"
    # Full round-trip fidelity on the fields the summary depends on.
    assert records[0].profit_tiers == ["2x", "5x"]
    assert records[1].exit_trigger == "flash_crash"
    assert records[1].realised_pnl_usd == -0.7


def test_load_missing_file_returns_empty(tmp_path) -> None:
    assert load_records(os.path.join(str(tmp_path), "nope.jsonl")) == []


def test_load_skips_blank_and_malformed_lines(tmp_path) -> None:
    path = os.path.join(str(tmp_path), "trades.jsonl")
    append_trade(_profit_report(), path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n")             # blank
        handle.write("not json{{{\n")  # malformed
    records = load_records(path)
    assert len(records) == 1  # the one good record survives
    assert records[0].symbol == "WIN"


def test_append_trade_never_raises_on_write_error() -> None:
    # A directory path is not a writable file: append_record raises, append_trade
    # must swallow it and return None rather than propagate into the loop.
    result = append_trade(_profit_report(), path=os.getcwd())
    assert result is None


def test_append_record_raises_on_bad_path() -> None:
    # The strict (non-best-effort) writer surfaces I/O errors to its caller.
    record = record_from_report(_profit_report())
    raised = False
    try:
        append_record(record, os.getcwd())  # a directory, not a file
    except OSError:
        raised = True
    assert raised
