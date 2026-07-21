"""Tests for the trade-journal performance summary (reporting.journal_summary).

Proves the six required metrics are computed correctly from a set of journalled
trades: total round-trips, win/loss + win rate, cumulative and average realised
PnL, the exit-reason breakdown (hard-exit triggers vs profit tiers, one bucket
per trade with hard-exit precedence), average hold time, and best/worst trade.
Also covers the empty-journal edge case.
"""

from __future__ import annotations

from reporting.journal_summary import build_summary, format_summary, summarize
from reporting.trade_journal import TradeRecord, append_record


def _record(
    symbol: str,
    pnl: float,
    *,
    exit_trigger=None,
    profit_tiers=None,
    hold_ticks: int = 3,
    hold_seconds: float = 180.0,
) -> TradeRecord:
    return TradeRecord(
        schema=1,
        recorded_at="2026-07-18T00:00:00+00:00",
        dry_run=True,
        mint_address=f"Mint{symbol}",
        symbol=symbol,
        strategy="momentum",
        entry_reason="",
        entry_price=0.001,
        entry_size=1000.0,
        entry_liquidity=100_000.0,
        realised_pnl_usd=pnl,
        fill_count=len(profit_tiers or []) or (1 if exit_trigger else 0),
        fully_exited=bool(exit_trigger),
        tokens_held=0.0 if exit_trigger else 100.0,
        exit_trigger=exit_trigger,
        exit_reason="",
        profit_tiers=list(profit_tiers or []),
        entry_time=None,
        exit_time=None,
        hold_ticks=hold_ticks,
        hold_seconds=hold_seconds,
        legs=[],
    )


def _sample_records():
    """A mixed journal: 2 winners, 2 losers, 1 break-even held-open."""
    return [
        _record("WIN1", 12.0, profit_tiers=["2x", "5x"], hold_ticks=5, hold_seconds=300.0),
        _record("WIN2", 3.0, profit_tiers=["2x"], hold_ticks=4, hold_seconds=240.0),
        _record("RUG1", -8.0, exit_trigger="flash_crash", hold_ticks=2, hold_seconds=120.0),
        _record("RUG2", -2.0, exit_trigger="liquidity_drop", hold_ticks=1, hold_seconds=60.0),
        _record("FLAT", 0.0, hold_ticks=5, hold_seconds=300.0),  # held open, break-even
    ]


def test_total_and_win_loss_breakdown() -> None:
    s = summarize(_sample_records())
    assert s.total_trades == 5
    assert s.wins == 2
    assert s.losses == 2
    assert s.breakeven == 1
    # Win rate is wins over ALL trades taken.
    assert s.win_rate_pct == 40.0


def test_cumulative_and_average_pnl() -> None:
    s = summarize(_sample_records())
    # 12 + 3 - 8 - 2 + 0 = 5.0 total; /5 trades = 1.0 avg.
    assert s.cumulative_pnl_usd == 5.0
    assert s.avg_pnl_usd == 1.0


def test_exit_reason_breakdown_one_bucket_per_trade() -> None:
    s = summarize(_sample_records())
    assert s.hard_exit_counts["flash_crash"] == 1
    assert s.hard_exit_counts["liquidity_drop"] == 1
    assert s.hard_exit_counts["hard_stop_loss"] == 0
    assert s.hard_exit_counts["coordinated_dump"] == 0
    assert s.hard_exit_counts["volume_collapse"] == 0
    # Two profit-taking trades (moonbag held), one held-open break-even.
    assert s.profit_take_trades == 2
    assert s.held_open_trades == 1
    # Buckets partition the trades exactly.
    hard = sum(s.hard_exit_counts.values())
    assert hard + s.profit_take_trades + s.held_open_trades == s.total_trades


def test_tier_firings_count_every_tier_hit() -> None:
    s = summarize(_sample_records())
    # WIN1 fired 2x and 5x; WIN2 fired 2x -> 2x twice, 5x once, 10x never.
    assert s.tier_firings["2x"] == 2
    assert s.tier_firings["5x"] == 1
    assert s.tier_firings["10x"] == 0


def test_hard_exit_takes_precedence_over_any_earlier_tier() -> None:
    # A trade that took profit at 2x then still hard-exited is a hard-exit trade,
    # not a profit-take trade — but the tier firing is still counted.
    rec = _record("MIX", -1.0, exit_trigger="hard_stop_loss", profit_tiers=["2x"])
    s = summarize([rec])
    assert s.hard_exit_counts["hard_stop_loss"] == 1
    assert s.profit_take_trades == 0
    assert s.tier_firings["2x"] == 1


def test_average_hold_time() -> None:
    s = summarize(_sample_records())
    # ticks: 5+4+2+1+5 = 17 / 5 = 3.4
    assert s.avg_hold_ticks == 3.4
    # seconds: 300+240+120+60+300 = 1020 / 5 = 204.0
    assert s.avg_hold_seconds == 204.0


def test_best_and_worst_trade() -> None:
    s = summarize(_sample_records())
    assert s.best is not None and s.best.symbol == "WIN1" and s.best.pnl_usd == 12.0
    assert s.worst is not None and s.worst.symbol == "RUG1" and s.worst.pnl_usd == -8.0


def test_empty_journal_summary_is_zeroed() -> None:
    s = summarize([])
    assert s.total_trades == 0
    assert s.best is None and s.worst is None
    assert s.win_rate_pct == 0.0
    # Every trigger/tier key is present (with 0) so the table is always complete.
    assert set(s.hard_exit_counts) == {
        "hard_stop_loss", "liquidity_drop", "coordinated_dump",
        "flash_crash", "volume_collapse",
    }
    text = format_summary(s)
    assert "no completed trades" in text


def test_build_summary_reads_from_disk(tmp_path) -> None:
    path = str(tmp_path / "trades.jsonl")
    for rec in _sample_records():
        append_record(rec, path)
    s, count = build_summary(path)
    assert count == 5
    assert s.total_trades == 5
    assert s.cumulative_pnl_usd == 5.0


def test_format_summary_renders_all_sections() -> None:
    text = format_summary(summarize(_sample_records()), source="trades.jsonl")
    for needle in (
        "Total trades", "Win rate", "Cumulative realised PnL",
        "Exit reasons", "flash_crash", "Profit tiers fired",
        "Avg hold per trade", "Best trade", "Worst trade",
    ):
        assert needle in text
