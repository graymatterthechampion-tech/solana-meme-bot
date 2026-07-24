"""Tests for the self-contained HTML report generator (reporting.build_report).

Covers: real-vs-demo filtering, aggregation (cum PnL, win/loss, exit buckets,
strategy split), the PRELIMINARY watermark on a small sample, self-containment
(no external references), and that generate() writes a valid HTML file.
"""

from __future__ import annotations

import os

from reporting.build_report import (
    RELIABLE_SAMPLE,
    build_data,
    generate,
    is_real,
    render_html,
    svg_bars,
)
from reporting.trade_journal import TradeRecord, append_record


def _rec(symbol, mint, pnl, *, strategy="dip", exit_trigger=None,
         profit_tiers=None, fully_exited=None, entry_time="2026-07-21T12:00:00+00:00",
         hold_ticks=1, hold_seconds=5.0) -> TradeRecord:
    if fully_exited is None:
        fully_exited = exit_trigger is not None
    return TradeRecord(
        schema=1, recorded_at=entry_time, dry_run=True, mint_address=mint,
        symbol=symbol, strategy=strategy, entry_reason="", entry_price=0.001,
        entry_size=1000.0, entry_liquidity=50_000.0, realised_pnl_usd=pnl,
        fill_count=1 if exit_trigger else 0, fully_exited=fully_exited,
        tokens_held=0.0, exit_trigger=exit_trigger, exit_reason="reason text",
        profit_tiers=list(profit_tiers or []), entry_time=entry_time,
        exit_time=entry_time, hold_ticks=hold_ticks, hold_seconds=hold_seconds, legs=[],
    )


REAL_MINT = "4tnZ57E3oWjaZKSQY1iwgqiD8ivAGk5TambivhTYpump"


def test_is_real_filters_demo() -> None:
    assert is_real(_rec("GMEBULL", REAL_MINT, 0.0)) is True
    assert is_real(_rec("MOON", "Mint_MOON", 1.0)) is False       # synthetic prefix
    assert is_real(_rec("DUMP", "shortaddr", -1.0)) is False      # demo symbol
    assert is_real(_rec("X", "tooshort", 0.0)) is False           # not a real mint


def _sample():
    return [
        _rec("GMEBULL", REAL_MINT, 0.0, strategy="dip"),                      # held open
        _rec("BABY", "GEPfztBZqdh8nZHaPsRxd4UzzwAc8wJRQvC9Rkuxpump", -73.47,
             strategy="dip", exit_trigger="flash_crash"),
        _rec("ACM", "4PRz3EwhbjrrX6YksMDuUzrXT51pr7CQtXNCravhpump", -0.74,
             strategy="dip", exit_trigger="volume_collapse"),
        _rec("HOP", "DUYw2p3NC6zDdsSrazV4JdDFKtRk2K4mw764EWs2pump", 0.0,
             strategy="momentum"),                                            # held open
        _rec("WIN", "89gZQFtEe3RJctXghdbEmht8SV2vQvcN4DNyjmappump", 5.0,
             strategy="momentum", profit_tiers=["2x"], fully_exited=True),
    ]


def test_build_data_aggregates() -> None:
    d = build_data(_sample())
    assert d.total == 5
    assert d.completed == 3          # 2 hard exits + 1 profit-take (fully_exited)
    assert d.open_held == 2
    assert d.wins == 1 and d.losses == 2 and d.breakeven == 2
    assert round(d.cum_pnl, 2) == round(-73.47 - 0.74 + 5.0, 2)
    assert d.equity[-1] == d.cum_pnl
    assert d.exit_counts["flash_crash"] == 1
    assert d.exit_counts["volume_collapse"] == 1
    assert d.exit_counts["held_open"] == 2       # GMEBULL, HOP
    assert d.exit_counts["profit_take"] == 1     # WIN (tier fired, no hard exit)
    assert d.tier_firings["2x"] == 1
    assert d.worst.symbol == "BABY" and d.best.symbol == "WIN"
    # strategy split
    assert d.by_strategy["dip"].pnl == round(-73.47 - 0.74, 2) or abs(
        d.by_strategy["dip"].pnl - (-74.21)) < 1e-6
    assert d.by_strategy["momentum"].completed == 1
    assert d.by_strategy["momentum"].win_rate == 100.0
    assert d.by_strategy["dip"].win_rate == 0.0


def test_render_is_self_contained_and_watermarked() -> None:
    html = render_html(build_data(_sample()))
    assert html.startswith("<!DOCTYPE html>")
    assert "PRELIMINARY" in html           # small sample -> watermarked
    assert "watermark" in html
    # No external resources at all (offline-openable).
    for token in ("http://", "https://", "<link", "src=", "cdn"):
        assert token not in html
    # Contains the charts and the table rows.
    assert html.count("<svg") == 3
    assert "BABY" in html and "ACM" in html


def test_no_watermark_when_sample_is_large() -> None:
    big = [
        _rec(f"T{i}", REAL_MINT[:-4] + f"{i:04d}"[-4:] + "pump", 1.0,
             exit_trigger="volume_collapse")
        for i in range(RELIABLE_SAMPLE + 1)
    ]
    # pad mints to be >=32 chars
    for r in big:
        assert is_real(r)
    html = render_html(build_data(big))
    # The .watermark CSS class is always defined; only the DIV + banner are gated.
    assert '<div class="watermark">' not in html
    assert "PRELIMINARY" not in html


def test_svg_bars_renders() -> None:
    svg = svg_bars([("a", 3, "--loss"), ("b", 0, "--good")])
    assert svg.startswith("<svg") and "</svg>" in svg


def test_generate_writes_file(tmp_path) -> None:
    journal = os.path.join(str(tmp_path), "trades.jsonl")
    for r in _sample():
        append_record(r, journal)
    # add a demo row that must be excluded
    append_record(_rec("MOON", "Mint_MOON", 9.0), journal)

    out = os.path.join(str(tmp_path), "report.html")
    data = generate(journal, out)
    assert os.path.exists(out)
    assert data.total == 5           # demo row dropped
    assert data.demo_dropped == 1
    assert open(out, encoding="utf-8").read().startswith("<!DOCTYPE html>")
