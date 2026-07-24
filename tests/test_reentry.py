"""Tests for the re-entry guard (safety.reentry) and its scan wiring.

Covers: cooldown blocks then clears; blacklist arms after N hard-exits, blocks
within its window, clears after it, and can be permanent; blacklist outranks
cooldown; repeated exits only push the expiry out; seeding from journal records
reconstructs state (incl. the ACM churn) with historical, not reset, clocks; and
that scan_and_evaluate skips a blocked mint before touching safety/entry.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import main
from reporting.trade_journal import TradeRecord
from safety.reentry import ReentryGuard

T0 = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)
MINT = "4PRz3EwhbjrrX6YksMDuUzrXT51pr7CQtXNCravhpump"


def _guard(**kw) -> ReentryGuard:
    base = dict(cooldown_seconds=3600.0, blacklist_hard_exits=1, blacklist_seconds=86_400.0)
    base.update(kw)
    return ReentryGuard(**base)


# --- cooldown ---------------------------------------------------------------

def test_unknown_mint_is_allowed() -> None:
    assert _guard().check(MINT, T0) is None


def test_cooldown_blocks_then_clears() -> None:
    g = _guard(cooldown_seconds=3600.0, blacklist_hard_exits=99)  # avoid blacklist
    g.record_entry(MINT, T0)
    assert g.check(MINT, T0 + timedelta(minutes=15)) is not None    # within 1h
    assert "cooldown" in g.check(MINT, T0 + timedelta(minutes=15))
    assert g.check(MINT, T0 + timedelta(minutes=61)) is None        # elapsed


def test_cooldown_zero_disables() -> None:
    g = _guard(cooldown_seconds=0.0, blacklist_hard_exits=99)
    g.record_entry(MINT, T0)
    assert g.check(MINT, T0 + timedelta(seconds=1)) is None


# --- blacklist --------------------------------------------------------------

def test_blacklist_after_one_hard_exit() -> None:
    g = _guard(blacklist_hard_exits=1, blacklist_seconds=86_400.0)
    g.record_hard_exit(MINT, "volume_collapse", T0)
    assert "blacklisted" in g.check(MINT, T0 + timedelta(hours=12))   # within 24h
    assert g.check(MINT, T0 + timedelta(hours=25)) is None            # expired


def test_blacklist_threshold_two() -> None:
    g = _guard(blacklist_hard_exits=2, cooldown_seconds=0.0)
    g.record_hard_exit(MINT, "volume_collapse", T0)
    assert g.check(MINT, T0 + timedelta(minutes=1)) is None           # 1 < 2, allowed
    g.record_hard_exit(MINT, "volume_collapse", T0 + timedelta(minutes=1))
    assert "blacklisted" in g.check(MINT, T0 + timedelta(minutes=2))  # 2 >= 2


def test_permanent_blacklist_when_seconds_nonpositive() -> None:
    g = _guard(blacklist_seconds=0.0)
    g.record_hard_exit(MINT, "flash_crash", T0)
    assert "permanent" in g.check(MINT, T0 + timedelta(days=3650))


def test_blacklist_outranks_cooldown() -> None:
    g = _guard(cooldown_seconds=60.0, blacklist_hard_exits=1)
    g.record_entry(MINT, T0)
    g.record_hard_exit(MINT, "flash_crash", T0)
    # Well past cooldown but inside blacklist -> blacklist reason wins.
    reason = g.check(MINT, T0 + timedelta(hours=5))
    assert reason is not None and "blacklisted" in reason


def test_repeated_exits_only_extend_expiry() -> None:
    g = _guard(blacklist_hard_exits=1, blacklist_seconds=3600.0)
    g.record_hard_exit(MINT, "volume_collapse", T0)
    g.record_hard_exit(MINT, "volume_collapse", T0 + timedelta(minutes=30))
    # Second exit at +30m pushes expiry to +90m; still blocked at +80m.
    assert g.check(MINT, T0 + timedelta(minutes=80)) is not None
    assert g.check(MINT, T0 + timedelta(minutes=91)) is None


# --- seeding from journal ---------------------------------------------------

def _rec(mint, entry_time, *, exit_trigger=None, exit_time=None) -> TradeRecord:
    return TradeRecord(
        schema=1, recorded_at=exit_time or entry_time, dry_run=True, mint_address=mint,
        symbol="ACM", strategy="dip", entry_reason="", entry_price=0.0004,
        entry_size=1000.0, entry_liquidity=50_000.0,
        realised_pnl_usd=-0.74 if exit_trigger else 0.0,
        fill_count=1 if exit_trigger else 0, fully_exited=bool(exit_trigger),
        tokens_held=0.0, exit_trigger=exit_trigger, exit_reason="",
        profit_tiers=[], entry_time=entry_time, exit_time=exit_time or entry_time,
        hold_ticks=1, hold_seconds=5.0, legs=[],
    )


def test_from_records_reconstructs_acm_churn() -> None:
    # ACM hard-exited repeatedly; a guard seeded from that history blacklists it.
    iso = lambda mins: (T0 + timedelta(minutes=mins)).isoformat()
    records = [
        _rec(MINT, iso(m), exit_trigger="volume_collapse", exit_time=iso(m))
        for m in (0, 15, 30, 45)
    ]
    now = T0 + timedelta(hours=1)
    g = ReentryGuard.from_records(records, now=now, blacklist_hard_exits=1, blacklist_seconds=86_400.0)
    assert g.is_blocked(MINT, now)                       # blacklisted from history
    assert MINT in g.blacklisted_mints(now)


def test_from_records_old_history_does_not_block() -> None:
    # A single entry 2h ago, no hard exit: cooldown (1h) already elapsed.
    g = ReentryGuard.from_records(
        [_rec(MINT, (T0 - timedelta(hours=2)).isoformat())],
        now=T0, cooldown_seconds=3600.0, blacklist_hard_exits=1,
    )
    assert g.check(MINT, T0) is None


def test_from_records_skips_mintless_and_bad_times() -> None:
    g = ReentryGuard.from_records(
        [_rec("", T0.isoformat()), _rec(MINT, "not-a-date")], now=T0
    )
    assert g.check(MINT, T0) is None  # unparseable entry time -> no cooldown armed


# --- scan wiring ------------------------------------------------------------

def test_scan_skips_blocked_mint_before_safety() -> None:
    """A blocked mint yields reentry_blocked and never reaches entry/market."""
    g = _guard()
    g.record_hard_exit(MINT, "flash_crash", datetime.now(timezone.utc))  # blacklist it

    reached: List[str] = []

    async def scanner() -> List[Dict[str, Any]]:
        return [{"mint": MINT, "symbol": "ACM"}, {"mint": "GoodMint" + "z" * 30, "symbol": "OK"}]

    async def market_provider(candidate: Dict[str, Any]):
        reached.append(candidate["mint"])
        return None  # both fail-closed after the guard; we only assert reach set

    sessions = asyncio.run(
        main.scan_and_evaluate(
            scanner=scanner,
            entry_market_provider=market_provider,
            reentry_guard=g,
        )
    )
    by = {s.mint_address: s.status for s in sessions}
    assert by[MINT] == "reentry_blocked"
    assert MINT not in reached                    # guard ran BEFORE market fetch
    assert "GoodMint" + "z" * 30 in reached       # non-blocked mint proceeded
