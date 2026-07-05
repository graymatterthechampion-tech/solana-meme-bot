"""Integration tests for the meta ranking layer in main.scan_and_evaluate.

Proves the contract end-to-end:
  * the meta boost is computed ONLY for tokens that entered (safe + BUY) — the
    provider is never even called for rejected-safety or no-entry tokens;
  * a strong-meta token that does not qualify is never bought (no trigger);
  * entered tokens are re-ranked by their boost;
  * absent meta signals never block or drop a token (it still enters, priority
    just stays at the neutral base).

No network: scanner, safety checker, entry-market provider, snapshot provider,
and the meta_boost_provider are all injected.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import main
from data.market_feed import mock_market_snapshot
from safety.rug_check import SafetyReport, _fail_closed_report
from scanner.meta_rank import MetaBoost, NEUTRAL_BOOST
from strategies.entry import EntryMarketData

NOW = time.time()
HOUR = 3600.0


def run(coro):
    return asyncio.run(coro)


def buy_market() -> EntryMarketData:
    """A dip-BUY-eligible market (proven runner, stabilised ~40% dip)."""
    return EntryMarketData(
        current_price=0.60,
        ath_price=1.00,
        ath_timestamp=NOW - 5 * HOUR,
        price_history=[1.00, 0.90, 0.75, 0.62, 0.61, 0.60, 0.605, 0.60],
        volume_history=[100.0, 100.0, 100.0, 100.0, 1000.0],
        current_liquidity=100_000.0,
        pre_dip_liquidity=110_000.0,
        now=NOW,
    )


def wait_market() -> EntryMarketData:
    """A shallow pullback -> both strategies WAIT (no position)."""
    return EntryMarketData(
        current_price=0.90,
        ath_price=1.00,
        ath_timestamp=NOW - 5 * HOUR,
        price_history=[1.00, 0.95, 0.92, 0.91, 0.90, 0.90],
        volume_history=[100.0, 100.0, 100.0, 100.0, 1000.0],
        current_liquidity=100_000.0,
        pre_dip_liquidity=110_000.0,
        now=NOW,
    )


def report(mint: str, *, passed: bool = True) -> SafetyReport:
    if not passed:
        return _fail_closed_report(mint, "LP not locked")
    return SafetyReport(
        mint_address=mint, lp_locked_or_burned=True, lp_lock_detail="LP burned",
        tax_pct=0.0, top10_holder_pct=5.0, holder_concentration_pass=True,
        funding_source_clustered=False, farmed_volume_flag=False,
        volume_to_mcap_ratio=0.1, passed=True, reasons=[],
    )


def candidate(mint: str, symbol: str = "TKN") -> Dict[str, Any]:
    return {"mint": mint, "symbol": symbol}


def make_scanner(cands):
    async def _scan():
        return list(cands)
    return _scan


def provider_for(markets):
    async def _provide(cand):
        return markets.get(cand["mint"])
    return _provide


def checker_for(reports):
    async def _check(mint):
        return reports[mint]
    return _check


def boost_provider(boosts: Dict[str, MetaBoost], calls: Optional[List[str]] = None):
    """Injected meta provider: records the mints it is called for."""
    async def _provider(mint: str, price: float) -> MetaBoost:
        if calls is not None:
            calls.append(mint)
        return boosts.get(mint, NEUTRAL_BOOST)
    return _provider


def evaluate(cands, markets, reports, boosts, calls=None):
    return run(
        main.scan_and_evaluate(
            max_candidates=10,
            max_iterations=2,
            scanner=make_scanner(cands),
            safety_checker=checker_for(reports),
            entry_market_provider=provider_for(markets),
            snapshot_provider=mock_market_snapshot,
            meta_boost_provider=boost_provider(boosts, calls),
        )
    )


def test_meta_boost_only_computed_for_entered_tokens() -> None:
    """The provider is invoked ONLY for entered tokens; others get no boost."""
    cands = [candidate("GOOD"), candidate("SHALLOW"), candidate("UNSAFE")]
    markets = {"GOOD": buy_market(), "SHALLOW": wait_market(), "UNSAFE": buy_market()}
    reports = {
        "GOOD": report("GOOD"),
        "SHALLOW": report("SHALLOW"),
        "UNSAFE": report("UNSAFE", passed=False),
    }
    boosts = {"GOOD": MetaBoost(boost=1.3, buyer_breadth=15, available=True)}
    calls: List[str] = []

    by_mint = {s.mint_address: s for s in evaluate(cands, markets, reports, boosts, calls)}

    # The gate: meta ran for the entered token ONLY.
    assert calls == ["GOOD"]

    good = by_mint["GOOD"]
    assert good.status == "entered"
    assert good.meta_boost is not None and good.meta_boost.boost == 1.3
    assert good.priority_score == 1.3

    for other in ("SHALLOW", "UNSAFE"):
        assert by_mint[other].meta_boost is None       # never scored
        assert by_mint[other].priority_score == 0.0


def test_strong_meta_cannot_trigger_a_buy() -> None:
    """A big meta boost on a non-qualifying (WAIT) token never buys it."""
    cands = [candidate("HYPE")]
    markets = {"HYPE": wait_market()}          # entry is WAIT regardless of meta
    reports = {"HYPE": report("HYPE")}
    boosts = {"HYPE": MetaBoost(boost=MetaBoost().boost + 0.5, available=True)}
    calls: List[str] = []

    session = evaluate(cands, markets, reports, boosts, calls)[0]

    assert session.status == "no_entry"        # meta did not force an entry
    assert session.position is None
    assert session.meta_boost is None
    assert calls == []                         # provider never even called


def test_entered_tokens_ranked_by_meta_boost() -> None:
    """Entered tokens are prioritised by their meta boost (broadest first)."""
    cands = [candidate("LOW"), candidate("HIGH")]   # discovery order: LOW first
    markets = {"LOW": buy_market(), "HIGH": buy_market()}
    reports = {"LOW": report("LOW"), "HIGH": report("HIGH")}
    boosts = {
        "LOW": MetaBoost(boost=1.05, buyer_breadth=2, available=True),
        "HIGH": MetaBoost(boost=1.40, buyer_breadth=18, kol_hit=True, available=True),
    }

    sessions = evaluate(cands, markets, reports, boosts)
    # Returned list stays in discovery order...
    assert [s.mint_address for s in sessions] == ["LOW", "HIGH"]
    # ...but the ranking overlay prioritises the higher meta boost.
    ranked = main.meta_ranked_entries(sessions)
    assert [s.mint_address for s in ranked] == ["HIGH", "LOW"]
    by_mint = {s.mint_address: s for s in sessions}
    assert by_mint["HIGH"].priority_score > by_mint["LOW"].priority_score


def test_absent_meta_signal_never_blocks_a_token() -> None:
    """A token whose signals are absent still ENTERS; priority just stays at base."""
    cands = [candidate("SILENT")]
    markets = {"SILENT": buy_market()}
    reports = {"SILENT": report("SILENT")}
    # Provider yields the neutral boost (e.g. Helius unavailable / fail-closed).
    boosts = {"SILENT": NEUTRAL_BOOST}

    session = evaluate(cands, markets, reports, boosts)[0]

    assert session.status == "entered"             # NOT blocked / dropped
    assert session.position is not None
    assert session.meta_boost is not None and session.meta_boost.boost == 1.0
    assert session.priority_score == 1.0           # neutral base, no penalty
    # And it still appears in the ranking (never filtered out).
    assert [s.mint_address for s in main.meta_ranked_entries([session])] == ["SILENT"]
