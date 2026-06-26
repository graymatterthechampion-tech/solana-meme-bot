"""Tests for the scan -> evaluate orchestration in main.scan_and_evaluate.

Proves the scanner is wired into the full pipeline: a mocked candidate list is
discovered, each surfaced mint is run through evaluate_and_trade (safety gate ->
entry decision -> managed loop), the per-scan cap is enforced, and fail-closed
candidates are recorded without ever reaching safety/entry.

No network: the scanner, safety checker, entry-market provider, and snapshot
provider are all injected.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import main
from data.market_feed import mock_market_snapshot
from safety.rug_check import SafetyReport, _fail_closed_report
from strategies.entry import EntryAction, EntryMarketData

NOW = time.time()
HOUR = 3600.0
ENTRY_PRICE = 0.001
ENTRY_LIQUIDITY = 100_000.0


def run(coro):
    return asyncio.run(coro)


def make_candidate(mint: str, symbol: str = "TKN") -> Dict[str, Any]:
    """A scanner-shaped candidate dict (the subset the orchestrator reads)."""
    return {
        "mint": mint,
        "symbol": symbol,
        "source_dex": "raydium",
        "market_cap_usd": 1_000_000.0,
        "liquidity_usd": ENTRY_LIQUIDITY,
        "volume_24h_usd": 500_000.0,
        "age_hours": 6.0,
        "price_usd": ENTRY_PRICE * 0.60,
        "pair_address": f"pair-{mint}",
    }


def make_scanner(candidates: List[Dict[str, Any]]):
    """An injected CandidateScanner returning a fixed list (no network)."""

    async def scanner() -> List[Dict[str, Any]]:
        return list(candidates)

    return scanner


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


async def buy_entry_market(candidate: Dict[str, Any]) -> EntryMarketData:
    """A BUY-eligible market for any candidate: proven runner, stabilised dip."""
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


def test_each_candidate_processed_through_pipeline() -> None:
    """Every surfaced candidate is run through safety -> entry -> loop."""
    candidates = [make_candidate(f"MINT{i}", f"TK{i}") for i in range(3)]

    # Spy on the safety checker so we can prove it ran once per candidate.
    seen_mints: List[str] = []

    async def spy_safe(mint: str) -> SafetyReport:
        seen_mints.append(mint)
        return await safe_checker(mint)

    sessions = run(
        main.scan_and_evaluate(
            max_candidates=10,
            max_iterations=3,
            scanner=make_scanner(candidates),
            safety_checker=spy_safe,
            entry_market_provider=buy_entry_market,
            snapshot_provider=mock_market_snapshot,
        )
    )

    # One session per candidate, in surfaced order, each fully processed.
    assert [s.mint_address for s in sessions] == ["MINT0", "MINT1", "MINT2"]
    assert seen_mints == ["MINT0", "MINT1", "MINT2"]
    assert all(s.status == "entered" for s in sessions)
    # Each reached the managed loop and ran the bounded iterations.
    for s in sessions:
        assert s.entry_decision is not None
        assert s.entry_decision.action is EntryAction.BUY
        assert s.position is not None and s.position.original_size > 0
        assert len(s.loop_outcomes) == 3


def test_max_candidates_caps_per_scan_work() -> None:
    """Only the first ``max_candidates`` surfaced tokens are evaluated."""
    candidates = [make_candidate(f"MINT{i}") for i in range(5)]
    evaluated: List[str] = []

    async def spy_safe(mint: str) -> SafetyReport:
        evaluated.append(mint)
        return await safe_checker(mint)

    sessions = run(
        main.scan_and_evaluate(
            max_candidates=2,
            max_iterations=1,
            scanner=make_scanner(candidates),
            safety_checker=spy_safe,
            entry_market_provider=buy_entry_market,
            snapshot_provider=mock_market_snapshot,
        )
    )

    # Cap honoured: only 2 processed even though 5 were surfaced.
    assert len(sessions) == 2
    assert [s.mint_address for s in sessions] == ["MINT0", "MINT1"]
    assert evaluated == ["MINT0", "MINT1"]


def test_mixed_outcomes_recorded_per_candidate() -> None:
    """Safety-fail, no-entry, and entered are each surfaced independently."""
    candidates = [
        make_candidate("UNSAFE"),
        make_candidate("SHALLOW"),
        make_candidate("GOOD"),
    ]

    async def mixed_safe(mint: str) -> SafetyReport:
        if mint == "UNSAFE":
            return _fail_closed_report(mint, "LP not locked")
        return await safe_checker(mint)

    async def mixed_entry(candidate: Dict[str, Any]) -> EntryMarketData:
        market = await buy_entry_market(candidate)
        if candidate["mint"] == "SHALLOW":
            # Only ~10% off ATH -> WAIT (no position).
            market.current_price = ENTRY_PRICE * 0.90
            market.price_history = [
                ENTRY_PRICE * m for m in (1.0, 0.95, 0.92, 0.91, 0.90, 0.90)
            ]
        return market

    sessions = run(
        main.scan_and_evaluate(
            max_candidates=10,
            max_iterations=2,
            scanner=make_scanner(candidates),
            safety_checker=mixed_safe,
            entry_market_provider=mixed_entry,
            snapshot_provider=mock_market_snapshot,
        )
    )

    status_by_mint = {s.mint_address: s.status for s in sessions}
    assert status_by_mint == {
        "UNSAFE": "rejected_safety",
        "SHALLOW": "no_entry",
        "GOOD": "entered",
    }


def test_no_market_data_fails_closed_before_safety() -> None:
    """A candidate with no buildable market data is skipped, safety untouched."""
    candidates = [make_candidate("NODATA")]
    safety_calls: List[str] = []

    async def spy_safe(mint: str) -> SafetyReport:  # pragma: no cover - must not run
        safety_calls.append(mint)
        return await safe_checker(mint)

    async def no_market(candidate: Dict[str, Any]) -> Optional[EntryMarketData]:
        return None

    sessions = run(
        main.scan_and_evaluate(
            scanner=make_scanner(candidates),
            safety_checker=spy_safe,
            entry_market_provider=no_market,
        )
    )

    assert len(sessions) == 1
    assert sessions[0].status == "no_market_data"
    assert sessions[0].position is None
    assert sessions[0].safety_report is None
    assert safety_calls == []  # never reached the safety gate


def test_empty_scan_evaluates_nothing() -> None:
    """No surfaced candidates -> no sessions, nothing evaluated."""
    sessions = run(
        main.scan_and_evaluate(
            scanner=make_scanner([]),
            safety_checker=safe_checker,
            entry_market_provider=buy_entry_market,
        )
    )
    assert sessions == []


async def _no_history(mint: str):
    """Injected history fetcher that yields nothing (no network, fail-closed)."""
    return None


def test_default_entry_market_provider_fails_closed_without_history() -> None:
    """The default provider builds data but no history -> entry SKIPs (no_entry).

    The OHLCV fetch is stubbed to None so the test stays offline; the real
    Birdeye/Dexscreener path is covered in test_price_history.
    """
    candidate = make_candidate("REAL")

    market = run(
        main._default_entry_market_provider(candidate, history_fetcher=_no_history)
    )
    assert market is not None
    assert market.current_price == ENTRY_PRICE * 0.60
    assert market.price_history == [] and market.volume_history == []

    # Run it through the full chain: safety passes, entry fails closed to SKIP.
    async def provider(c):
        return await main._default_entry_market_provider(c, history_fetcher=_no_history)

    sessions = run(
        main.scan_and_evaluate(
            scanner=make_scanner([candidate]),
            safety_checker=safe_checker,
            entry_market_provider=provider,
        )
    )
    assert len(sessions) == 1
    assert sessions[0].status == "no_entry"
    assert sessions[0].entry_decision is not None
    assert sessions[0].entry_decision.action is EntryAction.SKIP


def test_default_entry_market_provider_rejects_zero_price() -> None:
    """A candidate with no usable price yields None (fail-closed)."""
    candidate = make_candidate("ZERO")
    candidate["price_usd"] = 0.0
    assert run(main._default_entry_market_provider(candidate)) is None
