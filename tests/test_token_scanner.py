"""Tests for the read-only candidate scanner (scanner.token_scanner).

Drives scan_candidates against an injected ``pair_fetcher`` (no network),
covering: good candidates surfaced, out-of-band tokens filtered out, a too-new
token excluded, an API failure returning an empty list (fail-closed), and
de-duplication by mint.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from scanner.token_scanner import scan_candidates

NOW = 1_000_000_000.0          # fixed epoch seconds for deterministic age
NOW_MS = NOW * 1000.0
HOUR_MS = 3_600_000.0


def run(coro):
    return asyncio.run(coro)


def make_pair(
    mint: str = "MINT_GOOD",
    *,
    dex: str = "raydium",
    chain: str = "solana",
    symbol: str = "GOOD",
    market_cap: Any = 1_000_000.0,
    liquidity: Any = 50_000.0,
    volume_24h: Any = 120_000.0,
    age_hours: float = 6.0,
    pair_address: str = "PAIR",
    include_marketcap: bool = True,
    freeze_authority: Any = None,
    locked_lp_pct: Any = None,
) -> Dict[str, Any]:
    """Build a raw Dexscreener pair dict; defaults clear every filter.

    ``freeze_authority`` / ``locked_lp_pct``, when not None, inject the optional
    security signals the scanner reads opportunistically.
    """
    pair: Dict[str, Any] = {
        "chainId": chain,
        "dexId": dex,
        "pairAddress": pair_address,
        "baseToken": {"address": mint, "symbol": symbol},
        "priceUsd": "0.5",
        "liquidity": {"usd": liquidity},
        "volume": {"h24": volume_24h},
        "pairCreatedAt": NOW_MS - age_hours * HOUR_MS,
    }
    if include_marketcap:
        pair["marketCap"] = market_cap
    if freeze_authority is not None:
        pair["hasFreezeAuthority"] = freeze_authority
    if locked_lp_pct is not None:
        pair["lpLockedPct"] = locked_lp_pct
    return pair


def fetcher_for(pairs: List[Dict[str, Any]]):
    async def _fetch() -> List[Dict[str, Any]]:
        return pairs
    return _fetch


def test_good_candidates_surfaced() -> None:
    """Valid Solana pairs on allowed DEXs are surfaced, sorted by volume."""
    pairs = [
        make_pair("MINT_A", symbol="AAA", volume_24h=150_000.0),
        make_pair("MINT_B", symbol="BBB", dex="orca", volume_24h=200_000.0),
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))

    assert [c["mint"] for c in out] == ["MINT_B", "MINT_A"]  # volume desc
    b = out[0]
    assert b["source_dex"] == "orca"
    assert b["symbol"] == "BBB"
    assert b["market_cap_usd"] == 1_000_000.0
    assert b["liquidity_usd"] == 50_000.0
    assert b["volume_24h_usd"] == 200_000.0
    assert abs(b["age_hours"] - 6.0) < 1e-6


def test_out_of_band_tokens_filtered_out() -> None:
    """Tokens failing any numeric band / DEX gate are dropped."""
    pairs = [
        make_pair("KEEP", volume_24h=150_000.0),                 # passes
        make_pair("MCAP_LOW", market_cap=10_000.0),              # mcap < 50k
        make_pair("MCAP_HIGH", market_cap=100_000_000.0),        # mcap > 50M
        make_pair("LIQ_LOW", liquidity=20_000.0),                # liq < 30k
        make_pair("VOL_LOW", volume_24h=80_000.0),               # vol < 100k
        make_pair("BAD_DEX", dex="somenewdex"),                  # not allowed
        make_pair("WRONG_CHAIN", chain="ethereum"),              # not solana
        make_pair("NO_MCAP", include_marketcap=False),           # missing mcap (no fdv)
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))

    assert [c["mint"] for c in out] == ["KEEP"]


def test_too_new_token_excluded() -> None:
    """A pair younger than the minimum age is excluded (no proven run yet)."""
    pairs = [
        make_pair("ESTABLISHED", age_hours=6.0),
        make_pair("TOO_NEW", age_hours=3.0),   # < 6h default
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))

    assert [c["mint"] for c in out] == ["ESTABLISHED"]


def test_api_failure_returns_empty_list() -> None:
    """A fetch error fails closed to an empty list, never raising."""

    async def boom() -> List[Dict[str, Any]]:
        raise RuntimeError("dexscreener 503")

    out = run(scan_candidates(pair_fetcher=boom, now=NOW))
    assert out == []


def test_deduplicates_by_mint() -> None:
    """The same mint across multiple pairs surfaces once (highest volume)."""
    pairs = [
        make_pair("DUP", pair_address="P1", volume_24h=120_000.0),
        make_pair("DUP", pair_address="P2", dex="orca", volume_24h=300_000.0),
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))

    assert len(out) == 1
    assert out[0]["mint"] == "DUP"
    assert out[0]["volume_24h_usd"] == 300_000.0   # higher-volume pair kept
    assert out[0]["pair_address"] == "P2"


def test_partial_field_skipped_not_surfaced() -> None:
    """A pair missing a required numeric field is skipped (fail-closed)."""
    bad = make_pair("PARTIAL")
    bad["liquidity"] = {}  # no usd -> float(None) -> skip
    out = run(scan_candidates(pair_fetcher=fetcher_for([bad]), now=NOW))
    assert out == []


# --- Tighter filters --------------------------------------------------------

def test_min_liquidity_default_30k_and_configurable() -> None:
    """Default liquidity floor is $30k; below is dropped, and it is tunable."""
    pairs = [
        make_pair("THIN", liquidity=20_000.0),   # < 30k default
        make_pair("DEEP", liquidity=35_000.0),   # >= 30k default
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))
    assert [c["mint"] for c in out] == ["DEEP"]

    # Loosening the floor surfaces the thinner pool again.
    out_loose = run(
        scan_candidates(
            pair_fetcher=fetcher_for(pairs), now=NOW, min_liquidity_usd=15_000.0
        )
    )
    assert {c["mint"] for c in out_loose} == {"THIN", "DEEP"}


def test_min_volume_default_100k() -> None:
    """Default 24h volume floor is $100k."""
    pairs = [
        make_pair("QUIET", volume_24h=80_000.0),   # < 100k default
        make_pair("BUSY", volume_24h=250_000.0),   # >= 100k default
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))
    assert [c["mint"] for c in out] == ["BUSY"]


def test_min_age_default_6h() -> None:
    """Default minimum pair age is 6h (real trading history)."""
    pairs = [
        make_pair("FRESH", age_hours=5.0),        # < 6h default
        make_pair("SEASONED", age_hours=48.0),    # >= 6h default
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))
    assert [c["mint"] for c in out] == ["SEASONED"]


def test_only_established_dexes_allowed() -> None:
    """Only Raydium / Orca / Meteora pass; previously-tolerated venues drop."""
    pairs = [
        make_pair("RAY", dex="raydium"),
        make_pair("ORC", dex="orca"),
        make_pair("MET", dex="meteora"),
        make_pair("LIF", dex="lifinity"),   # no longer allowed
        make_pair("PHX", dex="phoenix"),    # no longer allowed
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))
    assert {c["mint"] for c in out} == {"RAY", "ORC", "MET"}


def test_ungraduated_pumpfun_excluded_even_if_dex_allowed() -> None:
    """An un-graduated Pump.fun pair is rejected even if the allow-list has it."""
    pairs = [make_pair("BONDING", dex="pumpfun")]
    # Explicitly widen the allow-list to include pumpfun; it must STILL drop.
    out = run(
        scan_candidates(
            pair_fetcher=fetcher_for(pairs),
            now=NOW,
            allowed_dexes=frozenset({"raydium", "orca", "meteora", "pumpfun"}),
        )
    )
    assert out == []


def test_freeze_authority_excluded_and_configurable() -> None:
    """A pair flagged with freeze authority is dropped by default; toggleable."""
    pairs = [
        make_pair("FROZEN", freeze_authority=True),
        make_pair("CLEAN", freeze_authority=False),
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))
    assert [c["mint"] for c in out] == ["CLEAN"]

    # Disabling the gate surfaces the flagged token again.
    out_off = run(
        scan_candidates(
            pair_fetcher=fetcher_for(pairs), now=NOW, exclude_freeze_authority=False
        )
    )
    assert {c["mint"] for c in out_off} == {"FROZEN", "CLEAN"}


def test_freeze_authority_address_form() -> None:
    """A real freeze-authority address is risky; the null/system address is safe."""
    risky = make_pair("RISKY")
    risky["freezeAuthority"] = "So11111111111111111111111111111111111111112"
    safe = make_pair("SAFE")
    safe["freezeAuthority"] = "11111111111111111111111111111111"  # revoked/null
    out = run(scan_candidates(pair_fetcher=fetcher_for([risky, safe]), now=NOW))
    assert [c["mint"] for c in out] == ["SAFE"]


def test_unlocked_lp_excluded_but_unknown_allowed() -> None:
    """0% locked LP is dropped; a locked LP passes; unknown lock is allowed."""
    pairs = [
        make_pair("UNLOCKED", locked_lp_pct=0.0),   # explicitly 0% -> drop
        make_pair("LOCKED", locked_lp_pct=100.0),   # locked -> keep
        make_pair("UNKNOWN"),                        # no signal -> keep
    ]
    out = run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW))
    assert {c["mint"] for c in out} == {"LOCKED", "UNKNOWN"}


def test_security_fields_default_none_when_absent() -> None:
    """Security signals surface as None when the pair carries no such data."""
    out = run(scan_candidates(pair_fetcher=fetcher_for([make_pair("PLAIN")]), now=NOW))
    assert len(out) == 1
    assert out[0]["freeze_authority"] is None
    assert out[0]["locked_lp_pct"] is None
