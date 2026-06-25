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
) -> Dict[str, Any]:
    """Build a raw Dexscreener pair dict; defaults clear every filter."""
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
    return pair


def fetcher_for(pairs: List[Dict[str, Any]]):
    async def _fetch() -> List[Dict[str, Any]]:
        return pairs
    return _fetch


def test_good_candidates_surfaced() -> None:
    """Valid Solana pairs on allowed DEXs are surfaced, sorted by volume."""
    pairs = [
        make_pair("MINT_A", symbol="AAA", volume_24h=80_000.0),
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
        make_pair("KEEP", volume_24h=90_000.0),                  # passes
        make_pair("MCAP_LOW", market_cap=10_000.0),              # mcap < 50k
        make_pair("MCAP_HIGH", market_cap=100_000_000.0),        # mcap > 50M
        make_pair("LIQ_LOW", liquidity=5_000.0),                 # liq < 10k
        make_pair("VOL_LOW", volume_24h=10_000.0),               # vol < 50k
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
        make_pair("TOO_NEW", age_hours=0.5),   # < 1h default
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
        make_pair("DUP", pair_address="P1", volume_24h=60_000.0),
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
