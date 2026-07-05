"""Tests for the read-only movers scanner (scanner.token_scanner).

Two layers:

  * FILTER/RANK layer — driven against an injected ``pair_fetcher`` that supplies
    already-enriched raw pairs (no network): band/DEX/age/security gates, the
    >=20% recent-momentum gate, de-duplication (deepest-liquidity pair), the
    volume/momentum blend ranking, and the surfaced-count cap.
  * SOURCE layer — driven against an ``httpx.MockTransport`` that simulates the
    Dexscreener boosts + token endpoints and the Birdeye trending endpoint, so
    aggregation across BOTH sources (and the no-key / per-source-failure paths)
    is exercised end-to-end without network.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import httpx
import pytest

from safety import rug_check
from scanner import token_scanner
from scanner.token_scanner import _blend_score, scan_candidates

NOW = 1_000_000_000.0          # fixed epoch seconds for deterministic age
NOW_MS = NOW * 1000.0
HOUR_MS = 3_600_000.0
KEY = "BIRDEYE_TEST_KEY"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # The Birdeye call reuses rug_check's retry/backoff globals; zero the backoff
    # so the failure test doesn't actually sleep between retries.
    monkeypatch.setattr(rug_check, "RATE_LIMIT_BACKOFF", 0.0)


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
    price_change_1h: Any = 5.0,
    price_change_6h: Any = 25.0,   # default clears the +20% momentum gate
    pair_address: str = "PAIR",
    include_marketcap: bool = True,
    include_pricechange: bool = True,
    freeze_authority: Any = None,
    locked_lp_pct: Any = None,
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
    if include_pricechange:
        pair["priceChange"] = {"h1": price_change_1h, "h6": price_change_6h}
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


def scan(pairs: List[Dict[str, Any]], **kwargs):
    return run(scan_candidates(pair_fetcher=fetcher_for(pairs), now=NOW, **kwargs))


# --- Filter / rank layer (injected pairs) -----------------------------------

def test_good_candidates_surfaced() -> None:
    """Valid Solana movers on allowed DEXs surface, ranked by the blend."""
    pairs = [
        make_pair("MINT_A", symbol="AAA", volume_24h=150_000.0),
        make_pair("MINT_B", symbol="BBB", dex="orca", volume_24h=200_000.0),
    ]
    out = scan(pairs)

    # Equal momentum -> higher volume wins the blend.
    assert [c["mint"] for c in out] == ["MINT_B", "MINT_A"]
    b = out[0]
    assert b["source_dex"] == "orca"
    assert b["symbol"] == "BBB"
    assert b["market_cap_usd"] == 1_000_000.0
    assert b["liquidity_usd"] == 50_000.0
    assert b["volume_24h_usd"] == 200_000.0
    assert b["recent_change_pct"] == 25.0
    assert abs(b["age_hours"] - 6.0) < 1e-6


def test_out_of_band_tokens_filtered_out() -> None:
    """Tokens failing any numeric band / DEX / chain gate are dropped."""
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
    out = scan(pairs)
    assert [c["mint"] for c in out] == ["KEEP"]


def test_too_new_token_excluded() -> None:
    """A pair younger than the minimum age is excluded (no proven run yet)."""
    pairs = [
        make_pair("ESTABLISHED", age_hours=6.0),
        make_pair("TOO_NEW", age_hours=3.0),   # < 6h default
    ]
    out = scan(pairs)
    assert [c["mint"] for c in out] == ["ESTABLISHED"]


def test_api_failure_returns_empty_list() -> None:
    """A fetch error fails closed to an empty list, never raising."""

    async def boom() -> List[Dict[str, Any]]:
        raise RuntimeError("dexscreener 503")

    out = run(scan_candidates(pair_fetcher=boom, now=NOW))
    assert out == []


def test_deduplicates_by_mint_keeps_highest_liquidity() -> None:
    """The same mint across pairs surfaces once — the DEEPEST-liquidity pair."""
    pairs = [
        make_pair("DUP", pair_address="P1", liquidity=80_000.0, volume_24h=120_000.0),
        make_pair("DUP", pair_address="P2", dex="orca", liquidity=40_000.0, volume_24h=300_000.0),
    ]
    out = scan(pairs)

    assert len(out) == 1
    assert out[0]["mint"] == "DUP"
    assert out[0]["liquidity_usd"] == 80_000.0   # deepest-liquidity pair kept
    assert out[0]["pair_address"] == "P1"


def test_partial_field_skipped_not_surfaced() -> None:
    """A pair missing a required numeric field is skipped (fail-closed)."""
    bad = make_pair("PARTIAL")
    bad["liquidity"] = {}  # no usd -> float(None) -> skip
    assert scan([bad]) == []


# --- Momentum gate (the +20% recent-change filter) --------------------------

def test_flat_and_falling_tokens_dropped() -> None:
    """Only tokens up >= 20% over 1h OR 6h survive; flat/falling are dropped."""
    pairs = [
        make_pair("UP_1H", price_change_1h=30.0, price_change_6h=-5.0),   # 1h qualifies
        make_pair("UP_6H", price_change_1h=-10.0, price_change_6h=22.0),  # 6h qualifies
        make_pair("FLAT", price_change_1h=5.0, price_change_6h=10.0),     # < 20% -> drop
        make_pair("FALLING", price_change_1h=-30.0, price_change_6h=-40.0),  # drop
        make_pair("NO_CHANGE", include_pricechange=False),               # unknown -> drop
    ]
    out = scan(pairs)
    assert {c["mint"] for c in out} == {"UP_1H", "UP_6H"}


def test_momentum_threshold_configurable() -> None:
    """The momentum floor is tunable (raise it to demand a stronger move)."""
    pairs = [
        make_pair("MILD", price_change_1h=25.0, price_change_6h=25.0),
        make_pair("STRONG", price_change_1h=60.0, price_change_6h=60.0),
    ]
    # Default 20% keeps both; a 50% floor keeps only the strong mover.
    assert {c["mint"] for c in scan(pairs)} == {"MILD", "STRONG"}
    assert [c["mint"] for c in scan(pairs, min_recent_change_pct=50.0)] == ["STRONG"]


# --- Blend ranking + cap ----------------------------------------------------

def test_blend_score_formula() -> None:
    """The blend is volume_weight*log10(vol) + momentum_weight*change_pct."""
    cand = {"volume_24h_usd": 100_000.0, "recent_change_pct": 20.0}
    score = _blend_score(cand, volume_weight=1.0, momentum_weight=0.05)
    assert math.isclose(score, math.log10(100_000.0) + 0.05 * 20.0)


def test_blend_ranks_a_big_mover_above_a_whale_volume_flat() -> None:
    """A strong mover outranks a much-higher-volume, barely-moving token."""
    pairs = [
        make_pair("WHALE", volume_24h=5_000_000.0, price_change_1h=20.0, price_change_6h=20.0),
        make_pair("MOVER", volume_24h=120_000.0, price_change_1h=120.0, price_change_6h=120.0),
    ]
    out = scan(pairs)
    assert [c["mint"] for c in out] == ["MOVER", "WHALE"]


def test_surfaced_count_capped() -> None:
    """At most ``max_candidates`` are surfaced (default 15; configurable)."""
    pairs = [
        make_pair(f"MINT{i}", volume_24h=100_000.0 + i * 1_000.0)
        for i in range(20)
    ]
    assert len(scan(pairs)) == 15               # default cap
    assert len(scan(pairs, max_candidates=3)) == 3


# --- DEX allow-list ---------------------------------------------------------

def test_only_established_dexes_allowed() -> None:
    """Raydium / Orca / Meteora / Pump.fun AMM pass; other venues drop."""
    pairs = [
        make_pair("RAY", dex="raydium"),
        make_pair("ORC", dex="orca"),
        make_pair("MET", dex="meteora"),
        make_pair("PSWAP", dex="pumpswap"),   # graduated Pump.fun AMM
        make_pair("LIF", dex="lifinity"),     # not allowed
        make_pair("PHX", dex="phoenix"),      # not allowed
    ]
    out = scan(pairs)
    assert {c["mint"] for c in out} == {"RAY", "ORC", "MET", "PSWAP"}


def test_ungraduated_pumpfun_excluded_even_if_dex_allowed() -> None:
    """An un-graduated Pump.fun (bonding-curve) pair is rejected outright."""
    pairs = [make_pair("BONDING", dex="pumpfun")]
    out = scan(
        pairs,
        allowed_dexes=frozenset({"raydium", "orca", "meteora", "pumpswap", "pumpfun"}),
    )
    assert out == []


# --- Security gates ---------------------------------------------------------

def test_freeze_authority_excluded_and_configurable() -> None:
    """A pair flagged with freeze authority is dropped by default; toggleable."""
    pairs = [
        make_pair("FROZEN", freeze_authority=True),
        make_pair("CLEAN", freeze_authority=False),
    ]
    assert [c["mint"] for c in scan(pairs)] == ["CLEAN"]
    out_off = scan(pairs, exclude_freeze_authority=False)
    assert {c["mint"] for c in out_off} == {"FROZEN", "CLEAN"}


def test_freeze_authority_address_form() -> None:
    """A real freeze-authority address is risky; the null/system address is safe."""
    risky = make_pair("RISKY")
    risky["freezeAuthority"] = "So11111111111111111111111111111111111111112"
    safe = make_pair("SAFE")
    safe["freezeAuthority"] = "11111111111111111111111111111111"  # revoked/null
    out = scan([risky, safe])
    assert [c["mint"] for c in out] == ["SAFE"]


def test_unlocked_lp_excluded_but_unknown_allowed() -> None:
    """0% locked LP is dropped; a locked LP passes; unknown lock is allowed."""
    pairs = [
        make_pair("UNLOCKED", locked_lp_pct=0.0),   # explicitly 0% -> drop
        make_pair("LOCKED", locked_lp_pct=100.0),   # locked -> keep
        make_pair("UNKNOWN"),                        # no signal -> keep
    ]
    out = scan(pairs)
    assert {c["mint"] for c in out} == {"LOCKED", "UNKNOWN"}


def test_security_fields_default_none_when_absent() -> None:
    """Security signals surface as None when the pair carries no such data."""
    out = scan([make_pair("PLAIN")])
    assert len(out) == 1
    assert out[0]["freeze_authority"] is None
    assert out[0]["locked_lp_pct"] is None


# --- Source layer (Dexscreener boosts + Birdeye trending via MockTransport) --

def make_transport(
    *,
    boost_top: Optional[List[Dict[str, Any]]] = None,
    boost_latest: Optional[List[Dict[str, Any]]] = None,
    birdeye_tokens: Optional[List[str]] = None,
    pairs_by_mint: Optional[Dict[str, Any]] = None,
    expect_birdeye_key: Optional[str] = None,
    birdeye_status: Optional[int] = None,
):
    """A MockTransport handler routing the scanner's read-only endpoints."""
    calls = {"boost_top": 0, "boost_latest": 0, "birdeye": 0, "tokens": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        path = request.url.path

        if path.endswith("/token-boosts/top/v1"):
            calls["boost_top"] += 1
            return httpx.Response(200, json=boost_top or [])
        if path.endswith("/token-boosts/latest/v1"):
            calls["boost_latest"] += 1
            return httpx.Response(200, json=boost_latest or [])
        if "token_trending" in url:
            calls["birdeye"] += 1
            if expect_birdeye_key is not None:
                assert request.headers.get("X-API-KEY") == expect_birdeye_key
            if birdeye_status and birdeye_status >= 500:
                return httpx.Response(birdeye_status, json={})
            tokens = [{"address": a} for a in (birdeye_tokens or [])]
            return httpx.Response(200, json={"data": {"tokens": tokens}})
        if "/latest/dex/tokens/" in path:
            calls["tokens"] += 1
            addrs = unquote(path.split("/latest/dex/tokens/")[1]).split(",")
            pairs: List[Dict[str, Any]] = []
            for addr in addrs:
                entry = (pairs_by_mint or {}).get(addr)
                if entry is None:
                    continue
                pairs.extend(entry if isinstance(entry, list) else [entry])
            return httpx.Response(200, json={"pairs": pairs})
        return httpx.Response(404, json={})

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def scan_network(handler, *, api_key: Optional[str] = KEY, **kwargs):
    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await scan_candidates(
                client=client, birdeye_api_key=api_key, now=NOW, **kwargs
            )
        finally:
            await client.aclose()

    return run(_run())


def test_aggregates_both_sources() -> None:
    """Addresses from BOTH Dexscreener boosts and Birdeye trending are enriched
    and surfaced, each stamped with its discovery source(s)."""
    handler = make_transport(
        boost_top=[{"chainId": "solana", "tokenAddress": "DEXMINT"}],
        boost_latest=[{"chainId": "ethereum", "tokenAddress": "NOT_SOL"}],  # dropped
        birdeye_tokens=["BIRDMINT"],
        pairs_by_mint={
            "DEXMINT": make_pair("DEXMINT", symbol="DEX", volume_24h=150_000.0),
            "BIRDMINT": make_pair("BIRDMINT", symbol="BIRD", dex="orca", volume_24h=200_000.0),
        },
        expect_birdeye_key=KEY,
    )
    out = scan_network(handler)

    by_mint = {c["mint"]: c for c in out}
    assert set(by_mint) == {"DEXMINT", "BIRDMINT"}
    assert by_mint["DEXMINT"]["discovery_sources"] == ["dexscreener"]
    assert by_mint["BIRDMINT"]["discovery_sources"] == ["birdeye"]
    # The non-Solana boost address was never enriched into a candidate.
    assert "NOT_SOL" not in by_mint
    assert handler.calls["birdeye"] == 1  # exactly one metered Birdeye call


def test_mint_from_both_sources_marked_both() -> None:
    """A mint surfaced by both sources records both discovery sources, once."""
    handler = make_transport(
        boost_top=[{"chainId": "solana", "tokenAddress": "SHARED"}],
        birdeye_tokens=["SHARED"],
        pairs_by_mint={"SHARED": make_pair("SHARED")},
    )
    out = scan_network(handler)
    assert len(out) == 1
    assert out[0]["discovery_sources"] == ["birdeye", "dexscreener"]


def test_no_birdeye_key_skips_birdeye_source() -> None:
    """With no API key the Birdeye endpoint is never called; Dexscreener works.

    An explicit empty key forces the no-key path hermetically, ignoring any
    ambient ``BIRDEYE_API_KEY`` in the environment/.env.
    """
    handler = make_transport(
        boost_top=[{"chainId": "solana", "tokenAddress": "DEXMINT"}],
        birdeye_tokens=["BIRDMINT"],
        pairs_by_mint={
            "DEXMINT": make_pair("DEXMINT"),
            "BIRDMINT": make_pair("BIRDMINT"),
        },
    )
    out = scan_network(handler, api_key="")

    assert [c["mint"] for c in out] == ["DEXMINT"]
    assert handler.calls["birdeye"] == 0        # no metered call without a key
    assert handler.calls["tokens"] == 1


def test_birdeye_failure_falls_back_to_dexscreener() -> None:
    """A Birdeye 5xx doesn't sink the scan; the Dexscreener source still surfaces."""
    handler = make_transport(
        boost_top=[{"chainId": "solana", "tokenAddress": "DEXMINT"}],
        birdeye_tokens=["BIRDMINT"],
        pairs_by_mint={"DEXMINT": make_pair("DEXMINT")},
        birdeye_status=503,
    )
    out = scan_network(handler)
    assert [c["mint"] for c in out] == ["DEXMINT"]


def test_no_movers_discovered_returns_empty() -> None:
    """No addresses from either source -> no token lookup, empty result."""
    handler = make_transport(boost_top=[], boost_latest=[], birdeye_tokens=[])
    out = scan_network(handler)
    assert out == []
    assert handler.calls["tokens"] == 0
