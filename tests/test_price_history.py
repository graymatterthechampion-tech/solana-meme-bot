"""Tests for the read-only OHLCV history layer (data.price_history).

Uses httpx.MockTransport to simulate Birdeye and Dexscreener (no network).
Covers: a full Birdeye history that lets evaluate_entry judge ATH / pullback;
Birdeye failure falling back to Dexscreener; both sources failing -> None ->
entry fails closed to SKIP; a too-short Birdeye result triggering fallback;
no API key skipping Birdeye; and the reused 429 retry/backoff path.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

import config
from data import price_history
from data.price_history import (
    build_entry_market_data,
    get_price_history,
)
from safety import rug_check
from strategies.entry import EntryAction, EntryMarketData, evaluate_entry

NOW = 1_700_000_000.0
HOUR = 3600.0
API_KEY = "test-birdeye-key"

# The real, verbatim body Birdeye returns (with HTTP 400) when the API plan does
# not grant the OHLCV/time-series endpoints. Confirmed against the live API: the
# same body comes back for /defi/ohlcv and /defi/v3/ohlcv, with or without the
# x-chain header, for any candle count — while /defi/price returns 200 on the
# same key. It is a permanent plan gate, not a malformed request.
BIRDEYE_PLAN_ERROR: Dict[str, Any] = {
    "success": False,
    "message": "Compute units usage limit exceeded",
}


def run(coro):
    return asyncio.run(coro)


# --- Birdeye fixtures --------------------------------------------------------

def proven_runner_items() -> List[Dict[str, Any]]:
    """OHLCV items for a proven runner: ATH 5h ago (10x volume spike), now in a
    stabilised ~40% dip. Drives evaluate_entry to BUY."""
    return [
        # (seconds_ago, open, high, low, close, volume)
        _item(6 * HOUR, 0.50, 0.50, 0.48, 0.50, 100.0),
        _item(5 * HOUR, 0.90, 1.00, 0.90, 1.00, 1000.0),   # ATH + volume spike
        _item(4 * HOUR, 0.85, 0.86, 0.78, 0.80, 100.0),
        _item(3 * HOUR, 0.75, 0.76, 0.69, 0.70, 100.0),
        _item(2 * HOUR, 0.66, 0.66, 0.60, 0.62, 100.0),
        _item(600.0, 0.61, 0.61, 0.59, 0.60, 100.0),
        _item(300.0, 0.60, 0.61, 0.60, 0.605, 100.0),
        _item(0.0, 0.605, 0.605, 0.60, 0.60, 100.0),
    ]


def _item(
    seconds_ago: float, o: float, h: float, low: float, c: float, v: float
) -> Dict[str, Any]:
    return {
        "unixTime": int(NOW - seconds_ago),
        "o": o,
        "h": h,
        "l": low,
        "c": c,
        "v": v,
    }


def birdeye_payload(items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    items = proven_runner_items() if items is None else items
    return {"success": True, "data": {"items": items}}


# --- Dexscreener fixtures ----------------------------------------------------

def dexscreener_payload(price: str = "0.60", chain: str = "solana") -> Dict[str, Any]:
    """Token payload with trailing price-change / volume buckets for the
    coarse fallback series."""
    return {
        "pairs": [
            {
                "chainId": chain,
                "priceUsd": price,
                "priceChange": {"m5": -0.5, "h1": -3.0, "h6": -20.0, "h24": -40.0},
                "volume": {
                    "m5": 5_000.0, "h1": 20_000.0, "h6": 80_000.0, "h24": 150_000.0,
                },
                "liquidity": {"usd": 100_000.0},
            }
        ]
    }


# --- GeckoTerminal fixtures --------------------------------------------------

def geckoterminal_pools_payload(
    pool: str = "POOL1", reserve_in_usd: str = "100000"
) -> Dict[str, Any]:
    """token->pools response with a single, deepest-liquidity Solana pool."""
    return {
        "data": [
            {
                "id": f"solana_{pool}",
                "type": "pool",
                "attributes": {"address": pool, "reserve_in_usd": reserve_in_usd},
            }
        ]
    }


def geckoterminal_ohlcv_payload(
    items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """OHLCV response in GeckoTerminal's shape: newest-first ``[ts,o,h,l,c,v]``
    rows under ``data.attributes.ohlcv_list`` (reuses the proven-runner series)."""
    items = proven_runner_items() if items is None else items
    rows = [[it["unixTime"], it["o"], it["h"], it["l"], it["c"], it["v"]] for it in items]
    rows.reverse()  # GeckoTerminal returns the most recent candle first
    return {
        "data": {
            "id": "solana_POOL1",
            "type": "ohlcv_request_response",
            "attributes": {"ohlcv_list": rows},
        }
    }


# --- Mock transport ----------------------------------------------------------

def make_handler(
    *,
    gecko_pools_json: Optional[Dict[str, Any]] = None,
    gecko_ohlcv_json: Optional[Dict[str, Any]] = None,
    gecko_raises: Optional[Exception] = None,
    gecko_status_sequence: Optional[List[int]] = None,
    birdeye_json: Optional[Dict[str, Any]] = None,
    birdeye_raises: Optional[Exception] = None,
    birdeye_status_sequence: Optional[List[int]] = None,
    dex_json: Optional[Dict[str, Any]] = None,
    dex_raises: Optional[Exception] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    state = {
        "gecko_pool_calls": 0, "gecko_ohlcv_calls": 0,
        "birdeye_calls": 0, "dex_calls": 0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)

        if "geckoterminal.com" in url:
            # OHLCV vs pool lookup share the host; the OHLCV path contains /ohlcv/.
            if "/ohlcv/" in url:
                state["gecko_ohlcv_calls"] += 1
                return httpx.Response(
                    200,
                    json=gecko_ohlcv_json
                    if gecko_ohlcv_json is not None
                    else geckoterminal_ohlcv_payload(),
                )
            state["gecko_pool_calls"] += 1
            if gecko_raises is not None:
                raise gecko_raises
            if gecko_status_sequence:
                idx = min(state["gecko_pool_calls"] - 1, len(gecko_status_sequence) - 1)
                status = gecko_status_sequence[idx]
                if status == 429:
                    return httpx.Response(429, headers={"Retry-After": "0"}, json={})
                if status >= 400:  # 5xx transient or 4xx permanent -> both fall through
                    return httpx.Response(status, json={})
            # Default: no pools -> GeckoTerminal yields nothing -> next source.
            return httpx.Response(
                200,
                json=gecko_pools_json if gecko_pools_json is not None else {"data": []},
            )

        if "birdeye.so" in url:
            state["birdeye_calls"] += 1
            # Birdeye must be called WITH the API key header.
            assert request.headers.get("X-API-KEY") == API_KEY
            if birdeye_raises is not None:
                raise birdeye_raises
            if birdeye_status_sequence:
                idx = min(state["birdeye_calls"] - 1, len(birdeye_status_sequence) - 1)
                status = birdeye_status_sequence[idx]
                if status == 429:
                    return httpx.Response(429, headers={"Retry-After": "0"}, json={})
                if status >= 500:
                    return httpx.Response(status, json={})
                if status >= 400:
                    # Verbatim body Birdeye returns for a plan-gated OHLCV call.
                    return httpx.Response(status, json=BIRDEYE_PLAN_ERROR)
            return httpx.Response(
                200,
                json=birdeye_json if birdeye_json is not None else birdeye_payload(),
            )

        if "dexscreener.com" in url:
            state["dex_calls"] += 1
            if dex_raises is not None:
                raise dex_raises
            return httpx.Response(
                200, json=dex_json if dex_json is not None else dexscreener_payload()
            )

        return httpx.Response(404, json={})

    handler.state = state  # type: ignore[attr-defined]
    return handler


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    # Reused retry/backoff path reads rug_check's globals; zero the backoff so
    # retry tests don't actually sleep.
    monkeypatch.setattr(rug_check, "RATE_LIMIT_BACKOFF", 0.0)
    # Disable the GeckoTerminal client-side throttle so tests never real-sleep
    # between its calls (the dedicated throttle test re-enables it locally).
    monkeypatch.setattr(price_history, "GECKO_MIN_CALL_INTERVAL", 0.0)


@pytest.fixture(autouse=True)
def _rearm_birdeye():
    # The Birdeye circuit breaker is process-global; re-arm it around every test
    # so a fatal-status test cannot leak into (and silently skip Birdeye in) the
    # next one.
    price_history.reset_birdeye_circuit()
    yield
    price_history.reset_birdeye_circuit()


def fetch(handler, *, api_key: Optional[str] = API_KEY):
    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await get_price_history(
                "MINT",
                client=client,
                birdeye_api_key=api_key,
                now=NOW,
                interval="5m",
                lookback_hours=24.0,
                min_candles=5,
            )
        finally:
            await client.aclose()

    return asyncio.run(_run())


# --- Tests -------------------------------------------------------------------

def test_birdeye_full_history_enables_entry_evaluation() -> None:
    """Full Birdeye candles -> real ATH/series so evaluate_entry can BUY."""
    history = fetch(make_handler())

    assert history is not None
    assert history.source == "birdeye"
    assert len(history) == 8
    assert history.interval_minutes == 5.0
    # Derived recent ATH = highest candle high (1.0), 5h ago.
    ath_price, ath_ts = history.ath()
    assert ath_price == pytest.approx(1.0)
    assert ath_ts == pytest.approx(NOW - 5 * HOUR)
    assert history.current_price == pytest.approx(0.60)

    # Feed it into the entry logic: it can now judge ATH/pullback -> BUY.
    market = build_entry_market_data(
        current_price=history.current_price,
        current_liquidity=100_000.0,
        market_cap_usd=1_000_000.0,
        history=history,
        now=NOW,
    )
    assert market.price_history == history.price_history
    assert market.volume_history == history.volume_history

    decision = run(evaluate_entry("MINT", market))
    assert decision.action is EntryAction.BUY
    assert decision.proven_runner is True
    assert decision.pullback_pct == pytest.approx(0.40, abs=0.01)
    assert decision.volume_spike_ratio == pytest.approx(10.0, abs=0.1)


def test_birdeye_failure_falls_back_to_dexscreener() -> None:
    """Birdeye transport error -> Dexscreener fallback series is used."""
    handler = make_handler(
        birdeye_raises=httpx.ConnectError("birdeye down"),
        dex_json=dexscreener_payload(),
    )
    history = fetch(handler)

    assert history is not None
    assert history.source == "dexscreener"
    assert len(history) >= 2
    # Birdeye was attempted (and retried) before the fallback.
    assert handler.state["birdeye_calls"] >= 1
    assert handler.state["dex_calls"] == 1
    # The coarse series still descends to the live price, current 0.60.
    assert history.current_price == pytest.approx(0.60)


def test_birdeye_plan_gate_400_falls_back_and_trips_circuit() -> None:
    """The real bug: a plan-gated 400 must fall back, NOT be retried, and must
    disable the Birdeye leg so later tokens skip it entirely."""
    handler = make_handler(
        birdeye_status_sequence=[400], dex_json=dexscreener_payload()
    )
    history = fetch(handler)

    # Fallback intact: the pipeline still gets usable history.
    assert history is not None
    assert history.source == "dexscreener"

    # A 400 is permanent -> exactly one attempt, no retry/backoff storm.
    assert handler.state["birdeye_calls"] == 1

    # Breaker tripped, and it records WHY (Birdeye's own message, not "400").
    reason = price_history.birdeye_status()
    assert reason is not None
    assert "400" in reason
    assert "Compute units usage limit exceeded" in reason


def test_birdeye_circuit_skips_birdeye_on_subsequent_tokens() -> None:
    """Once tripped, later calls must not touch Birdeye at all (no wasted
    round-trip per token per scan) and must still return fallback history."""
    handler = make_handler(
        birdeye_status_sequence=[400], dex_json=dexscreener_payload()
    )
    fetch(handler)
    assert handler.state["birdeye_calls"] == 1

    second = fetch(handler)  # next token, same process

    assert second is not None
    assert second.source == "dexscreener"
    assert handler.state["birdeye_calls"] == 1  # unchanged -> Birdeye skipped
    assert handler.state["dex_calls"] == 2


def test_birdeye_transient_5xx_does_not_trip_circuit() -> None:
    """A 5xx is transient: fall back for this token, but keep Birdeye ARMED."""
    handler = make_handler(
        birdeye_status_sequence=[500], dex_json=dexscreener_payload()
    )
    history = fetch(handler)

    assert history is not None
    assert history.source == "dexscreener"
    assert price_history.birdeye_status() is None  # still armed


def test_insufficient_birdeye_candles_falls_back() -> None:
    """Birdeye returns fewer than min_candles -> Dexscreener fallback used."""
    short = birdeye_payload(proven_runner_items()[:2])  # 2 < min 5
    handler = make_handler(birdeye_json=short, dex_json=dexscreener_payload())
    history = fetch(handler)

    assert history is not None
    assert history.source == "dexscreener"
    assert handler.state["dex_calls"] == 1


def test_no_api_key_skips_birdeye_uses_dexscreener(monkeypatch) -> None:
    """No Birdeye key -> Birdeye never called, Dexscreener fallback used.

    Null out the configured key so the test is hermetic: ``birdeye_api_key=None``
    resolves to ``config.BIRDEYE_API_KEY``, so a real key in a developer's local
    ``.env`` must not leak in and make Birdeye fire.
    """
    monkeypatch.setattr(config, "BIRDEYE_API_KEY", None)
    handler = make_handler(dex_json=dexscreener_payload())
    history = fetch(handler, api_key=None)

    assert history is not None
    assert history.source == "dexscreener"
    assert handler.state["birdeye_calls"] == 0  # skipped entirely
    assert handler.state["dex_calls"] == 1


def test_both_sources_fail_returns_none_and_entry_skips() -> None:
    """Both providers fail -> None -> empty history -> evaluate_entry SKIPs."""
    handler = make_handler(
        birdeye_raises=httpx.ConnectError("birdeye down"),
        dex_raises=httpx.ConnectError("dexscreener down"),
    )
    history = fetch(handler)
    assert history is None

    # No history -> empty-series snapshot -> entry fails closed to SKIP.
    market = build_entry_market_data(
        current_price=0.60,
        current_liquidity=100_000.0,
        market_cap_usd=1_000_000.0,
        history=None,
    )
    assert market.price_history == [] and market.volume_history == []
    assert isinstance(market, EntryMarketData)

    decision = run(evaluate_entry("MINT", market))
    assert decision.action is EntryAction.SKIP
    assert "history" in decision.reason.lower()


def test_both_sources_empty_returns_none() -> None:
    """Birdeye empty items + Dexscreener no pairs -> None (fail-closed)."""
    handler = make_handler(
        birdeye_json={"success": True, "data": {"items": []}},
        dex_json={"pairs": []},
    )
    history = fetch(handler)
    assert history is None
    assert handler.state["birdeye_calls"] >= 1
    assert handler.state["dex_calls"] == 1


def test_birdeye_rate_limit_retries_then_succeeds() -> None:
    """A 429 is retried (reused backoff path) and the retry succeeds on Birdeye."""
    handler = make_handler(birdeye_status_sequence=[429, 200])
    history = fetch(handler)

    assert history is not None
    assert history.source == "birdeye"
    assert handler.state["birdeye_calls"] == 2  # one 429 + one success


# --- GeckoTerminal (primary source) ------------------------------------------

def test_geckoterminal_full_history_is_primary_and_enables_entry() -> None:
    """GeckoTerminal real candles are used FIRST -> real ATH/series -> BUY, and
    neither Birdeye nor Dexscreener is touched."""
    handler = make_handler(
        gecko_pools_json=geckoterminal_pools_payload(),
        gecko_ohlcv_json=geckoterminal_ohlcv_payload(),
    )
    history = fetch(handler)

    assert history is not None
    assert history.source == "geckoterminal"
    assert len(history) == 8
    assert history.interval_minutes == 5.0
    # Derived recent ATH = highest candle high (1.0), 5h ago; live price 0.60.
    ath_price, ath_ts = history.ath()
    assert ath_price == pytest.approx(1.0)
    assert ath_ts == pytest.approx(NOW - 5 * HOUR)
    assert history.current_price == pytest.approx(0.60)

    # Primary succeeded -> exactly two Gecko calls (pool + OHLCV), no fallthrough.
    assert handler.state["gecko_pool_calls"] == 1
    assert handler.state["gecko_ohlcv_calls"] == 1
    assert handler.state["birdeye_calls"] == 0
    assert handler.state["dex_calls"] == 0

    # Feed it into the entry logic: real candles -> it can judge ATH/pullback -> BUY.
    market = build_entry_market_data(
        current_price=history.current_price,
        current_liquidity=100_000.0,
        market_cap_usd=1_000_000.0,
        history=history,
        now=NOW,
    )
    decision = run(evaluate_entry("MINT", market))
    assert decision.action is EntryAction.BUY
    assert decision.proven_runner is True
    assert decision.pullback_pct == pytest.approx(0.40, abs=0.01)
    assert decision.volume_spike_ratio == pytest.approx(10.0, abs=0.1)


def test_geckoterminal_picks_deepest_liquidity_pool() -> None:
    """When a token has several pools, the deepest by reserve_in_usd is queried."""
    pools = {
        "data": [
            {"attributes": {"address": "SHALLOW", "reserve_in_usd": "1000"}},
            {"attributes": {"address": "DEEP", "reserve_in_usd": "500000"}},
            {"attributes": {"address": "MID", "reserve_in_usd": "50000"}},
        ]
    }
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geckoterminal.com" in url and "/ohlcv/" in url:
            seen.append(url)
            return httpx.Response(200, json=geckoterminal_ohlcv_payload())
        if "geckoterminal.com" in url:
            return httpx.Response(200, json=pools)
        return httpx.Response(404, json={})

    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await get_price_history(
                "MINT", client=client, birdeye_api_key=API_KEY, now=NOW,
                interval="5m", lookback_hours=24.0, min_candles=5,
            )
        finally:
            await client.aclose()

    history = asyncio.run(_run())
    assert history is not None and history.source == "geckoterminal"
    assert len(seen) == 1
    assert "/pools/DEEP/ohlcv/" in seen[0]  # deepest pool, not SHALLOW/MID


def test_geckoterminal_rate_limited_backs_off_and_falls_through() -> None:
    """A 429 that survives the reused backoff falls through the chain to Birdeye
    (still real candles) — the bot is never blocked by the free-tier limit."""
    handler = make_handler(
        gecko_pools_json=geckoterminal_pools_payload(),
        gecko_status_sequence=[429],  # every attempt 429 -> retried, then exhausts
        birdeye_json=birdeye_payload(),
    )
    history = fetch(handler)

    assert history is not None
    assert history.source == "birdeye"
    # It backed off and RETRIED before giving up (more than one pool attempt).
    assert handler.state["gecko_pool_calls"] > 1
    # Then fell through to the next real-candle source.
    assert handler.state["birdeye_calls"] >= 1


def test_geckoterminal_failure_falls_through_to_birdeye() -> None:
    """A GeckoTerminal transport error falls through to Birdeye."""
    handler = make_handler(
        gecko_raises=httpx.ConnectError("gecko down"),
        birdeye_json=birdeye_payload(),
    )
    history = fetch(handler)

    assert history is not None
    assert history.source == "birdeye"
    assert handler.state["gecko_pool_calls"] >= 1
    assert handler.state["birdeye_calls"] >= 1


def test_geckoterminal_no_pool_falls_through_to_dexscreener() -> None:
    """No Gecko pool + Birdeye plan-gated -> coarse Dexscreener last resort."""
    handler = make_handler(
        gecko_pools_json={"data": []},        # token has no pool
        birdeye_status_sequence=[400],        # plan-gated, trips its breaker
        dex_json=dexscreener_payload(),
    )
    history = fetch(handler)

    assert history is not None
    assert history.source == "dexscreener"
    assert handler.state["gecko_pool_calls"] == 1
    assert handler.state["gecko_ohlcv_calls"] == 0  # no pool -> OHLCV never fetched


def test_gecko_throttle_spaces_out_calls(monkeypatch) -> None:
    """The client-side throttle blocks a second call until the min interval
    elapses, so the bot proactively stays under the free-tier rate limit."""
    monkeypatch.setattr(price_history, "GECKO_MIN_CALL_INTERVAL", 0.05)
    price_history.reset_gecko_throttle()

    async def _run() -> float:
        start = time.monotonic()
        await price_history._gecko_throttle()  # first: immediate
        await price_history._gecko_throttle()  # second: waits ~one interval
        return time.monotonic() - start

    elapsed = asyncio.run(_run())
    assert elapsed >= 0.05
