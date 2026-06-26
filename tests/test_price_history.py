"""Tests for the read-only OHLCV history layer (data.price_history).

Uses httpx.MockTransport to simulate Birdeye and Dexscreener (no network).
Covers: a full Birdeye history that lets evaluate_entry judge ATH / pullback;
Birdeye failure falling back to Dexscreener; both sources failing -> None ->
entry fails closed to SKIP; a too-short Birdeye result triggering fallback;
no API key skipping Birdeye; and the reused 429 retry/backoff path.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from data.price_history import (
    build_entry_market_data,
    get_price_history,
)
from safety import rug_check
from strategies.entry import EntryAction, EntryMarketData, evaluate_entry

NOW = 1_700_000_000.0
HOUR = 3600.0
API_KEY = "test-birdeye-key"


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


# --- Mock transport ----------------------------------------------------------

def make_handler(
    *,
    birdeye_json: Optional[Dict[str, Any]] = None,
    birdeye_raises: Optional[Exception] = None,
    birdeye_status_sequence: Optional[List[int]] = None,
    dex_json: Optional[Dict[str, Any]] = None,
    dex_raises: Optional[Exception] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    state = {"birdeye_calls": 0, "dex_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)

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


def test_insufficient_birdeye_candles_falls_back() -> None:
    """Birdeye returns fewer than min_candles -> Dexscreener fallback used."""
    short = birdeye_payload(proven_runner_items()[:2])  # 2 < min 5
    handler = make_handler(birdeye_json=short, dex_json=dexscreener_payload())
    history = fetch(handler)

    assert history is not None
    assert history.source == "dexscreener"
    assert handler.state["dex_calls"] == 1


def test_no_api_key_skips_birdeye_uses_dexscreener() -> None:
    """No Birdeye key -> Birdeye never called, Dexscreener fallback used."""
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
