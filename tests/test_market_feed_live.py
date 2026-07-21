"""Tests for the live management-loop feed in data.market_feed.

Uses httpx.MockTransport to simulate Dexscreener (liquidity) and Birdeye (1m
OHLCV) with no network. Covers: a full snapshot assembled from both sources,
peak-vs-rolling 15m volume derivation, and fail-closed behaviour on missing
liquidity, too few candles, or no Birdeye key.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

import config
from data.market_feed import (
    FINE_1M_OHLCV_SOURCES,
    ROLLING_WINDOW,
    get_live_market_snapshot,
)
from data.price_history import Candle, OHLCVHistory
from safety import rug_check

NOW = 1_700_000_000.0
API_KEY = "test-birdeye-key"


def run(coro):
    return asyncio.run(coro)


def candle(seconds_ago: float, close: float, vol: float,
           *, open_: Optional[float] = None) -> Dict[str, Any]:
    o = close if open_ is None else open_
    return {"unixTime": int(NOW - seconds_ago), "o": o, "h": max(o, close),
            "l": min(o, close), "c": close, "v": vol}


def birdeye_payload(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"success": True, "data": {"items": items}}


def dex_payload(liquidity: Optional[float] = 50_000.0,
                chain: str = "solana") -> Dict[str, Any]:
    pair: Dict[str, Any] = {"chainId": chain, "priceUsd": "1.20"}
    if liquidity is not None:
        pair["liquidity"] = {"usd": liquidity}
    return {"pairs": [pair]}


def make_handler(*, birdeye_json: Optional[Dict[str, Any]] = None,
                 dex_json: Optional[Dict[str, Any]] = None
                 ) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "birdeye.so" in url:
            return httpx.Response(200, json=birdeye_json or birdeye_payload([]))
        if "dexscreener.com" in url:
            return httpx.Response(200, json=dex_json or dex_payload())
        return httpx.Response(404, json={})
    return handler


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(rug_check, "RATE_LIMIT_BACKOFF", 0.0)


def fetch(handler, *, api_key: Optional[str] = API_KEY, entry_liq: float = 60_000.0):
    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await get_live_market_snapshot(
                "MINT", entry_liq, client=client, birdeye_api_key=api_key, now=NOW)
        finally:
            await client.aclose()
    return asyncio.run(_run())


# --- Tests ------------------------------------------------------------------

def test_full_live_snapshot_from_birdeye_and_dexscreener() -> None:
    # 18 flat-volume candles; last candle open 1.0 -> close 1.2.
    items = [candle((18 - i) * 60.0, close=1.0, vol=5.0) for i in range(17)]
    items.append(candle(0.0, close=1.2, vol=5.0, open_=1.0))
    snap = fetch(make_handler(birdeye_json=birdeye_payload(items),
                              dex_json=dex_payload(50_000.0)))

    assert snap is not None
    assert snap.current_price == pytest.approx(1.2)        # latest Birdeye close
    assert snap.current_liquidity == pytest.approx(50_000.0)  # Dexscreener
    assert snap.entry_liquidity == pytest.approx(60_000.0)    # passed through
    assert snap.rolling_volume_15m == pytest.approx(ROLLING_WINDOW * 5.0)  # 75
    assert snap.peak_volume_15m == pytest.approx(ROLLING_WINDOW * 5.0)
    assert snap.candle_1m_open == pytest.approx(1.0)
    assert snap.candle_1m_close == pytest.approx(1.2)
    # Trade-level signals stay inactive (Helius-only) -> benign defaults.
    assert snap.largest_single_sell == 0.0
    assert snap.top_wallet_sells == 0


def test_peak_volume_exceeds_rolling_when_early_spike() -> None:
    # First 15 candles heavy (vol 10), last 5 light (vol 1): early window peaks.
    items = ([candle((20 - i) * 60.0, close=1.0, vol=10.0) for i in range(15)]
             + [candle((5 - i) * 60.0, close=1.0, vol=1.0) for i in range(5)])
    snap = fetch(make_handler(birdeye_json=birdeye_payload(items)))

    assert snap is not None
    # trailing 15 = candles[5:20] = ten 10s + five 1s = 105
    assert snap.rolling_volume_15m == pytest.approx(105.0)
    # peak 15m window = the first 15 candles = 150
    assert snap.peak_volume_15m == pytest.approx(150.0)
    assert snap.peak_volume_15m > snap.rolling_volume_15m


def test_missing_liquidity_fails_closed() -> None:
    items = [candle((18 - i) * 60.0, close=1.0, vol=5.0) for i in range(18)]
    snap = fetch(make_handler(birdeye_json=birdeye_payload(items),
                              dex_json=dex_payload(liquidity=None)))
    assert snap is None  # no usable Dexscreener liquidity -> fail-closed


def test_insufficient_candles_fails_closed() -> None:
    items = [candle((5 - i) * 60.0, close=1.0, vol=5.0) for i in range(5)]  # < 15
    snap = fetch(make_handler(birdeye_json=birdeye_payload(items)))
    assert snap is None


def test_no_birdeye_key_fails_closed(monkeypatch) -> None:
    # Without a Birdeye key there are no real 1m candles; liquidity alone is
    # not a tradeable snapshot, so the feed fails closed. Null out the configured
    # key so a real key in a developer's local .env can't leak in (api_key=None
    # otherwise resolves to config.BIRDEYE_API_KEY).
    monkeypatch.setattr(config, "BIRDEYE_API_KEY", None)
    items = [candle((18 - i) * 60.0, close=1.0, vol=5.0) for i in range(18)]
    snap = fetch(make_handler(birdeye_json=birdeye_payload(items)), api_key=None)
    assert snap is None


# --- 1m OHLCV source acceptance ---------------------------------------------
# Regression: the live feed must accept ANY true 1m source (GeckoTerminal is the
# current primary), not just Birdeye. A stale `source == "birdeye"` guard threw
# away real GeckoTerminal candles and fail-closed every snapshot, so an entered
# position could never be managed.

def _history(source: str, *, n: int = 18) -> OHLCVHistory:
    """A valid n-candle 1m history from ``source``; last candle 1.0 -> 1.2."""
    candles = [
        Candle(
            timestamp=NOW - (n - i) * 60.0,
            open=1.0, high=1.2, low=1.0,
            close=1.0 if i < n - 1 else 1.2,
            volume=5.0,
        )
        for i in range(n)
    ]
    return OHLCVHistory("MINT", 1.0, source, candles)


def _fetch_with_history(history: Optional[OHLCVHistory]):
    """Drive get_live_market_snapshot with an injected history source.

    Dexscreener liquidity is still mocked (the feed always needs it); only the
    OHLCV history is swapped so we test the source-acceptance guard directly.
    """
    async def fake_fetch(_token: str, **_kwargs: Any) -> Optional[OHLCVHistory]:
        return history

    async def _run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(make_handler(dex_json=dex_payload()))
        )
        try:
            return await get_live_market_snapshot(
                "MINT", 60_000.0, client=client,
                history_fetcher=fake_fetch, now=NOW,
            )
        finally:
            await client.aclose()

    return asyncio.run(_run())


def test_geckoterminal_1m_history_is_accepted() -> None:
    snap = _fetch_with_history(_history("geckoterminal"))
    assert snap is not None                       # previously fail-closed
    assert snap.current_price == pytest.approx(1.2)
    assert snap.candle_1m_open == pytest.approx(1.0)
    assert snap.candle_1m_close == pytest.approx(1.2)


def test_coarse_dexscreener_history_fails_closed() -> None:
    # The coarse Dexscreener OHLCV fallback can't yield a real 1m candle, so a
    # snapshot must NOT be built from it (it would disarm flash-crash/vol checks).
    assert _fetch_with_history(_history("dexscreener")) is None


def test_fine_1m_sources_membership() -> None:
    assert "geckoterminal" in FINE_1M_OHLCV_SOURCES
    assert "birdeye" in FINE_1M_OHLCV_SOURCES
    assert "dexscreener" not in FINE_1M_OHLCV_SOURCES
