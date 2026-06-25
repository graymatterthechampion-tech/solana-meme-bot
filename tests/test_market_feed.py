"""Tests for the async market-data feed.

Covers the fail-closed contract: a good payload yields a populated
MarketData; any timeout, HTTP error, or missing field yields None. The async
httpx client is replaced with a lightweight fake so no network is touched.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx
import pytest

from data import market_feed
from data.market_feed import get_market_snapshot, mock_market_snapshot

ENTRY_LIQUIDITY = 100_000.0


def good_payload() -> Dict[str, Any]:
    return {
        "price": 0.002,
        "volume_15m": 40_000.0,
        "peak_volume_15m": 50_000.0,
        "liquidity": 90_000.0,
        "largest_single_sell": 100.0,
        "top_wallet_sells": 1,
        "candle_1m_open": 0.002,
        "candle_1m_close": 0.0019,
    }


class _FakeResponse:
    def __init__(self, json_data: Dict[str, Any]) -> None:
        self._json = json_data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return self._json


class _FakeClient:
    """Stands in for httpx.AsyncClient. Either returns a response or raises."""

    def __init__(
        self,
        response: Optional[_FakeResponse] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._exc = exc
        self.closed = False

    async def get(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response

    async def aclose(self) -> None:
        self.closed = True


def run(coro):
    return asyncio.run(coro)


def test_good_snapshot_is_populated() -> None:
    client = _FakeClient(response=_FakeResponse(good_payload()))
    snap = run(get_market_snapshot("MINT", ENTRY_LIQUIDITY, client=client))

    assert snap is not None
    assert snap.current_price == pytest.approx(0.002)
    assert snap.rolling_volume_15m == pytest.approx(40_000.0)
    assert snap.peak_volume_15m == pytest.approx(50_000.0)
    assert snap.current_liquidity == pytest.approx(90_000.0)
    # entry_liquidity is injected by the caller, not the feed.
    assert snap.entry_liquidity == pytest.approx(ENTRY_LIQUIDITY)
    assert snap.candle_1m_close == pytest.approx(0.0019)


def test_timeout_fails_closed_to_none() -> None:
    client = _FakeClient(exc=httpx.TimeoutException("timed out"))
    snap = run(get_market_snapshot("MINT", ENTRY_LIQUIDITY, client=client))
    assert snap is None  # no-action signal, never a partial/stale snapshot


def test_http_error_fails_closed_to_none() -> None:
    client = _FakeClient(exc=httpx.ConnectError("refused"))
    snap = run(get_market_snapshot("MINT", ENTRY_LIQUIDITY, client=client))
    assert snap is None


def test_missing_field_fails_closed_to_none() -> None:
    payload = good_payload()
    del payload["liquidity"]  # drop a required field
    client = _FakeClient(response=_FakeResponse(payload))
    snap = run(get_market_snapshot("MINT", ENTRY_LIQUIDITY, client=client))
    assert snap is None


def test_non_positive_price_fails_closed_to_none() -> None:
    payload = good_payload()
    payload["price"] = 0.0  # bad data, not a tradeable signal
    client = _FakeClient(response=_FakeResponse(payload))
    snap = run(get_market_snapshot("MINT", ENTRY_LIQUIDITY, client=client))
    assert snap is None


def test_injected_client_not_closed_by_feed() -> None:
    """A caller-supplied client must NOT be closed by the feed."""
    client = _FakeClient(response=_FakeResponse(good_payload()))
    run(get_market_snapshot("MINT", ENTRY_LIQUIDITY, client=client))
    assert client.closed is False


def test_mock_snapshot_runs_without_network() -> None:
    snap = run(mock_market_snapshot("MINT", ENTRY_LIQUIDITY))
    assert snap is not None
    assert snap.current_price > 0
    assert snap.entry_liquidity == pytest.approx(ENTRY_LIQUIDITY)
