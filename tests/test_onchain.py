"""Tests for the read-only Helius on-chain signal layer (data.onchain) and its
wiring into the live management feed.

All hermetic (httpx.MockTransport, no network). Covers: coordinated-dump signals
(largest single sell + distinct selling wallets), buyer breadth, KOL holdings,
fail-closed behaviour on Helius errors / no endpoint, the live feed populating
the dump fields, a Helius failure degrading to benign zeros (no crash, no false
pass), and a detected dump firing the hard-exit coordinated_dump trigger.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

import httpx
import pytest

from data.market_feed import ROLLING_WINDOW, get_live_market_snapshot
from data.onchain import OnchainFlow, get_kol_holdings, get_recent_flow
from safety import rug_check
from strategies.hard_exit import evaluate_hard_exit
from strategies.profit_taking import Position

MINT = "MINT"
POOL = "POOLVAULT"
POOL2 = "POOLVAULT2"
RPC = "https://mock.helius.invalid/"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    monkeypatch.setattr(rug_check, "RATE_LIMIT_BACKOFF", 0.0)


# --- JSON-RPC mock transport -------------------------------------------------

def swap_tx(entries: List[tuple]) -> Dict[str, Any]:
    """Build a jsonParsed transaction from ``(account_addr, owner, pre, post)``
    token-balance rows for :data:`MINT`."""
    keys = [{"pubkey": addr} for (addr, _o, _pre, _post) in entries]
    pre = [
        {"accountIndex": i, "mint": MINT, "owner": o, "uiTokenAmount": {"uiAmount": pre}}
        for i, (_a, o, pre, _post) in enumerate(entries)
    ]
    post = [
        {"accountIndex": i, "mint": MINT, "owner": o, "uiTokenAmount": {"uiAmount": post}}
        for i, (_a, o, _pre, post) in enumerate(entries)
    ]
    return {
        "transaction": {"message": {"accountKeys": keys}},
        "meta": {"err": None, "preTokenBalances": pre, "postTokenBalances": post},
    }


def rpc_handler(
    *,
    largest_value: Optional[List[Dict[str, Any]]] = None,
    signatures: Optional[List[Dict[str, Any]]] = None,
    txs: Optional[Dict[str, Any]] = None,
    token_accounts: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    error_method: Optional[str] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    calls: Dict[str, Any] = {"count": 0, "methods": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body["method"]
        params = body.get("params") or []
        calls["count"] += 1
        calls["methods"].append(method)

        if error_method and method == error_method:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1,
                                             "error": {"code": -32000, "message": "boom"}})
        if method == "getTokenLargestAccounts":
            return httpx.Response(200, json={"result": {"value": largest_value or []}})
        if method == "getSignaturesForAddress":
            return httpx.Response(200, json={"result": signatures or []})
        if method == "getTransaction":
            return httpx.Response(200, json={"result": (txs or {}).get(params[0])})
        if method == "getTokenAccountsByOwner":
            owner = params[0]
            return httpx.Response(
                200, json={"result": {"value": (token_accounts or {}).get(owner, [])}}
            )
        return httpx.Response(404, json={})

    handler.calls = calls  # type: ignore[attr-defined]
    return handler


def onchain_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- get_recent_flow ---------------------------------------------------------

DUMP_SIGS = [{"signature": f"s{i}", "err": None} for i in (1, 2, 3, 4)]
DUMP_TXS = {
    # walletA sells 60, walletB sells 300 (largest), walletC sells 50; pool +.
    "s1": swap_tx([("A_ACCT", "walletA", 100, 40), (POOL, "poolOwner", 1000, 1060)]),
    "s2": swap_tx([("B_ACCT", "walletB", 500, 200), (POOL, "poolOwner", 1060, 1360)]),
    "s3": swap_tx([("C_ACCT", "walletC", 80, 30), (POOL, "poolOwner", 1360, 1410)]),
    # walletD buys 80; pool -.
    "s4": swap_tx([("D_ACCT", "walletD", 0, 80), (POOL, "poolOwner", 1410, 1330)]),
}
LARGEST = [{"address": POOL, "uiAmount": 1e9}, {"address": POOL2, "uiAmount": 1e6}]


def test_recent_flow_detects_coordinated_dump() -> None:
    """Three distinct wallets net-selling -> selling_wallets=3, largest sell USD."""
    handler = rpc_handler(largest_value=LARGEST, signatures=DUMP_SIGS, txs=DUMP_TXS)

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_recent_flow(MINT, price=2.0, client=client, rpc_url=RPC)
        finally:
            await client.aclose()

    flow = run(_run())
    assert flow.available is True
    assert flow.selling_wallets == 3           # A, B, C
    assert flow.buyer_wallets == 1             # D
    assert flow.largest_single_sell_usd == pytest.approx(600.0)  # B: 300 * $2
    assert flow.scanned_transactions == 4


def test_recent_flow_counts_buyer_breadth() -> None:
    """Distinct net-buyers (pool sheds tokens) count as buyer breadth, not sells."""
    sigs = [{"signature": f"b{i}", "err": None} for i in (1, 2, 3)]
    txs = {
        "b1": swap_tx([("W1", "buyer1", 0, 50), (POOL, "p", 1000, 950)]),
        "b2": swap_tx([("W2", "buyer2", 10, 90), (POOL, "p", 950, 870)]),
        "b3": swap_tx([("W3", "buyer3", 0, 20), (POOL, "p", 870, 850)]),
    }
    handler = rpc_handler(largest_value=LARGEST, signatures=sigs, txs=txs)

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_recent_flow(MINT, price=1.0, client=client, rpc_url=RPC)
        finally:
            await client.aclose()

    flow = run(_run())
    assert flow.available is True
    assert flow.buyer_wallets == 3
    assert flow.selling_wallets == 0
    assert flow.largest_single_sell_usd == 0.0


def test_recent_flow_failclosed_on_rpc_error() -> None:
    """A Helius RPC error -> signal absent (zeros), never raises."""
    handler = rpc_handler(
        largest_value=LARGEST, signatures=DUMP_SIGS, txs=DUMP_TXS,
        error_method="getSignaturesForAddress",
    )

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_recent_flow(MINT, price=2.0, client=client, rpc_url=RPC)
        finally:
            await client.aclose()

    flow = run(_run())
    assert flow.available is False
    assert flow.largest_single_sell_usd == 0.0
    assert flow.selling_wallets == 0
    assert flow.buyer_wallets == 0


def test_recent_flow_no_endpoint_makes_no_call() -> None:
    """No configured RPC endpoint -> absent signal, and no request is made."""
    handler = rpc_handler(largest_value=LARGEST, signatures=DUMP_SIGS, txs=DUMP_TXS)

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_recent_flow(MINT, price=2.0, client=client, rpc_url="")
        finally:
            await client.aclose()

    flow = run(_run())
    assert flow.available is False
    assert handler.calls["count"] == 0


def test_recent_flow_nonpositive_price_absent() -> None:
    """A non-positive price can't value sells -> absent, no calls."""
    handler = rpc_handler(largest_value=LARGEST, signatures=DUMP_SIGS, txs=DUMP_TXS)

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_recent_flow(MINT, price=0.0, client=client, rpc_url=RPC)
        finally:
            await client.aclose()

    flow = run(_run())
    assert flow.available is False
    assert handler.calls["count"] == 0


# --- get_kol_holdings --------------------------------------------------------

def _token_account(ui_amount: float) -> Dict[str, Any]:
    return {"account": {"data": {"parsed": {"info": {
        "tokenAmount": {"uiAmount": ui_amount}}}}}}


def test_kol_holdings_hit() -> None:
    """A tracked wallet holding the token -> any_hold True with the wallet listed."""
    handler = rpc_handler(token_accounts={
        "KOL_A": [_token_account(1234.0)],
        "KOL_B": [],  # holds nothing
    })

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_kol_holdings(
                MINT, ["KOL_A", "KOL_B"], client=client, rpc_url=RPC
            )
        finally:
            await client.aclose()

    kol = run(_run())
    assert kol.available is True
    assert kol.any_hold is True
    assert kol.holding_wallets == ("KOL_A",)


def test_kol_holdings_miss() -> None:
    """No tracked wallet holds the token -> available, but no hit (never a block)."""
    handler = rpc_handler(token_accounts={"KOL_A": [], "KOL_B": [_token_account(0.0)]})

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_kol_holdings(
                MINT, ["KOL_A", "KOL_B"], client=client, rpc_url=RPC
            )
        finally:
            await client.aclose()

    kol = run(_run())
    assert kol.available is True
    assert kol.any_hold is False
    assert kol.holding_wallets == ()


def test_kol_holdings_empty_list_is_available_no_hit() -> None:
    """An empty curated list is a definitive no-hit with no RPC call."""
    handler = rpc_handler()

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_kol_holdings(MINT, [], client=client, rpc_url=RPC)
        finally:
            await client.aclose()

    kol = run(_run())
    assert kol.available is True
    assert kol.any_hold is False
    assert handler.calls["count"] == 0


def test_kol_holdings_failclosed_on_error() -> None:
    """A Helius error -> signal absent (available False), never raises."""
    handler = rpc_handler(error_method="getTokenAccountsByOwner")

    async def _run():
        client = onchain_client(handler)
        try:
            return await get_kol_holdings(MINT, ["KOL_A"], client=client, rpc_url=RPC)
        finally:
            await client.aclose()

    kol = run(_run())
    assert kol.available is False
    assert kol.any_hold is False


# --- Live feed wiring (Birdeye + Dexscreener + Helius flow) ------------------

NOW = 1_700_000_000.0


def _candle(seconds_ago: float, close: float, vol: float,
            *, open_: Optional[float] = None) -> Dict[str, Any]:
    o = close if open_ is None else open_
    return {"unixTime": int(NOW - seconds_ago), "o": o, "h": max(o, close),
            "l": min(o, close), "c": close, "v": vol}


def _feed_handler():
    items = [_candle((18 - i) * 60.0, close=1.0, vol=5.0) for i in range(18)]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "birdeye.so" in url:
            return httpx.Response(200, json={"success": True, "data": {"items": items}})
        if "dexscreener.com" in url:
            return httpx.Response(200, json={"pairs": [
                {"chainId": "solana", "priceUsd": "1.0", "liquidity": {"usd": 50_000.0}}]})
        return httpx.Response(404, json={})

    return handler


def _live_snapshot(flow_fetcher):
    async def _run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(_feed_handler()))
        try:
            return await get_live_market_snapshot(
                MINT, 60_000.0, client=client, birdeye_api_key="k",
                flow_fetcher=flow_fetcher, now=NOW,
            )
        finally:
            await client.aclose()

    return run(_run())


def test_live_feed_populates_dump_signals() -> None:
    """The feed carries the Helius dump signals into the snapshot."""
    async def flow(token, *, price, client, timeout, rpc_url):
        return OnchainFlow(
            largest_single_sell_usd=9_000.0, selling_wallets=4,
            buyer_wallets=2, scanned_transactions=12, available=True,
        )

    snap = _live_snapshot(flow)
    assert snap is not None
    assert snap.largest_single_sell == pytest.approx(9_000.0)
    assert snap.top_wallet_sells == 4
    # Core market fields are still assembled normally.
    assert snap.current_liquidity == pytest.approx(50_000.0)


def test_live_feed_flow_failure_is_benign() -> None:
    """A raising flow fetcher degrades to benign zeros — snapshot still valid."""
    async def boom(token, *, price, client, timeout, rpc_url):
        raise RuntimeError("helius 503")

    snap = _live_snapshot(boom)
    assert snap is not None                      # NOT fail-closed to None
    assert snap.largest_single_sell == 0.0       # dump signal simply absent
    assert snap.top_wallet_sells == 0
    assert snap.current_price == pytest.approx(1.0)  # rest of the snapshot intact


def test_detected_dump_fires_hard_exit() -> None:
    """End-to-end: a live snapshot carrying a dump fires coordinated_dump."""
    async def flow(token, *, price, client, timeout, rpc_url):
        # One sell worth 9000 USD vs 50k pool liquidity = 18% > 5% single-sell cap.
        return OnchainFlow(
            largest_single_sell_usd=9_000.0, selling_wallets=4,
            buyer_wallets=0, scanned_transactions=8, available=True,
        )

    snap = _live_snapshot(flow)
    assert snap is not None

    position = Position(entry_price=1.0, original_size=1000.0)
    decision = run(evaluate_hard_exit(position, snap))
    assert decision.should_exit is True
    assert decision.trigger == "coordinated_dump"
    assert decision.tokens_to_sell == 1000.0     # full remaining, moonbag included
