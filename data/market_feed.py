"""Async market-data feed.

Produces validated :class:`MarketData` snapshots that drive
``main.process_position``. The public interface is a single async function,
:func:`get_market_snapshot`, that returns ``Optional[MarketData]``.

FAIL-CLOSED CONTRACT
--------------------
On ANY error, timeout, or missing/invalid field the feed returns ``None`` — a
safe "no-action" signal — rather than stale or partial data. The loop must
treat ``None`` as "skip this iteration, do not trade": never evaluate exits or
profit-taking on incomplete data. A partial snapshot is more dangerous than no
snapshot, so we never synthesise missing fields.

CREDENTIALS
-----------
API keys / RPC URLs are read from ``.env`` via ``config`` and never hardcoded.

The live feed (:func:`get_live_market_snapshot`) is wired to real providers:
    * Dexscreener — current pool liquidity (USD).
    * Birdeye 1m OHLCV (via :mod:`data.price_history`) — price / 1m candle /
      rolling + peak 15m volume.
    * Helius RPC (via :mod:`data.onchain`, ``config.RPC_URL`` /
      ``config.HELIUS_API_KEY``) — read-only coordinated-dump signals (largest
      single sell, distinct selling wallets). Fail-closed and non-fatal.

The legacy ``_fetch_raw`` / :func:`get_market_snapshot` remain a placeholder
single-endpoint path (tests inject a mock client).

TODO(execution-realism): dry-run execution currently logs intended sells with
zero market friction. Before any live trading, model realistic SLIPPAGE,
PRIORITY/SWAP FEES, and FILL DELAY so dry-run PnL reflects reality and the
slippage caps in CLAUDE.md are actually exercised.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

import config
from data.onchain import OnchainFlow, get_recent_flow
from data.price_history import OHLCVHistory, get_price_history
from strategies.hard_exit import MarketData

logger = logging.getLogger(__name__)

# Default per-request timeout (seconds). A slow feed must fail-closed, not hang
# the trading loop.
DEFAULT_TIMEOUT: float = 5.0

# Placeholder base URL until real providers are wired. Real runs against this
# will simply fail-closed (return None); tests inject a mock client.
_PLACEHOLDER_BASE: str = "https://example.invalid/markets"

# --- Live feed (read-only) --------------------------------------------------
# Liquidity comes from Dexscreener; price + 1m candles + rolling volume come
# from Birdeye OHLCV (reused via data.price_history).
DEXSCREENER_TOKENS_URL: str = "https://api.dexscreener.com/latest/dex/tokens"
LIVE_INTERVAL: str = "1m"          # candle width for the management loop
LIVE_LOOKBACK_HOURS: float = 2.0   # enough 1m candles to find a 15m volume peak
ROLLING_WINDOW: int = 15           # candles per rolling window (15 x 1m = 15m)

# Async OHLCV source: mint -> recent history (injectable for tests).
HistoryFetcher = Callable[..., Awaitable[Optional[OHLCVHistory]]]

# Async read-only on-chain flow source (injectable for tests). Fail-closed: it
# returns an OnchainFlow (available=False, zeros) rather than raising.
FlowFetcher = Callable[..., Awaitable[OnchainFlow]]

# Every field the loop needs. If ANY is absent the snapshot is rejected.
REQUIRED_RAW_FIELDS: List[str] = [
    "price",
    "volume_15m",
    "peak_volume_15m",
    "liquidity",
    "largest_single_sell",
    "top_wallet_sells",
    "candle_1m_open",
    "candle_1m_close",
]


def _build_snapshot(raw: Dict[str, Any], entry_liquidity: float) -> MarketData:
    """Validate a raw payload into a :class:`MarketData`.

    Raises ``ValueError``/``TypeError`` if any required field is missing,
    null, non-numeric, or nonsensical (e.g. price <= 0). Callers treat any
    such failure as fail-closed (return ``None``).
    """
    missing = [k for k in REQUIRED_RAW_FIELDS if raw.get(k) is None]
    if missing:
        raise ValueError(f"missing/null fields: {missing}")

    snapshot = MarketData(
        current_price=float(raw["price"]),
        rolling_volume_15m=float(raw["volume_15m"]),
        peak_volume_15m=float(raw["peak_volume_15m"]),
        current_liquidity=float(raw["liquidity"]),
        entry_liquidity=float(entry_liquidity),
        largest_single_sell=float(raw["largest_single_sell"]),
        top_wallet_sells=int(raw["top_wallet_sells"]),
        candle_1m_open=float(raw["candle_1m_open"]),
        candle_1m_close=float(raw["candle_1m_close"]),
    )

    # Sanity floors: a non-positive price is bad data, not a tradeable signal.
    if snapshot.current_price <= 0:
        raise ValueError(f"non-positive price: {snapshot.current_price!r}")
    if snapshot.current_liquidity < 0 or snapshot.peak_volume_15m < 0:
        raise ValueError("negative liquidity/volume in payload")

    return snapshot


async def _fetch_raw(
    token_address: str,
    client: httpx.AsyncClient,
    *,
    timeout: float,
) -> Dict[str, Any]:
    """Fetch the raw market payload for a token.

    TODO(real-apis): this is a placeholder. Replace with real Dexscreener /
    Birdeye / Helius calls (see module docstring) and merge their fields.
    """
    headers: Dict[str, str] = {}
    if config.BIRDEYE_API_KEY:
        headers["X-API-KEY"] = config.BIRDEYE_API_KEY

    url = f"{_PLACEHOLDER_BASE}/tokens/{token_address}"
    resp = await client.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()
    return data


async def get_market_snapshot(
    token_address: str,
    entry_liquidity: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[MarketData]:
    """Return a validated market snapshot, or ``None`` (fail-closed).

    ``entry_liquidity`` is the liquidity recorded at position entry (the feed
    only fetches *live* market state). Pass an existing ``client`` to reuse a
    connection pool; otherwise a short-lived one is created and closed here.

    Returns ``None`` on any timeout, transport/HTTP error, or invalid/missing
    field. The loop must treat ``None`` as "no action this iteration".
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        raw = await _fetch_raw(token_address, client, timeout=timeout)
        return _build_snapshot(raw, entry_liquidity)
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        # httpx.HTTPError covers timeouts (TimeoutException), transport errors,
        # and non-2xx status (HTTPStatusError). Validation raises Value/Type/Key.
        logger.warning(
            "[feed] fail-closed for %s: %s -> no-action (None)",
            token_address,
            exc,
        )
        return None
    finally:
        if own_client:
            await client.aclose()


async def _fetch_dexscreener_liquidity(
    client: httpx.AsyncClient, token_address: str, *, timeout: float
) -> float:
    """Read-only GET of current pool liquidity (USD) from Dexscreener.

    Picks the Solana pair with the deepest liquidity. Raises ``ValueError`` if
    no pair carries a usable numeric liquidity (caller fails closed).
    """
    url = f"{DEXSCREENER_TOKENS_URL}/{token_address}"
    resp = await client.get(
        url, headers={"Accept": "application/json"}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()

    best_liq = -1.0
    for pair in data.get("pairs") or []:
        chain = str(pair.get("chainId", "")).lower()
        if chain and chain != "solana":
            continue
        raw_liq = (pair.get("liquidity") or {}).get("usd")
        if raw_liq is None:
            continue
        liq = float(raw_liq)
        if liq > best_liq:
            best_liq = liq
    if best_liq < 0:
        raise ValueError("no Solana pair with usable liquidity on Dexscreener")
    return best_liq


def _derive_volume_metrics(
    history: OHLCVHistory,
) -> "tuple[float, float, float, float]":
    """From chronological 1m candles derive the fields the loop's triggers need.

    Returns ``(rolling_volume_15m, peak_volume_15m, candle_1m_open,
    candle_1m_close)``: the trailing-15m volume sum, the maximum 15m rolling
    sum seen over the fetched window (the peak the volume-collapse trigger
    compares against), and the most recent 1m candle's open/close.
    """
    vols = history.volume_history
    n = len(vols)
    rolling_15m = sum(vols[-ROLLING_WINDOW:])

    if n >= ROLLING_WINDOW:
        window_sum = sum(vols[:ROLLING_WINDOW])
        peak = window_sum
        for i in range(ROLLING_WINDOW, n):
            window_sum += vols[i] - vols[i - ROLLING_WINDOW]
            peak = max(peak, window_sum)
    else:
        peak = rolling_15m  # not a full window yet -> peak == current

    last = history.candles[-1]
    return rolling_15m, peak, last.open, last.close


async def _safe_flow(
    fetch_flow: FlowFetcher,
    token_address: str,
    *,
    price: float,
    client: httpx.AsyncClient,
    timeout: float,
    rpc_url: Optional[str],
) -> OnchainFlow:
    """Call the flow fetcher, guaranteeing it never propagates an exception.

    ``get_recent_flow`` is already fail-closed, but this belt-and-braces guard
    means even a buggy injected fetcher degrades to an absent signal instead of
    crashing the feed or forcing a false-closed ``None`` snapshot.
    """
    try:
        return await fetch_flow(
            token_address, price=price, client=client, timeout=timeout, rpc_url=rpc_url
        )
    except Exception as exc:  # noqa: BLE001 — dump signal is strictly additive
        logger.warning(
            "[feed] on-chain flow fetch failed for %s: %s -> dump signal absent",
            token_address, exc,
        )
        return OnchainFlow()


async def get_live_market_snapshot(
    token_address: str,
    entry_liquidity: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    birdeye_api_key: Optional[str] = None,
    history_fetcher: Optional[HistoryFetcher] = None,
    flow_fetcher: Optional[FlowFetcher] = None,
    helius_rpc_url: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[MarketData]:
    """Assemble a REAL :class:`MarketData` snapshot for the management loop.

    Read-only and fail-closed (matches the ``snapshot_provider`` contract:
    ``async (token, entry_liquidity) -> Optional[MarketData]``). Sources:

        * Dexscreener — current pool liquidity (USD).
        * Birdeye 1m OHLCV (via :func:`data.price_history.get_price_history`) —
          current price (latest close), the latest 1m candle, and the rolling /
          peak 15m volume derived from the candles.
        * Helius (via :func:`data.onchain.get_recent_flow`) — the trade-level
          coordinated-dump signals ``largest_single_sell`` and
          ``top_wallet_sells``, valued in USD against the current price so they
          are comparable with the Dexscreener liquidity.

    The Helius flow fetch is fail-closed and NON-fatal: if it errors, times out,
    is unconfigured, or returns nothing, the two dump fields simply stay at their
    benign ``0`` and the rest of the snapshot is unaffected. That is safe by
    construction — those fields can only ever ADD a hard-exit trigger, never mask
    one, so an absent Helius signal can't make an unsafe position look safe.

    Returns ``None`` on any error, missing core data (price/liquidity), or fewer
    than one full 15m window of 1m candles — the loop treats ``None`` as
    "no action this iteration".
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    fetch_history = history_fetcher or get_price_history
    fetch_flow = flow_fetcher or get_recent_flow

    try:
        liquidity = await _fetch_dexscreener_liquidity(
            client, token_address, timeout=timeout
        )
        history = await fetch_history(
            token_address,
            interval=LIVE_INTERVAL,
            lookback_hours=LIVE_LOOKBACK_HOURS,
            min_candles=ROLLING_WINDOW,
            client=client,
            timeout=timeout,
            birdeye_api_key=birdeye_api_key,
            now=now,
        )
        # Need real 1m candles: the coarse Dexscreener fallback can't yield a
        # 1m candle or a true 15m volume, so anything but Birdeye fails closed.
        if (history is None or history.source != "birdeye"
                or len(history) < ROLLING_WINDOW):
            logger.warning(
                "[feed] %s no usable 1m candle history -> fail-closed (None)",
                token_address,
            )
            return None

        rolling_15m, peak_15m, c_open, c_close = _derive_volume_metrics(history)
        price = history.current_price

        # Helius coordinated-dump signals. Fail-closed and non-fatal: any error
        # leaves an absent (zeroed) flow, so the dump trigger stays dormant
        # rather than the whole snapshot failing closed to None.
        flow = await _safe_flow(
            fetch_flow, token_address, price=price, client=client,
            timeout=timeout, rpc_url=helius_rpc_url,
        )

        raw: Dict[str, Any] = {
            "price": price,                   # latest 1m close (Birdeye)
            "volume_15m": rolling_15m,
            "peak_volume_15m": peak_15m,
            "liquidity": liquidity,           # Dexscreener
            "largest_single_sell": flow.largest_single_sell_usd,  # Helius
            "top_wallet_sells": flow.selling_wallets,             # Helius
            "candle_1m_open": c_open,
            "candle_1m_close": c_close,
        }
        return _build_snapshot(raw, entry_liquidity)
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "[feed] live fail-closed for %s: %s -> no-action (None)",
            token_address,
            exc,
        )
        return None
    finally:
        if own_client:
            await client.aclose()


async def mock_market_snapshot(
    token_address: str,
    entry_liquidity: float = 100_000.0,
) -> MarketData:
    """In-memory stub snapshot (no network) so the loop can run end-to-end.

    Returns a deterministic, healthy snapshot that triggers no hard exit. Swap
    this for :func:`get_market_snapshot` once real providers are wired.
    """
    raw: Dict[str, Any] = {
        "price": 0.0015,
        "volume_15m": 50_000.0,
        "peak_volume_15m": 50_000.0,
        "liquidity": 100_000.0,
        "largest_single_sell": 0.0,
        "top_wallet_sells": 0,
        "candle_1m_open": 0.0015,
        "candle_1m_close": 0.0015,
    }
    return _build_snapshot(raw, entry_liquidity)


# --- Scenario mode ----------------------------------------------------------
# Scripted market paths for exercising the loop end-to-end without real data.

SCENARIOS: tuple = ("flat", "pump", "rug", "dump")

# Async provider matching main.run_loop's snapshot_provider contract.
ScenarioProvider = Callable[[str, float], Awaitable[Optional[MarketData]]]


def _make_snapshot(
    entry_liquidity: float,
    price: float,
    *,
    liquidity: Optional[float] = None,
    volume_15m: float = 50_000.0,
    peak_volume_15m: float = 50_000.0,
    largest_single_sell: float = 0.0,
    top_wallet_sells: int = 0,
    candle_open: Optional[float] = None,
    candle_close: Optional[float] = None,
) -> MarketData:
    """Build one validated snapshot. Defaults are hard-exit-healthy; the caller
    perturbs only the dimension a given scenario step needs to exercise."""
    raw: Dict[str, Any] = {
        "price": price,
        "volume_15m": volume_15m,
        "peak_volume_15m": peak_volume_15m,
        "liquidity": entry_liquidity if liquidity is None else liquidity,
        "largest_single_sell": largest_single_sell,
        "top_wallet_sells": top_wallet_sells,
        "candle_1m_open": price if candle_open is None else candle_open,
        "candle_1m_close": price if candle_close is None else candle_close,
    }
    return _build_snapshot(raw, entry_liquidity)


def _scenario_snapshots(
    name: str, entry_price: float, entry_liquidity: float
) -> List[MarketData]:
    """Return the scripted snapshot sequence for a scenario."""
    healthy_price = entry_price * 1.5

    if name == "flat":
        # Steady price ~1.5x: no hard exit, no profit tier -> HOLD (default).
        return [_make_snapshot(entry_liquidity, healthy_price)]

    if name == "pump":
        # Climb through 2x, 5x, 10x to exercise every profit-taking tier.
        return [
            _make_snapshot(entry_liquidity, entry_price * 2),
            _make_snapshot(entry_liquidity, entry_price * 5),
            _make_snapshot(entry_liquidity, entry_price * 10),
        ]

    if name == "rug":
        # Liquidity drops below 70% of entry -> hard-exit liquidity_drop.
        return [
            _make_snapshot(entry_liquidity, entry_price * 1.2),
            _make_snapshot(
                entry_liquidity, entry_price * 1.2, liquidity=entry_liquidity * 0.6
            ),
        ]

    if name == "dump":
        # Price crashes ~75% within one 1m candle -> hard-exit flash_crash.
        return [
            _make_snapshot(entry_liquidity, healthy_price),
            _make_snapshot(
                entry_liquidity,
                healthy_price * 0.25,
                candle_open=healthy_price,
                candle_close=healthy_price * 0.25,
            ),
        ]

    raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")


def make_scenario_provider(
    name: str,
    entry_price: float,
    entry_liquidity: float = 100_000.0,
) -> ScenarioProvider:
    """Build a stateful async provider that yields a scenario's snapshots in
    order over successive calls, then holds the final snapshot once exhausted.

    The returned callable matches ``main.run_loop``'s ``snapshot_provider``
    contract: ``async (token_address, entry_liquidity) -> MarketData``.
    """
    snapshots = _scenario_snapshots(name, entry_price, entry_liquidity)
    state = {"step": 0}

    async def provider(token_address: str, entry_liq: float) -> Optional[MarketData]:
        idx = min(state["step"], len(snapshots) - 1)
        state["step"] += 1
        return snapshots[idx]

    provider.__name__ = f"scenario_{name}"
    return provider
