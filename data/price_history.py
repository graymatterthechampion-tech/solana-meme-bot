"""Read-only historical OHLCV price-data layer.

Feeds REAL recent price/volume history into the post-pump dip-buy entry logic
(:mod:`strategies.entry`). Until this layer existed, scanned candidates reached
the entry decision with EMPTY history and always failed closed to SKIP; this
module fetches candles so :func:`strategies.entry.evaluate_entry` can actually
evaluate ATH age, pullback %, the consolidation band, and the volume spike.

STRICTLY READ-ONLY
------------------
HTTP GETs and pure derivation only. This module imports no signer, keypair, or
transaction-building code and never buys, sells, signs, or broadcasts.

DATA SOURCES (tried in priority order)
--------------------------------------
PRIMARY   — GeckoTerminal public API (``api.geckoterminal.com/api/v2``). FREE, no
            API key. Two read-only GETs per token: token -> deepest pool
            (``/networks/solana/tokens/{mint}/pools``), then real OHLCV candles for
            that pool (``/networks/solana/pools/{pool}/ohlcv/{timeframe}``). Returns
            true per-interval candles — the same fidelity as Birdeye — at zero cost,
            so it leads the chain.

            RATE-LIMITED: the free tier allows only ~30 calls/min (and can be as low
            as ~10). Every GeckoTerminal call is spaced by a strict client-side
            throttle (:func:`_gecko_throttle`, min interval
            :data:`GECKO_MIN_CALL_INTERVAL`) so the bot proactively stays under the
            limit. If a 429 slips through anyway it is retried with backoff by the
            shared request path and, on exhaustion, falls through to the next source
            — never blocking the trading loop.
SECONDARY — Birdeye OHLCV API (``/defi/ohlcv``). Requires an API key, read from
            ``.env`` via ``config.BIRDEYE_API_KEY`` (never hardcoded). Used only when
            GeckoTerminal yields nothing usable; resumes automatically once a Birdeye
            quota resets.

            PLAN-GATED: Birdeye answers OHLCV with ``400 {"success":false,
            "message":"Compute units usage limit exceeded"}`` on API plans that do
            not grant the historical/time-series endpoints — regardless of headers,
            candle count, or endpoint version (``/defi/ohlcv`` and ``/defi/v3/ohlcv``
            behave identically; ``/defi/price`` and ``/defi/token_overview`` still
            return 200 on the same key, which is how you tell a plan gate apart
            from an exhausted quota). That is a PERMANENT rejection, not a
            transient error, so the first one trips a circuit breaker
            (:func:`birdeye_status`) and every later call skips straight to the
            fallback instead of re-billing a round-trip per token per scan.
LAST-RESORT — Dexscreener token API (public, no key). When BOTH real-candle sources
            fail, are unconfigured, or return too little data, a COARSE recent series
            is reconstructed from Dexscreener's trailing price-change / volume
            buckets (m5/h1/h6/h24). Best-effort: enough to evaluate, never a
            substitute for true candles.

FAIL-CLOSED CONTRACT
--------------------
On ANY API error, timeout, rate-limit, or insufficient history BOTH sources
yield nothing and :func:`get_price_history` returns ``None``. The caller then
builds an empty-history snapshot and entry continues to fail closed to SKIP —
we never guess on a token whose recent run we cannot verify.

RATE LIMITS
-----------
Requests reuse the existing retry/backoff path
(:func:`safety.rug_check._request_json`): retry on 429/5xx/transient errors,
honoring a server ``Retry-After`` on 429, otherwise exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

import config
# Reuse the project's hardened read-only request helper (retry + backoff +
# Retry-After handling). Its rate-limit knobs live as module globals on
# rug_check, so tests zero them there and this path honors it too.
from safety.rug_check import _request_json
from strategies.entry import EntryMarketData

logger = logging.getLogger(__name__)

# --- Endpoints (read-only) --------------------------------------------------
GECKOTERMINAL_BASE: str = "https://api.geckoterminal.com/api/v2"
BIRDEYE_OHLCV_URL: str = "https://public-api.birdeye.so/defi/ohlcv"
DEXSCREENER_TOKENS_URL: str = "https://api.dexscreener.com/latest/dex/tokens"
CHAIN: str = "solana"

# --- Defaults (tunable) -----------------------------------------------------
DEFAULT_INTERVAL: str = "5m"          # candle width (Birdeye-style key, mapped per source)
DEFAULT_LOOKBACK_HOURS: float = 24.0  # how far back to request candles
DEFAULT_MIN_CANDLES: int = 5          # fewer real candles -> insufficient
DEFAULT_TIMEOUT: float = 8.0          # per-request timeout (seconds)
FALLBACK_MIN_POINTS: int = 2          # fewer fallback points -> unusable

# --- GeckoTerminal free-tier limits (tunable) -------------------------------
# The free tier caps OHLCV requests at 1000 candles and the whole API at ~30
# calls/min (and can be throttled to as low as ~10). We SELF-limit below 30 with
# headroom; each token costs two calls (pool lookup + OHLCV), so ~25 calls/min is
# ~12 tokens/min. If you observe 429s, lower GECKO_RATE_LIMIT_PER_MIN toward 10.
GECKO_OHLCV_MAX_LIMIT: int = 1000
GECKO_RATE_LIMIT_PER_MIN: float = 25.0
# Minimum seconds between two consecutive GeckoTerminal calls (module global so
# tests can zero it to skip real sleeping). Derived from the per-minute budget.
GECKO_MIN_CALL_INTERVAL: float = 60.0 / GECKO_RATE_LIMIT_PER_MIN

# Map a Birdeye-style candle key to GeckoTerminal's (timeframe, aggregate). Only
# widths the free OHLCV endpoint actually supports; anything else -> 5-minute.
_GECKO_TIMEFRAME: Dict[str, Tuple[str, int]] = {
    "1m": ("minute", 1), "5m": ("minute", 5), "15m": ("minute", 15),
    "1H": ("hour", 1), "4H": ("hour", 4), "12H": ("hour", 12),
    "1D": ("day", 1),
}

# Coarse, irregular spacing for the Dexscreener fallback series; the two most
# recent points (m5 -> now) are ~5 minutes apart, which is what the entry
# consolidation window keys off, so report 5m as the representative interval.
DEX_FALLBACK_INTERVAL_MINUTES: float = 5.0

# Birdeye candle ``type`` -> minutes, used as the entry sample interval.
_INTERVAL_MINUTES: Dict[str, float] = {
    "1m": 1.0, "3m": 3.0, "5m": 5.0, "15m": 15.0, "30m": 30.0,
    "1H": 60.0, "2H": 120.0, "4H": 240.0, "6H": 360.0, "8H": 480.0,
    "12H": 720.0, "1D": 1440.0,
}

# HTTP statuses from Birdeye that are PERMANENT rejections of the request as
# posed (bad params, dead key, or an endpoint the plan does not grant). Retrying
# or re-requesting these for the next token cannot help, so the first one trips
# the circuit breaker below. 429/5xx are deliberately absent: those ARE transient
# and are already retried with backoff inside _request_json.
BIRDEYE_FATAL_STATUSES: Tuple[int, ...] = (400, 401, 402, 403)

# Set to the reason string once Birdeye permanently rejects us; while set, the
# Birdeye leg is skipped entirely and we serve history from Dexscreener. Process
# lifetime only — restarting the bot (e.g. after upgrading the API plan) re-arms
# Birdeye. Tests reset it via reset_birdeye_circuit().
_birdeye_disabled_reason: Optional[str] = None

# Trailing Dexscreener buckets, oldest window first: (key, seconds_ago).
_DEX_WINDOWS: Tuple[Tuple[str, float], ...] = (
    ("h24", 86_400.0),
    ("h6", 21_600.0),
    ("h1", 3_600.0),
    ("m5", 300.0),
)

# Async read-only OHLCV source: mint -> history (or None, fail-closed).
HistoryFetcher = Callable[..., Awaitable[Optional["OHLCVHistory"]]]


@dataclass(frozen=True)
class Candle:
    """One OHLCV candle. ``timestamp`` is epoch seconds (candle open time)."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OHLCVHistory:
    """Validated recent OHLCV history for one mint.

    ``candles`` are chronological (oldest first). ``source`` records which
    provider produced it ("birdeye" | "dexscreener"). ``interval_minutes`` is
    the candle spacing the entry logic uses to map its consolidation window.
    """

    mint_address: str
    interval_minutes: float
    source: str
    candles: List[Candle]

    def __len__(self) -> int:
        return len(self.candles)

    @property
    def price_history(self) -> List[float]:
        """Closes, chronological (oldest first) — the entry price series."""
        return [c.close for c in self.candles]

    @property
    def volume_history(self) -> List[float]:
        """Per-candle volumes, chronological — drives the volume-spike check."""
        return [c.volume for c in self.candles]

    @property
    def current_price(self) -> float:
        """Most recent close."""
        return self.candles[-1].close

    def ath(self) -> Tuple[float, float]:
        """Recent all-time-high ``(price, timestamp)`` over the window, by the
        highest candle high."""
        peak = max(self.candles, key=lambda c: c.high)
        return peak.high, peak.timestamp


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float, or ``None`` if missing/non-numeric (fail-closed)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval_to_minutes(interval: str) -> float:
    """Map a Birdeye candle ``type`` to minutes (default 5m if unrecognised)."""
    return _INTERVAL_MINUTES.get(interval, 5.0)


def _gecko_timeframe(interval: str) -> Tuple[str, int]:
    """Map a candle key to GeckoTerminal ``(timeframe, aggregate)`` (default 5m)."""
    return _GECKO_TIMEFRAME.get(interval, ("minute", 5))


# --- GeckoTerminal client-side rate limiter ---------------------------------
# The free tier limits the WHOLE API (not per endpoint) to ~30 calls/min, so we
# serialise every GeckoTerminal call through one throttle and space consecutive
# calls by at least GECKO_MIN_CALL_INTERVAL. This is proactive (avoid the 429),
# complementing the reactive 429 backoff already in _request_json.

# monotonic timestamp of the last GeckoTerminal call let through the throttle.
_gecko_last_call: float = 0.0
# Serialises the throttle. Created lazily and rebound if the running event loop
# changes (each asyncio.run() in tests is a fresh loop), so a lock is never
# awaited from a loop it was not created on.
_gecko_lock: Optional[asyncio.Lock] = None
_gecko_lock_loop: Optional[asyncio.AbstractEventLoop] = None


def _gecko_throttle_lock() -> asyncio.Lock:
    """Return a lock bound to the currently running event loop."""
    global _gecko_lock, _gecko_lock_loop
    loop = asyncio.get_running_loop()
    if _gecko_lock is None or _gecko_lock_loop is not loop:
        _gecko_lock = asyncio.Lock()
        _gecko_lock_loop = loop
    return _gecko_lock


def reset_gecko_throttle() -> None:
    """Clear the throttle's last-call clock (used by tests)."""
    global _gecko_last_call
    _gecko_last_call = 0.0


async def _gecko_throttle() -> None:
    """Block until at least :data:`GECKO_MIN_CALL_INTERVAL` has passed since the
    previous GeckoTerminal call, so the free-tier per-minute budget is respected.

    A non-positive interval disables throttling entirely (tests set it to 0).
    """
    global _gecko_last_call
    interval = GECKO_MIN_CALL_INTERVAL
    if interval <= 0:
        return
    async with _gecko_throttle_lock():
        wait = _gecko_last_call + interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _gecko_last_call = time.monotonic()


# --- GeckoTerminal (primary) ------------------------------------------------

def _gecko_top_pool_address(data: Dict[str, Any]) -> Optional[str]:
    """Pick the deepest-liquidity pool address from a token->pools response.

    Each item's ``attributes.reserve_in_usd`` is the pool's total liquidity; the
    highest wins (GeckoTerminal usually returns them liquidity-sorted already, but
    we do not rely on that). Returns ``None`` if no pool carries an address.
    """
    best: Optional[str] = None
    best_liq = -1.0
    for pool in (data or {}).get("data") or []:
        attrs = pool.get("attributes") or {}
        address = attrs.get("address")
        if not address:
            continue
        liq = _as_float(attrs.get("reserve_in_usd")) or 0.0
        if liq > best_liq:
            best, best_liq = str(address), liq
    return best


def _geckoterminal_to_history(
    mint_address: str, data: Dict[str, Any], interval: str
) -> Optional[OHLCVHistory]:
    """Parse a GeckoTerminal OHLCV payload into history, or ``None`` if empty.

    ``data.attributes.ohlcv_list`` is a list of ``[unixTime, o, h, l, c, v]`` rows
    (GeckoTerminal returns them newest-first; we sort chronologically). Each row
    must be complete, numeric, and carry a positive close/high (fail-closed per row).
    """
    ohlcv_list = (
        ((data or {}).get("data") or {}).get("attributes") or {}
    ).get("ohlcv_list") or []
    candles: List[Candle] = []
    for row in ohlcv_list:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = _as_float(row[0])
        o = _as_float(row[1])
        h = _as_float(row[2])
        low = _as_float(row[3])
        c = _as_float(row[4])
        v = _as_float(row[5])
        if None in (ts, o, h, low, c, v):
            continue  # incomplete row -> skip (fail-closed per row)
        if c <= 0 or h <= 0:
            continue
        candles.append(
            Candle(
                timestamp=ts, open=o, high=h, low=low, close=c, volume=max(0.0, v)
            )
        )

    if not candles:
        return None
    candles.sort(key=lambda candle: candle.timestamp)
    return OHLCVHistory(
        mint_address=mint_address,
        interval_minutes=_interval_to_minutes(interval),
        source="geckoterminal",
        candles=candles,
    )


async def _fetch_geckoterminal(
    client: httpx.AsyncClient,
    mint_address: str,
    *,
    interval: str,
    lookback_hours: float,
    min_candles: int,
    timeout: float,
) -> Optional[OHLCVHistory]:
    """Read-only two-step GeckoTerminal fetch: token -> deepest pool -> OHLCV.

    FREE and keyless. Every call is spaced by :func:`_gecko_throttle` to stay under
    the free-tier rate limit. Returns ``None`` when the token has no pool or the
    OHLCV payload is empty; RAISES on transport/HTTP error (incl. a 429 that
    survived :func:`_request_json`'s backoff) so the caller falls through the chain.
    """
    # Step 1: token -> its pools, pick the deepest.
    pools_url = f"{GECKOTERMINAL_BASE}/networks/{CHAIN}/tokens/{mint_address}/pools"
    await _gecko_throttle()
    pools_data = await _request_json(
        client, "GET", pools_url, headers={"Accept": "application/json"},
        timeout=timeout, label="geckoterminal-pools",
    )
    pool_address = _gecko_top_pool_address(pools_data)
    if pool_address is None:
        logger.info(
            "[price_history] %s no GeckoTerminal pool -> next source", mint_address
        )
        return None

    # Step 2: pool -> OHLCV candles. Request only as many candles as the lookback
    # window needs (capped at the free-tier max), never fewer than min_candles.
    timeframe, aggregate = _gecko_timeframe(interval)
    interval_minutes = _interval_to_minutes(interval)
    wanted = math.ceil(lookback_hours * 60.0 / interval_minutes) if interval_minutes > 0 else min_candles
    limit = max(min_candles, min(GECKO_OHLCV_MAX_LIMIT, wanted))
    ohlcv_url = str(
        httpx.URL(
            f"{GECKOTERMINAL_BASE}/networks/{CHAIN}/pools/{pool_address}/ohlcv/{timeframe}",
            params={"aggregate": aggregate, "limit": limit},
        )
    )
    await _gecko_throttle()
    ohlcv_data = await _request_json(
        client, "GET", ohlcv_url, headers={"Accept": "application/json"},
        timeout=timeout, label="geckoterminal-ohlcv",
    )
    return _geckoterminal_to_history(mint_address, ohlcv_data, interval)


# --- Birdeye circuit breaker ------------------------------------------------

def birdeye_status() -> Optional[str]:
    """Why the Birdeye leg is disabled, or ``None`` if it is still armed."""
    return _birdeye_disabled_reason


def reset_birdeye_circuit() -> None:
    """Re-arm the Birdeye leg (used by tests, and on a fresh process)."""
    global _birdeye_disabled_reason
    _birdeye_disabled_reason = None


def _disable_birdeye(reason: str) -> None:
    """Trip the breaker once, logging loudly enough to be actionable."""
    global _birdeye_disabled_reason
    if _birdeye_disabled_reason is not None:
        return
    _birdeye_disabled_reason = reason
    logger.error(
        "[price_history] Birdeye permanently rejected us (%s) -> disabling the "
        "Birdeye leg for this process; serving COARSE Dexscreener history "
        "instead. Entry quality is degraded until the API plan/key is fixed.",
        reason,
    )


# --- Birdeye (primary) ------------------------------------------------------

def _birdeye_to_history(
    mint_address: str, data: Dict[str, Any], interval: str
) -> Optional[OHLCVHistory]:
    """Parse a Birdeye OHLCV payload into history, or ``None`` if no valid
    candles. Each item must carry a complete, numeric, positive-close OHLCV."""
    items = ((data or {}).get("data") or {}).get("items") or []
    candles: List[Candle] = []
    for item in items:
        ts = _as_float(item.get("unixTime"))
        o = _as_float(item.get("o"))
        h = _as_float(item.get("h"))
        low = _as_float(item.get("l"))
        c = _as_float(item.get("c"))
        v = _as_float(item.get("v"))
        if None in (ts, o, h, low, c, v):
            continue  # incomplete candle -> skip (fail-closed per item)
        if c <= 0 or h <= 0:
            continue
        candles.append(
            Candle(
                timestamp=ts, open=o, high=h, low=low, close=c, volume=max(0.0, v)
            )
        )

    if not candles:
        return None
    candles.sort(key=lambda candle: candle.timestamp)
    return OHLCVHistory(
        mint_address=mint_address,
        interval_minutes=_interval_to_minutes(interval),
        source="birdeye",
        candles=candles,
    )


async def _fetch_birdeye(
    client: httpx.AsyncClient,
    mint_address: str,
    *,
    interval: str,
    lookback_hours: float,
    timeout: float,
    api_key: Optional[str],
    now: float,
) -> Optional[OHLCVHistory]:
    """Read-only GET of Birdeye OHLCV candles.

    Returns ``None`` (skip -> caller falls back) when no API key is configured or
    the circuit breaker is already tripped, and trips the breaker on a permanent
    rejection (see :data:`BIRDEYE_FATAL_STATUSES`). Transient failures (429/5xx,
    timeouts) still raise after :func:`_request_json` exhausts its backoff, so
    the caller falls back for this token but Birdeye stays armed for the next.
    """
    if not api_key:
        logger.info(
            "[price_history] no Birdeye API key (BIRDEYE_API_KEY) -> skip Birdeye"
        )
        return None

    if _birdeye_disabled_reason is not None:
        logger.debug(
            "[price_history] Birdeye disabled (%s) -> fallback",
            _birdeye_disabled_reason,
        )
        return None

    time_to = int(now)
    time_from = int(now - lookback_hours * 3600.0)
    url = str(
        httpx.URL(
            BIRDEYE_OHLCV_URL,
            params={
                "address": mint_address,
                "type": interval,
                "time_from": time_from,
                "time_to": time_to,
            },
        )
    )
    headers = {
        "X-API-KEY": api_key,
        "x-chain": CHAIN,
        "accept": "application/json",
    }
    try:
        data = await _request_json(
            client, "GET", url, headers=headers, timeout=timeout, label="birdeye"
        )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in BIRDEYE_FATAL_STATUSES:
            # _request_json has already logged the full body; carry the message
            # into the breaker reason so the operator sees WHY, not just "400".
            _disable_birdeye(f"HTTP {status}: {_birdeye_message(exc.response)}")
            return None
        raise  # transient (429/5xx exhausted) -> fall back, stay armed

    return _birdeye_to_history(mint_address, data, interval)


def _birdeye_message(resp: httpx.Response) -> str:
    """Birdeye's own error text (``{"success":false,"message":...}``), if any."""
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001 — non-JSON error page
        return "<non-JSON body>"
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("error")
        if message:
            return str(message)
    return "<no message>"


# --- Dexscreener (fallback) -------------------------------------------------

def _best_dexscreener_pair(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pick the Solana pair with a valid price and the deepest liquidity."""
    best: Optional[Dict[str, Any]] = None
    best_liq = -1.0
    for pair in data.get("pairs") or []:
        chain = str(pair.get("chainId", "")).lower()
        if chain and chain != CHAIN:
            continue
        price = _as_float(pair.get("priceUsd"))
        if price is None or price <= 0:
            continue
        liq = _as_float((pair.get("liquidity") or {}).get("usd")) or 0.0
        if liq > best_liq:
            best, best_liq = pair, liq
    return best


def _dexscreener_to_history(
    mint_address: str, data: Dict[str, Any], now: float
) -> Optional[OHLCVHistory]:
    """Reconstruct a coarse recent series from Dexscreener trailing buckets.

    Prices are derived from trailing price-change percentages (price ``window``
    ago = current / (1 + pct/100)); per-bucket volumes are the non-overlapping
    increments of the cumulative window volumes. Returns ``None`` if fewer than
    :data:`FALLBACK_MIN_POINTS` usable points can be formed.
    """
    pair = _best_dexscreener_pair(data)
    if pair is None:
        return None

    price = float(pair["priceUsd"])  # validated in _best_dexscreener_pair
    price_change = pair.get("priceChange") or {}
    volume = pair.get("volume") or {}

    candles: List[Candle] = []
    for idx, (key, seconds_ago) in enumerate(_DEX_WINDOWS):
        change = _as_float(price_change.get(key))
        if change is None:
            continue
        denom = 1.0 + change / 100.0
        if denom <= 0:
            continue
        price_then = price / denom
        if price_then <= 0:
            continue
        # Non-overlapping increment vs the next (more recent, smaller) window.
        win_vol = _as_float(volume.get(key))
        next_vol = (
            _as_float(volume.get(_DEX_WINDOWS[idx + 1][0]))
            if idx + 1 < len(_DEX_WINDOWS)
            else 0.0
        )
        if win_vol is None:
            bucket_vol = 0.0
        else:
            inc = win_vol - (next_vol or 0.0)
            bucket_vol = inc if inc >= 0 else win_vol
        candles.append(
            Candle(
                timestamp=now - seconds_ago,
                open=price_then,
                high=price_then,
                low=price_then,
                close=price_then,
                volume=max(0.0, bucket_vol),
            )
        )

    # Most recent point: the live price, carrying the freshest (m5) volume.
    m5_vol = _as_float(volume.get("m5")) or 0.0
    candles.append(
        Candle(
            timestamp=now,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=max(0.0, m5_vol),
        )
    )

    candles.sort(key=lambda candle: candle.timestamp)
    if len(candles) < FALLBACK_MIN_POINTS:
        return None
    return OHLCVHistory(
        mint_address=mint_address,
        interval_minutes=DEX_FALLBACK_INTERVAL_MINUTES,
        source="dexscreener",
        candles=candles,
    )


async def _fetch_dexscreener(
    client: httpx.AsyncClient,
    mint_address: str,
    *,
    timeout: float,
    now: float,
) -> Optional[OHLCVHistory]:
    """Read-only GET of Dexscreener token data, reshaped into a coarse series.
    Raises on transport/HTTP error; the caller fails closed."""
    url = f"{DEXSCREENER_TOKENS_URL}/{mint_address}"
    data = await _request_json(
        client,
        "GET",
        url,
        headers={"Accept": "application/json"},
        timeout=timeout,
        label="dexscreener-ohlcv",
    )
    return _dexscreener_to_history(mint_address, data, now)


# --- Public entry point -----------------------------------------------------

async def get_price_history(
    mint_address: str,
    *,
    interval: str = DEFAULT_INTERVAL,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    min_candles: int = DEFAULT_MIN_CANDLES,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    birdeye_api_key: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[OHLCVHistory]:
    """Fetch recent OHLCV history for a mint through the source chain.

    Read-only and fail-closed. Tries sources in priority order and returns the
    first that yields at least ``min_candles`` real candles:
      1. GeckoTerminal (free, keyless, throttled) — real candles;
      2. Birdeye (if an API key is configured) — real candles;
      3. Dexscreener — a COARSE reconstructed series (accepted at
         ``FALLBACK_MIN_POINTS`` points), last resort only.
    On any error, empty/insufficient data, or missing config a source is skipped
    and the next is tried. Returns ``None`` if none yields usable history, so the
    caller's entry decision stays fail-closed (SKIP).

    ``birdeye_api_key`` defaults to ``config.BIRDEYE_API_KEY`` (read from .env).
    Pass an existing ``client`` to reuse a connection pool (and to inject a mock
    transport in tests); otherwise a short-lived one is created and closed here.
    """
    now = time.time() if now is None else now
    api_key = config.BIRDEYE_API_KEY if birdeye_api_key is None else birdeye_api_key

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)

    try:
        # --- PRIMARY: GeckoTerminal (free, real candles) ----------------------
        try:
            gecko = await _fetch_geckoterminal(
                client,
                mint_address,
                interval=interval,
                lookback_hours=lookback_hours,
                min_candles=min_candles,
                timeout=timeout,
            )
            if gecko is not None and len(gecko) >= min_candles:
                logger.info(
                    "[price_history] %s -> %d GeckoTerminal candle(s)",
                    mint_address,
                    len(gecko),
                )
                return gecko
            if gecko is not None:
                logger.warning(
                    "[price_history] %s GeckoTerminal returned %d candle(s) < min %d "
                    "-> trying next source",
                    mint_address,
                    len(gecko),
                    min_candles,
                )
        except Exception as exc:  # noqa: BLE001 — any error (incl. 429) -> next source
            logger.warning(
                "[price_history] %s GeckoTerminal failed: %s -> trying next source",
                mint_address,
                exc,
            )

        # --- SECONDARY: Birdeye -----------------------------------------------
        try:
            birdeye = await _fetch_birdeye(
                client,
                mint_address,
                interval=interval,
                lookback_hours=lookback_hours,
                timeout=timeout,
                api_key=api_key,
                now=now,
            )
            if birdeye is not None and len(birdeye) >= min_candles:
                logger.info(
                    "[price_history] %s -> %d Birdeye candle(s)",
                    mint_address,
                    len(birdeye),
                )
                return birdeye
            if birdeye is not None:
                logger.warning(
                    "[price_history] %s Birdeye returned %d candle(s) < min %d "
                    "-> trying next source",
                    mint_address,
                    len(birdeye),
                    min_candles,
                )
        except Exception as exc:  # noqa: BLE001 — any error -> next source
            logger.warning(
                "[price_history] %s Birdeye failed: %s -> trying next source",
                mint_address,
                exc,
            )

        # --- LAST RESORT: Dexscreener (coarse reconstruction) ----------------
        try:
            fallback = await _fetch_dexscreener(
                client, mint_address, timeout=timeout, now=now
            )
            if fallback is not None and len(fallback) >= FALLBACK_MIN_POINTS:
                logger.info(
                    "[price_history] %s -> %d Dexscreener fallback point(s)",
                    mint_address,
                    len(fallback),
                )
                return fallback
        except Exception as exc:  # noqa: BLE001 — any error -> fail closed
            logger.warning(
                "[price_history] %s Dexscreener fallback failed: %s",
                mint_address,
                exc,
            )

        logger.warning(
            "[price_history] %s no usable history from any source (fail-closed)",
            mint_address,
        )
        return None
    finally:
        if own_client:
            await client.aclose()


def build_entry_market_data(
    *,
    current_price: float,
    current_liquidity: float,
    market_cap_usd: Optional[float],
    history: Optional[OHLCVHistory],
    pre_dip_liquidity: Optional[float] = None,
    now: Optional[float] = None,
) -> EntryMarketData:
    """Assemble :class:`EntryMarketData` from a liquidity snapshot + OHLCV history.

    With usable ``history`` (>= 2 candles), the price/volume series, recent ATH
    (price + timestamp), current price, and sample interval all come from the
    candles — so :func:`strategies.entry.evaluate_entry` can judge ATH age,
    pullback, consolidation, and the volume spike. With no history it returns an
    EMPTY-history snapshot, which evaluate_entry rejects as insufficient (SKIP)
    rather than guessing.

    Liquidity is not part of OHLCV: ``current_liquidity`` is the live snapshot.
    ``pre_dip_liquidity`` defaults to ``0.0`` when unknown, which DISABLES the
    entry's drain check (see :func:`strategies.entry.evaluate_entry` step 5) —
    an honest "no signal" rather than defaulting to the current value, which
    would make the drain check silently pass and read as false comfort. There is
    no historical-liquidity source wired yet; supply a real figure to arm it.
    """
    now = time.time() if now is None else now
    pre_dip = 0.0 if pre_dip_liquidity is None else pre_dip_liquidity

    if history is not None and len(history) >= 2:
        ath_price, ath_timestamp = history.ath()
        return EntryMarketData(
            current_price=history.current_price,
            ath_price=ath_price,
            ath_timestamp=ath_timestamp,
            price_history=history.price_history,
            volume_history=history.volume_history,
            current_liquidity=current_liquidity,
            pre_dip_liquidity=pre_dip,
            sample_interval_minutes=history.interval_minutes,
            market_cap_usd=market_cap_usd,
            now=now,
        )

    # No usable history -> empty series -> evaluate_entry fails closed to SKIP.
    return EntryMarketData(
        current_price=current_price,
        ath_price=current_price,
        ath_timestamp=now,
        price_history=[],
        volume_history=[],
        current_liquidity=current_liquidity,
        pre_dip_liquidity=pre_dip,
        market_cap_usd=market_cap_usd,
        now=now,
    )
