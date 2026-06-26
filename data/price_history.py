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

DATA SOURCES
------------
PRIMARY  — Birdeye OHLCV API (``/defi/ohlcv``). Requires an API key, read from
           ``.env`` via ``config.BIRDEYE_API_KEY`` (never hardcoded). Returns
           real per-interval candles, the highest-fidelity source for ATH /
           pullback / consolidation / volume-spike math.
FALLBACK — Dexscreener token API (public, no key). When Birdeye fails, is
           unconfigured, or returns too little data, a COARSE recent series is
           reconstructed from Dexscreener's trailing price-change / volume
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

import logging
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
BIRDEYE_OHLCV_URL: str = "https://public-api.birdeye.so/defi/ohlcv"
DEXSCREENER_TOKENS_URL: str = "https://api.dexscreener.com/latest/dex/tokens"
CHAIN: str = "solana"

# --- Defaults (tunable) -----------------------------------------------------
DEFAULT_INTERVAL: str = "5m"          # Birdeye candle width
DEFAULT_LOOKBACK_HOURS: float = 24.0  # how far back to request candles
DEFAULT_MIN_CANDLES: int = 5          # fewer Birdeye candles -> insufficient
DEFAULT_TIMEOUT: float = 8.0          # per-request timeout (seconds)
FALLBACK_MIN_POINTS: int = 2          # fewer fallback points -> unusable

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
    """Read-only GET of Birdeye OHLCV candles. Returns ``None`` (skip) when no
    API key is configured; raises on transport/HTTP error so the caller falls
    back. The retry/backoff path is reused via :func:`_request_json`."""
    if not api_key:
        logger.info(
            "[price_history] no Birdeye API key (BIRDEYE_API_KEY) -> skip Birdeye"
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
    data = await _request_json(
        client, "GET", url, headers=headers, timeout=timeout, label="birdeye"
    )
    return _birdeye_to_history(mint_address, data, interval)


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
    """Fetch recent OHLCV history for a mint: Birdeye first, Dexscreener fallback.

    Read-only and fail-closed. Tries Birdeye (if an API key is configured) and
    accepts it only with at least ``min_candles`` candles; otherwise — on any
    error, empty/insufficient data, or no key — falls back to a coarse
    Dexscreener series. Returns ``None`` if neither source yields usable
    history, so the caller's entry decision stays fail-closed (SKIP).

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
        # --- PRIMARY: Birdeye -------------------------------------------------
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
                    "-> trying fallback",
                    mint_address,
                    len(birdeye),
                    min_candles,
                )
        except Exception as exc:  # noqa: BLE001 — any error -> fall back
            logger.warning(
                "[price_history] %s Birdeye failed: %s -> trying fallback",
                mint_address,
                exc,
            )

        # --- FALLBACK: Dexscreener -------------------------------------------
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

    Liquidity is not part of OHLCV: ``current_liquidity`` is the live snapshot
    and ``pre_dip_liquidity`` defaults to it (neutral drain check) unless the
    caller supplies a pre-dip figure.
    """
    now = time.time() if now is None else now
    pre_dip = current_liquidity if pre_dip_liquidity is None else pre_dip_liquidity

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
