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

TODO(real-apis): wire real providers in ``_fetch_raw``:
    * Dexscreener — GET https://api.dexscreener.com/latest/dex/tokens/{address}
      (no key) for price / liquidity / volume.
    * Birdeye — GET https://public-api.birdeye.so/... with header
      ``X-API-KEY: config.BIRDEYE_API_KEY`` for OHLCV / rolling volume.
    * Helius — RPC via ``config.RPC_URL`` (``config.HELIUS_API_KEY``) for
      on-chain wallet activity (largest single sell, coordinated top-wallet
      dumps). Cross-check fields across sources before trusting them.

TODO(execution-realism): dry-run execution currently logs intended sells with
zero market friction. Before any live trading, model realistic SLIPPAGE,
PRIORITY/SWAP FEES, and FILL DELAY so dry-run PnL reflects reality and the
slippage caps in CLAUDE.md are actually exercised.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

import config
from strategies.hard_exit import MarketData

logger = logging.getLogger(__name__)

# Default per-request timeout (seconds). A slow feed must fail-closed, not hang
# the trading loop.
DEFAULT_TIMEOUT: float = 5.0

# Placeholder base URL until real providers are wired. Real runs against this
# will simply fail-closed (return None); tests inject a mock client.
_PLACEHOLDER_BASE: str = "https://example.invalid/markets"

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
