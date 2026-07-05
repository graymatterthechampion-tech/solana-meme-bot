"""Read-only candidate token scanner: real, moving Solana setups.

DISCOVERY ONLY. This module's sole job is to discover and surface candidate
token mint addresses for the existing safety -> entry pipeline to evaluate. It
performs read-only HTTP GETs and pure filtering and returns a list of candidate
dicts. It NEVER buys, sells, signs, or broadcasts, and imports no signer,
keypair, or transaction-building code.

STRATEGY ALIGNMENT
------------------
The entry strategies (post-pump dip-buy and momentum/breakout) both act on
tokens that are ALREADY MOVING on established liquidity — not flat tokens and
not brand-new launches. So the scanner is built around *movers*: it discovers
tokens that are being promoted / trending, then keeps only those that are
demonstrably up recently AND deep/liquid/aged enough to trade.

CANDIDATE SOURCES (aggregated)
------------------------------
1. DEXSCREENER — the public token-boosts endpoints (``token-boosts/top`` and
   ``token-boosts/latest``), Dexscreener's public "promoted / trending" signal.
   No API key. Filtered to Solana.
2. BIRDEYE — the trending-token endpoint (``/defi/token_trending``), a
   higher-quality Solana mover feed. Requires an API key read from ``.env`` via
   ``config.BIRDEYE_API_KEY`` (never hardcoded). ONE ranked call per scan
   (``birdeye_limit`` addresses) to conserve compute units.

Both sources yield only mint ADDRESSES. Every discovered address is then
ENRICHED through a single batched Dexscreener token lookup
(``/latest/dex/tokens/{addrs}``, up to 30 mints per request), which returns the
uniform per-pair data (DEX, liquidity, 24h volume, market cap, pair age, and the
1h/6h price change) the pre-filter needs. One consistent filter path, whatever
the source.

PRE-FILTER (tradeable movers only; every threshold configurable)
----------------------------------------------------------------
* Solana pairs only — any non-Solana pair is rejected outright.
* Established Solana AMMs only — Raydium, Orca, Meteora, and the Pump.fun AMM
  (graduated ``pumpswap`` pools). Un-graduated bonding-curve venues (raw
  ``pumpfun``/``pump``/``moonshot``) are rejected even if the allow-list names
  them.
* Liquidity >= ``min_liquidity_usd`` (default $30k).
* 24h volume >= ``min_volume_24h_usd`` (default $100k).
* Market cap within [``min_market_cap_usd``, ``max_market_cap_usd``]
  (default $50k-$50M).
* Pair age >= ``min_age_hours`` (default 6h) — a real, provable trading history.
* Recent momentum: up by >= ``min_recent_change_pct`` (default +20%) over the 1h
  OR 6h window — this is what keeps FLAT tokens out and surfaces actual movers.
* Optional cheap security gates: an explicit freeze authority or an explicitly
  unlocked (0%) LP excludes the candidate; an unknown signal never does.

OUTPUT
------
De-duplicated by mint (keeping the DEEPEST-liquidity pair), ranked by a blend of
24h volume and recent price change (:func:`_blend_score`), and truncated to
``max_candidates`` (default 15).

FAIL-CLOSED CONTRACT
--------------------
On ANY API error, timeout, rate-limit, or missing/invalid field the scanner
skips that candidate/source, and on a total fetch failure it returns an EMPTY
list. It never surfaces unvalidated or partial data: a candidate is emitted only
when every required field is present and valid.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

import httpx

import config
# Reuse the project's hardened read-only request helper (retry + backoff +
# Retry-After handling) for the metered Birdeye call.
from safety.rug_check import _request_json

logger = logging.getLogger(__name__)

# --- Data sources (read-only) -----------------------------------------------
CHAIN_ID: str = "solana"
DEFAULT_TIMEOUT: float = 8.0

# Dexscreener public "promoted / trending" boosts (no API key). Returns a JSON
# array of {chainId, tokenAddress, ...}.
DEXSCREENER_BOOSTS_TOP: str = "https://api.dexscreener.com/token-boosts/top/v1"
DEXSCREENER_BOOSTS_LATEST: str = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_BOOST_URLS: tuple = (DEXSCREENER_BOOSTS_TOP, DEXSCREENER_BOOSTS_LATEST)

# Dexscreener batched token lookup: append a comma-separated list of up to 30
# mints -> {"pairs": [...]} with full per-pair stats. Used to enrich every
# discovered address (from EITHER source) into a uniform candidate.
DEXSCREENER_TOKENS: str = "https://api.dexscreener.com/latest/dex/tokens"
DEXSCREENER_TOKENS_BATCH: int = 30

# Birdeye trending tokens (higher-quality Solana movers). Requires an API key.
BIRDEYE_TRENDING: str = "https://public-api.birdeye.so/defi/token_trending"
BIRDEYE_TRENDING_LIMIT: int = 20   # single ranked call; keep small to save CUs

# Established Solana AMMs we trust. Deliberately narrow: venues that require a
# real, deployed liquidity pool. ``pumpswap`` is the Pump.fun AMM that GRADUATED
# tokens trade on. Tunable via the ``allowed_dexes`` parameter.
ALLOWED_DEXES: frozenset = frozenset({"raydium", "orca", "meteora", "pumpswap"})

# Bonding-curve / pre-graduation launchpads. A token trading on one of these has
# NOT migrated to a real AMM pool yet (e.g. an un-graduated Pump.fun launch), so
# it is rejected regardless of ``allowed_dexes``. The graduated Pump.fun AMM
# (``pumpswap``) is judged on its pool like any other established pair.
UNGRADUATED_VENUES: frozenset = frozenset({"pumpfun", "pump", "moonshot"})

# --- Pre-filter thresholds (tunable) ----------------------------------------
# Defaults are intentionally strict to keep rug-like / flat tokens out.
MIN_MARKET_CAP_USD: float = 50_000.0
MAX_MARKET_CAP_USD: float = 50_000_000.0
MIN_LIQUIDITY_USD: float = 30_000.0    # deep enough to exit without wrecking price
MIN_VOLUME_24H_USD: float = 100_000.0  # real, sustained trading interest
MIN_AGE_HOURS: float = 6.0             # old enough for a provable trading history
MIN_RECENT_CHANGE_PCT: float = 20.0    # up >= this over 1h OR 6h -> an actual mover

# --- Output shaping (tunable) -----------------------------------------------
DEFAULT_MAX_CANDIDATES: int = 15       # hard cap on how many movers we surface
BLEND_VOLUME_WEIGHT: float = 1.0       # weight on log10(24h volume)
BLEND_MOMENTUM_WEIGHT: float = 0.05    # weight on recent price-change percent

# Async read-only source of raw (already-enriched) Dexscreener pair dicts,
# injectable for tests to bypass the network discovery/enrichment path.
PairFetcher = Callable[[], Awaitable[List[Dict[str, Any]]]]

# --- Security-signal field conventions --------------------------------------
# Dexscreener's public data rarely carries these, but enriched responses and
# some proxies do. We read them opportunistically and only act on an explicitly
# bad value; a missing signal is treated as "unknown", never as a pass/fail.
_FREEZE_BOOL_KEYS: tuple = ("isFreezable", "freezable", "hasFreezeAuthority", "freezeAuthorityEnabled")
_FREEZE_ADDR_KEYS: tuple = ("freezeAuthority",)
_LOCK_PCT_KEYS: tuple = (
    "lockedLiquidityPct", "lpLockedPct", "lockedPct", "liquidityLockedPct", "lockedLpPct",
)
# Address values that mean "no authority set" (freeze authority revoked/absent).
_NULL_AUTHORITIES: frozenset = frozenset(
    {"", "none", "null", "0", "false", "11111111111111111111111111111111"}
)


def _as_float_or_zero(value: Any) -> float:
    """Coerce to float, or 0.0 if missing/non-numeric (fail-closed for a
    signed price-change field: unknown momentum reads as no momentum)."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _chunked(items: List[str], size: int):
    """Yield successive ``size``-length chunks of ``items``."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _dedupe_preserve(items: List[str]) -> List[str]:
    """De-duplicate ``items`` preserving first-seen order."""
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _security_sources(pair: Dict[str, Any]):
    """Yield the sub-dicts of a raw pair that may carry security signals."""
    yield pair
    for key in ("baseToken", "info", "security", "audit", "liquidity"):
        sub = pair.get(key)
        if isinstance(sub, dict):
            yield sub


def _extract_security(pair: Dict[str, Any]) -> tuple[Optional[bool], Optional[float]]:
    """Best-effort read of (freeze_authority, locked_lp_pct) from a raw pair.

    Returns ``None`` for either signal when it is not cheaply present. Never
    raises: any malformed value is treated as "unknown" for that signal so the
    optional enrichment can't break the read-only scan.
    """
    freeze: Optional[bool] = None
    locked_lp_pct: Optional[float] = None

    for src in _security_sources(pair):
        if freeze is None:
            for key in _FREEZE_BOOL_KEYS:
                val = src.get(key)
                if val is not None:
                    freeze = bool(val)  # True => freezable/authority present => risky
                    break
        if freeze is None:
            for key in _FREEZE_ADDR_KEYS:
                val = src.get(key)
                if val is not None:
                    # A real address => authority present (risky); a null/system
                    # address => authority revoked (safe).
                    freeze = str(val).strip().lower() not in _NULL_AUTHORITIES
                    break
        if locked_lp_pct is None:
            for key in _LOCK_PCT_KEYS:
                val = src.get(key)
                if val is not None:
                    try:
                        locked_lp_pct = float(val)
                    except (TypeError, ValueError):
                        locked_lp_pct = None
                    break

    return freeze, locked_lp_pct


def _parse_pair(pair: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
    """Validate one raw Dexscreener pair into a candidate dict, or ``None``.

    Returns ``None`` (fail-closed) if the pair is not on Solana or if ANY
    required field is missing/invalid — never a partial candidate. ``now`` is
    epoch seconds; ``pairCreatedAt`` is epoch milliseconds. Price-change fields
    are best-effort (default 0.0 => treated as "no momentum" by the filter).
    """
    try:
        if str(pair.get("chainId", "")).lower() != CHAIN_ID:
            return None

        base = pair.get("baseToken") or {}
        mint = base.get("address")
        if not mint:
            return None

        dex = str(pair.get("dexId", "")).lower()
        if not dex:
            return None

        # Liquidity (USD) and 24h volume are required and must be numeric.
        liquidity_usd = float((pair.get("liquidity") or {}).get("usd"))
        volume_24h_usd = float((pair.get("volume") or {}).get("h24"))

        # Market cap: prefer marketCap, fall back to fully-diluted valuation.
        mcap_raw = pair.get("marketCap")
        if mcap_raw is None:
            mcap_raw = pair.get("fdv")
        if mcap_raw is None:
            return None
        market_cap_usd = float(mcap_raw)

        # Age from pair creation time (ms). Required for the proven-run filter.
        created_ms = pair.get("pairCreatedAt")
        if created_ms is None:
            return None
        age_hours = (now * 1000.0 - float(created_ms)) / 3_600_000.0

        # Reject nonsensical numbers rather than surfacing them.
        if liquidity_usd < 0 or volume_24h_usd < 0 or market_cap_usd <= 0 or age_hours < 0:
            return None

        price_raw = pair.get("priceUsd")
        price_usd = float(price_raw) if price_raw is not None else 0.0

        # Recent momentum (percent). Best-effort: absent/invalid -> 0.0.
        change = pair.get("priceChange") or {}
        price_change_1h_pct = _as_float_or_zero(change.get("h1"))
        price_change_6h_pct = _as_float_or_zero(change.get("h6"))
        recent_change_pct = max(price_change_1h_pct, price_change_6h_pct)

        # Optional security enrichment: None when not cheaply present.
        freeze_authority, locked_lp_pct = _extract_security(pair)

        return {
            "mint": str(mint),
            "symbol": str(base.get("symbol") or "?"),
            "source_dex": dex,
            "market_cap_usd": market_cap_usd,
            "liquidity_usd": liquidity_usd,
            "volume_24h_usd": volume_24h_usd,
            "age_hours": age_hours,
            "price_usd": price_usd,
            "price_change_1h_pct": price_change_1h_pct,
            "price_change_6h_pct": price_change_6h_pct,
            "recent_change_pct": recent_change_pct,
            "pair_address": str(pair.get("pairAddress") or ""),
            # Populated by the network path; [] on the injected-fetcher path.
            "discovery_sources": [],
            # True => freeze authority present (risky); None => unknown.
            "freeze_authority": freeze_authority,
            # 0.0 => LP explicitly unlocked (risky); None => unknown.
            "locked_lp_pct": locked_lp_pct,
        }
    except (TypeError, ValueError):
        # Missing/null/non-numeric field -> fail closed for this candidate.
        return None


def _passes_filters(
    candidate: Dict[str, Any],
    *,
    allowed_dexes: frozenset,
    min_market_cap_usd: float,
    max_market_cap_usd: float,
    min_liquidity_usd: float,
    min_volume_24h_usd: float,
    min_age_hours: float,
    min_recent_change_pct: float,
    exclude_freeze_authority: bool,
    exclude_unlocked_lp: bool,
) -> bool:
    """Return True only if the candidate clears every pre-filter."""
    # Bonding-curve / un-graduated launchpad -> never a proven runner. Checked
    # first, and independently of ``allowed_dexes``, so widening the allow-list
    # can never let an un-graduated Pump.fun pair through.
    if candidate["source_dex"] in UNGRADUATED_VENUES:
        return False
    if candidate["source_dex"] not in allowed_dexes:
        return False
    if not (min_market_cap_usd <= candidate["market_cap_usd"] <= max_market_cap_usd):
        return False
    if candidate["liquidity_usd"] < min_liquidity_usd:
        return False
    if candidate["volume_24h_usd"] < min_volume_24h_usd:
        return False
    if candidate["age_hours"] < min_age_hours:
        return False
    # Momentum gate: must be UP by the threshold over the 1h OR 6h window. Flat
    # or falling tokens (recent_change_pct below the floor) are dropped.
    if candidate["recent_change_pct"] < min_recent_change_pct:
        return False
    # Optional security gates: only reject on an explicitly-bad signal; an
    # unknown (None) signal is not disqualifying.
    if exclude_freeze_authority and candidate.get("freeze_authority") is True:
        return False
    if exclude_unlocked_lp and candidate.get("locked_lp_pct") == 0.0:
        return False
    return True


def _blend_score(
    candidate: Dict[str, Any],
    *,
    volume_weight: float,
    momentum_weight: float,
) -> float:
    """Rank score blending 24h volume and recent momentum (higher = better).

    ``volume_weight * log10(volume) + momentum_weight * recent_change_pct``.
    Log-scaling volume keeps a whale-volume token from dwarfing a strong mover,
    while the momentum term rewards the fresher, faster run. Both weights are
    configurable so callers can lean the ranking toward liquidity or toward
    velocity.
    """
    volume = max(candidate["volume_24h_usd"], 1.0)
    return (
        volume_weight * math.log10(volume)
        + momentum_weight * candidate["recent_change_pct"]
    )


# --- Discovery + enrichment (read-only network) -----------------------------

async def _get_json(
    client: httpx.AsyncClient, url: str, *, timeout: float
) -> Any:
    """Read-only GET returning parsed JSON (list or dict). Raises on HTTP error;
    callers catch and fail closed."""
    resp = await client.get(
        url, headers={"Accept": "application/json"}, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


def _boost_addresses(data: Any) -> List[str]:
    """Extract Solana token addresses from a Dexscreener boosts payload (a JSON
    array of {chainId, tokenAddress})."""
    items = data if isinstance(data, list) else (data or {}).get("pairs") or []
    out: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("chainId", "")).lower() != CHAIN_ID:
            continue
        addr = item.get("tokenAddress") or item.get("address")
        if addr:
            out.append(str(addr))
    return out


def _birdeye_trending_addresses(data: Any) -> List[str]:
    """Extract token addresses from a Birdeye trending payload
    (``data.tokens[].address``)."""
    payload = (data or {}).get("data") or {}
    tokens = payload.get("tokens") or payload.get("items") or []
    out: List[str] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        addr = token.get("address") or token.get("mint")
        if addr:
            out.append(str(addr))
    return out


async def _discover_dexscreener(
    client: httpx.AsyncClient, *, timeout: float, boost_urls: tuple
) -> List[str]:
    """Discover Solana mover addresses from the Dexscreener boosts endpoints.

    Each endpoint is best-effort: one failing does not sink the other, and a
    total failure yields an empty list (the caller still has the Birdeye source).
    """
    mints: List[str] = []
    for url in boost_urls:
        try:
            data = await _get_json(client, url, timeout=timeout)
            mints.extend(_boost_addresses(data))
        except Exception as exc:  # noqa: BLE001 — best-effort source
            logger.warning("[scanner] Dexscreener boosts %s failed: %s", url, exc)
    return mints


async def _discover_birdeye(
    client: httpx.AsyncClient, *, timeout: float, api_key: Optional[str], limit: int
) -> List[str]:
    """Discover Solana mover addresses from Birdeye trending (one ranked call).

    Returns ``[]`` (and makes NO request) when no API key is configured, or on
    any error — the scan continues on the Dexscreener source alone.
    """
    if not api_key:
        logger.info(
            "[scanner] no Birdeye API key (BIRDEYE_API_KEY) -> skipping Birdeye source"
        )
        return []
    url = str(
        httpx.URL(
            BIRDEYE_TRENDING,
            params={"sort_by": "rank", "sort_type": "asc", "offset": 0, "limit": limit},
        )
    )
    headers = {"X-API-KEY": api_key, "x-chain": CHAIN_ID, "accept": "application/json"}
    try:
        data = await _request_json(
            client, "GET", url, headers=headers, timeout=timeout, label="birdeye-trending"
        )
        return _birdeye_trending_addresses(data)
    except Exception as exc:  # noqa: BLE001 — best-effort source
        logger.warning("[scanner] Birdeye trending failed: %s", exc)
        return []


async def _enrich_via_dexscreener(
    client: httpx.AsyncClient, mints: List[str], *, timeout: float,
    batch_size: int = DEXSCREENER_TOKENS_BATCH,
) -> List[Dict[str, Any]]:
    """Batch-fetch full pair data for ``mints`` via the Dexscreener token lookup.

    Up to ``batch_size`` mints per request. Each batch is best-effort; a failed
    batch is skipped (fail-closed) rather than sinking the whole scan.
    """
    pairs: List[Dict[str, Any]] = []
    for chunk in _chunked(mints, batch_size):
        url = f"{DEXSCREENER_TOKENS}/{','.join(chunk)}"
        try:
            data = await _get_json(client, url, timeout=timeout)
            pairs.extend((data or {}).get("pairs") or [])
        except Exception as exc:  # noqa: BLE001 — skip this batch, keep the rest
            logger.warning("[scanner] Dexscreener enrich batch failed: %s", exc)
    return pairs


async def _discover_and_enrich(
    client: httpx.AsyncClient,
    *,
    timeout: float,
    boost_urls: tuple,
    birdeye_api_key: Optional[str],
    birdeye_limit: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    """Aggregate both mover sources, then enrich every address into raw pairs.

    Returns ``(raw_pairs, sources_by_mint)`` where ``sources_by_mint`` records
    which source(s) surfaced each mint (for diagnostics / candidate stamping).
    """
    dex_mints = await _discover_dexscreener(client, timeout=timeout, boost_urls=boost_urls)
    birdeye_mints = await _discover_birdeye(
        client, timeout=timeout, api_key=birdeye_api_key, limit=birdeye_limit
    )

    sources_by_mint: Dict[str, Set[str]] = {}
    for mint in dex_mints:
        sources_by_mint.setdefault(mint, set()).add("dexscreener")
    for mint in birdeye_mints:
        sources_by_mint.setdefault(mint, set()).add("birdeye")

    mints = _dedupe_preserve(dex_mints + birdeye_mints)
    logger.info(
        "[scanner] discovered %d mover address(es) (dexscreener=%d, birdeye=%d)",
        len(mints), len(dex_mints), len(birdeye_mints),
    )
    if not mints:
        return [], sources_by_mint

    raw_pairs = await _enrich_via_dexscreener(client, mints, timeout=timeout)
    return raw_pairs, sources_by_mint


async def scan_candidates(
    *,
    pair_fetcher: Optional[PairFetcher] = None,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    birdeye_api_key: Optional[str] = None,
    boost_urls: tuple = DEXSCREENER_BOOST_URLS,
    birdeye_limit: int = BIRDEYE_TRENDING_LIMIT,
    allowed_dexes: frozenset = ALLOWED_DEXES,
    min_market_cap_usd: float = MIN_MARKET_CAP_USD,
    max_market_cap_usd: float = MAX_MARKET_CAP_USD,
    min_liquidity_usd: float = MIN_LIQUIDITY_USD,
    min_volume_24h_usd: float = MIN_VOLUME_24H_USD,
    min_age_hours: float = MIN_AGE_HOURS,
    min_recent_change_pct: float = MIN_RECENT_CHANGE_PCT,
    exclude_freeze_authority: bool = True,
    exclude_unlocked_lp: bool = True,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    volume_weight: float = BLEND_VOLUME_WEIGHT,
    momentum_weight: float = BLEND_MOMENTUM_WEIGHT,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Discover and surface moving Solana candidates, best movers first.

    Read-only. Aggregates the Dexscreener boosts and Birdeye trending sources,
    enriches every discovered address through one batched Dexscreener lookup,
    validates + pre-filters each pair (Solana AMM, liquidity/volume/mcap/age and
    the >= ``min_recent_change_pct`` recent-momentum gate), de-duplicates by mint
    (keeping the deepest-liquidity pair), ranks by :func:`_blend_score`, and
    returns at most ``max_candidates``.

    ``birdeye_api_key`` defaults to ``config.BIRDEYE_API_KEY`` (read from .env);
    with no key the Birdeye source is silently skipped. Inject ``pair_fetcher``
    to supply already-enriched raw pairs and bypass the network entirely, or
    ``client`` to drive the real path against a mock transport. On ANY fetch
    failure the scanner returns an EMPTY list — never partial or unvalidated data.
    """
    now = time.time() if now is None else now
    api_key = config.BIRDEYE_API_KEY if birdeye_api_key is None else birdeye_api_key
    sources_by_mint: Dict[str, Set[str]] = {}

    # --- Fetch (fail-closed to empty list on any error) --------------------
    try:
        if pair_fetcher is not None:
            raw_pairs = await pair_fetcher()
        else:
            own_client = client is None
            if own_client:
                client = httpx.AsyncClient(timeout=timeout)
            try:
                raw_pairs, sources_by_mint = await _discover_and_enrich(
                    client,
                    timeout=timeout,
                    boost_urls=boost_urls,
                    birdeye_api_key=api_key,
                    birdeye_limit=birdeye_limit,
                )
            finally:
                if own_client:
                    await client.aclose()
    except Exception as exc:  # noqa: BLE001 — read-only scan never raises out
        logger.warning("[scanner] fetch failed: %s -> empty candidate list", exc)
        return []

    # --- Validate + pre-filter ---------------------------------------------
    candidates: List[Dict[str, Any]] = []
    for pair in raw_pairs:
        parsed = _parse_pair(pair, now)
        if parsed is None:
            continue
        if not _passes_filters(
            parsed,
            allowed_dexes=allowed_dexes,
            min_market_cap_usd=min_market_cap_usd,
            max_market_cap_usd=max_market_cap_usd,
            min_liquidity_usd=min_liquidity_usd,
            min_volume_24h_usd=min_volume_24h_usd,
            min_age_hours=min_age_hours,
            min_recent_change_pct=min_recent_change_pct,
            exclude_freeze_authority=exclude_freeze_authority,
            exclude_unlocked_lp=exclude_unlocked_lp,
        ):
            continue
        candidates.append(parsed)

    # --- De-duplicate by mint, keeping the DEEPEST-liquidity pair ----------
    by_mint: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        mint = candidate["mint"]
        incumbent = by_mint.get(mint)
        if incumbent is None or candidate["liquidity_usd"] > incumbent["liquidity_usd"]:
            by_mint[mint] = candidate

    unique = list(by_mint.values())
    for candidate in unique:
        candidate["discovery_sources"] = sorted(sources_by_mint.get(candidate["mint"], set()))

    # --- Rank by the volume/momentum blend, then cap -----------------------
    unique.sort(
        key=lambda c: _blend_score(
            c, volume_weight=volume_weight, momentum_weight=momentum_weight
        ),
        reverse=True,
    )
    surfaced = unique[: max(0, max_candidates)]

    logger.info(
        "[scanner] surfaced %d mover(s) (cap=%d) from %d raw pair(s)",
        len(surfaced), max_candidates, len(raw_pairs),
    )
    return surfaced


# --- CLI: `python -m scanner.token_scanner` ---------------------------------

def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    """Human-readable rendering of the candidate list for the CLI."""
    if not candidates:
        return "No candidates surfaced."
    lines = [f"Surfaced {len(candidates)} mover(s):"]
    for c in candidates:
        lock = c.get("locked_lp_pct")
        lock_str = "lp=?" if lock is None else f"lp={lock:.0f}%"
        src = ",".join(c.get("discovery_sources") or []) or "?"
        lines.append(
            f"  {c['symbol']:<8} {c['mint']}  dex={c['source_dex']:<9} "
            f"mcap=${c['market_cap_usd']:,.0f}  liq=${c['liquidity_usd']:,.0f}  "
            f"vol24h=${c['volume_24h_usd']:,.0f}  chg={c['recent_change_pct']:+.0f}%  "
            f"age={c['age_hours']:.1f}h  {lock_str}  src={src}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Read-only: prints the candidate list, never trades."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m scanner.token_scanner",
        description="Read-only scanner for moving Solana candidates.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout (s)."
    )
    parser.add_argument(
        "--min-liquidity", type=float, default=MIN_LIQUIDITY_USD,
        help="Minimum pool liquidity in USD (default: %(default)s).",
    )
    parser.add_argument(
        "--min-volume", type=float, default=MIN_VOLUME_24H_USD,
        help="Minimum 24h volume in USD (default: %(default)s).",
    )
    parser.add_argument(
        "--min-age-hours", type=float, default=MIN_AGE_HOURS,
        help="Minimum pair age in hours (default: %(default)s).",
    )
    parser.add_argument(
        "--min-momentum", type=float, default=MIN_RECENT_CHANGE_PCT,
        help="Minimum recent price change %% over 1h/6h (default: %(default)s).",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES,
        help="Maximum movers to surface (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    candidates = asyncio.run(
        scan_candidates(
            timeout=args.timeout,
            min_liquidity_usd=args.min_liquidity,
            min_volume_24h_usd=args.min_volume,
            min_age_hours=args.min_age_hours,
            min_recent_change_pct=args.min_momentum,
            max_candidates=args.max_candidates,
        )
    )
    print(_format_candidates(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
