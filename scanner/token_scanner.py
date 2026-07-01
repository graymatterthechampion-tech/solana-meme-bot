"""Read-only candidate token scanner (Dexscreener).

DISCOVERY ONLY. This module's sole job is to discover and surface candidate
token mint addresses for the existing safety -> entry pipeline to evaluate. It
performs read-only HTTP GETs and pure filtering and returns a list of candidate
dicts. It NEVER buys, sells, signs, or broadcasts, and imports no signer,
keypair, or transaction-building code.

STRATEGY ALIGNMENT
------------------
The entry strategy (``strategies/entry.py``) is a *post-pump dip buy* on a
PROVEN RUNNER. So the scanner surfaces tokens with established liquidity and
trading history — NOT brand-new launches. It pre-filters on a market-cap band,
minimum liquidity, minimum 24h volume, and a minimum pair age before surfacing
anything, so brand-new / illiquid pairs never reach the pipeline.

To keep rug-like tokens out of the pipeline the defaults are deliberately
strict: only established AMMs (Raydium / Orca / Meteora) are trusted, un-graduated
Pump.fun bonding-curve pairs are rejected outright, and any pair carrying a
freeze-authority flag or an explicitly-unlocked (0%) LP is dropped when that
signal is cheaply present in the pair data. Every threshold is a keyword
argument so callers can loosen or tighten the gate without editing this module.

FAIL-CLOSED CONTRACT
--------------------
On ANY API error, timeout, rate-limit, or missing/invalid field the scanner
skips that candidate, and on a total fetch failure it returns an EMPTY list. It
never surfaces unvalidated or partial data: a candidate is emitted only when
every required field is present and valid. Security signals (freeze authority,
LP lock) are best-effort enrichment: absent = "unknown" and non-disqualifying;
only an explicitly-bad value excludes the candidate.

DATA SOURCE (read-only, no API key)
-----------------------------------
Dexscreener public search endpoint via httpx GET. Results are filtered to
Solana pairs on established DEXs (Raydium, Orca, Meteora), de-duplicated by
mint, and sorted by 24h volume.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# --- Data source (read-only, public; no API key) ----------------------------
DEXSCREENER_SEARCH: str = "https://api.dexscreener.com/latest/dex/search"
CHAIN_ID: str = "solana"
DEFAULT_TIMEOUT: float = 8.0

# Search terms used to surface active Solana pairs. Configurable; the results
# are always re-filtered to Solana + allowed DEXs in code.
DEFAULT_QUERIES: tuple = ("SOL/USDC", "SOL")

# Established Solana AMMs we trust for a "proven runner". Deliberately narrow:
# only venues that require a real, deployed liquidity pool. Brand-new/obscure or
# bonding-curve venues are excluded. Tunable via the ``allowed_dexes`` parameter.
ALLOWED_DEXES: frozenset = frozenset({"raydium", "orca", "meteora"})

# Bonding-curve / pre-graduation launchpads. A token trading on one of these has
# NOT migrated to a real AMM pool yet (e.g. an un-graduated Pump.fun launch), so
# it is rejected regardless of ``allowed_dexes``. A GRADUATED Pump.fun token
# trades on Raydium (dexId "raydium"), so it is judged on its pool like any
# other established pair — only the bonding-curve stage is excluded here.
UNGRADUATED_VENUES: frozenset = frozenset({"pumpfun", "pump", "moonshot"})

# --- Pre-filter thresholds (tunable) ----------------------------------------
# Defaults are intentionally strict to keep rug-like tokens out of the pipeline.
MIN_MARKET_CAP_USD: float = 50_000.0
MAX_MARKET_CAP_USD: float = 50_000_000.0
MIN_LIQUIDITY_USD: float = 30_000.0   # deep enough to exit without wrecking price
MIN_VOLUME_24H_USD: float = 100_000.0  # real, sustained trading interest
MIN_AGE_HOURS: float = 6.0   # old enough to have a real, provable trading history

# Async read-only source of raw Dexscreener pair dicts (injectable for tests).
PairFetcher = Callable[[], Awaitable[List[Dict[str, Any]]]]

# --- Security-signal field conventions --------------------------------------
# Dexscreener's public search rarely carries these, but enriched responses and
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
    epoch seconds; ``pairCreatedAt`` is epoch milliseconds.
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
            "pair_address": str(pair.get("pairAddress") or ""),
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
    # Optional security gates: only reject on an explicitly-bad signal; an
    # unknown (None) signal is not disqualifying.
    if exclude_freeze_authority and candidate.get("freeze_authority") is True:
        return False
    if exclude_unlocked_lp and candidate.get("locked_lp_pct") == 0.0:
        return False
    return True


async def _fetch_dexscreener_pairs(
    client: httpx.AsyncClient, queries: tuple, *, timeout: float
) -> List[Dict[str, Any]]:
    """Read-only GET of Dexscreener search results for each query, merged.

    Raises on transport/HTTP error; the caller (:func:`scan_candidates`) wraps
    this and fails closed to an empty list.
    """
    pairs: List[Dict[str, Any]] = []
    for query in queries:
        resp = await client.get(
            DEXSCREENER_SEARCH,
            params={"q": query},
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        pairs.extend(data.get("pairs") or [])
    return pairs


async def scan_candidates(
    *,
    pair_fetcher: Optional[PairFetcher] = None,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    queries: tuple = DEFAULT_QUERIES,
    allowed_dexes: frozenset = ALLOWED_DEXES,
    min_market_cap_usd: float = MIN_MARKET_CAP_USD,
    max_market_cap_usd: float = MAX_MARKET_CAP_USD,
    min_liquidity_usd: float = MIN_LIQUIDITY_USD,
    min_volume_24h_usd: float = MIN_VOLUME_24H_USD,
    min_age_hours: float = MIN_AGE_HOURS,
    exclude_freeze_authority: bool = True,
    exclude_unlocked_lp: bool = True,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Discover and surface candidate tokens, sorted by 24h volume (desc).

    Read-only. Fetches raw Dexscreener pairs (via ``pair_fetcher`` if injected,
    else a short-lived httpx client), validates + pre-filters each one, and
    de-duplicates by mint (highest-volume pair wins). On ANY fetch failure it
    returns an EMPTY list — never partial or unvalidated data.
    """
    now = time.time() if now is None else now

    # --- Fetch (fail-closed to empty list on any error) --------------------
    try:
        if pair_fetcher is not None:
            raw_pairs = await pair_fetcher()
        else:
            own_client = client is None
            if own_client:
                client = httpx.AsyncClient(timeout=timeout)
            try:
                raw_pairs = await _fetch_dexscreener_pairs(
                    client, queries, timeout=timeout
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
            exclude_freeze_authority=exclude_freeze_authority,
            exclude_unlocked_lp=exclude_unlocked_lp,
        ):
            continue
        candidates.append(parsed)

    # --- Sort by volume, then de-duplicate by mint (highest volume wins) ---
    candidates.sort(key=lambda c: c["volume_24h_usd"], reverse=True)
    seen: set = set()
    unique: List[Dict[str, Any]] = []
    for candidate in candidates:
        if candidate["mint"] in seen:
            continue
        seen.add(candidate["mint"])
        unique.append(candidate)

    logger.info(
        "[scanner] surfaced %d candidate(s) from %d raw pair(s)",
        len(unique),
        len(raw_pairs),
    )
    return unique


# --- CLI: `python -m scanner.token_scanner` ---------------------------------

def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    """Human-readable rendering of the candidate list for the CLI."""
    if not candidates:
        return "No candidates surfaced."
    lines = [f"Surfaced {len(candidates)} candidate(s):"]
    for c in candidates:
        lock = c.get("locked_lp_pct")
        lock_str = "lp=?" if lock is None else f"lp={lock:.0f}%"
        lines.append(
            f"  {c['symbol']:<8} {c['mint']}  dex={c['source_dex']:<9} "
            f"mcap=${c['market_cap_usd']:,.0f}  liq=${c['liquidity_usd']:,.0f}  "
            f"vol24h=${c['volume_24h_usd']:,.0f}  age={c['age_hours']:.1f}h  {lock_str}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Read-only: prints the candidate list, never trades."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m scanner.token_scanner",
        description="Read-only Dexscreener candidate token scanner.",
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
    args = parser.parse_args(argv)

    candidates = asyncio.run(
        scan_candidates(
            timeout=args.timeout,
            min_liquidity_usd=args.min_liquidity,
            min_volume_24h_usd=args.min_volume,
            min_age_hours=args.min_age_hours,
        )
    )
    print(_format_candidates(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
