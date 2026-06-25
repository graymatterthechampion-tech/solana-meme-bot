"""Read-only token safety / rug checks.

INSPECT-AND-REPORT ONLY. Nothing in this module buys, sells, signs, or
broadcasts. It performs read-only data gathering (HTTP GET / RPC reads) and
pure evaluation, returning a :class:`SafetyReport`. It deliberately imports no
signer, keypair, or transaction-building code.

FAIL-CLOSED CONTRACT
--------------------
Safety is asserted, never assumed. On any error, timeout, or missing field the
report is returned with ``passed=False`` and a conservative (unsafe) value for
every field, plus a reason. A token is never reported safe on incomplete data.

TODO(real-apis): wire read-only providers in ``_gather_token_data``:
    * Helius RPC (``config.RPC_URL`` / ``config.HELIUS_API_KEY``) —
      ``getTokenLargestAccounts``, account funding history for clustering.
    * RugCheck / GoPlus token-security — LP lock/burn status, buy/sell tax,
      mint/freeze authority, honeypot sellability.
    * Birdeye / Dexscreener — 24h volume, market cap, price action for the
      farmed-volume heuristic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# --- Thresholds (tunable) ---------------------------------------------------
MAX_TAX_PCT: float = 1.0          # fail if buy/sell tax >= 1%
MAX_TOP10_PCT: float = 15.0       # fail if top-10 holders own >= 15%
FARMED_VOL_RATIO: float = 1.0     # 24h volume >= market cap is suspicious...
FARMED_FLAT_PRICE_PCT: float = 5.0  # ...when 24h price moved <= 5% (flat)
DEFAULT_TIMEOUT: float = 5.0      # read-only request timeout (seconds)

_PLACEHOLDER_BASE: str = "https://example.invalid/safety"

# Addresses excluded from holder-concentration math: they are not "whales".
# TODO(addresses): expand with CEX hot wallets (Binance/Coinbase/OKX/Bybit),
# Raydium/Orca pool authorities, and the pump.fun bonding-curve program.
KNOWN_EXCLUDED_ADDRESSES: frozenset = frozenset(
    {
        "1nc1nerator11111111111111111111111111111111",  # SPL burn address
    }
)

# Async read-only data source: mint_address -> raw payload dict.
DataGatherer = Callable[[str], Awaitable[Dict[str, Any]]]


@dataclass
class SafetyReport:
    """Read-only assessment of a single token. Advisory output only."""

    mint_address: str
    lp_locked_or_burned: bool
    lp_lock_detail: str
    tax_pct: float
    top10_holder_pct: float
    holder_concentration_pass: bool
    funding_source_clustered: bool
    farmed_volume_flag: bool
    volume_to_mcap_ratio: float
    passed: bool
    reasons: List[str] = field(default_factory=list)


def _fail_closed_report(mint_address: str, reason: str) -> SafetyReport:
    """Build a conservative (unsafe) report when data can't be trusted."""
    return SafetyReport(
        mint_address=mint_address,
        lp_locked_or_burned=False,
        lp_lock_detail="unknown (data unavailable)",
        tax_pct=100.0,                  # assume worst case
        top10_holder_pct=100.0,         # assume worst case
        holder_concentration_pass=False,
        funding_source_clustered=True,  # assume risky
        farmed_volume_flag=True,        # assume risky
        volume_to_mcap_ratio=0.0,
        passed=False,
        reasons=[reason],
    )


def _require(raw: Dict[str, Any], key: str) -> Any:
    """Return ``raw[key]`` or raise ValueError if missing/null (fail-closed)."""
    value = raw.get(key)
    if value is None:
        raise ValueError(f"missing field: {key}")
    return value


def _top_holders(holders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the top-10 holders by percentage, excluding known non-whale
    addresses (exchanges, bonding curve, burn)."""
    considered = [
        h for h in holders if h.get("address") not in KNOWN_EXCLUDED_ADDRESSES
    ]
    considered.sort(key=lambda h: float(h["pct"]), reverse=True)
    return considered[:10]


def _evaluate(mint_address: str, raw: Dict[str, Any]) -> SafetyReport:
    """Defensively evaluate a raw payload into a SafetyReport.

    Each check is computed in isolation: if the data it needs is missing or
    malformed, ONLY that check is marked failed (fail-closed) with a reason —
    the other checks still run. ``passed`` is True only if EVERY check passes
    (no failed hard check and no risk flag set).
    """
    reasons: List[str] = []

    # 1. Liquidity: LP must be locked or burned.
    try:
        lp_locked_or_burned = bool(_require(raw, "lp_locked_or_burned"))
        lp_lock_detail = str(raw.get("lp_lock_detail") or "no detail provided")
        if not lp_locked_or_burned:
            reasons.append(f"LP not locked or burned ({lp_lock_detail})")
    except Exception as exc:  # noqa: BLE001 — fail-closed per check
        lp_locked_or_burned = False
        lp_lock_detail = "unknown (data unavailable)"
        reasons.append(f"liquidity check failed: {exc}")

    # 2. Tax: fail if buy OR sell tax >= 1%.
    try:
        tax_pct = max(
            float(_require(raw, "buy_tax_pct")), float(_require(raw, "sell_tax_pct"))
        )
        tax_pass = tax_pct < MAX_TAX_PCT
        if not tax_pass:
            reasons.append(f"tax too high: {tax_pct:.2f}% (>= {MAX_TAX_PCT:.0f}%)")
    except Exception as exc:  # noqa: BLE001
        tax_pct = 100.0
        tax_pass = False
        reasons.append(f"tax check failed: {exc}")

    # 3 & 4. Holder concentration + funding clustering (share the holder data).
    try:
        top10 = _top_holders(list(_require(raw, "holders")))
        top10_holder_pct = sum(float(h["pct"]) for h in top10)
        holder_concentration_pass = top10_holder_pct < MAX_TOP10_PCT
        if not holder_concentration_pass:
            reasons.append(
                f"top-10 holders own {top10_holder_pct:.2f}% (>= {MAX_TOP10_PCT:.0f}%)"
            )

        funders: Dict[str, List[str]] = {}
        for h in top10:
            src = h.get("funded_by")
            if src:
                funders.setdefault(str(src), []).append(str(h.get("address")))
        funding_source_clustered = any(len(addrs) >= 2 for addrs in funders.values())
        if funding_source_clustered:
            reasons.append("multiple top holders funded from the same source wallet")
    except Exception as exc:  # noqa: BLE001
        top10_holder_pct = 100.0
        holder_concentration_pass = False
        funding_source_clustered = True
        reasons.append(f"holder/funding check failed: {exc}")

    # 5 & 6. Farmed-volume heuristic + volume-to-market-cap ratio.
    try:
        volume = float(_require(raw, "volume_24h_usd"))
        market_cap = float(_require(raw, "market_cap_usd"))
        volume_to_mcap_ratio = volume / market_cap if market_cap > 0 else 0.0
        price_change = abs(float(_require(raw, "price_change_24h_pct")))
        farmed_volume_flag = (
            volume_to_mcap_ratio >= FARMED_VOL_RATIO
            and price_change <= FARMED_FLAT_PRICE_PCT
        )
        if farmed_volume_flag:
            reasons.append(
                f"farmed volume suspected: vol/mcap={volume_to_mcap_ratio:.2f} "
                f"with only {price_change:.1f}% price move"
            )
    except Exception as exc:  # noqa: BLE001
        volume_to_mcap_ratio = 0.0
        farmed_volume_flag = True
        reasons.append(f"volume check failed: {exc}")

    # passed only if EVERY check passes: hard checks AND no risk flags.
    passed = (
        lp_locked_or_burned
        and tax_pass
        and holder_concentration_pass
        and not funding_source_clustered
        and not farmed_volume_flag
    )

    return SafetyReport(
        mint_address=mint_address,
        lp_locked_or_burned=lp_locked_or_burned,
        lp_lock_detail=lp_lock_detail,
        tax_pct=tax_pct,
        top10_holder_pct=top10_holder_pct,
        holder_concentration_pass=holder_concentration_pass,
        funding_source_clustered=funding_source_clustered,
        farmed_volume_flag=farmed_volume_flag,
        volume_to_mcap_ratio=volume_to_mcap_ratio,
        passed=passed,
        reasons=reasons,
    )


async def _gather_token_data(
    mint_address: str,
    client: httpx.AsyncClient,
    *,
    timeout: float,
) -> Dict[str, Any]:
    """Read-only fetch of the raw safety payload for a mint.

    TODO(real-apis): placeholder. Replace with the read-only providers listed
    in the module docstring and merge their fields. Uses only HTTP GET.
    """
    headers: Dict[str, str] = {}
    if config.HELIUS_API_KEY:
        headers["Authorization"] = f"Bearer {config.HELIUS_API_KEY}"

    url = f"{_PLACEHOLDER_BASE}/token/{mint_address}"
    resp = await client.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data: Dict[str, Any] = resp.json()
    return data


async def evaluate_token_safety(
    mint_address: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    data_gatherer: Optional[DataGatherer] = None,
) -> SafetyReport:
    """Inspect a token and return a read-only :class:`SafetyReport`.

    This NEVER trades, signs, or broadcasts — it only reads and reports. On any
    timeout, transport/HTTP error, or missing/invalid field it fails closed:
    ``passed=False`` with conservative field values and an explanatory reason.

    ``data_gatherer`` (keyword-only) injects a read-only async data source for
    testing or for the no-network mock; when omitted, a short-lived httpx
    client performs the read.
    """
    try:
        if data_gatherer is not None:
            raw = await data_gatherer(mint_address)
        else:
            own_client = client is None
            if own_client:
                client = httpx.AsyncClient(timeout=timeout)
            try:
                raw = await _gather_token_data(mint_address, client, timeout=timeout)
            finally:
                if own_client:
                    await client.aclose()

        return _evaluate(mint_address, raw)
    except Exception as exc:  # noqa: BLE001 — total fail-closed: any error => UNSAFE
        # A safety gate must never raise into the trading loop and must never
        # report "safe" by accident. Any error at all -> conservative UNSAFE.
        logger.warning(
            "[rug_check] fail-closed for %s: %s -> reported UNSAFE", mint_address, exc
        )
        return _fail_closed_report(mint_address, f"safety data unavailable: {exc}")


async def mock_safe_token_data(mint_address: str) -> Dict[str, Any]:
    """No-network read-only stub returning a clean, passing token payload.

    Lets the safety check run end-to-end before real providers are wired.
    """
    return {
        "lp_locked_or_burned": True,
        "lp_lock_detail": "LP burned (100%)",
        "buy_tax_pct": 0.0,
        "sell_tax_pct": 0.0,
        "holders": [
            {"address": "Whale1", "pct": 3.0, "funded_by": "srcA"},
            {"address": "Whale2", "pct": 2.5, "funded_by": "srcB"},
            {"address": "Whale3", "pct": 2.0, "funded_by": "srcC"},
            {"address": "Whale4", "pct": 1.5, "funded_by": "srcD"},
            {"address": "Whale5", "pct": 1.0, "funded_by": "srcE"},
        ],
        "volume_24h_usd": 40_000.0,
        "market_cap_usd": 500_000.0,
        "price_change_24h_pct": 22.0,
    }
