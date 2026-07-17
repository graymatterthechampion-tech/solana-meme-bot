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

DATA SOURCES (read-only)
------------------------
``_gather_token_data`` pulls from public providers via httpx (no keys except
the optional Helius RPC from .env):
    * RugCheck public API — rugged flag, risk score/flags, LP lock/burn,
      transfer tax, holder distribution (knownAccounts-aware), market cap.
    * Dexscreener token API — 24h volume / price change for farmed-volume.
    * Helius RPC (``config.RPC_URL`` or ``config.HELIUS_API_KEY`` from .env) —
      holder-distribution fallback when RugCheck topHolders are absent.
Each provider is fetched independently and defensively: if one call errors,
times out, is rate-limited, or omits a field, only the checks that depend on it
fail closed — the others still evaluate.

TODO(real-apis): funding-source clustering still needs per-holder funding
history (e.g. Helius enhanced tx history); ``funded_by`` is left ``None`` for
now, so that signal is not yet evidenced.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# --- Thresholds (tunable) ---------------------------------------------------
MAX_TAX_PCT: float = 1.0          # fail if buy/sell tax >= 1%
# Fail the holder-concentration check when the top-10 non-pool holders own at
# least this FRACTION of supply. Deliberately calibrated to 0.30 (30%) for the
# Solana meme-coin market, where moderate holder concentration is normal — tune
# as needed. Only this bar is relaxed; every other safety check stays strict.
MAX_TOP10_HOLDER_PCT: float = 0.30  # fraction of supply (0.30 = 30%)
FARMED_VOL_RATIO: float = 1.0     # 24h volume >= market cap is suspicious...
FARMED_FLAT_PRICE_PCT: float = 5.0  # ...when 24h price moved <= 5% (flat)
LP_LOCK_MIN_PCT: float = 95.0     # treat LP as locked/burned at >= 95%
MAX_RISK_SCORE: float = 40.0      # fail if RugCheck score_normalised exceeds this (lower = safer)
DEFAULT_TIMEOUT: float = 8.0      # read-only request timeout (seconds)

# RugCheck risk levels considered high-severity (any one fails the token).
HIGH_RISK_LEVELS: frozenset = frozenset({"danger", "high", "critical"})
# topHolders whose owner is one of these knownAccounts types (or the system
# address) are pools/lockers, not real whales — excluded from concentration.
KNOWN_OWNER_EXCLUDE_TYPES: frozenset = frozenset({"AMM", "LOCKER"})
SYSTEM_ADDRESS: str = "11111111111111111111111111111111"

# --- Read-only data providers -----------------------------------------------
RUGCHECK_BASE: str = "https://api.rugcheck.xyz/v1"
HELIUS_MAINNET_BASE: str = "https://mainnet.helius-rpc.com"
DEXSCREENER_BASE: str = "https://api.dexscreener.com/latest/dex"  # no API key

# Rate-limit / transient-error handling (module globals so tests can zero them).
RATE_LIMIT_MAX_RETRIES: int = 3
RATE_LIMIT_BACKOFF: float = 0.5   # base seconds; exponential per attempt
RETRY_AFTER_MAX: float = 10.0     # cap any server-provided Retry-After
ERROR_BODY_LOG_CHARS: int = 400   # truncate logged 4xx bodies to this length

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

    # 0a. Rugged: explicit top-level kill switch — fail immediately if true.
    try:
        rugged = bool(_require(raw, "rugged"))
        if rugged:
            reasons.append("token flagged as RUGGED by RugCheck")
    except Exception as exc:  # noqa: BLE001 — fail-closed per check
        rugged = True
        reasons.append(f"rugged check failed: {exc}")

    # 0b. Risk gate: normalised score + any high-severity risk flag.
    try:
        score = float(_require(raw, "score_normalised"))
        flags = list(_require(raw, "risk_flags"))
        score_ok = score <= MAX_RISK_SCORE
        if not score_ok:
            reasons.append(
                f"risk score {score:.1f} exceeds max {MAX_RISK_SCORE:.0f}"
            )
        if flags:
            reasons.append("high-severity risks: " + ", ".join(flags))
        risk_gate_pass = score_ok and not flags
    except Exception as exc:  # noqa: BLE001
        risk_gate_pass = False
        reasons.append(f"risk gate failed: {exc}")

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
        # ``top10_holder_pct`` is a PERCENT (sum of per-holder pct); the tunable
        # threshold is a fraction of supply, so compare against it * 100.
        max_top10_pct = MAX_TOP10_HOLDER_PCT * 100.0
        top10_holder_pct = sum(float(h["pct"]) for h in top10)
        holder_concentration_pass = top10_holder_pct < max_top10_pct
        if not holder_concentration_pass:
            reasons.append(
                f"top-10 holders own {top10_holder_pct:.2f}% (>= {max_top10_pct:.0f}%)"
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

    # passed only if EVERY check passes: not rugged, risk gate clear, hard
    # checks all green, and no risk flags.
    passed = (
        not rugged
        and risk_gate_pass
        and lp_locked_or_burned
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


# --- Read-only HTTP layer (rate-limit aware) --------------------------------

def _helius_rpc_url() -> Optional[str]:
    """Resolve the Helius RPC endpoint from .env via config, or None."""
    if config.RPC_URL:
        return config.RPC_URL
    if config.HELIUS_API_KEY:
        return f"{HELIUS_MAINNET_BASE}/?api-key={config.HELIUS_API_KEY}"
    return None


def _error_body(resp: httpx.Response) -> str:
    """Best-effort truncated response body, for diagnosing a provider rejection.

    Never raises: a body we cannot read must not mask the HTTP error itself.
    """
    try:
        body = resp.text
    except Exception:  # noqa: BLE001 — undecodable/streamed body
        return "<unreadable body>"
    body = " ".join(body.split())  # collapse newlines so it stays one log line
    if not body:
        return "<empty body>"
    if len(body) > ERROR_BODY_LOG_CHARS:
        return f"{body[:ERROR_BODY_LOG_CHARS]}... (truncated)"
    return body


def _retry_after_seconds(resp: httpx.Response) -> Optional[float]:
    """Parse a Retry-After header (seconds), capped, or None."""
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return min(float(raw), RETRY_AFTER_MAX)
    except (TypeError, ValueError):
        return None


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float,
    label: str,
) -> Dict[str, Any]:
    """Perform one read-only request with retry on 429/5xx/transient errors.

    Honors a server ``Retry-After`` on 429, otherwise exponential backoff.
    Raises on exhaustion; callers fail those checks closed.

    A 4xx is a permanent rejection (bad params, or an endpoint the API plan does
    not grant), so it is NOT retried — but the provider's explanation lives in
    the response BODY, which a bare ``raise_for_status`` throws away. We log the
    truncated body before raising; the raised ``HTTPStatusError`` still carries
    the full response so callers can branch on it. The URL is never logged: it
    can embed an ``api-key`` query param (Helius).
    """
    last_error: Optional[Exception] = None

    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            resp = await client.request(
                method, url, json=json, headers=headers, timeout=timeout
            )
        except httpx.HTTPError as exc:  # timeouts, connect errors, etc.
            last_error = exc
            await asyncio.sleep(RATE_LIMIT_BACKOFF * (2 ** attempt))
            continue

        if resp.status_code == 429:
            delay = _retry_after_seconds(resp)
            if delay is None:
                delay = RATE_LIMIT_BACKOFF * (2 ** attempt)
            logger.warning("[rug_check] %s rate-limited (429); retry in %.2fs", label, delay)
            last_error = httpx.HTTPStatusError(
                "429 Too Many Requests", request=resp.request, response=resp
            )
            await asyncio.sleep(delay)
            continue

        if resp.status_code >= 500:
            last_error = httpx.HTTPStatusError(
                f"{resp.status_code} server error", request=resp.request, response=resp
            )
            await asyncio.sleep(RATE_LIMIT_BACKOFF * (2 ** attempt))
            continue

        if resp.status_code >= 400:
            logger.warning(
                "[rug_check] %s HTTP %d -> %s",
                label,
                resp.status_code,
                _error_body(resp),
            )
        resp.raise_for_status()
        data: Dict[str, Any] = resp.json()
        return data

    raise last_error or RuntimeError(f"{label}: request failed after retries")


async def _rpc(
    client: httpx.AsyncClient,
    url: str,
    rpc_method: str,
    params: List[Any],
    *,
    timeout: float,
) -> Any:
    """Call a read-only Solana JSON-RPC method (Helius). Raises on RPC error."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": rpc_method, "params": params}
    data = await _request_json(
        client, "POST", url, json=payload, timeout=timeout, label=f"helius:{rpc_method}"
    )
    if "error" in data:
        raise RuntimeError(f"helius {rpc_method} error: {data['error']}")
    return data.get("result")


# --- Provider fetch + parse -------------------------------------------------

def _parse_rugcheck(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map a live RugCheck v1 report into raw-schema fields, defensively.

    Field paths (RugCheck read endpoints need no API key):
      * rugged       -> top-level boolean kill switch
      * risk gate    -> score_normalised (lower = safer) + risks[].level
      * LP lock      -> markets[0].lp.lpLockedPct / lpLockedUSD
      * tax          -> transferFee.pct (token-2022 fee; 0 when none)
      * holders      -> topHolders[] (pct, owner), excluding AMM/LOCKER/system
                        owners via the knownAccounts map (so the pool/locker is
                        not counted as a whale)
      * market cap   -> token.supply (ui) * price

    Any field that is absent/malformed is omitted so the dependent check fails
    closed downstream rather than producing a wrong "safe".
    """
    out: Dict[str, Any] = {}

    # Rugged: explicit top-level kill switch.
    try:
        if data.get("rugged") is not None:
            out["rugged"] = bool(data["rugged"])
    except Exception:  # noqa: BLE001
        logger.debug("rugcheck rugged parse failed", exc_info=True)

    # Risk gate: normalised score (lower = safer) + high-severity risk flags.
    try:
        if data.get("score_normalised") is not None:
            out["score_normalised"] = float(data["score_normalised"])
        risks = data.get("risks")
        if isinstance(risks, list):
            out["risk_flags"] = [
                str(r.get("name") or "unknown")
                for r in risks
                if str(r.get("level", "")).lower() in HIGH_RISK_LEVELS
            ]
    except Exception:  # noqa: BLE001
        logger.debug("rugcheck risk parse failed", exc_info=True)

    # LP lock / burn: first market's lp block. lpLockedPct of 100 == fully
    # locked/burned; lpLockedUSD enriches the human-readable detail.
    try:
        markets = data.get("markets") or []
        lp = (markets[0].get("lp") or {}) if markets else {}
        lp_pct_raw = lp.get("lpLockedPct")
        if lp_pct_raw is not None:
            lp_pct = float(lp_pct_raw)
            detail = f"LP locked/burned {lp_pct:.1f}% (RugCheck"
            locked_usd = lp.get("lpLockedUSD")
            if locked_usd is not None:
                detail += f", ${float(locked_usd):,.0f} locked"
            detail += ")"
            out["lp_locked_or_burned"] = lp_pct >= LP_LOCK_MIN_PCT
            out["lp_lock_detail"] = detail
    except Exception:  # noqa: BLE001
        logger.debug("rugcheck LP parse failed", exc_info=True)

    # Transfer tax: top-level transferFee.pct (token-2022 fee). When there is
    # no fee config (token_extensions.transferFeeConfig is null), tax is 0.
    try:
        transfer_fee = data.get("transferFee")
        if isinstance(transfer_fee, dict) and transfer_fee.get("pct") is not None:
            tax = float(transfer_fee["pct"])
            out["buy_tax_pct"] = tax
            out["sell_tax_pct"] = tax
        else:
            ext = data.get("token_extensions")
            if (
                isinstance(ext, dict)
                and "transferFeeConfig" in ext
                and ext["transferFeeConfig"] is None
            ):
                out["buy_tax_pct"] = 0.0
                out["sell_tax_pct"] = 0.0
    except Exception:  # noqa: BLE001
        logger.debug("rugcheck tax parse failed", exc_info=True)

    # Holder distribution: topHolders[], EXCLUDING pool/locker/system owners
    # (via knownAccounts) so the liquidity pool is not counted as a holder.
    try:
        top_holders = data.get("topHolders")
        if isinstance(top_holders, list):
            known = data.get("knownAccounts") or {}
            holders: List[Dict[str, Any]] = []
            for h in top_holders:
                owner = h.get("owner")
                label_type = str((known.get(owner) or {}).get("type", "")).upper()
                if owner == SYSTEM_ADDRESS or label_type in KNOWN_OWNER_EXCLUDE_TYPES:
                    continue
                holders.append(
                    {"address": owner, "pct": float(h.get("pct") or 0.0), "funded_by": None}
                )
            out["holders"] = holders
    except Exception:  # noqa: BLE001
        logger.debug("rugcheck holders parse failed", exc_info=True)

    # Market cap: ui-supply * price.
    try:
        token = data.get("token") or {}
        price = data.get("price")
        if price is not None and token.get("supply") is not None:
            ui_supply = float(token["supply"]) / (10 ** int(token.get("decimals", 0)))
            out["market_cap_usd"] = ui_supply * float(price)
    except Exception:  # noqa: BLE001
        logger.debug("rugcheck mcap parse failed", exc_info=True)

    # NOTE: 24h volume / price change are NOT in the RugCheck report — they come
    # from Dexscreener (see _fetch_dexscreener), merged in during gathering.
    return out


async def _fetch_rugcheck(
    client: httpx.AsyncClient, mint_address: str, *, timeout: float
) -> Dict[str, Any]:
    """Fetch + parse the RugCheck report. Returns {} on any failure."""
    url = f"{RUGCHECK_BASE}/tokens/{mint_address}/report"
    try:
        # RugCheck read endpoints are public — NO API key / auth header.
        data = await _request_json(
            client, "GET", url, headers={"Accept": "application/json"},
            timeout=timeout, label="rugcheck",
        )
    except Exception as exc:  # noqa: BLE001 — any error -> omit, fail-closed
        logger.warning("[rug_check] RugCheck fetch failed for %s: %s", mint_address, exc)
        return {}
    return _parse_rugcheck(data)


def _parse_dexscreener(data: Dict[str, Any]) -> Dict[str, Any]:
    """Map a Dexscreener token response into raw volume / price-change fields.

    Uses the most active pair (highest 24h volume). Absent/malformed fields are
    omitted so the farmed-volume check fails closed downstream.
    """
    out: Dict[str, Any] = {}
    try:
        pairs = data.get("pairs") or []
        if pairs:
            best = max(
                pairs,
                key=lambda p: float((p.get("volume") or {}).get("h24") or 0.0),
            )
            vol = (best.get("volume") or {}).get("h24")
            if vol is not None:
                out["volume_24h_usd"] = float(vol)
            change = (best.get("priceChange") or {}).get("h24")
            if change is not None:
                out["price_change_24h_pct"] = float(change)
    except Exception:  # noqa: BLE001
        logger.debug("dexscreener parse failed", exc_info=True)
    return out


async def _fetch_dexscreener(
    client: httpx.AsyncClient, mint_address: str, *, timeout: float
) -> Dict[str, Any]:
    """Fetch 24h volume / price change from Dexscreener. Returns {} on failure.

    Dexscreener's token endpoint is public — no API key. On any error, timeout,
    rate-limit, or missing field this returns {}, leaving the farmed-volume
    check to fail closed.
    """
    url = f"{DEXSCREENER_BASE}/tokens/{mint_address}"
    try:
        data = await _request_json(
            client, "GET", url, headers={"Accept": "application/json"},
            timeout=timeout, label="dexscreener",
        )
    except Exception as exc:  # noqa: BLE001 — any error -> omit, fail-closed
        logger.warning("[rug_check] Dexscreener fetch failed for %s: %s", mint_address, exc)
        return {}
    return _parse_dexscreener(data)


async def _fetch_helius_holders(
    client: httpx.AsyncClient, mint_address: str, *, timeout: float
) -> Dict[str, Any]:
    """Fetch holder distribution via Helius RPC. Returns {} on any failure.

    Uses getTokenLargestAccounts + getTokenSupply (both read-only). ``pct`` is
    each largest account's share of total supply. ``funded_by`` is left None
    (funding history is not fetched yet — see module TODO).
    """
    rpc_url = _helius_rpc_url()
    if not rpc_url:
        logger.warning("[rug_check] no Helius RPC configured (RPC_URL/HELIUS_API_KEY)")
        return {}
    try:
        largest = await _rpc(
            client, rpc_url, "getTokenLargestAccounts", [mint_address], timeout=timeout
        )
        supply = await _rpc(
            client, rpc_url, "getTokenSupply", [mint_address], timeout=timeout
        )
        accounts = (largest or {}).get("value") or []
        total_ui = float(((supply or {}).get("value") or {}).get("uiAmount") or 0.0)
        if total_ui <= 0:
            return {}
        holders = [
            {
                "address": acct.get("address"),
                "pct": float(acct.get("uiAmount") or 0.0) / total_ui * 100.0,
                "funded_by": None,
            }
            for acct in accounts
        ]
        return {"holders": holders}
    except Exception as exc:  # noqa: BLE001 — any error -> omit, fail-closed
        logger.warning("[rug_check] Helius holder fetch failed for %s: %s", mint_address, exc)
        return {}


async def _gather_token_data(
    mint_address: str,
    client: httpx.AsyncClient,
    *,
    timeout: float,
) -> Dict[str, Any]:
    """Read-only gather from RugCheck (primary) + Dexscreener (volume), with a
    Helius holder fallback.

    RugCheck supplies rugged/risk/LP/tax/holders/mcap. Dexscreener supplies 24h
    volume / price change (RugCheck has neither). Holders come from RugCheck's
    knownAccounts-aware topHolders; only if those are absent do we fall back to
    Helius getTokenLargestAccounts (lower fidelity — cannot label pools). Each
    provider is fetched independently and never raises here: a failed provider
    omits its fields so only the dependent checks fail closed.
    """
    raw = dict(await _fetch_rugcheck(client, mint_address, timeout=timeout))
    # 24h volume / price action (RugCheck does not provide it).
    raw.update(await _fetch_dexscreener(client, mint_address, timeout=timeout))
    # Holder fallback to Helius only if RugCheck topHolders were absent.
    if "holders" not in raw:
        raw.update(await _fetch_helius_holders(client, mint_address, timeout=timeout))
    return raw


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
        "rugged": False,
        "score_normalised": 5.0,   # well below MAX_RISK_SCORE
        "risk_flags": [],          # no high-severity risks
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


# --- CLI: `python -m safety.rug_check <mint_address>` ------------------------

def _format_report(report: SafetyReport) -> str:
    """Human-readable rendering of a SafetyReport for the CLI."""
    verdict = "PASS (safe)" if report.passed else "FAIL (unsafe)"
    lines = [
        f"Token:   {report.mint_address}",
        f"Verdict: {verdict}",
        f"  LP locked/burned : {report.lp_locked_or_burned}  ({report.lp_lock_detail})",
        f"  Max tax          : {report.tax_pct:.2f}%",
        f"  Top-10 holders   : {report.top10_holder_pct:.2f}%  "
        f"(pass={report.holder_concentration_pass})",
        f"  Funding clustered: {report.funding_source_clustered}",
        f"  Farmed volume    : {report.farmed_volume_flag}",
        f"  Volume / mcap    : {report.volume_to_mcap_ratio:.3f}",
    ]
    if report.reasons:
        lines.append("  Reasons:")
        lines.extend(f"    - {reason}" for reason in report.reasons)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Read-only: prints a SafetyReport, never trades.

    Exit code 0 if the token passes all checks, 1 otherwise (incl. fail-closed).
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m safety.rug_check",
        description="Read-only token safety / rug check (RugCheck + Helius).",
    )
    parser.add_argument("mint_address", help="SPL token mint address to inspect")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Per-request timeout (s)."
    )
    args = parser.parse_args(argv)

    report = asyncio.run(
        evaluate_token_safety(args.mint_address, timeout=args.timeout)
    )
    print(_format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
