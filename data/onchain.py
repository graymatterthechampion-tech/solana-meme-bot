"""Read-only Helius on-chain signal layer.

Derives on-chain signals from Helius Solana JSON-RPC using READ methods only.
Nothing here signs, builds, or sends a transaction; it imports no signer or
keypair code. The Helius endpoint (and key) come from ``.env`` via ``config``
(``RPC_URL`` / ``HELIUS_API_KEY``), resolved by ``safety.rug_check``.

SIGNALS
-------
1. RECENT FLOW (:func:`get_recent_flow`) — from ``getTokenLargestAccounts`` +
   ``getSignaturesForAddress`` + ``getTransaction`` it derives, over a bounded
   window of recent swaps:
     * ``largest_single_sell_usd`` — the biggest single wallet sell (USD), and
     * ``selling_wallets`` — distinct wallets that net-sold,
   which POPULATE the management-loop's coordinated-dump hard-exit trigger; plus
     * ``buyer_wallets`` — distinct wallets that net-bought (buyer breadth),
   an organic-participation meta signal.
   Swap direction is classified by the liquidity-pool vault's balance change
   (the largest token accounts, excluded from buyer/seller counts), so a buy's
   pool-side leg is never mistaken for a wallet sell.
2. KOL HOLDINGS (:func:`get_kol_holdings`) — via ``getTokenAccountsByOwner``,
   whether any curated wallet (``config.KOL_WALLETS``) currently holds the token.

FAIL-CLOSED CONTRACT
--------------------
Every function is read-only and NEVER raises. On any Helius error, timeout,
rate-limit, unconfigured endpoint, or malformed data the signal is simply
ABSENT (``available=False`` and zeroed / empty). By construction:
    * the dump signal can only ever ADD an exit, never mask one — an absent
      signal just leaves the coordinated-dump trigger dormant; and
    * the buyer/KOL meta signals absent = no boost, never a false block.
The retry/backoff + Retry-After path is reused from ``safety.rug_check``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

import config
# Reuse the hardened read-only RPC helper (retry/backoff/Retry-After) and the
# .env-driven endpoint resolution, exactly as the other read-only layers do.
from safety.rug_check import _helius_rpc_url, _rpc

logger = logging.getLogger(__name__)

# --- Bounds / defaults (tunable) --------------------------------------------
DEFAULT_TIMEOUT: float = 8.0
DEFAULT_MAX_SIGNATURES: int = 25    # recent signatures to scan (bounds CU spend)
DEFAULT_POOL_EXCLUDE: int = 2       # largest token accounts treated as pool vaults
DEFAULT_MAX_KOL_WALLETS: int = 25   # cap KOL lookups (one RPC call each)


@dataclass(frozen=True)
class OnchainFlow:
    """Recent-swap flow signals for one mint.

    All fields are benign zero when ``available`` is False (fail-closed), so a
    caller can use them unconditionally: absent data simply contributes no dump
    signal and no buyer breadth.
    """

    largest_single_sell_usd: float = 0.0
    selling_wallets: int = 0
    buyer_wallets: int = 0
    scanned_transactions: int = 0
    available: bool = False


@dataclass(frozen=True)
class KolSignal:
    """Whether curated wallets hold a token. Benign (no hold) when unavailable."""

    any_hold: bool = False
    holding_wallets: Tuple[str, ...] = ()
    available: bool = False


def _as_float(value: Any) -> Optional[float]:
    """Coerce to float, or None if missing/non-numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _account_key(keys: List[Any], index: int) -> Optional[str]:
    """Resolve the pubkey at ``index`` of a jsonParsed message ``accountKeys``
    list (entries may be ``{"pubkey": ...}`` objects or bare strings)."""
    if not (0 <= index < len(keys)):
        return None
    key = keys[index]
    if isinstance(key, dict):
        return key.get("pubkey")
    return str(key)


def _mint_balances_by_index(
    entries: List[Dict[str, Any]], mint: str
) -> Dict[int, float]:
    """Map accountIndex -> uiAmount for the target mint's token-balance entries."""
    out: Dict[int, float] = {}
    for entry in entries or []:
        if entry.get("mint") != mint:
            continue
        idx = entry.get("accountIndex")
        amount = _as_float((entry.get("uiTokenAmount") or {}).get("uiAmount"))
        if idx is None or amount is None:
            continue
        out[int(idx)] = amount
    return out


def _index_meta(
    entries: List[Dict[str, Any]], keys: List[Any], mint: str
) -> Dict[int, Dict[str, Optional[str]]]:
    """Map accountIndex -> {owner, addr} for the mint's token-balance entries."""
    meta: Dict[int, Dict[str, Optional[str]]] = {}
    for entry in entries or []:
        if entry.get("mint") != mint:
            continue
        idx = entry.get("accountIndex")
        if idx is None:
            continue
        meta.setdefault(
            int(idx), {"owner": entry.get("owner"), "addr": _account_key(keys, int(idx))}
        )
    return meta


def _parse_tx_flow(
    tx_result: Dict[str, Any], mint: str, pool_accounts: Set[str]
) -> Optional[Tuple[float, Dict[str, float]]]:
    """Parse one transaction into ``(pool_delta, owner_deltas)`` for ``mint``.

    ``pool_delta`` is the net token change across the excluded pool vault
    accounts (positive => tokens flowed INTO the pool => sell pressure).
    ``owner_deltas`` maps each non-pool owner to its net token change. Returns
    ``None`` for a failed/irrelevant transaction.
    """
    meta = tx_result.get("meta") or {}
    if meta.get("err") is not None:
        return None  # failed tx — ignore
    message = (tx_result.get("transaction") or {}).get("message") or {}
    keys = message.get("accountKeys") or []

    pre = meta.get("preTokenBalances") or []
    post = meta.get("postTokenBalances") or []
    pre_amounts = _mint_balances_by_index(pre, mint)
    post_amounts = _mint_balances_by_index(post, mint)
    if not pre_amounts and not post_amounts:
        return None

    index_meta: Dict[int, Dict[str, Optional[str]]] = {}
    index_meta.update(_index_meta(pre, keys, mint))
    for idx, info in _index_meta(post, keys, mint).items():
        index_meta.setdefault(idx, info)

    pool_delta = 0.0
    owner_deltas: Dict[str, float] = {}
    for idx in set(pre_amounts) | set(post_amounts):
        delta = post_amounts.get(idx, 0.0) - pre_amounts.get(idx, 0.0)
        info = index_meta.get(idx, {})
        if info.get("addr") in pool_accounts:
            pool_delta += delta
            continue
        owner = info.get("owner")
        if owner is None:
            continue
        owner_deltas[owner] = owner_deltas.get(owner, 0.0) + delta
    return pool_delta, owner_deltas


async def get_recent_flow(
    mint: str,
    *,
    price: float,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    rpc_url: Optional[str] = None,
    max_signatures: int = DEFAULT_MAX_SIGNATURES,
    pool_exclude: int = DEFAULT_POOL_EXCLUDE,
) -> OnchainFlow:
    """Derive recent-swap flow signals for ``mint`` (read-only, fail-closed).

    Steps (all read RPC): ``getTokenLargestAccounts`` identifies the pool
    vault(s) to exclude; ``getSignaturesForAddress`` lists recent signatures;
    each ``getTransaction`` is parsed for per-wallet token deltas classified by
    pool direction. ``largest_single_sell_usd`` is the biggest single wallet
    sell (tokens sold * ``price``); ``selling_wallets`` / ``buyer_wallets`` are
    the distinct net sellers / buyers over the window.

    Returns an ``available=False`` zeroed :class:`OnchainFlow` on any failure,
    missing endpoint, or a non-positive ``price`` — NEVER raises. The dump
    fields it feeds can only add an exit, so a dormant signal is always safe.
    """
    resolved = rpc_url if rpc_url is not None else _helius_rpc_url()
    if not resolved:
        logger.info("[onchain] no Helius RPC configured -> flow signal absent")
        return OnchainFlow()
    if price <= 0:
        logger.info("[onchain] non-positive price -> cannot value sells, flow absent")
        return OnchainFlow()

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        # Pool vaults = the largest token accounts; excluded so a swap's
        # pool-side leg is never counted as a wallet buy/sell. Required — without
        # it we cannot tell sells from the pool's own balance moves, so a failure
        # here means "no reliable flow" (fail-closed), not a guessed signal.
        largest = await _rpc(
            client, resolved, "getTokenLargestAccounts", [mint], timeout=timeout
        )
        pool_accounts: Set[str] = {
            str(a.get("address"))
            for a in ((largest or {}).get("value") or [])[:max(0, pool_exclude)]
            if a.get("address")
        }

        sigs = await _rpc(
            client, resolved, "getSignaturesForAddress",
            [mint, {"limit": max_signatures}], timeout=timeout,
        )
        signatures = [
            s.get("signature")
            for s in (sigs or [])
            if isinstance(s, dict) and s.get("signature") and s.get("err") is None
        ]

        largest_sell_usd = 0.0
        sellers: Set[str] = set()
        buyers: Set[str] = set()
        scanned = 0
        for signature in signatures:
            try:
                tx = await _rpc(
                    client, resolved, "getTransaction",
                    [signature, {"encoding": "jsonParsed",
                                 "maxSupportedTransactionVersion": 0}],
                    timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001 — skip one bad tx, keep scanning
                logger.debug("[onchain] getTransaction %s failed: %s", signature, exc)
                continue
            if not tx:
                continue
            parsed = _parse_tx_flow(tx, mint, pool_accounts)
            if parsed is None:
                continue
            scanned += 1
            pool_delta, owner_deltas = parsed
            if pool_delta > 0:  # tokens into the pool -> sell pressure
                for owner, delta in owner_deltas.items():
                    if delta < 0:
                        sellers.add(owner)
                        largest_sell_usd = max(largest_sell_usd, -delta * price)
            elif pool_delta < 0:  # tokens out of the pool -> buy pressure
                for owner, delta in owner_deltas.items():
                    if delta > 0:
                        buyers.add(owner)

        return OnchainFlow(
            largest_single_sell_usd=largest_sell_usd,
            selling_wallets=len(sellers),
            buyer_wallets=len(buyers),
            scanned_transactions=scanned,
            available=True,
        )
    except Exception as exc:  # noqa: BLE001 — total fail-closed: signal absent
        logger.warning("[onchain] recent-flow fetch failed for %s: %s", mint, exc)
        return OnchainFlow()
    finally:
        if own_client:
            await client.aclose()


def _owner_holds_token(accounts: List[Dict[str, Any]]) -> bool:
    """True if any jsonParsed token account in ``accounts`` has uiAmount > 0."""
    for acct in accounts or []:
        info = (
            ((acct.get("account") or {}).get("data") or {})
            .get("parsed", {})
            .get("info", {})
        )
        amount = _as_float((info.get("tokenAmount") or {}).get("uiAmount"))
        if amount and amount > 0:
            return True
    return False


async def get_kol_holdings(
    mint: str,
    kol_wallets: Optional[List[str]] = None,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    rpc_url: Optional[str] = None,
    max_wallets: int = DEFAULT_MAX_KOL_WALLETS,
) -> KolSignal:
    """Whether any curated KOL wallet holds ``mint`` (read-only, fail-closed).

    Uses one ``getTokenAccountsByOwner`` read per wallet (bounded by
    ``max_wallets``). ``kol_wallets`` defaults to ``config.KOL_WALLETS`` (from
    .env). An empty list yields ``available=True, any_hold=False`` (nothing to
    check, definitively no hit). Any RPC error / missing endpoint yields
    ``available=False`` — NEVER raises. This is a meta boost signal only and can
    never trigger or block a trade.
    """
    wallets = config.KOL_WALLETS if kol_wallets is None else kol_wallets
    wallets = [w for w in (wallets or []) if w][:max(0, max_wallets)]
    if not wallets:
        return KolSignal(any_hold=False, holding_wallets=(), available=True)

    resolved = rpc_url if rpc_url is not None else _helius_rpc_url()
    if not resolved:
        logger.info("[onchain] no Helius RPC configured -> KOL signal absent")
        return KolSignal()

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        holders: List[str] = []
        for wallet in wallets:
            result = await _rpc(
                client, resolved, "getTokenAccountsByOwner",
                [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
                timeout=timeout,
            )
            if _owner_holds_token((result or {}).get("value") or []):
                holders.append(wallet)
        return KolSignal(
            any_hold=bool(holders),
            holding_wallets=tuple(holders),
            available=True,
        )
    except Exception as exc:  # noqa: BLE001 — total fail-closed: signal absent
        logger.warning("[onchain] KOL-holdings fetch failed for %s: %s", mint, exc)
        return KolSignal()
    finally:
        if own_client:
            await client.aclose()
