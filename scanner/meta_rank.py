"""Meta / KOL RANKING boost — a priority multiplier, never a trigger.

A :class:`MetaBoost` is a priority MULTIPLIER (>= 1.0) that RE-RANKS candidates
which have ALREADY passed the safety gate AND ALREADY produced a momentum-backed
BUY. It exists solely to prioritise the strongest *genuine* setups among tokens
that already qualified.

HARD GUARANTEES
---------------
* NOT a trigger: it is only ever computed by a caller that has already confirmed
  safety + a BUY (see ``main.evaluate_and_trade``, which calls it strictly on the
  "entered" branch). It takes no part in the safety or entry decision.
* NEVER lowers a bar: the boost has a hard FLOOR of 1.0. Absent / weak signals
  yield exactly 1.0 (neutral) — never a penalty that could veto, block, or shrink
  a token that already earned its entry. It can only RAISE priority.

SIGNALS (read-only, on-chain, hard-to-fake)
-------------------------------------------
Consumes the Helius signals from :mod:`data.onchain`:
  * ``get_recent_flow(...).buyer_wallets`` — buyer breadth: distinct recent
    buyer wallets (broad participation reads as organic).
  * ``get_kol_holdings(...).any_hold`` — whether any curated wallet
    (``config.KOL_WALLETS``) holds the token (a small flat bonus).

FAIL-CLOSED
-----------
Every entry point is read-only and NEVER raises. On any Helius error, timeout,
rate-limit, unconfigured endpoint, or missing data the corresponding component
contributes 0 (no boost), so the worst case is always a neutral 1.0 — an absent
signal can never block a good token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import httpx

from data.onchain import (
    DEFAULT_TIMEOUT,
    KolSignal,
    OnchainFlow,
    get_kol_holdings,
    get_recent_flow,
)

logger = logging.getLogger(__name__)

# --- Boost shaping (tunable) ------------------------------------------------
MAX_META_BOOST: float = 1.5        # hard ceiling on the priority multiplier
BUYER_BREADTH_TARGET: int = 20     # distinct buyers earning the full buyer bonus
BUYER_WEIGHT: float = 0.25         # max buyer-breadth bonus
KOL_WEIGHT: float = 0.15           # flat bonus when a tracked wallet holds


@dataclass(frozen=True)
class MetaBoost:
    """A ranking boost. ``boost`` in ``[1.0, MAX_META_BOOST]`` is the multiplier
    the caller applies to an already-qualified entry's priority.

    ``available`` records whether ANY on-chain signal was actually present; it is
    diagnostic only — behaviour never depends on it beyond the (already
    fail-closed) component maths.
    """

    boost: float = 1.0
    buyer_breadth: int = 0
    kol_hit: bool = False
    available: bool = False
    reason: str = "no meta signal (neutral)"

    @property
    def is_neutral(self) -> bool:
        """True when no boost was earned (a fail-closed / no-signal result)."""
        return self.boost <= 1.0


# The canonical no-boost result (the fail-closed floor).
NEUTRAL_BOOST: MetaBoost = MetaBoost()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_meta_boost(
    flow: Optional[OnchainFlow],
    kol: Optional[KolSignal],
    *,
    buyer_target: int = BUYER_BREADTH_TARGET,
    buyer_weight: float = BUYER_WEIGHT,
    kol_weight: float = KOL_WEIGHT,
    max_boost: float = MAX_META_BOOST,
) -> MetaBoost:
    """Combine the on-chain signals into a ``[1.0, max_boost]`` priority boost.

    Buyer breadth scales a bounded bonus (full at ``buyer_target`` distinct
    buyers); a KOL hold adds a flat bonus. TOTALLY fail-closed: a missing /
    unavailable signal contributes 0, so the result is never below 1.0. Performs
    NO gating — the caller must only invoke it for an already-safe, already-BUY
    token; it can never itself trigger or block a trade.
    """
    try:
        buyer_component = 0.0
        buyer_breadth = 0
        if flow is not None and flow.available:
            buyer_breadth = max(0, flow.buyer_wallets)
            if buyer_target > 0:
                buyer_component = _clamp(buyer_breadth / buyer_target, 0.0, 1.0) * buyer_weight

        kol_hit = bool(kol is not None and kol.available and kol.any_hold)
        kol_component = kol_weight if kol_hit else 0.0

        available = bool(
            (flow is not None and flow.available)
            or (kol is not None and kol.available)
        )
        boost = _clamp(1.0 + buyer_component + kol_component, 1.0, max_boost)
        if available:
            reason = (
                f"boost x{boost:.2f} | buyers={buyer_breadth} (+{buyer_component:.2f}) "
                f"kol={'yes' if kol_hit else 'no'} (+{kol_component:.2f})"
            )
        else:
            reason = "no meta signal available (neutral 1.00)"
        return MetaBoost(
            boost=boost,
            buyer_breadth=buyer_breadth,
            kol_hit=kol_hit,
            available=available,
            reason=reason,
        )
    except Exception as exc:  # noqa: BLE001 — ranking must never raise into a trade path
        logger.warning("[meta] boost computation failed: %s -> neutral", exc)
        return NEUTRAL_BOOST


async def fetch_meta_signals(
    mint: str,
    price: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    rpc_url: Optional[str] = None,
    kol_wallets: Optional[List[str]] = None,
) -> Tuple[OnchainFlow, KolSignal]:
    """Fetch the two read-only on-chain signals (both fail-closed, never raise).

    Reuses a single client for both calls when one is created here.
    """
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=timeout)
    try:
        flow = await get_recent_flow(
            mint, price=price, client=client, timeout=timeout, rpc_url=rpc_url
        )
        kol = await get_kol_holdings(
            mint, kol_wallets, client=client, timeout=timeout, rpc_url=rpc_url
        )
        return flow, kol
    finally:
        if own_client:
            await client.aclose()


async def rank_boost_for(
    mint: str,
    price: float,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = DEFAULT_TIMEOUT,
    rpc_url: Optional[str] = None,
    kol_wallets: Optional[List[str]] = None,
    buyer_target: int = BUYER_BREADTH_TARGET,
    buyer_weight: float = BUYER_WEIGHT,
    kol_weight: float = KOL_WEIGHT,
    max_boost: float = MAX_META_BOOST,
) -> MetaBoost:
    """Fetch on-chain signals for ``mint`` and compute its ranking boost.

    Read-only and totally fail-closed: any failure yields :data:`NEUTRAL_BOOST`
    (1.0), so a broken/absent Helius feed can only ever leave an entry
    un-boosted, never block or penalise it.
    """
    try:
        flow, kol = await fetch_meta_signals(
            mint, price, client=client, timeout=timeout,
            rpc_url=rpc_url, kol_wallets=kol_wallets,
        )
        return compute_meta_boost(
            flow, kol, buyer_target=buyer_target, buyer_weight=buyer_weight,
            kol_weight=kol_weight, max_boost=max_boost,
        )
    except Exception as exc:  # noqa: BLE001 — never let ranking raise
        logger.warning("[meta] rank_boost_for %s failed: %s -> neutral", mint, exc)
        return NEUTRAL_BOOST
