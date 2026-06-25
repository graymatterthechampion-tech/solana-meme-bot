"""Post-pump dip-buy entry-decision strategy.

Decides BUY / WAIT / SKIP for a token that has ALREADY passed the read-only
safety gate (``safety/rug_check``). This module assumes safety; it does not
re-run it. Its job is the *timing* call: buy the first orderly dip on a token
that has demonstrably run, not the top, and not a falling knife.

    BUY   open a position now (a stabilised dip on a proven runner), sized
    WAIT  proven runner, but the setup is not ready (too early / still falling)
    SKIP  not a proven runner, dropped too far, drained, or out of mcap band

DECISION PIPELINE (every gate fail-closed; a buy is the one irreversible act)
---------------------------------------------------------------------------
0. Input validation .............. bad/missing data            -> SKIP
1. Market-cap band ............... mcap outside [MIN, MAX]      -> SKIP
2. PROVEN RUNNER gate ............ must pass to be considered:
     * new ATH within ``ath_max_age_hours`` (default 12h), else stale -> SKIP
     * volume spike >= ``min_volume_spike``x baseline (default 5x), else SKIP
3. PULLBACK trigger .............. retrace off ATH:
     * younger than ``ath_min_age_hours`` (default 2h)          -> WAIT (hot)
     * retrace < ``pullback_min_pct`` (default 30%)             -> WAIT (early)
     * retrace > ``pullback_max_pct`` (default 50%)             -> SKIP (dying)
4. CONSOLIDATION check ........... price must have stabilised within a tight
     band (default 10%) over the last K minutes (default 5); still dropping
     vertically (a falling knife)                               -> WAIT
5. LIQUIDITY check ............... current liquidity must clear the floor AND
     not be significantly drained vs its pre-dip level          -> SKIP
6. SIZING -> BUY ................. 1% (configurable) of portfolio balance,
     additionally capped at a fraction of pool liquidity (MEV guard).

Sizing / MEV: notional is the smaller of the portfolio max-allocation and a
fraction of pool liquidity (``MAX_POOL_FRACTION``), limiting our own price
impact and the sandwich surface on low-liquidity pools (CLAUDE.md MEV rules).

Position risk notes: each BUY carries a configurable time-stop and stop-loss
(advisory, for the caller's exit management — this module only opens).

Safety: dry-run by default. Every outcome logs a structured
"Dry Run Entry: BUY/WAIT/SKIP" line with full reasoning and (on BUY) sizing.
Nothing is ever signed or broadcast.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from strategies.profit_taking import Position

logger = logging.getLogger(__name__)

# --- Proven-runner gate (tunable) -------------------------------------------
ATH_MAX_AGE_HOURS: float = 12.0     # ATH older than this is a stale runner
ATH_MIN_AGE_HOURS: float = 2.0      # ATH younger than this is still too hot
MIN_VOLUME_SPIKE: float = 5.0       # peak volume must be >= this x baseline

# --- Pullback band (tunable) ------------------------------------------------
PULLBACK_MIN_PCT: float = 0.30      # below this retrace: too early (WAIT)
PULLBACK_MAX_PCT: float = 0.50      # above this retrace: breakdown (SKIP)

# --- Consolidation / falling-knife guard (tunable) --------------------------
CONSOLIDATION_BAND_PCT: float = 0.10   # max range over the window to be "tight"
CONSOLIDATION_MINUTES: float = 5.0     # window that must be tight (minutes)

# --- Liquidity (tunable) ----------------------------------------------------
MIN_LIQUIDITY_USD: float = 10_000.0    # absolute pool-liquidity floor
MAX_LIQUIDITY_DRAIN_PCT: float = 0.30  # max drain vs pre-dip liquidity

# --- Market-cap band (tunable) ----------------------------------------------
MIN_MARKET_CAP_USD: float = 50_000.0       # below: too small / illiquid
MAX_MARKET_CAP_USD: float = 50_000_000.0   # above: too large to run again

# --- Sizing & risk notes (tunable) ------------------------------------------
DEFAULT_PORTFOLIO_USD: float = 10_000.0    # portfolio balance if not supplied
PORTFOLIO_ALLOCATION_PCT: float = 0.01     # max-allocation guardrail: 1%
MAX_POOL_FRACTION: float = 0.01            # cap buy at this fraction of pool
DEFAULT_TIME_STOP_MINUTES: float = 240.0   # advisory time-stop on the position
DEFAULT_STOP_LOSS_PCT: float = 0.20        # advisory stop-loss below entry


class EntryAction(str, Enum):
    """Tri-state entry decision. ``str`` mixin so it logs/serialises as its
    plain name."""

    BUY = "BUY"
    WAIT = "WAIT"
    SKIP = "SKIP"


@dataclass
class EntryMarketData:
    """Market inputs for a post-pump dip-buy decision.

    ``price_history`` / ``volume_history`` are recent samples in chronological
    order (oldest first); ``sample_interval_minutes`` is the spacing between
    price samples, used to map the consolidation window (minutes) onto samples.
    ``ath_timestamp`` and ``now`` are epoch seconds. ``pre_dip_liquidity`` is
    the pool liquidity recorded before the pullback (for the drain check).
    """

    current_price: float
    ath_price: float
    ath_timestamp: float
    price_history: List[float]
    volume_history: List[float]
    current_liquidity: float
    pre_dip_liquidity: float
    sample_interval_minutes: float = 1.0
    market_cap_usd: Optional[float] = None
    now: float = field(default_factory=time.time)


@dataclass
class EntryDecision:
    """Result of an entry evaluation.

    ``position`` and the sizing / stop fields are populated only on BUY. The
    diagnostic fields are filled whenever computed so WAIT/SKIP are explainable.
    """

    action: EntryAction
    mint_address: str
    reason: str
    proven_runner: bool = False
    ath_age_hours: float = 0.0
    pullback_pct: float = 0.0
    volume_spike_ratio: float = 0.0
    consolidated: bool = False
    position: Optional[Position] = None
    entry_price: float = 0.0
    entry_liquidity: float = 0.0
    notional_usd: float = 0.0
    time_stop_minutes: float = 0.0
    stop_loss_pct: float = 0.0
    stop_loss_price: float = 0.0


def _baseline_average(volume_history: List[float]) -> Optional[float]:
    """Baseline average volume = mean excluding the single peak (so the spike
    does not inflate its own baseline), or ``None`` if it cannot be computed."""
    if len(volume_history) < 2:
        return None
    rest = list(volume_history)
    rest.remove(max(volume_history))  # drop one occurrence of the peak
    if not rest:
        return None
    avg = sum(rest) / len(rest)
    return avg if avg > 0 else None


def _is_consolidated(
    price_history: List[float],
    sample_interval_minutes: float,
    window_minutes: float,
    band_pct: float,
) -> Optional[bool]:
    """Return whether the most recent ``window_minutes`` of price sit inside a
    ``band_pct`` range (stabilised), or ``None`` if there is not enough data.

    A wide range over the window means the price is still moving vertically — a
    falling knife — so the caller treats both ``False`` and ``None`` as WAIT.
    """
    interval = sample_interval_minutes if sample_interval_minutes > 0 else 1.0
    n = max(2, round(window_minutes / interval))
    window = price_history[-n:]
    if len(window) < 2:
        return None
    hi, lo = max(window), min(window)
    if hi <= 0:
        return None
    return (hi - lo) / hi <= band_pct


def _log_decision(decision: EntryDecision) -> None:
    """Emit the structured dry-run line for any outcome."""
    if decision.action is EntryAction.BUY:
        logger.info(
            "Dry Run Entry: BUY | mint=%s reason=%s | entry=%.8f "
            "pullback=%.1f%% off ATH ath_age=%.1fh vol_spike=%.1fx "
            "tokens=%.6f notional=$%.2f (<= %.0f%% pool) | "
            "stop_loss=%.8f (-%.0f%%) time_stop=%.0fmin",
            decision.mint_address, decision.reason, decision.entry_price,
            decision.pullback_pct * 100, decision.ath_age_hours,
            decision.volume_spike_ratio,
            decision.position.original_size if decision.position else 0.0,
            decision.notional_usd, MAX_POOL_FRACTION * 100,
            decision.stop_loss_price, decision.stop_loss_pct * 100,
            decision.time_stop_minutes,
        )
    else:
        logger.info(
            "Dry Run Entry: %s | mint=%s reason=%s",
            decision.action.value, decision.mint_address, decision.reason,
        )


def _decision(
    action: EntryAction, mint_address: str, reason: str, **extra: object
) -> EntryDecision:
    """Build, log, and return a non-BUY decision (BUY is built inline)."""
    decision = EntryDecision(
        action=action, mint_address=mint_address, reason=reason, **extra  # type: ignore[arg-type]
    )
    _log_decision(decision)
    return decision


async def evaluate_entry(
    mint_address: str,
    market: EntryMarketData,
    *,
    dry_run: bool = True,
    ath_min_age_hours: float = ATH_MIN_AGE_HOURS,
    ath_max_age_hours: float = ATH_MAX_AGE_HOURS,
    min_volume_spike: float = MIN_VOLUME_SPIKE,
    pullback_min_pct: float = PULLBACK_MIN_PCT,
    pullback_max_pct: float = PULLBACK_MAX_PCT,
    consolidation_band_pct: float = CONSOLIDATION_BAND_PCT,
    consolidation_minutes: float = CONSOLIDATION_MINUTES,
    portfolio_balance: float = DEFAULT_PORTFOLIO_USD,
    portfolio_allocation_pct: float = PORTFOLIO_ALLOCATION_PCT,
    time_stop_minutes: float = DEFAULT_TIME_STOP_MINUTES,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> EntryDecision:
    """Decide BUY / WAIT / SKIP for an already-safety-checked token.

    See the module docstring for the full pipeline. Every gate is fail-closed:
    bad data, a failed proven-runner gate, an over-extended drop, a drained
    pool, or an out-of-band market cap all yield SKIP; a proven runner whose
    dip has not stabilised yields WAIT; a clean stabilised dip yields a sized
    BUY with advisory stop-loss / time-stop notes. ``dry_run`` only gates the
    structured log line; this module never signs or broadcasts regardless.
    """
    try:
        # --- 0. Fail-closed input validation --------------------------------
        if market.current_price <= 0 or market.ath_price <= 0:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"invalid price data (current={market.current_price!r}, "
                f"ath={market.ath_price!r})",
            )
        if not market.price_history or not market.volume_history:
            return _decision(
                EntryAction.SKIP, mint_address,
                "insufficient price/volume history (fail-closed)",
            )

        # --- 1. Market-cap band ---------------------------------------------
        if market.market_cap_usd is not None and (
            market.market_cap_usd < MIN_MARKET_CAP_USD
            or market.market_cap_usd > MAX_MARKET_CAP_USD
        ):
            return _decision(
                EntryAction.SKIP, mint_address,
                f"market cap ${market.market_cap_usd:,.0f} outside band "
                f"[${MIN_MARKET_CAP_USD:,.0f}, ${MAX_MARKET_CAP_USD:,.0f}]",
            )

        # Trust the data: use the higher of supplied ATH and observed peak.
        ath_price = max(market.ath_price, max(market.price_history))
        ath_age_hours = (market.now - market.ath_timestamp) / 3600.0
        if ath_age_hours < 0:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"ATH timestamp is in the future (age={ath_age_hours:.2f}h)",
            )

        # --- 2. PROVEN RUNNER gate (must pass to be considered) -------------
        avg_volume = _baseline_average(market.volume_history)
        if avg_volume is None:
            return _decision(
                EntryAction.SKIP, mint_address,
                "cannot establish a baseline volume average (fail-closed)",
                ath_age_hours=ath_age_hours,
            )
        volume_spike_ratio = max(market.volume_history) / avg_volume

        if volume_spike_ratio < min_volume_spike:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"not a proven runner: volume spike {volume_spike_ratio:.1f}x "
                f"< required {min_volume_spike:.1f}x",
                ath_age_hours=ath_age_hours, volume_spike_ratio=volume_spike_ratio,
            )
        if ath_age_hours > ath_max_age_hours:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"no recent ATH: last ATH {ath_age_hours:.1f}h ago "
                f"(> {ath_max_age_hours:.0f}h window)",
                ath_age_hours=ath_age_hours, volume_spike_ratio=volume_spike_ratio,
            )

        # Proven runner from here on.
        pullback_pct = (ath_price - market.current_price) / ath_price

        # --- 3. PULLBACK trigger --------------------------------------------
        if ath_age_hours < ath_min_age_hours:
            return _decision(
                EntryAction.WAIT, mint_address,
                f"proven runner but ATH only {ath_age_hours:.1f}h ago "
                f"(< {ath_min_age_hours:.0f}h); too hot, let the dip form",
                proven_runner=True, ath_age_hours=ath_age_hours,
                pullback_pct=pullback_pct, volume_spike_ratio=volume_spike_ratio,
            )
        if pullback_pct < pullback_min_pct:
            return _decision(
                EntryAction.WAIT, mint_address,
                f"pullback only {pullback_pct * 100:.1f}% off ATH "
                f"(< {pullback_min_pct * 100:.0f}%); too early",
                proven_runner=True, ath_age_hours=ath_age_hours,
                pullback_pct=pullback_pct, volume_spike_ratio=volume_spike_ratio,
            )
        if pullback_pct > pullback_max_pct:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"dropped too far: {pullback_pct * 100:.1f}% off ATH "
                f"(> {pullback_max_pct * 100:.0f}%); likely dying",
                proven_runner=True, ath_age_hours=ath_age_hours,
                pullback_pct=pullback_pct, volume_spike_ratio=volume_spike_ratio,
            )

        # --- 4. CONSOLIDATION check (avoid falling knives) ------------------
        consolidated = _is_consolidated(
            market.price_history, market.sample_interval_minutes,
            consolidation_minutes, consolidation_band_pct,
        )
        if not consolidated:  # False or None -> not stabilised
            return _decision(
                EntryAction.WAIT, mint_address,
                f"not stabilised: price has not held within "
                f"{consolidation_band_pct * 100:.0f}% over the last "
                f"{consolidation_minutes:.0f}min (falling knife)",
                proven_runner=True, ath_age_hours=ath_age_hours,
                pullback_pct=pullback_pct, volume_spike_ratio=volume_spike_ratio,
                consolidated=False,
            )

        # --- 5. LIQUIDITY check (floor + drain vs pre-dip) ------------------
        if market.current_liquidity < MIN_LIQUIDITY_USD:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"liquidity ${market.current_liquidity:,.0f} below floor "
                f"${MIN_LIQUIDITY_USD:,.0f}",
                proven_runner=True, ath_age_hours=ath_age_hours,
                pullback_pct=pullback_pct, volume_spike_ratio=volume_spike_ratio,
                consolidated=True,
            )
        if market.pre_dip_liquidity > 0:
            drained = 1.0 - market.current_liquidity / market.pre_dip_liquidity
            if drained > MAX_LIQUIDITY_DRAIN_PCT:
                return _decision(
                    EntryAction.SKIP, mint_address,
                    f"liquidity drained {drained * 100:.1f}% vs pre-dip "
                    f"${market.pre_dip_liquidity:,.0f} (> "
                    f"{MAX_LIQUIDITY_DRAIN_PCT * 100:.0f}%)",
                    proven_runner=True, ath_age_hours=ath_age_hours,
                    pullback_pct=pullback_pct,
                    volume_spike_ratio=volume_spike_ratio, consolidated=True,
                )

        # --- 6. SIZING -> BUY ----------------------------------------------
        budget_usd = portfolio_balance * portfolio_allocation_pct
        notional_usd = min(budget_usd, market.current_liquidity * MAX_POOL_FRACTION)
        tokens = notional_usd / market.current_price
        if tokens <= 0:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"computed non-positive size ({tokens!r})",
                proven_runner=True, ath_age_hours=ath_age_hours,
                pullback_pct=pullback_pct, volume_spike_ratio=volume_spike_ratio,
                consolidated=True,
            )

        position = Position(
            entry_price=market.current_price, original_size=tokens
        )
        stop_loss_price = market.current_price * (1.0 - stop_loss_pct)

        decision = EntryDecision(
            action=EntryAction.BUY,
            mint_address=mint_address,
            reason="stabilised dip on a proven runner",
            proven_runner=True,
            ath_age_hours=ath_age_hours,
            pullback_pct=pullback_pct,
            volume_spike_ratio=volume_spike_ratio,
            consolidated=True,
            position=position,
            entry_price=market.current_price,
            entry_liquidity=market.current_liquidity,
            notional_usd=notional_usd,
            time_stop_minutes=time_stop_minutes,
            stop_loss_pct=stop_loss_pct,
            stop_loss_price=stop_loss_price,
        )
        _log_decision(decision)
        # Live signing/broadcasting is intentionally NOT implemented here; the
        # caller executes the buy under an explicit live flag.
        return decision
    except Exception as exc:  # noqa: BLE001 — total fail-closed: never buy on error
        return _decision(EntryAction.SKIP, mint_address, f"entry error: {exc}")
