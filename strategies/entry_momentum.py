"""Momentum / breakout entry-decision strategy.

The second entry module, alongside the post-pump dip-buy (:mod:`strategies.entry`).
Both decide BUY / WAIT / SKIP for a token that has ALREADY passed the read-only
safety gate (``safety/rug_check``); neither re-runs it. Where the dip-buy waits
for an orderly pullback on a proven runner, this one does the opposite trade:
it buys *into* a strong, volume-confirmed breakout — but refuses to buy the top.

    BUY   strong, volume-confirmed upward momentum that is NOT overextended
    WAIT  momentum too weak yet, or the volume move is a single unconfirmed spike
    SKIP  overextended (RSI overbought OR already run past the run-up cap),
          bad data, drained liquidity, or an out-of-band market cap

DECISION PIPELINE (every gate fail-closed; a buy is the one irreversible act)
---------------------------------------------------------------------------
0. Input validation .............. bad/missing data                 -> SKIP
1. Market-cap band ............... mcap outside [MIN, MAX]          -> SKIP
2. MOMENTUM trigger .............. price change over the last N minutes must
     clear ``min_momentum_pct`` (default +15% over 15min), else too weak -> WAIT
3. OVEREXTENSION guards (critical — do NOT buy the top). Either fires -> SKIP:
     * run-up from the recent base exceeds ``max_runup_pct`` (default +100%)
     * RSI on the series exceeds ``rsi_overbought`` (default 75)
4. VOLUME confirmation ........... the move must ride SUSTAINED elevated volume,
     not one candle: window-average volume >= ``volume_confirm_ratio`` x the
     pre-move baseline, a majority of the window's candles above baseline, and
     no single candle dominating the window; otherwise a lone spike    -> WAIT
5. LIQUIDITY floor ............... current liquidity below the floor  -> SKIP
6. SIZING -> BUY ................. 1% (configurable) of portfolio balance,
     additionally capped at a fraction of pool liquidity (MEV guard).

Sizing / MEV mirrors the dip-buy exactly: notional is the smaller of the
portfolio max-allocation and a fraction of pool liquidity (``MAX_POOL_FRACTION``),
limiting our own price impact and the sandwich surface on low-liquidity pools
(CLAUDE.md MEV rules). Prefer smaller sizes on thin pools.

Position risk notes: each BUY carries a configurable time-stop and stop-loss
(advisory, for the caller's exit management — this module only opens). Breakout
entries default to a tighter stop than the dip-buy.

Safety: dry-run by default. Every outcome logs a structured
"Dry Run Momentum Entry: BUY/WAIT/SKIP" line with full reasoning and (on BUY)
sizing. Nothing is ever signed or broadcast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Reuse the shared tri-state action and market-data container so a single
# safety->entry pipeline can feed BOTH entry strategies interchangeably.
from strategies.entry import EntryAction, EntryMarketData
from strategies.profit_taking import Position

logger = logging.getLogger(__name__)

# --- Momentum trigger (tunable) ---------------------------------------------
MOMENTUM_LOOKBACK_MINUTES: float = 15.0   # N: window the move is measured over
MIN_MOMENTUM_PCT: float = 0.15            # X: min price gain over the window

# --- Volume confirmation (tunable) ------------------------------------------
VOLUME_CONFIRM_RATIO: float = 1.5         # window avg vol >= this x baseline
VOLUME_SUSTAIN_FRACTION: float = 0.5      # >= this share of window above baseline
MAX_PEAK_VOLUME_SHARE: float = 0.6        # one candle may not exceed this share

# --- Overextension guards (tunable) — do NOT buy the top --------------------
RSI_PERIOD: int = 14                      # lookback (in deltas) for RSI
RSI_OVERBOUGHT: float = 75.0              # RSI above this -> overbought -> SKIP
MAX_RUNUP_PCT: float = 1.00               # run-up above this off base -> SKIP
RUNUP_BASE_LOOKBACK_MINUTES: float = 60.0  # window whose low is the "recent base"

# --- Liquidity (tunable) ----------------------------------------------------
MIN_LIQUIDITY_USD: float = 10_000.0       # absolute pool-liquidity floor

# --- Market-cap band (tunable) ----------------------------------------------
MIN_MARKET_CAP_USD: float = 50_000.0       # below: too small / illiquid
MAX_MARKET_CAP_USD: float = 50_000_000.0   # above: too large to run again

# --- Sizing & risk notes (tunable) ------------------------------------------
DEFAULT_PORTFOLIO_USD: float = 10_000.0    # portfolio balance if not supplied
PORTFOLIO_ALLOCATION_PCT: float = 0.01     # max-allocation guardrail: 1%
MAX_POOL_FRACTION: float = 0.01            # cap buy at this fraction of pool
DEFAULT_TIME_STOP_MINUTES: float = 120.0   # advisory time-stop (tighter breakout)
DEFAULT_STOP_LOSS_PCT: float = 0.15        # advisory stop-loss below entry


@dataclass
class MomentumDecision:
    """Result of a momentum entry evaluation.

    Shares the core shape of :class:`strategies.entry.EntryDecision` (``action``,
    ``mint_address``, ``reason``, ``position`` + sizing/stop fields populated only
    on BUY) so a caller can treat either strategy's decision uniformly. The
    diagnostic fields are filled whenever computed so WAIT/SKIP are explainable.
    """

    action: EntryAction
    mint_address: str
    reason: str
    momentum_pct: float = 0.0
    run_up_pct: float = 0.0
    rsi: Optional[float] = None
    overextended: bool = False
    volume_confirmed: bool = False
    volume_ratio: float = 0.0
    peak_volume_share: float = 0.0
    position: Optional[Position] = None
    entry_price: float = 0.0
    entry_liquidity: float = 0.0
    notional_usd: float = 0.0
    time_stop_minutes: float = 0.0
    stop_loss_pct: float = 0.0
    stop_loss_price: float = 0.0


def _sample_count(minutes: float, interval_minutes: float) -> int:
    """Number of samples spanning ``minutes`` at ``interval_minutes`` spacing
    (at least 1). A non-positive interval is treated as 1 minute."""
    interval = interval_minutes if interval_minutes > 0 else 1.0
    return max(1, round(minutes / interval))


def _price_momentum(
    prices: List[float], current_price: float, window_samples: int
) -> Optional[Tuple[float, float]]:
    """Return ``(reference_price, momentum_pct)`` for the move over the last
    ``window_samples`` samples, or ``None`` if it cannot be computed.

    The reference is the close ``window_samples`` back (clamped to the oldest
    available sample); ``momentum_pct`` is ``(current - reference) / reference``.
    """
    if len(prices) < 2:
        return None
    ref_index = max(0, len(prices) - 1 - window_samples)
    reference = prices[ref_index]
    if reference <= 0:
        return None
    return reference, (current_price - reference) / reference


def _run_up_from_base(
    prices: List[float], current_price: float, base_samples: int
) -> Optional[Tuple[float, float]]:
    """Return ``(base_price, run_up_pct)`` where ``base_price`` is the lowest
    close over the last ``base_samples`` samples (the recent base the token ran
    up from), or ``None`` if it cannot be computed."""
    if not prices:
        return None
    base_window = prices[max(0, len(prices) - 1 - base_samples):]
    if not base_window:
        return None
    base = min(base_window)
    if base <= 0:
        return None
    return base, (current_price - base) / base


def _compute_rsi(prices: List[float], period: int) -> Optional[float]:
    """Classic RSI over the last ``period`` price deltas (simple-average form).

    Returns ``None`` when there are too few points (< 2 deltas). With no losses
    in the window RSI is 100 (pure up-move); with no gains it is 0.
    """
    if len(prices) < 3:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    window = deltas[-period:] if len(deltas) >= period else deltas
    if not window:
        return None
    avg_gain = sum(d for d in window if d > 0) / len(window)
    avg_loss = sum(-d for d in window if d < 0) / len(window)
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@dataclass
class _VolumeCheck:
    """Outcome of the sustained-volume confirmation."""

    confirmed: bool
    ratio: float
    peak_share: float
    detail: str


def _volume_confirmation(
    volumes: List[float],
    window_samples: int,
    *,
    confirm_ratio: float,
    sustain_fraction: float,
    max_peak_share: float,
) -> _VolumeCheck:
    """Judge whether the recent volume move is a SUSTAINED elevation rather than
    a single spike.

    Confirmed requires ALL of: window-average volume >= ``confirm_ratio`` x the
    pre-move baseline average (elevated), at least ``sustain_fraction`` of the
    window's candles at/above that baseline (sustained), and no single candle
    exceeding ``max_peak_share`` of the window's total volume (not one spike).
    """
    if len(volumes) < 2:
        return _VolumeCheck(False, 0.0, 0.0, "insufficient volume history")

    window_n = min(len(volumes), max(2, window_samples))
    window = volumes[-window_n:]
    baseline = volumes[:-window_n]
    if not baseline:
        return _VolumeCheck(False, 0.0, 0.0, "no pre-move volume baseline")

    baseline_avg = sum(baseline) / len(baseline)
    window_avg = sum(window) / len(window)
    window_total = sum(window)
    peak_share = (max(window) / window_total) if window_total > 0 else 1.0
    fraction_above = sum(1 for v in window if v >= baseline_avg) / len(window)

    if baseline_avg <= 0:
        # No prior volume to compare against: treat any real window volume as
        # elevated but still demand it be sustained and not a lone spike.
        ratio = float("inf") if window_total > 0 else 0.0
    else:
        ratio = window_avg / baseline_avg

    elevated = ratio >= confirm_ratio
    sustained = fraction_above >= sustain_fraction
    not_single_spike = peak_share <= max_peak_share
    confirmed = elevated and sustained and not_single_spike

    if confirmed:
        detail = "sustained elevated volume"
    elif not elevated:
        detail = f"volume only {ratio:.1f}x baseline (< {confirm_ratio:.1f}x)"
    elif not not_single_spike:
        detail = (
            f"single volume spike: one candle is {peak_share * 100:.0f}% of the "
            f"window (> {max_peak_share * 100:.0f}%)"
        )
    else:
        detail = (
            f"volume not sustained: only {fraction_above * 100:.0f}% of candles "
            f"above baseline (< {sustain_fraction * 100:.0f}%)"
        )
    return _VolumeCheck(confirmed, ratio if ratio != float("inf") else 0.0,
                        peak_share, detail)


def _log_decision(decision: MomentumDecision) -> None:
    """Emit the structured dry-run line for any outcome."""
    if decision.action is EntryAction.BUY:
        logger.info(
            "Dry Run Momentum Entry: BUY | mint=%s reason=%s | entry=%.8f "
            "momentum=+%.1f%% run_up=+%.1f%% rsi=%s vol=%.1fx (peak %.0f%%) | "
            "tokens=%.6f notional=$%.2f (<= %.0f%% pool) | "
            "stop_loss=%.8f (-%.0f%%) time_stop=%.0fmin",
            decision.mint_address, decision.reason, decision.entry_price,
            decision.momentum_pct * 100, decision.run_up_pct * 100,
            f"{decision.rsi:.0f}" if decision.rsi is not None else "n/a",
            decision.volume_ratio, decision.peak_volume_share * 100,
            decision.position.original_size if decision.position else 0.0,
            decision.notional_usd, MAX_POOL_FRACTION * 100,
            decision.stop_loss_price, decision.stop_loss_pct * 100,
            decision.time_stop_minutes,
        )
    else:
        logger.info(
            "Dry Run Momentum Entry: %s | mint=%s reason=%s",
            decision.action.value, decision.mint_address, decision.reason,
        )


def _decision(
    action: EntryAction, mint_address: str, reason: str, **extra: object
) -> MomentumDecision:
    """Build, log, and return a non-BUY decision (BUY is built inline)."""
    decision = MomentumDecision(
        action=action, mint_address=mint_address, reason=reason, **extra  # type: ignore[arg-type]
    )
    _log_decision(decision)
    return decision


async def evaluate_entry_momentum(
    mint_address: str,
    market: EntryMarketData,
    *,
    dry_run: bool = True,
    momentum_lookback_minutes: float = MOMENTUM_LOOKBACK_MINUTES,
    min_momentum_pct: float = MIN_MOMENTUM_PCT,
    volume_confirm_ratio: float = VOLUME_CONFIRM_RATIO,
    volume_sustain_fraction: float = VOLUME_SUSTAIN_FRACTION,
    max_peak_volume_share: float = MAX_PEAK_VOLUME_SHARE,
    rsi_period: int = RSI_PERIOD,
    rsi_overbought: float = RSI_OVERBOUGHT,
    max_runup_pct: float = MAX_RUNUP_PCT,
    runup_base_lookback_minutes: float = RUNUP_BASE_LOOKBACK_MINUTES,
    min_liquidity_usd: float = MIN_LIQUIDITY_USD,
    portfolio_balance: float = DEFAULT_PORTFOLIO_USD,
    portfolio_allocation_pct: float = PORTFOLIO_ALLOCATION_PCT,
    time_stop_minutes: float = DEFAULT_TIME_STOP_MINUTES,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
) -> MomentumDecision:
    """Decide BUY / WAIT / SKIP for an already-safety-checked token on momentum.

    See the module docstring for the full pipeline. Every gate is fail-closed:
    bad data, an out-of-band market cap, an overbought RSI, a run past the
    run-up cap, or a drained/thin pool all yield SKIP; weak momentum or an
    unconfirmed (single-spike) volume move yields WAIT; a strong, volume-
    confirmed, not-overextended breakout yields a sized BUY with advisory
    stop-loss / time-stop notes. ``dry_run`` only gates the structured log line;
    this module never signs or broadcasts regardless.
    """
    try:
        # --- 0. Fail-closed input validation --------------------------------
        if market.current_price <= 0:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"invalid price data (current={market.current_price!r})",
            )
        if len(market.price_history) < 2 or not market.volume_history:
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

        interval = market.sample_interval_minutes
        current = market.current_price

        # --- 2. MOMENTUM trigger --------------------------------------------
        window_samples = _sample_count(momentum_lookback_minutes, interval)
        momentum = _price_momentum(market.price_history, current, window_samples)
        if momentum is None:
            return _decision(
                EntryAction.SKIP, mint_address,
                "cannot measure momentum (invalid/insufficient price history)",
            )
        _, momentum_pct = momentum
        if momentum_pct < min_momentum_pct:
            return _decision(
                EntryAction.WAIT, mint_address,
                f"momentum only +{momentum_pct * 100:.1f}% over "
                f"~{momentum_lookback_minutes:.0f}min "
                f"(< +{min_momentum_pct * 100:.0f}%); too weak, wait for the move",
                momentum_pct=momentum_pct,
            )

        # --- 3. OVEREXTENSION guards (critical — do NOT buy the top) --------
        base_samples = _sample_count(runup_base_lookback_minutes, interval)
        run_up = _run_up_from_base(market.price_history, current, base_samples)
        if run_up is None:
            return _decision(
                EntryAction.SKIP, mint_address,
                "cannot establish a recent base for the run-up check (fail-closed)",
                momentum_pct=momentum_pct,
            )
        _, run_up_pct = run_up
        if run_up_pct > max_runup_pct:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"overextended: already +{run_up_pct * 100:.0f}% off the recent "
                f"base (> +{max_runup_pct * 100:.0f}%); do not buy the top",
                momentum_pct=momentum_pct, run_up_pct=run_up_pct, overextended=True,
            )

        rsi = _compute_rsi(market.price_history, rsi_period)
        if rsi is not None and rsi > rsi_overbought:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"overbought: RSI {rsi:.0f} > {rsi_overbought:.0f}; "
                f"do not buy the top",
                momentum_pct=momentum_pct, run_up_pct=run_up_pct, rsi=rsi,
                overextended=True,
            )

        # --- 4. VOLUME confirmation (sustained, not a single spike) ---------
        volume = _volume_confirmation(
            market.volume_history, window_samples,
            confirm_ratio=volume_confirm_ratio,
            sustain_fraction=volume_sustain_fraction,
            max_peak_share=max_peak_volume_share,
        )
        if not volume.confirmed:
            return _decision(
                EntryAction.WAIT, mint_address,
                f"momentum not volume-confirmed: {volume.detail}",
                momentum_pct=momentum_pct, run_up_pct=run_up_pct, rsi=rsi,
                volume_confirmed=False, volume_ratio=volume.ratio,
                peak_volume_share=volume.peak_share,
            )

        # --- 5. LIQUIDITY floor ---------------------------------------------
        if market.current_liquidity < min_liquidity_usd:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"liquidity ${market.current_liquidity:,.0f} below floor "
                f"${min_liquidity_usd:,.0f}",
                momentum_pct=momentum_pct, run_up_pct=run_up_pct, rsi=rsi,
                volume_confirmed=True, volume_ratio=volume.ratio,
                peak_volume_share=volume.peak_share,
            )

        # --- 6. SIZING -> BUY ----------------------------------------------
        budget_usd = portfolio_balance * portfolio_allocation_pct
        notional_usd = min(budget_usd, market.current_liquidity * MAX_POOL_FRACTION)
        tokens = notional_usd / current
        if tokens <= 0:
            return _decision(
                EntryAction.SKIP, mint_address,
                f"computed non-positive size ({tokens!r})",
                momentum_pct=momentum_pct, run_up_pct=run_up_pct, rsi=rsi,
                volume_confirmed=True, volume_ratio=volume.ratio,
                peak_volume_share=volume.peak_share,
            )

        position = Position(entry_price=current, original_size=tokens)
        stop_loss_price = current * (1.0 - stop_loss_pct)

        decision = MomentumDecision(
            action=EntryAction.BUY,
            mint_address=mint_address,
            reason="strong volume-confirmed breakout, not overextended",
            momentum_pct=momentum_pct,
            run_up_pct=run_up_pct,
            rsi=rsi,
            overextended=False,
            volume_confirmed=True,
            volume_ratio=volume.ratio,
            peak_volume_share=volume.peak_share,
            position=position,
            entry_price=current,
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
        return _decision(EntryAction.SKIP, mint_address, f"momentum entry error: {exc}")
