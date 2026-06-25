"""Hard-exit strategy.

Independent, defensive exit triggers evaluated on every loop iteration. Per
CLAUDE.md these checks run BEFORE profit-taking; if ANY trigger fires the bot
sells the ENTIRE remaining position — moonbag included — and skips profit
logic for that iteration.

Triggers (any one is sufficient):
    1. hard_stop_loss   price drops >= 20% below entry price
    2. volume_collapse  rolling 15m volume drops > 70% from its peak
    3. liquidity_drop   current pool liquidity < 70% of entry liquidity
    4. coordinated_dump a single sell > 5% of pool liquidity, OR several top
                        wallets selling together
    5. flash_crash      price drops >= 70% within a single 1-minute candle

Safety: dry-run by default. Nothing is signed or broadcast — a structured
"Dry Run Exit" breakdown is logged and the decision is returned for the caller
to execute under an explicit live flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from strategies.profit_taking import Position

logger = logging.getLogger(__name__)

# --- Trigger thresholds ------------------------------------------------------
STOP_LOSS_DROP: float = 0.20          # exit if price falls >= 20% below entry
VOLUME_COLLAPSE_DROP: float = 0.70    # exit if 15m volume falls > 70% from peak
LIQUIDITY_FLOOR: float = 0.70         # exit if liquidity < 70% of entry
SINGLE_SELL_LIQ_PCT: float = 0.05     # exit if one sell > 5% of pool liquidity
COORDINATED_WALLET_THRESHOLD: int = 3  # exit if >= 3 top wallets dump together
FLASH_CRASH_DROP: float = 0.70        # exit if a 1m candle drops >= 70%


@dataclass
class MarketData:
    """Snapshot of market state for one evaluation.

    ``current_price`` is required. Event-style fields default to benign values
    so a caller only needs to populate the signals it has.
    """

    current_price: float
    rolling_volume_15m: float = 0.0
    peak_volume_15m: float = 0.0
    current_liquidity: float = 0.0
    entry_liquidity: float = 0.0
    largest_single_sell: float = 0.0   # value, same units as liquidity
    top_wallet_sells: int = 0          # count of top wallets selling together
    candle_1m_open: float = 0.0
    candle_1m_close: float = 0.0


@dataclass
class ExitDecision:
    """Result of a hard-exit evaluation."""

    should_exit: bool
    trigger: Optional[str] = None
    reason: Optional[str] = None
    tokens_to_sell: float = 0.0  # full remaining position, moonbag included


# --- Individual trigger checks ----------------------------------------------
# Each returns a human-readable reason string when it fires, else None.

def _check_hard_stop_loss(pos: Position, md: MarketData) -> Optional[str]:
    floor = pos.entry_price * (1.0 - STOP_LOSS_DROP)
    if md.current_price <= floor:
        drop = (1.0 - md.current_price / pos.entry_price) * 100
        return (
            f"price {md.current_price:.8f} is {drop:.1f}% below entry "
            f"{pos.entry_price:.8f} (>= {STOP_LOSS_DROP * 100:.0f}% stop-loss)"
        )
    return None


def _check_flash_crash(pos: Position, md: MarketData) -> Optional[str]:
    if md.candle_1m_open <= 0:
        return None
    drop = (md.candle_1m_open - md.candle_1m_close) / md.candle_1m_open
    if drop >= FLASH_CRASH_DROP:
        return (
            f"1m candle dropped {drop * 100:.1f}% (open {md.candle_1m_open:.8f} "
            f"-> close {md.candle_1m_close:.8f}, >= {FLASH_CRASH_DROP * 100:.0f}%)"
        )
    return None


def _check_liquidity_drop(pos: Position, md: MarketData) -> Optional[str]:
    if md.entry_liquidity <= 0:
        return None
    floor = md.entry_liquidity * LIQUIDITY_FLOOR
    if md.current_liquidity < floor:
        pct = md.current_liquidity / md.entry_liquidity * 100
        return (
            f"liquidity {md.current_liquidity:.2f} is {pct:.1f}% of entry "
            f"{md.entry_liquidity:.2f} (< {LIQUIDITY_FLOOR * 100:.0f}% floor)"
        )
    return None


def _check_volume_collapse(pos: Position, md: MarketData) -> Optional[str]:
    if md.peak_volume_15m <= 0:
        return None
    floor = md.peak_volume_15m * (1.0 - VOLUME_COLLAPSE_DROP)  # 30% of peak
    if md.rolling_volume_15m < floor:
        drop = (1.0 - md.rolling_volume_15m / md.peak_volume_15m) * 100
        return (
            f"15m volume {md.rolling_volume_15m:.2f} dropped {drop:.1f}% from "
            f"peak {md.peak_volume_15m:.2f} (> {VOLUME_COLLAPSE_DROP * 100:.0f}%)"
        )
    return None


def _check_coordinated_dump(pos: Position, md: MarketData) -> Optional[str]:
    if (
        md.current_liquidity > 0
        and md.largest_single_sell > md.current_liquidity * SINGLE_SELL_LIQ_PCT
    ):
        pct = md.largest_single_sell / md.current_liquidity * 100
        return (
            f"single sell {md.largest_single_sell:.2f} is {pct:.1f}% of pool "
            f"liquidity (> {SINGLE_SELL_LIQ_PCT * 100:.0f}%)"
        )
    if md.top_wallet_sells >= COORDINATED_WALLET_THRESHOLD:
        return (
            f"{md.top_wallet_sells} top wallets selling together "
            f"(>= {COORDINATED_WALLET_THRESHOLD})"
        )
    return None


# Ordered list of (trigger_name, check_fn). Checked in this order; the first
# trigger to fire wins (any single trigger forces a full exit regardless).
_CHECKS: List[Tuple[str, Callable[[Position, MarketData], Optional[str]]]] = [
    ("flash_crash", _check_flash_crash),
    ("hard_stop_loss", _check_hard_stop_loss),
    ("liquidity_drop", _check_liquidity_drop),
    ("volume_collapse", _check_volume_collapse),
    ("coordinated_dump", _check_coordinated_dump),
]


async def evaluate_hard_exit(
    position: Position,
    market_data: MarketData,
    dry_run: bool = True,
) -> ExitDecision:
    """Evaluate all hard-exit triggers for one loop iteration.

    Returns an :class:`ExitDecision`. If any trigger fires, ``should_exit`` is
    True, ``trigger``/``reason`` identify it, and ``tokens_to_sell`` is the
    FULL remaining position (moonbag included). If none fire, returns a
    no-exit decision with ``tokens_to_sell == 0``.

    In dry-run mode (the default) nothing is signed or broadcast — a structured
    "Dry Run Exit" breakdown is logged for the firing trigger.
    """
    for trigger_name, check_fn in _CHECKS:
        reason = check_fn(position, market_data)
        if reason is None:
            continue

        decision = ExitDecision(
            should_exit=True,
            trigger=trigger_name,
            reason=reason,
            tokens_to_sell=position.remaining,
        )

        if dry_run:
            logger.warning(
                "Dry Run Exit | trigger=%s reason=%s tokens_to_sell=%.6f "
                "(FULL remaining position, moonbag included)",
                decision.trigger,
                decision.reason,
                decision.tokens_to_sell,
            )
        # Live signing/broadcasting is intentionally NOT implemented here; the
        # caller executes the full exit under an explicit live flag.

        return decision

    return ExitDecision(should_exit=False)
