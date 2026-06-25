"""Tiered profit-taking strategy.

Implements the profit-taking ladder from CLAUDE.md. Each tier fires exactly
once over the lifetime of a position, guarded by boolean flags so it cannot
re-trigger across trading-loop iterations:

    >= 2x  entry price -> sell 50% of the ORIGINAL position size
    >= 5x  entry price -> sell 25% of the ORIGINAL position size
    >= 10x entry price -> sell 15% of the ORIGINAL position size

That leaves a 10% moonbag that is never auto-sold by this module.

All fractions are taken against the ORIGINAL size (not the remaining
balance), so the realised sells are additive: 50% + 25% + 15% = 90%, with a
10% moonbag retained.

Gap handling: if a single price reading has jumped past several thresholds at
once (e.g. straight from 1x to 10x), every eligible unsold tier fires in the
same call. A jump straight to 10x therefore sells 50% + 25% + 15% = 90% in one
evaluation. The once-only flags still guarantee each tier fires exactly once.

Safety: this module is dry-run by default. It never signs or broadcasts a
transaction — it logs a structured "Dry Run Sell" breakdown and returns the
intended actions for the caller to execute under an explicit live flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Fraction of the original position that is never auto-sold.
MOONBAG_FRACTION: float = 0.10

# (label, price multiple threshold, fraction of ORIGINAL to sell, flag name).
# Ordered ascending by threshold so lower tiers are evaluated/sold first.
_TIERS: List[Tuple[str, float, float, str]] = [
    ("2x", 2.0, 0.50, "sold_2x"),
    ("5x", 5.0, 0.25, "sold_5x"),
    ("10x", 10.0, 0.15, "sold_10x"),
]


@dataclass
class Position:
    """An open position being tracked for profit-taking.

    Attributes:
        entry_price: Price per token at entry. Must be > 0.
        original_size: Token amount bought at entry. Must be > 0.
        remaining: Tokens still held. Defaults to ``original_size``.
        sold_2x / sold_5x / sold_10x: Tier guards. Once a tier fires it flips
            to True and can never re-trigger.
    """

    entry_price: float
    original_size: float
    remaining: float = field(default=None)  # type: ignore[assignment]
    sold_2x: bool = False
    sold_5x: bool = False
    sold_10x: bool = False

    def __post_init__(self) -> None:
        if self.entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {self.entry_price!r}")
        if self.original_size <= 0:
            raise ValueError(f"original_size must be > 0, got {self.original_size!r}")
        if self.remaining is None:
            self.remaining = self.original_size

    @property
    def moonbag_tokens(self) -> float:
        """The protected moonbag amount, in tokens."""
        return self.original_size * MOONBAG_FRACTION


@dataclass
class SellAction:
    """A profit-taking sell decision for the caller to execute."""

    tier: str
    price_multiple: float
    fraction_of_original: float
    tokens_to_sell: float
    tokens_remaining_after: float


def _current_multiple(position: Position, current_price: float) -> float:
    """Return current price as a multiple of the entry price."""
    return current_price / position.entry_price


async def evaluate_take_profit(
    position: Position,
    current_price: float,
    dry_run: bool = True,
) -> List[SellAction]:
    """Evaluate the full profit-taking ladder for one price reading.

    Fires EVERY eligible unsold tier in a single call, in ascending order. If
    the price gapped past several thresholds at once, all of them fire now
    (e.g. a jump straight to 10x sells 50% + 25% + 15% in this one call).
    Returns the list of :class:`SellAction` executed this call, in tier order;
    an empty list means no tier triggered.

    When a tier fires, its guard flag is set and ``position.remaining`` is
    reduced so the tier cannot re-trigger and the simulation stays consistent.
    In dry-run mode (the default) nothing is signed or broadcast — a structured
    "Dry Run Sell" breakdown is logged for each tier that fires.
    """
    actions: List[SellAction] = []

    if current_price <= 0:
        logger.warning("Ignoring non-positive current_price=%r", current_price)
        return actions

    multiple = _current_multiple(position, current_price)

    for label, threshold, fraction, flag_name in _TIERS:
        already_sold: bool = getattr(position, flag_name)
        if already_sold or multiple < threshold:
            continue

        tokens_to_sell = position.original_size * fraction

        # Mark the tier fired before anything else so it can never re-trigger,
        # even if the caller's execution path raises later.
        setattr(position, flag_name, True)
        position.remaining = max(0.0, position.remaining - tokens_to_sell)

        action = SellAction(
            tier=label,
            price_multiple=multiple,
            fraction_of_original=fraction,
            tokens_to_sell=tokens_to_sell,
            tokens_remaining_after=position.remaining,
        )
        actions.append(action)

        if dry_run:
            logger.info(
                "Dry Run Sell | tier=%s price_multiple=%.2fx fraction_sold=%.0f%% "
                "tokens_sold=%.6f tokens_remaining=%.6f moonbag=%.6f",
                action.tier,
                action.price_multiple,
                action.fraction_of_original * 100,
                action.tokens_to_sell,
                action.tokens_remaining_after,
                position.moonbag_tokens,
            )
        # Live signing/broadcasting is intentionally NOT implemented here; the
        # caller executes the returned actions under an explicit live flag.

    return actions
