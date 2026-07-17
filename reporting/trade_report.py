"""Clean per-trade reporting.

The managed loop emits a line every tick (holds, fills, meta boosts, ...). Over
a continuous run that cumulative stream is too noisy to answer the only two
questions that matter after the fact:

    * WHY did the position exit?
    * What was the per-trade (and per-sell) PnL?

This module collapses one entered position's whole lifecycle into a single
:class:`TradeReport` and a compact, ASCII-only text block. It is pure and
read-only: it reads an already-completed ``TradeSession`` and computes nothing
that touches the network, a signer, or a broadcast.

PnL convention matches the rest of the bot: realised PnL for a sell is its
simulated net proceeds (after price impact, fees, and fill-delay drift — see
:mod:`execution.fill_simulator`) minus the cost basis of the tokens sold
(``tokens * entry_price``). A held moonbag has no realised PnL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from strategies.entry_momentum import MomentumDecision

if TYPE_CHECKING:  # avoid a runtime import cycle (main imports this module).
    from main import LoopOutcome, TradeSession

logger = logging.getLogger(__name__)

# Terminal status (main.TradeSession.status) that actually opened a position.
_ENTERED_STATUS: str = "entered"


@dataclass(frozen=True)
class TradeLeg:
    """One executed (dry-run) sell within a trade's lifecycle.

    ``kind`` is ``"profit"`` (a profit-taking tier) or ``"hard_exit"`` (the full
    defensive exit). ``label`` is the human tag used in the report, e.g.
    ``"PROFIT/2x"`` or ``"HARD_EXIT/hard_stop_loss"``.
    """

    label: str
    kind: str
    tokens: float
    price: float
    net_proceeds_usd: float
    pnl_usd: float
    over_cap: bool


@dataclass(frozen=True)
class TradeReport:
    """A clean, self-contained summary of one entered position's lifecycle."""

    mint_address: str
    symbol: str
    strategy: str
    entry_reason: str
    entry_price: float
    entry_size: float
    entry_liquidity: float
    legs: List[TradeLeg] = field(default_factory=list)
    exit_trigger: Optional[str] = None
    exit_reason: str = ""
    fully_exited: bool = False
    tokens_held: float = 0.0
    realised_pnl_usd: float = 0.0
    fill_count: int = 0


def _strategy_name(entry_decision: object) -> str:
    """Name the strategy that produced ``entry_decision`` (dip vs momentum).

    Both decision types are structurally identical, so the concrete class is the
    only reliable discriminator. Unknown types degrade to ``"?"`` rather than
    guessing.
    """
    if isinstance(entry_decision, MomentumDecision):
        return "momentum"
    if entry_decision is None:
        return "?"
    # The dip decision (strategies.entry.EntryDecision) is the only other type
    # the pipeline produces; name it without importing it to keep this decoupled.
    if type(entry_decision).__name__ == "EntryDecision":
        return "dip"
    return "?"


def _legs_from_outcome(
    outcome: "LoopOutcome", entry_price: float
) -> List[TradeLeg]:
    """Build the executed sell legs for one loop outcome.

    A ``hard_exit`` outcome carries its single full-exit fill alongside the
    firing :class:`~strategies.hard_exit.ExitDecision`; a ``profit_taking``
    outcome pairs each :class:`~strategies.profit_taking.SellAction` with the
    fill it produced (same order). A ``hold`` outcome has no fills and yields no
    legs. Legs are matched positionally with ``zip`` — the fill list is built
    one-per-action/exit by the loop, so the pairing is exact.
    """
    legs: List[TradeLeg] = []

    if outcome.path == "hard_exit" and outcome.exit_decision is not None:
        trigger = outcome.exit_decision.trigger or "hard_exit"
        for fill in outcome.fills:
            legs.append(
                _leg(f"HARD_EXIT/{trigger}", "hard_exit", fill, entry_price)
            )
    elif outcome.path == "profit_taking":
        for action, fill in zip(outcome.sell_actions, outcome.fills):
            legs.append(
                _leg(f"PROFIT/{action.tier}", "profit", fill, entry_price)
            )

    return legs


def _leg(label: str, kind: str, fill: object, entry_price: float) -> TradeLeg:
    """Assemble one :class:`TradeLeg` from a fill, computing its realised PnL."""
    tokens = float(getattr(fill, "requested_tokens", 0.0))
    pnl = float(getattr(fill, "net_proceeds_usd", 0.0)) - tokens * entry_price
    return TradeLeg(
        label=label,
        kind=kind,
        tokens=tokens,
        price=float(getattr(fill, "quoted_price", 0.0)),
        net_proceeds_usd=float(getattr(fill, "net_proceeds_usd", 0.0)),
        pnl_usd=pnl,
        over_cap=bool(getattr(fill, "exceeded_slippage_cap", False)),
    )


def _describe_exit(
    session: "TradeSession",
) -> tuple[Optional[str], str, bool]:
    """Explain how the managed loop ended: (trigger, reason, fully_exited).

    Precedence follows the loop's own priority. A hard exit is terminal (the
    loop stops the instant one fires), so if any outcome took the ``hard_exit``
    path that is THE exit reason. Otherwise the loop simply ran out its bounded
    iterations with the position still open — report whether any profit tier
    fired so the moonbag/remaining is explained rather than looking abandoned.
    """
    outcomes = [o for o in session.loop_outcomes if o is not None]

    for outcome in outcomes:
        if outcome.path == "hard_exit" and outcome.exit_decision is not None:
            trigger = outcome.exit_decision.trigger
            reason = outcome.exit_decision.reason or "hard exit fired"
            return trigger, f"HARD EXIT [{trigger}] - {reason}", True

    tiers = [
        action.tier
        for outcome in outcomes
        if outcome.path == "profit_taking"
        for action in outcome.sell_actions
    ]
    if tiers:
        return (
            None,
            f"no hard exit; profit tiers {','.join(tiers)} sold, moonbag held",
            False,
        )

    ticks = len(outcomes)
    return (
        None,
        f"no exit trigger fired over {ticks} tick(s); position held open",
        False,
    )


def build_trade_report(session: "TradeSession") -> Optional[TradeReport]:
    """Build a :class:`TradeReport` for one entered position, or ``None``.

    Returns ``None`` for any session that never opened a position (a rejected,
    no-entry, or no-market-data candidate has no trade to report). For an
    entered session, walks every loop outcome to collect the executed sell legs,
    explains the exit, and totals the realised PnL. Pure — no side effects.
    """
    if session.status != _ENTERED_STATUS or session.position is None:
        return None

    position = session.position
    entry_price = position.entry_price
    decision = session.entry_decision

    legs: List[TradeLeg] = []
    for outcome in session.loop_outcomes:
        if outcome is None:
            continue
        legs.extend(_legs_from_outcome(outcome, entry_price))

    trigger, exit_reason, fully_exited = _describe_exit(session)

    entry_liquidity = (
        float(getattr(decision, "entry_liquidity", 0.0)) if decision else 0.0
    )
    entry_reason = str(getattr(decision, "reason", "")) if decision else ""

    return TradeReport(
        mint_address=session.mint_address,
        symbol=session.symbol,
        strategy=_strategy_name(decision),
        entry_reason=entry_reason,
        entry_price=entry_price,
        entry_size=position.original_size,
        entry_liquidity=entry_liquidity,
        legs=legs,
        exit_trigger=trigger,
        exit_reason=exit_reason,
        fully_exited=fully_exited,
        tokens_held=0.0 if fully_exited else position.remaining,
        realised_pnl_usd=sum(leg.pnl_usd for leg in legs),
        fill_count=len(legs),
    )


def _signed(amount: float) -> str:
    """Format a USD amount with an explicit sign, e.g. ``+$1.99`` / ``-$0.40``."""
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.2f}"


def format_trade_report(report: TradeReport) -> str:
    """Render a :class:`TradeReport` as a compact, ASCII-only text block.

    ASCII-only (hyphens, no em dashes / box glyphs) so it renders cleanly in a
    standard Windows console without encoding errors, matching the startup
    banner's constraint.
    """
    rule = "-" * 66
    lines: List[str] = [
        rule,
        f"TRADE {report.symbol} [{report.strategy}]  {report.mint_address}",
        (
            f"  entry : {report.entry_size:.6f} @ {report.entry_price:.8f}  "
            f"liq=${report.entry_liquidity:,.0f}"
            + (f"  ({report.entry_reason})" if report.entry_reason else "")
        ),
    ]

    if report.legs:
        lines.append("  sells :")
        for leg in report.legs:
            flag = "  OVER-CAP" if leg.over_cap else ""
            lines.append(
                f"    {leg.label:<26} {leg.tokens:.6f} @ {leg.price:.8f}  "
                f"net=${leg.net_proceeds_usd:.2f}  pnl={_signed(leg.pnl_usd)}{flag}"
            )
    else:
        lines.append("  sells : none executed")

    lines.append(f"  exit  : {report.exit_reason}")

    held = (
        "position fully closed"
        if report.fully_exited
        else f"{report.tokens_held:.6f} tokens held"
    )
    lines.append(
        f"  result: realised PnL {_signed(report.realised_pnl_usd)} over "
        f"{report.fill_count} fill(s); {held}"
    )
    lines.append(rule)
    return "\n".join(lines)


def log_trade_report(session: "TradeSession") -> Optional[TradeReport]:
    """Build and log the clean trade report for one session (if it entered).

    Returns the report (or ``None`` when there was no trade to report) so callers
    can also fold it into aggregate reporting without rebuilding it.
    """
    report = build_trade_report(session)
    if report is not None:
        logger.info("\n%s", format_trade_report(report))
    return report
