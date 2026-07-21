"""Performance summary over the persisted trade journal.

Reads the append-only journal written by :mod:`reporting.trade_journal` and
computes the after-the-fact performance picture mandated by the project's
reporting needs:

    1. Total trades (completed round-trips).
    2. Win / loss breakdown and win rate.
    3. Cumulative realised PnL (after simulated slippage/fees) and average
       PnL per trade.
    4. Exit-reason breakdown: how many exited via each hard-exit trigger
       (hard_stop_loss / liquidity_drop / coordinated_dump / flash_crash /
       volume_collapse) versus profit-taking tiers (2x / 5x / 10x).
    5. Average hold time per trade (loop iterations, and wall-clock seconds).
    6. Best and worst single trade.

Pure and read-only: it consumes an already-written journal file and prints a
compact, ASCII-only table. It never touches the network, a signer, or a
broadcast. Run it standalone::

    python -m reporting.journal_summary --path trades.jsonl
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from reporting.trade_journal import TradeRecord, load_records

# The canonical hard-exit triggers, in the order CLAUDE.md lists them, so the
# breakdown always shows every trigger (0 included) in a stable order.
HARD_EXIT_TRIGGERS: Tuple[str, ...] = (
    "hard_stop_loss",
    "liquidity_drop",
    "coordinated_dump",
    "flash_crash",
    "volume_collapse",
)

# The profit-taking tiers, ascending, so the breakdown is always complete.
PROFIT_TIERS: Tuple[str, ...] = ("2x", "5x", "10x")


@dataclass(frozen=True)
class BestWorst:
    """The extreme single trade in one direction."""

    symbol: str
    mint_address: str
    pnl_usd: float


@dataclass
class Summary:
    """Everything the printed performance table needs, precomputed."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate_pct: float = 0.0
    cumulative_pnl_usd: float = 0.0
    avg_pnl_usd: float = 0.0
    # Exit classification (mutually exclusive, one bucket per trade).
    hard_exit_counts: Dict[str, int] = field(default_factory=dict)
    profit_take_trades: int = 0     # >=1 profit tier fired, no hard exit
    held_open_trades: int = 0       # no hard exit and no tier fired
    # Tier firings (NOT mutually exclusive: one trade can fire several tiers).
    tier_firings: Dict[str, int] = field(default_factory=dict)
    avg_hold_ticks: float = 0.0
    avg_hold_seconds: float = 0.0
    best: Optional[BestWorst] = None
    worst: Optional[BestWorst] = None


def _classify_exit(record: TradeRecord) -> str:
    """Bucket a trade into exactly one exit class.

    Precedence mirrors the loop: a hard exit is terminal, so if one fired it is
    THE exit reason regardless of any earlier profit tier. Otherwise the trade
    is a profit-take (at least one tier fired, moonbag held) or was simply held
    open to the end of its managed window.
    """
    if record.exit_trigger:
        return f"hard_exit:{record.exit_trigger}"
    if record.profit_tiers:
        return "profit_take"
    return "held_open"


def summarize(records: List[TradeRecord]) -> Summary:
    """Compute the full performance :class:`Summary` from journalled trades.

    Pure aggregation over the record list; no I/O. An empty list yields a
    zeroed summary (every count 0, no best/worst) rather than an error.
    """
    summary = Summary()
    summary.hard_exit_counts = {t: 0 for t in HARD_EXIT_TRIGGERS}
    summary.tier_firings = {t: 0 for t in PROFIT_TIERS}

    if not records:
        return summary

    summary.total_trades = len(records)

    pnl_total = 0.0
    ticks_total = 0
    seconds_total = 0.0
    best: Optional[BestWorst] = None
    worst: Optional[BestWorst] = None

    for record in records:
        pnl = record.realised_pnl_usd
        pnl_total += pnl

        if pnl > 0:
            summary.wins += 1
        elif pnl < 0:
            summary.losses += 1
        else:
            summary.breakeven += 1

        bucket = _classify_exit(record)
        if bucket.startswith("hard_exit:"):
            trigger = bucket.split(":", 1)[1]
            # Guard against an unknown/legacy trigger name.
            summary.hard_exit_counts[trigger] = (
                summary.hard_exit_counts.get(trigger, 0) + 1
            )
        elif bucket == "profit_take":
            summary.profit_take_trades += 1
        else:
            summary.held_open_trades += 1

        for tier in record.profit_tiers:
            summary.tier_firings[tier] = summary.tier_firings.get(tier, 0) + 1

        ticks_total += record.hold_ticks
        seconds_total += record.hold_seconds

        extreme = BestWorst(record.symbol, record.mint_address, pnl)
        if best is None or pnl > best.pnl_usd:
            best = extreme
        if worst is None or pnl < worst.pnl_usd:
            worst = extreme

    summary.cumulative_pnl_usd = pnl_total
    summary.avg_pnl_usd = pnl_total / summary.total_trades
    # Win rate is wins over all trades taken; break-even/held trades are neither
    # a win nor a loss but still count as trades in the denominator.
    summary.win_rate_pct = 100.0 * summary.wins / summary.total_trades
    summary.avg_hold_ticks = ticks_total / summary.total_trades
    summary.avg_hold_seconds = seconds_total / summary.total_trades
    summary.best = best
    summary.worst = worst
    return summary


def _signed(amount: float) -> str:
    """Format a USD amount with an explicit sign, e.g. ``+$1.99`` / ``-$0.40``."""
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):,.2f}"


def format_summary(summary: Summary, *, source: str = "") -> str:
    """Render a :class:`Summary` as a compact, ASCII-only performance table.

    ASCII-only (hyphens, no box glyphs / em dashes) so it renders cleanly in a
    standard Windows console, matching the rest of the bot's reporting.
    """
    rule = "=" * 62
    thin = "-" * 62
    lines: List[str] = [rule, "TRADE JOURNAL PERFORMANCE SUMMARY"]
    if source:
        lines.append(f"source: {source}")
    lines.append(rule)

    if summary.total_trades == 0:
        lines.append("no completed trades journalled yet.")
        lines.append(rule)
        return "\n".join(lines)

    # 1 + 2. Totals, win/loss, win rate.
    wl = f"{summary.wins} / {summary.losses} / {summary.breakeven}"
    lines.append(f"  {'Total trades (round-trips)':<34}{summary.total_trades:>10}")
    lines.append(f"  {'Wins / Losses / Break-even':<34}{wl:>10}")
    lines.append(f"  {'Win rate':<34}{summary.win_rate_pct:>9.1f}%")
    lines.append(thin)

    # 3. PnL.
    lines.append(
        f"  {'Cumulative realised PnL':<34}"
        f"{_signed(summary.cumulative_pnl_usd):>10}"
    )
    lines.append(
        f"  {'Average PnL per trade':<34}{_signed(summary.avg_pnl_usd):>10}"
    )
    lines.append("  (realised = net of simulated slippage, fees, fill drift)")
    lines.append(thin)

    # 4. Exit-reason breakdown.
    lines.append("  Exit reasons (one bucket per trade):")
    for trigger in HARD_EXIT_TRIGGERS:
        count = summary.hard_exit_counts.get(trigger, 0)
        lines.append(f"    {('hard exit / ' + trigger):<32}{count:>10}")
    lines.append(f"    {'profit-taking (moonbag held)':<32}"
                 f"{summary.profit_take_trades:>10}")
    lines.append(f"    {'held open (no trigger)':<32}"
                 f"{summary.held_open_trades:>10}")
    lines.append("  Profit tiers fired (a trade may hit several):")
    for tier in PROFIT_TIERS:
        count = summary.tier_firings.get(tier, 0)
        lines.append(f"    {('tier ' + tier):<32}{count:>10}")
    lines.append(thin)

    # 5. Hold time.
    lines.append(
        f"  {'Avg hold per trade':<34}{summary.avg_hold_ticks:>7.1f} ticks"
    )
    lines.append(
        f"  {'  wall-clock equivalent':<34}{summary.avg_hold_seconds:>6.1f} s"
    )
    lines.append("  (1 tick = one management loop iteration = --interval s live)")
    lines.append(thin)

    # 6. Best / worst.
    if summary.best is not None:
        lines.append(
            f"  {'Best trade':<20}{summary.best.symbol:<10}"
            f"{_signed(summary.best.pnl_usd):>30}"
        )
    if summary.worst is not None:
        lines.append(
            f"  {'Worst trade':<20}{summary.worst.symbol:<10}"
            f"{_signed(summary.worst.pnl_usd):>30}"
        )
    lines.append(rule)
    return "\n".join(lines)


def build_summary(path: str) -> Tuple[Summary, int]:
    """Load the journal at ``path`` and summarise it. Returns (summary, count)."""
    records = load_records(path)
    return summarize(records), len(records)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the standalone summary tool."""
    parser = argparse.ArgumentParser(
        description="Summarise the persisted trade journal (read-only)."
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Journal file to read (default: config.TRADE_JOURNAL_PATH).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Print the performance summary for the configured (or given) journal."""
    args = parse_args(argv)
    path = args.path
    if path is None:
        import config

        path = config.TRADE_JOURNAL_PATH

    summary, count = build_summary(path)
    print(format_summary(summary, source=f"{path} ({count} trade(s))"))


if __name__ == "__main__":
    main()
