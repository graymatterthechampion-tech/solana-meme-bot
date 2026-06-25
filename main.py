"""Solana meme bot entry point.

Wires the strategy modules into a single per-position evaluation step with a
strict ordering mandated by CLAUDE.md:

    1. evaluate_hard_exit() runs FIRST. If it signals an exit, execute the full
       exit (dry-run log) and SKIP profit-taking entirely for that position.
    2. Only if no hard exit fires, evaluate_take_profit() runs.

Safety: the bot runs in dry-run mode by default. Live execution requires the
explicit `--live` flag. Before any live network commands, confirm devnet vs
mainnet-beta.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import config
from data.market_feed import mock_market_snapshot
from strategies import hard_exit, profit_taking
from strategies.hard_exit import ExitDecision, MarketData
from strategies.profit_taking import Position, SellAction

# Async source that yields a snapshot (or None, fail-closed) for a token.
SnapshotProvider = Callable[[str, float], Awaitable[Optional[MarketData]]]

logger = logging.getLogger(__name__)


@dataclass
class LoopOutcome:
    """The result of evaluating one position for one loop iteration.

    ``path`` is one of:
        "hard_exit"     a hard-exit trigger fired; full exit, profit skipped
        "profit_taking" no hard exit; one or more profit tiers fired
        "hold"          no hard exit and no profit tier fired
    """

    path: str
    exit_decision: Optional[ExitDecision] = None
    sell_actions: List[SellAction] = field(default_factory=list)


async def process_position(
    position: Position,
    market_data: MarketData,
    dry_run: bool = True,
) -> LoopOutcome:
    """Evaluate one position for one loop iteration in the mandated order.

    Hard-exit checks run first. If ``should_exit`` is True the full exit is
    executed (dry-run logged inside ``evaluate_hard_exit``) and profit-taking
    is skipped entirely for this position. Otherwise profit-taking runs.

    Calls ``profit_taking.evaluate_take_profit`` via the module attribute (not
    a bound import) so the ordering guarantee stays observable/testable.
    """
    # --- 1. Hard-exit checks FIRST ------------------------------------------
    exit_decision = await hard_exit.evaluate_hard_exit(
        position, market_data, dry_run=dry_run
    )
    if exit_decision.should_exit:
        logger.warning(
            "[loop] path=HARD_EXIT trigger=%s reason=%s tokens=%.6f "
            "-> full exit, profit-taking SKIPPED",
            exit_decision.trigger,
            exit_decision.reason,
            exit_decision.tokens_to_sell,
        )
        return LoopOutcome(path="hard_exit", exit_decision=exit_decision)

    # --- 2. Profit-taking only if no hard exit fired ------------------------
    sell_actions = await profit_taking.evaluate_take_profit(
        position, market_data.current_price, dry_run=dry_run
    )
    if sell_actions:
        logger.info(
            "[loop] path=PROFIT_TAKING tiers=%s remaining=%.6f",
            [a.tier for a in sell_actions],
            position.remaining,
        )
        return LoopOutcome(path="profit_taking", sell_actions=sell_actions)

    logger.info(
        "[loop] path=HOLD no hard exit, no profit tier (price=%.8f remaining=%.6f)",
        market_data.current_price,
        position.remaining,
    )
    return LoopOutcome(path="hold")


async def run_loop(
    position: Position,
    token_address: str,
    entry_liquidity: float,
    *,
    max_iterations: int,
    dry_run: bool = True,
    interval: float = 0.0,
    snapshot_provider: SnapshotProvider = mock_market_snapshot,
) -> List[Optional[LoopOutcome]]:
    """Run the trading loop for a bounded number of iterations.

    Each iteration: fetch a snapshot, SKIP on ``None`` (fail-closed — never
    trade on incomplete data), otherwise evaluate via :func:`process_position`.
    Returns one entry per iteration (``None`` for a skipped/fail-closed tick).

    ``max_iterations`` bounds the loop so it never runs forever. The loop stops
    early once a hard exit fires, since the position is then fully closed.
    ``snapshot_provider`` is injectable for testing; it defaults to the
    no-network mock source.
    """
    outcomes: List[Optional[LoopOutcome]] = []

    for i in range(max_iterations):
        snapshot = await snapshot_provider(token_address, entry_liquidity)

        if snapshot is None:
            logger.warning(
                "[loop %d/%d] no snapshot (fail-closed) -> SKIP, no trade",
                i + 1,
                max_iterations,
            )
            outcomes.append(None)
        else:
            outcome = await process_position(position, snapshot, dry_run=dry_run)
            logger.info(
                "[loop %d/%d] outcome=%s", i + 1, max_iterations, outcome.path
            )
            outcomes.append(outcome)

            if outcome.path == "hard_exit":
                logger.info(
                    "[loop %d/%d] position fully exited -> stopping loop",
                    i + 1,
                    max_iterations,
                )
                break

        if interval > 0:
            await asyncio.sleep(interval)

    return outcomes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Dry-run defaults to ON. `--dry-run` is accepted for explicitness, while
    `--live` is the explicit opt-in required to disable dry-run.
    """
    parser = argparse.ArgumentParser(description="Solana meme bot")

    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Simulate without broadcasting transactions (default: ON).",
    )
    parser.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        help="Disable dry-run and execute live transactions (explicit opt-in).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of loop iterations to run against the mock feed (default: 5).",
    )

    return parser.parse_args()


# 5-row ASCII block glyphs, all uniform-width, for the startup banner.
_GLYPHS: dict[str, List[str]] = {
    "H": ["H   H", "H   H", "HHHHH", "H   H", "H   H"],
    "E": ["EEEEE", "E    ", "EEEE ", "E    ", "EEEEE"],
    "L": ["L    ", "L    ", "L    ", "L    ", "LLLLL"],
    "I": ["IIIII", "  I  ", "  I  ", "  I  ", "IIIII"],
    "X": ["X   X", " X X ", "  X  ", " X X ", "X   X"],
    "V": ["V   V", "V   V", "V   V", " V V ", "  V  "],
    "C": [" CCCC", "C    ", "C    ", "C    ", " CCCC"],
    "T": ["TTTTT", "  T  ", "  T  ", "  T  ", "  T  "],
    "O": [" OOO ", "O   O", "O   O", "O   O", " OOO "],
    "R": ["RRRR ", "R   R", "RRRR ", "R  R ", "R   R"],
    " ": ["     ", "     ", "     ", "     ", "     "],
}


def _render_block(text: str) -> List[str]:
    """Render text into 5 rows of ASCII block letters (alignment guaranteed)."""
    return [" ".join(_GLYPHS[ch][row] for ch in text) for row in range(5)]


def print_banner(mode: str, network: str) -> None:
    """Print the startup banner once. Cosmetic only; uses plain print().

    ASCII-only (hyphen, not em dash) so it renders safely in a standard
    ~80-col terminal without Windows console encoding errors.
    """
    rule = "=" * 70
    print()
    print(rule)
    for line in _render_block("HELIX VECTOR"):
        print(line)
    print()
    print("        v1.0 - Solana Execution Bot")
    print(f"        mode: {mode}   network: {network}")
    print(rule)
    print()
    # Flush so the banner lands before any (stderr) log output when piped.
    sys.stdout.flush()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print_banner(mode, config.SOLANA_NETWORK)
    logger.info("[startup] Helix Vector 1.0 | mode=%s network=%s", mode, config.SOLANA_NETWORK)

    if not args.dry_run:
        logger.info("[startup] LIVE mode requested — trading logic not implemented yet.")

    # Demo run against the no-network mock feed so the loop exercises end-to-end.
    # TODO: replace mock_market_snapshot with data.market_feed.get_market_snapshot
    # (real Dexscreener/Birdeye/Helius) once the data stream is wired up.
    demo_position = Position(entry_price=0.001, original_size=1000.0)
    outcomes = asyncio.run(
        run_loop(
            demo_position,
            token_address="DEMO_MINT",
            entry_liquidity=100_000.0,
            max_iterations=args.iterations,
            dry_run=args.dry_run,
        )
    )
    logger.info("[shutdown] ran %d iterations against mock feed", len(outcomes))


if __name__ == "__main__":
    main()
