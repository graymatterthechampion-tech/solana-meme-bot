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
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import config
from strategies import hard_exit, profit_taking
from strategies.hard_exit import ExitDecision, MarketData
from strategies.profit_taking import Position, SellAction

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

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    logger.info("[startup] mode=%s network=%s", mode, config.SOLANA_NETWORK)

    if not args.dry_run:
        logger.info("[startup] LIVE mode requested — trading logic not implemented yet.")

    # TODO: drive process_position() from a live data feed (price, volume,
    # liquidity, candle, wallet activity) once the data stream is wired up.


if __name__ == "__main__":
    main()
