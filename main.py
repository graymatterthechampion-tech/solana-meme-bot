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
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional

import config
from data.market_feed import SCENARIOS, make_scenario_provider, mock_market_snapshot
from safety.rug_check import (
    SafetyReport,
    evaluate_token_safety,
    mock_safe_token_data,
)
from strategies import hard_exit, profit_taking
from strategies.entry import (
    EntryAction,
    EntryDecision,
    EntryMarketData,
    evaluate_entry,
)
from strategies.hard_exit import ExitDecision, MarketData
from strategies.profit_taking import Position, SellAction

# Async safety gate: mint -> read-only SafetyReport.
SafetyChecker = Callable[[str], Awaitable[SafetyReport]]

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


@dataclass
class TradeSession:
    """The full lifecycle outcome for one candidate token.

    ``status`` is one of:
        "rejected_safety" safety gate failed; entry never evaluated
        "no_entry"        safety passed but entry was WAIT/SKIP; no position
        "entered"         safety + BUY; a position was opened and the loop ran
    """

    mint_address: str
    status: str
    safety_report: Optional[SafetyReport] = None
    entry_decision: Optional[EntryDecision] = None
    position: Optional[Position] = None
    loop_outcomes: List[Optional[LoopOutcome]] = field(default_factory=list)


async def _default_safety_checker(mint_address: str) -> SafetyReport:
    """Default gate: the live read-only safety / rug check."""
    return await evaluate_token_safety(mint_address)


async def evaluate_and_trade(
    mint_address: str,
    entry_market: EntryMarketData,
    *,
    max_iterations: int,
    dry_run: bool = True,
    safety_checker: SafetyChecker = _default_safety_checker,
    snapshot_provider: SnapshotProvider = mock_market_snapshot,
) -> TradeSession:
    """Run the full candidate pipeline: safety gate -> entry -> manage loop.

    Order (every gate fail-closed; nothing is signed or broadcast):

        1. Safety gate FIRST. If ``evaluate_token_safety`` does not pass, log
           and SKIP — the token never reaches entry and no position is opened.
        2. Entry decision. Only a BUY opens a position; WAIT/SKIP open nothing.
        3. On BUY, hand the sized :class:`Position` to :func:`run_loop`, which
           manages it with the mandated hard-exit-first priority.

    Returns a :class:`TradeSession` describing how far the candidate got.
    """
    # --- 1. Safety gate FIRST ----------------------------------------------
    report = await safety_checker(mint_address)
    if not report.passed:
        logger.warning(
            "[trade] %s SKIP — failed safety gate (%s); entry not evaluated",
            mint_address,
            "; ".join(report.reasons) or "unsafe",
        )
        return TradeSession(
            mint_address=mint_address,
            status="rejected_safety",
            safety_report=report,
        )

    logger.info("[trade] %s passed safety gate -> evaluating entry", mint_address)

    # --- 2. Entry decision: only BUY opens a position ----------------------
    decision = await evaluate_entry(mint_address, entry_market, dry_run=dry_run)
    if decision.action is not EntryAction.BUY:
        logger.info(
            "[trade] %s no position opened (entry=%s: %s)",
            mint_address,
            decision.action.value,
            decision.reason,
        )
        return TradeSession(
            mint_address=mint_address,
            status="no_entry",
            safety_report=report,
            entry_decision=decision,
        )

    # --- 3. BUY: open the position and manage it in the loop ---------------
    position = decision.position
    assert position is not None  # guaranteed on a BUY decision
    logger.info(
        "[trade] %s ENTER — opened position size=%.6f @ %.8f (entry_liq=%.2f)",
        mint_address,
        position.original_size,
        position.entry_price,
        decision.entry_liquidity,
    )
    outcomes = await run_loop(
        position,
        token_address=mint_address,
        entry_liquidity=decision.entry_liquidity,
        max_iterations=max_iterations,
        dry_run=dry_run,
        snapshot_provider=snapshot_provider,
    )
    return TradeSession(
        mint_address=mint_address,
        status="entered",
        safety_report=report,
        entry_decision=decision,
        position=position,
        loop_outcomes=outcomes,
    )


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
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="flat",
        help="Scripted market path to feed the loop (default: flat).",
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

    # Demo run against no-network mocks so the WHOLE chain exercises end-to-end:
    # safety gate -> entry decision -> (on BUY) the managed loop.
    # TODO: replace the mock safety gather, demo entry market, and scenario
    # provider with the real data stream (Dexscreener/Birdeye/Helius) once wired.
    async def demo_safety_checker(mint: str) -> SafetyReport:
        return await evaluate_token_safety(mint, data_gatherer=mock_safe_token_data)

    demo_entry_price = 0.001
    entry_liquidity = 100_000.0
    # A BUY-eligible candidate: proven runner (ATH 5h ago, 10x volume spike),
    # ~40% pullback, stabilised — so the demo reaches the loop.
    now = time.time()
    demo_market = EntryMarketData(
        current_price=demo_entry_price * 0.60,
        ath_price=demo_entry_price,
        ath_timestamp=now - 5 * 3600.0,
        price_history=[
            demo_entry_price * m
            for m in (1.0, 0.9, 0.75, 0.62, 0.61, 0.60, 0.605, 0.60)
        ],
        volume_history=[100.0, 100.0, 100.0, 100.0, 1000.0],
        current_liquidity=entry_liquidity,
        pre_dip_liquidity=entry_liquidity * 1.1,
        now=now,
    )
    # The loop manages the opened position along the chosen scenario path.
    provider = make_scenario_provider(
        args.scenario,
        entry_price=demo_entry_price * 0.60,
        entry_liquidity=entry_liquidity,
    )
    logger.info("[startup] scenario=%s", args.scenario)
    session = asyncio.run(
        evaluate_and_trade(
            "DEMO_MINT",
            demo_market,
            max_iterations=args.iterations,
            dry_run=args.dry_run,
            safety_checker=demo_safety_checker,
            snapshot_provider=provider,
        )
    )
    logger.info(
        "[shutdown] status=%s ran %d loop iterations (scenario=%s)",
        session.status,
        len(session.loop_outcomes),
        args.scenario,
    )


if __name__ == "__main__":
    main()
