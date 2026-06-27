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
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import config
from data.market_feed import get_live_market_snapshot, mock_market_snapshot
from data.price_history import (
    OHLCVHistory,
    build_entry_market_data,
    get_price_history,
)
from execution.fill_simulator import FillResult, simulate_fill
from safety.rug_check import (
    SafetyReport,
    evaluate_token_safety,
)
from scanner.token_scanner import scan_candidates
from strategies import hard_exit, profit_taking
from strategies.entry import (
    EntryAction,
    EntryDecision,
    EntryMarketData,
    evaluate_entry,
)
from strategies.hard_exit import ExitDecision, MarketData
from strategies.profit_taking import Position, SellAction

# Default cap on how many surfaced candidates to evaluate per scan.
DEFAULT_MAX_CANDIDATES: int = 5

# Async safety gate: mint -> read-only SafetyReport.
SafetyChecker = Callable[[str], Awaitable[SafetyReport]]
# Async candidate discovery: () -> list of candidate dicts (mint + stats).
CandidateScanner = Callable[[], Awaitable[List[Dict[str, Any]]]]
# Async per-candidate detailed market builder for the entry decision.
EntryMarketProvider = Callable[[Dict[str, Any]], Awaitable[Optional[EntryMarketData]]]
# Async read-only OHLCV source: mint -> recent history (or None, fail-closed).
HistoryFetcher = Callable[[str], Awaitable[Optional[OHLCVHistory]]]

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
    fills: List[FillResult] = field(default_factory=list)


def _simulate_sell(
    position: Position, tokens: float, market_data: MarketData, label: str
) -> FillResult:
    """Model the realistic outcome of one dry-run sell and log net proceeds/PnL.

    Uses the live snapshot's price and pool liquidity so the simulated fill
    reflects real price impact + fees + fill-delay drift (see
    :mod:`execution.fill_simulator`). Realised PnL is net proceeds minus the
    cost basis (tokens * entry price). Nothing is signed or broadcast.
    """
    fill = simulate_fill(
        tokens, market_data.current_price, market_data.current_liquidity
    )
    pnl = fill.net_proceeds_usd - tokens * position.entry_price
    logger.info(
        "[fill] %s tokens=%.6f @ %.8f notional=$%.2f slippage=%.2f%%%s "
        "net=$%.2f pnl=$%.2f",
        label, tokens, market_data.current_price, fill.notional_usd,
        fill.total_slippage_pct * 100,
        " OVER-CAP" if fill.exceeded_slippage_cap else "",
        fill.net_proceeds_usd, pnl,
    )
    return fill


def realised_pnl_usd(
    outcomes: List[Optional[LoopOutcome]], entry_price: float
) -> float:
    """Sum simulated realised PnL (net proceeds - cost basis) across all fills."""
    total = 0.0
    for outcome in outcomes:
        if outcome is None:
            continue
        for fill in outcome.fills:
            total += fill.net_proceeds_usd - fill.requested_tokens * entry_price
    return total


async def process_position(
    position: Position,
    market_data: MarketData,
    dry_run: bool = True,
) -> LoopOutcome:
    """Evaluate one position for one loop iteration in the mandated order.

    Hard-exit checks run first. If ``should_exit`` is True the full exit is
    executed (dry-run logged inside ``evaluate_hard_exit``) and profit-taking
    is skipped entirely for this position. Otherwise profit-taking runs. Each
    sell is run through the fill simulator so dry-run PnL reflects real friction.

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
        fill = _simulate_sell(
            position, exit_decision.tokens_to_sell, market_data,
            f"HARD_EXIT/{exit_decision.trigger}",
        )
        return LoopOutcome(
            path="hard_exit", exit_decision=exit_decision, fills=[fill]
        )

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
        fills = [
            _simulate_sell(position, a.tokens_to_sell, market_data, f"PROFIT/{a.tier}")
            for a in sell_actions
        ]
        return LoopOutcome(
            path="profit_taking", sell_actions=sell_actions, fills=fills
        )

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
        "no_market_data"  no per-candidate market data could be built (fail-
                          closed); safety/entry never evaluated
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
    pnl = realised_pnl_usd(outcomes, position.entry_price)
    n_fills = sum(len(o.fills) for o in outcomes if o is not None)
    logger.info(
        "[trade] %s managed -> simulated realised PnL = $%.2f over %d fill(s)",
        mint_address, pnl, n_fills,
    )
    return TradeSession(
        mint_address=mint_address,
        status="entered",
        safety_report=report,
        entry_decision=decision,
        position=position,
        loop_outcomes=outcomes,
    )


async def _default_entry_market_provider(
    candidate: Dict[str, Any],
    *,
    history_fetcher: HistoryFetcher = get_price_history,
) -> Optional[EntryMarketData]:
    """Build per-candidate :class:`EntryMarketData` for the entry decision.

    The scanner surfaces a point-in-time snapshot (price / liquidity / market
    cap) but NOT the OHLCV history the post-pump dip-buy entry needs. This
    fetches recent history via the read-only price-data layer
    (:func:`data.price_history.get_price_history` — Birdeye primary, Dexscreener
    fallback) and assembles it into the entry inputs. The fetch is fail-closed:
    on any error / insufficient data it yields no history, and
    :func:`data.price_history.build_entry_market_data` then produces an
    EMPTY-history snapshot so ``evaluate_entry`` fails closed to SKIP rather
    than guessing. ``history_fetcher`` is injectable for tests.

    Returns ``None`` (fail-closed) only if the candidate carries no usable price.
    """
    try:
        price = float(candidate.get("price_usd") or 0.0)
        liquidity = float(candidate.get("liquidity_usd") or 0.0)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None

    mint = str(candidate.get("mint") or "")
    history = await history_fetcher(mint) if mint else None
    return build_entry_market_data(
        current_price=price,
        current_liquidity=liquidity,
        market_cap_usd=candidate.get("market_cap_usd"),
        history=history,
    )


async def scan_and_evaluate(
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_iterations: int = 5,
    dry_run: bool = True,
    scanner: CandidateScanner = scan_candidates,
    safety_checker: SafetyChecker = _default_safety_checker,
    entry_market_provider: EntryMarketProvider = _default_entry_market_provider,
    snapshot_provider: SnapshotProvider = mock_market_snapshot,
) -> List[TradeSession]:
    """Discover live candidates and run each through the full pipeline.

    Read-only / dry-run discovery + evaluation; nothing is signed or broadcast.
    Steps:

        1. ``scanner()`` surfaces the live candidate list (fail-closed to []).
        2. Process at most ``max_candidates`` of them (a hard cap on per-scan
           work). Candidates are scanner-sorted by 24h volume (highest first).
        3. For each, build per-candidate market data; on ``None`` fail closed
           and record "no_market_data". Otherwise run :func:`evaluate_and_trade`
           (safety gate -> entry -> managed loop) and record its outcome.

    Logs a per-candidate line showing how far each got and an aggregate summary
    of the terminal statuses. Returns one :class:`TradeSession` per candidate
    that was processed (i.e. up to the cap).
    """
    cap = max(0, max_candidates)
    candidates = await scanner()
    total = len(candidates)
    if not candidates:
        logger.info("[scan] no candidates surfaced -> nothing to evaluate")
        return []

    selected = candidates[:cap]
    logger.info(
        "[scan] surfaced %d candidate(s); evaluating %d (cap=%d)",
        total,
        len(selected),
        cap,
    )

    sessions: List[TradeSession] = []
    for idx, candidate in enumerate(selected, start=1):
        mint = str(candidate.get("mint") or "")
        symbol = str(candidate.get("symbol") or "?")
        if not mint:
            logger.warning(
                "[scan %d/%d] candidate missing mint -> SKIP", idx, len(selected)
            )
            continue

        entry_market = await entry_market_provider(candidate)
        if entry_market is None:
            logger.warning(
                "[scan %d/%d] %s (%s) no market data -> SKIP (fail-closed)",
                idx,
                len(selected),
                symbol,
                mint,
            )
            sessions.append(
                TradeSession(mint_address=mint, status="no_market_data")
            )
            continue

        session = await evaluate_and_trade(
            mint,
            entry_market,
            max_iterations=max_iterations,
            dry_run=dry_run,
            safety_checker=safety_checker,
            snapshot_provider=snapshot_provider,
        )
        logger.info(
            "[scan %d/%d] %s (%s) -> %s",
            idx,
            len(selected),
            symbol,
            mint,
            session.status,
        )
        sessions.append(session)

    summary = Counter(s.status for s in sessions)
    logger.info(
        "[scan] done: processed %d candidate(s) -> %s",
        len(sessions),
        dict(summary),
    )
    return sessions


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
        "--max-candidates",
        dest="max_candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help=(
            "Max surfaced candidates to evaluate per scan "
            f"(default: {DEFAULT_MAX_CANDIDATES})."
        ),
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

    # Live, read-only discovery -> per-candidate dry-run evaluation. The scanner
    # surfaces real Dexscreener candidates; each runs through the full pipeline
    # (safety gate -> real-OHLCV entry decision -> on BUY, the managed loop on
    # the live market feed, with simulated-friction fills). Nothing is ever
    # signed or broadcast.
    logger.info(
        "[scan] discovering candidates (read-only); cap=%d", args.max_candidates
    )
    sessions = asyncio.run(
        scan_and_evaluate(
            max_candidates=args.max_candidates,
            max_iterations=args.iterations,
            dry_run=args.dry_run,
            snapshot_provider=get_live_market_snapshot,
        )
    )
    summary = Counter(s.status for s in sessions)
    logger.info(
        "[shutdown] processed %d candidate(s): %s",
        len(sessions),
        dict(summary),
    )


if __name__ == "__main__":
    main()
