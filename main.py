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
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import config
from data.market_feed import get_live_market_snapshot, mock_market_snapshot
from data.price_history import (
    OHLCVHistory,
    build_entry_market_data,
    get_price_history,
)
from execution.fill_simulator import FillResult, simulate_fill
from reporting.journal_summary import build_summary, format_summary
from reporting.trade_journal import append_trade, load_records
from reporting.trade_report import log_trade_report
from safety.reentry import ReentryGuard
from safety.rug_check import (
    SafetyReport,
    evaluate_token_safety,
)
from scanner.meta_rank import MetaBoost, NEUTRAL_BOOST, rank_boost_for
from scanner.token_scanner import scan_candidates
from strategies import hard_exit, profit_taking
from strategies.entry import (
    EntryAction,
    EntryDecision,
    EntryMarketData,
    evaluate_entry,
)
from strategies.entry_momentum import MomentumDecision, evaluate_entry_momentum
from strategies.hard_exit import ExitDecision, MarketData
from strategies.profit_taking import Position, SellAction

# Default cap on how many surfaced candidates to evaluate per scan.
DEFAULT_MAX_CANDIDATES: int = 5

# Default seconds to sleep between cycles in continuous (--loop) mode.
DEFAULT_INTERVAL_SECONDS: float = 60.0

# --- Entry strategy selection ----------------------------------------------
STRATEGY_DIP: str = "dip"              # post-pump dip-buy only
STRATEGY_MOMENTUM: str = "momentum"    # momentum/breakout only
STRATEGY_BOTH: str = "both"            # try dip first, then momentum; first BUY wins
STRATEGY_CHOICES: tuple[str, ...] = (STRATEGY_DIP, STRATEGY_MOMENTUM, STRATEGY_BOTH)
DEFAULT_STRATEGY: str = STRATEGY_BOTH

# Either entry strategy's decision; both share the ``action`` / ``position`` /
# ``entry_liquidity`` shape the pipeline reads, so callers treat them uniformly.
AnyEntryDecision = Union[EntryDecision, MomentumDecision]

# How informative a non-BUY decision's reason is. A WAIT is a near-miss ("valid
# setup, timing not ready") and explains the pass-on better than a SKIP
# ("did not qualify"), so it ranks higher. Used only to pick which strategy's
# reason to surface in "both" mode when neither returns BUY.
_ACTION_INFORMATIVENESS: Dict[EntryAction, int] = {
    EntryAction.SKIP: 0,
    EntryAction.WAIT: 1,
    EntryAction.BUY: 2,
}

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

# Async meta ranking boost for an already-entered token: (mint, entry_price) ->
# MetaBoost (>= 1.0). Injectable; the default is a no-network neutral boost so
# library/test use never touches Helius. main() wires the live provider.
MetaBoostProvider = Callable[[str, float], Awaitable[MetaBoost]]

# Async one-cycle scan+evaluate (defaults to scan_and_evaluate); injectable in
# tests. Accepts the scan_and_evaluate keyword arguments and returns its sessions.
ContinuousScan = Callable[..., Awaitable[List["TradeSession"]]]
# Async sleep between cycles (defaults to asyncio.sleep); injectable in tests.
AsyncSleep = Callable[[float], Awaitable[None]]

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
    # Per-fill detail is DEBUG: the clean end-of-trade TradeReport summarises
    # every sell, so this line is only needed when tracing a single fill.
    logger.debug(
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
        # Per-tick path detail is DEBUG; the exit and its reason are surfaced
        # cleanly in the end-of-trade TradeReport (exit_reason).
        logger.debug(
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
        logger.debug(
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

    logger.debug(
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
            logger.debug(
                "[loop %d/%d] outcome=%s", i + 1, max_iterations, outcome.path
            )
            outcomes.append(outcome)

            if outcome.path == "hard_exit":
                logger.debug(
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
        "reentry_blocked" the re-entry guard skipped this mint (cooldown or
                          blacklist); safety/entry never evaluated
        "rejected_safety" safety gate failed; entry never evaluated
        "no_entry"        safety passed but entry was WAIT/SKIP; no position
        "entered"         safety + BUY; a position was opened and the loop ran

    ``meta_boost`` / ``priority_score`` are populated ONLY for "entered" tokens
    (safe + momentum-backed BUY). The meta layer is a ranking boost, never a
    trigger: it is computed strictly after the BUY and feeds nothing upstream.
    ``priority_score`` is the base priority times the meta boost; it is 0.0 for
    every non-entered session (nothing was bought to rank).
    """

    mint_address: str
    status: str
    safety_report: Optional[SafetyReport] = None
    entry_decision: Optional[AnyEntryDecision] = None
    position: Optional[Position] = None
    loop_outcomes: List[Optional[LoopOutcome]] = field(default_factory=list)
    symbol: str = "?"
    meta_boost: Optional[MetaBoost] = None
    priority_score: float = 0.0


# Base priority every entered token starts from before the meta boost is applied.
META_BASE_PRIORITY: float = 1.0


def meta_ranked_entries(sessions: List["TradeSession"]) -> List["TradeSession"]:
    """Return the ENTERED sessions ordered by meta-boosted priority (desc).

    Only tokens that actually opened a position are rankable; everything else has
    no priority to compare. The sort is stable, so equal-priority entries keep
    their original (discovery) order. This is a pure overlay — it never drops or
    re-orders the source ``sessions`` list itself.
    """
    entered = [s for s in sessions if s.status == "entered"]
    entered.sort(key=lambda s: s.priority_score, reverse=True)
    return entered


async def _default_safety_checker(mint_address: str) -> SafetyReport:
    """Default gate: the live read-only safety / rug check."""
    return await evaluate_token_safety(mint_address)


async def _neutral_meta_boost_provider(mint_address: str, price: float) -> MetaBoost:
    """Default meta provider: a no-network neutral boost (1.0).

    Keeps library/test use fully hermetic — the meta ranking is inert unless a
    live provider is explicitly wired in (see :func:`_live_meta_boost_provider`,
    used by ``main()``). A neutral boost can only ever leave an entry un-ranked,
    never block it.
    """
    return NEUTRAL_BOOST


async def _live_meta_boost_provider(mint_address: str, price: float) -> MetaBoost:
    """Live meta provider: read-only Helius buyer-breadth + KOL-holdings boost.

    Fail-closed (:func:`scanner.meta_rank.rank_boost_for` never raises and
    returns a neutral 1.0 on any error / missing endpoint).
    """
    return await rank_boost_for(mint_address, price)


async def evaluate_entry_strategy(
    mint_address: str,
    entry_market: EntryMarketData,
    *,
    strategy: str = DEFAULT_STRATEGY,
    dry_run: bool = True,
) -> AnyEntryDecision:
    """Run the selected entry strategy (or both) and return one decision.

    * ``"dip"``      — only the post-pump dip-buy (:func:`evaluate_entry`).
    * ``"momentum"`` — only the breakout (:func:`evaluate_entry_momentum`).
    * ``"both"``     — evaluate the dip FIRST, then momentum, and take the first
      that returns BUY. If neither BUYs, surface the MORE INFORMATIVE decision
      (a WAIT near-miss over a SKIP rejection), tie-breaking to the dip (the
      primary strategy) so the reported reason is deterministic.

    ``evaluate_entry`` is looked up via the module global so tests that patch
    ``main.evaluate_entry`` still intercept it.
    """
    if strategy == STRATEGY_MOMENTUM:
        return await evaluate_entry_momentum(mint_address, entry_market, dry_run=dry_run)
    if strategy == STRATEGY_DIP:
        return await evaluate_entry(mint_address, entry_market, dry_run=dry_run)

    # "both": dip is primary; the first BUY wins.
    dip_decision = await evaluate_entry(mint_address, entry_market, dry_run=dry_run)
    if dip_decision.action is EntryAction.BUY:
        return dip_decision
    momentum_decision = await evaluate_entry_momentum(
        mint_address, entry_market, dry_run=dry_run
    )
    if momentum_decision.action is EntryAction.BUY:
        return momentum_decision

    # Neither bought: report whichever pass-on reason is more informative. The
    # dip is compared first, so an equally-ranked momentum decision does NOT
    # displace it (deterministic tie-break to the primary strategy).
    if (
        _ACTION_INFORMATIVENESS[momentum_decision.action]
        > _ACTION_INFORMATIVENESS[dip_decision.action]
    ):
        return momentum_decision
    return dip_decision


async def evaluate_and_trade(
    mint_address: str,
    entry_market: EntryMarketData,
    *,
    max_iterations: int,
    dry_run: bool = True,
    strategy: str = DEFAULT_STRATEGY,
    symbol: str = "?",
    safety_checker: SafetyChecker = _default_safety_checker,
    snapshot_provider: SnapshotProvider = mock_market_snapshot,
    meta_boost_provider: MetaBoostProvider = _neutral_meta_boost_provider,
    journal_path: Optional[str] = None,
) -> TradeSession:
    """Run the full candidate pipeline: safety gate -> entry -> manage loop.

    Order (every gate fail-closed; nothing is signed or broadcast):

        1. Safety gate FIRST. If ``evaluate_token_safety`` does not pass, log
           and SKIP — the token never reaches entry and no position is opened.
        2. Entry decision. Only a BUY opens a position; WAIT/SKIP open nothing.
        3. On BUY, hand the sized :class:`Position` to :func:`run_loop`, which
           manages it with the mandated hard-exit-first priority.
        4. ONLY on a completed BUY, compute the meta ranking boost
           (``meta_boost_provider``) from read-only on-chain signals. This is a
           pure ranking annotation — it runs after the trade is decided and
           feeds nothing upstream, so it can never trigger a buy or relax the
           safety/entry bars. It is fail-closed to a neutral 1.0.

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
            symbol=symbol,
        )

    logger.debug(
        "[trade] %s passed safety gate -> evaluating entry (strategy=%s)",
        mint_address, strategy,
    )

    # --- 2. Entry decision: only BUY opens a position ----------------------
    decision = await evaluate_entry_strategy(
        mint_address, entry_market, strategy=strategy, dry_run=dry_run
    )
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
            symbol=symbol,
        )

    # --- 3. BUY: open the position and manage it in the loop ---------------
    position = decision.position
    assert position is not None  # guaranteed on a BUY decision
    logger.debug(
        "[trade] %s ENTER — opened position size=%.6f @ %.8f (entry_liq=%.2f)",
        mint_address,
        position.original_size,
        position.entry_price,
        decision.entry_liquidity,
    )
    entry_time = datetime.now(timezone.utc).isoformat()
    outcomes = await run_loop(
        position,
        token_address=mint_address,
        entry_liquidity=decision.entry_liquidity,
        max_iterations=max_iterations,
        dry_run=dry_run,
        snapshot_provider=snapshot_provider,
    )
    exit_time = datetime.now(timezone.utc).isoformat()

    # --- 4. META RANKING (boost only; strictly AFTER the BUY is decided) ----
    # Consumes read-only on-chain signals (buyer breadth + KOL holdings). It can
    # only ever RAISE this entry's priority relative to other entries; it is
    # fail-closed to a neutral 1.0 and can never veto, block, or shrink a token.
    boost = await meta_boost_provider(mint_address, position.entry_price)
    priority = META_BASE_PRIORITY * boost.boost
    logger.debug(
        "[meta] %s (%s) rank boost=x%.2f priority=%.2f | %s",
        mint_address, symbol, boost.boost, priority, boost.reason,
    )
    session = TradeSession(
        mint_address=mint_address,
        status="entered",
        safety_report=report,
        entry_decision=decision,
        position=position,
        loop_outcomes=outcomes,
        symbol=symbol,
        meta_boost=boost,
        priority_score=priority,
    )
    # Emit the one clean, human-readable trade report for this position: entry,
    # every simulated sell with its PnL, why it exited, and the realised total.
    # This is the INFO-level signal that replaces the per-tick chatter above.
    report = log_trade_report(session)

    # Durably journal the completed trade (best-effort, append-only) so it can
    # be analysed after the run and across runs. Only ever writes a local file;
    # nothing is signed or broadcast. Disabled unless a journal path is given
    # (library/test callers pass None and stay hermetic).
    if report is not None and journal_path:
        append_trade(
            report,
            journal_path,
            dry_run=dry_run,
            entry_time=entry_time,
            exit_time=exit_time,
            hold_ticks=len(outcomes),
        )
    return session


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
    strategy: str = DEFAULT_STRATEGY,
    scanner: CandidateScanner = scan_candidates,
    safety_checker: SafetyChecker = _default_safety_checker,
    entry_market_provider: EntryMarketProvider = _default_entry_market_provider,
    snapshot_provider: SnapshotProvider = mock_market_snapshot,
    meta_boost_provider: MetaBoostProvider = _neutral_meta_boost_provider,
    journal_path: Optional[str] = None,
    reentry_guard: Optional[ReentryGuard] = None,
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

        # Re-entry guard (runs BEFORE any network fetch / safety / entry work):
        # skip mints we recently traded (cooldown) or that burned us (blacklist).
        if reentry_guard is not None:
            skip_reason = reentry_guard.check(mint, datetime.now(timezone.utc))
            if skip_reason is not None:
                logger.info(
                    "[scan %d/%d] %s (%s) -> SKIP re-entry: %s",
                    idx, len(selected), symbol, mint, skip_reason,
                )
                sessions.append(
                    TradeSession(mint_address=mint, status="reentry_blocked", symbol=symbol)
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
                TradeSession(mint_address=mint, status="no_market_data", symbol=symbol)
            )
            continue

        session = await evaluate_and_trade(
            mint,
            entry_market,
            max_iterations=max_iterations,
            dry_run=dry_run,
            strategy=strategy,
            symbol=symbol,
            safety_checker=safety_checker,
            snapshot_provider=snapshot_provider,
            meta_boost_provider=meta_boost_provider,
            journal_path=journal_path,
        )
        # Feed the guard: an entry arms the cooldown; any hard exit this trade
        # took counts toward the blacklist. Only "entered" sessions opened a
        # position, so nothing else can move the guard's state.
        if reentry_guard is not None and session.status == "entered":
            now = datetime.now(timezone.utc)
            reentry_guard.record_entry(mint, now)
            for outcome in session.loop_outcomes:
                if (outcome is not None and outcome.path == "hard_exit"
                        and outcome.exit_decision is not None):
                    reentry_guard.record_hard_exit(
                        mint, outcome.exit_decision.trigger, now
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

    # Meta-boosted priority order of the genuine (entered) setups. Ranking only;
    # the returned list stays in discovery order and no entry is added/removed.
    ranked = meta_ranked_entries(sessions)
    if ranked:
        logger.info(
            "[meta] entry priority (best first): %s",
            [
                f"{s.symbol}:{s.mint_address}=x{(s.meta_boost.boost if s.meta_boost else 1.0):.2f}"
                for s in ranked
            ],
        )
    return sessions


def session_pnl(session: TradeSession) -> float:
    """Simulated realised PnL (USD) for one candidate's managed loop.

    Zero unless a position was actually opened (``status == "entered"``); for
    every other terminal status no fills occurred, so there is nothing to sum.
    """
    if session.position is None:
        return 0.0
    return realised_pnl_usd(session.loop_outcomes, session.position.entry_price)


@dataclass
class CumulativeStats:
    """Running totals across every cycle of a continuous (--loop) run.

    Accumulated in place so the totals survive even if the run is interrupted
    mid-cycle (the caller keeps a reference and reports it on shutdown):

        cycles          completed scan/evaluate cycles
        total_candidates candidates processed across all cycles
        status_counts   per-terminal-status tallies (entered, no_entry, ...)
        total_pnl_usd   summed simulated realised PnL across all cycles
    """

    cycles: int = 0
    total_candidates: int = 0
    status_counts: Counter = field(default_factory=Counter)
    total_pnl_usd: float = 0.0

    def record_cycle(self, sessions: List[TradeSession]) -> None:
        """Fold one cycle's sessions into the running totals."""
        self.cycles += 1
        self.total_candidates += len(sessions)
        self.status_counts.update(s.status for s in sessions)
        self.total_pnl_usd += sum(session_pnl(s) for s in sessions)


def _log_cumulative_summary(stats: CumulativeStats) -> None:
    """Log the cumulative summary for a finished/interrupted continuous run."""
    logger.info(
        "[shutdown] continuous run: %d cycle(s), %d candidate(s) scanned, "
        "statuses=%s, total simulated PnL=$%.2f",
        stats.cycles,
        stats.total_candidates,
        dict(stats.status_counts),
        stats.total_pnl_usd,
    )


def _print_journal_summary(journal_path: str) -> None:
    """Print the persisted-journal performance summary at end of a run.

    Read-only: reads the append-only journal and renders the ASCII table. Safe
    no-op (prints an empty-journal notice) if nothing was recorded this run.
    """
    summary, count = build_summary(journal_path)
    print()
    print(format_summary(summary, source=f"{journal_path} ({count} trade(s))"))


async def run_continuous(
    *,
    interval: float = DEFAULT_INTERVAL_SECONDS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_iterations: int = 5,
    dry_run: bool = True,
    strategy: str = DEFAULT_STRATEGY,
    scan: ContinuousScan = scan_and_evaluate,
    snapshot_provider: SnapshotProvider = mock_market_snapshot,
    meta_boost_provider: MetaBoostProvider = _neutral_meta_boost_provider,
    sleep: AsyncSleep = asyncio.sleep,
    max_cycles: Optional[int] = None,
    stats: Optional[CumulativeStats] = None,
    journal_path: Optional[str] = None,
    reentry_guard: Optional[ReentryGuard] = None,
) -> CumulativeStats:
    """Re-run ``scan_and_evaluate`` every ``interval`` seconds until interrupted.

    Read-only / dry-run, exactly like the single pass — nothing is signed or
    broadcast. Each cycle runs the full discovery->evaluate pipeline and folds
    its sessions into ``stats``. Between cycles the loop sleeps ``interval``
    seconds (skipped after the final cycle).

    Stopping:
        * In production ``max_cycles`` is ``None`` (run forever); the loop ends
          when the user sends Ctrl+C. The resulting ``KeyboardInterrupt`` (or
          the ``CancelledError`` asyncio raises into the task on shutdown) is
          caught here and the loop stops cleanly with totals intact.
        * ``max_cycles`` bounds the loop for tests/automation: it runs exactly
          that many cycles then returns.

    ``stats`` is accumulated IN PLACE; pass one in so a caller still holds the
    running totals if the loop is interrupted. ``scan`` and ``sleep`` are
    injectable so tests can drive cycles without network or real delays.
    """
    if stats is None:
        stats = CumulativeStats()

    cycle = 0
    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            logger.info(
                "[loop] === cycle %d start (dry_run=%s, interval=%.0fs) ===",
                cycle,
                dry_run,
                interval,
            )
            scan_kwargs: Dict[str, Any] = dict(
                max_candidates=max_candidates,
                max_iterations=max_iterations,
                dry_run=dry_run,
                strategy=strategy,
                snapshot_provider=snapshot_provider,
                meta_boost_provider=meta_boost_provider,
            )
            # Only forward the journal path when journalling is enabled, so the
            # hermetic default (None) leaves the scan call signature unchanged.
            if journal_path:
                scan_kwargs["journal_path"] = journal_path
            # Pass the SAME guard object every cycle so cooldown/blacklist state
            # accumulates across the run (not just within one scan).
            if reentry_guard is not None:
                scan_kwargs["reentry_guard"] = reentry_guard
            sessions = await scan(**scan_kwargs)
            stats.record_cycle(sessions)
            logger.info(
                "[loop] cycle %d done: %d candidate(s) %s | "
                "cumulative PnL=$%.2f over %d cycle(s)",
                cycle,
                len(sessions),
                dict(Counter(s.status for s in sessions)),
                stats.total_pnl_usd,
                stats.cycles,
            )

            # Don't sleep after the final bounded cycle.
            if max_cycles is not None and cycle >= max_cycles:
                break
            if interval > 0:
                await sleep(interval)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info(
            "[loop] interrupted after %d completed cycle(s) — stopping cleanly",
            stats.cycles,
        )

    return stats


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
    parser.add_argument(
        "--loop",
        dest="loop",
        action="store_true",
        default=False,
        help=(
            "Run continuously, re-scanning every --interval seconds until "
            "interrupted with Ctrl+C (default: single pass)."
        ),
    )
    parser.add_argument(
        "--interval",
        dest="interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help=(
            "Seconds to sleep between cycles in --loop mode "
            f"(default: {DEFAULT_INTERVAL_SECONDS:.0f})."
        ),
    )
    parser.add_argument(
        "--strategy",
        dest="strategy",
        choices=STRATEGY_CHOICES,
        default=DEFAULT_STRATEGY,
        help=(
            "Entry strategy to evaluate each safety-passed candidate with: "
            "'dip' (post-pump dip-buy), 'momentum' (breakout), or 'both' "
            "(dip first, then momentum; first BUY wins) "
            f"(default: {DEFAULT_STRATEGY})."
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
    logger.info(
        "[startup] Helix Vector 1.0 | mode=%s network=%s strategy=%s",
        mode, config.SOLANA_NETWORK, args.strategy,
    )

    if not args.dry_run:
        logger.info("[startup] LIVE mode requested — trading logic not implemented yet.")

    # Re-entry guard, seeded from the persisted journal so cooldown/blacklist
    # survive across separate runs (the observed ACM churn was 7 separate
    # processes). Read-only: it only ever SKIPS a re-buy.
    now0 = datetime.now(timezone.utc)
    reentry_guard = ReentryGuard.from_records(
        load_records(config.TRADE_JOURNAL_PATH),
        now=now0,
        cooldown_seconds=config.REENTRY_COOLDOWN_SECONDS,
        blacklist_hard_exits=config.REENTRY_BLACKLIST_HARD_EXITS,
        blacklist_seconds=config.REENTRY_BLACKLIST_SECONDS,
    )
    logger.info(
        "[startup] re-entry guard: cooldown=%.0fs blacklist>=%d hard-exit(s) for %.0fs "
        "| seeded %d mint(s), %d currently blacklisted",
        config.REENTRY_COOLDOWN_SECONDS, config.REENTRY_BLACKLIST_HARD_EXITS,
        config.REENTRY_BLACKLIST_SECONDS, reentry_guard.tracked,
        len(reentry_guard.blacklisted_mints(now0)),
    )

    # Continuous mode: keep re-scanning every --interval seconds until Ctrl+C.
    # Stats are accumulated in a local object so the cumulative summary is still
    # reported if the run is interrupted mid-cycle. Read-only/dry-run throughout.
    if args.loop:
        logger.info(
            "[startup] continuous mode — re-scanning every %.0fs (Ctrl+C to stop)",
            args.interval,
        )
        stats = CumulativeStats()
        try:
            asyncio.run(
                run_continuous(
                    interval=args.interval,
                    max_candidates=args.max_candidates,
                    max_iterations=args.iterations,
                    dry_run=args.dry_run,
                    strategy=args.strategy,
                    snapshot_provider=get_live_market_snapshot,
                    meta_boost_provider=_live_meta_boost_provider,
                    stats=stats,
                    journal_path=config.TRADE_JOURNAL_PATH,
                    reentry_guard=reentry_guard,
                )
            )
        except KeyboardInterrupt:
            # Ctrl+C can surface here (out of asyncio.run) rather than inside the
            # coroutine; either way the in-place stats are intact.
            logger.info("[loop] keyboard interrupt received — shutting down")
        _log_cumulative_summary(stats)
        _print_journal_summary(config.TRADE_JOURNAL_PATH)
        return

    # Single pass: live, read-only discovery -> per-candidate dry-run evaluation.
    # The scanner surfaces real Dexscreener candidates; each runs through the
    # full pipeline (safety gate -> real-OHLCV entry decision -> on BUY, the
    # managed loop on the live market feed, with simulated-friction fills).
    # Nothing is ever signed or broadcast.
    logger.info(
        "[scan] discovering candidates (read-only); cap=%d", args.max_candidates
    )
    sessions = asyncio.run(
        scan_and_evaluate(
            max_candidates=args.max_candidates,
            max_iterations=args.iterations,
            dry_run=args.dry_run,
            strategy=args.strategy,
            snapshot_provider=get_live_market_snapshot,
            meta_boost_provider=_live_meta_boost_provider,
            journal_path=config.TRADE_JOURNAL_PATH,
            reentry_guard=reentry_guard,
        )
    )
    summary = Counter(s.status for s in sessions)
    logger.info(
        "[shutdown] processed %d candidate(s): %s",
        len(sessions),
        dict(summary),
    )
    _print_journal_summary(config.TRADE_JOURNAL_PATH)


if __name__ == "__main__":
    main()
