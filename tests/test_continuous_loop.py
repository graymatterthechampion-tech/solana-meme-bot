"""Tests for continuous (--loop) mode in main.run_continuous.

Proves the continuous runner re-runs the scan/evaluate pipeline cycle after
cycle, accumulates cumulative stats (cycle count, candidates, per-status tallies
and simulated PnL), stops cleanly when bounded by ``max_cycles``, and shuts down
gracefully — with totals intact — on a Ctrl+C (KeyboardInterrupt).

No network and no real delays: the per-cycle ``scan`` and the inter-cycle
``sleep`` are both injected.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

import main
from execution.fill_simulator import simulate_fill
from main import LoopOutcome, TradeSession
from strategies.profit_taking import Position


def run(coro):
    return asyncio.run(coro)


def make_scan(sessions: List[TradeSession], calls: List[Dict[str, Any]]):
    """An injected per-cycle scan that records its kwargs and returns sessions."""

    async def scan(**kwargs: Any) -> List[TradeSession]:
        calls.append(kwargs)
        return list(sessions)

    return scan


def test_bounded_loop_runs_n_cycles_then_stops() -> None:
    """``max_cycles`` runs exactly N cycles, sleeping only between them."""
    sessions = [TradeSession("A", "entered"), TradeSession("B", "no_entry")]
    calls: List[Dict[str, Any]] = []
    sleeps: List[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    stats = run(
        main.run_continuous(
            max_cycles=3,
            interval=5.0,
            scan=make_scan(sessions, calls),
            sleep=sleep,
        )
    )

    # Ran exactly 3 cycles; slept between cycles but NOT after the last one.
    assert stats.cycles == 3
    assert len(calls) == 3
    assert sleeps == [5.0, 5.0]

    # Each cycle's kwargs are forwarded to the scan pipeline.
    assert set(calls[0]) == {
        "max_candidates",
        "max_iterations",
        "dry_run",
        "strategy",
        "snapshot_provider",
        "meta_boost_provider",
    }
    assert calls[0]["dry_run"] is True

    # Cumulative tallies fold every cycle's sessions together.
    assert stats.total_candidates == 6
    assert dict(stats.status_counts) == {"entered": 3, "no_entry": 3}
    # No positions opened -> no simulated PnL.
    assert stats.total_pnl_usd == 0.0


def test_keyboard_interrupt_stops_cleanly_with_cumulative_stats() -> None:
    """A Ctrl+C mid-run is caught; completed-cycle totals survive intact."""
    sessions = [TradeSession("A", "entered")]
    calls: List[Dict[str, Any]] = []

    async def boom_sleep(_seconds: float) -> None:
        # Simulate Ctrl+C arriving after the 2nd cycle's scan completes.
        if len(calls) >= 2:
            raise KeyboardInterrupt

    stats = main.CumulativeStats()
    result = run(
        main.run_continuous(
            interval=1.0,
            scan=make_scan(sessions, calls),
            sleep=boom_sleep,
            stats=stats,
            max_cycles=None,  # would run forever if not interrupted
        )
    )

    # Returned the same in-place stats object, without propagating the interrupt.
    assert result is stats
    assert stats.cycles == 2
    assert len(calls) == 2
    assert dict(stats.status_counts) == {"entered": 2}


def test_cumulative_pnl_accumulates_across_cycles() -> None:
    """Simulated realised PnL sums across every cycle of the run."""
    position = Position(entry_price=0.001, original_size=1000.0)
    fill = simulate_fill(500.0, 0.003, 100_000.0)  # sell 500 @ 3x into a deep pool
    outcome = LoopOutcome(path="profit_taking", fills=[fill])
    session = TradeSession(
        "GOOD", "entered", position=position, loop_outcomes=[outcome]
    )

    # Independently computed per-cycle PnL: net proceeds minus cost basis.
    per_cycle_pnl = fill.net_proceeds_usd - 500.0 * 0.001
    assert per_cycle_pnl > 0.0  # a 3x sell is profitable net of friction

    async def scan(**_kwargs: Any) -> List[TradeSession]:
        return [session]

    async def sleep(_seconds: float) -> None:
        return None

    stats = run(
        main.run_continuous(max_cycles=4, interval=0.0, scan=scan, sleep=sleep)
    )

    assert stats.cycles == 4
    assert stats.total_candidates == 4
    assert stats.total_pnl_usd == pytest.approx(4 * per_cycle_pnl)
