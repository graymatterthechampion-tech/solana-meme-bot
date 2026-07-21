"""Append-only persistence for completed trades.

The clean :class:`~reporting.trade_report.TradeReport` answers "why did THIS
position exit and what did it make" for a single trade, but it lives only in
memory: the loop builds it, logs it, and drops it. Nothing survives the run, so
performance cannot be reviewed after the fact or compared across runs.

This module fixes that with the smallest possible footprint: it flattens one
finished :class:`TradeReport` (plus the timing the loop captured around it) into
a JSON-serialisable :class:`TradeRecord` and APPENDS it as one line to a local
JSON Lines file (``trades.jsonl`` by default). JSON Lines — one JSON object per
line — is chosen so every write is a pure append: we never load, mutate, and
rewrite the whole history, so an interrupted run can at worst lose the trade in
flight, never corrupt the ones already recorded.

SAFETY / SCOPE
--------------
* Read-only with respect to the market: this only writes a local file. Nothing
  here fetches, signs, or broadcasts anything.
* The record holds only simulated PnL and market metrics — no keys, secrets, or
  wallet material — but it is runtime output, so the file is git-ignored.
* Writing is best-effort: :func:`append_trade` never raises into the trading
  loop. A disk error is logged and swallowed so a journalling problem can never
  take down (or alter) the run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # avoid any import cost / cycle at runtime.
    from reporting.trade_report import TradeReport

logger = logging.getLogger(__name__)

# Schema version stamped on every record so a future format change can be
# detected and migrated rather than silently misread.
JOURNAL_SCHEMA: int = 1


@dataclass(frozen=True)
class TradeRecord:
    """One completed trade, flattened for durable JSON Lines storage.

    Every field the performance summary needs is precomputed here so the
    analyser never has to re-derive anything from raw loop internals:

        * win/loss + PnL         -> ``realised_pnl_usd``
        * exit-reason breakdown  -> ``exit_trigger`` (hard exit) and
          ``profit_tiers`` (which of 2x/5x/10x fired)
        * hold time              -> ``hold_ticks`` and ``hold_seconds``
        * best/worst trade       -> ``realised_pnl_usd`` + ``symbol``
    """

    schema: int
    recorded_at: str          # ISO-8601 UTC timestamp the record was written
    dry_run: bool
    mint_address: str
    symbol: str
    strategy: str
    entry_reason: str
    entry_price: float
    entry_size: float
    entry_liquidity: float
    realised_pnl_usd: float
    fill_count: int
    fully_exited: bool
    tokens_held: float
    exit_trigger: Optional[str]   # hard-exit trigger name, else None
    exit_reason: str
    profit_tiers: List[str]       # e.g. ["2x", "5x"] — tiers that fired
    entry_time: Optional[str]     # ISO-8601 UTC; None if the loop didn't time it
    exit_time: Optional[str]
    hold_ticks: int               # managed loop iterations the position lived
    hold_seconds: float           # wall-clock entry->exit (0.0 if untimed)
    legs: List[Dict[str, Any]] = field(default_factory=list)


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


def _profit_tiers(report: "TradeReport") -> List[str]:
    """Extract the profit tiers that fired, in order (``["2x", "5x", ...]``).

    Profit legs are labelled ``"PROFIT/<tier>"`` by the report builder; a hard
    exit is labelled ``"HARD_EXIT/<trigger>"`` and is deliberately excluded.
    """
    tiers: List[str] = []
    for leg in report.legs:
        if leg.kind == "profit" and leg.label.startswith("PROFIT/"):
            tiers.append(leg.label.split("/", 1)[1])
    return tiers


def _hold_seconds(entry_time: Optional[str], exit_time: Optional[str]) -> float:
    """Wall-clock seconds between two ISO timestamps (0.0 if either is absent)."""
    if not entry_time or not exit_time:
        return 0.0
    try:
        start = datetime.fromisoformat(entry_time)
        end = datetime.fromisoformat(exit_time)
    except ValueError:
        return 0.0
    return max(0.0, (end - start).total_seconds())


def record_from_report(
    report: "TradeReport",
    *,
    dry_run: bool = True,
    entry_time: Optional[str] = None,
    exit_time: Optional[str] = None,
    hold_ticks: int = 0,
) -> TradeRecord:
    """Flatten a finished :class:`TradeReport` into a storable :class:`TradeRecord`.

    Pure — no I/O, no clock read except the ``recorded_at`` stamp. The caller
    supplies the entry/exit timestamps and iteration count it observed around
    :func:`main.run_loop`; everything else comes straight off the report.
    """
    legs = [
        {
            "label": leg.label,
            "kind": leg.kind,
            "tokens": leg.tokens,
            "price": leg.price,
            "net_proceeds_usd": leg.net_proceeds_usd,
            "pnl_usd": leg.pnl_usd,
            "over_cap": leg.over_cap,
        }
        for leg in report.legs
    ]
    return TradeRecord(
        schema=JOURNAL_SCHEMA,
        recorded_at=_now_iso(),
        dry_run=dry_run,
        mint_address=report.mint_address,
        symbol=report.symbol,
        strategy=report.strategy,
        entry_reason=report.entry_reason,
        entry_price=report.entry_price,
        entry_size=report.entry_size,
        entry_liquidity=report.entry_liquidity,
        realised_pnl_usd=report.realised_pnl_usd,
        fill_count=report.fill_count,
        fully_exited=report.fully_exited,
        tokens_held=report.tokens_held,
        exit_trigger=report.exit_trigger,
        exit_reason=report.exit_reason,
        profit_tiers=_profit_tiers(report),
        entry_time=entry_time,
        exit_time=exit_time,
        hold_ticks=hold_ticks,
        hold_seconds=_hold_seconds(entry_time, exit_time),
        legs=legs,
    )


def append_record(record: TradeRecord, path: str) -> None:
    """Append one record as a single JSON line to ``path`` (creating it if new).

    Raises on I/O error — use :func:`append_trade` for the best-effort variant
    the trading loop calls. Any parent directory in ``path`` is created first.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=True, sort_keys=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_trade(
    report: "TradeReport",
    path: str,
    *,
    dry_run: bool = True,
    entry_time: Optional[str] = None,
    exit_time: Optional[str] = None,
    hold_ticks: int = 0,
) -> Optional[TradeRecord]:
    """Best-effort: journal one completed trade, never raising into the loop.

    Builds the record and appends it. On ANY error (disk full, permissions,
    serialisation) the failure is logged and swallowed, and ``None`` is
    returned — a journalling problem must never abort or alter a trading run.
    Returns the written :class:`TradeRecord` on success.
    """
    try:
        record = record_from_report(
            report,
            dry_run=dry_run,
            entry_time=entry_time,
            exit_time=exit_time,
            hold_ticks=hold_ticks,
        )
        append_record(record, path)
        return record
    except Exception as exc:  # noqa: BLE001 — journalling is strictly optional
        logger.warning(
            "[journal] failed to persist trade for %s (%s): %s -> skipped",
            getattr(report, "symbol", "?"),
            getattr(report, "mint_address", "?"),
            exc,
        )
        return None


def load_records(path: str) -> List[TradeRecord]:
    """Read every journalled trade back from a JSON Lines file.

    Skips blank lines and any single malformed/legacy line (logged, not fatal)
    so one bad append can never make the whole history unreadable. Returns an
    empty list if the file does not exist yet.
    """
    if not os.path.exists(path):
        return []

    records: List[TradeRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(_record_from_dict(data))
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "[journal] skipping unreadable line %d in %s: %s",
                    lineno, path, exc,
                )
    return records


def _record_from_dict(data: Dict[str, Any]) -> TradeRecord:
    """Rebuild a :class:`TradeRecord` from a decoded JSON object.

    Tolerant of missing optional keys (older/partial records) so the analyser
    degrades gracefully rather than refusing to read a slightly-older journal.
    """
    return TradeRecord(
        schema=int(data.get("schema", 0)),
        recorded_at=str(data.get("recorded_at", "")),
        dry_run=bool(data.get("dry_run", True)),
        mint_address=str(data.get("mint_address", "")),
        symbol=str(data.get("symbol", "?")),
        strategy=str(data.get("strategy", "?")),
        entry_reason=str(data.get("entry_reason", "")),
        entry_price=float(data.get("entry_price", 0.0)),
        entry_size=float(data.get("entry_size", 0.0)),
        entry_liquidity=float(data.get("entry_liquidity", 0.0)),
        realised_pnl_usd=float(data.get("realised_pnl_usd", 0.0)),
        fill_count=int(data.get("fill_count", 0)),
        fully_exited=bool(data.get("fully_exited", False)),
        tokens_held=float(data.get("tokens_held", 0.0)),
        exit_trigger=data.get("exit_trigger"),
        exit_reason=str(data.get("exit_reason", "")),
        profit_tiers=list(data.get("profit_tiers", []) or []),
        entry_time=data.get("entry_time"),
        exit_time=data.get("exit_time"),
        hold_ticks=int(data.get("hold_ticks", 0)),
        hold_seconds=float(data.get("hold_seconds", 0.0)),
        legs=list(data.get("legs", []) or []),
    )
