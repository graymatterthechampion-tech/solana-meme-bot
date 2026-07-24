"""Re-entry guard — cooldown + blacklist to stop same-token churn.

The first real dry-run exposed a churn pathology: the bot re-bought the SAME
mint over and over (ACM entered 7 times in ~1h50m), each time volume-collapsing
out within one tick for pure friction loss, because nothing remembered that we
had just traded — or just been burned by — that token.

This module is that memory. A :class:`ReentryGuard` tracks, per mint, when we
last entered it and how many times it has triggered a defensive HARD exit, and
answers one question before the safety/entry gates run:

    should we even consider this mint right now?

Two independent brakes:

* **Cooldown** — after we ENTER a mint we refuse to re-enter it for
  ``cooldown_seconds`` regardless of how that trade turned out. This alone kills
  the rapid re-buy churn (a 1h cooldown would have collapsed ACM's 7 entries to
  1) and stops us re-opening tokens we're already holding.
* **Blacklist** — a hard exit (flash_crash / volume_collapse / liquidity_drop /
  coordinated_dump / hard_stop_loss) is the token telling us it is dying or
  rugging. After ``blacklist_hard_exits`` such exits the mint is blacklisted for
  ``blacklist_seconds`` (or permanently, for the process, when that is <= 0).

SCOPE / SAFETY
--------------
Pure in-memory bookkeeping over timestamps and counts. It only ever BLOCKS an
entry — it can never open, size, sign, or broadcast anything, and it fail-closes
in the safe direction (when unsure it does not block). It can be seeded from the
persisted trade journal (:meth:`ReentryGuard.from_records`) so the cooldown and
blacklist survive across separate process runs — which is exactly how the
observed churn slipped through (each re-buy was a fresh ``python main.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Sentinel "until" for a permanent (process-lifetime) blacklist.
_PERMANENT: datetime = datetime.max.replace(tzinfo=timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    """Return ``dt`` as a timezone-aware UTC datetime (assume UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp to aware UTC, or ``None`` if unparseable."""
    if not value:
        return None
    try:
        return _as_utc(datetime.fromisoformat(value))
    except ValueError:
        return None


@dataclass
class _TokenState:
    """Per-mint memory: last entry time, hard-exit count, blacklist expiry."""

    last_entry_at: Optional[datetime] = None
    hard_exit_count: int = 0
    blacklisted_until: Optional[datetime] = None  # None = not blacklisted


@dataclass
class ReentryGuard:
    """Cooldown + blacklist gate keyed by mint address.

    Times are timezone-aware UTC. ``blacklist_seconds <= 0`` means a blacklist,
    once armed, lasts for the lifetime of this guard (permanent).
    """

    cooldown_seconds: float = 3600.0
    blacklist_hard_exits: int = 1
    blacklist_seconds: float = 86_400.0
    _states: Dict[str, _TokenState] = field(default_factory=dict)

    # --- queries ------------------------------------------------------------

    def check(self, mint: str, now: datetime) -> Optional[str]:
        """Reason to SKIP this mint right now, or ``None`` if it's tradeable.

        Blacklist is checked before cooldown (it's the stronger signal). Both are
        fail-open: an unknown mint, or one whose windows have elapsed, returns
        ``None`` and is allowed through to the normal safety/entry gates.
        """
        state = self._states.get(mint)
        if state is None:
            return None
        now = _as_utc(now)

        if state.blacklisted_until is not None and now < state.blacklisted_until:
            if state.blacklisted_until == _PERMANENT:
                return (
                    f"blacklisted (permanent) after {state.hard_exit_count} "
                    f"hard-exit(s)"
                )
            return (
                f"blacklisted after {state.hard_exit_count} hard-exit(s) until "
                f"{state.blacklisted_until.isoformat(timespec='seconds')}"
            )

        if state.last_entry_at is not None and self.cooldown_seconds > 0:
            elapsed = (now - state.last_entry_at).total_seconds()
            if 0 <= elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                return (
                    f"cooldown {remaining:.0f}s remaining (entered "
                    f"{state.last_entry_at.isoformat(timespec='seconds')})"
                )
        return None

    def is_blocked(self, mint: str, now: datetime) -> bool:
        """True if :meth:`check` would block this mint now."""
        return self.check(mint, now) is not None

    # --- updates ------------------------------------------------------------

    def record_entry(self, mint: str, now: datetime) -> None:
        """Remember that we ENTERED ``mint`` at ``now`` (arms the cooldown)."""
        if not mint:
            return
        self._states.setdefault(mint, _TokenState()).last_entry_at = _as_utc(now)

    def record_hard_exit(self, mint: str, trigger: Optional[str], now: datetime) -> None:
        """Record a defensive hard exit; arm/extend the blacklist at threshold."""
        if not mint:
            return
        now = _as_utc(now)
        state = self._states.setdefault(mint, _TokenState())
        state.hard_exit_count += 1
        if state.hard_exit_count < self.blacklist_hard_exits:
            return

        if self.blacklist_seconds <= 0:
            state.blacklisted_until = _PERMANENT
        else:
            until = now + timedelta(seconds=self.blacklist_seconds)
            # Take the later expiry so repeated exits only ever push it out.
            state.blacklisted_until = (
                until if state.blacklisted_until is None
                else max(state.blacklisted_until, until)
            )
        logger.debug(
            "[reentry] %s blacklisted (%d hard-exit(s), trigger=%s) until %s",
            mint, state.hard_exit_count, trigger,
            "permanent" if state.blacklisted_until == _PERMANENT
            else state.blacklisted_until.isoformat(timespec="seconds"),
        )

    # --- observability ------------------------------------------------------

    @property
    def tracked(self) -> int:
        """How many distinct mints the guard is currently remembering."""
        return len(self._states)

    def blacklisted_mints(self, now: datetime) -> List[str]:
        """Mints currently blacklisted at ``now`` (expired ones excluded)."""
        now = _as_utc(now)
        return [
            m for m, s in self._states.items()
            if s.blacklisted_until is not None and now < s.blacklisted_until
        ]

    # --- construction -------------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: Iterable["object"],
        *,
        now: datetime,
        cooldown_seconds: float = 3600.0,
        blacklist_hard_exits: int = 1,
        blacklist_seconds: float = 86_400.0,
    ) -> "ReentryGuard":
        """Seed a guard by replaying persisted journal records in time order.

        Each record contributes its entry (arming cooldown from ``entry_time``)
        and, if it hard-exited, its exit (arming/extending the blacklist from
        ``exit_time``). Because expiries are anchored to the ACTUAL historical
        times, a cooldown or blacklist that has already elapsed by ``now`` simply
        won't block — the replay reconstructs state, it does not reset the clock.

        Accepts any objects exposing ``mint_address``, ``entry_time``,
        ``exit_trigger`` and ``exit_time`` (i.e. journal ``TradeRecord``s);
        malformed/mint-less rows are skipped.
        """
        guard = cls(
            cooldown_seconds=cooldown_seconds,
            blacklist_hard_exits=blacklist_hard_exits,
            blacklist_seconds=blacklist_seconds,
        )

        def _key(r: object) -> str:
            return (getattr(r, "entry_time", "") or getattr(r, "recorded_at", "") or "")

        for r in sorted(records, key=_key):
            mint = str(getattr(r, "mint_address", "") or "")
            if not mint:
                continue
            entered = _parse_iso(getattr(r, "entry_time", None)) \
                or _parse_iso(getattr(r, "recorded_at", None))
            if entered is not None:
                guard.record_entry(mint, entered)
            trigger = getattr(r, "exit_trigger", None)
            if trigger:
                exited = _parse_iso(getattr(r, "exit_time", None)) or entered or _as_utc(now)
                guard.record_hard_exit(mint, trigger, exited)
        return guard
