"""Incremental run-state projection cache.

Without a cache, a live view (``GET /api/runs/{id}``) calls ``replay_run`` on
every poll, folding the run's events from seq 1 each time -- at ~700 ms poll
cadence a long run re-folds its entire history on every poll. This module
caches the projected :class:`RunState` plus the cursor (next seq to read) per
run, so a poll folds only the events past the cursor onto the cached state.

The cache is run-id keyed and holds ``(cursor, RunState)``. A cold cache seeds
from ``RunState(run_id=...)`` (cursor 0); ``project`` folds the given tail
(assumed to start at the cursor) onto the cached state and advances the
cursor by the number of events folded. The caller reads the tail with
``ledger.events_since(run_id, since=cache.cursor(run_id))`` so the fold stays
incremental -- this module never touches the ledger itself, keeping it free
of storage coupling.
"""

from __future__ import annotations

from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.replay import fold_events
from northstack.domain.run_state import RunState
from northstack.events.envelope import EventEnvelope


class ProjectionCache:
    """Per-run ``(cursor, RunState)`` cache for incremental live views."""

    __slots__ = ("_states",)

    def __init__(self) -> None:
        self._states: dict[str, tuple[int, RunState]] = {}

    def cursor(self, run_id: str) -> int:
        """The next seq to read for ``run_id`` (0 on a cold cache)."""
        entry = self._states.get(run_id)
        return entry[0] if entry is not None else 0

    def state(self, run_id: str) -> RunState | None:
        """The cached projected state, or None on a cold cache."""
        entry = self._states.get(run_id)
        return entry[1] if entry is not None else None

    def project(
        self,
        run_id: str,
        ledger: Ledger,
        tail: list[EventEnvelope],
    ) -> RunState:
        """Fold ``tail`` onto the cached state and advance the cursor.

        ``tail`` MUST start at the cursor (i.e. be the result of
        ``ledger.events_since(run_id, since=cache.cursor(run_id))``). A cold
        cache seeds from an empty ``RunState`` (cursor 0). The folded state is
        cached and returned; the cursor advances by ``len(tail)``.
        """
        entry = self._states.get(run_id)
        if entry is None:
            base = RunState(run_id=run_id)
            cursor = 0
        else:
            cursor, base = entry
        folded = fold_events(base, tail)
        cursor = cursor + len(tail)
        self._states[run_id] = (cursor, folded)
        return folded

    def invalidate(self, run_id: str) -> None:
        """Drop the cached state for ``run_id`` (e.g. on a stop/restart)."""
        self._states.pop(run_id, None)
