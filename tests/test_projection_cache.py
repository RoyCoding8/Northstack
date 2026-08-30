"""Incremental live view: projection cache keyed by run id.

Without a cache, ``GET /api/runs/{id}`` calls ``replay_run`` on every poll,
folding the run's events from seq 1 each time (~700 ms poll cadence).  The
projection cache holds ``(cursor, RunState)`` per run so a poll folds only
the events past the cursor.  This file pins that with a call-count spy on
the event reader: seeding a cache, then polling twice with no new events
between the polls must NOT re-read from seq 1 -- the second poll reads zero
new events, and a poll after one new event reads exactly that one.
"""

from __future__ import annotations

from pathlib import Path

from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.projection_cache import ProjectionCache
from northstack.domain.status import RunStatus
from northstack.events.catalog import RequestAccepted, StatusChanged
from northstack.events.envelope import EventEnvelope


class _SpyLedger(Ledger):
    """A Ledger that counts events_since() calls and the ``since`` args."""

    def __init__(self, path: Path) -> None:
        super().__init__(path=path)
        self.since_args: list[int] = []
        self.events_calls = 0

    def events_since(self, run_id: str, since: int = 0, limit: int = 500) -> list[EventEnvelope]:
        self.since_args.append(since)
        return super().events_since(run_id, since=since, limit=limit)

    def events(self, run_id: str) -> list[EventEnvelope]:
        self.events_calls += 1
        return super().events(run_id)


_LEGAL_STEPS = [
    RunStatus.CONTRACTED,
    RunStatus.PLANNED,
    RunStatus.EXECUTING,
    RunStatus.VERIFYING,
]


def _seed(ledger: Ledger, run_id: str, n: int) -> int:
    """Append n events (request + legal status advances) -> last seq.

    RequestAccepted lands the run at INTAKE; each subsequent event advances
    the status along the legal INTAKE->contracted->planned->executing->verifying
    chain so the projection's state-machine check never sees an illegal flip.
    """
    ledger.append_next(run_id, RequestAccepted(goal="g", workspace_root="/ws", budget=None))
    for i in range(1, n):
        ledger.append_next(run_id, StatusChanged(status=_LEGAL_STEPS[(i - 1) % len(_LEGAL_STEPS)]))
    return n


class TestProjectionCacheIncremental:
    def test_two_polls_with_no_new_events_read_zero_after_first(self, tmp_path: Path) -> None:
        ledger = _SpyLedger(tmp_path / "ledger.db")
        run_id = "run-inc"
        _seed(ledger, run_id, 3)

        cache = ProjectionCache()

        # First poll: cold cache -> read from the cursor (0), fold all 3.
        state1 = cache.project(run_id, ledger, ledger.events_since(run_id, since=0))
        assert state1.events_replayed == 3
        assert cache.cursor(run_id) == 3

        # Second poll with NO new events: the cache hands the route a cursor
        # of 3, events_since(since=3) returns [], and the state is unchanged.
        tail = ledger.events_since(run_id, since=cache.cursor(run_id))
        state2 = cache.project(run_id, ledger, tail)
        assert state2.events_replayed == 3, "no new events -> events_replayed unchanged"
        assert cache.cursor(run_id) == 3
        # The spy recorded the cursor advancing past the seeded events.
        assert ledger.since_args[-1] == 3, "second poll must read past the cursor, not from 0"

    def test_poll_after_one_new_event_folds_only_that_event(self, tmp_path: Path) -> None:
        ledger = _SpyLedger(tmp_path / "ledger.db")
        run_id = "run-inc2"
        _seed(ledger, run_id, 2)

        cache = ProjectionCache()
        state1 = cache.project(run_id, ledger, ledger.events_since(run_id, since=0))
        assert state1.events_replayed == 2
        assert cache.cursor(run_id) == 2

        # Append exactly one new event -- the next legal step after the seed.
        # seed(2) emits RequestAccepted (-> INTAKE) then CONTRACTED, so the
        # legal next flip is PLANNED.
        ledger.append_next(run_id, StatusChanged(status=RunStatus.PLANNED))

        # The route reads only past the cursor (2) -> exactly one new event.
        tail = ledger.events_since(run_id, since=cache.cursor(run_id))
        assert len(tail) == 1, "should read exactly the one new event"
        state2 = cache.project(run_id, ledger, tail)
        assert state2.events_replayed == 3
        assert cache.cursor(run_id) == 3
        # The full-history reader was never invoked after the cold poll.
        assert ledger.events_calls == 0, "incremental path must not call events()"

    def test_project_does_not_replay_from_seq_1_on_warm_cache(self, tmp_path: Path) -> None:
        ledger = _SpyLedger(tmp_path / "ledger.db")
        run_id = "run-warm"
        _seed(ledger, run_id, 4)

        cache = ProjectionCache()
        cache.project(run_id, ledger, ledger.events_since(run_id, since=0))
        # Three more polls, no new events.
        for _ in range(3):
            tail = ledger.events_since(run_id, since=cache.cursor(run_id))
            cache.project(run_id, ledger, tail)
        # events() (full replay from seq 1) must never have been called.
        assert ledger.events_calls == 0
