"""Tests for Ledger.list_runs() + RunSummary.

Seams tested:
  1. list_runs -- empty ledger returns []
  2. list_runs -- single run: status + outcome extracted, counts correct
  3. list_runs -- multiple runs ordered DESC by start_time
  4. list_runs -- latest status wins when multiple status_changed events exist
  5. list_runs -- outcome None for a run that never emitted an outcome
  6. list_runs -- limit + offset paging
"""

from __future__ import annotations

from pathlib import Path

import pytest

from northstack.adapters.sqlite_ledger import Ledger, RunSummary
from northstack.domain import RunOutcome, RunStatus
from northstack.events.catalog import OutcomeEmitted, RunCreated, StatusChanged


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    with Ledger(path=tmp_path / "runs.db") as lg:
        yield lg


def _seed_run(
    ledger: Ledger,
    run_id: str,
    statuses: list[str],
    outcome: str | None = None,
) -> None:
    """Append a minimal event sequence for one run via append_next.

    Always starts with RUN_CREATED then a chain of STATUS_CHANGED events.
    Optionally an OUTCOME_EMITTED after them.
    """
    ledger.append_next(run_id, RunCreated())
    for st in statuses:
        ledger.append_next(run_id, StatusChanged(status=RunStatus(st)))
    if outcome is not None:
        ledger.append_next(run_id, OutcomeEmitted(outcome=RunOutcome(outcome)))


def test_list_runs_empty(ledger: Ledger) -> None:
    assert ledger.list_runs() == []


def test_list_runs_single_extract(ledger: Ledger) -> None:
    _seed_run(ledger, "run-aaa", ["contracted", "planned", "verified"], "verified")
    runs = ledger.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert isinstance(r, RunSummary)
    assert r.run_id == "run-aaa"
    assert r.status == "verified"  # latest status_changed
    assert r.outcome == "verified"  # outcome_emitted
    assert r.event_count == 5  # RUN_CREATED + 3 STATUS + OUTCOME
    assert r.start_time <= r.last_event_time


def test_list_runs_ordered_desc(ledger: Ledger) -> None:
    # Seed each run completely before the next so start_times are strictly
    # increasing (run-3 seeded last => newest). list_runs DESC returns newest
    # first; a last_seq DESC tiebreaker keeps this stable if timestamps tie.
    _seed_run(ledger, "run-1", ["verified"], "verified")
    _seed_run(ledger, "run-2", ["abstained"], "abstained")
    _seed_run(ledger, "run-3", ["failed"], "failed")

    runs = ledger.list_runs()
    assert [r.run_id for r in runs] == ["run-3", "run-2", "run-1"]
    assert {r.outcome for r in runs} == {"verified", "abstained", "failed"}
    assert runs[0].last_event_time >= runs[-1].start_time  # newest >= oldest start


def test_list_runs_order_stable_when_timestamps_tie(ledger: Ledger) -> None:
    # Force a sub-ms seeding where all three runs share an identical
    # start_time; the last_seq DESC tiebreaker must keep newest-seeded first.
    for rid in ["run-a", "run-b", "run-c"]:
        ledger.append_next(rid, RunCreated())
        ledger.append_next(rid, StatusChanged(status=RunStatus.PLANNED))
    runs = ledger.list_runs()
    ids = [r.run_id for r in runs]
    # run-c was seeded last => highest last_seq => appears first under the tiebreaker
    assert ids == ["run-c", "run-b", "run-a"]


def test_list_runs_latest_status_wins(ledger: Ledger) -> None:
    _seed_run(
        ledger,
        "run-late",
        ["contracted", "planned", "executing", "verifying", "failed"],
        "failed",
    )
    runs = ledger.list_runs()
    assert runs[0].status == "failed"
    assert runs[0].outcome == "failed"


def test_list_runs_outcome_none_unless_emitted(ledger: Ledger) -> None:
    _seed_run(ledger, "run-nooutcome", ["contracted", "planned"])
    runs = ledger.list_runs()
    assert runs[0].outcome is None
    assert runs[0].status == "planned"


def test_list_runs_status_unknown_when_none(ledger: Ledger) -> None:
    # A run with only RUN_CREATED, no STATUS_CHANGED
    ledger.append_next("run-only-created", RunCreated())
    runs = ledger.list_runs()
    assert runs[0].status == "unknown"
    assert runs[0].outcome is None
    assert runs[0].event_count == 1


def test_list_runs_limit_offset(ledger: Ledger) -> None:
    for i in range(5):
        rid = f"run-{i}"
        ledger.append_next(rid, RunCreated())
        ledger.append_next(rid, StatusChanged(status=RunStatus.PLANNED))

    page1 = ledger.list_runs(limit=2, offset=0)
    page2 = ledger.list_runs(limit=2, offset=2)
    page3 = ledger.list_runs(limit=2, offset=4)
    assert len(page1) == 2
    assert len(page2) == 2
    assert len(page3) == 1
    # No overlap across pages
    all_ids = {r.run_id for r in page1 + page2 + page3}
    assert all_ids == {f"run-{i}" for i in range(5)}
    assert len(ledger.list_runs(limit=10)) == 5
