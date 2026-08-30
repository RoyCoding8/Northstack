"""RunSupervisor: one owner of a run's task + ledger + workspace + heartbeat.

One ``RunSupervisor`` per run replaces the former hand-synced ``app.state``
dicts in ``interfaces/web/routes_runs.py`` (``active_runs``,
``run_ledgers``, ``run_workspaces``).  This file pins the two contracts
that substitution must preserve:

1. **Release exactly once.** A run's task body, the cancellation path,
   and the server shutdown path can all race to release the per-run
   handles. ``release()`` is idempotent: the ledger is closed, the task
   is dropped, and the workspace path is cleared from live access exactly
   once, no matter how many callers race. The supervisor tracks
   ``released`` and a spy ledger records a single close.

2. **Cancellation still writes a terminal event.** On
   ``asyncio.CancelledError`` the run coroutine (``Company.run_async``)
   emits a terminal ``OutcomeEmitted`` + ``StatusChanged`` before
   re-raising -- that is the run's own contract, already honoured in
   ``orchestrator.py``. The supervisor's job is to make sure that once
   the cancelled task finishes, ``release()`` has run exactly once and
   the ledger the events landed on is readable up to that terminal
   event before it is closed. This test drives a cancellable coroutine
   that mirrors ``run_async``'s cancellation handler and asserts the
   terminal event is durable in the ledger after release.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.run_supervisor import RunSupervisor
from northstack.domain import RunOutcome
from northstack.domain.status import RunStatus
from northstack.events.catalog import OutcomeEmitted, StatusChanged


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(path=tmp_path / "ledger.db")


def _seed_terminal(ledger: Ledger, run_id: str) -> None:
    """Mirror run_async's CancelledError handler: a terminal outcome+status."""
    ledger.append_next(run_id, OutcomeEmitted(outcome=RunOutcome.FAILED, reason="cancelled"))
    ledger.append_next(run_id, StatusChanged(status=RunStatus.FAILED))


class _CloseSpyLedger(Ledger):
    """A Ledger that counts close() calls to prove release-once."""

    def __init__(self, path: Path) -> None:
        super().__init__(path=path)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class TestRunSupervisorReleaseOnce:
    def test_release_is_idempotent_and_closes_ledger_once(self, tmp_path: Path) -> None:
        ledger = _CloseSpyLedger(tmp_path / "ledger.db")
        run_id = "run-release"
        # A completed (never started) task placeholder: None is a legal
        # "no task to await" state for a run that already finished.
        sup = RunSupervisor(run_id=run_id, ledger=ledger, workspace=str(tmp_path), task=None)

        sup.release()
        sup.release()  # second caller racing the same teardown path
        sup.release()  # and the shutdown path

        assert ledger.close_calls == 1, "ledger must be closed exactly once"
        assert sup.released is True
        # After release the supervisor no longer exposes live handles.
        assert sup.ledger is None
        assert sup.task is None


class TestRunSupervisorCancellationWritesTerminalThenReleasesOnce:
    @pytest.mark.asyncio
    async def test_cancelled_run_emits_terminal_and_releases_once(self, tmp_path: Path) -> None:
        """On CancelledError the run still writes a terminal OutcomeEmitted +
        status before re-raising, and the supervisor releases the task, ledger
        and workspace exactly once.

        This mirrors the real lifecycle: routes_runs wraps
        ``company.run_async(...)`` in a task; ``run_async``'s own
        ``except asyncio.CancelledError`` emits the terminal pair and
        re-raises; the supervisor's release (run from the task body's
        finally OR the shutdown path) must run exactly once and leave
        the terminal event durable in the closed ledger's history.
        """
        ledger = _CloseSpyLedger(tmp_path / "ledger.db")
        run_id = "run-cancel"

        async def _run_async_like() -> None:
            # Mirrors Company.run_async's cancellation contract: catch
            # CancelledError, emit terminal pair, re-raise.
            try:
                try:
                    await asyncio.Event().wait()  # block forever
                except asyncio.CancelledError:
                    _seed_terminal(ledger, run_id)
                    raise
            finally:
                # The task body releases the supervisor exactly once here.
                sup.release()

        task = asyncio.create_task(_run_async_like())
        sup = RunSupervisor(run_id=run_id, ledger=ledger, workspace=str(tmp_path), task=task)

        # Let the task start, then cancel it (as /stop or shutdown would).
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The terminal pair is durable on disk -- a reader that missed the
        # live handle (the route falls through to the on-disk ledger after
        # release) replays the same events.
        disk = Ledger(path=tmp_path / "ledger.db")
        try:
            events = disk.events(run_id)
        finally:
            disk.close()
        kinds = [type(e.payload).__name__ for e in events]
        assert "OutcomeEmitted" in kinds, "cancelled run must emit a terminal outcome"
        assert "StatusChanged" in kinds, "cancelled run must emit a terminal status"
        outcome_ev = next(e for e in events if isinstance(e.payload, OutcomeEmitted))
        assert outcome_ev.payload.outcome == RunOutcome.FAILED

        # Handles released exactly once, even though release() is reachable
        # from both the task-body finally and the shutdown path.
        sup.release()  # shutdown path racing the already-released run
        assert ledger.close_calls == 1
        assert sup.released is True
