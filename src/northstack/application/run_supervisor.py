"""One owner of a run's lifecycle: task + ledger + workspace + heartbeat.

The supervisor is the only object that holds the per-run task, ledger
handle and workspace path, and the only place those handles are released.

Release contract: ``release()`` is idempotent. The task body, the
cancellation path, and server shutdown can all race to release the
per-run handles; whichever runs first tears them down and every later
caller is a no-op. The ledger is closed, the task reference dropped, and
the live accessors return None -- each exactly once. The workspace PATH
is retained for finished-run discovery. The terminal event on
cancellation is ``Company.run_async``'s own contract; the supervisor
guarantees the ledger it lands on stays readable up to that point, then
closes it once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstack.adapters.sqlite_ledger import Ledger


class RunSupervisor:
    """Owns the per-run task + ledger handle + workspace path + heartbeat.

    Held in a single registry keyed by run id, and released exactly once
    when the run reaches a terminal state or is cancelled.
    """

    __slots__ = (
        "_heartbeat_cb",
        "_ledger",
        "_released",
        "_run_id",
        "_task",
        "_workspace",
    )

    def __init__(
        self,
        *,
        run_id: str,
        ledger: Ledger | None,
        workspace: str | None,
        task: asyncio.Task[None] | None,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._ledger = ledger
        self._workspace = workspace
        self._task = task
        self._heartbeat_cb = heartbeat
        self._released = False

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def ledger(self) -> Ledger | None:
        return self._ledger if not self._released else None

    @property
    def workspace(self) -> str | None:
        return self._workspace

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task if not self._released else None

    @property
    def released(self) -> bool:
        return self._released

    @property
    def is_active(self) -> bool:
        """A run is active while its task exists and has not finished."""
        if self._released or self._task is None:
            return False
        return not self._task.done()

    def bind_task(self, task: asyncio.Task[None]) -> None:
        """Attach the asyncio task created for this run.

        Called by the start handler right after ``asyncio.create_task``.
        """
        if self._released:
            task.cancel()  # released before the task was bound: tear it down
            return
        self._task = task

    def heartbeat(self) -> None:
        """Record a per-cell progress beat for the stall detector."""
        if self._heartbeat_cb is not None:
            self._heartbeat_cb()

    def release(self) -> None:
        """Release the per-run handles exactly once. Idempotent.

        Drops the task reference and closes the ledger -- each exactly
        once, no matter how many callers race. The workspace PATH is
        retained (a string, not a handle) so finished-run discovery still
        works. Safe to call from the task body's ``finally``, the
        cancellation path, and the server shutdown path.
        """
        if self._released:
            return
        self._released = True
        self._task = None
        ledger = self._ledger
        self._ledger = None
        if ledger is not None:
            try:
                ledger.close()
            except Exception:  # noqa: BLE001, S110 - best-effort teardown, never raise
                pass
