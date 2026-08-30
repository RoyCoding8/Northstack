"""The emit facade: the one way a producer writes to the ledger.

Producers hold an ``EventStream`` bound to a run and emit typed payloads.  They
never allocate a sequence number, never touch the hash chain, and never build a
payload dict -- so a producer cannot invent a field the catalog does not know.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Protocol

from northstack.events.catalog import EventPayload
from northstack.events.envelope import EventEnvelope

_APPEND_ERRORS = (ValueError, TypeError, OSError, RuntimeError, sqlite3.Error)


class EventAppender(Protocol):
    """The slice of the ledger an emitter needs."""

    def append_next(self, run_id: str, payload: EventPayload) -> EventEnvelope: ...


class EventStream:
    """Run-scoped emitter.

    ``ledger=None`` is a first-class no-op so an orchestrator can run without a
    ledger (tests, dry runs) without every call site testing for it.

    Two entry points: ``emit`` (synchronous, for the sync CLI path) and
    ``emit_async`` (offloads the blocking ``append_next`` to a worker thread via
    ``asyncio.to_thread`` so the event loop is not stalled on sqlite I/O).
    ``Ledger.append_next`` is serialized by an in-process ``RLock``, so
    concurrent ``emit_async`` appends get sequential sequence numbers + a
    correct prev_hash rather than racing.
    """

    def __init__(self, ledger: EventAppender | None, run_id: str) -> None:
        self._ledger = ledger
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def emit(self, payload: EventPayload) -> EventEnvelope | None:
        if self._ledger is None:
            return None
        return self._ledger.append_next(self._run_id, payload)

    async def emit_async(self, payload: EventPayload) -> EventEnvelope | None:
        if self._ledger is None:
            return None
        append = asyncio.create_task(
            asyncio.to_thread(self._ledger.append_next, self._run_id, payload)
        )
        try:
            return await asyncio.shield(append)
        except asyncio.CancelledError as cancelled:
            while not append.done():
                try:
                    await asyncio.shield(append)
                except asyncio.CancelledError:
                    continue
                except _APPEND_ERRORS:
                    break
            try:
                append.result()
            except _APPEND_ERRORS as exc:
                cancelled.add_note(f"append completion failed: {type(exc).__name__}: {exc}")
            raise cancelled
