"""Tests proving the event schema-evolution ladder and the emit facade.

Two seams:

  - ``northstack.events.upcast`` -- the schema-evolution ladder. At v1 the
    UPCASTERS registry is empty, so ``register`` and the ladder loop are dead
    code until a future format change arrives. These tests PROVE the ladder by
    monkeypatching CURRENT_SCHEMA_VERSION to 3 and simulating that future.
  - ``northstack.events.stream`` -- the emit facade. ``EventStream`` never
    allocates a sequence, never touches the hash chain, and must serialize
    concurrent async appends so the ledger reports contiguous sequences.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import northstack.events.upcast as upcast_mod
from northstack.events.catalog import CURRENT_SCHEMA_VERSION, EventKind, RunCreated
from northstack.events.envelope import EventEnvelope
from northstack.events.errors import LedgerCorruption, UnknownSchemaVersion
from northstack.events.stream import EventAppender, EventStream
from northstack.events.upcast import register, upcast


# upcast -- the schema-evolution ladder


def test_upcast_current_version_passes_through_untouched():
    """A payload already at the current version is returned equal and unmutated."""
    payload = {"kind": "run_created", "schema_version": CURRENT_SCHEMA_VERSION, "extra": 1}
    before = dict(payload)
    result = upcast(EventKind.RUN_CREATED, payload, seq=1)
    assert result == before
    assert payload == before


def test_upcast_missing_schema_version_is_treated_as_v1():
    """A payload lacking a schema_version key is read as version 1."""
    payload = {"kind": "run_created"}
    result = upcast(EventKind.RUN_CREATED, payload, seq=2)
    assert result == payload


def test_upcast_future_version_raises_unknown_schema_version():
    """A version newer than this build is refused, naming seq, kind, versions."""
    with pytest.raises(UnknownSchemaVersion) as exc:
        upcast(EventKind.RUN_CREATED, {"kind": "run_created", "schema_version": 5}, seq=7)
    assert exc.value.seq == 7
    assert exc.value.kind == "run_created"
    assert exc.value.version == 5
    assert exc.value.maximum == CURRENT_SCHEMA_VERSION


@pytest.mark.parametrize("bad_version", ["2", 2.0, None])
def test_upcast_non_int_schema_version_raises_ledger_corruption(bad_version):
    """A non-int schema_version is corruption, naming the offending seq."""
    with pytest.raises(LedgerCorruption) as exc:
        upcast(EventKind.RUN_CREATED, {"kind": "run_created", "schema_version": bad_version}, seq=3)
    assert exc.value.seq == 3


def test_upcast_ladder_applies_both_steps(monkeypatch):
    """With CURRENT=3 and both steps registered, upcast applies v1->v2->v3 in order."""
    monkeypatch.setattr(upcast_mod, "CURRENT_SCHEMA_VERSION", 3)
    monkeypatch.setattr(upcast_mod, "UPCASTERS", {})
    seen: dict[int, dict] = {}

    @register(EventKind.RUN_CREATED, 1)
    def to_v2(data: dict) -> dict:
        seen[1] = dict(data)
        data = dict(data)
        data["stage"] = "v2"
        return data

    @register(EventKind.RUN_CREATED, 2)
    def to_v3(data: dict) -> dict:
        seen[2] = dict(data)
        data = dict(data)
        data["stage"] = "v3"
        return data

    result = upcast(EventKind.RUN_CREATED, {"kind": "run_created", "schema_version": 1}, seq=1)
    assert result["schema_version"] == 3
    assert result["stage"] == "v3"
    assert seen[1] == {"kind": "run_created", "schema_version": 1}
    assert seen[2]["stage"] == "v2"


def test_upcast_gap_raises_loudly(monkeypatch):
    """A missing rung must refuse rather than partially parse the payload."""
    monkeypatch.setattr(upcast_mod, "CURRENT_SCHEMA_VERSION", 3)
    monkeypatch.setattr(upcast_mod, "UPCASTERS", {})

    @register(EventKind.RUN_CREATED, 2)
    def to_v3(data: dict) -> dict:
        return data

    with pytest.raises(LedgerCorruption, match="no upcaster from schema_version 1"):
        upcast(EventKind.RUN_CREATED, {"kind": "run_created", "schema_version": 1}, seq=9)


def test_register_returns_function_and_populates_registry(monkeypatch):
    """``register`` returns the wrapped function unchanged and keys it by (kind, from)."""
    monkeypatch.setattr(upcast_mod, "UPCASTERS", {})

    def my_step(data: dict) -> dict:
        return data

    decorated = register(EventKind.RUN_CREATED, 1)(my_step)
    assert decorated is my_step
    assert upcast_mod.UPCASTERS[(EventKind.RUN_CREATED, 1)] is my_step


def test_upcast_registration_is_kind_scoped(monkeypatch):
    """A registered run for one kind does not leak into a different kind's ladder."""
    monkeypatch.setattr(upcast_mod, "CURRENT_SCHEMA_VERSION", 3)
    monkeypatch.setattr(upcast_mod, "UPCASTERS", {})

    @register(EventKind.RUN_CREATED, 2)
    def to_v3(data: dict) -> dict:
        return data

    with pytest.raises(LedgerCorruption, match="no upcaster from schema_version 1"):
        upcast(EventKind.STATUS_CHANGED, {"kind": "status_changed", "schema_version": 1}, seq=4)


# EventStream -- the emit facade


class FakeAppender(EventAppender):
    """A no-op ledger seam that records calls and returns a real envelope."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object, EventEnvelope]] = []
        self._lock = threading.Lock()
        self._next = 0

    def append_next(self, run_id: str, payload: object) -> EventEnvelope:
        with self._lock:
            self._next += 1
            seq = self._next
        envelope = EventEnvelope(run_id=run_id, seq=seq, payload=payload)
        self.calls.append((run_id, payload, envelope))
        return envelope


def test_emit_without_ledger_returns_none():
    """With ledger=None, the sync emit is a silent no-op."""
    stream = EventStream(None, "r1")
    assert stream.emit(RunCreated()) is None


async def test_emit_async_without_ledger_returns_none():
    """With ledger=None, the async emit is a silent no-op."""
    stream = EventStream(None, "r1")
    assert await stream.emit_async(RunCreated()) is None


def test_emit_returns_appender_result_untouched():
    """emit returns what append_next returned and forwards run_id + payload."""
    appender = FakeAppender()
    stream = EventStream(appender, "r1")
    payload = RunCreated()
    result = stream.emit(payload)
    assert result is appender.calls[0][2]
    assert result.payload is payload
    assert appender.calls[0][0] == "r1"


async def test_emit_async_returns_appender_result_untouched():
    """emit_async returns what append_next returned and forwards run_id + payload."""
    appender = FakeAppender()
    stream = EventStream(appender, "r1")
    payload = RunCreated()
    result = await stream.emit_async(payload)
    assert result is appender.calls[0][2]
    assert result.payload is payload
    assert appender.calls[0][0] == "r1"


def test_stream_run_id_returns_constructor_value():
    """The run_id property echoes the constructor argument."""
    assert EventStream(None, "run-xyz").run_id == "run-xyz"


async def test_emit_async_concurrent_sequences_are_contiguous():
    """Concurrent emit_async appends serialize into distinct, contiguous sequences."""
    appender = FakeAppender()
    stream = EventStream(appender, "r1")
    results = await asyncio.gather(*[stream.emit_async(RunCreated()) for _ in range(50)])
    seqs = [e.seq for e in results]
    assert len(set(seqs)) == 50
    assert sorted(seqs) == list(range(1, 51))


async def test_emit_async_cancellation_waits_for_started_append() -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    class BlockingAppender(FakeAppender):
        def append_next(self, run_id: str, payload: object) -> EventEnvelope:
            started.set()
            assert release.wait(2)
            try:
                return super().append_next(run_id, payload)
            finally:
                finished.set()

    task = asyncio.create_task(EventStream(BlockingAppender(), "r1").emit_async(RunCreated()))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    try:
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


async def test_emit_async_cancellation_before_append_starts_is_durable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_to_thread(function, *args):
        entered.set()
        await release.wait()
        return function(*args)

    monkeypatch.setattr("northstack.events.stream.asyncio.to_thread", delayed_to_thread)
    appender = FakeAppender()
    task = asyncio.create_task(EventStream(appender, "r1").emit_async(RunCreated()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(appender.calls) == 1


async def test_emit_async_cancellation_is_not_replaced_by_append_failure() -> None:
    started, release = threading.Event(), threading.Event()

    class FailingAppender(FakeAppender):
        def append_next(self, run_id: str, payload: object) -> EventEnvelope:
            started.set()
            assert release.wait(2)
            raise OSError("injected append failure")

    task = asyncio.create_task(EventStream(FailingAppender(), "r1").emit_async(RunCreated()))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
