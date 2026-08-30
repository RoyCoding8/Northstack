"""OpenTelemetry spans over the worker's :class:`WorkerEvent` stream.

Optional by construction: ``opentelemetry-api`` is an extra, and with it
absent :func:`cell_trace` yields a tracer whose sink is ``None``.  The worker
then runs untraced instead of the package failing to import -- tracing is an
observation, and an observation may never be a reason a cell cannot start.

The event stream is flat but the shape it describes is nested, so the mapping
is: one span per cell, one child span per model turn (open at
``turn_started``, closed by the next one or by the cell), and model/tool
outcomes recorded on the turn they belong to.  Tool calls are span events
rather than spans of their own -- the stream reports a completion and its
duration, never a start, so a child span would have to invent its own
begin timestamp.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

from northstack.application.worker import WorkerEvent, WorkerEventKind

try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover - exercised by the no-otel install
    trace = None  # type: ignore[assignment]

WorkerEventSink = Callable[[WorkerEvent], Awaitable[None]]

_MODEL_ATTRS = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_tokens": "gen_ai.usage.cache_read_tokens",
    "finish_reason": "gen_ai.response.finish_reason",
    "cost_usd": "northstack.cost_usd",
    "tool_calls": "northstack.tool_calls",
}


class _CellTrace:
    """Holds the cell span and the turn span currently open beneath it."""

    def __init__(self, tracer: Any, cell_span: Any) -> None:
        self._tracer = tracer
        self._cell = cell_span
        self._turn: Any = None

    def close(self) -> None:
        self._end_turn()
        self._cell.end()

    def _end_turn(self) -> None:
        if self._turn is not None:
            self._turn.end()
            self._turn = None

    async def sink(self, event: WorkerEvent) -> None:
        detail = event.detail
        if event.kind is WorkerEventKind.TURN_STARTED:
            self._end_turn()
            self._turn = self._tracer.start_span(
                "northstack.turn",
                context=trace.set_span_in_context(self._cell),
                attributes={"northstack.turn": event.turn, **_scalars(detail)},
            )
        elif event.kind is WorkerEventKind.CONTEXT_COMPACTED:
            self._cell.add_event("context_compacted", _scalars(detail))
        elif event.kind is WorkerEventKind.MODEL_CALL_COMPLETED and self._turn is not None:
            self._turn.set_attributes(
                {_MODEL_ATTRS[k]: v for k, v in _scalars(detail).items() if k in _MODEL_ATTRS}
            )
        elif event.kind is WorkerEventKind.MODEL_CALL_FAILED and self._turn is not None:
            self._turn.set_status(trace.Status(trace.StatusCode.ERROR, detail.get("error", "")))
            self._turn.set_attributes({"northstack.error_kind": detail.get("error_kind", "")})
        elif event.kind is WorkerEventKind.TOOL_CALL_COMPLETED and self._turn is not None:
            self._turn.add_event("tool_call", _scalars(detail))


def _scalars(detail: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in detail.items() if isinstance(v, str | int | float | bool)}


@contextlib.contextmanager
def cell_trace(run_id: str, cell_id: str, profile: str) -> Iterator[WorkerEventSink | None]:
    """Open a span for one cell attempt; yield the sink that fills it in."""
    if trace is None:
        yield None
        return
    tracer = trace.get_tracer("northstack")
    span = tracer.start_span(
        "northstack.cell",
        attributes={
            "northstack.run_id": run_id,
            "northstack.cell_id": cell_id,
            "gen_ai.request.model": profile,
        },
    )
    state = _CellTrace(tracer, span)
    try:
        yield state.sink
    finally:
        state.close()


def fanout(*sinks: WorkerEventSink | None) -> WorkerEventSink:
    """One sink that feeds several. A ``None`` member is simply not wired."""
    live = [s for s in sinks if s is not None]

    async def sink(event: WorkerEvent) -> None:
        for one in live:
            await one(event)

    return sink
