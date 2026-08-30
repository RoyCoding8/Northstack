"""Tests for the OpenTelemetry spans built over the worker event stream.

Covers:
  - One cell span, one child span per turn, closed when the cell closes
  - Model usage lands on the turn it belongs to, under GenAI attribute names
  - A failed model call marks its turn span ERROR
  - Tool calls and compactions are recorded as span events
  - ``fanout`` feeds every live sink and tolerates a ``None`` member
"""

from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from northstack.application.tracing import cell_trace, fanout
from northstack.application.worker import WorkerEvent, WorkerEventKind


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The raw attribute, not get_tracer_provider(): that resolves None to a
    # ProxyTracerProvider, and restoring the proxy *as* the global provider
    # makes it delegate to itself -- every later get_tracer() recurses.
    previous = trace._TRACER_PROVIDER
    trace._TRACER_PROVIDER = provider
    yield exporter
    trace._TRACER_PROVIDER = previous
    provider.shutdown()


def _event(kind: WorkerEventKind, turn: int = 1, **detail) -> WorkerEvent:
    return WorkerEvent(kind=kind, turn=turn, detail=detail)


async def _drive(*events: WorkerEvent) -> None:
    with cell_trace("run-1", "cell-1", "cheap-worker") as sink:
        assert sink is not None
        for event in events:
            await sink(event)


def _by_name(exporter, name: str) -> list:
    return [s for s in exporter.get_finished_spans() if s.name == name]


class TestSpanShape:
    @pytest.mark.asyncio
    async def test_one_cell_span_carries_the_identity(self, spans):
        await _drive()
        (cell,) = _by_name(spans, "northstack.cell")
        assert cell.attributes["northstack.run_id"] == "run-1"
        assert cell.attributes["northstack.cell_id"] == "cell-1"
        assert cell.attributes["gen_ai.request.model"] == "cheap-worker"

    @pytest.mark.asyncio
    async def test_each_turn_gets_its_own_child_span(self, spans):
        await _drive(
            _event(WorkerEventKind.TURN_STARTED, turn=1, messages=2),
            _event(WorkerEventKind.MODEL_CALL_COMPLETED, turn=1),
            _event(WorkerEventKind.TURN_STARTED, turn=2, messages=5),
        )
        turns = _by_name(spans, "northstack.turn")
        assert [t.attributes["northstack.turn"] for t in turns] == [1, 2]
        (cell,) = _by_name(spans, "northstack.cell")
        assert all(t.parent.span_id == cell.context.span_id for t in turns)

    @pytest.mark.asyncio
    async def test_an_open_turn_is_closed_by_the_cell(self, spans):
        await _drive(_event(WorkerEventKind.TURN_STARTED))
        assert len(_by_name(spans, "northstack.turn")) == 1


class TestAttributes:
    @pytest.mark.asyncio
    async def test_usage_lands_under_genai_names(self, spans):
        await _drive(
            _event(WorkerEventKind.TURN_STARTED),
            _event(
                WorkerEventKind.MODEL_CALL_COMPLETED,
                input_tokens=1200,
                output_tokens=90,
                cost_usd=0.0,
                finish_reason="end_turn",
            ),
        )
        (turn,) = _by_name(spans, "northstack.turn")
        assert turn.attributes["gen_ai.usage.input_tokens"] == 1200
        assert turn.attributes["gen_ai.usage.output_tokens"] == 90
        assert turn.attributes["gen_ai.response.finish_reason"] == "end_turn"
        assert turn.attributes["northstack.cost_usd"] == 0.0

    @pytest.mark.asyncio
    async def test_a_failed_call_marks_the_turn_as_error(self, spans):
        await _drive(
            _event(WorkerEventKind.TURN_STARTED),
            _event(WorkerEventKind.MODEL_CALL_FAILED, error_kind="provider", error="HTTP 502"),
        )
        (turn,) = _by_name(spans, "northstack.turn")
        assert turn.status.status_code is StatusCode.ERROR
        assert turn.status.description == "HTTP 502"
        assert turn.attributes["northstack.error_kind"] == "provider"

    @pytest.mark.asyncio
    async def test_tool_calls_are_events_on_their_turn(self, spans):
        await _drive(
            _event(WorkerEventKind.TURN_STARTED),
            _event(WorkerEventKind.TOOL_CALL_COMPLETED, tool="read", ok=True, duration_ms=4.0),
            _event(WorkerEventKind.TOOL_CALL_COMPLETED, tool="write", ok=False, duration_ms=9.0),
        )
        (turn,) = _by_name(spans, "northstack.turn")
        assert [e.name for e in turn.events] == ["tool_call", "tool_call"]
        assert [e.attributes["tool"] for e in turn.events] == ["read", "write"]
        assert turn.events[1].attributes["ok"] is False

    @pytest.mark.asyncio
    async def test_compaction_is_an_event_on_the_cell(self, spans):
        """It happens between turns, so the turn span would be the wrong home."""
        await _drive(
            _event(WorkerEventKind.CONTEXT_COMPACTED, turn=0, rounds_dropped=3),
            _event(WorkerEventKind.TURN_STARTED),
        )
        (cell,) = _by_name(spans, "northstack.cell")
        assert [e.name for e in cell.events] == ["context_compacted"]
        assert cell.events[0].attributes["rounds_dropped"] == 3

    @pytest.mark.asyncio
    async def test_non_scalar_detail_is_dropped(self, spans):
        await _drive(_event(WorkerEventKind.TURN_STARTED, messages=2, payload={"secret": "value"}))
        (turn,) = _by_name(spans, "northstack.turn")
        assert "payload" not in turn.attributes
        assert turn.attributes["messages"] == 2


class TestFanout:
    @pytest.mark.asyncio
    async def test_every_live_sink_sees_the_event(self):
        a, b = [], []

        async def to(seen):
            async def sink(event):
                seen.append(event.kind)

            return sink

        sink = fanout(await to(a), None, await to(b))
        await sink(_event(WorkerEventKind.TURN_STARTED))
        assert a == b == [WorkerEventKind.TURN_STARTED]

    @pytest.mark.asyncio
    async def test_all_none_is_a_working_no_op(self):
        await fanout(None, None)(_event(WorkerEventKind.TURN_STARTED))
