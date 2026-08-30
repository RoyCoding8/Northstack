"""Tests for the worker's context management: eviction, compaction, truncation fuse.

Covers:
  - Whole-round eviction never orphans a tool result from its call
  - The pinned prefix (system + objective) survives compaction
  - Compaction is a no-op below the trigger ratio and fires above it
  - The elision note lands on the objective, not as a second user turn
  - The MAX_TOKENS nudge loop terminates without relying on ``max_calls``
  - The WorkerEvent sink sees every turn, and a raising sink never fails a cell
"""

from __future__ import annotations

import tempfile

import pytest

from northstack.adapters.providers.wire import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelResponse,
    Usage,
)
from northstack.adapters.providers.gateway import HTTPProviderError, ModelGateway
from northstack.adapters.workspace.restricted import RestrictedWorkspace
from northstack.application.worker import (
    _CONTEXT_TARGET_RATIO,
    _CONTEXT_TRIGGER_RATIO,
    _ELISION_SENTINEL,
    _MAX_CONSECUTIVE_TRUNCATIONS,
    _MIN_RETAINED_SPANS,
    NativeWorker,
    WorkerEvent,
    WorkerEventKind,
    WorkerEventSink,
    _compact_messages,
    _evictable_spans,
    _note_elision,
)
from northstack.config import ModelProfile, NorthStackConfig, Protocol
from northstack.domain import Budget, GraphCell, WorkContract


def _conversation(rounds: int, filler: str = "x" * 400) -> list[ModelMessage]:
    """system + objective, then ``rounds`` of (assistant with tool call, 2 results)."""
    messages = [
        ModelMessage(role=MessageRole.SYSTEM, content="system"),
        ModelMessage(role=MessageRole.USER, content="objective"),
    ]
    for i in range(rounds):
        messages.append(ModelMessage(role=MessageRole.ASSISTANT, content=f"call {i} {filler}"))
        messages.append(ModelMessage(role=MessageRole.TOOL, content=f"result {i}a {filler}"))
        messages.append(ModelMessage(role=MessageRole.TOOL, content=f"result {i}b {filler}"))
    return messages


def _chars(messages: list[ModelMessage]) -> int:
    return sum(len(m.content) for m in messages)


def _assert_well_formed(messages: list[ModelMessage]) -> None:
    """Every TOOL message must follow the ASSISTANT turn that called it."""
    assert messages[0].role == MessageRole.SYSTEM
    assert messages[1].role == MessageRole.USER
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.TOOL:
            assert i > 0, "a tool result may never be the first message"
            prior = [m.role for m in messages[:i]]
            last_non_tool = next(r for r in reversed(prior) if r != MessageRole.TOOL)
            assert last_non_tool == MessageRole.ASSISTANT, (
                f"tool result at {i} is orphaned; nearest non-tool ancestor is {last_non_tool}"
            )


class TestEvictableSpans:
    def test_pins_system_and_objective(self):
        pinned, spans = _evictable_spans(_conversation(3))
        assert pinned == 2
        assert spans == [(2, 5), (5, 8), (8, 11)]

    def test_span_bundles_assistant_with_its_results(self):
        _, spans = _evictable_spans(_conversation(2))
        assert all(end - start == 3 for start, end in spans)

    def test_no_user_objective_pins_only_system(self):
        messages = [ModelMessage(role=MessageRole.SYSTEM, content="s")] + _conversation(1)[2:]
        pinned, _ = _evictable_spans(messages)
        assert pinned == 1


class TestCompaction:
    def test_no_op_below_trigger(self):
        messages = _conversation(4)
        before = list(messages)
        assert _compact_messages(messages, _chars, _chars(messages) * 10) == 0
        assert messages == before

    def test_fires_above_trigger_and_reaches_target(self):
        messages = _conversation(12)
        window = int(_chars(messages) / (_CONTEXT_TRIGGER_RATIO + 0.1))
        dropped = _compact_messages(messages, _chars, window)
        assert dropped > 0
        assert _chars(messages) <= window * _CONTEXT_TARGET_RATIO

    def test_eviction_never_orphans_a_tool_result(self):
        messages = _conversation(12)
        _compact_messages(messages, _chars, int(_chars(messages) / 2))
        _assert_well_formed(messages)

    def test_evicts_oldest_first(self):
        messages = _conversation(8)
        _compact_messages(messages, _chars, int(_chars(messages) / 2))
        assert "call 7" in messages[-3].content
        assert not any("call 0" in m.content for m in messages)

    def test_retains_a_floor_of_spans_when_window_is_impossible(self):
        messages = _conversation(10)
        _compact_messages(messages, _chars, 1)
        _, spans = _evictable_spans(messages)
        assert len(spans) == _MIN_RETAINED_SPANS
        _assert_well_formed(messages)

    def test_pinned_prefix_survives(self):
        messages = _conversation(10)
        _compact_messages(messages, _chars, 1)
        assert messages[0].content == "system"
        assert messages[1].content == "objective"


class TestElisionNote:
    def test_note_lands_on_the_objective(self):
        messages = _conversation(3)
        _note_elision(messages, 4)
        assert messages[1].role == MessageRole.USER
        assert messages[1].content.startswith("objective")
        assert "4 earlier tool round(s)" in messages[1].content
        assert sum(1 for m in messages if m.role == MessageRole.USER) == 1

    def test_note_is_rewritten_not_appended(self):
        messages = _conversation(3)
        _note_elision(messages, 4)
        _note_elision(messages, 9)
        assert messages[1].content.count(_ELISION_SENTINEL) == 1
        assert "9 earlier tool round(s)" in messages[1].content

    def test_no_objective_is_a_no_op(self):
        messages = [ModelMessage(role=MessageRole.SYSTEM, content="s")]
        _note_elision(messages, 3)
        assert messages[0].content == "s"


class TestTruncationFuse:
    """A MAX_TOKENS response with no tool calls appends a nudge and loops. With
    ``max_calls=0`` (the default) that loop has no other exit, so the fuse must
    stand on its own consecutive-truncation counter.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", ["partial answer", ""])
    async def test_fuse_trips_with_unlimited_max_calls(self, text: str):
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            return ModelResponse(
                text=text,
                finish_reason=FinishReason.MAX_TOKENS,
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="openai",
                model="test-model",
            )

        result = await _run_worker(fn, max_calls=0)
        assert calls["n"] == _MAX_CONSECUTIVE_TRUNCATIONS
        assert not result.ok and result.error_kind == "provider"
        assert "consecutive turn(s)" in result.error

    @pytest.mark.asyncio
    async def test_fewer_than_n_truncations_still_completes(self):
        finishes = [FinishReason.MAX_TOKENS] * 2 + [FinishReason.END_TURN]

        async def fn(req, prof, c, k):
            reason = finishes.pop(0) if finishes else FinishReason.END_TURN
            return ModelResponse(
                text="answer",
                finish_reason=reason,
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="openai",
                model="test-model",
            )

        result = await _run_worker(fn, max_calls=0)
        assert result.ok


def _recorder() -> tuple[list[WorkerEvent], WorkerEventSink]:
    seen: list[WorkerEvent] = []

    async def sink(event: WorkerEvent) -> None:
        seen.append(event)

    return seen, sink


class TestEventSink:
    @pytest.mark.asyncio
    async def test_a_clean_turn_publishes_start_and_completion(self):
        seen, sink = _recorder()

        async def fn(req, prof, c, k):
            return ModelResponse(
                text="done",
                finish_reason=FinishReason.END_TURN,
                usage=Usage(input_tokens=11, output_tokens=7),
                provider="openai",
                model="test-model",
            )

        result = await _run_worker(fn, max_calls=0, on_event=sink)
        assert result.ok
        assert [e.kind for e in seen] == [
            WorkerEventKind.TURN_STARTED,
            WorkerEventKind.MODEL_CALL_COMPLETED,
        ]
        assert all(e.turn == 1 for e in seen)
        assert seen[1].detail["output_tokens"] == 7
        assert seen[1].detail["finish_reason"] == FinishReason.END_TURN.value

    @pytest.mark.asyncio
    async def test_provider_error_publishes_a_failure(self):
        seen, sink = _recorder()

        async def fn(req, prof, c, k):
            raise HTTPProviderError(
                "Provider returned HTTP 502", status_code=502, provider="openai", model="m"
            )

        result = await _run_worker(fn, max_calls=0, on_event=sink)
        assert not result.ok
        assert seen[-1].kind == WorkerEventKind.MODEL_CALL_FAILED
        assert seen[-1].detail["error_kind"] == "provider"

    @pytest.mark.asyncio
    async def test_turn_number_advances_across_turns(self):
        seen, sink = _recorder()
        finishes = [FinishReason.MAX_TOKENS, FinishReason.END_TURN]

        async def fn(req, prof, c, k):
            return ModelResponse(
                text="partial",
                finish_reason=finishes.pop(0) if finishes else FinishReason.END_TURN,
                usage=Usage(input_tokens=1, output_tokens=1),
                provider="openai",
                model="test-model",
            )

        await _run_worker(fn, max_calls=0, on_event=sink)
        assert [e.turn for e in seen] == [1, 1, 2, 2]

    @pytest.mark.asyncio
    async def test_a_raising_sink_never_fails_the_cell(self):
        async def sink(event: WorkerEvent) -> None:
            raise RuntimeError("observer exploded")

        async def fn(req, prof, c, k):
            return ModelResponse(
                text="done",
                finish_reason=FinishReason.END_TURN,
                usage=Usage(input_tokens=1, output_tokens=1),
                provider="openai",
                model="test-model",
            )

        result = await _run_worker(fn, max_calls=0, on_event=sink)
        assert result.ok and result.text == "done"


async def _run_worker(fn, *, max_calls: int, on_event=None):
    profile = ModelProfile(
        name="cheap-worker",
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test-model",
        max_concurrency=4,
        max_output_tokens=4096,
    )
    gateway = ModelGateway(NorthStackConfig(name="test", profiles=[profile]))
    adapter = type("Adapter", (), {"complete": staticmethod(fn)})()
    gateway._adapters[Protocol.OPENAI_CHAT] = adapter
    cell = GraphCell(
        id="cell-1",
        name="Test Cell",
        contract=WorkContract(
            id="wc-1",
            objective="x",
            budget=Budget(token_limit=100_000, cost_limit_usd=5.0, max_calls=max_calls),
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        return await NativeWorker(gateway, RestrictedWorkspace(td)).run(
            cell, "cheap-worker", [], on_event=on_event
        )
