"""Tests for parallel tool execution within one model round.

Covers:
  - Consecutive read-only calls batch; a mutating call runs alone, in place
  - The per-round cap bounds how many tools open at once
  - Reads in a batch actually overlap on the wall clock
  - Results are appended in call order, never completion order
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any

import pytest

from northstack.adapters.providers.wire import (
    FinishReason,
    MessageRole,
    ModelResponse,
    ToolCall,
    Usage,
)
from northstack.adapters.providers.gateway import ModelGateway
from northstack.adapters.workspace.restricted import RestrictedWorkspace, ToolResult
from northstack.application.tools.registry import ToolContext, ToolRegistry
from northstack.application.worker import (
    _MAX_PARALLEL_TOOL_CALLS,
    NativeWorker,
    _tool_batches,
)
from northstack.config import ModelProfile, NorthStackConfig, Protocol
from northstack.domain import Budget, GraphCell, WorkContract

_PARAMS: dict[str, Any] = {"type": "object", "properties": {}}


class _FakeTool:
    """A tool that sleeps, records when it ran, and never touches the disk."""

    def __init__(self, name: str, *, mutating: bool, delay: float = 0.0) -> None:
        self.name = name
        self.description = name
        self.parameters = _PARAMS
        self.mutating = mutating
        self._delay = delay
        self.log: list[str] = []

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        self.log.append(f"enter {self.name}")
        await asyncio.sleep(self._delay)
        self.log.append(f"exit {self.name}")
        return ToolResult(ok=True, operation=self.name, data=self.name.encode())


def _registry(*tools: _FakeTool) -> ToolRegistry:
    return ToolRegistry(list(tools))


def _calls(*names: str) -> list[ToolCall]:
    return [ToolCall(id=f"c{i}", name=n, arguments={}) for i, n in enumerate(names)]


def _names(batches: list[list[ToolCall]]) -> list[list[str]]:
    return [[c.name for c in batch] for batch in batches]


class TestBatching:
    def test_consecutive_reads_batch(self):
        registry = _registry(_FakeTool("read", mutating=False))
        assert _names(_tool_batches(_calls("read", "read", "read"), registry)) == [["read"] * 3]

    def test_a_mutation_runs_alone_and_in_place(self):
        registry = _registry(_FakeTool("read", mutating=False), _FakeTool("write", mutating=True))
        batches = _tool_batches(_calls("read", "read", "write", "read", "read"), registry)
        assert _names(batches) == [["read", "read"], ["write"], ["read", "read"]]

    def test_back_to_back_mutations_never_share_a_batch(self):
        registry = _registry(_FakeTool("write", mutating=True))
        assert _names(_tool_batches(_calls("write", "write"), registry)) == [
            ["write"],
            ["write"],
        ]

    def test_the_cap_bounds_a_greedy_round(self):
        registry = _registry(_FakeTool("read", mutating=False))
        batches = _tool_batches(_calls(*["read"] * (_MAX_PARALLEL_TOOL_CALLS + 2)), registry)
        assert [len(b) for b in batches] == [_MAX_PARALLEL_TOOL_CALLS, 2]

    def test_an_unregistered_tool_is_not_treated_as_mutating(self):
        """It resolves to ``Unknown tool`` without touching the workspace, so
        holding a batch open for it would cost latency for nothing.
        """
        registry = _registry(_FakeTool("read", mutating=False))
        assert _names(_tool_batches(_calls("read", "ghost"), registry)) == [["read", "ghost"]]

    def test_no_calls_is_no_batches(self):
        assert _tool_batches([], _registry()) == []


class TestConcurrentExecution:
    @pytest.mark.asyncio
    async def test_reads_in_a_batch_overlap(self):
        slow = _FakeTool("read", mutating=False, delay=0.15)
        started = time.perf_counter()
        result, messages = await _run_round(_registry(slow), _calls(*["read"] * 4))
        elapsed = time.perf_counter() - started

        assert result.ok
        assert len(messages) == 4
        assert elapsed < 0.45, f"4x0.15s reads took {elapsed:.2f}s -- they ran serially"
        assert slow.log[:4] == ["enter read"] * 4

    @pytest.mark.asyncio
    async def test_a_mutation_does_not_overlap_its_neighbours(self):
        read = _FakeTool("read", mutating=False, delay=0.05)
        write = _FakeTool("write", mutating=True, delay=0.01)
        order: list[str] = []
        read.log = write.log = order

        await _run_round(_registry(read, write), _calls("read", "write", "read"))
        assert order == [
            "enter read",
            "exit read",
            "enter write",
            "exit write",
            "enter read",
            "exit read",
        ]

    @pytest.mark.asyncio
    async def test_results_follow_call_order_not_completion_order(self):
        registry = _registry(
            _FakeTool("slow", mutating=False, delay=0.12),
            _FakeTool("fast", mutating=False, delay=0.0),
        )
        _, messages = await _run_round(registry, _calls("slow", "fast", "slow"))
        assert [m.content for m in messages] == ["slow", "fast", "slow"]


async def _run_round(registry: ToolRegistry, calls: list[ToolCall]):
    """Drive one tool round, then a clean answer; return (result, tool messages)."""
    responses = [
        ModelResponse(
            text="",
            finish_reason=FinishReason.TOOL_USE,
            tool_calls=calls,
            usage=Usage(input_tokens=1, output_tokens=1),
            provider="openai",
            model="test-model",
        ),
        ModelResponse(
            text="done",
            finish_reason=FinishReason.END_TURN,
            usage=Usage(input_tokens=1, output_tokens=1),
            provider="openai",
            model="test-model",
        ),
    ]
    seen: list[Any] = []

    async def fn(req, prof, c, k):
        seen[:] = req.messages
        return responses.pop(0) if responses else responses[-1]

    profile = ModelProfile(
        name="cheap-worker",
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test-model",
        max_concurrency=4,
        max_output_tokens=4096,
    )
    gateway = ModelGateway(NorthStackConfig(name="test", profiles=[profile]))
    gateway._adapters[Protocol.OPENAI_CHAT] = type("Adapter", (), {"complete": staticmethod(fn)})()
    cell = GraphCell(
        id="cell-1",
        name="Test Cell",
        contract=WorkContract(
            id="wc-1",
            objective="x",
            budget=Budget(token_limit=100_000, cost_limit_usd=5.0, max_calls=0),
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        worker = NativeWorker(gateway, RestrictedWorkspace(td), tool_registry=registry)
        result = await worker.run(cell, "cheap-worker", registry.advertised())
    return result, [m for m in seen if m.role == MessageRole.TOOL]
