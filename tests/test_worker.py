"""Tests for NativeWorker: bounded model/tool loop.

Covers:
  - Worker read/write/tool loop execution
  - Lease release on completion and error
  - Budget termination (max calls, tool rounds, tokens, wall time, cost)
  - Tool validation against JSON schema (jsonschema library)
  - Schema repair (one bounded attempt)
  - Structured failure on budget, provider, schema, or tool errors
  - Only allowlisted mediated tools with JSON schemas
  - Named command-profile tool execution
  - Mutation lease acquisition and release
  - WorkerResult structured output
  - profile_name as explicit kwarg (contract.id != profile.name regression)
  - Cost enforcement with actual configured pricing
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from northstack.adapters.providers.gateway import (
    AuthProviderError,
    CapabilityError,
    HTTPProviderError,
    ModelGateway,
    ProviderError,
)
from northstack.adapters.providers.wire import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    Usage,
)
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace, ToolResult
from northstack.application.tools.registry import ToolRegistry
from northstack.application.worker import (
    NativeWorker,
    WorkerResult,
    _estimate_request_input_tokens,
    _try_parse_json_response,
    _validate_tool_args,
    _validate_with_jsonschema,
)
from northstack.config import ModelProfile, NorthStackConfig, Protocol
from northstack.domain import Budget, GraphCell, WorkContract

# Helpers


def _make_profile(
    name: str = "cheap-worker",
    input_price: float = 1.0,
    output_price: float = 5.0,
    max_concurrency: int = 4,
    max_output_tokens: int = 4096,
) -> ModelProfile:
    return ModelProfile(
        name=name,
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test-model",
        max_concurrency=max_concurrency,
        requests_per_minute=1000,
        input_price_per_million_usd=input_price,
        output_price_per_million_usd=output_price,
        max_output_tokens=max_output_tokens,
    )


def _make_config(profiles: list[ModelProfile] | None = None) -> NorthStackConfig:
    return NorthStackConfig(name="test", profiles=profiles or [_make_profile()])


def _make_cell(
    objective: str = "Build a hello world app",
    budget_tokens: int = 100_000,
    budget_cost: float = 5.0,
    max_calls: int = 0,
    max_tool_rounds: int = 0,
    max_wall_time: float = 0.0,
    max_retries: int = 0,
    allowed_tools: list[str] | None = None,
) -> GraphCell:
    return GraphCell(
        id="cell-1",
        name="Test Cell",
        contract=WorkContract(
            id="wc-1",
            objective=objective,
            budget=Budget(
                token_limit=budget_tokens,
                cost_limit_usd=budget_cost,
                max_calls=max_calls,
                max_tool_rounds=max_tool_rounds,
                max_wall_time_seconds=max_wall_time,
                max_retries=max_retries,
            ),
            allowed_tools=allowed_tools or [],
        ),
    )


def _wire_adapter(gateway: ModelGateway, complete_fn):
    """Wire a mock adapter into gateway.

    complete_fn signature: async (request, profile, client, api_key).
    """
    mock_adapter = MagicMock()
    mock_adapter.complete = complete_fn
    gateway._adapters[Protocol.OPENAI_CHAT] = mock_adapter


def _ok_response(text="Done!", **kw):
    defaults = {
        "text": text,
        "finish_reason": FinishReason.END_TURN,
        "usage": Usage(input_tokens=10, output_tokens=5),
        "provider": "openai",
        "model": "test-model",
    }
    defaults.update(kw)
    return ModelResponse(**defaults)


def _tool_response(call_id, tool_name, arguments=None, **kw):
    defaults = {
        "text": "",
        "tool_calls": [ToolCall(id=call_id, name=tool_name, arguments=arguments or {})],
        "finish_reason": FinishReason.TOOL_USE,
        "usage": Usage(input_tokens=10, output_tokens=5),
        "provider": "openai",
        "model": "test-model",
    }
    defaults.update(kw)
    return ModelResponse(**defaults)


# Tool arg validation


class TestValidateToolArgs:
    def test_valid_args(self):
        td = {
            "get_weather": ToolDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        }
        assert (
            _validate_tool_args(
                ToolCall(id="tc1", name="get_weather", arguments={"city": "NYC"}), td
            )
            is None
        )

    def test_unknown_tool(self):
        assert "Unknown tool" in _validate_tool_args(ToolCall(id="tc1", name="x", arguments={}), {})

    def test_missing_required(self):
        td = {
            "g": ToolDefinition(
                name="g",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                },
            )
        }
        assert "Missing required" in _validate_tool_args(
            ToolCall(id="t", name="g", arguments={}), td
        )

    def test_wrong_type(self):
        td = {
            "g": ToolDefinition(
                name="g", parameters={"type": "object", "properties": {"x": {"type": "string"}}}
            )
        }
        assert "must be string" in _validate_tool_args(
            ToolCall(id="t", name="g", arguments={"x": 123}), td
        )


# JSON schema validation (jsonschema library)


class TestJsonschemaValidation:
    def test_valid(self):
        s = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
        assert _validate_with_jsonschema({"x": "hi"}, s) is None

    def test_missing_required(self):
        s = {"type": "object", "required": ["x"]}
        assert _validate_with_jsonschema({}, s) is not None

    def test_additional_properties_rejected(self):
        s = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        assert _validate_with_jsonschema({"x": "hi", "y": 1}, s) is not None

    def test_enum(self):
        s = {"type": "object", "properties": {"v": {"type": "string", "enum": ["a", "b"]}}}
        assert _validate_with_jsonschema({"v": "a"}, s) is None
        assert _validate_with_jsonschema({"v": "c"}, s) is not None

    def test_oneof(self):
        s = {
            "type": "object",
            "properties": {"v": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
        }
        assert _validate_with_jsonschema({"v": "x"}, s) is None
        assert _validate_with_jsonschema({"v": 1}, s) is None
        assert _validate_with_jsonschema({"v": [1]}, s) is not None

    def test_too_large(self):
        s = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(10000)}}
        assert _validate_with_jsonschema({}, s) is not None


# JSON parsing


class TestJsonParsing:
    def test_valid(self):
        r, e = _try_parse_json_response('{"x": 1}', None)
        assert r == {"x": 1} and e is None

    def test_code_block(self):
        r, _ = _try_parse_json_response('```json\n{"x":1}\n```', None)
        assert r == {"x": 1}

    def test_invalid(self):
        r, e = _try_parse_json_response("nope", None)
        assert r is None and e is not None

    def test_schema_pass(self):
        s = {"type": "object", "required": ["x"]}
        r, e = _try_parse_json_response('{"x":1}', s)
        assert r is not None and e is None

    def test_schema_fail(self):
        r, e = _try_parse_json_response('{"y":1}', {"type": "object", "required": ["x"]})
        assert r is None and e is not None


# Basic execution


class TestBasicExecution:
    @pytest.mark.asyncio
    async def test_simple_text(self):
        async def fn(req, prof, c, k):
            return _ok_response("Hello!")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            result = await w.run(_make_cell(), profile_name="cheap-worker", tool_defs=[])
            assert result.ok and result.text == "Hello!" and result.tool_rounds == 0
            assert (result.api_calls, result.successful_api_calls) == (1, 1)

    @pytest.mark.asyncio
    async def test_tool_call_round(self):
        call_n = 0

        async def fn(req, prof, client, key):
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                return _tool_response("tc1", "read", {"path": "f.txt"})
            return _ok_response("processed")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "f.txt").write_text("hello")
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ],
            )
            assert r.ok and r.text == "processed" and r.tool_calls_made == 1

    @pytest.mark.asyncio
    async def test_streaming_gateway_beats_progress_per_delta(self):
        """A streaming-capable gateway's complete_stream is used and each delta
        re-beats the progress callback -- a minutes-long stream keeps the stall
        detector satisfied.  The gateway is injected as a plain object; no
        monkeypatch."""
        from northstack.adapters.providers.wire import TextDelta

        beats: list[str] = []
        seen_deltas: list[str] = []

        class _StreamingGateway:
            def profile(self, name):
                return _make_profile(name)

            async def complete(self, request):
                raise AssertionError("complete() must not run when complete_stream exists")

            async def complete_stream(self, request, *, on_delta=None):
                for d in (TextDelta(text="par"), TextDelta(text="tial")):
                    if on_delta is not None:
                        on_delta(d)
                        seen_deltas.append(d.text)
                return _ok_response("partial")

        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(_StreamingGateway(), RestrictedWorkspace(td))
            result = await w.run(
                _make_cell(),
                profile_name="cheap-worker",
                tool_defs=[],
                on_progress=lambda: beats.append("beat"),
            )
        assert result.ok and result.text == "partial"
        assert seen_deltas == ["par", "tial"]
        # record_call fires once per response plus one beat per delta.
        assert len(beats) >= 3

    @pytest.mark.asyncio
    async def test_multi_tool_calls(self):
        call_n = 0

        async def fn(req, prof, client, key):
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                return ModelResponse(
                    text="",
                    tool_calls=[
                        ToolCall(id="t1", name="read", arguments={"path": "a.txt"}),
                        ToolCall(id="t2", name="read", arguments={"path": "b.txt"}),
                    ],
                    finish_reason=FinishReason.TOOL_USE,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    provider="openai",
                    model="m",
                )
            return _ok_response("both")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.txt").write_text("a")
            (Path(td) / "b.txt").write_text("b")
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ],
            )
            assert r.tool_calls_made == 2


# Empty MAX_TOKENS response must not be accepted as success
# (live finding: reasoning models that exhaust max_output_tokens on reasoning
#  return content=null; the adapter yields text="" + finish=MAX_TOKENS, and
#  with no tool calls the worker must NOT treat that as ok=True)


class TestEmptyMaxTokensNotSuccess:
    @pytest.mark.asyncio
    async def test_empty_max_tokens_no_tools_is_not_success(self):
        """An empty-text MAX_TOKENS response with no tool calls is a transient
        truncation failure, not a completed answer."""

        async def fn(req, prof, c, k):
            return _ok_response(
                "",
                finish_reason=FinishReason.MAX_TOKENS,
                usage=Usage(input_tokens=10, output_tokens=4096),
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(max_calls=1), "cheap-worker", [])
            assert not r.ok, "empty MAX_TOKENS response wrongly accepted as success"
            assert r.error_kind == "provider"

    @pytest.mark.asyncio
    async def test_nonempty_max_tokens_still_truncation_failure(self):
        """Even if some text leaked out, MAX_TOKENS with no tool calls means the
        model was truncated mid-answer -- not a clean completion. Treat as
        transient so recovery can retry with a larger output budget."""

        async def fn(req, prof, c, k):
            return _ok_response(
                "partial answer...",
                finish_reason=FinishReason.MAX_TOKENS,
                usage=Usage(input_tokens=10, output_tokens=4096),
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(max_calls=1), "cheap-worker", [])
            assert not r.ok
            assert r.error_kind == "provider"

    @pytest.mark.asyncio
    async def test_the_second_nudge_stops_asking_for_more_prose(self):
        """A model that truncates twice is rambling. "Continue the answer" buys
        another ramble, so the second nudge must name the two shapes that end
        the loop instead.
        """
        seen: list[str] = []

        async def fn(req, prof, c, k):
            seen.append(req.messages[-1].content or "")
            return _ok_response(
                "and furthermore...",
                finish_reason=FinishReason.MAX_TOKENS,
                usage=Usage(input_tokens=10, output_tokens=4096),
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(max_calls=0), "cheap-worker", [])
        assert not r.ok and r.error_kind == "provider"
        assert len(seen) == 3, f"fuse should stop at 3 turns, saw {len(seen)}"
        assert "more concisely" in seen[1]
        assert "single tool call" in seen[2]

    @pytest.mark.asyncio
    async def test_empty_max_tokens_then_real_answer_recovers(self):
        """If the first call is truncated empty but a retry yields a real answer,
        the worker should surface the recovered answer as success."""

        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _ok_response(
                    "",
                    finish_reason=FinishReason.MAX_TOKENS,
                    usage=Usage(input_tokens=10, output_tokens=4096),
                )
            return _ok_response("The real answer.")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(max_calls=3), "cheap-worker", [])
            assert r.ok and r.text == "The real answer."


# Budget


class TestBudget:
    @pytest.mark.asyncio
    async def test_max_calls(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            return _tool_response(f"t{n}", "read", {"path": "."})

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(max_calls=2),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ],
            )
            assert not r.ok and "API call limit" in r.error

    @pytest.mark.asyncio
    async def test_max_tool_rounds(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            return _tool_response(f"t{n}", "read", {"path": "."})

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(max_tool_rounds=1),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ],
            )
            assert not r.ok and "Tool round limit" in r.error

    @pytest.mark.asyncio
    async def test_wall_time(self):
        cancelled = False

        async def fn(req, prof, c, k):
            nonlocal cancelled
            try:
                await asyncio.sleep(1)
                return _tool_response("t1", "read", {"path": "."})
            finally:
                cancelled = True

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            started = time.perf_counter()
            r = await w.run(
                _make_cell(max_wall_time=0.02),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ],
            )
            assert not r.ok and "Wall time" in r.error
            assert time.perf_counter() - started < 0.5
            assert cancelled
            assert (r.api_calls, r.successful_api_calls) == (1, 0)

    @pytest.mark.asyncio
    async def test_wall_time_cancels_tool_call(self):
        cancelled = False

        class SlowTool:
            name, description, parameters, mutating = "slow", "slow", {"type": "object"}, False

            async def execute(self, ctx, args):  # type: ignore[no-untyped-def]
                nonlocal cancelled
                try:
                    await asyncio.sleep(1)
                    return ToolResult(ok=True, output="late")
                finally:
                    cancelled = True

        async def fn(req, prof, c, k):
            return _tool_response("t1", "slow", {})

        gw = ModelGateway(_make_config())
        _wire_adapter(gw, fn)
        registry = ToolRegistry([SlowTool()])
        with tempfile.TemporaryDirectory() as td:
            worker = NativeWorker(gw, RestrictedWorkspace(td), tool_registry=registry)
            started = time.perf_counter()
            result = await worker.run(
                _make_cell(max_wall_time=0.02, allowed_tools=["slow"]),
                "cheap-worker",
                registry.advertised(),
            )
        assert not result.ok and result.error_kind == "budget"
        assert time.perf_counter() - started < 0.5
        assert cancelled


# Request input reservation


class TestRequestInputReservation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(("delta", "called"), [(0, False), (1, True)])
    async def test_cell_budget_boundary_reserves_estimated_input(
        self, delta: int, called: bool
    ) -> None:
        prompt = "Objective: Build a hello world app"
        probe = ModelRequest(
            profile_name="cheap-worker",
            messages=[ModelMessage(role=MessageRole.USER, content=prompt)],
            max_output_tokens=1,
        )
        estimate = _estimate_request_input_tokens(probe)
        requests: list[ModelRequest] = []

        async def fn(request, profile, client, api_key):  # type: ignore[no-untyped-def]
            requests.append(request)
            return _ok_response()

        gateway = ModelGateway(_make_config())
        _wire_adapter(gateway, fn)
        with tempfile.TemporaryDirectory() as directory:
            result = await NativeWorker(gateway, RestrictedWorkspace(directory)).run(
                _make_cell(budget_tokens=estimate + delta), "cheap-worker", []
            )
        assert bool(requests) is called
        if called:
            assert result.ok and requests[0].max_output_tokens == 1
        else:
            assert not result.ok and result.error_kind == "budget"


# Provider errors


class TestProviderErrors:
    @pytest.mark.asyncio
    async def test_provider_error(self):
        async def fn(req, prof, c, k):
            raise ProviderError("down", provider="openai", model="m")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(), "cheap-worker", [])
            assert not r.ok and r.error_kind == "provider" and "down" in r.error
            assert (r.api_calls, r.successful_api_calls) == (1, 0)

    @pytest.mark.asyncio
    async def test_unknown_profile(self):
        cfg = _make_config()
        gw = ModelGateway(cfg)
        w = NativeWorker(gw, MagicMock(spec=RestrictedWorkspace))
        r = await w.run(_make_cell(), "nonexistent", [])
        assert not r.ok and r.error_kind == "configuration" and "Profile not found" in r.error

    @pytest.mark.asyncio
    async def test_capability_error_is_not_transient_provider_failure(self):
        async def fn(req, prof, c, k):
            raise CapabilityError("unsupported feature", provider="openai", model="m")

        gw = ModelGateway(_make_config())
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            result = await NativeWorker(gw, RestrictedWorkspace(td)).run(
                _make_cell(), "cheap-worker", []
            )
        assert not result.ok and result.error_kind == "capability"


# Transient provider-error retry (task #20)


class TestTransientRetry:
    """The worker makes ONE ``gateway.complete()`` call per turn and does NOT
    retry transient provider errors internally -- retry is the
    orchestrator/cell_runner's job under central accounting.  A transient
    provider error surfaces as a single
    ``WorkerResult(ok=False, error_kind="provider")``. These tests pin that
    contract: whatever the provider fault, the worker calls the gateway exactly
    once and returns the typed failure; it never multiplies the orchestrator's
    retry budget by its own internal loop (the squared-retry bug).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    async def test_transient_http_status_surfaces_as_single_failure(self, status: int):
        """A retryable 503 is no longer retried in the worker -- one call,
        ``ok=False``, ``error_kind="provider"``. Retry moves to the orchestrator.
        """
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            raise HTTPProviderError(
                f"Provider returned HTTP {status}",
                status_code=status,
                provider="openai",
                model="m",
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(), "cheap-worker", [])
        assert calls["n"] == 1, f"worker must make exactly 1 call, got {calls['n']}"
        assert not r.ok and r.error_kind == "provider"
        assert (r.api_calls, r.successful_api_calls) == (1, 0)

    @pytest.mark.asyncio
    async def test_http_400_is_not_classified_transient(self):
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            raise HTTPProviderError(
                "Provider returned HTTP 400", status_code=400, provider="openai", model="m"
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(), "cheap-worker", [])
        assert calls["n"] == 1, f"worker must make exactly 1 call, got {calls['n']}"
        assert not r.ok and r.error_kind == "configuration"

    @pytest.mark.asyncio
    async def test_auth_error_surfaces_as_single_failure(self):
        """AuthProviderError is not transient -- and with no internal retry loop
        at all, *every* provider fault fails in one call. Auth still surfaces as
        ``ok=False``, ``error_kind="provider"``.
        """
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            raise AuthProviderError("Authentication failed", provider="openai", model="m")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(), "cheap-worker", [])
        assert calls["n"] == 1, "worker must make exactly 1 call"
        assert not r.ok and r.error_kind == "authentication" and "Authentication" in r.error

    @pytest.mark.asyncio
    async def test_max_retries_does_not_drive_worker_attempts(self):
        """``max_retries`` no longer controls worker-internal attempts -- the
        worker ignores it for provider calls and makes exactly 1. The retry cap
        is the orchestrator's. A persistent 503 with ``max_retries=1`` is a
        single ``ok=False`` call, not 2.
        """
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            raise HTTPProviderError(
                "Provider returned HTTP 503", status_code=503, provider="openai", model="m"
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        cell = GraphCell(
            id="cell-1",
            name="Test Cell",
            contract=WorkContract(
                id="wc-1",
                objective="x",
                budget=Budget(
                    token_limit=100_000,
                    cost_limit_usd=5.0,
                    max_calls=0,
                    max_tool_rounds=0,
                    max_wall_time_seconds=0.0,
                    max_retries=1,
                ),
                allowed_tools=[],
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(cell, "cheap-worker", [])
        # max_retries=1 no longer buys a second worker attempt; exactly 1 call.
        assert calls["n"] == 1, f"expected exactly 1 worker call, got {calls['n']}"
        assert not r.ok and r.error_kind == "provider"

    @pytest.mark.asyncio
    async def test_bare_provider_error_surfaces_as_single_failure(self):
        """A bare ProviderError (config/protocol mismatch) fails in one call --
        as does every provider fault now that the worker does not retry. The
        retryable/non-retryable distinction is moot in the worker.
        """
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            raise ProviderError("unsupported protocol", provider="openai", model="m")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(), "cheap-worker", [])
        assert calls["n"] == 1, "worker must make exactly 1 call"
        assert not r.ok and r.error_kind == "provider" and "unsupported" in r.error

    @pytest.mark.asyncio
    async def test_409_is_not_classified_transient(self):
        """HTTP 409 Conflict is a semantic/state error -- and with no internal
        retry loop, it (like every provider fault) fails in one call. The worker
        does not waste a retry budget on an idempotent fail because it has no
        retry budget.
        """
        calls = {"n": 0}

        async def fn(req, prof, c, k):
            calls["n"] += 1
            raise HTTPProviderError(
                "Resource conflict", status_code=409, provider="openai", model="m"
            )

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(), "cheap-worker", [])
        assert calls["n"] == 1, "worker must make exactly 1 call"
        assert not r.ok and r.error_kind == "configuration"


# Mid-cell-crash durability (resume seam probe)


class TestMidCellCrashDurability:
    """Phase-1 /diagnosing-bugs probe: when a cell succeeds a tool round then
    crashes on a later round, does an orchestrator retry (a fresh worker.run)
    re-execute the already-succeeded tool round, or resume from it?

    On a mid-cell crash after a succeeded tool round, an orchestrator retry
    (a fresh worker.run) resumes from that round rather than re-executing it:
    CELL_COMPLETED is emitted only on success and the carried conversation
    is replayed as the worker's starting context. This probe drives that real path and
    asserts the OPPOSITE (resume, don't re-run). If it goes red, the gap is
    confirmed and a checkpoint seam is the fix; if green, the gap is falsified
    and this becomes a regression test only.
    """

    @pytest.mark.asyncio
    async def test_succeeded_tool_round_not_rerun_on_retry(self):
        """The worker makes ONE call per turn and does not retry internally, so
        a mid-cell crash (503 on call 2) escapes immediately as ``ok=False``.
        The orchestrator's BACKOFF_RETRY is simulated here by a second
        ``worker.run`` resumed from the first attempt's messages
        (``resume_from_messages``). The claim under test: the round-1 read
        that already succeeded in run #1 must NOT re-execute in run #2 -- its
        result is carried by the resumed conversation, not re-issued. This
        pins the resume seam.
        """
        read_calls = {"n": 0}

        cfg = _make_config()
        gw = ModelGateway(cfg)

        # Per-run model-call index. Run #1: call 1 = tool round (read
        # succeeds); call 2 = transient 503 mid-cell. With no internal retry,
        # the worker escapes run #1 on that first 503. Run #2 (resumed from
        # r1.messages): the model already has the read result in context, so a
        # well-behaved model returns the final answer directly without
        # re-issuing the tool call.
        run_idx = {"run": 0}

        async def fn(req, prof, c, k):
            run_idx["run"] += 1
            if run_idx["run"] == 1:
                return _tool_response("c1", "read", {"path": "data.txt"})
            if run_idx["run"] == 2:
                raise HTTPProviderError(
                    "upstream 503 mid-cell", status_code=503, provider="openai", model="m"
                )
            # run #2 (resumed): model has the prior read result in context;
            # returns the final answer without re-issuing the tool call.
            return _ok_response("done after retry")

        _wire_adapter(gw, fn)

        with tempfile.TemporaryDirectory() as td:
            ws = RestrictedWorkspace(td)
            # Seed a file the read tool can return.
            (Path(td) / "data.txt").write_text("payload", encoding="utf-8")
            # Count every workspace.read invocation across both runs.
            real_read = ws.read

            def counting_read(path="."):
                read_calls["n"] += 1
                return real_read(path)

            ws.read = counting_read  # type: ignore[assignment]

            w = NativeWorker(gw, ws)
            cell = _make_cell(
                allowed_tools=["read"],
                max_calls=0,
                max_tool_rounds=0,
                max_retries=1,
            )
            read_tool = ToolDefinition(
                name="read",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            )

            r1 = await w.run(cell, "cheap-worker", [read_tool])
            # Run #1 failed mid-cell with a provider error (transient). The
            # worker made exactly one failing call (call 2) and escaped -- no
            # internal retry.
            assert not r1.ok and r1.error_kind == "provider", (
                f"run #1 should fail mid-cell with provider error, got {r1.error_kind}"
            )
            # Orchestrator BACKOFF_RETRY simulated by resuming from r1's
            # conversation -- the seam this test pins.
            r2 = await w.run(
                cell,
                "cheap-worker",
                [read_tool],
                resume_from_messages=r1.messages,
            )

        assert r2.ok and r2.text == "done after retry"
        # CLAIM UNDER TEST: the round-1 read that already succeeded in run #1
        # must NOT re-execute in run #2 -- its result should resume, not rerun.
        # Predicted RED today: count == 2 (each run calls read fresh). When the
        # checkpoint seam lands, this flips to 1.
        assert read_calls["n"] == 1, (
            "succeeded tool round re-ran on retry: workspace.read was called "
            f"{read_calls['n']} time(s) across two worker.run calls. A mid-cell "
            "crash should resume from the succeeded round, not re-execute it."
        )


# Lease management


class TestLease:
    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _tool_response("t1", "write", {"path": "o.txt", "content": "d"})
            return _ok_response("done")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            ws = RestrictedWorkspace(td)
            ac = rc = False
            orig_a, orig_r = ws.acquire_lease, ws.release_lease

            def ta(o):
                nonlocal ac
                ac = True
                return orig_a(o)

            def tr(t):
                nonlocal rc
                rc = True
                return orig_r(t)

            ws.acquire_lease, ws.release_lease = ta, tr
            w = NativeWorker(gw, ws)
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="write",
                        description="W",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    )
                ],
            )
            assert ac and rc and r.ok

    @pytest.mark.asyncio
    async def test_contention_error(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _tool_response("t1", "write", {"path": "f.txt", "content": "x"})
            return _ok_response("ok")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            ws = RestrictedWorkspace(td)
            ws.acquire_lease = lambda o: None
            w = NativeWorker(gw, ws)
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="write",
                        description="W",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    )
                ],
            )
            assert len(r.tool_results) > 0 and r.tool_results[0].is_error

    @pytest.mark.asyncio
    async def test_mutating_flag_drives_lease(self):
        """Lease acquisition reads ``Tool.mutating`` from the registry, not a
        hardcoded name tuple. ``cmd_lint`` is mutating-by-flag (a subprocess can
        mutate the workspace) but is NOT in the former
        ``("write", "create", "replace", "patch")`` tuple -- so the old name
        check would run the mutating subprocess WITHOUT a lease. The flag is the
        single authority, so a mutating-by-flag tool acquires the lease; a
        non-mutating-by-flag tool (``read``) never does."""
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _tool_response("t1", "cmd_lint", {})
            return _ok_response("ok")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        cmd_profiles = {"lint": CommandProfile(name="lint", argv=["ruff", "check", "."])}
        with tempfile.TemporaryDirectory() as td:
            ws = RestrictedWorkspace(td)
            acquired: list[str] = []
            orig_a, orig_r = ws.acquire_lease, ws.release_lease

            def ta(owner):
                acquired.append(owner)
                return orig_a(owner)

            def tr(t):
                return orig_r(t)

            ws.acquire_lease, ws.release_lease = ta, tr
            w = NativeWorker(gw, ws, command_profiles=cmd_profiles)
            await w.run(
                _make_cell(allowed_tools=["cmd_lint"]),
                "cheap-worker",
                ToolRegistry.with_defaults(command_profiles=cmd_profiles).advertised(),
            )
            # cmd_lint is mutating-by-flag -> the worker must acquire a lease.
            assert acquired, "mutating-by-flag tool did not acquire a lease"
            assert ws._lease is None, "lease not released after run"

    @pytest.mark.asyncio
    async def test_read_only_tool_does_not_acquire_lease(self):
        """A non-mutating-by-flag tool (``read``) must never acquire the
        mutation lease -- the flag is authoritative, so read-only dispatch skips
        the lease path entirely."""
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _tool_response("t1", "read", {"path": "."})
            return _ok_response("ok")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            ws = RestrictedWorkspace(td)
            acquired: list[str] = []
            orig_a, orig_r = ws.acquire_lease, ws.release_lease

            def ta(owner):
                acquired.append(owner)
                return orig_a(owner)

            def tr(t):
                return orig_r(t)

            ws.acquire_lease, ws.release_lease = ta, tr
            w = NativeWorker(gw, ws)
            await w.run(
                _make_cell(),
                "cheap-worker",
                ToolRegistry.with_defaults(command_profiles={}).advertised(),
            )
            assert not acquired, f"read-only tool acquired a lease: {acquired}"

    @pytest.mark.asyncio
    async def test_released_on_exit(self):
        async def fn(req, prof, c, k):
            return _tool_response("t1", "write", {"path": "f.txt", "content": "x"})

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            ws = RestrictedWorkspace(td)
            w = NativeWorker(gw, ws)
            await w.run(
                _make_cell(max_calls=1),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="write",
                        description="W",
                        parameters={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    )
                ],
            )
            assert ws._lease is None


# Tool validation


class TestToolValidation:
    @pytest.mark.asyncio
    async def test_invalid_args(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _tool_response("t1", "read", {})
            return _ok_response("ok")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    )
                ],
            )
            assert r.ok and r.tool_calls_made == 1

    @pytest.mark.asyncio
    async def test_trusts_advertised_tool_defs(self):
        """The worker sends ``tool_defs`` verbatim: the orchestrator already
        applied ``contract.allowed_tools`` at advertisement time (the single
        enforcement point), so no worker-side re-filtering happens."""
        sent = []

        async def fn(req, prof, c, k):
            sent.extend(req.tools)
            return _ok_response()

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            await w.run(
                _make_cell(allowed_tools=["read"]),
                "cheap-worker",
                [
                    ToolDefinition(name="read", description="R"),
                    ToolDefinition(name="write", description="W"),
                ],
            )
            names = [t.name for t in sent]
            assert names == ["read", "write"]


# Schema repair


class TestSchemaRepair:
    @pytest.mark.asyncio
    async def test_repair_success(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _ok_response('{"name":"t"}')
            return _ok_response('{"id":"1","name":"t"}')

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [],
                output_json_schema={
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                    "required": ["id", "name"],
                },
            )
            assert r.ok and r.parsed_output["id"] == "1"

    @pytest.mark.asyncio
    async def test_repair_failure(self):
        async def fn(req, prof, c, k):
            return _ok_response("not json")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(),
                "cheap-worker",
                [],
                output_json_schema={"type": "object", "required": ["x"]},
            )
            assert not r.ok and r.error_kind == "schema"

    @pytest.mark.asyncio
    async def test_repair_budget_exhausted(self):
        async def fn(req, prof, c, k):
            return _ok_response("bad")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(max_calls=1),
                "cheap-worker",
                [],
                output_json_schema={"type": "object", "required": ["x"]},
            )
            assert not r.ok


# Command profiles


class TestCommandProfiles:
    @pytest.mark.asyncio
    async def test_exposed_as_tool(self):
        sent = []

        async def fn(req, prof, c, k):
            sent.extend(req.tools)
            return _ok_response()

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        cmd_profiles = {"lint": CommandProfile(name="lint", argv=["ruff", "check", "."])}
        # The orchestrator advertises the registry's ToolDefinition list (Step
        # 5); the worker no longer rebuilds cmd_* itself. So this test -- which
        # stands in for the orchestrator's advertisement -- passes exactly the
        # advertised set, mirroring production.
        advertised = ToolRegistry.with_defaults(command_profiles=cmd_profiles).advertised()
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(
                gw,
                RestrictedWorkspace(td),
                command_profiles=cmd_profiles,
            )
            await w.run(_make_cell(), "cheap-worker", advertised)
            tool_names = [t.name for t in sent]
            assert "cmd_lint" in tool_names
            # Regression: cmd_* tools must not be duplicated. The live CLI used to
            # 400 because BOTH orchestration and the worker built cmd_* tools,
            # producing duplicate tool names the upstream rejected. Now the
            # single registry advertises each tool once (orchestration advertises,
            # the worker does not rebuild), so there can be exactly one.
            assert tool_names.count("cmd_lint") == 1, tool_names
            assert len(tool_names) == len(set(tool_names)), tool_names

    @pytest.mark.asyncio
    async def test_execution(self):
        n = 0

        async def fn(req, prof, c, k):
            nonlocal n
            n += 1
            if n == 1:
                return _tool_response("t1", "cmd_lint", {})
            return _ok_response("ok")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(
                gw,
                RestrictedWorkspace(td),
                command_profiles={"lint": CommandProfile(name="lint", argv=["ruff", "check", "."])},
            )
            r = await w.run(_make_cell(allowed_tools=["cmd_lint"]), "cheap-worker", [])
            assert r.ok and r.tool_calls_made == 1


# Regression: contract.id != profile.name


class TestProfileNameRegression:
    @pytest.mark.asyncio
    async def test_contract_id_differs_from_profile_name(self):
        async def fn(req, prof, c, k):
            assert req.profile_name == "cheap-worker"
            return _ok_response("ok")

        cfg = _make_config()
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        cell = GraphCell(
            id="random-id",
            name="F",
            contract=WorkContract(
                id="contract-abc",
                objective="Do it",
                budget=Budget(token_limit=10000, cost_limit_usd=1.0),
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(cell, "cheap-worker", [])
            assert r.ok


# Cost enforcement


class TestCostEnforcement:
    @pytest.mark.asyncio
    async def test_cost_budget_exceeded(self):
        # The worker no longer enforces cost/token limits -- it is the single
        # owner of *spend reporting*, not of spend *authority*. A call whose
        # cost blows past the cell budget must still return ok=True, carrying
        # the actual spend; the orchestrator's BudgetAuthority
        # abstains at the wave boundary (covered by
        # test_budget_exceeded_abstains_and_marks_exhausted). Asserting the
        # worker used to abort here would pin the removed enforcement the
        # design centralised upward.
        async def fn(req, prof, c, k):
            return _ok_response(usage=Usage(input_tokens=100000, output_tokens=50000))

        profile = _make_profile(input_price=10.0, output_price=30.0)  # $2.50/call
        cfg = _make_config([profile])
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(budget_cost=1.0, budget_tokens=1_000_000), "cheap-worker", []
            )
            # The worker succeeds and reports the over-budget spend; it does
            # not gate on cost. Enforcement is the orchestrator's job.
            assert r.ok, f"worker must not enforce cost: {r.error}"
            assert r.total_cost_usd == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_free_tier_succeeds(self):
        async def fn(req, prof, c, k):
            return _ok_response(usage=Usage(input_tokens=1000, output_tokens=500))

        profile = _make_profile(input_price=0.0, output_price=0.0)
        cfg = _make_config([profile])
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(_make_cell(budget_cost=0.001), "cheap-worker", [])
            assert r.ok and r.total_cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_preflight_does_not_double_count_historical_input_cost(self):
        # The worker's preflight cost guard is gone entirely (spend authority
        # now lives in the orchestrator's BudgetAuthority). This test keeps the
        # tool-round setup that used to exercise the preflight, now asserting
        # the new contract: the worker completes both calls and reports the
        # *true* cumulative spend -- it never aborts on cost, so
        # there is no preflight to double-count historical input tokens. The
        # regression the old test guarded (re-pricing 100k historical input
        # tokens a second time and tripping early) is now structurally
        # impossible: there is no preflight to trip.
        call_n = 0

        async def fn(req, prof, client, key):
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                return _tool_response(
                    "tc1",
                    "read",
                    {"path": "f.txt"},
                    usage=Usage(input_tokens=100_000, output_tokens=0),
                )
            return _ok_response("done", usage=Usage(input_tokens=10, output_tokens=0))

        profile = _make_profile(input_price=10.0, output_price=0.0)  # $1.0 per 100k input tokens
        cfg = _make_config([profile])
        gw = ModelGateway(cfg)
        _wire_adapter(gw, fn)
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "f.txt").write_text("hello")
            w = NativeWorker(gw, RestrictedWorkspace(td))
            r = await w.run(
                _make_cell(budget_cost=1.5, budget_tokens=1_000_000, allowed_tools=["read"]),
                "cheap-worker",
                [
                    ToolDefinition(
                        name="read",
                        description="R",
                        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                    )
                ],
            )
            # No preflight remains: the worker runs to completion and reports
            # the true spend ($1.0 first call + $0.0001 second = $1.0001),
            # never aborting on cost.
            assert r.ok, f"worker must not abort on cost: {r.error}"
            assert r.text == "done"
            assert r.tool_calls_made == 1
            assert r.total_cost_usd == pytest.approx(1.0001)


# Gateway limiting


class TestGatewayLimiting:
    @pytest.mark.asyncio
    async def test_shared_limiter(self):
        cfg = _make_config()
        gw = ModelGateway(cfg)
        assert gw._get_limiter("cheap-worker") is gw._get_limiter("cheap-worker")

    @pytest.mark.asyncio
    async def test_singleton_no_overlap(self):
        from northstack.adapters.providers.gateway import ModelLimiter

        lim = ModelLimiter("s", max_concurrency=1, rpm=1000)
        mx, cur, lk = 0, 0, asyncio.Lock()

        async def t():
            nonlocal mx, cur
            async with lk:
                cur += 1
                mx = max(mx, cur)
            await asyncio.sleep(0.02)
            async with lk:
                cur -= 1

        await asyncio.gather(*[lim.run(t) for _ in range(5)])
        assert mx == 1


# WorkerResult


class TestWorkerResult:
    def test_frozen(self):
        r = WorkerResult(ok=True, text="h")
        with pytest.raises(Exception):
            r.text = "x"  # type: ignore[misc]

    def test_fields(self):
        r = WorkerResult(
            ok=True,
            text="d",
            tool_calls_made=3,
            tool_rounds=2,
            total_input_tokens=100,
            total_output_tokens=50,
        )
        assert r.tool_calls_made == 3 and r.tool_rounds == 2

    def test_error_kind(self):
        assert WorkerResult(ok=False, error="e", error_kind="budget").error_kind == "budget"

    def test_parsed_output(self):
        assert WorkerResult(ok=True, text="{}", parsed_output={"x": 1}).parsed_output == {"x": 1}


class TestCellPromptCarriesItsCriteria:
    """A cell is graded on criteria it must therefore be shown."""

    @staticmethod
    def _cell(indices):
        from northstack.domain import (
            Budget,
            CommandCriterion,
            FileDiffCriterion,
            WorkContract,
        )
        from northstack.domain.graph import GraphCell

        contract = WorkContract(
            id="wc-1",
            version=1,
            objective="ship topwords",
            deliverables=["topwords.py"],
            budget=Budget(token_limit=100, cost_limit_usd=0.0),
            acceptance_criteria=[
                FileDiffCriterion(
                    description="tests package exists",
                    path="tests/__init__.py",
                    must_exist=True,
                ),
                CommandCriterion(description="suite passes", command_name="test", exit_code=0),
            ],
        )
        return GraphCell(
            id="c0",
            name="c0",
            mode="mutating",
            contract=contract,
            acceptance_criterion_indices=indices,
        )

    def _prompt(self, cell):
        return NativeWorker._build_user_prompt(None, cell)  # type: ignore[arg-type]

    def test_assigned_criteria_reach_the_prompt(self):
        prompt = self._prompt(self._cell([0, 1]))
        assert "tests/__init__.py" in prompt
        assert "tests package exists" in prompt
        assert "suite passes" in prompt

    def test_only_the_cells_own_criteria_are_shown(self):
        prompt = self._prompt(self._cell([1]))
        assert "suite passes" in prompt
        assert "tests/__init__.py" not in prompt

    def test_a_cell_with_no_criteria_gets_no_acceptance_block(self):
        assert "accepted only if" not in self._prompt(self._cell([]))

    def test_an_out_of_range_index_is_skipped(self):
        assert "accepted only if" not in self._prompt(self._cell([7]))
