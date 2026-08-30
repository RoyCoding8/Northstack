"""NativeWorker: bounded model/tool loop with structured schemas.

Public seam:
  - NativeWorker.run(cell, profile_name, tools, workspace, ...) -> WorkerResult
  - WorkerResult / WorkerError -> typed outcomes

The worker performs a bounded model/tool loop using only
RestrictedWorkspace/WebReader mediated tools and structured schemas.
It never assumes exact provider identity -- capability flags govern behavior.

The worker calls gateway.complete(request) directly; the gateway applies
per-profile concurrency and RPM limiting internally.  The worker must NOT
import or use ModelLimiter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

import jsonschema
from pydantic import BaseModel, ConfigDict, Field

from northstack.adapters.providers.gateway import (
    AuthProviderError,
    CapabilityError,
    ProviderConfigurationError,
    ProviderError,
)
from northstack.adapters.providers.pricing import compute_cost_usd
from northstack.adapters.providers.wire import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.adapters.workspace.webfetch import WebReader
from northstack.application.json_extraction import strip_code_fence
from northstack.application.tools.registry import ToolRegistry
from northstack.domain.graph import GraphCell
from northstack.ports.protocols import GatewayPort

T = TypeVar("T")


class _WallTimeExceeded(Exception):
    pass


async def _within_wall_time(operation: Awaitable[T], remaining: float | None) -> T:
    timeout = asyncio.timeout(remaining)
    try:
        async with timeout:
            return await operation
    except TimeoutError:
        if timeout.expired():
            raise _WallTimeExceeded from None
        raise


def _provider_error_kind(error: ProviderError) -> str:
    return (
        "authentication"
        if isinstance(error, AuthProviderError)
        else "capability"
        if isinstance(error, CapabilityError)
        else "configuration"
        if isinstance(error, ProviderConfigurationError)
        else "provider"
        if error.is_transient()
        else "configuration"
    )


def _estimate_request_input_tokens(request: ModelRequest) -> int:
    payload = request.model_dump(
        mode="json",
        exclude={"profile_name", "max_output_tokens"},
    )
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    return max(1, (size + 3) // 4)


_CONTEXT_TRIGGER_RATIO = 0.78
_CONTEXT_TARGET_RATIO = 0.62
_MIN_RETAINED_SPANS = 3
_ELISION_SENTINEL = "\n\n[context management:"


def _render_criteria(cell: GraphCell) -> str:
    """The cell's own acceptance criteria, one checkable line each."""
    criteria = cell.contract.acceptance_criteria
    lines = []
    for i in cell.acceptance_criterion_indices:
        if not 0 <= i < len(criteria):
            continue
        c = criteria[i]
        detail = {
            k: v
            for k, v in c.model_dump(exclude_none=True).items()
            if k not in ("kind", "description")
        }
        kind = getattr(c.kind, "value", c.kind)
        lines.append(f"- [{kind}] {c.description}" + (f" -- {detail}" if detail else ""))
    return "\n".join(lines)


_MAX_CONSECUTIVE_TRUNCATIONS = 3
_TRUNCATION_NUDGES = (
    "Your previous response was truncated before you finished (the output token "
    "limit was reached). Continue and give the complete final answer now, more "
    "concisely if needed.",
    "Truncated again. Stop explaining and act: reply with a single tool call, or "
    "a final answer under 200 words. No preamble, no restating the plan.",
)
_MAX_PARALLEL_TOOL_CALLS = 6


def _evictable_spans(messages: list[ModelMessage]) -> tuple[int, list[tuple[int, int]]]:
    """Length of the pinned prefix, then the ``(start, end)`` spans droppable whole.

    A span is one turn plus the tool results answering it. Dropping a span
    whole is what keeps every tool result attached to the call that produced
    it -- an orphaned result is rejected on the wire.
    """
    pinned = 0
    while pinned < len(messages) and messages[pinned].role == MessageRole.SYSTEM:
        pinned += 1
    if pinned < len(messages) and messages[pinned].role == MessageRole.USER:
        pinned += 1

    spans: list[tuple[int, int]] = []
    start = pinned
    while start < len(messages):
        end = start + 1
        while end < len(messages) and messages[end].role == MessageRole.TOOL:
            end += 1
        spans.append((start, end))
        start = end
    return pinned, spans


def _note_elision(messages: list[ModelMessage], elided: int) -> None:
    """Record the elision on the pinned objective rather than as its own turn.

    A separate marker message would sit next to the objective as a second
    consecutive user turn, which some providers reject.
    """
    pinned, _ = _evictable_spans(messages)
    if pinned == 0 or messages[pinned - 1].role != MessageRole.USER:
        return
    objective = messages[pinned - 1]
    base = objective.content.split(_ELISION_SENTINEL)[0]
    messages[pinned - 1] = objective.model_copy(
        update={
            "content": (
                f"{base}{_ELISION_SENTINEL} {elided} earlier tool round(s) were dropped "
                "to fit the model's context window. Re-read any file you need rather "
                "than relying on memory of it.]"
            )
        }
    )


def _compact_messages(
    messages: list[ModelMessage],
    estimate: Callable[[list[ModelMessage]], int],
    window_tokens: int,
) -> int:
    """Drop the oldest whole rounds until the request fits the profile's window.

    Returns how many rounds were dropped. Eviction is deterministic rather
    than a summarising model call: it costs nothing, adds no latency, and
    cannot itself fail mid-cell.
    """
    if estimate(messages) <= window_tokens * _CONTEXT_TRIGGER_RATIO:
        return 0

    target = window_tokens * _CONTEXT_TARGET_RATIO
    dropped = 0
    while estimate(messages) > target:
        _, spans = _evictable_spans(messages)
        if len(spans) <= _MIN_RETAINED_SPANS:
            break
        start, end = spans[0]
        del messages[start:end]
        dropped += 1
    return dropped


logger = logging.getLogger(__name__)


_MAX_SCHEMA_VALIDATION_BYTES = 256 * 1024


class WorkerEventKind(str, Enum):
    """The observable moments inside one cell's model/tool loop."""

    TURN_STARTED = "turn_started"
    CONTEXT_COMPACTED = "context_compacted"
    MODEL_CALL_COMPLETED = "model_call_completed"
    MODEL_CALL_FAILED = "model_call_failed"
    TOOL_CALL_COMPLETED = "tool_call_completed"


class WorkerEvent(BaseModel):
    """One moment in the loop, published to whoever is watching.

    A single sink rather than a callback per lifecycle point: the ledger
    projection, tracing, and any future observer all consume this one stream,
    so adding a moment never means widening a protocol.
    """

    model_config = ConfigDict(frozen=True)

    kind: WorkerEventKind
    turn: int = Field(ge=0, description="1-based model turn; 0 before the first")
    detail: dict[str, Any] = Field(default_factory=dict)


WorkerEventSink = Callable[[WorkerEvent], Awaitable[None]]


class WorkerResult(BaseModel):
    """Structured result from NativeWorker.run().

    Covers success, budget exhaustion, provider errors, schema failures,
    and tool execution errors.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    ok: bool = Field(description="Whether the worker completed successfully")
    text: str = Field(default="", description="Final text output from the model")
    tool_calls_made: int = Field(default=0, ge=0, description="Total tool calls executed")
    tool_rounds: int = Field(default=0, ge=0, description="Total model/tool round trips")
    api_calls: int = Field(default=0, ge=0, description="Total gateway API calls attempted")
    successful_api_calls: int = Field(
        default=0, ge=0, description="Gateway calls that returned a response"
    )
    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    wall_time_ms: int = Field(default=0, ge=0)
    error: str = Field(default="", description="Error message if not ok")
    error_kind: str = Field(
        default="",
        description="Error category: budget, provider, schema, tool, or empty",
    )
    parsed_output: dict[str, Any] | None = Field(
        default=None,
        description="Parsed JSON output when output_json_schema is used",
    )
    tool_results: list[ToolResultMessage] = Field(
        default_factory=list,
        description="All tool results produced during the run",
    )
    messages: list[ModelMessage] = Field(
        default_factory=list,
        description="Full conversation history after completion",
    )


class AttemptAccounting:
    """Accumulate one run's tallies and emit a single :class:`WorkerResult`.

    The worker's loop mutates the running counters (``record_call`` /
    ``record_round``) and then calls ``.success(...)`` or ``.failure(...)`` to
    materialise a :class:`WorkerResult`. Both append the final assistant
    message (if any) to ``messages`` before building the result, so the
    returned conversation history is complete. ``wall_time_ms`` is read at
    result-build time from the ``start_time`` captured on construction.
    """

    def __init__(
        self,
        start_time: float,
        messages: list[ModelMessage],
        tool_results: list[ToolResultMessage],
        on_progress: Callable[[], None] | None = None,
    ) -> None:
        self._start_time = start_time
        self._messages = messages
        self._tool_results = tool_results
        self._on_progress = on_progress or (lambda: None)
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost = 0.0
        self._rounds = 0
        self._api_calls = 0
        self._successful_api_calls = 0

    def on_delta_beat(self, _delta: object) -> None:
        """Progress beat per streamed delta.

        A stream can run for minutes before its final usage lands; without
        these beats the stall detector would read the quiet stretch as a
        pinned cell.  The delta itself is deliberately discarded -- raw model
        output never enters accounting or the ledger.
        """
        self._on_progress()

    def record_round(self) -> None:
        """One model/tool round trip completed."""
        self._rounds += 1
        self._on_progress()

    def record_call(self, input_tokens: int, output_tokens: int, cost: float) -> None:
        """Account one gateway response's spend."""
        self._successful_api_calls += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._cost += cost
        self._on_progress()

    @property
    def api_calls(self) -> int:
        return self._api_calls

    def record_attempt(self) -> None:
        self._api_calls += 1
        self._on_progress()

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def cost(self) -> float:
        return self._cost

    def _wall_ms(self) -> int:
        return int((time.perf_counter() - self._start_time) * 1000)

    def _with_final(self, text: str | None, *, append: bool) -> list[ModelMessage]:
        """Return the message history, optionally appending the final answer."""
        if not append or text is None or text == "":
            return self._messages
        return self._messages + [ModelMessage(role=MessageRole.ASSISTANT, content=text)]

    def success(
        self,
        text: str,
        *,
        parsed_output: dict[str, Any] | None = None,
        append_final: bool = True,
    ) -> WorkerResult:
        return WorkerResult(
            ok=True,
            text=text,
            parsed_output=parsed_output,
            tool_calls_made=len(self._tool_results),
            tool_rounds=self._rounds,
            api_calls=self._api_calls,
            successful_api_calls=self._successful_api_calls,
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
            total_cost_usd=self._cost,
            wall_time_ms=self._wall_ms(),
            tool_results=self._tool_results,
            messages=self._with_final(text, append=append_final),
        )

    def failure(
        self,
        kind: str,
        detail: str,
        *,
        text: str = "",
        append_final: bool = False,
    ) -> WorkerResult:
        return WorkerResult(
            ok=False,
            error=detail,
            error_kind=kind,
            text=text,
            tool_calls_made=len(self._tool_results),
            tool_rounds=self._rounds,
            api_calls=self._api_calls,
            successful_api_calls=self._successful_api_calls,
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
            total_cost_usd=self._cost,
            wall_time_ms=self._wall_ms(),
            tool_results=self._tool_results,
            messages=self._with_final(text, append=append_final),
        )


class WorkerError(ProviderError):
    """Error during worker execution."""


def _validate_tool_args(
    tool_call: ToolCall,
    tool_defs: dict[str, ToolDefinition],
) -> str | None:
    """Validate tool call arguments against the definition.

    Returns an error message if invalid, None if valid.
    """
    tool_def = tool_defs.get(tool_call.name)
    if tool_def is None:
        return f"Unknown tool: {tool_call.name}"

    params = tool_def.parameters
    required = params.get("required", [])
    props = params.get("properties", {})

    for req_field in required:
        if req_field not in tool_call.arguments:
            return f"Missing required argument '{req_field}' for tool '{tool_call.name}'"

    for arg_name, arg_value in tool_call.arguments.items():
        if arg_name in props:
            expected_type = props[arg_name].get("type")
            if expected_type == "string" and not isinstance(arg_value, str):
                return f"Argument '{arg_name}' for tool '{tool_call.name}' must be string"
            elif expected_type == "integer" and not isinstance(arg_value, int):
                return f"Argument '{arg_name}' for tool '{tool_call.name}' must be integer"
            elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                return f"Argument '{arg_name}' for tool '{tool_call.name}' must be number"
            elif expected_type == "boolean" and not isinstance(arg_value, bool):
                return f"Argument '{arg_name}' for tool '{tool_call.name}' must be boolean"

    return None


def _tool_batches(calls: list[ToolCall], registry: ToolRegistry) -> list[list[ToolCall]]:
    """Split one round's tool calls into groups that may run concurrently.

    A model that asks to read six files should wait for the slowest, not the
    sum.  Only the reads are independent though: the workspace mutation lease
    is a single exclusive lock, and a read issued after a write must observe
    that write.  So a mutating call runs alone and in place, and consecutive
    read-only calls between two mutations batch together -- which preserves the
    sequential ordering the model asked for wherever that ordering can matter.

    The cap keeps one greedy round from opening an unbounded number of
    subprocesses and file handles at once.
    """
    batches: list[list[ToolCall]] = []
    previous_was_mutating = True
    for call in calls:
        tool = registry.get(call.name)
        mutating = tool is not None and tool.mutating
        if mutating or previous_was_mutating or len(batches[-1]) >= _MAX_PARALLEL_TOOL_CALLS:
            batches.append([call])
        else:
            batches[-1].append(call)
        previous_was_mutating = mutating
    return batches


def _build_tool_result_message(
    tool_call: ToolCall,
    result: str,
    is_error: bool = False,
) -> ToolResultMessage:
    """Build a tool result message from a tool call and result."""
    return ToolResultMessage(
        tool_call_id=tool_call.id,
        content=result,
        is_error=is_error,
    )


def _validate_with_jsonschema(
    data: Any,
    schema: dict[str, Any],
) -> str | None:
    """Validate data against a JSON schema using jsonschema library.

    Returns error message or None if valid.
    Bounds schema and input sizes for safety.
    """
    schema_bytes = len(json.dumps(schema).encode())
    if schema_bytes > _MAX_SCHEMA_VALIDATION_BYTES:
        return f"Schema too large ({schema_bytes} bytes, max {_MAX_SCHEMA_VALIDATION_BYTES})"

    data_bytes = len(json.dumps(data).encode())
    if data_bytes > _MAX_SCHEMA_VALIDATION_BYTES:
        return f"Input data too large ({data_bytes} bytes, max {_MAX_SCHEMA_VALIDATION_BYTES})"

    try:
        jsonschema.validate(
            instance=data,
            schema=schema,
            format_checker=jsonschema.FormatChecker(),
        )
    except jsonschema.ValidationError as e:
        path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "(root)"
        return f"Validation error at {path}: {e.message}"
    except jsonschema.SchemaError as e:
        return f"Invalid schema: {e.message}"

    return None


def _try_parse_json_response(
    text: str,
    schema: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Try to parse text as JSON and validate against schema.

    Returns (parsed_dict, error_or_none).
    """
    if not text.strip():
        return None, "Empty response"

    json_text = strip_code_fence(text)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    if schema is not None:
        error = _validate_with_jsonschema(parsed, schema)
        if error:
            return None, error

    return parsed, None


class _WorkerToolContext:
    """The worker's concrete :class:`ToolContext` handed to each tool's
    ``execute``.

    A narrow view of the worker's state: the mediated workspace, the SSRF-
    protected web reader (``None`` when unset), the configured command
    profiles (resolved by ``cmd_*`` tools), and the active mutation lease.
    The registry's tools reach through this and never see the whole
    ``NativeWorker``.
    """

    __slots__ = ("command_profiles", "lease", "web_reader", "workspace")

    def __init__(
        self,
        *,
        workspace: RestrictedWorkspace,
        web_reader: WebReader | None,
        command_profiles: dict[str, CommandProfile],
        lease: str | None,
    ) -> None:
        self.workspace = workspace
        self.web_reader = web_reader
        self.command_profiles = command_profiles
        self.lease = lease


class NativeWorker:
    """Bounded model/tool loop using mediated tools.

    The worker:
      1. Sends the request to the model via gateway.complete()
      2. If the model requests tool calls, validates and executes them
      3. Appends tool results as messages and loops
      4. Stops on final text output, budget exhaustion, or error
      5. Manages mutation leases for workspace write tools
      6. Performs at most one schema repair attempt if output_json_schema is set

    Budget enforcement:
      - max_calls: total API calls
      - max_tool_rounds: total tool-call round trips
      - token_limit: cumulative input+output tokens
      - cost_limit_usd: cumulative cost
      - max_wall_time_seconds: wall-clock time

    The worker does NOT import or use ModelLimiter.  Gateway.complete()
    applies per-profile concurrency and RPM limiting internally.
    """

    def __init__(
        self,
        gateway: GatewayPort,
        workspace: RestrictedWorkspace,
        *,
        web_reader: WebReader | None = None,
        command_profiles: dict[str, CommandProfile] | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._gateway = gateway
        self._workspace = workspace
        self._web_reader = web_reader
        self._command_profiles = command_profiles or {}
        self._tool_registry = tool_registry or ToolRegistry.with_defaults(
            command_profiles=self._command_profiles
        )

    async def run(
        self,
        cell: GraphCell,
        profile_name: str,
        tool_defs: list[ToolDefinition],
        *,
        system_prompt: str = "",
        output_json_schema: dict[str, Any] | None = None,
        resume_from_messages: list[ModelMessage] | None = None,
        on_progress: Callable[[], None] | None = None,
        on_checkpoint: Callable[[list[ModelMessage]], None] | None = None,
        on_event: WorkerEventSink | None = None,
    ) -> WorkerResult:
        """Run a bounded model/tool loop for a cell.

        Args:
            cell: The cell containing the work contract and budget.
            profile_name: Which ModelProfile to use (router supplies this).
            tool_defs: Tool definitions to expose to the model.
            system_prompt: System prompt for the model.
            output_json_schema: Optional JSON-Schema for structured output.
            resume_from_messages: Prior conversation history from a crashed
                attempt. When set, the loop resumes from it instead of
                re-issuing the user prompt, so already-succeeded tool rounds
                are not re-executed on a BACKOFF_RETRY. Mid-cell crash
                durability.
            on_checkpoint: Invoked with the full conversation after each
                completed tool round so the caller can persist run-resume
                state to disk.
            on_event: Observer for the loop's :class:`WorkerEvent` stream. A
                sink that raises is logged and ignored -- observation never
                fails the cell.

        Returns:
            WorkerResult with the outcome.
        """
        start_time = time.perf_counter()
        budget = cell.contract.budget

        def remaining_wall() -> float | None:
            return (
                None
                if budget.max_wall_time_seconds <= 0
                else max(0.0, budget.max_wall_time_seconds - (time.perf_counter() - start_time))
            )

        messages: list[ModelMessage] = []
        tool_results_all: list[ToolResultMessage] = []
        acct = AttemptAccounting(start_time, messages, tool_results_all, on_progress)

        try:
            profile = self._gateway.profile(profile_name)
        except ProviderError as e:
            return acct.failure(_provider_error_kind(e), str(e))

        tool_map = {t.name: t for t in tool_defs}

        mutation_lease: str | None = None
        schema_repair_attempted = False

        try:
            if resume_from_messages:
                messages.extend(resume_from_messages)
            else:
                if system_prompt:
                    messages.append(
                        ModelMessage(
                            role=MessageRole.SYSTEM,
                            content=system_prompt,
                        )
                    )
                messages.append(
                    ModelMessage(
                        role=MessageRole.USER,
                        content=self._build_user_prompt(cell),
                    )
                )

            def _request_for(msgs: list[ModelMessage]) -> ModelRequest:
                return ModelRequest(
                    profile_name=profile_name,
                    messages=msgs,
                    system="" if any(m.role == MessageRole.SYSTEM for m in msgs) else system_prompt,
                    tools=list(tool_defs),
                    output_json_schema=output_json_schema,
                    max_output_tokens=1,
                )

            total_elided = 0
            truncated_turns = 0
            turn = 0
            durations: dict[str, float] = {}

            async def emit(kind: WorkerEventKind, **detail: Any) -> None:
                if on_event is None:
                    return
                try:
                    await on_event(WorkerEvent(kind=kind, turn=turn, detail=detail))
                except Exception:
                    logger.warning("worker event sink failed on %s", kind.value, exc_info=True)

            while True:
                turn += 1
                if budget.max_wall_time_seconds > 0:
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= budget.max_wall_time_seconds:
                        return acct.failure("budget", "Wall time limit exceeded")

                if budget.max_calls > 0 and acct.api_calls >= budget.max_calls:
                    return acct.failure("budget", "API call limit exceeded")

                if budget.max_tool_rounds > 0 and acct.rounds >= budget.max_tool_rounds:
                    return acct.failure("budget", "Tool round limit exceeded")

                elided = _compact_messages(
                    messages,
                    lambda msgs: _estimate_request_input_tokens(_request_for(msgs)),
                    profile.context_window_tokens,
                )
                if elided:
                    total_elided += elided
                    _note_elision(messages, total_elided)
                    await emit(
                        WorkerEventKind.CONTEXT_COMPACTED,
                        rounds_dropped=elided,
                        rounds_dropped_total=total_elided,
                        window_tokens=profile.context_window_tokens,
                    )

                request = _request_for(messages)
                await emit(
                    WorkerEventKind.TURN_STARTED,
                    messages=len(messages),
                    input_tokens_estimate=_estimate_request_input_tokens(request),
                )
                available_output = profile.max_output_tokens
                if budget.token_limit is not None:
                    available_output = (
                        budget.token_limit
                        - acct.input_tokens
                        - acct.output_tokens
                        - _estimate_request_input_tokens(request)
                    )
                if available_output <= 0:
                    return acct.failure(
                        "budget", "Token budget cannot cover the next request input"
                    )
                request = request.model_copy(
                    update={"max_output_tokens": min(available_output, profile.max_output_tokens)}
                )

                try:
                    complete_stream = getattr(self._gateway, "complete_stream", None)
                    acct.record_attempt()
                    operation = (
                        complete_stream(request, on_delta=acct.on_delta_beat)
                        if complete_stream is not None
                        else self._gateway.complete(request)
                    )
                    response = await _within_wall_time(operation, remaining_wall())
                except _WallTimeExceeded:
                    await emit(WorkerEventKind.MODEL_CALL_FAILED, error_kind="budget")
                    return acct.failure("budget", "Wall time limit exceeded during provider call")
                except ProviderError as e:
                    kind = _provider_error_kind(e)
                    await emit(WorkerEventKind.MODEL_CALL_FAILED, error_kind=kind, error=str(e))
                    return acct.failure(kind, str(e))

                call_cost = compute_cost_usd(
                    response.usage,
                    profile.input_price_per_million_usd,
                    profile.output_price_per_million_usd,
                )
                acct.record_call(
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    call_cost,
                )
                await emit(
                    WorkerEventKind.MODEL_CALL_COMPLETED,
                    finish_reason=response.finish_reason.value,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    cache_read_tokens=response.usage.cache_read_tokens,
                    cache_creation_tokens=response.usage.cache_creation_tokens,
                    cost_usd=call_cost,
                    tool_calls=len(response.tool_calls),
                )

                if not response.tool_calls:
                    if output_json_schema is not None:
                        parsed, schema_error = _try_parse_json_response(
                            response.text,
                            output_json_schema,
                        )
                        if schema_error and not schema_repair_attempted:
                            if budget.max_calls > 0 and acct.api_calls >= budget.max_calls:
                                return acct.failure(
                                    "schema",
                                    f"Schema validation failed and no budget "
                                    f"for repair: {schema_error}",
                                    text=response.text,
                                    append_final=True,
                                )

                            schema_repair_attempted = True
                            messages.append(
                                ModelMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=response.text,
                                )
                            )
                            messages.append(
                                ModelMessage(
                                    role=MessageRole.USER,
                                    content=(
                                        f"Your response failed schema validation: {schema_error}\n"
                                        "Please fix and output valid JSON matching the schema."
                                    ),
                                )
                            )
                            continue  # retry loop for repair

                        if parsed is not None:
                            return acct.success(response.text, parsed_output=parsed)
                        else:
                            return acct.failure(
                                "schema",
                                f"Schema validation failed after repair: {schema_error}",
                                text=response.text,
                                append_final=True,
                            )

                    if response.finish_reason == FinishReason.MAX_TOKENS:
                        truncated_turns += 1
                        out_of_calls = budget.max_calls > 0 and acct.api_calls >= budget.max_calls
                        if out_of_calls or truncated_turns >= _MAX_CONSECUTIVE_TRUNCATIONS:
                            return acct.failure(
                                "provider",
                                (
                                    "Model response truncated (finish=max_tokens) with no "
                                    f"tool calls on {truncated_turns} consecutive turn(s); "
                                    f"output budget was {profile.max_output_tokens} tokens"
                                ),
                                text=response.text,
                                append_final=True,
                            )
                        if response.text:
                            messages.append(
                                ModelMessage(
                                    role=MessageRole.ASSISTANT,
                                    content=response.text,
                                )
                            )
                        messages.append(
                            ModelMessage(
                                role=MessageRole.USER,
                                content=_TRUNCATION_NUDGES[
                                    min(truncated_turns, len(_TRUNCATION_NUDGES)) - 1
                                ],
                            )
                        )
                        continue  # retry the truncated turn

                    return acct.success(response.text)

                truncated_turns = 0
                assistant_msg = ModelMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.text,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                durations.clear()

                async def resolve(tc: ToolCall) -> ToolResultMessage:
                    nonlocal mutation_lease
                    started_at = time.perf_counter()
                    try:
                        validation_error = _validate_tool_args(tc, tool_map)
                        if validation_error:
                            return _build_tool_result_message(tc, validation_error, is_error=True)

                        tool = self._tool_registry.get(tc.name)
                        if tool is not None and tool.mutating:
                            mutation_lease = self._workspace.acquire_lease("native-worker")
                            if mutation_lease is None:
                                return _build_tool_result_message(
                                    tc,
                                    "Cannot acquire mutation lease -- "
                                    "workspace is locked by another process",
                                    is_error=True,
                                )
                        return await self._execute_tool(tc, tool_map, mutation_lease)
                    finally:
                        durations[tc.id] = round((time.perf_counter() - started_at) * 1000, 1)

                for batch in _tool_batches(response.tool_calls, self._tool_registry):
                    try:
                        results = await _within_wall_time(
                            asyncio.gather(*(resolve(tc) for tc in batch)), remaining_wall()
                        )
                    except _WallTimeExceeded:
                        return acct.failure("budget", "Wall time limit exceeded during tool call")

                    for tc, result in zip(batch, results, strict=True):
                        tool_results_all.append(result)
                        messages.append(
                            ModelMessage(
                                role=MessageRole.TOOL,
                                content=result.content,
                                tool_call_id=tc.id,
                            )
                        )
                        await emit(
                            WorkerEventKind.TOOL_CALL_COMPLETED,
                            tool=tc.name,
                            ok=not result.is_error,
                            duration_ms=durations[tc.id],
                            result_bytes=len(result.content),
                        )

                acct.record_round()
                if on_checkpoint is not None:
                    on_checkpoint(list(messages))

        finally:
            if mutation_lease:
                self._workspace.release_lease(mutation_lease)

    def _build_user_prompt(self, cell: GraphCell) -> str:
        """Build the user prompt from a cell's contract."""
        parts = [f"Objective: {cell.contract.objective}"]
        for label, value in (
            ("Scope", cell.contract.scope),
            ("Deliverables", ", ".join(cell.contract.deliverables)),
            ("Constraints", ", ".join(cell.contract.constraints)),
            ("Forbidden", ", ".join(cell.contract.forbidden_outcomes)),
        ):
            if value:
                parts.append(f"{label}: {value}")
        criteria = _render_criteria(cell)
        if criteria:
            parts.append(f"Your work is accepted only if all of these hold:\n{criteria}")
        return "\n".join(parts)

    async def _execute_tool(
        self,
        tool_call: ToolCall,
        tool_map: dict[str, ToolDefinition],
        lease: str | None,
    ) -> ToolResultMessage:
        """Execute a single tool call via the tool registry.

        The worker keeps no ``if/elif`` name chain: every tool's behaviour
        lives in :class:`ToolRegistry`, and dispatch is a lookup. The registry
        returns the workspace's typed :class:`ToolResult`; this method renders
        it into a ``ToolResultMessage`` -- success text, a truncation marker,
        or an error.
        """
        name = tool_call.name
        args = tool_call.arguments

        ctx = _WorkerToolContext(
            workspace=self._workspace,
            web_reader=self._web_reader,
            command_profiles=self._command_profiles,
            lease=lease,
        )

        try:
            result = await self._tool_registry.execute(name, ctx, args)
            if result is None:
                return _build_tool_result_message(
                    tool_call,
                    f"Unknown tool: {name}",
                    is_error=True,
                )

            if result.ok:
                content = result.data.decode("utf-8", errors="replace") if result.data else ""
                if result.truncated:
                    content += f"\n[truncated, {result.total_bytes} total bytes]"
                return _build_tool_result_message(tool_call, content)
            else:
                detail = result.error or f"Tool '{name}' failed"
                if result.data:
                    detail += "\n" + result.data.decode("utf-8", errors="replace")
                    if result.truncated:
                        detail += f"\n[truncated, {result.total_bytes} total bytes]"
                return _build_tool_result_message(tool_call, detail, is_error=True)

        except Exception as e:  # noqa: BLE001 -- safety net for tool execution
            return _build_tool_result_message(
                tool_call,
                f"Tool execution error: {e}",
                is_error=True,
            )
