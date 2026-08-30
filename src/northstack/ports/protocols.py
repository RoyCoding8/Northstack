"""Protocols the application layer talks through."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from northstack.adapters.providers.wire import ModelRequest, ToolDefinition
from northstack.adapters.workspace.restricted import ToolResult
from northstack.config import ModelProfile
from northstack.domain.graph import GraphCell
from northstack.domain.outcome import ArtifactRef
from northstack.events.catalog import EventPayload
from northstack.events.envelope import EventEnvelope


@runtime_checkable
class EventSink(Protocol):
    """Append-only destination for events."""

    def append_next(self, run_id: str, payload: EventPayload) -> EventEnvelope: ...

    def events(self, run_id: str) -> list[EventEnvelope]: ...

    def events_since(
        self, run_id: str, since: int = 0, limit: int = 500
    ) -> list[EventEnvelope]: ...


@runtime_checkable
class ArtifactSink(Protocol):
    """Content-addressed blob storage."""

    def write(self, content: bytes, *, media_type: str) -> ArtifactRef: ...

    def read(self, ref: ArtifactRef) -> bytes: ...


@runtime_checkable
class GatewayPort(Protocol):
    """Minimal gateway surface every model-backed role needs.

    ``complete`` returns whatever the caller validates downstream; concrete
    gateways return ``ModelResponse``.  ``profile`` exposes the resolved
    ModelProfile so callers can size requests without knowing the config.
    """

    async def complete(self, request: ModelRequest) -> Any: ...

    def profile(self, profile_name: str) -> ModelProfile: ...

    async def close(self) -> None: ...


@runtime_checkable
class WorkspacePort(Protocol):
    """Restricted filesystem the worker and verifier act through."""

    def read(self, path: str) -> ToolResult: ...

    def write(self, path: str, content: bytes) -> ToolResult: ...

    def list(self, path: str) -> ToolResult: ...


@runtime_checkable
class MemoryPort(Protocol):
    """Long-term, namespaced knowledge that outlives a single run.

    Deliberately narrower than the ledger's ``EventSink``: memory answers
    "what is relevant to this objective", not "what happened in run X", so it
    offers relevance search and no ordering guarantee at all.
    """

    def remember(self, namespace: str, text: str, *, source: str = "") -> Any: ...

    def recall(self, namespace: str, query: str, *, limit: int = 5) -> list[Any]: ...


class WorkerPort(Protocol):
    """Executes one cell attempt against a profile.

    The three observers are part of the contract, not an optional extra the
    caller probes for: a worker that silently drops ``on_event`` leaves the
    cell opaque, and one that rejects the kwarg surfaces as a fake provider
    error through the runner's catch-all.
    """

    async def run(
        self,
        cell: GraphCell,
        profile_name: str,
        tool_defs: list[ToolDefinition],
        *,
        system_prompt: str = "",
        output_json_schema: dict[str, Any] | None = None,
        resume_from_messages: list[Any] | None = None,
        on_progress: Callable[[], None] | None = None,
        on_checkpoint: Callable[[list[Any]], None] | None = None,
        on_event: Callable[[Any], Awaitable[None]] | None = None,
    ) -> Any: ...


class Clock(Protocol):
    """Monotonic time source; injectable so tests do not wait."""

    def __call__(self) -> float: ...


class Sleeper(Protocol):
    """Async delay seam; injectable so tests do not wait."""

    async def __call__(self, seconds: float) -> None: ...
