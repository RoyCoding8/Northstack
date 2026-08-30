"""In-process provider fakes.

``httpx.MockTransport`` is the seam: adapters and ``_execute_request`` already
take the client as a parameter, so a canned transport exercises the real POST,
status pipeline and JSON parsing without a network or a mocked coroutine.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from northstack.adapters.artifacts import ArtifactStore
from northstack.domain.outcome import ArtifactRef

OPENAI_OK: dict[str, Any] = {
    "model": "test-model",
    "choices": [
        {
            "message": {"role": "assistant", "content": "hello", "tool_calls": []},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
}

ANTHROPIC_OK: dict[str, Any] = {
    "model": "test-model",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 11, "output_tokens": 7},
}


class Recorder:
    """Captures every request the transport saw."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def header(self, name: str) -> str | None:
        return self.last.headers.get(name)


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[httpx.AsyncClient, Recorder]:
    """An AsyncClient whose transport is ``handler``, plus a request recorder."""
    recorder = Recorder()

    def wrapped(request: httpx.Request) -> httpx.Response:
        recorder.requests.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(wrapped)), recorder


def responds(
    payload: dict[str, Any] | None = None,
    *,
    status: int = 200,
    text: str | None = None,
) -> tuple[httpx.AsyncClient, Recorder]:
    """A client returning one canned response for every request."""

    def handler(_: httpx.Request) -> httpx.Response:
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload if payload is not None else {})

    return client_for(handler)


def raises(error: Exception) -> tuple[httpx.AsyncClient, Recorder]:
    """A client whose transport raises ``error`` instead of responding."""

    def handler(_: httpx.Request) -> httpx.Response:
        raise error

    return client_for(handler)


class FailingArtifactStore(ArtifactStore):
    """An ArtifactStore whose write always fails with OSError."""

    def write(self, content: bytes, *, media_type: str) -> ArtifactRef:
        raise OSError("disk full")


class RecordingArtifactStore(ArtifactStore):
    """An ArtifactStore that records what was written alongside storing it."""

    def __init__(self, base_path: Path | str) -> None:
        super().__init__(base_path)
        self.writes: list[tuple[bytes, str]] = []

    def write(self, content: bytes, *, media_type: str) -> ArtifactRef:
        self.writes.append((content, media_type))
        return super().write(content, media_type=media_type)
