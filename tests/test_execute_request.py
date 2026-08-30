"""Regression lock: shared HTTP request+error-handling in providers.

OpenAIAdapter.complete and AnthropicAdapter.complete duplicated the same
post -> status/auth -> json -> error pipeline; it is centralized in
``_execute_request``. This test guards the dedup against silent regression.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from northstack.adapters.providers import gateway as gateway_module
from northstack.adapters.providers.gateway import (
    AuthProviderError,
    HTTPProviderError,
    ProviderError,
    _execute_request,
)

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "northstack"
    / "adapters"
    / "providers"
    / "gateway.py"
)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gateway_module, "_retry_sleep", _noop_sleep)


async def _noop_sleep(_seconds: float) -> None:
    return


def _adapters_calling_post_directly() -> list[str]:
    """Adapter methods that call client.post directly instead of _execute_request."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name != "complete":
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Await)
                and isinstance(sub.value, ast.Call)
                and isinstance(sub.value.func, ast.Attribute)
                and sub.value.func.attr == "post"
            ):
                hits.append(node.name)
    return hits


def test_complete_methods_do_not_call_post_directly() -> None:
    hits = _adapters_calling_post_directly()
    assert not hits, (
        f"Adapter.complete methods call client.post directly: {hits}; use _execute_request instead."
    )


def _mock(status_code: int, payload: object | None = None, text: str | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    if payload is not None:
        r.json.return_value = payload
    if text is not None:
        r.text = text
    return r


@pytest.mark.asyncio
async def test_execute_request_returns_parsed_json() -> None:
    payload = {"ok": True}
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _mock(200, payload=payload)
    data = await _execute_request(
        client, "http://x/v1", {}, {}, "openai", "gpt-4", redacted_base_url="http://x"
    )
    assert data == payload


@pytest.mark.asyncio
async def test_execute_request_401_raises_auth() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _mock(401, text="Unauthorized")
    with pytest.raises(AuthProviderError):
        await _execute_request(
            client, "http://x/v1", {}, {}, "openai", "gpt-4", redacted_base_url="http://x"
        )


@pytest.mark.asyncio
async def test_execute_request_non_200_raises_http() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _mock(500, text="boom")
    with pytest.raises(HTTPProviderError) as exc:
        await _execute_request(
            client, "http://x/v1", {}, {}, "anthropic", "claude", redacted_base_url="http://x"
        )
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_execute_request_network_error_raises_provider() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.side_effect = httpx.ConnectError("nope")
    with pytest.raises(ProviderError) as exc:
        await _execute_request(
            client, "http://x/v1", {}, {}, "openai", "gpt-4", redacted_base_url="http://x"
        )
    assert not isinstance(exc.value, AuthProviderError)
    assert not isinstance(exc.value, HTTPProviderError)
    # Transport retries are bounded: initial attempt + 2 retries.
    assert client.post.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_execute_request_retries_transient_status_then_succeeds(status: int) -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.side_effect = [_mock(status, text="down"), _mock(200, payload={"ok": True})]
    data = await _execute_request(
        client, "http://x/v1", {}, {}, "openai", "gpt-4", redacted_base_url="http://x"
    )
    assert data == {"ok": True}
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_execute_request_does_not_retry_client_errors() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _mock(422, text="bad request")
    with pytest.raises(HTTPProviderError) as exc:
        await _execute_request(
            client, "http://x/v1", {}, {}, "openai", "gpt-4", redacted_base_url="http://x"
        )
    assert exc.value.status_code == 422
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_execute_request_malformed_json_raises_provider() -> None:
    r = MagicMock()
    r.status_code = 200
    r.json.side_effect = json.JSONDecodeError("no", "", 0)
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = r
    with pytest.raises(ProviderError) as exc:
        await _execute_request(
            client, "http://x/v1", {}, {}, "anthropic", "claude", redacted_base_url="http://x"
        )
    assert not isinstance(exc.value, AuthProviderError)
    assert not isinstance(exc.value, HTTPProviderError)


@pytest.mark.asyncio
async def test_execute_request_profile_retry_overrides_bound_attempts() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.side_effect = httpx.ConnectError("nope")
    with pytest.raises(ProviderError):
        await _execute_request(
            client,
            "http://x/v1",
            {},
            {},
            "openai",
            "gpt-4",
            redacted_base_url="http://x",
            retries=4,
            backoff_seconds=(0.1, 0.2, 0.3, 0.4),
        )
    assert client.post.await_count == 5


@pytest.mark.asyncio
async def test_execute_request_zero_retries_makes_single_attempt() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _mock(503, text="down")
    with pytest.raises(HTTPProviderError) as exc:
        await _execute_request(
            client,
            "http://x/v1",
            {},
            {},
            "openai",
            "gpt-4",
            redacted_base_url="http://x",
            retries=0,
        )
    assert exc.value.status_code == 503
    assert client.post.await_count == 1


@pytest.mark.asyncio
async def test_execute_request_backoff_shorter_than_retries_clamps() -> None:
    sleeps: list[float] = []

    async def _record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.side_effect = [
        _mock(502, text="a"),
        _mock(503, text="b"),
        _mock(200, payload={"ok": True}),
    ]
    data = await _execute_request(
        client,
        "http://x/v1",
        {},
        {},
        "openai",
        "gpt-4",
        redacted_base_url="http://x",
        retries=2,
        backoff_seconds=[0.25],
        sleep_fn=_record_sleep,
    )
    assert data == {"ok": True}
    assert sleeps == [0.25, 0.25]


@pytest.mark.asyncio
async def test_execute_request_propagates_provider_and_model_labels() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _mock(500, text="boom")
    with pytest.raises(HTTPProviderError) as exc:
        await _execute_request(
            client, "http://x/v1", {}, {}, "anthropic", "claude-3", redacted_base_url="http://x"
        )
    assert exc.value.provider == "anthropic"
    assert exc.value.model == "claude-3"
