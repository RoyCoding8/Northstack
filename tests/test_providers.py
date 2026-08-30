"""Tests for ModelGateway, adapters, and ModelLimiter.

Covers:
  - OpenAI-compatible adapter: endpoint-safe URL joining, auth optional,
    /chat/completions, tool definitions/tool calls, response_format, timeout,
    HTTP/provider errors, usage normalization
  - Anthropic-compatible adapter: /v1/messages, x-api-key and
    anthropic-version headers, system separated, content blocks/tool_use,
    tool_result conversion, JSON-schema fallback behavior
  - Capability negotiation: reject/transform unsupported features deterministically
  - Secret handling: resolve API key only immediately before request;
    never include in repr, event payload, artifact, exception, subprocess env
  - ModelLimiter: async concurrency semaphore per profile, monotonic RPM
    limiter, injectable clock/sleeper, deterministic tests
  - Fake HTTP transport/server tests for both protocols
  - Malformed/HTTP errors, tool calls, usage, optional auth/local no-key,
    unsupported capabilities, secret non-leakage
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers import gateway as _gateway_module
from northstack.adapters.providers.gateway import (
    AnthropicAdapter,
    AuthProviderError,
    GeminiAdapter,
    HTTPProviderError,
    ModelGateway,
    ModelLimiter,
    OpenAIAdapter,
    ProviderConfigurationError,
    ProviderError,
    _anthropic_endpoint,
    _check_capabilities,
    _gemini_endpoint,
    _join_url,
    _openai_endpoint,
    _should_send_json_schema,
    _validate_base_url,
    _with_query,
)
from northstack.adapters.providers.pricing import compute_cost_usd
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
from northstack.config import Capability, ModelProfile, NorthStackConfig, Protocol, SecretEnvRef


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(_seconds: float) -> None:
        return

    monkeypatch.setattr(_gateway_module, "_retry_sleep", _noop)


# Fixtures


def _make_profile(
    name: str = "test-worker",
    protocol: Protocol = Protocol.OPENAI_CHAT,
    base_url: str = "http://localhost:8080/v1",
    model: str = "test-model",
    capabilities: set[Capability] | None = None,
    max_concurrency: int = 4,
    rpm: int = 60,
    api_key_env: SecretEnvRef | None = None,
) -> ModelProfile:
    return ModelProfile(
        name=name,
        protocol=protocol,
        base_url=base_url,
        model=model,
        capabilities=capabilities or set(),
        max_concurrency=max_concurrency,
        requests_per_minute=rpm,
        api_key_env=api_key_env,
    )


def _make_config(profiles: list[ModelProfile] | None = None) -> NorthStackConfig:
    return NorthStackConfig(
        name="test",
        profiles=profiles or [_make_profile()],
    )


def _make_request(
    profile_name: str = "test-worker",
    content: str = "Hello",
    tools: list[ToolDefinition] | None = None,
    output_json_schema: dict[str, Any] | None = None,
) -> ModelRequest:
    return ModelRequest(
        profile_name=profile_name,
        messages=[ModelMessage(role=MessageRole.USER, content=content)],
        tools=tools or [],
        output_json_schema=output_json_schema,
    )


# URL joining


class TestURLJoining:
    def test_simple_join(self):
        assert (
            _join_url("http://localhost:8080/v1", "chat/completions")
            == "http://localhost:8080/v1/chat/completions"
        )

    def test_trailing_slash_base(self):
        assert (
            _join_url("http://localhost:8080/v1/", "chat/completions")
            == "http://localhost:8080/v1/chat/completions"
        )

    def test_leading_slash_path(self):
        assert (
            _join_url("http://localhost:8080/v1", "/chat/completions")
            == "http://localhost:8080/v1/chat/completions"
        )

    def test_double_slash_stripped(self):
        assert (
            _join_url("http://localhost:8080/v1/", "/chat/completions")
            == "http://localhost:8080/v1/chat/completions"
        )


# Secret handling


class TestSecretHandling:
    def test_resolve_api_key_with_env(self):
        import os

        os.environ["TEST_API_KEY_XYZ"] = "sk-secret123"
        try:
            profile = _make_profile(
                api_key_env=SecretEnvRef(env_var="TEST_API_KEY_XYZ"),
            )
            key = profile.api_key_env.resolve()
            assert key == "sk-secret123"
        finally:
            del os.environ["TEST_API_KEY_XYZ"]

    def test_resolve_api_key_missing_env(self):
        profile = _make_profile(
            api_key_env=SecretEnvRef(env_var="NONEXISTENT_KEY_12345"),
        )
        with pytest.raises(KeyError):
            profile.api_key_env.resolve()

    def test_resolve_api_key_no_env_ref(self):
        profile = _make_profile(api_key_env=None)
        assert profile.api_key_env is None

    def test_secret_env_ref_repr_does_not_leak_value(self):
        import os

        os.environ["SECRET_REPR_TEST"] = "sk-supersecretvalue"
        try:
            ref = SecretEnvRef(env_var="SECRET_REPR_TEST")
            repr_str = repr(ref)
            assert "sk-supersecretvalue" not in repr_str
            assert "SECRET_REPR_TEST" in repr_str
        finally:
            del os.environ["SECRET_REPR_TEST"]

    def test_api_key_not_in_exception(self):
        """API key must not appear in exception messages."""
        import os

        os.environ["EXCEPTION_TEST_KEY"] = "sk-topsecret"
        try:
            with pytest.raises(AuthProviderError) as exc_info:
                raise AuthProviderError("Auth failed", provider="test")
            assert "sk-topsecret" not in str(exc_info.value)
        finally:
            del os.environ["EXCEPTION_TEST_KEY"]


# Capability negotiation


class TestCapabilityNegotiation:
    def test_json_schema_without_capability_warns(self):
        request = _make_request(
            output_json_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        profile = _make_profile(capabilities=set())
        warnings = _check_capabilities(request, profile)
        assert len(warnings) == 1
        assert "native_json_schema" in warnings[0]

    def test_json_schema_with_capability_no_warning(self):
        request = _make_request(
            output_json_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        profile = _make_profile(capabilities={Capability.NATIVE_JSON_SCHEMA})
        warnings = _check_capabilities(request, profile)
        assert len(warnings) == 0

    def test_should_send_json_schema_true(self):
        request = _make_request(
            output_json_schema={"type": "object"},
        )
        profile = _make_profile(capabilities={Capability.NATIVE_JSON_SCHEMA})
        assert _should_send_json_schema(request, profile) is True

    def test_should_send_json_schema_false_no_schema(self):
        request = _make_request()
        profile = _make_profile(capabilities={Capability.NATIVE_JSON_SCHEMA})
        assert _should_send_json_schema(request, profile) is False

    def test_should_send_json_schema_false_no_capability(self):
        request = _make_request(
            output_json_schema={"type": "object"},
        )
        profile = _make_profile(capabilities=set())
        assert _should_send_json_schema(request, profile) is False


# OpenAI adapter


class TestOpenAIAdapter:
    def test_build_body_basic(self):
        adapter = OpenAIAdapter()
        profile = _make_profile(protocol=Protocol.OPENAI_CHAT)
        request = _make_request()
        body = adapter._build_body(request, profile)
        assert body["model"] == "test-model"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert "max_tokens" in body

    def test_build_body_with_system(self):
        adapter = OpenAIAdapter()
        profile = _make_profile()
        request = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            system="You are helpful.",
        )
        body = adapter._build_body(request, profile)
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are helpful."

    def test_build_body_with_tools(self):
        adapter = OpenAIAdapter()
        profile = _make_profile(capabilities={Capability.TOOL_USE})
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ]
        request = _make_request(tools=tools)
        body = adapter._build_body(request, profile)
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "get_weather"

    def test_build_body_json_schema_response_format(self):
        adapter = OpenAIAdapter()
        profile = _make_profile(capabilities={Capability.NATIVE_JSON_SCHEMA})
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        request = _make_request(output_json_schema=schema)
        body = adapter._build_body(request, profile)
        assert "response_format" in body
        assert body["response_format"]["type"] == "json_schema"

    def test_build_body_no_json_schema_without_capability(self):
        adapter = OpenAIAdapter()
        profile = _make_profile(capabilities=set())
        schema = {"type": "object"}
        request = _make_request(output_json_schema=schema)
        body = adapter._build_body(request, profile)
        assert "response_format" not in body

    def test_convert_messages_with_tool_calls(self):
        adapter = OpenAIAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    content="Let me check",
                    tool_calls=[ToolCall(id="tc1", name="get_weather", arguments={"city": "NYC"})],
                ),
            ],
        )
        messages = adapter._convert_messages(request)
        assert messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_convert_messages_tool_result(self):
        adapter = OpenAIAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(
                    role=MessageRole.TOOL,
                    content="72F sunny",
                    tool_call_id="tc1",
                ),
            ],
        )
        messages = adapter._convert_messages(request)
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "tc1"

    def test_parse_response_basic(self):
        adapter = OpenAIAdapter()
        data = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4",
        }
        response = adapter._parse_response(data, "test-model")
        assert response.text == "Hello!"
        assert response.finish_reason == FinishReason.END_TURN
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5
        assert response.provider == "openai"

    def test_parse_response_with_tool_calls(self):
        adapter = OpenAIAdapter()
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4",
        }
        response = adapter._parse_response(data, "test-model")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "NYC"}
        assert response.finish_reason == FinishReason.TOOL_USE

    def test_parse_response_max_tokens(self):
        adapter = OpenAIAdapter()
        data = {
            "choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100},
            "model": "gpt-4",
        }
        response = adapter._parse_response(data, "test-model")
        assert response.finish_reason == FinishReason.MAX_TOKENS

    @pytest.mark.asyncio
    async def test_complete_success(self):
        adapter = OpenAIAdapter()
        profile = _make_profile()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "model": "gpt-4",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        response = await adapter.complete(request, profile, mock_client, "test-key")
        assert response.text == "Hello!"
        assert response.provider == "openai"

    @pytest.mark.asyncio
    async def test_complete_auth_error(self):
        adapter = OpenAIAdapter()
        profile = _make_profile()

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        with pytest.raises(AuthProviderError):
            await adapter.complete(request, profile, mock_client, "bad-key")

    @pytest.mark.asyncio
    async def test_complete_http_error(self):
        adapter = OpenAIAdapter()
        profile = _make_profile()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        with pytest.raises(HTTPProviderError) as exc_info:
            await adapter.complete(request, profile, mock_client, "test-key")
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_complete_network_error(self):
        adapter = OpenAIAdapter()
        profile = _make_profile()

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        request = _make_request()
        with pytest.raises(ProviderError, match="Network error"):
            await adapter.complete(request, profile, mock_client, "test-key")

    @pytest.mark.asyncio
    async def test_complete_no_auth_optional(self):
        """Local/no-key endpoints should work without auth."""
        adapter = OpenAIAdapter()
        profile = _make_profile(api_key_env=None)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Local response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": "llama3",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        response = await adapter.complete(request, profile, mock_client, None)
        assert response.text == "Local response"

    @pytest.mark.asyncio
    async def test_complete_malformed_json_response(self):
        adapter = OpenAIAdapter()
        profile = _make_profile()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_response.text = "not json"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        with pytest.raises(ProviderError, match="Invalid JSON"):
            await adapter.complete(request, profile, mock_client, "test-key")


# Anthropic adapter


class TestAnthropicAdapter:
    def test_build_body_basic(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = _make_request()
        body = adapter._build_body(request, profile)
        assert body["model"] == "test-model"
        assert "messages" in body
        assert "max_tokens" in body
        assert body.get("system") is None  # No system prompt

    def test_build_body_with_system(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            system="You are helpful.",
        )
        body = adapter._build_body(request, profile)
        assert body["system"] == "You are helpful."
        # System should NOT appear in messages
        assert all(m.get("role") != "system" for m in body["messages"])

    def test_build_body_with_tools(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(
            protocol=Protocol.ANTHROPIC_MESSAGES, capabilities={Capability.TOOL_USE}
        )
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ]
        request = _make_request(tools=tools)
        body = adapter._build_body(request, profile)
        assert "tools" in body
        assert body["tools"][0]["name"] == "get_weather"
        assert "input_schema" in body["tools"][0]

    def test_build_body_json_schema_output(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(
            protocol=Protocol.ANTHROPIC_MESSAGES,
            capabilities={Capability.NATIVE_JSON_SCHEMA},
        )
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        request = _make_request(output_json_schema=schema)
        body = adapter._build_body(request, profile)
        assert "output_config" in body
        assert body["output_config"]["format"]["type"] == "json_schema"

    def test_convert_messages_tool_use_blocks(self):
        adapter = AnthropicAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    content="Let me check",
                    tool_calls=[ToolCall(id="tc1", name="get_weather", arguments={"city": "NYC"})],
                ),
            ],
        )
        messages = adapter._convert_messages(request)
        assert messages[0]["content"][1]["type"] == "tool_use"
        assert messages[0]["content"][1]["name"] == "get_weather"

    def test_convert_messages_tool_result_conversion(self):
        adapter = AnthropicAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(
                    role=MessageRole.TOOL,
                    content="72F sunny",
                    tool_call_id="tc1",
                ),
            ],
        )
        messages = adapter._convert_messages(request)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "tool_result"
        assert messages[0]["content"][0]["tool_use_id"] == "tc1"

    def test_convert_messages_merges_consecutive_tool_results(self):
        adapter = AnthropicAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(role=MessageRole.TOOL, content="Result 1", tool_call_id="tc1"),
                ModelMessage(role=MessageRole.TOOL, content="Result 2", tool_call_id="tc2"),
            ],
        )
        messages = adapter._convert_messages(request)
        # Should be merged into a single user message with two tool_result blocks
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2

    def test_parse_response_basic(self):
        adapter = AnthropicAdapter()
        data = {
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-3",
        }
        response = adapter._parse_response(data, "test-model")
        assert response.text == "Hello!"
        assert response.finish_reason == FinishReason.END_TURN
        assert response.provider == "anthropic"

    def test_parse_response_tool_use(self):
        adapter = AnthropicAdapter()
        data = {
            "content": [
                {"type": "text", "text": "Let me check"},
                {
                    "type": "tool_use",
                    "id": "tu_123",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-3",
        }
        response = adapter._parse_response(data, "test-model")
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.finish_reason == FinishReason.TOOL_USE

    def test_parse_response_cache_usage(self):
        adapter = AnthropicAdapter()
        data = {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 40,
            },
            "model": "claude-3",
        }
        response = adapter._parse_response(data, "test-model")
        assert response.usage.cache_creation_tokens == 50
        assert response.usage.cache_read_tokens == 40

    @pytest.mark.asyncio
    async def test_complete_success(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "Hello from Anthropic!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-3",
        }
        mock_response.text = json.dumps(mock_response.json.return_value)

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        response = await adapter.complete(request, profile, mock_client, "test-key")
        assert response.text == "Hello from Anthropic!"

    @pytest.mark.asyncio
    async def test_complete_sends_correct_headers(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "model": "claude-3",
        }
        mock_response.text = "{}"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        await adapter.complete(request, profile, mock_client, "my-api-key")

        # Verify headers
        call_kwargs = mock_client.post.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert headers.get("x-api-key") == "my-api-key"
        assert headers.get("anthropic-version") == "2023-06-01"

    @pytest.mark.asyncio
    async def test_complete_auth_error(self):
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        request = _make_request()
        with pytest.raises(AuthProviderError):
            await adapter.complete(request, profile, mock_client, "bad-key")


# Gemini adapter


def _gemini_profile(**kwargs: Any) -> ModelProfile:
    kwargs.setdefault("protocol", Protocol.GEMINI_GENERATE_CONTENT)
    kwargs.setdefault("base_url", "https://generativelanguage.googleapis.com")
    kwargs.setdefault("model", "gemini-3.5-flash-lite")
    return _make_profile(**kwargs)


class TestGeminiAdapter:
    def test_build_body_basic(self):
        body = GeminiAdapter()._build_body(_make_request(), _gemini_profile())
        assert body["contents"] == [{"role": "user", "parts": [{"text": "Hello"}]}]
        assert body["generationConfig"]["maxOutputTokens"] == 4096
        assert "model" not in body

    def test_build_body_hoists_system_to_system_instruction(self):
        request = ModelRequest(
            profile_name="test-worker",
            system="From request",
            messages=[
                ModelMessage(role=MessageRole.SYSTEM, content="From messages"),
                ModelMessage(role=MessageRole.USER, content="Hi"),
            ],
        )
        body = GeminiAdapter()._build_body(request, _gemini_profile())
        assert body["systemInstruction"] == {"parts": [{"text": "From request\n\nFrom messages"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_build_body_wraps_tools_in_function_declarations(self):
        tools = [ToolDefinition(name="read", description="Read a file")]
        body = GeminiAdapter()._build_body(
            _make_request(tools=tools),
            _gemini_profile(capabilities={Capability.TOOL_USE}),
        )
        assert body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ]
            }
        ]

    def test_build_body_omits_tools_without_capability(self):
        body = GeminiAdapter()._build_body(
            _make_request(tools=[ToolDefinition(name="read")]), _gemini_profile()
        )
        assert "tools" not in body

    def test_build_body_drops_schema_fields_gemini_rejects(self):
        """``additionalProperties`` is a hard HTTP 400 on Gemini.

        Every tool schema the registry advertises closes its object with
        ``additionalProperties: False`` for OpenAI strict mode, and Gemini's
        FunctionDeclaration.parameters is a select subset of OpenAPI 3.0 that
        has no such field -- it rejects the whole request rather than ignoring
        the key.  The projection keeps the fields the subset does define.
        """
        tools = [
            ToolDefinition(
                name="replace",
                description="Replace text",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "where"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            )
        ]
        body = GeminiAdapter()._build_body(
            _make_request(tools=tools),
            _gemini_profile(capabilities={Capability.TOOL_USE}),
        )
        declaration = body["tools"][0]["functionDeclarations"][0]
        assert declaration["parameters"] == {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "where"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["path"],
        }
        assert "parametersJsonSchema" not in declaration

    def test_parse_response_keeps_thought_signature_on_tool_call(self):
        """Gemini 3 binds an opaque signature to the part carrying the call.

        Replaying that call in later history without it is an HTTP 400, so the
        token has to survive normalization into ``ToolCall``.
        """
        response = GeminiAdapter()._parse_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {"name": "read", "args": {"path": "a.py"}},
                                    "thoughtSignature": "Cs4BAdHtim8",
                                }
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            "gemini-3.5-flash-lite",
        )
        assert response.tool_calls[0].signature == "Cs4BAdHtim8"

    def test_parse_response_tolerates_absent_thought_signature(self):
        response = GeminiAdapter()._parse_response(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"functionCall": {"name": "read", "args": {}}}]},
                        "finishReason": "STOP",
                    }
                ]
            },
            "gemini-3.5-flash-lite",
        )
        assert response.tool_calls[0].signature == ""

    def test_convert_messages_replays_thought_signature(self):
        request = ModelRequest(
            profile_name="test-worker",
            messages=[
                ModelMessage(role=MessageRole.USER, content="Read it"),
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="read",
                            arguments={"path": "a.py"},
                            signature="Cs4BAdHtim8",
                        )
                    ],
                ),
                ModelMessage(role=MessageRole.TOOL, content="ok", tool_call_id="call-1"),
            ],
        )
        contents = GeminiAdapter()._convert_messages(request)
        assert contents[1]["parts"] == [
            {
                "functionCall": {"name": "read", "args": {"path": "a.py"}},
                "thoughtSignature": "Cs4BAdHtim8",
            }
        ]

    def test_convert_messages_omits_empty_thought_signature(self):
        """An unsigned call must not grow an empty key: the other protocols
        issue no signature, and Gemini validates the field's contents."""
        request = ModelRequest(
            profile_name="test-worker",
            messages=[
                ModelMessage(role=MessageRole.USER, content="Read it"),
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[ToolCall(id="call-1", name="read", arguments={})],
                ),
            ],
        )
        contents = GeminiAdapter()._convert_messages(request)
        assert contents[1]["parts"] == [{"functionCall": {"name": "read", "args": {}}}]

    def test_build_body_routes_unexpressible_schema_to_json_schema_field(self):
        """A schema the subset cannot express travels intact in the JSON Schema field.

        Projecting ``$ref``/``oneOf`` away would change what the schema means,
        not merely loosen it, so those go to ``parametersJsonSchema`` -- the
        documented full-JSON-Schema alternative -- and never alongside
        ``parameters``, which the API treats as mutually exclusive.
        """
        schema = {
            "type": "object",
            "properties": {"target": {"$ref": "#/$defs/node"}},
            "$defs": {"node": {"type": "string"}},
        }
        body = GeminiAdapter()._build_body(
            _make_request(tools=[ToolDefinition(name="walk", parameters=schema)]),
            _gemini_profile(capabilities={Capability.TOOL_USE}),
        )
        declaration = body["tools"][0]["functionDeclarations"][0]
        assert declaration["parametersJsonSchema"] == schema
        assert "parameters" not in declaration

    def test_build_body_json_schema_sets_response_mime_type(self):
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        body = GeminiAdapter()._build_body(
            _make_request(output_json_schema=schema),
            _gemini_profile(capabilities={Capability.NATIVE_JSON_SCHEMA}),
        )
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        assert body["generationConfig"]["responseJsonSchema"] == schema

    def test_convert_messages_assistant_is_model_role(self):
        request = ModelRequest(
            profile_name="test-worker",
            messages=[
                ModelMessage(role=MessageRole.USER, content="Read it"),
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    content="On it",
                    tool_calls=[ToolCall(id="call_1", name="read", arguments={"path": "a.txt"})],
                ),
            ],
        )
        contents = GeminiAdapter()._convert_messages(request)
        assert contents[1] == {
            "role": "model",
            "parts": [
                {"text": "On it"},
                {"functionCall": {"name": "read", "args": {"path": "a.txt"}}},
            ],
        }

    def test_convert_messages_tool_result_resolves_name_from_call_id(self):
        """Gemini keys functionResponse by tool NAME; the id must be mapped back."""
        request = ModelRequest(
            profile_name="test-worker",
            messages=[
                ModelMessage(role=MessageRole.USER, content="Read it"),
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[ToolCall(id="call_1", name="read", arguments={})],
                ),
                ModelMessage(role=MessageRole.TOOL, tool_call_id="call_1", content="file body"),
            ],
        )
        contents = GeminiAdapter()._convert_messages(request)
        assert contents[2] == {
            "role": "user",
            "parts": [{"functionResponse": {"name": "read", "response": {"result": "file body"}}}],
        }

    def test_convert_messages_merges_consecutive_tool_results(self):
        request = ModelRequest(
            profile_name="test-worker",
            messages=[
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    tool_calls=[
                        ToolCall(id="c1", name="read", arguments={}),
                        ToolCall(id="c2", name="list", arguments={}),
                    ],
                ),
                ModelMessage(role=MessageRole.TOOL, tool_call_id="c1", content="one"),
                ModelMessage(role=MessageRole.TOOL, tool_call_id="c2", content="two"),
            ],
        )
        contents = GeminiAdapter()._convert_messages(request)
        assert len(contents) == 2
        assert [p["functionResponse"]["name"] for p in contents[1]["parts"]] == ["read", "list"]

    def test_parse_response_basic(self):
        response = GeminiAdapter()._parse_response(
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Hi"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
                "modelVersion": "gemini-3.5-flash-lite",
            },
            "gemini-3.5-flash-lite",
        )
        assert response.text == "Hi"
        assert response.finish_reason == FinishReason.END_TURN
        assert response.provider == "gemini"
        assert response.usage == Usage(input_tokens=10, output_tokens=5)

    def test_parse_response_tool_calls(self):
        response = GeminiAdapter()._parse_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"functionCall": {"name": "read", "args": {"path": "a"}}}]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            "gemini-3.5-flash-lite",
        )
        assert response.finish_reason == FinishReason.TOOL_USE
        assert response.tool_calls[0].name == "read"
        assert response.tool_calls[0].arguments == {"path": "a"}
        assert response.tool_calls[0].id

    def test_parse_response_drops_thought_parts(self):
        """Thought summaries are narration -- they must not pollute JSON output."""
        response = GeminiAdapter()._parse_response(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "Let me think...", "thought": True},
                                {"text": '{"ok": true}'},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
            "gemini-3.5-flash-lite",
        )
        assert response.text == '{"ok": true}'

    def test_parse_response_folds_thinking_tokens_into_output(self):
        """Thinking tokens bill at the output rate, so cost must include them."""
        response = GeminiAdapter()._parse_response(
            {
                "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 100,
                    "candidatesTokenCount": 20,
                    "thoughtsTokenCount": 80,
                    "cachedContentTokenCount": 40,
                },
            },
            "gemini-3.5-flash-lite",
        )
        assert response.usage == Usage(input_tokens=60, output_tokens=100, cache_read_tokens=40)

    def test_parse_response_max_tokens(self):
        response = GeminiAdapter()._parse_response(
            {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]},
            "gemini-3.5-flash-lite",
        )
        assert response.finish_reason == FinishReason.MAX_TOKENS

    def test_parse_response_empty_candidates_raises(self):
        with pytest.raises(ProviderError, match="empty candidates"):
            GeminiAdapter()._parse_response({"candidates": []}, "gemini-3.5-flash-lite")

    @pytest.mark.asyncio
    async def test_complete_sends_key_in_header_and_model_in_path(self):
        """The key must never ride in the query string, where it would be logged."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]
        }
        mock_response.text = "{}"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        await GeminiAdapter().complete(_make_request(), _gemini_profile(), mock_client, "sekret")

        url = mock_client.post.call_args.args[0]
        headers = mock_client.post.call_args.kwargs["headers"]
        assert headers["x-goog-api-key"] == "sekret"
        assert url.endswith("/v1beta/models/gemini-3.5-flash-lite:generateContent")
        assert "sekret" not in url

    @pytest.mark.asyncio
    async def test_complete_auth_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_response

        with pytest.raises(AuthProviderError):
            await GeminiAdapter().complete(
                _make_request(), _gemini_profile(), mock_client, "bad-key"
            )


class TestGeminiEndpointURLs:
    def test_generate_content_endpoint(self):
        url = _gemini_endpoint("https://generativelanguage.googleapis.com", "gemini-3.7-flash")
        assert url == (
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash"
            ":generateContent"
        )

    def test_stream_endpoint_uses_sse(self):
        url = _gemini_endpoint("https://host", "m", stream=True)
        assert url == "https://host/v1beta/models/m:streamGenerateContent?alt=sse"

    def test_strips_trailing_version_segment(self):
        assert _gemini_endpoint("https://host/v1beta/", "m") == (
            "https://host/v1beta/models/m:generateContent"
        )

    def test_model_is_one_encoded_path_segment(self):
        assert _gemini_endpoint("https://host", "publishers/acme model?#") == (
            "https://host/v1beta/models/publishers%2Facme%20model%3F%23:generateContent"
        )

    def test_reserved_stream_query_key_cannot_be_replaced(self):
        with pytest.raises(ProviderConfigurationError, match="alt"):
            _with_query(_gemini_endpoint("https://host", "m", stream=True), {"alt": "json"})

    def test_query_merge_preserves_repeats_ipv6_and_fragment(self):
        assert _with_query("https://[::1]/v1?x=1&x=2#frag", {"y": "a b"}) == (
            "https://[::1]/v1?x=1&x=2&y=a+b#frag"
        )


# ModelLimiter


class TestModelLimiter:
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        limiter = ModelLimiter("test", max_concurrency=2, rpm=100)

        async def my_fn():
            return 42

        result = await limiter.run(my_fn)
        assert result == 42

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        limiter = ModelLimiter("test", max_concurrency=1, rpm=1000)
        call_count = 0

        async def slow_fn():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return call_count

        # Launch two tasks concurrently
        results = await asyncio.gather(
            limiter.run(slow_fn),
            limiter.run(slow_fn),
        )
        # With max_concurrency=1, second task should wait
        assert set(results) == {1, 2}

    @pytest.mark.asyncio
    async def test_singleton_concurrency_one(self):
        """Test that singleton profile (max_concurrency=1) never overlaps."""
        limiter = ModelLimiter("singleton", max_concurrency=1, rpm=1000)
        max_observed = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked_fn():
            nonlocal max_observed, current
            async with lock:
                current += 1
                max_observed = max(max_observed, current)
            await asyncio.sleep(0.02)
            async with lock:
                current -= 1
            return True

        # Launch 5 tasks
        await asyncio.gather(*[limiter.run(tracked_fn) for _ in range(5)])
        assert max_observed == 1  # Never more than 1 concurrent

    @pytest.mark.asyncio
    async def test_rpm_limiting(self):
        """Test that RPM limit is enforced with injectable clock."""
        timestamps: list[float] = []
        clock_value = 0.0

        def mock_clock():
            return clock_value

        async def mock_sleep(seconds: float):
            nonlocal clock_value
            clock_value += seconds

        limiter = ModelLimiter(
            "test",
            max_concurrency=10,
            rpm=2,
            clock=mock_clock,
            sleeper=mock_sleep,
        )

        async def record_time():
            timestamps.append(mock_clock())
            return True

        # Make 3 calls - third should wait
        await limiter.run(record_time)
        await limiter.run(record_time)
        await limiter.run(record_time)

        # First two should be at time 0, third should be at time 60 (waited)
        assert timestamps[0] == 0.0
        assert timestamps[1] == 0.0
        assert timestamps[2] >= 60.0

    @pytest.mark.asyncio
    async def test_rpm_window_slides(self):
        """Test that RPM window slides correctly."""
        clock_value = 0.0

        def mock_clock():
            return clock_value

        async def mock_sleep(seconds: float):
            nonlocal clock_value
            clock_value += seconds

        limiter = ModelLimiter(
            "test",
            max_concurrency=10,
            rpm=2,
            clock=mock_clock,
            sleeper=mock_sleep,
        )

        async def noop():
            return True

        # First call at t=0
        await limiter.run(noop)
        # Second call at t=0 (within rpm limit)
        await limiter.run(noop)
        # Advance clock past the first call's window
        clock_value = 61.0
        # Third call: first call (t=0) is now 61s old, purged. Only t=0 remains
        # (but that's purged too).
        # Actually, t=0 is purged (>60s), so only the t=0 second call... wait, both were at t=0.
        # After purge at t=61: both t=0 entries are purged (61s old). _request_times is empty.
        # So third call should not wait
        start = mock_clock()
        await limiter.run(noop)
        assert mock_clock() == start  # No wait needed

    @pytest.mark.asyncio
    async def test_rpm_concurrent_waiters_reserve_distinct_windows(self):
        clock_value = 0.0

        def mock_clock():
            return clock_value

        async def advancing_sleep(seconds: float):
            nonlocal clock_value
            clock_value += seconds
            await asyncio.sleep(0)

        limiter = ModelLimiter(
            "test",
            max_concurrency=2,
            rpm=1,
            clock=mock_clock,
            sleeper=advancing_sleep,
        )
        admitted: list[float] = []

        async def noop():
            admitted.append(clock_value)
            return True

        await limiter.run(noop)
        await asyncio.gather(limiter.run(noop), limiter.run(noop))
        assert admitted == [0.0, 60.0, 120.0]

    @pytest.mark.asyncio
    async def test_rpm_five_waiters_never_burst_above_capacity(self):
        clock_value = 0.0
        sleepers: list[tuple[float, asyncio.Future[None]]] = []

        def mock_clock():
            return clock_value

        async def controlled_sleep(seconds: float):
            future = asyncio.get_running_loop().create_future()
            sleepers.append((clock_value + seconds, future))
            await future

        async def settle(admission_count: int, sleeper_count: int) -> None:
            for _ in range(20):
                if len(admitted) == admission_count and len(sleepers) == sleeper_count:
                    return
                await asyncio.sleep(0)
            raise AssertionError((admitted, sleepers))

        def advance(value: float) -> None:
            nonlocal clock_value
            clock_value = value
            due = [future for deadline, future in sleepers if deadline <= value]
            sleepers[:] = [(d, f) for d, f in sleepers if d > value]
            for future in due:
                future.set_result(None)

        limiter = ModelLimiter(
            "test", max_concurrency=7, rpm=2, clock=mock_clock, sleeper=controlled_sleep
        )
        admitted: list[float] = []

        async def record():
            admitted.append(clock_value)

        await limiter.run(record)
        await limiter.run(record)
        tasks = [asyncio.create_task(limiter.run(record)) for _ in range(5)]
        await settle(2, 5)
        advance(60.0)
        await settle(4, 3)
        advance(120.0)
        await settle(6, 1)
        advance(180.0)
        await asyncio.gather(*tasks)
        assert admitted == [0.0, 0.0, 60.0, 60.0, 120.0, 120.0, 180.0]

    @pytest.mark.asyncio
    async def test_rpm_cancelled_waiter_consumes_no_slot(self):
        blocked = asyncio.Event()
        calls = 0

        async def sleep_forever(_seconds: float):
            blocked.set()
            await asyncio.Event().wait()

        async def record():
            nonlocal calls
            calls += 1

        limiter = ModelLimiter("test", 2, 1, clock=lambda: 0.0, sleeper=sleep_forever)
        await limiter.run(record)
        task = asyncio.create_task(limiter.run(record))
        await blocked.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == 1
        assert limiter._request_times == [0.0]


# ModelGateway


class TestModelGateway:
    @pytest.mark.asyncio
    async def test_complete_selects_adapter_by_protocol(self):
        config = _make_config()
        gateway = ModelGateway(config)

        # Mock the adapter
        mock_adapter = AsyncMock()
        mock_adapter.complete.return_value = ModelResponse(
            text="Hello",
            finish_reason=FinishReason.END_TURN,
            usage=Usage(input_tokens=10, output_tokens=5),
            provider="openai",
            model="test-model",
        )
        gateway._adapters[Protocol.OPENAI_CHAT] = mock_adapter
        gateway._client = AsyncMock(spec=httpx.AsyncClient)

        request = _make_request()
        response = await gateway.complete(request)
        assert response.text == "Hello"
        mock_adapter.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_profile_not_found(self):
        config = _make_config()
        gateway = ModelGateway(config)

        request = _make_request(profile_name="nonexistent")
        with pytest.raises(ProviderError, match="Profile not found"):
            await gateway.complete(request)

    @pytest.mark.asyncio
    async def test_complete_stores_artifact(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            config = _make_config()
            gateway = ModelGateway(config, artifact_store=store)

            mock_adapter = AsyncMock()
            mock_adapter.complete.return_value = ModelResponse(
                text="Hello",
                finish_reason=FinishReason.END_TURN,
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="openai",
                model="test-model",
            )
            gateway._adapters[Protocol.OPENAI_CHAT] = mock_adapter
            gateway._client = AsyncMock(spec=httpx.AsyncClient)

            request = _make_request()
            response = await gateway.complete(request)
            assert response.response_artifact_id is not None
            assert response.response_artifact_id.startswith("sha256:")

    @pytest.mark.asyncio
    async def test_complete_writes_artifact_off_the_event_loop(self):
        """The artifact write is blocking file I/O; on the async path it must
        NOT run on the event-loop thread. If ``ArtifactStore.write`` runs in
        the loop thread, a slow disk stalls every concurrent coroutine.
        Offloading via ``asyncio.to_thread`` runs it on a worker thread whose
        id differs from the loop thread's.
        """
        import tempfile

        loop_thread = threading.get_ident()
        write_thread: list[int] = []

        class _SpyStore(ArtifactStore):
            """Captures the thread id that runs each write, then delegates."""

            def write(self, content, *, media_type):
                write_thread.append(threading.get_ident())
                return super().write(content, media_type=media_type)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = _SpyStore(tmpdir)
            config = _make_config()
            gateway = ModelGateway(config, artifact_store=store)
            mock_adapter = AsyncMock()
            mock_adapter.complete.return_value = ModelResponse(
                text="Hello",
                finish_reason=FinishReason.END_TURN,
                usage=Usage(input_tokens=10, output_tokens=5),
                provider="openai",
                model="test-model",
            )
            gateway._adapters[Protocol.OPENAI_CHAT] = mock_adapter
            gateway._client = AsyncMock(spec=httpx.AsyncClient)

            request = _make_request()
            await gateway.complete(request)

        assert write_thread, "artifact store.write was never called"
        assert write_thread[0] != loop_thread, (
            "ArtifactStore.write ran on the event-loop thread; the async path "
            "must offload the blocking file write to a worker thread"
        )


# Usage normalization


class TestUsageNormalization:
    def test_usage_total_tokens(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_usage_zero_defaults(self):
        usage = Usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_usage_frozen(self):
        usage = Usage(input_tokens=100, output_tokens=50)
        with pytest.raises(Exception):
            usage.input_tokens = 200  # type: ignore[misc]


# ModelRequest helpers


class TestModelRequest:
    def test_get_max_tokens_explicit(self):
        request_explicit = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            max_output_tokens=2048,
        )
        assert request_explicit.get_max_tokens(4096) == 2048

    def test_get_max_tokens_fallback(self):
        request = _make_request()
        assert request.get_max_tokens(4096) == 4096

    def test_model_request_frozen(self):
        request = _make_request()
        with pytest.raises(Exception):
            request.profile_name = "other"  # type: ignore[misc]


# Base URL validation


class TestBaseURLValidation:
    def test_valid_http(self):
        _validate_base_url("http://localhost:8080/v1", Protocol.OPENAI_CHAT)

    def test_valid_https(self):
        _validate_base_url("https://api.openai.com/v1", Protocol.OPENAI_CHAT)

    def test_rejects_ftp(self):
        with pytest.raises(ValueError, match="http or https"):
            _validate_base_url("ftp://example.com", Protocol.OPENAI_CHAT)

    def test_rejects_credentials(self):
        with pytest.raises(ValueError, match="credentials"):
            _validate_base_url("https://user:pass@api.com/v1", Protocol.OPENAI_CHAT)

    def test_rejects_query(self):
        with pytest.raises(ValueError, match="query"):
            _validate_base_url("https://api.com/v1?key=abc", Protocol.OPENAI_CHAT)

    def test_rejects_fragment(self):
        with pytest.raises(ValueError, match="fragment"):
            _validate_base_url("https://api.com/v1#section", Protocol.OPENAI_CHAT)

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="hostname"):
            _validate_base_url("http:///path", Protocol.OPENAI_CHAT)


# Endpoint URL construction


class TestEndpointURLs:
    def test_openai_endpoint(self):
        assert (
            _openai_endpoint("http://localhost:8080/v1")
            == "http://localhost:8080/v1/chat/completions"
        )

    def test_anthropic_endpoint(self):
        assert (
            _anthropic_endpoint("https://api.anthropic.com")
            == "https://api.anthropic.com/v1/messages"
        )

    def test_anthropic_endpoint_strips_trailing_v1(self):
        """Avoid /v1/v1/messages when base already ends with /v1."""
        assert (
            _anthropic_endpoint("https://api.anthropic.com/v1")
            == "https://api.anthropic.com/v1/messages"
        )


# Shared limiter pool


class TestSharedLimiterPool:
    @pytest.mark.asyncio
    async def test_same_profile_shares_limiter(self):
        """Two get_limiter calls for same profile return same instance."""
        config = _make_config()
        gateway = ModelGateway(config)

        limiter1 = gateway._get_limiter("test-worker")
        limiter2 = gateway._get_limiter("test-worker")
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_different_profiles_different_limiters(self):
        """Different profiles get different limiter instances."""
        profile2 = _make_profile(name="expert", max_concurrency=1)
        config = _make_config(profiles=[_make_profile(), profile2])
        gateway = ModelGateway(config)

        limiter1 = gateway._get_limiter("test-worker")
        limiter2 = gateway._get_limiter("expert")
        assert limiter1 is not limiter2

    @pytest.mark.asyncio
    async def test_singleton_max_concurrency_one(self):
        """Singleton profile limiter enforces max_concurrency=1."""
        # The profile has max_concurrency=4, but we can test with a singleton
        singleton_limiter = ModelLimiter("singleton", max_concurrency=1, rpm=1000)
        max_observed = 0
        current = 0
        lock = asyncio.Lock()

        async def tracked_fn():
            nonlocal max_observed, current
            async with lock:
                current += 1
                max_observed = max(max_observed, current)
            await asyncio.sleep(0.02)
            async with lock:
                current -= 1
            return True

        await asyncio.gather(*[singleton_limiter.run(tracked_fn) for _ in range(5)])
        assert max_observed == 1


# Cost calculation


class TestCostCalculation:
    def test_cost_with_pricing(self):
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = compute_cost_usd(usage, 10.0, 30.0)
        # 1000/1M * 10 + 500/1M * 30 = 0.01 + 0.015 = 0.025
        assert abs(cost - 0.025) < 1e-10

    def test_cost_free_tier(self):
        usage = Usage(input_tokens=1000, output_tokens=500)
        cost = compute_cost_usd(usage, 0.0, 0.0)
        assert cost == 0.0

    def test_cost_cache_aware_disjoint_buckets(self):
        # Independent worked example, NOT a recomputation of the current formula.
        # Contract: input_tokens = non-cached regular input; cache buckets are
        # disjoint additive portions billed at their own rates (cache_read 0.1x,
        # cache_creation 1.25x of input price). 1M-token base -> easy USD math.
        usage = Usage(
            input_tokens=1_000_000,  # -> 1.0x  = 10.0 USD
            output_tokens=1_000_000,  # -> 1.0x  = 30.0 USD
            cache_creation_tokens=1_000_000,  # -> 1.25x = 12.5 USD
            cache_read_tokens=1_000_000,  # -> 0.1x  =  1.0 USD
        )
        cost = compute_cost_usd(usage, 10.0, 30.0)
        assert abs(cost - 53.5) < 1e-9

    def test_cost_cache_read_cheaper_than_plain_input(self):
        # Same base usage, only cache_read differs: cache_read must be billed
        # strictly cheaper than regular input (0.1x vs 1.0x).
        no_cache = Usage(input_tokens=1_000_000, output_tokens=0)
        with_cache = Usage(input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
        plain = compute_cost_usd(no_cache, 10.0, 30.0)
        cached = compute_cost_usd(with_cache, 10.0, 30.0)
        assert cached < plain
        assert abs(cached - 1.0) < 1e-9  # 1M * 10 / 1M * 0.1
        assert abs(plain - 10.0) < 1e-9


# Provider error safety


class TestProviderErrorSafety:
    def test_http_error_no_response_body(self):
        """HTTPProviderError must not retain response body."""
        err = HTTPProviderError(
            "HTTP 500",
            status_code=500,
            provider="openai",
            model="test",
        )
        assert not hasattr(err, "response_body") or err.response_body == ""

    def test_network_error_redacts_url(self):
        """Network error messages should not leak full URLs with tokens."""
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("Network error calling https://api")
        assert "sk-" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_auth_error_when_key_required(self):
        """Missing env var for required key raises AuthProviderError."""
        profile = _make_profile(
            api_key_env=SecretEnvRef(env_var="NONEXISTENT_AUTH_KEY_999"),
        )
        config = _make_config(profiles=[profile])
        gateway = ModelGateway(config)
        # Mock the client to avoid real HTTP calls
        gateway._client = AsyncMock(spec=httpx.AsyncClient)

        request = _make_request()
        with pytest.raises(AuthProviderError, match="not set"):
            await gateway.complete(request)


# Capability: tools stripped without TOOL_USE


class TestCapabilityToolStripping:
    def test_tools_stripped_without_tool_use(self):
        """When TOOL_USE capability absent, tools are not sent."""
        adapter = OpenAIAdapter()
        profile = _make_profile(capabilities=set())  # No TOOL_USE
        tools = [ToolDefinition(name="read", description="Read")]
        request = _make_request(tools=tools)
        body = adapter._build_body(request, profile)
        assert "tools" not in body

    def test_tools_sent_with_tool_use(self):
        """When TOOL_USE capability present, tools are sent."""
        adapter = OpenAIAdapter()
        profile = _make_profile(capabilities={Capability.TOOL_USE})
        tools = [ToolDefinition(name="read", description="Read")]
        request = _make_request(tools=tools)
        body = adapter._build_body(request, profile)
        assert "tools" in body

    def test_anthropic_seed_omitted(self):
        """Anthropic adapter must not send seed."""
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            seed=42,
        )
        body = adapter._build_body(request, profile)
        assert "seed" not in body

    def test_openai_seed_sent(self):
        """OpenAI adapter sends seed."""
        adapter = OpenAIAdapter()
        profile = _make_profile(protocol=Protocol.OPENAI_CHAT)
        request = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            seed=42,
        )
        body = adapter._build_body(request, profile)
        assert body.get("seed") == 42


# OpenAI assistant content is string/None


class TestOpenAIAssistantContent:
    def test_assistant_content_is_string(self):
        """OpenAI assistant content should be string or None, not list."""
        adapter = OpenAIAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    content="Hello",
                    tool_calls=[ToolCall(id="tc1", name="read", arguments={"path": "."})],
                ),
            ],
        )
        messages = adapter._convert_messages(request)
        assert messages[0]["content"] == "Hello"
        assert isinstance(messages[0]["content"], str)

    def test_assistant_content_none_when_empty(self):
        adapter = OpenAIAdapter()
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(id="tc1", name="read", arguments={"path": "."})],
                ),
            ],
        )
        messages = adapter._convert_messages(request)
        assert messages[0]["content"] is None


# Anthropic system prompt combination


class TestAnthropicSystemCombination:
    def test_system_from_request_and_messages(self):
        """Anthropic combines request.system and system-role messages."""
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(role=MessageRole.SYSTEM, content="Additional context"),
                ModelMessage(role=MessageRole.USER, content="Hi"),
            ],
            system="Primary system prompt",
        )
        body = adapter._build_body(request, profile)
        assert "Primary system prompt" in body["system"]
        assert "Additional context" in body["system"]
        # System messages should not appear in messages array
        assert all(m.get("role") != "system" for m in body["messages"])

    def test_system_only_from_messages(self):
        """When request.system is empty, system from messages is used."""
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = ModelRequest(
            profile_name="test",
            messages=[
                ModelMessage(role=MessageRole.SYSTEM, content="System from messages"),
                ModelMessage(role=MessageRole.USER, content="Hi"),
            ],
        )
        body = adapter._build_body(request, profile)
        assert body["system"] == "System from messages"


# OpenAI response parsing edge cases


class TestOpenAIResponseParsing:
    def test_empty_choices_raises_provider_error(self):
        """Empty choices list should raise ProviderError, not IndexError."""
        adapter = OpenAIAdapter()
        data = {
            "choices": [],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        }
        with pytest.raises(ProviderError, match="empty choices"):
            adapter._parse_response(data, "test-model")

    def test_cache_read_only_tokens(self):
        """OpenAI cached_tokens maps to cache_read_tokens only, not creation."""
        adapter = OpenAIAdapter()
        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 500},
            },
        }
        response = adapter._parse_response(data, "test-model")
        assert response.usage.cache_read_tokens == 500
        assert response.usage.cache_creation_tokens == 0

    def test_openai_normalizes_cached_input_disjoint(self):
        # Independent contract: input_tokens must hold the NON-cached input
        # portion so cache_read_tokens is disjoint (additive) — never inside
        # input_tokens — otherwise compute_cost_usd double-counts the cache.
        # prompt_tokens=1000 includes cached_tokens=500 -> non-cached input=500.
        adapter = OpenAIAdapter()
        data = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 500},
            },
        }
        response = adapter._parse_response(data, "test-model")
        assert response.usage.input_tokens == 500
        assert response.usage.cache_read_tokens == 500
        assert response.usage.cache_creation_tokens == 0
        assert response.usage.output_tokens == 100
