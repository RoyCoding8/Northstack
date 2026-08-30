"""Cross-protocol conformance: the three plugs must behave identically.

Every test here asks one neutral question of all three protocols at once.  The
per-adapter suites in ``test_providers.py`` and ``test_streaming.py`` are
vertical -- one adapter, one question -- and by construction cannot observe a
*difference between* adapters.  That difference is exactly what makes a profile
swap safe or unsafe, so it gets its own file.

A failure here is an agnosticism leak: swapping ``protocol`` in a profile would
change an answer the caller is entitled to treat as provider-neutral.
"""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from northstack.adapters.providers.gateway import (
    _AUTH_HEADERS,
    _PROVIDER_LABELS,
    _TOOL_ERROR_PREFIX,
    AnthropicAdapter,
    GeminiAdapter,
    OpenAIAdapter,
    ProviderProtocolError,
    StreamAssembler,
    _anthropic_stream_chunks,
    _build_headers,
    _gemini_endpoint,
    _gemini_stream_chunks,
    _openai_endpoint,
    _openai_stream_chunks,
    _with_query,
)
from northstack.adapters.providers.wire import (
    FinishReason,
    ImageContent,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)
from northstack.config import Capability, ModelProfile, Protocol

ADAPTERS = {
    Protocol.OPENAI_CHAT: OpenAIAdapter(),
    Protocol.ANTHROPIC_MESSAGES: AnthropicAdapter(),
    Protocol.GEMINI_GENERATE_CONTENT: GeminiAdapter(),
}
ALL_PROTOCOLS = list(ADAPTERS)
FULL_CAPS = {
    Capability.TOOL_USE,
    Capability.NATIVE_JSON_SCHEMA,
    Capability.STREAMING,
    Capability.VISION,
}
PIXEL = "iVBORw0KGgo="


def profile(protocol: Protocol, **kw: Any) -> ModelProfile:
    return ModelProfile(
        name=f"p-{protocol.value}",
        protocol=protocol,
        base_url="https://example.test",
        model="m-1",
        max_concurrency=1,
        capabilities=kw.pop("capabilities", FULL_CAPS),
        **kw,
    )


def build(protocol: Protocol, request: ModelRequest, **kw: Any) -> dict[str, Any]:
    return ADAPTERS[protocol]._build_body(request, profile(protocol, **kw))


def user(content: str = "hi", **kw: Any) -> ModelRequest:
    return ModelRequest(
        profile_name="p", messages=[ModelMessage(role=MessageRole.USER, content=content, **kw)]
    )


class FakeSSE:
    """Minimal stand-in for httpx.Response.aiter_lines over an SSE body."""

    def __init__(self, body: str) -> None:
        self._body = body

    async def aiter_lines(self) -> Any:
        for line in self._body.split("\n"):
            yield line

    async def aiter_bytes(self) -> Any:
        yield self._body.encode()


def drain(parser: Any, sse: str) -> Any:
    async def run() -> Any:
        assembler = StreamAssembler()
        async for delta in parser(FakeSSE(sse)):
            assembler.update(delta)
        return assembler.build("probe", "m-1")

    return asyncio.run(run())


# request shaping


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_system_prompt_survives_every_protocol(protocol: Protocol) -> None:
    request = ModelRequest(profile_name="p", system="SENTINEL", messages=user().messages)
    assert "SENTINEL" in json.dumps(build(protocol, request))


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_system_role_message_equals_system_field(protocol: Protocol) -> None:
    """Callers may express the system prompt either way; the wire must not care."""
    as_field = build(protocol, ModelRequest(profile_name="p", system="S", messages=user().messages))
    as_message = build(
        protocol,
        ModelRequest(
            profile_name="p",
            messages=[ModelMessage(role=MessageRole.SYSTEM, content="S"), *user().messages],
        ),
    )
    assert as_field == as_message


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_max_output_tokens_reaches_the_wire(protocol: Protocol) -> None:
    body = build(
        protocol,
        ModelRequest(profile_name="p", max_output_tokens=1234, messages=user().messages),
    )
    assert "1234" in json.dumps(body)


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_tools_stripped_without_capability(protocol: Protocol) -> None:
    body = build(
        protocol,
        ModelRequest(
            profile_name="p",
            messages=user().messages,
            tools=[ToolDefinition(name="t", description="", parameters={})],
        ),
        capabilities=set(),
    )
    assert "tools" not in body


# tool round trip

TOOL_TURN = [
    ModelMessage(role=MessageRole.USER, content="run it"),
    ModelMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=[ToolCall(id="call_1", name="do_thing", arguments={"x": 1})],
    ),
    ModelMessage(role=MessageRole.TOOL, content="RESULT", tool_call_id="call_1"),
]


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_tool_result_reaches_the_wire(protocol: Protocol) -> None:
    body = build(
        protocol,
        ModelRequest(
            profile_name="p",
            messages=TOOL_TURN,
            tools=[ToolDefinition(name="do_thing", description="d", parameters={})],
        ),
    )
    wire = json.dumps(body)
    assert "RESULT" in wire
    assert "do_thing" in wire


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_tool_failure_is_distinguishable_from_success(protocol: Protocol) -> None:
    """A failed tool result must not read as a successful one.

    Anthropic and Gemini flag it structurally; OpenAI's tool role has no such
    field, so the marker goes inline where the model can still see it.
    """
    failed = [*TOOL_TURN[:2], ModelMessage(**{**TOOL_TURN[2].model_dump(), "is_error": True})]
    ok = json.dumps(build(protocol, ModelRequest(profile_name="p", messages=TOOL_TURN)))
    err = json.dumps(build(protocol, ModelRequest(profile_name="p", messages=failed)))
    assert ok != err
    marker = {
        Protocol.OPENAI_CHAT: _TOOL_ERROR_PREFIX,
        Protocol.ANTHROPIC_MESSAGES: '"is_error": true',
        Protocol.GEMINI_GENERATE_CONTENT: '"error"',
    }[protocol]
    assert marker in err and marker not in ok


def test_tool_result_lifts_into_a_message_without_losing_the_flag() -> None:
    lifted = ModelMessage.from_tool_result(
        ToolResultMessage(tool_call_id="c1", content="boom", is_error=True)
    )
    assert (lifted.role, lifted.tool_call_id, lifted.content, lifted.is_error) == (
        MessageRole.TOOL,
        "c1",
        "boom",
        True,
    )


# vision


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_images_reach_the_wire_in_the_protocol_shape(protocol: Protocol) -> None:
    body = build(protocol, user(images=[ImageContent(media_type="image/png", data=PIXEL)]))
    wire = json.dumps(body)
    assert PIXEL in wire
    assert {
        Protocol.OPENAI_CHAT: "image_url",
        Protocol.ANTHROPIC_MESSAGES: "base64",
        Protocol.GEMINI_GENERATE_CONTENT: "inlineData",
    }[protocol] in wire


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_images_stripped_without_vision_capability(protocol: Protocol) -> None:
    """An unsupported feature is dropped deterministically, exactly as tools are,
    rather than sent and rejected at the far end."""
    request = user(images=[ImageContent(media_type="image/png", data=PIXEL)])
    stripped = build(protocol, request, capabilities={Capability.TOOL_USE})
    assert PIXEL not in json.dumps(stripped)
    assert stripped == build(protocol, user(), capabilities={Capability.TOOL_USE})


def test_image_media_type_is_the_three_way_intersection() -> None:
    with pytest.raises(ValidationError):
        ImageContent(media_type="image/tiff", data=PIXEL)


# streaming equivalence

OPENAI_SSE = (
    'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)
ANTHROPIC_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":'
    '{"usage":{"input_tokens":10}}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"lo"}}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":2}}\n\n'
)
GEMINI_SSE = (
    'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n'
    'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],'
    '"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}\n\n'
)
EQUIVALENCE_CASES = {
    Protocol.OPENAI_CHAT: (
        OPENAI_SSE,
        {
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
        _openai_stream_chunks,
    ),
    Protocol.ANTHROPIC_MESSAGES: (
        ANTHROPIC_SSE,
        {
            "content": [{"type": "text", "text": "Hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
        _anthropic_stream_chunks,
    ),
    Protocol.GEMINI_GENERATE_CONTENT: (
        GEMINI_SSE,
        {
            "candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2},
        },
        _gemini_stream_chunks,
    ),
}


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_stream_assembles_to_the_non_streaming_response(protocol: Protocol) -> None:
    sse, raw, parser = EQUIVALENCE_CASES[protocol]
    streamed, parsed = drain(parser, sse), ADAPTERS[protocol]._parse_response(raw, "m-1")
    assert (streamed.text, streamed.finish_reason, streamed.usage) == (
        parsed.text,
        parsed.finish_reason,
        parsed.usage,
    )


THOUGHT_CASES = {
    Protocol.ANTHROPIC_MESSAGES: (
        {
            "content": [
                {"type": "thinking", "thinking": "LEAKED"},
                {"type": "text", "text": "answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {},
        },
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"thinking_delta","thinking":"LEAKED"}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
        '"delta":{"type":"text_delta","text":"answer"}}\n\n'
        'event: message_delta\ndata: {"type":"message_delta",'
        '"delta":{"stop_reason":"end_turn"}}\n\n',
        _anthropic_stream_chunks,
    ),
    Protocol.GEMINI_GENERATE_CONTENT: (
        {
            "candidates": [
                {
                    "content": {"parts": [{"text": "LEAKED", "thought": True}, {"text": "answer"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
        },
        'data: {"candidates":[{"content":{"parts":['
        '{"text":"LEAKED","thought":true},{"text":"answer"}]},"finishReason":"STOP"}]}\n\n',
        _gemini_stream_chunks,
    ),
}


@pytest.mark.parametrize("protocol", list(THOUGHT_CASES))
def test_reasoning_traces_dropped_on_both_transports(protocol: Protocol) -> None:
    """Streamed and completed text for one response must be the same bytes, or a
    JSON-parsing caller sees different answers depending on the transport."""
    raw, sse, parser = THOUGHT_CASES[protocol]
    assert ADAPTERS[protocol]._parse_response(raw, "m-1").text == "answer"
    assert drain(parser, sse).text == "answer"


# response normalization

OK_RESPONSES = {
    Protocol.OPENAI_CHAT: {
        "choices": [{"message": {"content": "x"}, "finish_reason": "wat"}],
        "usage": {},
    },
    Protocol.ANTHROPIC_MESSAGES: {
        "content": [{"type": "text", "text": "x"}],
        "stop_reason": "wat",
        "usage": {},
    },
    Protocol.GEMINI_GENERATE_CONTENT: {
        "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": "WAT"}],
        "usageMetadata": {},
    },
}


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_unknown_finish_reason_degrades_not_crashes(protocol: Protocol) -> None:
    parsed = ADAPTERS[protocol]._parse_response(OK_RESPONSES[protocol], "m-1")
    assert parsed.finish_reason is FinishReason.END_TURN


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_provider_label_matches_the_protocol_registry(protocol: Protocol) -> None:
    """The streaming path labels responses from ``_PROVIDER_LABELS``; the parsers
    hard-code the same strings.  Cost attribution splits if they drift."""
    parsed = ADAPTERS[protocol]._parse_response(OK_RESPONSES[protocol], "m-1")
    assert parsed.provider == _PROVIDER_LABELS[protocol]


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_empty_response_raises_on_every_protocol(protocol: Protocol) -> None:
    empty = {
        Protocol.OPENAI_CHAT: {"choices": []},
        Protocol.ANTHROPIC_MESSAGES: {"content": [], "stop_reason": "end_turn", "usage": {}},
        Protocol.GEMINI_GENERATE_CONTENT: {"candidates": []},
    }[protocol]
    with pytest.raises(ProviderProtocolError) as raised:
        ADAPTERS[protocol]._parse_response(empty, "m-1")
    assert raised.value.status_code == 200


def test_malformed_openai_container_raises_a_typed_protocol_error() -> None:
    with pytest.raises(ProviderProtocolError) as raised:
        OpenAIAdapter()._parse_response({"choices": 1}, "m-1")

    assert (raised.value.provider, raised.value.model, raised.value.field_path) == (
        "openai",
        "m-1",
        "response",
    )


def test_protocol_error_bounds_field_path() -> None:
    error = ProviderProtocolError(
        "malformed", provider="openai", model="m-1", field_path="x" * 1000
    )
    assert len(error.field_path) == 160


MALFORMED_RESPONSES = [
    (Protocol.OPENAI_CHAT, {"choices": {"SECRET": "x"}}),
    (Protocol.OPENAI_CHAT, {"choices": [{"message": []}]}),
    (Protocol.OPENAI_CHAT, {"choices": [{"message": {"tool_calls": 1}}]}),
    (Protocol.ANTHROPIC_MESSAGES, {"content": {"SECRET": "x"}}),
    (Protocol.ANTHROPIC_MESSAGES, {"content": [{"type": "text"}], "usage": 1}),
    (Protocol.GEMINI_GENERATE_CONTENT, {"candidates": {"SECRET": "x"}}),
    (Protocol.GEMINI_GENERATE_CONTENT, {"candidates": [{"content": []}]}),
    (
        Protocol.GEMINI_GENERATE_CONTENT,
        {"candidates": [{"content": {"parts": []}}], "usageMetadata": 1},
    ),
]


@pytest.mark.parametrize(("protocol", "response"), MALFORMED_RESPONSES)
def test_malformed_response_shapes_never_leak_raw_exceptions_or_content(
    protocol: Protocol, response: dict[str, Any]
) -> None:
    with pytest.raises(ProviderProtocolError) as raised:
        ADAPTERS[protocol]._parse_response(response, "m-1")

    assert raised.value.provider == _PROVIDER_LABELS[protocol]
    assert raised.value.model == "m-1"
    assert raised.value.status_code == 200
    assert "SECRET" not in str(raised.value)


@pytest.mark.parametrize(
    ("provider", "parser", "sse"),
    [
        ("openai", _openai_stream_chunks, 'data: {"choices":1}\n\n'),
        (
            "anthropic",
            _anthropic_stream_chunks,
            'event: message_start\ndata: {"type":"message_start","message":[]}\n\n',
        ),
        ("gemini", _gemini_stream_chunks, 'data: {"candidates":1}\n\n'),
    ],
)
def test_malformed_stream_shapes_raise_typed_protocol_errors(
    provider: str, parser: Any, sse: str
) -> None:
    with pytest.raises(ProviderProtocolError) as raised:
        drain(parser, sse)

    assert (raised.value.provider, raised.value.field_path) == (provider, "stream event")
    assert raised.value.status_code == 200


@pytest.mark.parametrize(
    ("provider", "parser"),
    [
        ("openai", _openai_stream_chunks),
        ("anthropic", _anthropic_stream_chunks),
        ("gemini", _gemini_stream_chunks),
    ],
)
def test_malformed_sse_json_is_never_a_silent_terminal(provider: str, parser: Any) -> None:
    with pytest.raises(ProviderProtocolError) as raised:
        drain(parser, "data: {not-json}\n\n")
    assert (raised.value.provider, raised.value.status_code) == (provider, 200)


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
@pytest.mark.parametrize("value", [None, [], {}, 0, "scalar", [None]])
def test_required_response_collection_shape_matrix(protocol: Protocol, value: Any) -> None:
    field = {
        Protocol.OPENAI_CHAT: "choices",
        Protocol.ANTHROPIC_MESSAGES: "content",
        Protocol.GEMINI_GENERATE_CONTENT: "candidates",
    }[protocol]
    with pytest.raises(ProviderProtocolError):
        ADAPTERS[protocol]._parse_response({field: value}, "m-1")


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_unknown_additive_response_fields_are_ignored(protocol: Protocol) -> None:
    original = deepcopy(OK_RESPONSES[protocol])
    extended = deepcopy(original)
    extended["future_extension"] = {"nested": [1, 2, 3]}
    collection = {
        Protocol.OPENAI_CHAT: "choices",
        Protocol.ANTHROPIC_MESSAGES: "content",
        Protocol.GEMINI_GENERATE_CONTENT: "candidates",
    }[protocol]
    extended[collection][0]["future_extension"] = True
    baseline = ADAPTERS[protocol]._parse_response(original, "m-1")
    parsed = ADAPTERS[protocol]._parse_response(extended, "m-1")
    assert parsed == baseline


@pytest.mark.parametrize(
    "parser", [_openai_stream_chunks, _anthropic_stream_chunks, _gemini_stream_chunks]
)
def test_unknown_stream_events_are_ignored(parser: Any) -> None:
    response = drain(
        parser,
        'event: future_event\ndata: {"type":"future_event","future":{"x":1}}\n\n',
    )
    assert (response.text, response.tool_calls, response.finish_reason) == (
        "",
        [],
        FinishReason.END_TURN,
    )


# endpoint escape hatch


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_default_auth_header_per_protocol(protocol: Protocol) -> None:
    name, prefix = _AUTH_HEADERS[protocol]
    assert _build_headers(profile(protocol), "KEY")[name] == f"{prefix}KEY"


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_auth_header_override_sends_the_key_raw(protocol: Protocol) -> None:
    """Azure OpenAI wants ``api-key: <key>`` with no scheme prefix."""
    headers = _build_headers(profile(protocol, auth_header="api-key"), "KEY")
    assert headers["api-key"] == "KEY"
    assert _AUTH_HEADERS[protocol][0] not in headers


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_extra_headers_cannot_displace_the_credential(protocol: Protocol) -> None:
    name, prefix = _AUTH_HEADERS[protocol]
    headers = _build_headers(profile(protocol, extra_headers={"X-Title": "NorthStack"}), "KEY")
    assert headers["X-Title"] == "NorthStack"
    assert headers[name] == f"{prefix}KEY"


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_no_credential_header_without_a_key(protocol: Protocol) -> None:
    assert _AUTH_HEADERS[protocol][0] not in _build_headers(profile(protocol), None)


@pytest.mark.parametrize(
    "header", ["Authorization", "x-api-key", "X-Goog-Api-Key", "api-key", "Cookie"]
)
def test_extra_headers_rejects_credentials(header: str) -> None:
    """A key pasted here would sit in plaintext TOML and ride into every config
    view; api_key_env resolves at call time and is redacted everywhere."""
    with pytest.raises(ValidationError):
        profile(Protocol.OPENAI_CHAT, extra_headers={header: "secret"})


def test_extra_query_appends_to_a_bare_url() -> None:
    url = _with_query(
        _openai_endpoint("https://x.openai.azure.com/openai/deployments/d"),
        {"api-version": "2024-10-21"},
    )
    assert url.endswith("/chat/completions?api-version=2024-10-21")


def test_extra_query_preserves_params_the_endpoint_already_set() -> None:
    url = _with_query(_gemini_endpoint("https://example.test", "m-1", stream=True), {"key2": "v2"})
    assert "alt=sse" in url and url.endswith("&key2=v2")


# structured output

PERMISSIVE_SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}}
STRICT_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
@pytest.mark.parametrize("schema", [PERMISSIVE_SCHEMA, STRICT_SCHEMA])
def test_json_schema_reaches_the_wire_on_every_protocol(
    protocol: Protocol, schema: dict[str, Any]
) -> None:
    request = ModelRequest(profile_name="p", messages=user().messages, output_json_schema=schema)
    assert '"x"' in json.dumps(build(protocol, request))


@pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
def test_json_schema_stripped_without_capability(protocol: Protocol) -> None:
    request = ModelRequest(
        profile_name="p", messages=user().messages, output_json_schema=STRICT_SCHEMA
    )
    assert build(protocol, request, capabilities=set()) == build(
        protocol, user(), capabilities=set()
    )


def test_openai_strict_mode_only_when_the_schema_qualifies() -> None:
    """A schema with optional fields is legal on Anthropic and Gemini and 400s
    under OpenAI strict mode.  Forcing it into strict shape would make those
    fields mandatory, so it is sent non-strict instead."""

    def strict_flag(schema: dict[str, Any]) -> bool:
        request = ModelRequest(
            profile_name="p", messages=user().messages, output_json_schema=schema
        )
        flag = build(Protocol.OPENAI_CHAT, request)["response_format"]["json_schema"]["strict"]
        return bool(flag)

    assert strict_flag(STRICT_SCHEMA)
    assert not strict_flag(PERMISSIVE_SCHEMA)
    assert not strict_flag(
        {
            **STRICT_SCHEMA,
            "properties": {"x": {"type": "string"}, "y": PERMISSIVE_SCHEMA},
            "required": ["x", "y"],
        }
    ), "a nested permissive object must disqualify the whole schema"


def test_token_limit_param_selects_the_wire_spelling() -> None:
    """OpenAI's own reasoning models reject ``max_tokens``; third-party
    OpenAI-compatible gateways want it."""
    default = build(Protocol.OPENAI_CHAT, ModelRequest(profile_name="p", messages=user().messages))
    reasoning = build(
        Protocol.OPENAI_CHAT,
        ModelRequest(profile_name="p", messages=user().messages),
        token_limit_param="max_completion_tokens",
    )
    assert "max_tokens" in default and "max_completion_tokens" not in default
    assert "max_completion_tokens" in reasoning and "max_tokens" not in reasoning
