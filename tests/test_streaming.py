"""Streaming: SSE parsing, delta assembly, gateway stream_complete.

Locks the invariant that a streamed response assembles into the identical
ModelResponse the non-streaming parser produces, and that stream_complete
falls back for non-streaming profiles and stores the same normalized
artifact.  All seams are injected (fake responses, sleep_fn, stub adapters) --
no monkeypatching.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress

import httpx
import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers import gateway as gateway_module
from northstack.adapters.providers.gateway import (
    AuthProviderError,
    ModelGateway,
    OpenAIAdapter,
    ProviderError,
    ProviderProtocolError,
    _anthropic_stream_chunks,
    _gemini_stream_chunks,
    _open_stream_with_retry,
    _openai_stream_chunks,
    assemble_streamed_response,
)
from northstack.adapters.providers.wire import (
    FinishDelta,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextDelta,
    ToolCallDelta,
    Usage,
    UsageDelta,
)
from northstack.config import Capability, ModelProfile, NorthStackConfig, Protocol


# Seams


class FakeStreamResponse:
    """Duck-typed httpx.Response exposing exactly what the streaming path
    consumes: ``status_code``, ``aiter_lines()``, ``aclose()``."""

    def __init__(self, lines: list[str] | None = None, *, error: Exception | None = None):
        self.status_code = 200
        self._lines = lines or []
        self._error = error
        self.closed = False
        self.close_count = 0

    async def aiter_lines(self):  # type: ignore[no-untyped-def]
        for line in self._lines:
            yield line
        if self._error is not None:
            raise self._error

    async def aiter_bytes(self):  # type: ignore[no-untyped-def]
        for line in self._lines:
            yield (line + "\n").encode()
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True
        self.close_count += 1


class FakeStreamClient:
    def __init__(self, response: FakeStreamResponse) -> None:
        self.response = response
        self.request: dict = {}

    def build_request(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        self.request = {"method": method, "url": url, **kwargs}
        return object()

    async def send(self, _request, *, stream=False):  # type: ignore[no-untyped-def]
        assert stream
        return self.response


@asynccontextmanager
async def _local_http_server(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.close()
        await server.wait_closed()


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(OSError, ConnectionError):
        await writer.wait_closed()


def _sse_lines(*events: str) -> list[str]:
    """Split ``data: ...`` blocks on blank lines the way the wire does."""
    lines: list[str] = []
    for event in events:
        lines.extend(event.split("\n"))
        lines.append("")
    return lines


def _profile(name: str = "p", **overrides: object) -> dict:
    base = {
        "name": name,
        "protocol": Protocol.OPENAI_CHAT,
        "base_url": "http://127.0.0.1:9/v1",
        "model": "test-model",
        "max_concurrency": 2,
        "requests_per_minute": 60,
        "capabilities": {Capability.STREAMING, Capability.TOOL_USE},
        "roles": {"worker"},
    }
    return {**base, **overrides}


def _config(*profiles: dict) -> NorthStackConfig:
    return NorthStackConfig.model_validate({"name": "t", "profiles": list(profiles)})


def _request(profile_name: str = "p") -> ModelRequest:
    return ModelRequest(
        profile_name=profile_name, messages=[ModelMessage(role="user", content="go")]
    )


async def _record_sleep(_seconds: float) -> None:
    return


class _StubAdapter:
    """Injected adapter seam: canned deltas without touching the network."""

    def __init__(self, deltas: list | None = None) -> None:
        self._deltas = deltas or []
        self.stream_calls: list[tuple[str, str]] = []

    async def stream(self, request, profile, client, api_key):  # type: ignore[no-untyped-def]
        self.stream_calls.append((request.profile_name, profile.name))
        for d in self._deltas:
            yield d

    async def complete(self, request, profile, client, api_key):  # type: ignore[no-untyped-def]
        raise AssertionError("complete() must not be called for a STREAMING profile")


class _StubCompleteAdapter(_StubAdapter):
    """Fallback-path seam: complete() returns a canned ModelResponse."""

    async def complete(self, request, profile, client, api_key):  # type: ignore[no-untyped-def]
        return ModelResponse(
            text="done",
            finish_reason=FinishReason.END_TURN,
            provider="openai",
            model=profile.model,
        )


def _gateway_with(profile_kwargs: dict, adapter) -> tuple[ModelGateway, NorthStackConfig]:
    cfg = _config({**_profile("p"), **profile_kwargs})
    gw = ModelGateway(cfg)
    gw._adapters[cfg.profiles[0].protocol] = adapter  # type: ignore[assignment]
    return gw, cfg


# SSE event splitting


@pytest.mark.asyncio
async def test_sse_events_splits_openai_chunks() -> None:
    resp = FakeStreamResponse(
        _sse_lines(
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        )
    )
    events = [e async for e in gateway_module._sse_events(resp)]  # type: ignore[arg-type]
    assert len(events) == 3
    assert events[0][1] is not None
    assert events[0][1]["choices"][0]["delta"]["content"] == "Hel"
    assert events[2] == ("", None)  # [DONE]: unparseable sentinel


@pytest.mark.asyncio
async def test_sse_events_handles_anthropic_event_pairs_and_comments() -> None:
    resp = FakeStreamResponse(
        _sse_lines(
            ": keep-alive comment",
            "event: message_start",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}',
            "event: ping",
            "data: {}",
        )
    )
    events = [(n, d) async for n, d in gateway_module._sse_events(resp)]  # type: ignore[arg-type]
    assert events == [
        ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}),
        ("ping", {}),
    ]


@pytest.mark.asyncio
async def test_sse_events_tolerates_missing_trailing_blank_line() -> None:
    resp = FakeStreamResponse(['data: {"a":1}'])  # no terminating blank line
    events = [e async for e in gateway_module._sse_events(resp)]  # type: ignore[arg-type]
    assert events == [("", {"a": 1})]


@pytest.mark.asyncio
async def test_sse_utf8_codepoint_survives_every_byte_boundary() -> None:
    payload = 'data: {"choices":[{"delta":{"content":"😀"}}]}\n\ndata: [DONE]\n\n'.encode()

    class Chunks(httpx.AsyncByteStream):
        def __init__(self, split: int):
            self.split = split

        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield payload[: self.split]
            yield payload[self.split :]

    for split in range(1, len(payload)):
        response = httpx.Response(200, stream=Chunks(split))
        deltas = [delta async for delta in _openai_stream_chunks(response)]
        assert [delta.text for delta in deltas if isinstance(delta, TextDelta)] == ["😀"]
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\r\n", "\r"])
async def test_sse_line_endings_survive_every_byte_boundary(separator: str) -> None:
    payload = separator.join(
        ['data: {"choices":[{"delta":{"content":"ok"}}]}', "", "data: [DONE]", "", ""]
    ).encode()

    class Chunks(httpx.AsyncByteStream):
        def __init__(self, split: int):
            self.split = split

        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield payload[: self.split]
            yield payload[self.split :]

    for split in range(1, len(payload)):
        response = httpx.Response(200, stream=Chunks(split))
        deltas = [delta async for delta in _openai_stream_chunks(response, strict=True)]
        assert [delta.text for delta in deltas if isinstance(delta, TextDelta)] == ["ok"]
        await response.aclose()


@pytest.mark.asyncio
async def test_invalid_utf8_stream_is_protocol_error() -> None:
    class Invalid(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"\xff"}}]}\n\n'

    response = httpx.Response(200, stream=Invalid())
    with pytest.raises(ProviderProtocolError):
        _ = [delta async for delta in _openai_stream_chunks(response)]
    await response.aclose()


# Chunk parsers -> StreamDeltas


@pytest.mark.asyncio
async def test_openai_chunks_yield_text_usage_and_finish() -> None:
    resp = FakeStreamResponse(
        _sse_lines(
            'data: {"choices":[{"delta":{"role":"assistant"}}]}',
            'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    },
                }
            ),
            "data: [DONE]",
        )
    )
    deltas = [d async for d in _openai_stream_chunks(resp)]  # type: ignore[arg-type]
    texts = [d.text for d in deltas if isinstance(d, TextDelta)]
    finishes = [d for d in deltas if isinstance(d, FinishDelta)]
    usages = [d for d in deltas if isinstance(d, UsageDelta)]
    assert texts == ["Hello"]
    assert len(finishes) == 1 and finishes[0].finish_reason == FinishReason.END_TURN
    assert usages[-1].usage.input_tokens == 7  # 10 prompt - 3 cached read
    assert usages[-1].usage.output_tokens == 4


@pytest.mark.asyncio
async def test_openai_tool_call_fragments_carry_name_by_index() -> None:
    resp = FakeStreamResponse(
        _sse_lines(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"fs_read","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"{\\"path\\": "}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"\\"a.txt\\"}"}}]}}]}',
            "data: [DONE]",
        )
    )
    deltas = [d async for d in _openai_stream_chunks(resp)]  # type: ignore[arg-type]
    tool_deltas = [d for d in deltas if isinstance(d, ToolCallDelta)]
    assert tool_deltas[0].id == "call_1" and tool_deltas[0].name == "fs_read"
    joined = "".join(d.arguments_fragment for d in tool_deltas)
    assert json.loads(joined) == {"path": "a.txt"}
    # Name carried forward onto later fragments of the same index.
    assert all(d.name == "fs_read" for d in tool_deltas)


@pytest.mark.asyncio
async def test_anthropic_chunks_merge_usage_across_events() -> None:
    resp = FakeStreamResponse(
        _sse_lines(
            "event: message_start",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":12,'
            '"cache_creation_input_tokens":2,"cache_read_input_tokens":3}}}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":'
            '{"type":"tool_use","id":"tcu_1","name":"fs_read"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"hi"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"input_json_delta","partial_json":"{\\"x\\":1}"}}',
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
            '"usage":{"output_tokens":6}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        )
    )
    deltas = [d async for d in _anthropic_stream_chunks(resp)]  # type: ignore[arg-type]
    kinds = [d.type for d in deltas]
    assert kinds == ["tool_call", "text", "tool_call", "usage", "finish"]
    usage_delta = next(d for d in deltas if isinstance(d, UsageDelta))
    assert usage_delta.usage.input_tokens == 12
    assert usage_delta.usage.cache_creation_tokens == 2
    assert usage_delta.usage.cache_read_tokens == 3
    assert usage_delta.usage.output_tokens == 6
    assert deltas[-1].finish_reason == FinishReason.TOOL_USE


@pytest.mark.asyncio
async def test_gemini_chunks_emit_whole_tool_calls_at_running_indices() -> None:
    """Gemini never fragments a functionCall, so each one is a complete delta."""
    resp = FakeStreamResponse(
        _sse_lines(
            'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}',
            'data: {"candidates":[{"content":{"parts":['
            '{"functionCall":{"name":"fs_read","args":{"path":"a"}}},'
            '{"functionCall":{"name":"fs_list","args":{}}}]}}]}',
            'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}],'
            '"usageMetadata":{"promptTokenCount":30,"candidatesTokenCount":4,'
            '"thoughtsTokenCount":6,"cachedContentTokenCount":10}}',
        )
    )
    deltas = [d async for d in _gemini_stream_chunks(resp)]  # type: ignore[arg-type]
    assert [d.type for d in deltas] == ["text", "tool_call", "tool_call", "usage", "finish"]
    tool_deltas = [d for d in deltas if isinstance(d, ToolCallDelta)]
    assert [d.index for d in tool_deltas] == [0, 1]
    assert [d.name for d in tool_deltas] == ["fs_read", "fs_list"]
    assert json.loads(tool_deltas[0].arguments_fragment) == {"path": "a"}
    usage_delta = next(d for d in deltas if isinstance(d, UsageDelta))
    assert usage_delta.usage == Usage(input_tokens=20, output_tokens=10, cache_read_tokens=10)
    assert deltas[-1].finish_reason == FinishReason.END_TURN


@pytest.mark.asyncio
async def test_gemini_safety_stop_maps_to_error_finish() -> None:
    resp = FakeStreamResponse(
        _sse_lines('data: {"candidates":[{"content":{"parts":[]},"finishReason":"SAFETY"}]}')
    )
    deltas = [d async for d in _gemini_stream_chunks(resp)]  # type: ignore[arg-type]
    assert deltas[-1].finish_reason == FinishReason.ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parser", "lines"),
    [
        (_openai_stream_chunks, ['data: {"choices":[{"delta":{"content":"x"}}]}', ""]),
        (
            _anthropic_stream_chunks,
            [
                "event: content_block_delta",
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}',
                "",
            ],
        ),
        (
            _gemini_stream_chunks,
            ['data: {"candidates":[{"content":{"parts":[{"text":"x"}]}}]}', ""],
        ),
    ],
)
async def test_strict_stream_rejects_clean_eof_without_terminal(parser, lines) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ProviderProtocolError) as caught:
        _ = [delta async for delta in parser(FakeStreamResponse(lines), strict=True)]
    assert caught.value.field_path == "stream.terminal"


@pytest.mark.asyncio
async def test_anthropic_ignores_data_after_terminal() -> None:
    response = FakeStreamResponse(
        _sse_lines(
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"leak"}}',
        )
    )
    deltas = [delta async for delta in _anthropic_stream_chunks(response, strict=True)]
    assert not [delta for delta in deltas if isinstance(delta, TextDelta)]


@pytest.mark.asyncio
async def test_openai_ignores_data_after_terminal() -> None:
    response = FakeStreamResponse(
        _sse_lines(
            "data: [DONE]",
            'data: {"choices":[{"delta":{"content":"leak"}}]}',
        )
    )
    deltas = [delta async for delta in _openai_stream_chunks(response, strict=True)]
    assert not [delta for delta in deltas if isinstance(delta, TextDelta)]


@pytest.mark.asyncio
async def test_gemini_ignores_data_after_terminal() -> None:
    response = FakeStreamResponse(
        _sse_lines(
            'data: {"candidates":[{"content":{"parts":[]},"finishReason":"STOP"}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"leak"}]}}]}',
        )
    )
    deltas = [delta async for delta in _gemini_stream_chunks(response, strict=True)]
    assert not [delta for delta in deltas if isinstance(delta, TextDelta)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parser", "lines"),
    [
        (_openai_stream_chunks, _sse_lines("data: [DONE]", "data: [DONE]")),
        (
            _anthropic_stream_chunks,
            _sse_lines(
                'event: message_stop\ndata: {"type":"message_stop"}',
                'event: message_stop\ndata: {"type":"message_stop"}',
            ),
        ),
        (
            _gemini_stream_chunks,
            _sse_lines(
                'data: {"candidates":[{"finishReason":"STOP"}]}',
                'data: {"candidates":[{"finishReason":"STOP"}]}',
            ),
        ),
    ],
)
async def test_duplicate_terminal_frames_emit_one_finish(parser, lines) -> None:  # type: ignore[no-untyped-def]
    deltas = [delta async for delta in parser(FakeStreamResponse(lines), strict=True)]
    assert len([delta for delta in deltas if isinstance(delta, FinishDelta)]) == 1


# Assembler equivalence: streamed == non-streamed ModelResponse


def _openai_nonstream_body() -> dict:
    return {
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Answer text",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "fs_read", "arguments": '{"path": "a.txt"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 6},
        },
    }


def _equivalent_stream_lines() -> list[str]:
    """SSE chunks whose assembly must equal the non-streamed parse above."""
    pieces: list[dict] = [
        {"choices": [{"delta": {"content": "Answer"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " text"}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_x",
                                "type": "function",
                                "function": {"name": "fs_read", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"path": '}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 6},
            },
        },
    ]
    lines: list[str] = []
    for piece in pieces:
        lines.append("data: " + json.dumps(piece))
        lines.append("")
    lines.append("data: [DONE]")
    return lines


@pytest.mark.asyncio
async def test_streamed_assembly_equals_nonstreamed_parse() -> None:
    adapter = OpenAIAdapter()
    nonstreamed = adapter._parse_response(_openai_nonstream_body(), "test-model")
    streamed = await assemble_streamed_response(
        _openai_stream_chunks(FakeStreamResponse(_equivalent_stream_lines())),  # type: ignore[arg-type]
        "openai",
        "test-model",
    )

    assert streamed.text == nonstreamed.text
    assert streamed.finish_reason == nonstreamed.finish_reason
    assert streamed.usage.model_dump() == nonstreamed.usage.model_dump()
    assert streamed.provider == nonstreamed.provider == "openai"
    assert streamed.model == nonstreamed.model == "test-model"
    assert [tc.model_dump() for tc in streamed.tool_calls] == [
        tc.model_dump() for tc in nonstreamed.tool_calls
    ]


@pytest.mark.asyncio
async def test_stream_without_finish_or_usage_degrades_like_parser() -> None:
    """A stream cut short still yields END_TURN + zero usage -- no crash,
    matching the parser's defaults."""
    resp = FakeStreamResponse(_sse_lines('data: {"choices":[{"delta":{"content":"partial"}}]}'))
    response = await assemble_streamed_response(
        _openai_stream_chunks(resp),
        "openai",
        "test-model",  # type: ignore[arg-type]
    )
    assert response.text == "partial"
    assert response.finish_reason == FinishReason.END_TURN
    assert response.usage == Usage()


@pytest.mark.asyncio
async def test_undecodable_tool_args_degrade_to_empty_dict() -> None:
    resp = FakeStreamResponse(
        _sse_lines(
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
            '"function":{"name":"t","arguments":""}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"not json"}}]}}]}',
        )
    )
    response = await assemble_streamed_response(
        _openai_stream_chunks(resp),
        "openai",
        "test-model",  # type: ignore[arg-type]
    )
    assert response.tool_calls[0].arguments == {}


# Gateway-level stream_complete


@pytest.mark.asyncio
async def test_gateway_falls_back_for_non_streaming_profile() -> None:
    plain = _profile("p", capabilities={Capability.TOOL_USE})
    cfg = _config(plain)
    gw = ModelGateway(cfg)
    adapter = _StubCompleteAdapter([])
    gw._adapters[cfg.profiles[0].protocol] = adapter  # type: ignore[assignment]

    deltas = [d async for d in gw.stream_complete(_request("p"))]

    assert [(d.type, getattr(d, "finish_reason", None)) for d in deltas] == [
        ("usage", None),
        ("finish", FinishReason.END_TURN),
    ]
    assert not adapter.stream_calls


@pytest.mark.asyncio
async def test_gateway_streams_deltas_under_capability() -> None:
    adapter = _StubAdapter(
        [
            TextDelta(text="chunk-1"),
            UsageDelta(usage=Usage(input_tokens=3, output_tokens=2)),
            FinishDelta(finish_reason=FinishReason.END_TURN),
        ]
    )
    gw, _cfg = _gateway_with({}, adapter)

    seen = [d async for d in gw.stream_complete(_request("p"))]

    assert [d.type for d in seen] == ["text", "usage", "finish"]
    assert adapter.stream_calls == [("p", "p")]


@pytest.mark.asyncio
async def test_gateway_stores_artifact_for_streamed_response(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    adapter = _StubAdapter([TextDelta(text="hello")])
    gw, _cfg = _gateway_with({}, adapter)
    gw._artifact_store = store

    _deltas = [d async for d in gw.stream_complete(_request("p"))]

    blobs = [
        p for shard in (tmp_path / "artifacts").iterdir() if shard.is_dir() for p in shard.iterdir()
    ]
    assert len(blobs) == 1
    stored = json.loads(blobs[0].read_text(encoding="utf-8"))
    assert stored["provider"] == "openai"
    assert stored["text"] == "hello"


@pytest.mark.asyncio
async def test_gateway_supports_streaming_reflects_capability() -> None:
    gw_yes, cfg = _gateway_with({}, _StubAdapter())
    plain_cfg = _config(_profile("q", capabilities=set()))
    gw_no = ModelGateway(plain_cfg)
    assert gw_yes.supports_streaming(cfg.profiles[0].name)
    assert not gw_no.supports_streaming(plain_cfg.profiles[0].name)


# complete_stream: consumer opt-in path


@pytest.mark.asyncio
async def test_complete_stream_returns_response_and_reports_deltas() -> None:
    deltas_seen: list[str] = []
    adapter = _StubAdapter(
        [
            TextDelta(text="answer"),
            UsageDelta(usage=Usage(input_tokens=5, output_tokens=1)),
            FinishDelta(finish_reason=FinishReason.END_TURN),
        ]
    )
    gw, _cfg = _gateway_with({}, adapter)

    response = await gw.complete_stream(
        _request("p"), on_delta=lambda d: deltas_seen.append(d.type)
    )

    assert response.text == "answer"
    assert response.finish_reason == FinishReason.END_TURN
    assert response.usage.input_tokens == 5
    assert deltas_seen == ["text", "usage", "finish"]


@pytest.mark.asyncio
async def test_complete_stream_stamps_artifact_digest(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    gw, _cfg = _gateway_with({}, _StubAdapter([TextDelta(text="hi")]))
    gw._artifact_store = store

    response = await gw.complete_stream(_request("p"))

    assert response.response_artifact_id is not None
    stored = store.read_by_digest(response.response_artifact_id)
    assert json.loads(stored)["text"] == "hi"


@pytest.mark.asyncio
async def test_complete_stream_falls_back_to_complete_without_capability() -> None:
    plain = _profile("p", capabilities={Capability.TOOL_USE})
    cfg = _config(plain)
    gw = ModelGateway(cfg)

    class _RecordingCompleteAdapter(_StubAdapter):
        def __init__(self) -> None:
            super().__init__([])
            self.complete_calls: list[str] = []

        async def complete(self, request, profile, client, api_key):  # type: ignore[no-untyped-def]
            self.complete_calls.append(request.profile_name)
            return ModelResponse(
                text="nonstreamed",
                finish_reason=FinishReason.END_TURN,
                provider="openai",
                model=profile.model,
            )

    adapter = _RecordingCompleteAdapter()
    gw._adapters[cfg.profiles[0].protocol] = adapter  # type: ignore[assignment]

    response = await gw.complete_stream(_request("p"), on_delta=lambda _d: None)

    assert response.text == "nonstreamed"
    assert adapter.complete_calls == ["p"]


# Retry-until-first-byte on the streaming open path


class _OpenResponse:
    def __init__(self, status_code: int, close_error: Exception | None = None) -> None:
        self.status_code = status_code
        self.close_error = close_error
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1
        if self.close_error:
            raise self.close_error


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_open_stream_retries_transient_then_returns_200(status: int) -> None:
    profile = ModelProfile(**_profile())
    responses = [_OpenResponse(status), _OpenResponse(200)]
    attempts: list[int] = []

    async def _open() -> httpx.Response:
        attempts.append(1)
        return responses[len(attempts) - 1]  # type: ignore[return-value]

    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    resp = await _open_stream_with_retry(_open, profile, "openai", "test-model", sleep_fn=_sleep)
    assert resp.status_code == 200
    assert len(attempts) == 2
    assert responses[0].close_count == 1
    assert sleeps == [1.5]


@pytest.mark.asyncio
async def test_open_stream_exhausts_retries_on_connect_error() -> None:
    profile = ModelProfile(**_profile())
    attempts: list[int] = []

    async def _open() -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("nope")

    with pytest.raises(ProviderError):
        await _open_stream_with_retry(
            _open, profile, "openai", "test-model", sleep_fn=_record_sleep
        )
    assert len(attempts) == profile.transport_retries + 1


@pytest.mark.asyncio
async def test_open_stream_401_is_terminal_without_retry() -> None:
    profile = ModelProfile(**_profile(transport_retries=3))
    attempts: list[int] = []
    response = _OpenResponse(401)

    async def _open() -> httpx.Response:
        attempts.append(1)
        return response  # type: ignore[return-value]

    with pytest.raises(AuthProviderError):
        await _open_stream_with_retry(
            _open, profile, "openai", "test-model", sleep_fn=_record_sleep
        )
    assert len(attempts) == 1
    assert response.close_count == 1


@pytest.mark.asyncio
async def test_open_stream_401_preserves_auth_error_when_close_fails() -> None:
    profile = ModelProfile(**_profile())
    response = _OpenResponse(401, RuntimeError("close failed"))

    async def _open() -> httpx.Response:
        return response  # type: ignore[return-value]

    with pytest.raises(AuthProviderError):
        await _open_stream_with_retry(_open, profile, "openai", "test-model")
    assert response.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [201, 400, 403, 429, 500])
async def test_open_stream_non_200_closes_exactly_once(status: int) -> None:
    profile = ModelProfile(**_profile(transport_retries=0))
    response = _OpenResponse(status)

    async def _open() -> httpx.Response:
        return response  # type: ignore[return-value]

    with pytest.raises(ProviderError):
        await _open_stream_with_retry(_open, profile, "openai", "test-model")
    assert response.close_count == 1


@pytest.mark.asyncio
async def test_openai_yields_before_the_response_body_finishes() -> None:
    started, release = asyncio.Event(), asyncio.Event()

    class DelayedBody(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            started.set()
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            await release.wait()
            yield b"data: [DONE]\n\n"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=DelayedBody())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stream = gateway_module._stream_openai(_request(), ModelProfile(**_profile()), client, None)
    first = asyncio.create_task(anext(stream))
    await started.wait()
    try:
        delta = await asyncio.wait_for(asyncio.shield(first), 0.1)
        assert isinstance(delta, TextDelta) and delta.text == "first"
    finally:
        release.set()
        await first
        await stream.aclose()
        await client.aclose()


@pytest.mark.asyncio
async def test_real_http_stream_yields_before_tail_arrives() -> None:
    first_sent, release = asyncio.Event(), asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
            b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
        )
        await writer.drain()
        first_sent.set()
        await release.wait()
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()
        await _close_writer(writer)

    async with _local_http_server(handler) as base_url:
        async with httpx.AsyncClient() as client:
            stream = gateway_module._stream_openai(
                _request(), ModelProfile(**_profile(base_url=base_url)), client, None
            )
            first_task = asyncio.create_task(anext(stream))
            await first_sent.wait()
            first = await asyncio.wait_for(first_task, 0.5)
            assert isinstance(first, TextDelta) and first.text == "first"
            release.set()
            assert isinstance(await asyncio.wait_for(anext(stream), 0.5), FinishDelta)
            await stream.aclose()


@pytest.mark.asyncio
async def test_stream_cancellation_before_headers_cancels_send() -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    class HeaderClient:
        def build_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return object()

        async def send(self, _request, *, stream=False):  # type: ignore[no-untyped-def]
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    stream = gateway_module._stream_openai(
        _request(),
        ModelProfile(**_profile()),
        HeaderClient(),
        None,  # type: ignore[arg-type]
    )
    task = asyncio.create_task(anext(stream))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_real_http_cancellation_before_headers_closes_connection() -> None:
    request_read, closed = asyncio.Event(), asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        request_read.set()
        await reader.read()
        closed.set()
        await _close_writer(writer)

    async with _local_http_server(handler) as base_url:
        async with httpx.AsyncClient() as client:
            stream = gateway_module._stream_openai(
                _request(), ModelProfile(**_profile(base_url=base_url)), client, None
            )
            task = asyncio.create_task(anext(stream))
            await request_read.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), 1)


@pytest.mark.asyncio
async def test_stream_cancellation_between_deltas_closes_response() -> None:
    waiting = asyncio.Event()

    class WaitingResponse(FakeStreamResponse):
        async def aiter_bytes(self):  # type: ignore[no-untyped-def]
            yield b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
            waiting.set()
            await asyncio.Event().wait()

    response = WaitingResponse()
    stream = gateway_module._stream_openai(
        _request(),
        ModelProfile(**_profile()),
        FakeStreamClient(response),
        None,  # type: ignore[arg-type]
    )
    assert isinstance(await anext(stream), TextDelta)
    task = asyncio.create_task(anext(stream))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.close_count == 1


@pytest.mark.asyncio
async def test_real_http_cancellation_between_deltas_closes_connection() -> None:
    first_sent, closed = asyncio.Event(), asyncio.Event()

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
            b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n'
        )
        await writer.drain()
        first_sent.set()
        await reader.read()
        closed.set()
        await _close_writer(writer)

    async with _local_http_server(handler) as base_url:
        async with httpx.AsyncClient() as client:
            stream = gateway_module._stream_openai(
                _request(), ModelProfile(**_profile(base_url=base_url)), client, None
            )
            first_task = asyncio.create_task(anext(stream))
            await first_sent.wait()
            assert isinstance(await first_task, TextDelta)
            task = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(closed.wait(), 1)


@pytest.mark.asyncio
async def test_response_closes_before_terminal_delta_is_delivered() -> None:
    response = FakeStreamResponse(_sse_lines("data: [DONE]"))
    stream = gateway_module._stream_openai(
        _request(),
        ModelProfile(**_profile()),
        FakeStreamClient(response),
        None,  # type: ignore[arg-type]
    )
    terminal = await anext(stream)
    assert isinstance(terminal, FinishDelta)
    assert response.close_count == 1
    await stream.aclose()
    assert response.close_count == 1


@pytest.mark.asyncio
async def test_mid_stream_failure_is_terminal_provider_error() -> None:
    """Once headers arrive, a dropped connection surfaces ProviderError --
    never a silent partial answer and never another attempt."""
    profile = ModelProfile(**_profile())
    resp = FakeStreamResponse(
        _sse_lines('data: {"choices":[{"delta":{"content":"half"}}]}'),
        error=httpx.StreamError("connection reset"),
    )

    deltas = []
    with pytest.raises(ProviderError):
        async for delta in gateway_module._stream_openai(
            _request(),
            profile,
            FakeStreamClient(resp),
            None,  # type: ignore[arg-type]
        ):
            deltas.append(delta)
    # The one delta delivered before the drop was still yielded...
    assert [d.text for d in deltas if isinstance(d, TextDelta)] == ["half"]
    # ...and the half-consumed response was closed exactly once.
    assert resp.closed


@pytest.mark.asyncio
async def test_gemini_stream_posts_to_sse_endpoint_with_header_key() -> None:
    """The key rides the x-goog-api-key header; the URL stays secret-free."""
    profile = ModelProfile(
        **_profile(
            protocol=Protocol.GEMINI_GENERATE_CONTENT,
            base_url="https://generativelanguage.googleapis.com",
            model="gemini-3.5-flash-lite",
        )
    )
    resp = FakeStreamResponse(
        _sse_lines(
            'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}'
        )
    )
    client = FakeStreamClient(resp)

    deltas = [
        d
        async for d in gateway_module._stream_gemini(
            _request(),
            profile,
            client,  # type: ignore[arg-type]
            "sekret",
        )
    ]

    assert client.request["url"].endswith(
        "/v1beta/models/gemini-3.5-flash-lite:streamGenerateContent?alt=sse"
    )
    assert client.request["headers"]["x-goog-api-key"] == "sekret"
    assert "sekret" not in client.request["url"]
    assert [d.text for d in deltas if isinstance(d, TextDelta)] == ["ok"]
    assert resp.closed


@pytest.mark.asyncio
async def test_gemini_mid_stream_failure_is_terminal_provider_error() -> None:
    profile = ModelProfile(
        **_profile(
            protocol=Protocol.GEMINI_GENERATE_CONTENT,
            base_url="https://generativelanguage.googleapis.com",
        )
    )
    resp = FakeStreamResponse(
        _sse_lines('data: {"candidates":[{"content":{"parts":[{"text":"half"}]}}]}'),
        error=httpx.StreamError("connection reset"),
    )

    with pytest.raises(ProviderError):
        async for _delta in gateway_module._stream_gemini(
            _request(),
            profile,
            FakeStreamClient(resp),  # type: ignore[arg-type]
            None,
        ):
            pass
    assert resp.closed


# Limiter held across the stream lifecycle


@pytest.mark.asyncio
async def test_concurrency_slot_is_held_until_last_delta() -> None:
    """Two concurrent streams on a max_concurrency=1 profile serialize -- the
    slot spans open-to-close, not open-to-first-delta."""
    import asyncio
    import time

    cfg = _config(_profile("solo", max_concurrency=1))
    gw = ModelGateway(cfg)

    class _GatedAdapter:
        async def stream(self, request, profile, client, api_key):  # type: ignore[no-untyped-def]
            yield TextDelta(text="x")
            await asyncio.sleep(0.05)  # hold the slot mid-stream
            yield FinishDelta(finish_reason=FinishReason.END_TURN)

    gw._adapters[cfg.profiles[0].protocol] = _GatedAdapter()  # type: ignore[assignment]

    async def collect() -> list[str]:
        return [d.type async for d in gw.stream_complete(_request("solo"))]

    start = time.perf_counter()
    first, second = await asyncio.gather(collect(), collect())
    elapsed = time.perf_counter() - start

    assert first == ["text", "finish"] and second == ["text", "finish"]
    # Serialized streams each sleep 0.05s => total >= 0.1s; an overlapping run
    # (slot released early) would finish in ~0.05s.
    assert elapsed >= 0.09
