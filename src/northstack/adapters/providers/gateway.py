"""ModelGateway, protocol adapters, and rate limiting.

Public seam:
  - ModelGateway(config, artifact_store?) -> complete(ModelRequest) -> ModelResponse
  - ModelLimiter.run(profile, callable) -> structurally enforces concurrency + RPM
  - ProviderError / HTTPProviderError / AuthProviderError -> typed exception hierarchy

Adapters are selected from ModelProfile.protocol.  Secrets are resolved only
at request time and never appear in repr, event payloads, artifacts, exceptions,
or subprocess environments.

Process-local limitation: ModelLimiter instances are process-local.  Each
ModelGateway owns a shared limiter pool keyed by profile name, so every
request for the same profile within one gateway shares one semaphore and
RPM window.  Gateways in separate processes do NOT share rate limits.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, ParamSpec, TypeVar
from typing import Protocol as TypingProtocol
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlsplit, urlunsplit

import httpx

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.wire import (
    FinishDelta,
    FinishReason,
    ImageContent,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    StreamDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    Usage,
    UsageDelta,
)
from northstack.config import Capability, ModelProfile, NorthStackConfig, Protocol
from northstack.domain.url_policy import validate_provider_url

T = TypeVar("T")
P = ParamSpec("P")

logger = logging.getLogger(__name__)

MAX_SCHEMA_BYTES = 64 * 1024
MAX_INPUT_BYTES = 1024 * 1024
_MAX_ERROR_BODY_LEN = 512
_MAX_PROTOCOL_FIELD_PATH = 160


class ProviderError(Exception):
    """Base error for model-provider failures."""

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model

    def is_transient(self) -> bool:
        return True


class ProviderProtocolError(ProviderError):
    """Malformed successful response from a provider protocol."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        model: str = "",
        field_path: str = "response",
        status_code: int = 200,
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.field_path = field_path[:_MAX_PROTOCOL_FIELD_PATH]
        self.status_code = status_code

    def is_transient(self) -> bool:
        return False


def _protocol_parser(provider: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorate(parser: Callable[P, T]) -> Callable[P, T]:
        @wraps(parser)
        def normalized(*args: P.args, **kwargs: P.kwargs) -> T:
            model = str(kwargs.get("model") or (args[2] if len(args) > 2 else ""))
            try:
                return parser(*args, **kwargs)
            except ProviderError:
                raise
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise ProviderProtocolError(
                    f"Malformed {provider} response at response",
                    provider=provider,
                    model=model,
                ) from error

        return normalized

    return decorate


def _protocol_stream(
    provider: str,
) -> Callable[
    [Callable[..., AsyncIterator[StreamDelta]]], Callable[..., AsyncIterator[StreamDelta]]
]:
    def decorate(
        parser: Callable[..., AsyncIterator[StreamDelta]],
    ) -> Callable[..., AsyncIterator[StreamDelta]]:
        @wraps(parser)
        async def normalized(
            *args: Any, model: str = "", **kwargs: Any
        ) -> AsyncIterator[StreamDelta]:
            try:
                async for delta in parser(*args, **kwargs):
                    yield delta
            except ProviderError:
                raise
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise ProviderProtocolError(
                    f"Malformed {provider} stream event",
                    provider=provider,
                    model=model,
                    field_path="stream event",
                ) from error

        return normalized

    return decorate


class HTTPProviderError(ProviderError):
    """HTTP-level failure from a model provider.

    response_body is NEVER retained.  Only status_code and a safe message are
    exposed -- the message may carry the provider's own ``error.message``,
    redacted and length-bounded, because a bare status code leaves a transient
    failure unattributable.  A redacted artifact is stored if an ArtifactStore
    is available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        provider: str = "",
        model: str = "",
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.status_code = status_code

    def is_transient(self) -> bool:
        if self.status_code in _TRANSIENT_STATUS:
            return True
        return self.status_code == 413 and bool(_RATE_LIMIT_BODY.search(str(self)))


class AuthProviderError(ProviderError):
    """Authentication failure -- bad or missing API key."""

    def is_transient(self) -> bool:
        return False


class CapabilityError(ProviderError):
    """Requested feature not supported by the profile."""

    def is_transient(self) -> bool:
        return False


class ProviderConfigurationError(ProviderError):
    """Invalid or missing provider configuration."""

    def is_transient(self) -> bool:
        return False


def _check_capabilities(
    request: ModelRequest,
    profile: ModelProfile,
) -> list[str]:
    """Return a list of unsupported feature warnings.

    Features that are unsupported but requested are either rejected or
    transformed deterministically (e.g. JSON-Schema fallback).
    """
    warnings: list[str] = []

    if (
        request.output_json_schema is not None
        and Capability.NATIVE_JSON_SCHEMA not in profile.capabilities
    ):
        warnings.append(
            "output_json_schema requested but lacks native_json_schema capability; "
            "will use prompt-based JSON output with validation"
        )

    if request.seed is not None and profile.protocol == Protocol.ANTHROPIC_MESSAGES:
        warnings.append(
            "seed requested but Anthropic protocol does not support seed; "
            "it will be omitted from the request"
        )

    if request.tools and Capability.TOOL_USE not in profile.capabilities:
        warnings.append(
            "tools requested but profile lacks TOOL_USE capability; "
            "tools will be stripped from the request"
        )

    if any(m.images for m in request.messages) and Capability.VISION not in profile.capabilities:
        warnings.append(
            "images attached but profile lacks VISION capability; "
            "images will be stripped from the request"
        )

    return warnings


def _should_send_json_schema(request: ModelRequest, profile: ModelProfile) -> bool:
    """Return True if we can send native JSON-Schema output format."""
    return (
        request.output_json_schema is not None
        and Capability.NATIVE_JSON_SCHEMA in profile.capabilities
    )


def _strict_ready(schema: Any) -> bool:
    """Whether OpenAI's strict json_schema mode accepts the schema unmodified.

    Strict requires every object to close ``additionalProperties`` and list
    every property in ``required``.  Rewriting the schema to fit would silently
    make the caller's optional fields mandatory, so a schema that does not
    already qualify is sent non-strict -- the same schema Anthropic and Gemini
    accept must not 400 on OpenAI alone.
    """
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object" and (
        schema.get("additionalProperties") is not False
        or set(schema.get("properties") or {}) != set(schema.get("required") or [])
    ):
        return False
    nested = [
        *(
            v
            for k in ("properties", "$defs", "definitions")
            for v in (schema.get(k) or {}).values()
        ),
        *(v for k in ("anyOf", "allOf", "oneOf", "prefixItems") for v in (schema.get(k) or [])),
        schema.get("items"),
    ]
    return all(_strict_ready(node) for node in nested)


_GEMINI_SCHEMA_FIELDS = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "maxItems",
        "minItems",
        "properties",
        "required",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
        "pattern",
        "example",
        "anyOf",
        "propertyOrdering",
        "default",
        "items",
        "minimum",
        "maximum",
    }
)

_GEMINI_STRUCTURAL_KEYWORDS = frozenset(
    {"$ref", "$defs", "definitions", "allOf", "oneOf", "not", "prefixItems", "const"}
)


def _gemini_needs_json_schema(schema: Any) -> bool:
    """Whether ``schema`` uses a keyword the Gemini Schema subset cannot express.

    Recursion only follows the containers the subset itself defines
    (``properties``, ``anyOf``, ``items``); every other container --
    ``$defs``, ``allOf``, ``prefixItems`` -- is itself a structural keyword and
    so is caught before the walk needs to descend into it.
    """
    if not isinstance(schema, Mapping):
        return False
    if _GEMINI_STRUCTURAL_KEYWORDS & schema.keys():
        return True
    properties = schema.get("properties")
    any_of = schema.get("anyOf")
    children = [
        *(properties.values() if isinstance(properties, Mapping) else ()),
        *(any_of if isinstance(any_of, list) else ()),
        schema.get("items"),
    ]
    return any(_gemini_needs_json_schema(node) for node in children)


def _gemini_schema_subset(schema: Any) -> Any:
    """Project ``schema`` onto the fields Gemini's Schema type accepts.

    Only ever drops constraints -- ``additionalProperties: False`` is the one
    that matters in practice, and the tool boundary validates arguments anyway,
    so a loosened advertisement cannot admit a call the workspace would honour.
    """
    if not isinstance(schema, Mapping):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_FIELDS:
            continue
        if key == "properties" and isinstance(value, Mapping):
            out[key] = {k: _gemini_schema_subset(v) for k, v in value.items()}
        elif key == "anyOf" and isinstance(value, list):
            out[key] = [_gemini_schema_subset(v) for v in value]
        elif key == "items":
            out[key] = _gemini_schema_subset(value)
        else:
            out[key] = value
    return out


def _gemini_tool_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """The FunctionDeclaration parameter field and value for ``schema``.

    Simple schemas go in ``parameters`` (the subset every Gemini-compatible
    endpoint understands); anything needing full JSON Schema goes in
    ``parametersJsonSchema``.  The two fields are mutually exclusive.
    """
    if _gemini_needs_json_schema(schema):
        return {"parametersJsonSchema": schema}
    return {"parameters": _gemini_schema_subset(schema)}


def _visible_request(request: ModelRequest, profile: ModelProfile) -> ModelRequest:
    """The request as the profile may actually receive it.

    Images drop out for a profile without VISION, mirroring how tools drop out
    without TOOL_USE -- an unsupported feature is stripped deterministically
    rather than sent and rejected at the far end.
    """
    if Capability.VISION in profile.capabilities or not any(m.images for m in request.messages):
        return request
    return request.model_copy(
        update={"messages": [m.model_copy(update={"images": []}) for m in request.messages]}
    )


def _tool_result_text(msg: ModelMessage) -> str:
    """Tool-result text with the failure marker OpenAI's tool role cannot carry
    structurally.  Anthropic and Gemini flag the error in the block itself."""
    return f"{_TOOL_ERROR_PREFIX}{msg.content}" if msg.is_error else msg.content


def _anthropic_image_block(img: ImageContent) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": img.media_type, "data": img.data},
    }


def _validate_base_url(url: str, protocol: Protocol) -> None:
    """Backward-compatible syntax validator used by provider unit tests."""
    validate_provider_url(url, credentialed=False)


def _join_url(base: str, path: str) -> str:
    """Join a base URL and a path, ensuring no double slashes."""
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


def _openai_endpoint(base: str) -> str:
    """Build OpenAI chat completions URL from base."""
    return _join_url(base, "chat/completions")


def _anthropic_endpoint(base: str) -> str:
    """Build Anthropic messages URL from base.

    The base_url is the API root (e.g. https://api.anthropic.com).
    We append /v1/messages exactly once.
    """
    base = base.rstrip("/")
    base = base.removesuffix("/v1")
    return _join_url(base, "v1/messages")


def _gemini_endpoint(base: str, model: str, *, stream: bool = False) -> str:
    """Build a Gemini generateContent URL from base and model.

    The base_url is the API root (e.g. https://generativelanguage.googleapis.com);
    the API version segment is appended exactly once.  The model is part of the
    path, not the body, and the API key travels in a header -- never the query
    string, which would leak the secret into logs and error messages.
    """
    base = base.rstrip("/").removesuffix("/v1beta")
    method = "streamGenerateContent" if stream else "generateContent"
    url = _join_url(base, f"v1beta/models/{quote(model, safe='')}:{method}")
    return _with_query(url, {"alt": "sse"}) if stream else url


def _with_query(url: str, extra: Mapping[str, str]) -> str:
    """Append a profile's static query parameters, preserving any the endpoint
    builder already set (Gemini's streaming URL carries ``?alt=sse``)."""
    if not extra:
        return url
    parts = urlsplit(url)
    existing = parse_qsl(parts.query, keep_blank_values=True)
    conflicts = sorted({key for key, _ in existing} & extra.keys())
    if conflicts:
        raise ProviderConfigurationError(f"Reserved query parameter(s): {', '.join(conflicts)}")
    query = urlencode([*existing, *extra.items()])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _resolve_api_key(profile: ModelProfile) -> str | None:
    """Resolve the API key at call time.  Returns None for no-key endpoints."""
    if profile.api_key_env is None:
        return None
    try:
        return profile.api_key_env.resolve()
    except KeyError:
        return None


def _require_api_key(profile: ModelProfile) -> str:
    """Resolve API key, raising AuthProviderError if missing.

    Call this when the profile declares api_key_env -- we must NOT silently
    treat a missing key as a no-key endpoint.
    """
    if profile.api_key_env is None:
        raise AuthProviderError(
            "Profile requires an API key but api_key_env is not configured",
            provider=profile.protocol.value,
            model=profile.model,
        )
    try:
        return profile.api_key_env.resolve()
    except KeyError:
        raise AuthProviderError(
            f"API key environment variable {profile.api_key_env.env_var} is not set",
            provider=profile.protocol.value,
            model=profile.model,
        ) from None


def _redact_url(url: str) -> str:
    """Redact a URL for safe inclusion in error messages."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.hostname}"


_SECRET_SHAPED = re.compile(r"sk-[A-Za-z0-9_\-]{8,}|[Bb]earer\s+\S+|[A-Za-z0-9_\-]{32,}")


def _safe_error_snippet(body: str) -> str:
    """Lift the provider's own reason out of an error envelope, redacted.

    Only ``error.message`` is taken -- never the raw body, which can echo the
    request back and with it the whole conversation. Anything key-shaped
    inside that message is masked, and the result is length-bounded.
    """
    try:
        detail: Any = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    for key in ("error", "message"):
        if isinstance(detail, dict) and key in detail:
            detail = detail[key]
    if not isinstance(detail, str):
        return ""
    return _SECRET_SHAPED.sub("<redacted>", detail).strip()[:200]


def _http_error_message(status_code: int, body: str) -> str:
    """``Provider returned HTTP N`` plus the provider's reason when it gave one."""
    reason = _safe_error_snippet(body)
    return f"Provider returned HTTP {status_code}" + (f": {reason}" if reason else "")


async def _error_body(resp: httpx.Response) -> str:
    """Read a failed streaming response's body; never raise from the error path."""
    try:
        body: bytes = await resp.aread()
    except (AttributeError, httpx.HTTPError, OSError, RuntimeError):
        return ""
    return body.decode("utf-8", errors="replace")


def _store_provider_error_artifact(
    artifact_store: ArtifactStore | None,
    status_code: int,
    provider: str,
    model: str,
) -> str | None:
    """Store a redacted, bounded provider-error artifact.

    Returns artifact digest or None.  Never stores raw response body.
    """
    if artifact_store is None:
        return None
    try:
        error_payload = json.dumps(
            {
                "error": True,
                "status_code": status_code,
                "provider": provider,
                "model": model,
                "timestamp": time.time(),
            },
            sort_keys=True,
        )
        ref = artifact_store.write(error_payload.encode(), media_type="application/json")
        return ref.digest
    except OSError:
        logger.warning("Failed to store provider error artifact", exc_info=True)
        return None


class AdapterProtocol(TypingProtocol):
    """Protocol that all wire-format adapters must satisfy."""

    async def complete(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> ModelResponse:
        """Send the request and return a normalized response."""
        ...

    def stream(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream normalized deltas for the request.

        Synchronous generator over an awaited transport stream; the first
        HTTP byte is awaited on first ``__anext__`` so pre-stream failures
        raise here, inside the caller's retry window.
        """
        ...


_TRANSIENT_STATUS = frozenset({429, 502, 503, 504})
_RATE_LIMIT_BODY = re.compile(
    r"tokens per minute|requests per minute|TPM|RPM|rate limit", re.IGNORECASE
)
_TRANSPORT_RETRIES = 2
_RETRY_BACKOFF_SECONDS = (1.5, 6.0)
_retry_sleep = asyncio.sleep

_OPENAI_FINISH = {
    "stop": FinishReason.END_TURN,
    "tool_calls": FinishReason.TOOL_USE,
    "length": FinishReason.MAX_TOKENS,
    "content_filter": FinishReason.ERROR,
}
_ANTHROPIC_FINISH = {
    "end_turn": FinishReason.END_TURN,
    "tool_use": FinishReason.TOOL_USE,
    "max_tokens": FinishReason.MAX_TOKENS,
    "stop_sequence": FinishReason.STOP_SEQUENCE,
}
_GEMINI_FINISH = {
    "STOP": FinishReason.END_TURN,
    "MAX_TOKENS": FinishReason.MAX_TOKENS,
    "SAFETY": FinishReason.ERROR,
    "RECITATION": FinishReason.ERROR,
    "PROHIBITED_CONTENT": FinishReason.ERROR,
    "SPII": FinishReason.ERROR,
    "MALFORMED_FUNCTION_CALL": FinishReason.ERROR,
    "OTHER": FinishReason.ERROR,
}
_PROVIDER_LABELS = {
    Protocol.OPENAI_CHAT: "openai",
    Protocol.ANTHROPIC_MESSAGES: "anthropic",
    Protocol.GEMINI_GENERATE_CONTENT: "gemini",
}

_ANTHROPIC_API_VERSION = "2023-06-01"

_AUTH_HEADERS = {
    Protocol.OPENAI_CHAT: ("Authorization", "Bearer "),
    Protocol.ANTHROPIC_MESSAGES: ("x-api-key", ""),
    Protocol.GEMINI_GENERATE_CONTENT: ("x-goog-api-key", ""),
}
_STATIC_HEADERS: dict[Protocol, dict[str, str]] = {
    Protocol.ANTHROPIC_MESSAGES: {"anthropic-version": _ANTHROPIC_API_VERSION},
}

_TOOL_ERROR_PREFIX = "[tool_error] "


def _build_headers(profile: ModelProfile, api_key: str | None) -> dict[str, str]:
    """Assemble request headers for a profile.

    Profile-supplied headers land before the credential so a stray entry can
    never displace it; ``auth_header`` overrides the header NAME only, and
    sends the key raw -- Azure OpenAI wants ``api-key: <key>`` with no scheme.
    """
    headers = {
        "Content-Type": "application/json",
        **_STATIC_HEADERS.get(profile.protocol, {}),
        **profile.extra_headers,
    }
    if api_key:
        name, prefix = _AUTH_HEADERS[profile.protocol]
        if profile.auth_header:
            name, prefix = profile.auth_header, ""
        headers[name] = f"{prefix}{api_key}"
    return headers


async def _execute_request(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    provider_label: str,
    model: str,
    *,
    redacted_base_url: str,
    timeout_seconds: float = 300.0,
    retries: int = _TRANSPORT_RETRIES,
    backoff_seconds: Sequence[float] = _RETRY_BACKOFF_SECONDS,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """POST, then run the shared status/auth/json error pipeline. Returns parsed JSON.

    Transport-level transient failures (connect errors, 502/503/504) are
    retried with a short fixed backoff: a failed request billed nothing, and
    one pool hiccup otherwise kills one-shot callers (falsifier, reviewers)
    that have no outer retry loop. Semantic failures (4xx, bad JSON) are not
    retried -- the caller's recovery law owns those. Retry counts and backoff
    come from the profile so a pool's temperament is config, not code.
    """
    sleeper = sleep_fn if sleep_fn is not None else _retry_sleep
    for attempt in range(retries + 1):
        try:
            resp = await client.post(url, json=body, headers=headers, timeout=timeout_seconds)
        except httpx.RequestError as e:
            if attempt < retries:
                await sleeper(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                continue
            raise ProviderError(
                f"Network error calling {redacted_base_url}",
                provider=provider_label,
                model=model,
            ) from e
        if resp.status_code in _TRANSIENT_STATUS and attempt < retries:
            await sleeper(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
            continue
        break

    if resp.status_code == 401:
        raise AuthProviderError(
            "Authentication failed -- check API key",
            provider=provider_label,
            model=model,
        )

    if resp.status_code != 200:
        raise HTTPProviderError(
            _http_error_message(resp.status_code, resp.text),
            status_code=resp.status_code,
            provider=provider_label,
            model=model,
        )

    try:
        parsed: dict[str, Any] = resp.json()
        return parsed
    except json.JSONDecodeError as e:
        raise ProviderError(
            "Invalid JSON in response",
            provider=provider_label,
            model=model,
        ) from e


def _sse_payload(data_lines: list[str]) -> dict[str, Any] | None:
    raw = "\n".join(data_lines)
    if raw == "[DONE]":
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("SSE data must be a JSON object")
    return payload


def _split_stream_lines(text: str, *, final: bool = False) -> tuple[list[str], str]:
    lines: list[str] = []
    start = index = 0
    while index < len(text):
        if text[index] == "\n":
            lines.append(text[start:index])
            start = index = index + 1
            continue
        if text[index] == "\r":
            if index + 1 == len(text) and not final:
                break
            lines.append(text[start:index])
            index += 2 if index + 1 < len(text) and text[index + 1] == "\n" else 1
            start = index
            continue
        index += 1
    remainder = text[start:]
    if final and remainder:
        lines.append(remainder)
        remainder = ""
    return lines, remainder


async def _utf8_stream_lines(response: httpx.Response) -> AsyncIterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    buffered = ""
    async for chunk in response.aiter_bytes():
        lines, buffered = _split_stream_lines(buffered + decoder.decode(chunk))
        for line in lines:
            yield line
    lines, _ = _split_stream_lines(buffered + decoder.decode(b"", final=True), final=True)
    for line in lines:
        yield line


async def _sse_events(response: httpx.Response) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
    """Yield ``(event_name, parsed_data)`` pairs from an SSE response.

    Handles bare ``data:`` lines (OpenAI) and ``event:``/``data:`` pairs
    (Anthropic). OpenAI's ``[DONE]`` sentinel yields ``None``; malformed JSON
    is rejected. Comment lines are skipped per the SSE spec.
    """
    event_name = ""
    data_lines: list[str] = []
    async for line in _utf8_stream_lines(response):
        if line == "":
            if data_lines:
                yield event_name, _sse_payload(data_lines)
                event_name = ""
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        yield event_name, _sse_payload(data_lines)


def _require_stream_terminal(provider: str, terminal: bool) -> None:
    if not terminal:
        raise ProviderProtocolError(
            f"{provider} stream ended before its terminal frame",
            provider=provider,
            field_path="stream.terminal",
        )


@_protocol_stream("openai")
async def _openai_stream_chunks(
    response: httpx.Response,
    *,
    strict: bool = False,
) -> AsyncIterator[StreamDelta]:
    """Parse OpenAI chat-completions SSE chunks into StreamDeltas."""
    tool_names: dict[int, str] = {}
    finish_reason: FinishReason | None = None
    terminal = False
    async for _event_name, data in _sse_events(response):
        if data is None:
            terminal = True
            break
        choices = data.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text:
            yield TextDelta(text=text)
        for tc in delta.get("tool_calls") or []:
            idx = int(tc.get("index", 0))
            fn = tc.get("function") or {}
            name = fn.get("name") or tool_names.get(idx, "")
            if fn.get("name"):
                tool_names[idx] = fn["name"]
            yield ToolCallDelta(
                index=idx,
                id=tc.get("id") or "",
                name=name,
                arguments_fragment=fn.get("arguments") or "",
            )
        fr = choice.get("finish_reason")
        if fr is not None:
            finish_reason = _OPENAI_FINISH.get(fr, FinishReason.END_TURN)
        usage_raw = data.get("usage")
        if isinstance(usage_raw, dict) and (
            usage_raw.get("prompt_tokens") or usage_raw.get("completion_tokens")
        ):
            details = usage_raw.get("prompt_tokens_details")
            cached_read = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
            prompt_tokens = usage_raw.get("prompt_tokens", 0)
            yield UsageDelta(
                usage=Usage(
                    input_tokens=max(0, prompt_tokens - cached_read),
                    output_tokens=usage_raw.get("completion_tokens", 0),
                    cache_creation_tokens=0,
                    cache_read_tokens=cached_read,
                )
            )
    if strict:
        _require_stream_terminal("openai", terminal)
    yield FinishDelta(finish_reason=finish_reason or FinishReason.END_TURN)


@_protocol_stream("anthropic")
async def _anthropic_stream_chunks(
    response: httpx.Response,
    *,
    strict: bool = False,
) -> AsyncIterator[StreamDelta]:
    """Parse Anthropic messages SSE events into StreamDeltas."""
    finish_reason: FinishReason | None = None
    input_usage: Usage | None = None
    output_tokens = 0
    terminal = False
    async for event_name, data in _sse_events(response):
        etype = (data or {}).get("type") or event_name
        if etype == "message_stop":
            terminal = True
            break
        if etype == "message_start":
            message = (data or {}).get("message")
            if message is not None and not isinstance(message, dict):
                raise TypeError("message_start.message must be an object")
            msg_usage = (message or {}).get("usage") or {}
            if msg_usage:
                input_usage = Usage(
                    input_tokens=msg_usage.get("input_tokens", 0),
                    cache_creation_tokens=msg_usage.get("cache_creation_input_tokens", 0),
                    cache_read_tokens=msg_usage.get("cache_read_input_tokens", 0),
                )
        elif etype == "content_block_start":
            block = (data or {}).get("content_block") or {}
            if block.get("type") == "tool_use":
                yield ToolCallDelta(
                    index=int((data or {}).get("index", 0)),
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                )
        elif etype == "content_block_delta":
            delta = (data or {}).get("delta") or {}
            if delta.get("type") == "text_delta":
                yield TextDelta(text=delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                yield ToolCallDelta(
                    index=int((data or {}).get("index", 0)),
                    arguments_fragment=delta.get("partial_json", ""),
                )
        elif etype == "message_delta":
            stop = ((data or {}).get("delta") or {}).get("stop_reason")
            if stop is not None:
                finish_reason = _ANTHROPIC_FINISH.get(stop, FinishReason.END_TURN)
            msg_usage = (data or {}).get("usage") or {}
            output_tokens = max(output_tokens, msg_usage.get("output_tokens", 0))
    if strict:
        _require_stream_terminal("anthropic", terminal)
    if input_usage is not None:
        yield UsageDelta(usage=input_usage.model_copy(update={"output_tokens": output_tokens}))
    else:
        yield UsageDelta(usage=Usage(output_tokens=output_tokens))
    yield FinishDelta(finish_reason=finish_reason or FinishReason.END_TURN)


def _gemini_usage(raw: dict[str, Any]) -> Usage:
    """Normalize Gemini usageMetadata to the disjoint-bucket contract.

    ``promptTokenCount`` is the TOTAL input including the cached prefix, so
    the cached count is subtracted out.  Thinking tokens are billed at the
    output rate and Gemini reports them separately, so they fold into output.
    """
    cached = raw.get("cachedContentTokenCount", 0)
    return Usage(
        input_tokens=max(0, raw.get("promptTokenCount", 0) - cached),
        output_tokens=raw.get("candidatesTokenCount", 0) + raw.get("thoughtsTokenCount", 0),
        cache_read_tokens=cached,
    )


@_protocol_stream("gemini")
async def _gemini_stream_chunks(
    response: httpx.Response,
    *,
    strict: bool = False,
) -> AsyncIterator[StreamDelta]:
    """Parse Gemini streamGenerateContent SSE chunks into StreamDeltas.

    Each chunk is a whole GenerateContentResponse.  Unlike OpenAI and
    Anthropic, Gemini never fragments a functionCall across chunks -- args
    arrive complete -- so each one emits a single self-contained delta at the
    next running index.

    Thought parts are dropped on exactly the same condition as
    ``GeminiAdapter._parse_response``: the streamed and completed text for one
    response must be the same bytes, or a JSON-parsing caller sees different
    answers depending on the transport.
    """
    finish_reason: FinishReason | None = None
    usage = Usage()
    tool_index = 0
    terminal = False
    async for _event_name, data in _sse_events(response):
        if data is None:
            break
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("text") and not part.get("thought"):
                yield TextDelta(text=part["text"])
            call = part.get("functionCall")
            if call:
                yield ToolCallDelta(
                    index=tool_index,
                    id=call.get("id") or str(uuid.uuid4()),
                    name=call.get("name", ""),
                    arguments_fragment=json.dumps(call.get("args") or {}),
                    signature=part.get("thoughtSignature") or "",
                )
                tool_index += 1
        fr = candidate.get("finishReason")
        if fr is not None:
            finish_reason = _GEMINI_FINISH.get(fr, FinishReason.END_TURN)
            terminal = True
        usage_raw = data.get("usageMetadata")
        if isinstance(usage_raw, dict) and usage_raw:
            usage = _gemini_usage(usage_raw)
        if terminal:
            break
    if strict:
        _require_stream_terminal("gemini", terminal)
    yield UsageDelta(usage=usage)
    yield FinishDelta(finish_reason=finish_reason or FinishReason.END_TURN)


async def _open_stream_with_retry(
    open_fn: Callable[[], Awaitable[httpx.Response]],
    profile: ModelProfile,
    provider_label: str,
    model: str,
    *,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> httpx.Response:
    """Open a streaming POST with the same transient-retry policy as
    ``_execute_request``. Retries apply only until a response with headers
    arrives; once streaming starts, mid-stream failure is terminal -- a
    half-consumed response cannot be safely replayed above this layer.
    """
    retries = profile.transport_retries
    backoff = tuple(profile.transport_retry_backoff_seconds)
    sleeper = sleep_fn if sleep_fn is not None else _retry_sleep
    for attempt in range(retries + 1):
        try:
            resp = await open_fn()
        except httpx.RequestError as e:
            if attempt < retries:
                await sleeper(backoff[min(attempt, len(backoff) - 1)])
                continue
            raise ProviderError(
                f"Network error calling {_redact_url(profile.base_url)}",
                provider=provider_label,
                model=model,
            ) from e
        if resp.status_code == 401:
            error = AuthProviderError(
                "Authentication failed -- check API key",
                provider=provider_label,
                model=model,
            )
            try:
                await resp.aclose()
            except (OSError, RuntimeError, httpx.HTTPError) as cleanup_error:
                error.add_note(f"response cleanup failed: {type(cleanup_error).__name__}")
            raise error
        if resp.status_code in _TRANSIENT_STATUS and attempt < retries:
            await resp.aclose()
            await sleeper(backoff[min(attempt, len(backoff) - 1)])
            continue
        if resp.status_code != 200:
            message = _http_error_message(resp.status_code, await _error_body(resp))
            await resp.aclose()
            raise HTTPProviderError(
                message,
                status_code=resp.status_code,
                provider=provider_label,
                model=model,
            )
        return resp
    raise ProviderError("unreachable", provider=provider_label, model=model)


async def _post_stream(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> httpx.Response:
    request = client.build_request("POST", url, json=body, headers=headers, timeout=timeout_seconds)
    return await client.send(request, stream=True)


async def _yield_stream_response(
    response: httpx.Response,
    deltas: AsyncIterator[StreamDelta],
    profile: ModelProfile,
    provider: str,
) -> AsyncIterator[StreamDelta]:
    closed = False
    error: BaseException | None = None

    async def close() -> None:
        nonlocal closed
        if not closed:
            closed = True
            await response.aclose()

    try:
        async for delta in deltas:
            if isinstance(delta, FinishDelta):
                await close()
            yield delta
    except httpx.StreamError as stream_error:
        error = ProviderError(
            f"Stream interrupted calling {_redact_url(profile.base_url)}",
            provider=provider,
            model=profile.model,
        )
        raise error from stream_error
    except (ProviderError, asyncio.CancelledError, GeneratorExit) as active_error:
        error = active_error
        raise
    except (OSError, RuntimeError, httpx.HTTPError) as cleanup_error:
        error = ProviderError(
            f"Failed to close {provider} stream response",
            provider=provider,
            model=profile.model,
        )
        raise error from cleanup_error
    finally:
        if not closed:
            try:
                await close()
            except (OSError, RuntimeError, httpx.HTTPError) as cleanup_error:
                if error is None:
                    raise ProviderError(
                        f"Failed to close {provider} stream response",
                        provider=provider,
                        model=profile.model,
                    ) from cleanup_error
                if isinstance(error, Exception):
                    error.add_note(f"response cleanup failed: {type(cleanup_error).__name__}")


async def _stream_openai(
    request: ModelRequest,
    profile: ModelProfile,
    client: httpx.AsyncClient,
    api_key: str | None,
) -> AsyncIterator[StreamDelta]:
    headers = _build_headers(profile, api_key)
    body = OpenAIAdapter()._build_body(request, profile)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    url = _with_query(_openai_endpoint(profile.base_url), profile.extra_query)

    async def _open() -> httpx.Response:
        return await _post_stream(client, url, body, headers, profile.request_timeout_seconds)

    resp = await _open_stream_with_retry(_open, profile, "openai", profile.model)
    deltas = _openai_stream_chunks(
        resp, strict=profile.strict_stream_completion, model=profile.model
    )
    async for delta in _yield_stream_response(resp, deltas, profile, "openai"):
        yield delta


async def _stream_anthropic(
    request: ModelRequest,
    profile: ModelProfile,
    client: httpx.AsyncClient,
    api_key: str | None,
) -> AsyncIterator[StreamDelta]:
    headers = _build_headers(profile, api_key)
    body = AnthropicAdapter()._build_body(request, profile)
    body["stream"] = True
    url = _with_query(_anthropic_endpoint(profile.base_url), profile.extra_query)

    async def _open() -> httpx.Response:
        return await _post_stream(client, url, body, headers, profile.request_timeout_seconds)

    resp = await _open_stream_with_retry(_open, profile, "anthropic", profile.model)
    deltas = _anthropic_stream_chunks(
        resp, strict=profile.strict_stream_completion, model=profile.model
    )
    async for delta in _yield_stream_response(resp, deltas, profile, "anthropic"):
        yield delta


async def _stream_gemini(
    request: ModelRequest,
    profile: ModelProfile,
    client: httpx.AsyncClient,
    api_key: str | None,
) -> AsyncIterator[StreamDelta]:
    headers = _build_headers(profile, api_key)
    body = GeminiAdapter()._build_body(request, profile)
    url = _with_query(
        _gemini_endpoint(profile.base_url, profile.model, stream=True), profile.extra_query
    )

    async def _open() -> httpx.Response:
        return await _post_stream(client, url, body, headers, profile.request_timeout_seconds)

    resp = await _open_stream_with_retry(_open, profile, "gemini", profile.model)
    deltas = _gemini_stream_chunks(
        resp, strict=profile.strict_stream_completion, model=profile.model
    )
    async for delta in _yield_stream_response(resp, deltas, profile, "gemini"):
        yield delta


class StreamAssembler:
    """Folds StreamDeltas into the ModelResponse the non-streaming parser
    would have produced -- the invariant equivalence tests pin.  Tool-call
    argument fragments concatenate per index; an undecodable tail degrades to
    empty arguments exactly as ``_parse_response`` does for malformed
    provider arguments."""

    def __init__(self) -> None:
        self._text: list[str] = []
        self._tools: dict[int, dict[str, str]] = {}
        self._usage = Usage()
        self._finish: FinishReason | None = None

    def update(self, delta: StreamDelta) -> None:
        if isinstance(delta, TextDelta):
            self._text.append(delta.text)
        elif isinstance(delta, ToolCallDelta):
            buf = self._tools.setdefault(
                delta.index, {"id": "", "name": "", "args": "", "signature": ""}
            )
            if delta.id:
                buf["id"] = delta.id
            if delta.name:
                buf["name"] = delta.name
            if delta.signature:
                buf["signature"] = delta.signature
            buf["args"] += delta.arguments_fragment
        elif isinstance(delta, UsageDelta):
            self._usage = delta.usage
        elif isinstance(delta, FinishDelta):
            self._finish = delta.finish_reason

    def build(self, provider: str, model: str) -> ModelResponse:
        tool_calls = [
            ToolCall(
                id=self._tools[i]["id"] or str(uuid.uuid4()),
                name=self._tools[i]["name"],
                arguments=_safe_json_object(self._tools[i]["args"]),
                signature=self._tools[i]["signature"],
            )
            for i in sorted(self._tools)
        ]
        return ModelResponse(
            text="".join(self._text),
            tool_calls=tool_calls,
            finish_reason=self._finish or FinishReason.END_TURN,
            usage=self._usage,
            provider=provider,
            model=model,
        )


async def assemble_streamed_response(
    deltas: AsyncIterator[StreamDelta],
    provider: str,
    model: str,
) -> ModelResponse:
    assembler = StreamAssembler()
    async for d in deltas:
        assembler.update(d)
    return assembler.build(provider, model)


def _safe_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class OpenAIAdapter:
    """Adapter for OpenAI-compatible /chat/completions endpoints.

    OpenAI assistant content is string or None, never a list of typed blocks.
    """

    async def complete(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> ModelResponse:
        headers = _build_headers(profile, api_key)

        body = self._build_body(request, profile)

        url = _with_query(_openai_endpoint(profile.base_url), profile.extra_query)

        data = await _execute_request(
            client,
            url,
            body,
            headers,
            "openai",
            profile.model,
            redacted_base_url=_redact_url(profile.base_url),
            timeout_seconds=profile.request_timeout_seconds,
            retries=profile.transport_retries,
            backoff_seconds=tuple(profile.transport_retry_backoff_seconds),
        )

        return self._parse_response(data, profile.model)

    def stream(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> AsyncIterator[StreamDelta]:
        return _stream_openai(request, profile, client, api_key)

    def _build_body(self, request: ModelRequest, profile: ModelProfile) -> dict[str, Any]:
        """Build OpenAI chat completions request body."""
        request = _visible_request(request, profile)
        messages = self._convert_messages(request)
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})

        body: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            profile.token_limit_param: request.get_max_tokens(profile.max_output_tokens),
        }

        if request.tools and Capability.TOOL_USE in profile.capabilities:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in request.tools
            ]

        if _should_send_json_schema(request, profile):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": _strict_ready(request.output_json_schema),
                    "schema": request.output_json_schema,
                },
            }

        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.seed is not None:
            body["seed"] = request.seed

        return body

    @staticmethod
    def _content(msg: ModelMessage) -> Any:
        """Plain string unless images are attached, in which case OpenAI wants
        a typed content-part list."""
        if not msg.images:
            return msg.content
        parts: list[dict[str, Any]] = [{"type": "text", "text": msg.content}] if msg.content else []
        return parts + [
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img.media_type};base64,{img.data}"},
            }
            for img in msg.images
        ]

    def _convert_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        """Convert provider-neutral messages to OpenAI format.

        OpenAI assistant content is string or None, never a list -- except
        when images make it a list of typed content parts.
        """
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                tool_calls_raw = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or None,
                        "tool_calls": tool_calls_raw,
                    }
                )
            elif msg.role == MessageRole.TOOL:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": _tool_result_text(msg),
                    }
                )
            else:
                messages.append(
                    {
                        "role": msg.role.value,
                        "content": self._content(msg),
                    }
                )
        return messages

    @_protocol_parser("openai")
    def _parse_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        """Parse OpenAI chat completions response into ModelResponse."""
        choices = data.get("choices", [])
        if not choices:
            raise ProviderProtocolError(
                "Response contains empty choices",
                provider="openai",
                model=model,
                field_path="choices",
            )
        choice = choices[0]
        message = choice.get("message", {})

        text = message.get("content") or ""
        finish_reason = _OPENAI_FINISH.get(
            choice.get("finish_reason", "stop"), FinishReason.END_TURN
        )

        tool_calls: list[ToolCall] = []
        for tc_raw in message.get("tool_calls", []):
            func = tc_raw.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc_raw.get("id", str(uuid.uuid4())),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )

        usage_raw = data.get("usage", {})
        cached_read = (
            usage_raw.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            if isinstance(usage_raw.get("prompt_tokens_details"), dict)
            else 0
        )
        prompt_tokens = usage_raw.get("prompt_tokens", 0)
        usage = Usage(
            input_tokens=max(0, prompt_tokens - cached_read),
            output_tokens=usage_raw.get("completion_tokens", 0),
            cache_creation_tokens=0,
            cache_read_tokens=cached_read,
        )

        if tool_calls:
            finish_reason = FinishReason.TOOL_USE

        return ModelResponse(
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            provider="openai",
            model=data.get("model", model),
        )


_EPHEMERAL = {"type": "ephemeral"}


def _mark_cache_breakpoint(message: dict[str, Any]) -> None:
    """Cache everything up to and including ``message``.

    Two breakpoints per request: one static (tools + system, identical for the
    whole cell) and this rolling one on the newest turn, so each turn reads the
    previous turn's cache and writes only its own delta.  A plain string
    content is normalised to one text block first -- ``cache_control`` lives on
    a block, never on the message.
    """
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return
        content = [{"type": "text", "text": content}]
        message["content"] = content
    if content:
        content[-1]["cache_control"] = _EPHEMERAL


class AnthropicAdapter:
    """Adapter for Anthropic-compatible /v1/messages endpoints.

    Sends x-api-key and anthropic-version headers.  System prompt is
    separated from messages.  Content blocks include tool_use and
    tool_result conversion.

    Seed is NOT sent (Anthropic does not support it).
    """

    API_VERSION = _ANTHROPIC_API_VERSION

    async def complete(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> ModelResponse:
        headers = _build_headers(profile, api_key)

        body = self._build_body(request, profile)

        url = _with_query(_anthropic_endpoint(profile.base_url), profile.extra_query)

        data = await _execute_request(
            client,
            url,
            body,
            headers,
            "anthropic",
            profile.model,
            redacted_base_url=_redact_url(profile.base_url),
            timeout_seconds=profile.request_timeout_seconds,
            retries=profile.transport_retries,
            backoff_seconds=tuple(profile.transport_retry_backoff_seconds),
        )

        return self._parse_response(data, profile.model)

    def stream(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> AsyncIterator[StreamDelta]:
        return _stream_anthropic(request, profile, client, api_key)

    def _build_body(self, request: ModelRequest, profile: ModelProfile) -> dict[str, Any]:
        """Build Anthropic messages request body."""
        request = _visible_request(request, profile)
        messages = self._convert_messages(request)

        body: dict[str, Any] = {
            "model": profile.model,
            "max_tokens": request.get_max_tokens(profile.max_output_tokens),
            "messages": messages,
        }

        caching = Capability.PROMPT_CACHING in profile.capabilities
        if caching and messages:
            _mark_cache_breakpoint(messages[-1])

        system_parts: list[str] = []
        if request.system:
            system_parts.append(request.system)
        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM and msg.content:
                system_parts.append(msg.content)
        if system_parts:
            system_text = "\n\n".join(system_parts)
            body["system"] = (
                [{"type": "text", "text": system_text, "cache_control": _EPHEMERAL}]
                if caching
                else system_text
            )

        if request.tools and Capability.TOOL_USE in profile.capabilities:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in request.tools
            ]

        if _should_send_json_schema(request, profile):
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.output_json_schema,
                },
            }

        if request.temperature is not None:
            body["temperature"] = request.temperature

        return body

    def _convert_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        """Convert provider-neutral messages to Anthropic format.

        Anthropic uses content blocks.  System messages are extracted to
        top-level.  Tool results become tool_result blocks.
        """
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                continue

            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                content: list[dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": content})

            elif msg.role == MessageRole.TOOL:
                result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                }
                if msg.content:
                    result_block["content"] = msg.content
                if msg.is_error:
                    result_block["is_error"] = True
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].append(result_block)
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [result_block],
                        }
                    )

            elif msg.images:
                text_block = [{"type": "text", "text": msg.content}] if msg.content else []
                messages.append(
                    {
                        "role": msg.role.value,
                        "content": [_anthropic_image_block(img) for img in msg.images] + text_block,
                    }
                )

            else:
                messages.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return messages

    @_protocol_parser("anthropic")
    def _parse_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        """Parse Anthropic messages response into ModelResponse."""
        content_blocks = data.get("content", [])
        if not content_blocks:
            if data.get("stop_reason") == "max_tokens":
                return ModelResponse(
                    text="",
                    tool_calls=[],
                    finish_reason=FinishReason.MAX_TOKENS,
                    provider="anthropic",
                    model=model,
                )
            raise ProviderProtocolError(
                "Response contains empty content",
                provider="anthropic",
                model=model,
                field_path="content",
            )
        finish_reason = _ANTHROPIC_FINISH.get(
            data.get("stop_reason", "end_turn"), FinishReason.END_TURN
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", str(uuid.uuid4())),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        usage_raw = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_raw.get("input_tokens", 0),
            output_tokens=usage_raw.get("output_tokens", 0),
            cache_creation_tokens=usage_raw.get("cache_creation_input_tokens", 0),
            cache_read_tokens=usage_raw.get("cache_read_input_tokens", 0),
        )

        if tool_calls:
            finish_reason = FinishReason.TOOL_USE

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            provider="anthropic",
            model=data.get("model", model),
        )


_GEMINI_ROLES = {MessageRole.ASSISTANT: "model", MessageRole.TOOL: "user"}


class GeminiAdapter:
    """Adapter for the Gemini API's generateContent endpoints.

    Three shape differences from the other two protocols drive this code.
    The model is addressed in the URL path rather than the body.  Assistant
    turns are role ``model``, and tool results are ordinary ``user`` turns
    carrying ``functionResponse`` parts.  Those parts key off the tool NAME,
    not a call id, so ``_convert_messages`` carries an id -> name map forward
    from the assistant turn that requested the call.
    """

    async def complete(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> ModelResponse:
        headers = _build_headers(profile, api_key)

        body = self._build_body(request, profile)

        url = _with_query(_gemini_endpoint(profile.base_url, profile.model), profile.extra_query)

        data = await _execute_request(
            client,
            url,
            body,
            headers,
            "gemini",
            profile.model,
            redacted_base_url=_redact_url(profile.base_url),
            timeout_seconds=profile.request_timeout_seconds,
            retries=profile.transport_retries,
            backoff_seconds=tuple(profile.transport_retry_backoff_seconds),
        )

        return self._parse_response(data, profile.model)

    def stream(
        self,
        request: ModelRequest,
        profile: ModelProfile,
        client: httpx.AsyncClient,
        api_key: str | None,
    ) -> AsyncIterator[StreamDelta]:
        return _stream_gemini(request, profile, client, api_key)

    def _build_body(self, request: ModelRequest, profile: ModelProfile) -> dict[str, Any]:
        """Build a Gemini generateContent request body."""
        request = _visible_request(request, profile)
        generation: dict[str, Any] = {
            "maxOutputTokens": request.get_max_tokens(profile.max_output_tokens)
        }
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if request.seed is not None:
            generation["seed"] = request.seed
        if _should_send_json_schema(request, profile):
            generation["responseMimeType"] = "application/json"
            generation["responseJsonSchema"] = request.output_json_schema

        body: dict[str, Any] = {
            "contents": self._convert_messages(request),
            "generationConfig": generation,
        }

        system_parts = [request.system] if request.system else []
        system_parts += [
            m.content for m in request.messages if m.role == MessageRole.SYSTEM and m.content
        ]
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        if request.tools and Capability.TOOL_USE in profile.capabilities:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            **_gemini_tool_parameters(t.parameters),
                        }
                        for t in request.tools
                    ]
                }
            ]

        return body

    def _convert_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        """Convert provider-neutral messages to Gemini ``contents``.

        System messages are hoisted to systemInstruction by ``_build_body``
        and skipped here.  Consecutive tool results merge into one user turn,
        matching the Anthropic adapter's tool_result batching.
        """
        call_names: dict[str, str] = {}
        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                continue

            parts: list[dict[str, Any]] = []
            if msg.role == MessageRole.TOOL:
                parts.append(
                    {
                        "functionResponse": {
                            "name": call_names.get(msg.tool_call_id, msg.tool_call_id),
                            "response": {"error" if msg.is_error else "result": msg.content},
                        }
                    }
                )
                if contents and contents[-1]["role"] == "user":
                    contents[-1]["parts"].extend(parts)
                    continue
            else:
                if msg.content:
                    parts.append({"text": msg.content})
                parts += [
                    {"inlineData": {"mimeType": img.media_type, "data": img.data}}
                    for img in msg.images
                ]
                for tc in msg.tool_calls:
                    call_names[tc.id] = tc.name
                    call_part: dict[str, Any] = {
                        "functionCall": {"name": tc.name, "args": tc.arguments}
                    }
                    if tc.signature:
                        call_part["thoughtSignature"] = tc.signature
                    parts.append(call_part)

            contents.append({"role": _GEMINI_ROLES.get(msg.role, "user"), "parts": parts})

        return contents

    @_protocol_parser("gemini")
    def _parse_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        """Parse a Gemini generateContent response into ModelResponse."""
        candidates = data.get("candidates", [])
        if not candidates:
            raise ProviderProtocolError(
                "Response contains empty candidates",
                provider="gemini",
                model=model,
                field_path="candidates",
            )
        candidate = candidates[0]

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        content = candidate.get("content")
        if content is not None and not isinstance(content, dict):
            raise ProviderProtocolError(
                "Malformed gemini response at candidates[0].content",
                provider="gemini",
                model=model,
                field_path="candidates[0].content",
            )
        for part in (content or {}).get("parts") or []:
            if part.get("text") and not part.get("thought"):
                text_parts.append(part["text"])
            call = part.get("functionCall")
            if call:
                tool_calls.append(
                    ToolCall(
                        id=call.get("id") or str(uuid.uuid4()),
                        name=call.get("name", ""),
                        arguments=call.get("args") or {},
                        signature=part.get("thoughtSignature") or "",
                    )
                )

        finish_reason = _GEMINI_FINISH.get(
            candidate.get("finishReason", "STOP"), FinishReason.END_TURN
        )
        if tool_calls:
            finish_reason = FinishReason.TOOL_USE

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=_gemini_usage(data.get("usageMetadata", {})),
            provider="gemini",
            model=data.get("modelVersion", model),
        )


_ADAPTERS: dict[Protocol, type[AdapterProtocol]] = {
    Protocol.OPENAI_CHAT: OpenAIAdapter,
    Protocol.ANTHROPIC_MESSAGES: AnthropicAdapter,
    Protocol.GEMINI_GENERATE_CONTENT: GeminiAdapter,
}


def _get_adapter(protocol: Protocol) -> AdapterProtocol:
    """Get the adapter for a wire protocol."""
    cls = _ADAPTERS.get(protocol)
    if cls is None:
        raise ProviderError(f"Unsupported protocol: {protocol}")
    return cls()


class ModelLimiter:
    """Structurally enforces per-profile max concurrency and RPM.

    Scoped by profile name.  Injectable clock/sleeper for deterministic tests.

    GLM/singleton profiles (max_concurrency=1) never overlap with
    high-concurrency profiles -- each profile gets its own semaphore.

    Process-local: limiters are NOT shared across processes.  Use a single
    ModelGateway (which owns a shared limiter pool) to get structural
    enforcement across multiple callers within one process.
    """

    def __init__(
        self,
        profile_name: str,
        max_concurrency: int,
        rpm: int,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._profile_name = profile_name
        self._max_concurrency = max(1, max_concurrency)
        self._rpm = max(1, rpm)
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or _default_sleep

        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

    @property
    def profile_name(self) -> str:
        return self._profile_name

    async def _enforce_rpm(self) -> None:
        """Reserve one request slot in the rolling RPM window."""
        while True:
            async with self._lock:
                now = self._clock()
                self._request_times = [t for t in self._request_times if now - t < 60.0]
                if len(self._request_times) < self._rpm:
                    self._request_times.append(now)
                    return
                wait_seconds = self._request_times[0] + 60.0 - now
            await self._sleeper(wait_seconds)

    async def run(self, callable_fn: Callable[[], Awaitable[T]]) -> T:
        """Execute callable within concurrency + RPM limits.

        The callable is invoked while holding both the concurrency semaphore
        and after RPM checking.
        """
        async with self._semaphore:
            await self._enforce_rpm()
            return await callable_fn()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[None]:
        """Hold the concurrency slot + RPM slot across an await sequence.

        Streaming uses this instead of ``run`` because a stream cannot be
        expressed as one awaited callable: the slot must stay held from open
        to last delta so RPM accounting matches billed usage.
        """
        async with self._semaphore:
            await self._enforce_rpm()
            yield


async def _default_sleep(seconds: float) -> None:
    """Default async sleeper."""
    await asyncio.sleep(seconds)


class ModelGateway:
    """Model completion gateway.

    Selects the appropriate adapter from the profile's protocol, resolves
    secrets at call time, enforces capabilities, and stores normalized
    response artifacts.

    Owns a shared limiter pool keyed by profile name: every request for the
    same profile within this gateway shares one semaphore and RPM window.
    This provides structural single-flight enforcement across multiple
    NativeWorker instances that share the same gateway.

    Constructor:
        ModelGateway(config, artifact_store=None)

    Usage:
        response = await gateway.complete(request)
    """

    def __init__(
        self,
        config: NorthStackConfig,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self._config = config
        self._artifact_store = artifact_store
        self._adapters: dict[Protocol, AdapterProtocol] = {}
        self._client: httpx.AsyncClient | None = None
        self._limiters: dict[str, ModelLimiter] = {}

    def profile(self, profile_name: str) -> ModelProfile:
        """Look up a ModelProfile by name.

        Returns the profile for pricing, output limits, capabilities, etc.
        Raises ProviderError if not found.
        """
        return self._get_profile(profile_name)

    def _get_limiter(self, profile_name: str) -> ModelLimiter:
        """Get or create a shared limiter for a profile.

        All callers for the same profile_name within this gateway share
        the same semaphore and RPM window.  Process-local only -- two
        gateway instances do NOT coordinate.  The control plane
        composition root must provide a single gateway per process.
        """
        if profile_name not in self._limiters:
            p = self._get_profile(profile_name)
            self._limiters[profile_name] = ModelLimiter(
                profile_name=p.name,
                max_concurrency=p.max_concurrency,
                rpm=p.requests_per_minute,
            )
        return self._limiters[profile_name]

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient()
        return self._client

    def _get_profile(self, profile_name: str) -> ModelProfile:
        """Look up a profile by name from the config."""
        for p in self._config.profiles:
            if p.name == profile_name:
                return p
        raise ProviderConfigurationError(f"Profile not found: {profile_name}")

    def _get_adapter(self, protocol: Protocol) -> AdapterProtocol:
        """Get or cache the adapter for a protocol."""
        if protocol not in self._adapters:
            self._adapters[protocol] = _get_adapter(protocol)
        return self._adapters[protocol]

    async def _complete_unlimited(self, request: ModelRequest) -> ModelResponse:
        """Complete a model request without limiter (internal).

        Resolves the profile, checks capabilities, delegates to the adapter,
        and stores a normalized response artifact.
        """
        profile = self._get_profile(request.profile_name)

        warnings = _check_capabilities(request, profile)
        for w in warnings:
            logger.info("Capability negotiation: %s", w)

        api_key = None if profile.api_key_env is None else _require_api_key(profile)

        adapter = self._get_adapter(profile.protocol)
        client = await self._get_client()
        response = await adapter.complete(request, profile, client, api_key)

        digest = await self._store_response_artifact(response)
        if digest is not None:
            response = response.model_copy(update={"response_artifact_id": digest})

        return response

    async def _store_response_artifact(self, response: ModelResponse) -> str | None:
        """Persist the normalized response; returns its digest or None.

        The write is blocking file I/O (hash + mkdir + write_bytes); on this
        async path it is offloaded to a worker thread so a slow disk cannot
        stall the loop or a concurrently gathered coroutine.
        """
        if self._artifact_store is None:
            return None
        try:
            artifact_json = json.dumps(
                {
                    "provider": response.provider,
                    "model": response.model,
                    "text": response.text,
                    "tool_calls": [tc.model_dump() for tc in response.tool_calls],
                    "finish_reason": response.finish_reason.value,
                    "usage": response.usage.model_dump(),
                },
                sort_keys=True,
            )
            ref = await asyncio.to_thread(
                self._artifact_store.write,
                artifact_json.encode(),
                media_type="application/json",
            )
        except OSError:
            logger.warning("Failed to store response artifact", exc_info=True)
            return None
        return ref.digest

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete a model request, applying per-profile concurrency + RPM limits.

        The limiter is owned by this gateway.  Every caller of complete()
        for the same profile_name shares one semaphore and RPM window.
        Direct gateway users and NativeWorker both get limiting -- no
        double-wrapping.
        """
        limiter = self._get_limiter(request.profile_name)
        return await limiter.run(lambda: self._complete_unlimited(request))

    def supports_streaming(self, profile_name: str) -> bool:
        """True when the profile declares Capability.STREAMING."""
        return Capability.STREAMING in self._get_profile(profile_name).capabilities

    async def stream_complete(self, request: ModelRequest) -> AsyncIterator[StreamDelta]:
        """Stream deltas under the same limits and artifact contract as complete().

        The whole stream lifecycle (open through last delta) holds the
        profile's concurrency semaphore and RPM slot.  Profiles without
        Capability.STREAMING fall back to a single-shot complete() wrapped as
        usage+finish deltas, so callers need no branching.  The normalized
        response is stored as an artifact exactly as complete() does -- the
        audit trail cannot tell streamed from non-streamed calls apart.
        """
        profile = self._get_profile(request.profile_name)

        if Capability.STREAMING not in profile.capabilities:
            response = await self.complete(request)
            yield UsageDelta(usage=response.usage)
            yield FinishDelta(finish_reason=response.finish_reason)
            return

        api_key = None if profile.api_key_env is None else _require_api_key(profile)
        adapter = self._get_adapter(profile.protocol)
        client = await self._get_client()
        limiter = self._get_limiter(request.profile_name)
        provider_label = _PROVIDER_LABELS[profile.protocol]
        assembler = StreamAssembler()

        async with limiter.session():
            async for delta in adapter.stream(request, profile, client, api_key):
                assembler.update(delta)
                yield delta

        await self._store_response_artifact(assembler.build(provider_label, profile.model))

    async def complete_stream(
        self,
        request: ModelRequest,
        *,
        on_delta: Callable[[StreamDelta], None] | None = None,
    ) -> ModelResponse:
        """Complete via streaming, returning the same ModelResponse as complete().

        Opt-in consumer path: the caller receives one ``ModelResponse`` (tool
        calls drive NativeWorker's loop) while ``on_delta`` observes each raw
        delta live -- the worker passes its heartbeat there so a long stream
        keeps beating the stall detector instead of looking pinned.  Deltas
        never enter the ledger; the stored normalized artifact is identical to
        complete()'s, so replay and the audit trail cannot distinguish the
        transport.
        """
        profile = self._get_profile(request.profile_name)
        if Capability.STREAMING not in profile.capabilities:
            return await self.complete(request)

        api_key = None if profile.api_key_env is None else _require_api_key(profile)
        adapter = self._get_adapter(profile.protocol)
        client = await self._get_client()
        limiter = self._get_limiter(request.profile_name)
        provider_label = _PROVIDER_LABELS[profile.protocol]
        assembler = StreamAssembler()

        async with limiter.session():
            async for delta in adapter.stream(request, profile, client, api_key):
                assembler.update(delta)
                if on_delta is not None:
                    on_delta(delta)

        response = assembler.build(provider_label, profile.model)
        digest = await self._store_response_artifact(response)
        if digest is not None:
            response = response.model_copy(update={"response_artifact_id": digest})
        return response

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
