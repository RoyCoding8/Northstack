"""Tests for the provider gateway's secret handling, error pipeline and adapter dispatch.

These paths (_resolve_api_key, _require_api_key, _store_provider_error_artifact,
_execute_request, _get_adapter and the adapters over a fake transport) are not
exercised by any other test. Uses tests/helpers/fake_gateway.py as the seam.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import httpx
import pytest

from northstack.adapters.providers import gateway as _gateway_module
from northstack.adapters.providers.gateway import (
    AnthropicAdapter,
    AuthProviderError,
    HTTPProviderError,
    OpenAIAdapter,
    ProviderError,
    _execute_request,
    _get_adapter,
    _require_api_key,
    _resolve_api_key,
    _store_provider_error_artifact,
)
from northstack.adapters.providers.wire import (
    MessageRole,
    ModelMessage,
    ModelRequest,
)
from northstack.config import Protocol, SecretEnvRef

from tests.helpers.fake_gateway import (
    ANTHROPIC_OK,
    FailingArtifactStore,
    OPENAI_OK,
    RecordingArtifactStore,
    raises,
    responds,
)


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(_seconds: float) -> None:
        return

    monkeypatch.setattr(_gateway_module, "_retry_sleep", _noop)


# Reused construction helpers (copied from tests/test_providers.py so we do not
# invent a different ModelProfile shape)


def _make_profile(
    name: str = "test-worker",
    protocol: Protocol = Protocol.OPENAI_CHAT,
    base_url: str = "http://localhost:8080/v1",
    model: str = "test-model",
    max_concurrency: int = 4,
    rpm: int = 60,
    api_key_env: SecretEnvRef | None = None,
) -> Any:
    from northstack.config import ModelProfile

    return ModelProfile(
        name=name,
        protocol=protocol,
        base_url=base_url,
        model=model,
        capabilities=set(),
        max_concurrency=max_concurrency,
        requests_per_minute=rpm,
        api_key_env=api_key_env,
    )


def _make_request(
    profile_name: str = "test-worker",
    content: str = "Hello",
) -> ModelRequest:
    return ModelRequest(
        profile_name=profile_name,
        messages=[ModelMessage(role=MessageRole.USER, content=content)],
        tools=[],
        output_json_schema=None,
    )


# A: secret resolution


class TestTransientClassification:
    """A 413 is two different failures wearing one status code: a per-minute
    token ceiling (clears itself) and an oversized request (never will).
    """

    @pytest.mark.parametrize(
        "message, transient",
        [
            ("HTTP 413: Request too large ... tokens per minute (TPM): Limit 8000", True),
            ("HTTP 413: rate limit reached for this organization", True),
            ("HTTP 413: payload exceeds the model context window", False),
            ("HTTP 400: malformed tool schema", False),
        ],
    )
    def test_413_splits_on_the_body(self, message, transient):
        status = int(message.split()[1].rstrip(":"))
        assert HTTPProviderError(message, status_code=status).is_transient() is transient

    @pytest.mark.parametrize("status", [429, 502, 503, 504])
    def test_the_usual_transients_are_unchanged(self, status):
        assert HTTPProviderError("busy", status_code=status).is_transient()


class TestSecretResolution:
    def test_resolve_api_key_none_when_no_env_ref(self):
        profile = _make_profile(api_key_env=None)
        assert _resolve_api_key(profile) is None

    def test_resolve_api_key_none_when_env_var_missing(self):
        profile = _make_profile(api_key_env=SecretEnvRef(env_var="NONEXISTENT_KEY_A2_12345"))
        assert _resolve_api_key(profile) is None

    def test_resolve_api_key_value_when_env_var_set(self, monkeypatch):
        monkeypatch.setenv("SET_KEY_A3", "sk-a3-value")
        profile = _make_profile(api_key_env=SecretEnvRef(env_var="SET_KEY_A3"))
        assert _resolve_api_key(profile) == "sk-a3-value"

    def test_require_api_key_raises_when_no_env_ref(self):
        profile = _make_profile(api_key_env=None)
        with pytest.raises(AuthProviderError) as exc_info:
            _require_api_key(profile)
        assert exc_info.value.provider == profile.protocol.value
        assert exc_info.value.model == profile.model

    def test_require_api_key_raises_and_names_missing_env(self):
        env_var = "NONEXISTENT_KEY_A5_67890"
        profile = _make_profile(api_key_env=SecretEnvRef(env_var=env_var))
        with pytest.raises(AuthProviderError) as exc_info:
            _require_api_key(profile)
        assert env_var in str(exc_info.value)

    def test_require_api_key_value_when_env_var_set(self, monkeypatch):
        monkeypatch.setenv("SET_KEY_A6", "sk-a6-value")
        profile = _make_profile(api_key_env=SecretEnvRef(env_var="SET_KEY_A6"))
        assert _require_api_key(profile) == "sk-a6-value"

    def test_secret_value_never_embedded_in_profile(self, monkeypatch):
        """The key resolves out of the env but is never carried on the profile itself."""
        secret = "sk-FAKESECRET_A7_DO_NOT_LEAK"
        monkeypatch.setenv("SET_KEY_A7", secret)
        profile = _make_profile(api_key_env=SecretEnvRef(env_var="SET_KEY_A7"))
        assert _require_api_key(profile) == secret
        assert secret not in repr(profile)
        assert secret not in str(profile.model_dump())


# B: error artifact


class TestErrorArtifact:
    def test_store_none_when_no_store(self):
        assert _store_provider_error_artifact(None, 500, "openai", "m") is None

    def test_store_writes_once_with_digest(self, tmp_path):
        store = RecordingArtifactStore(tmp_path)
        digest = _store_provider_error_artifact(store, 500, "openai", "m")
        assert isinstance(digest, str) and digest.startswith("sha256:")
        assert len(store.writes) == 1

    def test_store_payload_keys_and_values(self, tmp_path):
        store = RecordingArtifactStore(tmp_path)
        digest = _store_provider_error_artifact(store, 429, "anthropic", "claude-x")
        assert digest is not None
        payload = json.loads(store.writes[0][0])
        assert set(payload.keys()) == {
            "error",
            "status_code",
            "provider",
            "model",
            "timestamp",
        }
        assert payload["status_code"] == 429
        assert payload["provider"] == "anthropic"
        assert payload["model"] == "claude-x"
        assert payload["error"] is True

    def test_store_cannot_carry_a_response_body(self, tmp_path):
        """No body can leak: the payload is built from arguments, and there is no body argument."""
        assert set(inspect.signature(_store_provider_error_artifact).parameters) == {
            "artifact_store",
            "status_code",
            "provider",
            "model",
        }
        store = RecordingArtifactStore(tmp_path)
        digest = _store_provider_error_artifact(store, 503, "openai", "m")
        assert digest is not None
        written = store.writes[0][0]
        assert store.read_by_digest(digest) == written
        assert written.decode() == json.dumps(json.loads(written), sort_keys=True)

    def test_store_returns_none_on_oserror(self, tmp_path):
        store = FailingArtifactStore(tmp_path)
        assert _store_provider_error_artifact(store, 500, "openai", "m") is None

    def test_store_media_type_json(self, tmp_path):
        store = RecordingArtifactStore(tmp_path)
        _store_provider_error_artifact(store, 500, "openai", "m")
        assert store.writes[0][1] == "application/json"


# C: _execute_request pipeline


class TestExecuteRequestPipeline:
    async def test_status_200_valid_json_returns_dict(self):
        client, _ = responds({"ok": True})
        async with client:
            result = await _execute_request(
                client,
                "http://localhost/v1/chat/completions",
                {},
                {},
                "openai",
                "m",
                redacted_base_url="http://localhost",
            )
        assert result == {"ok": True}

    async def test_status_401_raises_auth(self):
        client, _ = responds(status=401)
        with pytest.raises(AuthProviderError) as exc_info:
            async with client:
                await _execute_request(
                    client,
                    "http://localhost/v1/chat/completions",
                    {},
                    {},
                    "openai",
                    "m",
                    redacted_base_url="http://localhost",
                )
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "m"

    async def test_status_500_raises_http_with_code(self):
        client, _ = responds(status=500)
        with pytest.raises(HTTPProviderError) as exc_info:
            async with client:
                await _execute_request(
                    client,
                    "http://localhost/v1/chat/completions",
                    {},
                    {},
                    "openai",
                    "m",
                    redacted_base_url="http://localhost",
                )
        assert exc_info.value.status_code == 500

    async def test_status_429_raises_http_with_code(self):
        client, _ = responds(status=429)
        with pytest.raises(HTTPProviderError) as exc_info:
            async with client:
                await _execute_request(
                    client,
                    "http://localhost/v1/chat/completions",
                    {},
                    {},
                    "openai",
                    "m",
                    redacted_base_url="http://localhost",
                )
        assert exc_info.value.status_code == 429

    async def test_status_200_non_json_raises_provider(self):
        client, _ = responds(text="this is not json", status=200)
        with pytest.raises(ProviderError, match="Invalid JSON"):
            async with client:
                await _execute_request(
                    client,
                    "http://localhost/v1/chat/completions",
                    {},
                    {},
                    "openai",
                    "m",
                    redacted_base_url="http://localhost",
                )

    async def test_transport_connect_error_redacts_url(self):
        client, _ = raises(httpx.ConnectError("connection refused"))
        with pytest.raises(ProviderError) as exc_info:
            async with client:
                await _execute_request(
                    client,
                    "https://api.example.com/v1/chat/completions?trace=abc",
                    {},
                    {},
                    "openai",
                    "m",
                    redacted_base_url="https://api.example.com",
                )
        msg = str(exc_info.value)
        assert "https://api.example.com" in msg
        assert "/chat/completions" not in msg
        assert "?trace=abc" not in msg
        assert exc_info.value.provider == "openai"
        assert exc_info.value.model == "m"

    async def test_all_errors_carry_provider_and_model(self):
        for status in (401, 500, 429):
            client, _ = responds(status=status)
            with pytest.raises(ProviderError) as exc_info:
                async with client:
                    await _execute_request(
                        client,
                        "http://localhost/v1/chat/completions",
                        {},
                        {},
                        "anthropic",
                        "anthropic-model",
                        redacted_base_url="http://localhost",
                    )
            assert exc_info.value.provider == "anthropic"
            assert exc_info.value.model == "anthropic-model"


# D: adapter dispatch


class TestAdapterDispatch:
    def test_get_adapter_openai(self):
        adapter = _get_adapter(Protocol.OPENAI_CHAT)
        assert isinstance(adapter, OpenAIAdapter)

    def test_get_adapter_anthropic(self):
        adapter = _get_adapter(Protocol.ANTHROPIC_MESSAGES)
        assert isinstance(adapter, AnthropicAdapter)

    def test_get_adapter_returns_new_each_call(self):
        a1 = _get_adapter(Protocol.OPENAI_CHAT)
        a2 = _get_adapter(Protocol.OPENAI_CHAT)
        assert a1 is not a2


# E: adapters end-to-end over the fake transport


class TestAdaptersEndToEnd:
    async def test_openai_complete_text_and_usage(self):
        client, _ = responds(OPENAI_OK)
        adapter = OpenAIAdapter()
        profile = _make_profile(protocol=Protocol.OPENAI_CHAT)
        request = _make_request()
        async with client:
            response = await adapter.complete(request, profile, client, None)
        assert response.text == "hello"
        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 7

    async def test_anthropic_complete_text_and_usage(self):
        client, _ = responds(ANTHROPIC_OK)
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = _make_request()
        async with client:
            response = await adapter.complete(request, profile, client, None)
        assert response.text == "hello"
        assert response.usage.input_tokens == 11
        assert response.usage.output_tokens == 7

    async def test_openai_sends_bearer_auth_header(self):
        client, recorder = responds(OPENAI_OK)
        adapter = OpenAIAdapter()
        profile = _make_profile(protocol=Protocol.OPENAI_CHAT)
        request = _make_request()
        async with client:
            await adapter.complete(request, profile, client, "sk-openai-e3")
        assert recorder.header("authorization") == "Bearer sk-openai-e3"

    async def test_anthropic_sends_x_api_key_header(self):
        client, recorder = responds(ANTHROPIC_OK)
        adapter = AnthropicAdapter()
        profile = _make_profile(protocol=Protocol.ANTHROPIC_MESSAGES)
        request = _make_request()
        async with client:
            await adapter.complete(request, profile, client, "sk-anthropic-e4")
        assert recorder.header("x-api-key") == "sk-anthropic-e4"

    async def test_error_str_does_not_leak_api_key(self):
        """The key reaches the wire, the 500 path runs, and the error still does not carry it."""
        api_key = "sk-SUPERSECRETKEY_E5_DO_NOT_LEAK"
        client, recorder = responds(status=500)
        adapter = OpenAIAdapter()
        profile = _make_profile(protocol=Protocol.OPENAI_CHAT)
        request = _make_request()
        with pytest.raises(HTTPProviderError) as exc_info:
            async with client:
                await adapter.complete(request, profile, client, api_key)
        assert exc_info.value.status_code == 500
        assert recorder.header("authorization") == f"Bearer {api_key}"
        assert api_key not in str(exc_info.value)
        assert api_key not in repr(exc_info.value)
