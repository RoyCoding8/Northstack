"""Tests for config_toml.py -- the writable-config serializer.

Seams tested:
  1. config_to_toml -- round-trips a minimal config (load -> emit -> parse -> load) equal
  2. config_to_toml -- round-trips the real northstack.toml on modeled fields
  3. api_key_env=None omits the key entirely (from_toml reproduces None)
  4. api_key_env emits the env-var NAME only, never a value
  5. `tier` is NEVER emitted (derived property, not a field)
  6. commands emit the 5 base fields; isolation/docker_image only when non-default
  7. roles/capabilities emitted as sorted value-string lists
  8. default-valued optional fields are omitted (tidy + exact round-trip)
  9. save_config_to_toml writes a parseable file
  10. emitted TOML is parseable by stdlib tomllib
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from northstack.adapters.config_toml import config_to_toml, save_config_to_toml
from northstack.config import (
    Capability,
    CommandConfig,
    ModelProfile,
    NorthStackConfig,
    Protocol,
    Role,
    RunConfig,
    SecretEnvRef,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "northstack.toml"


# Helpers


def _sig(c: NorthStackConfig) -> tuple:
    """A comparable signature over all modeled fields."""
    return (
        c.name,
        [
            (
                p.name,
                p.protocol.value,
                p.base_url,
                p.model,
                p.max_concurrency,
                sorted(r.value for r in p.roles),
                sorted(cc.value for cc in p.capabilities),
                p.api_key_env.env_var if p.api_key_env is not None else None,
                p.allow_insecure_http,
                p.requests_per_minute,
                p.context_window_tokens,
                p.max_output_tokens,
                p.request_timeout_seconds,
                p.strict_stream_completion,
                p.transport_retries,
                tuple(p.transport_retry_backoff_seconds),
                p.input_price_per_million_usd,
                p.output_price_per_million_usd,
                p.auth_header,
                dict(p.extra_headers),
                dict(p.extra_query),
                p.token_limit_param,
            )
            for p in c.profiles
        ],
        [
            (
                cmd.name,
                tuple(cmd.argv),
                cmd.timeout_seconds,
                cmd.max_output_bytes,
                list(cmd.env_allowlist),
                cmd.isolation,
                cmd.docker_image,
            )
            for cmd in c.commands
        ],
        (
            c.run.default_budget_tokens,
            c.run.default_budget_cost_usd,
            c.run.stall_window_seconds,
            c.run.planner_mode,
            c.run.falsifier_mode,
            c.run.calibration_path,
        ),
        [(e.role.value, list(e.profiles)) for e in c.routing],
    )


def _roundtrip(c: NorthStackConfig, tmp_path: Path) -> NorthStackConfig:
    """Emit to a temp file, re-load via from_toml."""
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    return NorthStackConfig.from_toml(out)


def test_roundtrip_minimal(tmp_path: Path) -> None:
    c = NorthStackConfig(name="minimal")
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


def test_roundtrip_preserves_all_profile_transport_fields(tmp_path: Path) -> None:
    profile = ModelProfile(
        name="custom",
        protocol=Protocol.OPENAI_CHAT,
        base_url="https://example.com/v1",
        model="custom-model",
        max_concurrency=3,
        allow_insecure_http=True,
        request_timeout_seconds=17.5,
        strict_stream_completion=False,
        transport_retries=1,
        transport_retry_backoff_seconds=[0.25],
        auth_header="X-Custom-Key",
        extra_headers={"X-Client": "northstack"},
        extra_query={"api-version": "2026-01-01"},
        token_limit_param="max_completion_tokens",
    )
    config = NorthStackConfig(name="transport", profiles=[profile])
    assert _sig(config) == _sig(_roundtrip(config, tmp_path))


@pytest.mark.skipif(
    not REPO_CONFIG.exists(),
    reason="local operator config northstack.toml is gitignored; not present in CI",
)
def test_roundtrip_real_company_toml(tmp_path: Path) -> None:
    """The on-disk northstack.toml round-trips on all modeled fields."""
    c = NorthStackConfig.from_toml(REPO_CONFIG)
    rt = _roundtrip(c, tmp_path)
    assert _sig(c) == _sig(rt)


def test_api_key_env_none_omitted(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        profiles=[
            ModelProfile(
                name="local",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost",
                model="m",
                roles={Role.WORKER},
                max_concurrency=1,
            )
        ],
    )
    toml = config_to_toml(c)
    assert "api_key_env" not in toml  # key omitted entirely
    rt = _roundtrip(c, tmp_path)
    assert rt.profiles[0].api_key_env is None
    assert _sig(c) == _sig(rt)


def test_api_key_env_emits_name_only(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        profiles=[
            ModelProfile(
                name="remote",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost",
                model="m",
                api_key_env=SecretEnvRef(env_var="MY_API_KEY"),
                roles={Role.WORKER},
                max_concurrency=1,
            )
        ],
    )
    toml = config_to_toml(c)
    assert "MY_API_KEY" in toml  # env-var name present
    assert "sk-" not in toml  # no secret value shape
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))
    # never a value field next to the name
    assert "value" not in toml


def test_tier_never_emitted(tmp_path: Path) -> None:
    # A profile priced like an expert (tier 3) must NOT serialize `tier`.
    c = NorthStackConfig(
        name="t",
        profiles=[
            ModelProfile(
                name="expert",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost",
                model="m",
                roles={Role.ORCHESTRATOR},
                max_concurrency=1,
                input_price_per_million_usd=15.0,
                output_price_per_million_usd=75.0,
            )
        ],
    )
    assert c.profiles[0].tier == 3  # sanity: tier is derived
    toml = config_to_toml(c)
    assert "tier" not in toml
    # re-loading must still work (from_toml tolerates unknown keys, but the
    # point is we don't introduce a misleading editable tier field)
    rt = _roundtrip(c, tmp_path)
    assert rt.profiles[0].tier == 3
    assert _sig(c) == _sig(rt)


def test_commands_emit_exactly_five_fields(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        commands=[
            CommandConfig(
                name="lint",
                argv=["ruff", "check", "."],
                timeout_seconds=30.0,
                max_output_bytes=65_536,
                env_allowlist=["PATH", "HOME"],
            )
        ],
    )
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    raw = tomllib.loads(out.read_text("utf-8"))
    cmd = raw["northstack"]["commands"][0]
    assert set(cmd.keys()) == {
        "name",
        "argv",
        "timeout_seconds",
        "max_output_bytes",
        "env_allowlist",
    }
    assert cmd["env_allowlist"] == ["PATH", "HOME"]
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


def test_command_isolation_emitted_when_non_default(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        commands=[
            CommandConfig(
                name="scan",
                argv=["python", "-m", "scan"],
                isolation="docker",
                docker_image="python:3.12-slim",
            )
        ],
    )
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    cmd = tomllib.loads(out.read_text("utf-8"))["northstack"]["commands"][0]
    assert cmd["isolation"] == "docker"
    assert cmd["docker_image"] == "python:3.12-slim"
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


def test_run_non_default_fields_emitted_and_round_trip(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        run=RunConfig(
            default_budget_tokens=0,
            default_budget_cost_usd=0.0,
            stall_window_seconds=30.0,
            planner_mode="model",
            falsifier_mode="model",
            calibration_path="calibration.jsonl",
        ),
    )
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    run = tomllib.loads(out.read_text("utf-8"))["northstack"]["run"]
    assert run["default_budget_tokens"] == 0
    assert run["stall_window_seconds"] == 30.0
    assert run["planner_mode"] == "model"
    assert run["falsifier_mode"] == "model"
    assert run["calibration_path"] == "calibration.jsonl"
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


def test_roles_capabilities_sorted_value_strings(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        profiles=[
            ModelProfile(
                name="p",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost",
                model="m",
                roles={Role.SPECIALIST, Role.WORKER, Role.REVIEWER},  # unsorted
                capabilities={Capability.STREAMING, Capability.TOOL_USE},  # unsorted
                max_concurrency=1,
            )
        ],
    )
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    raw = tomllib.loads(out.read_text("utf-8"))
    p = raw["northstack"]["profiles"][0]
    assert p["roles"] == sorted(["specialist", "worker", "reviewer"])
    assert p["capabilities"] == sorted(["streaming", "tool_use"])
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


def test_default_fields_omitted(tmp_path: Path) -> None:
    c = NorthStackConfig(
        name="t",
        profiles=[
            ModelProfile(
                name="p",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost",
                model="m",
                roles={Role.WORKER},
                max_concurrency=1,
                # all defaults: requests_per_minute=60, ctx=128000, out=4096, prices=0
            )
        ],
    )
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    raw = tomllib.loads(out.read_text("utf-8"))
    p = raw["northstack"]["profiles"][0]
    assert "requests_per_minute" not in p
    assert "context_window_tokens" not in p
    assert "max_output_tokens" not in p
    assert "input_price_per_million_usd" not in p
    assert "output_price_per_million_usd" not in p
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


def test_run_default_omitted(tmp_path: Path) -> None:
    c = NorthStackConfig(name="t", run=RunConfig())  # defaults
    out = tmp_path / "rt.toml"
    save_config_to_toml(c, out)
    raw = tomllib.loads(out.read_text("utf-8"))
    assert "run" not in raw["northstack"]
    assert _sig(c) == _sig(_roundtrip(c, tmp_path))


@pytest.mark.skipif(
    not REPO_CONFIG.exists(),
    reason="local operator config northstack.toml is gitignored; not present in CI",
)
def test_emitted_is_stdlib_parseable(tmp_path: Path) -> None:
    """The emitted string must parse with stdlib tomllib (the from_toml parser)."""
    c = NorthStackConfig.from_toml(REPO_CONFIG)
    toml_str = config_to_toml(c)
    # Must not raise
    parsed = tomllib.loads(toml_str)
    assert "northstack" in parsed
    assert parsed["northstack"]["name"] == c.name
