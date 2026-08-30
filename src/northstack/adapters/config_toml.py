"""Serialize a ``NorthStackConfig`` back to TOML.

The counterpart to ``NorthStackConfig.from_toml``.  Emits exactly the shape
``from_toml`` reads so a round-trip (load -> serialize -> parse -> load) is
lossless for all operator-editable fields.

Fidelity rules (verified against ``config.py``):
  - ``ModelProfile.tier`` is a DERIVED property (config.py:140-154), not a
    field.  It must NEVER be emitted -- a stray ``tier`` key would be a
    silent extra in the TOML (from_toml ignores unknown profile keys, but
    emitting it would mislead operators into thinking tier is editable).
  - ``api_key_env`` is a ``SecretEnvRef | None``.  Only the env-var NAME is
    serialized (never a value).  When None, the key is OMITTED entirely so
    ``from_toml``'s ``p.get("api_key_env")`` truthiness check reproduces None.
  - ``roles`` / ``capabilities`` are enums; emit sorted value-string lists.
  - ``commands`` always emit name, argv, timeout_seconds, max_output_bytes,
    env_allowlist; isolation/docker_image emit only when non-default.
  - ``run`` emits every non-default RunConfig field (budgets, stall window,
    planner_mode, falsifier_mode, calibration_path).
  - ``routing`` emits ``role`` (value str) + ``profiles`` (list[str]).

Secrets: this writer never sees or writes a secret VALUE -- only the env-var
name.  Round-trip safety is tested in ``tests/test_config_toml.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel

from northstack.adapters.atomic_io import atomic_write_text
from northstack.config import (
    CommandConfig,
    ModelProfile,
    NorthStackConfig,
    RouteMapping,
    RunConfig,
)


def _default(model: type[BaseModel], field: str) -> Any:
    """The pydantic ``Field(default=...)`` for ``field`` -- the single source of
    truth for "is this value still the default?" so the writer and the loader
    (``from_toml`` validating through pydantic) can never disagree.
    """
    return model.model_fields[field].get_default(call_default_factory=True)


def _profile_to_dict(profile: ModelProfile) -> dict[str, Any]:
    """Serialize a ModelProfile to a TOML-table dict matching from_toml's reads."""
    d: dict[str, Any] = {
        "name": profile.name,
        "protocol": profile.protocol.value,
        "base_url": profile.base_url,
        "model": profile.model,
        "max_concurrency": profile.max_concurrency,
    }
    if profile.api_key_env is not None:
        d["api_key_env"] = profile.api_key_env.env_var
    if profile.allow_insecure_http:
        d["allow_insecure_http"] = True
    d["roles"] = sorted(r.value for r in profile.roles)
    d["capabilities"] = sorted(c.value for c in profile.capabilities)
    for fld in (
        "requests_per_minute",
        "context_window_tokens",
        "max_output_tokens",
        "request_timeout_seconds",
        "strict_stream_completion",
        "transport_retries",
        "transport_retry_backoff_seconds",
        "input_price_per_million_usd",
        "output_price_per_million_usd",
        "auth_header",
        "extra_headers",
        "extra_query",
        "token_limit_param",
    ):
        value = getattr(profile, fld)
        if value != _default(ModelProfile, fld):
            d[fld] = value
    return d


def _command_to_dict(command: CommandConfig) -> dict[str, Any]:
    """Serialize a CommandConfig -- the fields from_toml's ``CommandConfig``
    model expects. The 5 core fields always emit; isolation fields emit only
    when non-default, keeping default round-trips byte-stable."""
    d: dict[str, Any] = {
        "name": command.name,
        "argv": list(command.argv),
        "timeout_seconds": command.timeout_seconds,
        "max_output_bytes": command.max_output_bytes,
        "env_allowlist": list(command.env_allowlist),
    }
    if command.isolation != "host":
        d["isolation"] = command.isolation
    if command.docker_image:
        d["docker_image"] = command.docker_image
    return d


def _routing_to_dict(entry: RouteMapping) -> dict[str, Any]:
    """Serialize a RouteMapping -- role value str + ordered profiles list."""
    return {
        "role": entry.role.value,
        "profiles": list(entry.profiles),
    }


def config_to_toml(config: NorthStackConfig) -> str:
    """Return a TOML string that ``from_toml`` can parse back into an equal config."""
    company: dict[str, Any] = {"name": config.name}

    if config.profiles:
        company["profiles"] = [_profile_to_dict(p) for p in config.profiles]
    if config.commands:
        company["commands"] = [_command_to_dict(c) for c in config.commands]

    run_dict: dict[str, Any] = {}
    for fld in (
        "default_budget_tokens",
        "default_budget_cost_usd",
        "stall_window_seconds",
        "planner_mode",
        "falsifier_mode",
        "calibration_path",
    ):
        value = getattr(config.run, fld)
        if value != _default(RunConfig, fld):
            run_dict[fld] = value
    if run_dict:
        company["run"] = run_dict

    if config.routing:
        company["routing"] = [_routing_to_dict(e) for e in config.routing]

    return tomli_w.dumps({"northstack": company})


def save_config_to_toml(config: NorthStackConfig, path: Path) -> Path:
    """Write the config to ``path`` as TOML. Returns the resolved path."""
    path = Path(path)
    atomic_write_text(path, config_to_toml(config))
    return path
