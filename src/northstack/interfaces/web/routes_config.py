"""Config editing + command-test endpoints for the web control surface.

All read responses embed a per-profile ``key_status`` (env-var name + resolved
OK/UNSET, never a value) and the derived ``tier``.  Writes route through the
``ConfigStore`` which validates-by-construction: a bad edit raises
``ValidationError``/``ValueError`` -> HTTP 400 with the message, and the
in-memory store is left unchanged.

Routing presets are strategies derived from the current eligible profiles and
include availability metadata for the UI.  Applying one derives it again from
the current store and routes it through ``ConfigStore.update_routing`` so every
reference and role declaration is validated atomically.

Command "test" endpoint: dry-executes a named command in a scratch directory
and returns bounded stdout/stderr/exit.  This is NOT a sandbox (documented in
the response) -- it is a convenience preview of argv behavior.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from northstack.adapters.workspace.commands import run_command
from northstack.config import (
    Capability,
    CommandConfig,
    ModelProfile,
    NonNegativeFiniteFloat,
    Protocol,
    Role,
    RouteMapping,
    RunConfig,
    SecretEnvRef,
)
from northstack.interfaces.web.config_store import ConfigStore

router = APIRouter(tags=["config"])


_PRESET_META = {
    "single_expert": (
        "Single expert",
        "Use one profile that declares every role.",
    ),
    "cheap_fanout": (
        "Cheap fan-out",
        "Use up to two lowest-cost eligible profiles per role.",
    ),
    "strong_integrator": (
        "Strong integrator",
        "Use the highest-cost eligible profile for each role.",
    ),
}


def _profile_cost(profile: ModelProfile) -> float:
    return profile.input_price_per_million_usd + profile.output_price_per_million_usd


def _routing_presets(store: ConfigStore) -> list[dict[str, Any]]:
    profiles = list(store.get().profiles)
    roles = list(Role)
    eligible = {role: [p for p in profiles if role in p.roles] for role in roles}
    missing = [role.value for role in roles if not eligible[role]]

    all_role_profiles = [p for p in profiles if all(role in p.roles for role in roles)]
    if all_role_profiles:
        expert = max(
            all_role_profiles,
            key=lambda p: (_profile_cost(p), len(p.capabilities), p.context_window_tokens, p.name),
        )
        single_routing = {role.value: [expert.name] for role in roles}
        single_reason = None
    else:
        single_routing = {}
        single_reason = "No profile declares all five roles."

    if missing:
        unavailable_reason = f"No eligible profile for: {', '.join(missing)}."
        cheap_routing: dict[str, list[str]] = {}
        strong_routing: dict[str, list[str]] = {}
    else:
        unavailable_reason = None
        cheap_routing = {
            role.value: [
                p.name
                for p in sorted(
                    eligible[role],
                    key=lambda p: (_profile_cost(p), -p.max_concurrency, p.name),
                )[:2]
            ]
            for role in roles
        }
        strong_routing = {
            role.value: [
                max(
                    eligible[role],
                    key=lambda p: (
                        _profile_cost(p),
                        len(p.capabilities),
                        p.context_window_tokens,
                        p.name,
                    ),
                ).name
            ]
            for role in roles
        }

    routings = {
        "single_expert": (single_routing, single_reason),
        "cheap_fanout": (cheap_routing, unavailable_reason),
        "strong_integrator": (strong_routing, unavailable_reason),
    }
    return [
        {
            "id": preset_id,
            "label": _PRESET_META[preset_id][0],
            "description": _PRESET_META[preset_id][1],
            "routing": routing,
            "available": reason is None,
            "reason": reason,
        }
        for preset_id, (routing, reason) in routings.items()
    ]


def _store(request: Request) -> ConfigStore:
    store: ConfigStore = request.app.state.store
    return store


class NameBody(BaseModel):
    name: str


class ProfileBody(BaseModel):
    name: str
    protocol: str
    base_url: str
    model: str
    api_key_env: str | None = None  # env-var NAME only
    allow_insecure_http: bool = False
    roles: list[str] = []
    capabilities: list[str] = []
    max_concurrency: int = 1
    requests_per_minute: int = 60
    context_window_tokens: int = 128_000
    max_output_tokens: int = 4_096
    request_timeout_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    strict_stream_completion: bool = True
    transport_retries: int = Field(default=2, ge=0, le=5)
    transport_retry_backoff_seconds: list[NonNegativeFiniteFloat] = Field(
        default_factory=lambda: [1.5, 6.0], max_length=5
    )
    input_price_per_million_usd: float = 0.0
    output_price_per_million_usd: float = 0.0
    auth_header: str | None = None
    extra_headers: dict[str, str] = {}
    extra_query: dict[str, str] = {}
    token_limit_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"  # noqa: S105


class DuplicateBody(BaseModel):
    new_name: str


class CommandBody(BaseModel):
    name: str
    argv: list[str]
    timeout_seconds: float = 10.0
    max_output_bytes: int = 65_536
    env_allowlist: list[str] = ["PATH"]
    isolation: Literal["host", "docker"] = "host"
    docker_image: str = ""


class RunBody(BaseModel):
    default_budget_tokens: int = 100_000
    default_budget_cost_usd: float = 5.0
    stall_window_seconds: float = 0.0
    planner_mode: Literal["single", "model"] = "single"
    falsifier_mode: Literal["off", "model"] = "off"
    calibration_path: str = ""


class RoutingBody(BaseModel):
    routing: list[dict[str, Any]]  # [{"role": "worker", "profiles": ["p1"]}]


def _build_profile(b: ProfileBody) -> ModelProfile:
    return ModelProfile(
        name=b.name,
        protocol=Protocol(b.protocol),
        base_url=b.base_url,
        model=b.model,
        api_key_env=SecretEnvRef(env_var=b.api_key_env) if b.api_key_env else None,
        allow_insecure_http=b.allow_insecure_http,
        roles={Role(r) for r in b.roles} if b.roles else set(),
        capabilities={Capability(c) for c in b.capabilities} if b.capabilities else set(),
        max_concurrency=b.max_concurrency,
        requests_per_minute=b.requests_per_minute,
        context_window_tokens=b.context_window_tokens,
        max_output_tokens=b.max_output_tokens,
        request_timeout_seconds=b.request_timeout_seconds,
        strict_stream_completion=b.strict_stream_completion,
        transport_retries=b.transport_retries,
        transport_retry_backoff_seconds=b.transport_retry_backoff_seconds,
        input_price_per_million_usd=b.input_price_per_million_usd,
        output_price_per_million_usd=b.output_price_per_million_usd,
        auth_header=b.auth_header,
        extra_headers=b.extra_headers,
        extra_query=b.extra_query,
        token_limit_param=b.token_limit_param,
    )


def _build_command(b: CommandBody) -> CommandConfig:
    return CommandConfig(
        name=b.name,
        argv=b.argv,
        timeout_seconds=b.timeout_seconds,
        max_output_bytes=b.max_output_bytes,
        env_allowlist=b.env_allowlist,
        isolation=b.isolation,
        docker_image=b.docker_image,
    )


def _build_routing(entries: list[dict[str, Any]]) -> list[RouteMapping]:
    return [RouteMapping(role=Role(e["role"]), profiles=list(e["profiles"])) for e in entries]


def _bad(err: Exception) -> HTTPException:
    """Map a config validation failure to a concise, operator-readable 400."""
    if isinstance(err, ValidationError):
        messages = []
        for item in err.errors(include_url=False, include_input=False):
            message = str(item.get("msg", "Invalid configuration"))
            if message.startswith("Value error, "):
                message = message.removeprefix("Value error, ")
            location = ".".join(str(part) for part in item.get("loc", ()))
            messages.append(f"{location}: {message}" if location else message)
        return HTTPException(status_code=400, detail="; ".join(messages))
    return HTTPException(status_code=400, detail=str(err))


def _mutate(request: Request, mutate: Callable[[ConfigStore], None]) -> dict[str, Any]:
    """Apply one store edit; a rejected edit is a 400 and leaves the store unchanged."""
    try:
        mutate(_store(request))
    except (ValueError, ValidationError) as e:
        raise _bad(e) from e
    return _store(request).view()


@router.get("/config")
def get_config(request: Request) -> dict[str, Any]:
    return _store(request).view()


@router.get("/secrets/status")
def get_secrets_status(request: Request) -> dict[str, Any]:
    """Per-profile env-var name + resolved OK/UNSET. Never a value."""
    s = _store(request)
    c = s.get()
    return {
        "profiles": [
            {
                "name": p.name,
                "api_key_env": p.api_key_env.env_var if p.api_key_env else None,
                "key_status": s.profile_view(p)["key_status"],
            }
            for p in c.profiles
        ]
    }


@router.patch("/config/name")
def patch_name(body: NameBody, request: Request) -> dict[str, Any]:
    return _mutate(request, lambda s: s.update_name(body.name))


@router.put("/config/run")
def put_run(body: RunBody, request: Request) -> dict[str, Any]:
    return _mutate(
        request,
        lambda s: s.update_run(
            RunConfig(
                default_budget_tokens=body.default_budget_tokens,
                default_budget_cost_usd=body.default_budget_cost_usd,
                stall_window_seconds=body.stall_window_seconds,
                planner_mode=body.planner_mode,
                falsifier_mode=body.falsifier_mode,
                calibration_path=body.calibration_path,
            )
        ),
    )


@router.put("/config/routing")
def put_routing(body: RoutingBody, request: Request) -> dict[str, Any]:
    return _mutate(request, lambda s: s.update_routing(_build_routing(body.routing)))


@router.get("/config/routing/presets")
def get_routing_presets(request: Request) -> dict[str, Any]:
    return {"presets": _routing_presets(_store(request))}


@router.post("/config/routing/presets/{preset_id}/apply")
def apply_routing_preset(preset_id: str, request: Request) -> dict[str, Any]:
    preset = next(
        (item for item in _routing_presets(_store(request)) if item["id"] == preset_id),
        None,
    )
    if preset is None:
        raise HTTPException(status_code=404, detail=f"unknown preset: {preset_id}")
    if not preset["available"]:
        raise HTTPException(status_code=400, detail=preset["reason"])
    entries = [{"role": role, "profiles": profiles} for role, profiles in preset["routing"].items()]
    return _mutate(request, lambda s: s.update_routing(_build_routing(entries)))


@router.post("/config/profiles")
def add_profile(body: ProfileBody, request: Request) -> dict[str, Any]:
    return _mutate(request, lambda s: s.add_profile(_build_profile(body)))


@router.put("/config/profiles/{name}")
def update_profile(name: str, body: ProfileBody, request: Request) -> dict[str, Any]:
    return _mutate(request, lambda s: s.update_profile(name, _build_profile(body)))


@router.delete("/config/profiles/{name}")
def delete_profile(
    name: str, request: Request, remove_from_routing: bool = False
) -> dict[str, Any]:
    return _mutate(
        request, lambda s: s.delete_profile(name, remove_from_routing=remove_from_routing)
    )


@router.post("/config/profiles/{name}/duplicate")
def duplicate_profile(name: str, body: DuplicateBody, request: Request) -> dict[str, Any]:
    return _mutate(request, lambda s: s.duplicate_profile(name, body.new_name))


@router.post("/config/commands")
def add_command(body: CommandBody, request: Request) -> dict[str, Any]:
    try:
        _store(request).add_command(_build_command(body))
    except (ValueError, ValidationError) as e:
        raise _bad(e) from e
    return _store(request).view()


@router.put("/config/commands/{name}")
def update_command(name: str, body: CommandBody, request: Request) -> dict[str, Any]:
    try:
        _store(request).update_command(name, _build_command(body))
    except (ValueError, ValidationError) as e:
        raise _bad(e) from e
    return _store(request).view()


@router.delete("/config/commands/{name}")
def delete_command(name: str, request: Request) -> dict[str, Any]:
    try:
        _store(request).delete_command(name)
    except (ValueError, ValidationError) as e:
        raise _bad(e) from e
    return _store(request).view()


@router.post("/config/commands/{name}/test")
def test_command(name: str, request: Request) -> dict[str, Any]:
    """Dry-run a named command in a scratch dir. NOT a sandbox.

    Returns bounded stdout/stderr/exit.  Documents ``isolated: False`` so the
    UI can label it clearly.  The scratch dir is temp, not the real workspace.
    """
    s = _store(request)
    c = s.get()
    cmd = next((cmd for cmd in c.commands if cmd.name == name), None)
    if cmd is None:
        raise HTTPException(status_code=404, detail=f"unknown command: {name}")
    if not cmd.argv:
        raise HTTPException(status_code=400, detail="command has no argv")

    with tempfile.TemporaryDirectory(prefix="northstack-cmd-test-") as scratch:
        result = run_command(
            list(cmd.argv),
            cwd=scratch,
            env_allowlist=list(cmd.env_allowlist),
            timeout_seconds=cmd.timeout_seconds,
            max_output_bytes=cmd.max_output_bytes,
        )
        return {
            "name": cmd.name,
            "argv": list(cmd.argv),
            "exit_code": result.exit_code if result.exit_code is not None else -1,
            "stdout": result.stdout.decode("utf-8", errors="replace"),
            "stderr": (result.stderr.decode("utf-8", errors="replace") or result.error),
            "truncated": result.truncated,
            "isolated": False,
        }


@router.post("/config/save")
def save_config(request: Request) -> dict[str, Any]:
    path = _store(request).save_to_toml()
    return {"saved": str(path), "config": _store(request).view()}


@router.post("/config/validate")
def validate_config(request: Request) -> dict[str, Any]:
    _mutate(request, lambda s: s.validate())
    return {"valid": True, "config": _store(request).view()}


@router.post("/config/reload")
def reload_config(request: Request) -> dict[str, Any]:
    try:
        return _mutate(request, lambda s: s.reload())
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/config/reset")
def reset_config(request: Request) -> dict[str, Any]:
    _store(request).reset()
    return _store(request).view()


class TomlBody(BaseModel):
    text: str


@router.get("/config/toml")
def get_toml(request: Request) -> dict[str, Any]:
    return {"text": _store(request).toml_document()}


@router.put("/config/toml")
def put_toml(body: TomlBody, request: Request) -> dict[str, Any]:
    return _mutate(request, lambda s: s.apply_toml(body.text))
