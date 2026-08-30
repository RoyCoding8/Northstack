"""In-memory editable ``NorthStackConfig`` store for the web control surface.

Editing discipline:
  - The live config is a frozen ``NorthStackConfig``.  Every mutation builds
    new lists and constructs a fresh ``NorthStackConfig(**fields)``, which
    runs ALL existing pydantic validators (unique names, routing-references-
    known-profile, role-declared-on-profile, duplicate routing entries) as the
    validation gate.  Invalid edits raise ``ValidationError`` -> the store is
    left UNCHANGED (the old config survives) and the API returns 400.
  - Edits live in memory until an explicit ``save_to_toml``; ``reload`` discards
    in-memory edits back to the on-disk file.

Secrets:
  - The store NEVER accepts or holds a secret VALUE -- only the env-var name
    string.  ``SecretEnvRef`` carries the name; resolution (OK/UNSET) is
    computed on read via ``key_status``, mirroring ``tui._key_status``.

Section preservation (important): the on-disk ``northstack.toml`` may carry
operator-authored sections the model does NOT know (e.g. ``[northstack.workspace]``,
``[northstack.web_fetch]``).  ``from_toml`` silently drops them.  Naively writing
the serialized config would therefore LOSE those sections on save.  To avoid
that footgun, ``save_to_toml`` preserves any top-level ``[northstack.*]`` section
that the model does not own by re-parsing the existing file (if present) and
splicing the unknown sections back into the emitted document.
"""

from __future__ import annotations

import threading
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from northstack.adapters.atomic_io import atomic_write_text
from northstack.adapters.config_toml import config_to_toml
from northstack.config import (
    CommandConfig,
    ModelProfile,
    NorthStackConfig,
    RouteMapping,
    RunConfig,
    SecretEnvRef,
)

_MODELED_COMPANY_KEYS = frozenset({"name", "profiles", "commands", "run", "routing"})


def key_status(profile: ModelProfile) -> str:
    """Delegate to ModelProfile.key_status (single source)."""
    return profile.key_status()


class ConfigStore:
    """Thread-safe in-memory editable config with explicit TOML persistence."""

    def __init__(self, config: NorthStackConfig, toml_path: Path) -> None:
        self._lock = threading.Lock()
        self._config = config.model_copy(deep=True)
        self._toml_path = Path(toml_path)
        self._dirty = False
        self._unknown = self._unknown_from_company(self._read_existing_company() or {})

    def get(self) -> NorthStackConfig:
        """Return a snapshot of the current in-memory config (frozen, safe to share)."""
        with self._lock:
            return self._config.model_copy(deep=True)

    def unsaved(self) -> bool:
        """True if in-memory edits have not been written to TOML since the last save/reload."""
        with self._lock:
            return self._dirty

    @staticmethod
    def profile_view(profile: ModelProfile) -> dict[str, Any]:
        """A UI-safe view of a profile: never a secret value, includes derived tier + key_status."""
        return {
            "name": profile.name,
            "protocol": profile.protocol.value,
            "base_url": profile.base_url,
            "model": profile.model,
            "api_key_env": profile.api_key_env.env_var if profile.api_key_env else None,
            "allow_insecure_http": profile.allow_insecure_http,
            "key_status": key_status(profile),
            "roles": sorted(r.value for r in profile.roles),
            "capabilities": sorted(c.value for c in profile.capabilities),
            "max_concurrency": profile.max_concurrency,
            "requests_per_minute": profile.requests_per_minute,
            "context_window_tokens": profile.context_window_tokens,
            "max_output_tokens": profile.max_output_tokens,
            "request_timeout_seconds": profile.request_timeout_seconds,
            "strict_stream_completion": profile.strict_stream_completion,
            "transport_retries": profile.transport_retries,
            "transport_retry_backoff_seconds": list(profile.transport_retry_backoff_seconds),
            "input_price_per_million_usd": profile.input_price_per_million_usd,
            "output_price_per_million_usd": profile.output_price_per_million_usd,
            "auth_header": profile.auth_header,
            "extra_headers": dict(profile.extra_headers),
            "extra_query": dict(profile.extra_query),
            "token_limit_param": profile.token_limit_param,
            "tier": profile.tier,
        }

    def view(self) -> dict[str, Any]:
        """Full UI-safe config view with per-profile key_status + derived tier + unsaved flag."""
        with self._lock:
            c = self._config
            return {
                "name": c.name,
                "profiles": [self.profile_view(p) for p in c.profiles],
                "commands": [
                    {
                        "name": cmd.name,
                        "argv": list(cmd.argv),
                        "timeout_seconds": cmd.timeout_seconds,
                        "max_output_bytes": cmd.max_output_bytes,
                        "env_allowlist": list(cmd.env_allowlist),
                        "isolation": cmd.isolation,
                        "docker_image": cmd.docker_image,
                    }
                    for cmd in c.commands
                ],
                "run": {
                    "default_budget_tokens": c.run.default_budget_tokens,
                    "default_budget_cost_usd": c.run.default_budget_cost_usd,
                    "stall_window_seconds": c.run.stall_window_seconds,
                    "planner_mode": c.run.planner_mode,
                    "falsifier_mode": c.run.falsifier_mode,
                    "calibration_path": c.run.calibration_path,
                },
                "routing": [
                    {"role": e.role.value, "profiles": list(e.profiles)} for e in c.routing
                ],
                "unsaved": self._dirty,
            }

    def _replace(self, new_config: NorthStackConfig) -> None:
        with self._lock:
            self._config = new_config
            self._dirty = True

    def _try(self, builder: Callable[[NorthStackConfig], NorthStackConfig]) -> NorthStackConfig:
        """Build a candidate config via ``builder(current)``; validate by
        construction; on success replace + return; on ValidationError raise it
        (store unchanged).  Any other exception propagates."""
        with self._lock:
            candidate = builder(self._config)
            self._config = candidate
            self._dirty = True
            return candidate

    def _rebuild(self, c: NorthStackConfig, **changes: Any) -> NorthStackConfig:
        """Copy ``c`` with only ``changes`` replaced; untouched lists are defensively copied."""
        fields: dict[str, Any] = {
            "name": c.name,
            "run": c.run,
            "profiles": list(c.profiles),
            "routing": list(c.routing),
            "commands": list(c.commands),
        }
        return NorthStackConfig(**{**fields, **changes})

    def update_name(self, name: str) -> None:
        if not name or not name.strip():
            raise ValueError("name must be non-empty")
        self._try(lambda c: self._rebuild(c, name=name))

    def add_profile(self, profile: ModelProfile) -> None:
        def build(c: NorthStackConfig) -> NorthStackConfig:
            names = [p.name for p in c.profiles] + [profile.name]
            if len(names) != len(set(names)):
                raise ValueError(f"profile name already exists: {profile.name}")
            return self._rebuild(c, profiles=[*c.profiles, profile])

        self._try(build)

    def update_profile(self, name: str, profile: ModelProfile) -> None:
        def build(c: NorthStackConfig) -> NorthStackConfig:
            if not c.profile(name):
                raise ValueError(f"unknown profile: {name}")
            if profile.name != name and c.profile(profile.name):
                raise ValueError(f"profile name already exists: {profile.name}")
            new_profiles = [(profile if p.name == name else p) for p in c.profiles]
            new_routing = [
                RouteMapping(
                    role=e.role,
                    profiles=[(profile.name if n == name else n) for n in e.profiles],
                )
                for e in c.routing
            ]
            return self._rebuild(c, profiles=new_profiles, routing=new_routing)

        self._try(build)

    def delete_profile(self, name: str, *, remove_from_routing: bool = False) -> None:
        """Delete a profile, optionally removing its routing references atomically.

        The default remains fail-safe for non-UI callers.  The web UI opts into
        ``remove_from_routing`` after an explicit confirmation so a routed
        profile can be removed without a fragile two-request partial update.
        Routing entries whose final profile is removed are dropped entirely.
        """

        def build(c: NorthStackConfig) -> NorthStackConfig:
            if not c.profile(name):
                raise ValueError(f"unknown profile: {name}")
            referenced_roles = [e.role.value for e in c.routing if name in e.profiles]
            if referenced_roles and not remove_from_routing:
                raise ValueError(
                    f"cannot delete profile {name!r}: still routed to role(s) "
                    f"{referenced_roles}. Remove it from routing first."
                )
            new_routing = []
            for entry in c.routing:
                remaining = [
                    profile_name for profile_name in entry.profiles if profile_name != name
                ]
                if remaining:
                    new_routing.append(RouteMapping(role=entry.role, profiles=remaining))
            return self._rebuild(
                c, profiles=[p for p in c.profiles if p.name != name], routing=new_routing
            )

        self._try(build)

    def duplicate_profile(self, name: str, new_name: str) -> None:
        """Clone an existing profile under ``new_name`` (a template starting point)."""

        def build(c: NorthStackConfig) -> NorthStackConfig:
            src = c.profile(name)
            if src is None:
                raise ValueError(f"unknown profile: {name}")
            if c.profile(new_name):
                raise ValueError(f"profile name already exists: {new_name}")
            clone = ModelProfile(
                name=new_name,
                protocol=src.protocol,
                base_url=src.base_url,
                model=src.model,
                api_key_env=SecretEnvRef(env_var=src.api_key_env.env_var)
                if src.api_key_env is not None
                else None,
                allow_insecure_http=src.allow_insecure_http,
                roles=set(src.roles),
                capabilities=set(src.capabilities),
                max_concurrency=src.max_concurrency,
                requests_per_minute=src.requests_per_minute,
                context_window_tokens=src.context_window_tokens,
                max_output_tokens=src.max_output_tokens,
                input_price_per_million_usd=src.input_price_per_million_usd,
                output_price_per_million_usd=src.output_price_per_million_usd,
                auth_header=src.auth_header,
                extra_headers=dict(src.extra_headers),
                extra_query=dict(src.extra_query),
                token_limit_param=src.token_limit_param,
            )
            return self._rebuild(c, profiles=[*c.profiles, clone])

        self._try(build)

    def add_command(self, command: CommandConfig) -> None:
        def build(c: NorthStackConfig) -> NorthStackConfig:
            names = [cmd.name for cmd in c.commands] + [command.name]
            if len(names) != len(set(names)):
                raise ValueError(f"command name already exists: {command.name}")
            return self._rebuild(c, commands=[*c.commands, command])

        self._try(build)

    def update_command(self, name: str, command: CommandConfig) -> None:
        def build(c: NorthStackConfig) -> NorthStackConfig:
            if not any(cmd.name == name for cmd in c.commands):
                raise ValueError(f"unknown command: {name}")
            if command.name != name and any(cmd.name == command.name for cmd in c.commands):
                raise ValueError(f"command name already exists: {command.name}")
            new_commands = [(command if cmd.name == name else cmd) for cmd in c.commands]
            return self._rebuild(c, commands=new_commands)

        self._try(build)

    def delete_command(self, name: str) -> None:
        def build(c: NorthStackConfig) -> NorthStackConfig:
            if not any(cmd.name == name for cmd in c.commands):
                raise ValueError(f"unknown command: {name}")
            return self._rebuild(c, commands=[cmd for cmd in c.commands if cmd.name != name])

        self._try(build)

    def update_run(self, run: RunConfig) -> None:
        self._try(lambda c: self._rebuild(c, run=run))

    def update_routing(self, routing: list[RouteMapping]) -> None:
        self._try(lambda c: self._rebuild(c, routing=list(routing)))

    def validate(self) -> None:
        """Re-run validators on the current in-memory config (dry-run save).

        Construction already validated on every edit, so this is a cheap
        no-op that re-asserts the current state is valid.  Raises
        ``ValidationError`` only if the stored object somehow became invalid.
        """
        with self._lock:
            NorthStackConfig(
                name=self._config.name,
                profiles=list(self._config.profiles),
                commands=list(self._config.commands),
                run=self._config.run,
                routing=list(self._config.routing),
            )

    def toml_document(self) -> str:
        """The full in-memory document: modeled fields plus unknown sections."""
        with self._lock:
            return self._merge_locked(self._config)

    def apply_toml(self, text: str) -> None:
        """Replace in-memory config from a full TOML document.

        Modeled ``[northstack]`` keys validate through ``NorthStackConfig``.
        Any other ``[northstack.*]`` table is kept as an unknown section and
        written back on save, so the UI can edit workspace/web_fetch too.
        """
        try:
            raw = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"invalid TOML: {e}") from e
        config, unknown = self._parse_raw(raw)
        with self._lock:
            self._config = config
            self._unknown = unknown
            self._dirty = True

    def save_to_toml(self) -> Path:
        """Persist the current in-memory config to ``toml_path``.

        Preserves unknown top-level ``[northstack.*]`` sections held in memory
        (loaded from disk, or replaced via ``apply_toml``) so saving via the
        web UI does not silently drop operator-authored policy.
        Returns the written path and clears the dirty flag.
        """
        with self._lock:
            new_text = self._merge_locked(self._config)
            atomic_write_text(self._toml_path, new_text)
            self._dirty = False
            return self._toml_path

    def _merge_locked(self, config: NorthStackConfig) -> str:
        """Emit the modeled config, splicing back unknown [northstack.*] sections."""
        modeled_text = config_to_toml(config)
        modeled = tomllib.loads(modeled_text)
        company = dict(modeled["northstack"])
        for key, val in self._unknown.items():
            if key not in company:
                company[key] = val
        modeled["northstack"] = company
        import tomli_w

        return tomli_w.dumps(modeled)

    @staticmethod
    def _unknown_from_company(company: dict[str, Any]) -> dict[str, Any]:
        return {key: val for key, val in company.items() if key not in _MODELED_COMPANY_KEYS}

    @classmethod
    def _parse_raw(cls, raw: dict[str, Any]) -> tuple[NorthStackConfig, dict[str, Any]]:
        company = raw.get("northstack")
        if not isinstance(company, dict):
            raise ValueError("TOML must contain a [northstack] table")
        modeled = {key: val for key, val in company.items() if key in _MODELED_COMPANY_KEYS}
        return NorthStackConfig.model_validate(modeled), cls._unknown_from_company(company)

    def _read_existing_company(self) -> dict[str, Any] | None:
        if not self._toml_path.exists():
            return None
        try:
            with open(self._toml_path, "rb") as f:
                raw = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            return None
        company = raw.get("northstack")
        return company if isinstance(company, dict) else None

    def reload(self) -> None:
        """Discard in-memory edits; reload from ``toml_path``."""
        with self._lock:
            if not self._toml_path.exists():
                raise FileNotFoundError(f"config file not found: {self._toml_path}")
            with open(self._toml_path, "rb") as file:
                config, unknown = self._parse_raw(tomllib.load(file))
            self._config, self._unknown = config, unknown
            self._dirty = False

    def reset(self) -> None:
        """Reset the in-memory config to a minimal default (name only)."""
        with self._lock:
            self._config = NorthStackConfig(name=self._config.name)
            self._dirty = True
