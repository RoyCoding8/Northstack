"""Configuration loading for northstack.

Public seam: callers create a NorthStackConfig from a TOML file.
Secret values are referenced by env-var name, never resolved or logged
in model representations.  API keys must not have plaintext defaults.
"""

from __future__ import annotations

import enum
import os
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from northstack.domain.budget import Budget
from northstack.domain.container_policy import validate_docker_image
from northstack.domain.secrets_policy import is_secret_field_name, validate_env_allowlist
from northstack.domain.url_policy import validate_provider_url

NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_SAFE_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _check_unique(items: list[str], kind: str) -> None:
    folded = [item.casefold() for item in items]
    if len(folded) != len(set(folded)):
        dupes = {item for item in items if folded.count(item.casefold()) > 1}
        raise ValueError(f"Duplicate {kind}: {dupes}")


class SecretEnvRef(BaseModel):
    """Reference to a secret stored in an environment variable.

    The value is never stored in the model -- only the env-var name.
    Callers call .resolve() at the point of use.

    No default field: API key defaults defeat the purpose of env-based
    secret management.  If the env var is unset, resolve() raises.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    env_var: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="Environment variable name",
    )

    def resolve(self) -> str:
        """Read the secret from the environment. Raises KeyError if unset."""
        val = os.environ.get(self.env_var)
        if val is not None:
            return val
        raise KeyError(
            f"Environment variable {self.env_var} is not set. Set it in your shell or .env file."
        )

    def __repr__(self) -> str:
        return f"SecretEnvRef(env_var={self.env_var!r})"


class Protocol(str, enum.Enum):
    """Wire protocol for model provider communication."""

    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GEMINI_GENERATE_CONTENT = "gemini_generate_content"


class Capability(str, enum.Enum):
    """Model capabilities that affect routing decisions."""

    TOOL_USE = "tool_use"
    NATIVE_JSON_SCHEMA = "native_json_schema"
    VISION = "vision"
    STREAMING = "streaming"
    PROMPT_CACHING = "prompt_caching"


class Role(str, enum.Enum):
    """Worker roles in the northstack org chart.

    Roles are capability/responsibility slots, not personas. The operator
    assigns a named model profile to each role in ``[northstack.routing]`` so
    that menial work can run on a cheap, small-model profile while the
    biggest orchestrator runs on a strong, frontier-model profile.
    """

    WORKER = "worker"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    SPECIALIST = "specialist"
    ORCHESTRATOR = "orchestrator"


class ModelProfile(BaseModel):
    """Provider-neutral model profile.

    Describes a model endpoint with protocol, capabilities, rate limits,
    context window, and pricing -- without assuming a specific provider.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=_SAFE_IDENTIFIER,
        description="Profile identifier (e.g. cheap-worker)",
    )
    protocol: Protocol = Field(description="Wire protocol for this endpoint")
    base_url: str = Field(min_length=1, description="API base URL (OpenAI-compatible or Anthropic)")
    model: str = Field(min_length=1, description="Model identifier on the provider")
    api_key_env: SecretEnvRef | None = Field(
        default=None,
        description="Env var holding the API key; None for local/no-key endpoints",
    )
    allow_insecure_http: bool = Field(
        default=False,
        description="Allow credentials over HTTP to a non-loopback trusted proxy",
    )
    roles: set[Role] = Field(
        default_factory=lambda: {Role.WORKER},
        description="Roles this profile can fill",
    )
    capabilities: set[Capability] = Field(
        default_factory=lambda: {Capability.TOOL_USE},
        description="Model capabilities for routing decisions",
    )
    max_concurrency: int = Field(ge=1, description="Max parallel workers for this profile")
    requests_per_minute: int = Field(default=60, ge=1, description="Rate limit (RPM)")
    context_window_tokens: int = Field(default=128_000, ge=1, description="Max input context")
    max_output_tokens: int = Field(default=4_096, ge=1, description="Max output tokens")
    request_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description=(
            "Per-request HTTP timeout; reasoning models can legitimately take "
            "minutes on long prompts before the first byte arrives"
        ),
    )
    strict_stream_completion: bool = Field(
        default=True,
        description="Reject streams that end without the protocol terminal frame",
    )
    transport_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Retries for transient transport failures (connect errors, 502/503/504)",
    )
    transport_retry_backoff_seconds: list[NonNegativeFiniteFloat] = Field(
        default_factory=lambda: [1.5, 6.0],
        max_length=5,
        description="Backoff before each transport retry; entry i precedes retry i+1",
    )
    input_price_per_million_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Input cost per million tokens (USD); 0 = free/local",
    )
    output_price_per_million_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Output cost per million tokens (USD); 0 = free/local",
    )
    auth_header: str | None = Field(
        default=None,
        description=(
            "Override the header the resolved API key is written to, sent raw with "
            "no scheme prefix (Azure OpenAI wants 'api-key'). None uses the "
            "protocol default"
        ),
    )
    extra_headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Static headers added to every request to this endpoint (gateway "
            "attribution, API-version pins, beta opt-ins). Never credentials"
        ),
    )
    extra_query: dict[str, str] = Field(
        default_factory=dict,
        description="Static query parameters appended to every request URL",
    )
    token_limit_param: Literal["max_tokens", "max_completion_tokens"] = Field(
        default="max_tokens",
        description=(
            "Wire spelling of the output-token limit on openai_chat endpoints; "
            "OpenAI's own reasoning models require max_completion_tokens"
        ),
    )

    @field_validator("extra_headers")
    @classmethod
    def _reject_credential_headers(cls, v: dict[str, str]) -> dict[str, str]:
        """Credentials belong in api_key_env, which resolves at call time and is
        redacted everywhere.  A key pasted into extra_headers would sit in
        plaintext TOML and ride into every config view."""
        folded = [name.casefold() for name in v]
        if len(folded) != len(set(folded)):
            raise ValueError("extra_headers names must be unique case-insensitively")
        if invalid := sorted(name for name in v if _HEADER_NAME.fullmatch(name) is None):
            raise ValueError(f"extra_headers contains invalid header names: {', '.join(invalid)}")
        if any(any(char in value for char in "\r\n\0") for value in v.values()):
            raise ValueError("extra_headers values must not contain control characters")
        if collisions := sorted(name for name in v if is_secret_field_name(name)):
            raise ValueError(
                f"extra_headers must not carry credentials: {', '.join(collisions)}; "
                "use api_key_env (with auth_header to change the header name)"
            )
        return v

    @field_validator("auth_header")
    @classmethod
    def _validate_auth_header(cls, v: str | None) -> str | None:
        if v is not None and _HEADER_NAME.fullmatch(v) is None:
            raise ValueError("auth_header is not a valid HTTP header name")
        return v

    @field_validator("extra_query")
    @classmethod
    def _reject_credential_query(cls, v: dict[str, str]) -> dict[str, str]:
        if collisions := sorted(name for name in v if is_secret_field_name(name)):
            raise ValueError(
                f"extra_query must not carry credentials: {', '.join(collisions)}; use api_key_env"
            )
        return v

    @field_validator("api_key_env", mode="before")
    @classmethod
    def _coerce_api_key_env(cls, v: object) -> object:
        """TOML stores the env-var name as a bare string; the model holds a
        SecretEnvRef.  Coerce str -> SecretEnvRef so ``from_toml`` can validate
        the parsed table straight through pydantic without hand-mapping the
        field (the ``Field(default=...)`` on ``api_key_env`` is then the single
        source of truth for "absent means None / no key").
        """
        if isinstance(v, str):
            return SecretEnvRef(env_var=v)
        return v

    @model_validator(mode="after")
    def validate_base_url_policy(self) -> ModelProfile:
        if self.transport_retries and not self.transport_retry_backoff_seconds:
            raise ValueError("transport retries require at least one backoff duration")
        if self.auth_header and self.auth_header.casefold() in map(
            str.casefold, self.extra_headers
        ):
            raise ValueError("auth_header must not collide with extra_headers")
        validate_provider_url(
            self.base_url,
            credentialed=self.api_key_env is not None,
            allow_insecure_http=self.allow_insecure_http,
        )
        return self

    def key_status(self) -> str:
        """Env-var name + resolved OK/UNSET. Never the value. Single source."""
        if self.api_key_env is None:
            return "no key"
        name = self.api_key_env.env_var
        return f"env:{name} {'OK' if os.environ.get(name) else 'UNSET'}"

    @property
    def tier(self) -> int:
        """Derived tier based on pricing and capabilities.

        1 = cheap/fast, 2 = mid-range, 3 = expert/strong.
        Local/free endpoints always tier 1.
        """
        avg_price = (self.input_price_per_million_usd + self.output_price_per_million_usd) / 2
        if avg_price == 0.0:
            return 1
        if avg_price < 2.0:
            return 1
        if avg_price < 15.0:
            return 2
        return 3


class RunConfig(BaseModel):
    """Default run-level settings."""

    model_config = ConfigDict(allow_inf_nan=False)

    default_budget_tokens: int = Field(default=100_000, ge=0)
    default_budget_cost_usd: float = Field(default=5.0, ge=0.0)
    stall_window_seconds: float = Field(default=0.0, ge=0.0)
    planner_mode: Literal["single", "model"] = Field(default="single")
    falsifier_mode: Literal["off", "model"] = Field(default="off")
    calibration_path: str = Field(default="")
    memory_enabled: bool = Field(default=False)

    def default_budget(self) -> Budget:
        """The run budget these defaults describe. Single source for the mapping."""
        return Budget(
            token_limit=self.default_budget_tokens or None,
            cost_limit_usd=self.default_budget_cost_usd or None,
        )

    def budget_summary(self) -> str:
        """One-line rendering of :meth:`default_budget` for operator surfaces."""
        b = self.default_budget()
        tokens = f"{b.token_limit:,} tokens" if b.token_limit is not None else "unlimited tokens"
        cost = f"${b.cost_limit_usd:.2f}" if b.cost_limit_usd is not None else "unlimited cost"
        return f"{tokens}, {cost}"


class CommandConfig(BaseModel):
    """Provider-neutral named subprocess command configuration."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=60, pattern=_SAFE_IDENTIFIER)
    argv: list[str] = Field(min_length=1)
    timeout_seconds: float = Field(default=10.0, ge=0.0)
    max_output_bytes: int = Field(default=65_536, ge=0)
    env_allowlist: list[str] = Field(default_factory=lambda: ["PATH"])
    isolation: Literal["host", "docker"] = Field(default="host")
    docker_image: str = Field(default="", description="Required when isolation = docker")

    @field_validator("env_allowlist", mode="before")
    @classmethod
    def _validate_env_allowlist(cls, v: list[str]) -> list[str]:
        """Reject sensitive env names -- mirrors CommandProfile."""
        return validate_env_allowlist(v)

    @model_validator(mode="after")
    def _validate_docker_isolation(self) -> CommandConfig:
        if self.isolation == "docker":
            try:
                validate_docker_image(self.docker_image)
            except ValueError as error:
                raise ValueError(f"command {self.name!r}: {error}") from error
        return self


class RouteMapping(BaseModel):
    """Explicit role -> ordered model-profile assignment.

    The operator edits this table (not profile pricing) to decide which
    model runs at each level: a cheap, small-model profile for ``worker``,
    a strong, frontier-model profile for ``orchestrator``, and so on. Within a
    role, profiles are tried in the order written (a fallback chain); a
    profile named here must also declare the corresponding ``Role`` tag as
    a sanity guard so a mislabelled profile cannot silently serve a role.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role = Field(description="Role this assignment governs")
    profiles: list[str] = Field(
        min_length=1,
        description="Ordered profile names eligible for this role (fallback chain)",
    )


class NorthStackConfig(BaseModel):
    """Top-level configuration loaded from TOML."""

    name: str = Field(min_length=1)
    profiles: list[ModelProfile] = Field(default_factory=list)
    commands: list[CommandConfig] = Field(default_factory=list)
    run: RunConfig = Field(default_factory=RunConfig)
    routing: list[RouteMapping] = Field(
        default_factory=list,
        description="Explicit role -> ordered profile names; empty = legacy role-tag filtering",
    )

    def profile(self, name: str) -> ModelProfile | None:
        """Lookup a model profile by name; None if absent."""
        for profile in self.profiles:
            if profile.name == name:
                return profile
        return None

    def role_map(self) -> dict[Role, list[str]]:
        """role -> ordered profile names from the routing table."""
        return {entry.role: list(entry.profiles) for entry in self.routing}

    @model_validator(mode="after")
    def _validate_unique_names(self) -> NorthStackConfig:
        _check_unique([p.name for p in self.profiles], "profile names")
        _check_unique([c.name for c in self.commands], "command names")
        return self

    @model_validator(mode="after")
    def _validate_routing(self) -> NorthStackConfig:
        profile_names = {p.name for p in self.profiles}
        seen_roles: set[Role] = set()
        for entry in self.routing:
            if entry.role in seen_roles:
                raise ValueError(f"Duplicate routing entry for role {entry.role.value!r}")
            seen_roles.add(entry.role)
            missing = [n for n in entry.profiles if n not in profile_names]
            if missing:
                raise ValueError(
                    f"routing entry for role {entry.role.value!r} references "
                    f"unknown profile(s): {missing}"
                )
            for name in entry.profiles:
                profile = next(p for p in self.profiles if p.name == name)
                if entry.role not in profile.roles:
                    raise ValueError(
                        f"profile {name!r} is routed to role {entry.role.value!r} "
                        f"but does not declare that role "
                        f"(roles={sorted(r.value for r in profile.roles)})"
                    )
        return self

    @classmethod
    def from_toml(cls, path: Path) -> NorthStackConfig:
        """Load configuration from a TOML file.

        The parsed ``[northstack]`` table is validated straight through pydantic,
        so the ``Field(default=...)`` declarations ARE the defaults -- there is no
        second hand-mapped source of truth (no ``p.get("requests_per_minute", 60)``
        duplicating the ``Field(default=60)`` above).  ``profiles``, ``commands``,
        ``run`` and ``routing`` are passed as raw tables/lists and pydantic coerces
        enums, sets, ``SecretEnvRef`` (via the ``api_key_env`` before-validator)
        and applies every default itself.

        Uses stdlib tomllib (Python 3.11+) -- no extra dependency needed.
        """
        import tomllib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        company = raw.get("northstack", {})
        return cls.model_validate(company)
