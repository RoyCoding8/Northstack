"""Tests for configuration loading at the public seam."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

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
from northstack.domain import BudgetUsage

# SecretEnvRef (no default allowed)


class TestSecretEnvRef:
    def test_create_ref(self):
        ref = SecretEnvRef(env_var="MY_API_KEY")
        assert ref.env_var == "MY_API_KEY"

    def test_no_default_field(self):
        """SecretEnvRef must not accept a default -- API keys come from env only."""
        with pytest.raises(ValidationError):
            SecretEnvRef(env_var="KEY", default="sk-plaintext")  # type: ignore[call-arg]

    def test_resolve_reads_env(self, monkeypatch):
        monkeypatch.setenv("TEST_SECRET_KEY", "secret-value-123")
        ref = SecretEnvRef(env_var="TEST_SECRET_KEY")
        assert ref.resolve() == "secret-value-123"

    def test_resolve_missing_raises(self):
        ref = SecretEnvRef(env_var="NONEXISTENT_KEY_FOR_TEST_99999")
        with pytest.raises(KeyError):
            ref.resolve()

    def test_repr_hides_no_value(self):
        ref = SecretEnvRef(env_var="HIDDEN_KEY")
        r = repr(ref)
        assert "HIDDEN_KEY" in r

    def test_frozen(self):
        ref = SecretEnvRef(env_var="KEY")
        with pytest.raises(ValidationError):
            ref.env_var = "OTHER"  # type: ignore[misc]

    @pytest.mark.parametrize("name", ["BAD NAME", "1START", "BAD=VALUE", "BAD\nNAME"])
    def test_rejects_nonportable_environment_names(self, name):
        with pytest.raises(ValidationError):
            SecretEnvRef(env_var=name)


# ModelProfile (provider-neutral)


class TestModelProfile:
    @pytest.mark.parametrize("name", ["../escape", "space name", "<script>", "a" * 65])
    def test_rejects_unsafe_profile_names(self, name):
        with pytest.raises(ValidationError):
            ModelProfile(
                name=name,
                protocol=Protocol.OPENAI_CHAT,
                base_url="https://example.com/v1",
                model="model",
                max_concurrency=1,
            )

    @pytest.mark.parametrize(
        "values",
        [
            {"request_timeout_seconds": float("inf")},
            {"transport_retry_backoff_seconds": [float("inf")]},
            {"transport_retry_backoff_seconds": [-1.0]},
            {"input_price_per_million_usd": float("inf")},
            {"transport_retries": 1, "transport_retry_backoff_seconds": []},
        ],
    )
    def test_rejects_non_finite_transport_values(self, values):
        with pytest.raises(ValidationError):
            ModelProfile(
                name="invalid",
                protocol=Protocol.OPENAI_CHAT,
                base_url="https://example.com/v1",
                model="model",
                max_concurrency=1,
                **values,
            )

    def test_create_profile(self):
        p = ModelProfile(
            name="cheap-worker",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://localhost:8080/v1",
            model="mimo-v2.5",
            max_concurrency=8,
            roles={Role.WORKER},
            capabilities={Capability.TOOL_USE},
        )
        assert p.name == "cheap-worker"
        assert p.protocol == Protocol.OPENAI_CHAT
        assert p.max_concurrency == 8
        assert p.tier == 1  # local/free endpoint

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("extra_headers", {"X-Client-Secret": "plaintext"}),
            ("extra_headers", {"X-Access-Token": "plaintext"}),
            ("extra_query", {"api-key": "plaintext"}),
            ("extra_query", {"client_secret": "plaintext"}),
            ("extra_query", {"sig": "plaintext"}),
        ],
    )
    def test_rejects_plaintext_credentials_in_static_transport_fields(self, field, value):
        with pytest.raises(ValidationError, match=field):
            ModelProfile(
                name="unsafe",
                protocol=Protocol.OPENAI_CHAT,
                base_url="https://example.com/v1",
                model="model",
                max_concurrency=1,
                **{field: value},
            )

    def test_rejects_extra_header_colliding_with_custom_auth_header(self):
        with pytest.raises(ValidationError, match="auth_header"):
            ModelProfile(
                name="unsafe",
                protocol=Protocol.OPENAI_CHAT,
                base_url="https://example.com/v1",
                model="model",
                max_concurrency=1,
                auth_header="X-Custom-Auth",
                extra_headers={"x-custom-auth": "plaintext"},
            )

    @pytest.mark.parametrize(
        "values",
        [
            {"auth_header": "Bad Header"},
            {"extra_headers": {"Bad Header": "value"}},
            {"extra_headers": {"X-Test": "safe\r\nInjected: value"}},
            {"extra_headers": {"X-Test": "one", "x-test": "two"}},
        ],
    )
    def test_rejects_invalid_custom_headers(self, values):
        with pytest.raises(ValidationError):
            ModelProfile(
                name="invalid",
                protocol=Protocol.OPENAI_CHAT,
                base_url="https://example.com/v1",
                model="model",
                max_concurrency=1,
                **values,
            )

    @pytest.mark.parametrize(
        "base_url",
        [
            "ftp://example.com",
            "https://user:pass@example.com/v1",
            "https://example.com/v1?key=value",
            "https://example.com/v1#fragment",
            "http:///missing-host",
        ],
    )
    def test_rejects_invalid_base_url_at_construction(self, base_url):
        with pytest.raises(ValidationError, match="base_url"):
            ModelProfile(
                name="invalid",
                protocol=Protocol.OPENAI_CHAT,
                base_url=base_url,
                model="model",
                max_concurrency=1,
            )

    def test_credentialed_remote_http_requires_explicit_opt_in(self):
        with pytest.raises(ValidationError, match="must use https"):
            ModelProfile(
                name="remote-http",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://proxy.example.com/v1",
                model="model",
                api_key_env=SecretEnvRef(env_var="API_KEY"),
                max_concurrency=1,
            )

        profile = ModelProfile(
            name="trusted-proxy",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://proxy.example.com/v1",
            model="model",
            api_key_env=SecretEnvRef(env_var="API_KEY"),
            allow_insecure_http=True,
            max_concurrency=1,
        )
        assert profile.allow_insecure_http is True

    @pytest.mark.parametrize(
        "base_url", ["http://localhost:8080/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"]
    )
    def test_credentialed_loopback_http_is_allowed(self, base_url):
        profile = ModelProfile(
            name="local",
            protocol=Protocol.OPENAI_CHAT,
            base_url=base_url,
            model="model",
            api_key_env=SecretEnvRef(env_var="API_KEY"),
            max_concurrency=1,
        )
        assert profile.base_url == base_url

    def test_profile_with_api_key(self):
        p = ModelProfile(
            name="with-key",
            protocol=Protocol.OPENAI_CHAT,
            base_url="https://api.example.com/v1",
            model="model-a",
            api_key_env=SecretEnvRef(env_var="API_KEY"),
            max_concurrency=4,
        )
        assert p.api_key_env is not None
        assert p.api_key_env.env_var == "API_KEY"

    def test_profile_no_api_key_local(self):
        p = ModelProfile(
            name="local",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://localhost:11434/v1",
            model="llama3",
            api_key_env=None,
            max_concurrency=1,
        )
        assert p.api_key_env is None

    def test_derived_tier_pricing(self):
        cheap = ModelProfile(
            name="cheap",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            max_concurrency=1,
            input_price_per_million_usd=0.5,
            output_price_per_million_usd=1.5,
        )
        mid = ModelProfile(
            name="mid",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            max_concurrency=1,
            input_price_per_million_usd=5.0,
            output_price_per_million_usd=10.0,
        )
        expert = ModelProfile(
            name="expert",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            max_concurrency=1,
            input_price_per_million_usd=15.0,
            output_price_per_million_usd=75.0,
        )
        assert cheap.tier == 1
        assert mid.tier == 2
        assert expert.tier == 3

    def test_profile_frozen(self):
        p = ModelProfile(
            name="x",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            max_concurrency=1,
        )
        with pytest.raises(ValidationError):
            p.name = "y"  # type: ignore[misc]

    def test_invalid_protocol_rejected(self):
        with pytest.raises(ValidationError):
            ModelProfile(
                name="x",
                protocol="bogus",  # type: ignore[arg-type]
                base_url="http://x",
                model="m",
                max_concurrency=1,
            )


# NorthStackConfig: loading from TOML


class TestNorthStackConfig:
    def test_load_from_toml_file(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            [northstack]
            name = "TestCo"

            [northstack.run]
            default_budget_tokens = 50000
            default_budget_cost_usd = 2.0

            [[northstack.commands]]
            name = "test"
            argv = ["python", "-m", "pytest", "-q"]
            timeout_seconds = 120.0
            max_output_bytes = 131072
            env_allowlist = ["PATH"]

            [[northstack.profiles]]
            name = "cheap-worker"
            protocol = "openai_chat"
            base_url = "http://localhost:8080/v1"
            model = "mimo-v2.5"
            max_concurrency = 8
            api_key_env = "MIMO_API_KEY"
            roles = ["worker"]
            capabilities = ["tool_use"]
            requests_per_minute = 120
            context_window_tokens = 128000
            max_output_tokens = 8192

            [[northstack.profiles]]
            name = "expert"
            protocol = "openai_chat"
            base_url = "http://localhost:8090/v1"
            model = "glm-5.2"
            max_concurrency = 1
            api_key_env = "GLM_API_KEY"
            roles = ["planner", "reviewer"]
            capabilities = ["tool_use", "native_json_schema"]
            requests_per_minute = 30
            context_window_tokens = 200000
            max_output_tokens = 16384
            input_price_per_million_usd = 10.0
            output_price_per_million_usd = 30.0
        """)
        config_path = tmp_path / "northstack.toml"
        config_path.write_text(toml_content)

        config = NorthStackConfig.from_toml(config_path)
        assert config.name == "TestCo"
        assert len(config.profiles) == 2
        assert config.profiles[0].name == "cheap-worker"
        assert config.profiles[0].protocol == Protocol.OPENAI_CHAT
        assert config.profiles[0].max_concurrency == 8
        assert config.profiles[0].tier == 1
        assert config.profiles[1].name == "expert"
        assert config.profiles[1].max_concurrency == 1
        assert Role.PLANNER in config.profiles[1].roles
        assert config.commands[0].name == "test"
        assert config.commands[0].argv == ["python", "-m", "pytest", "-q"]
        assert config.run.default_budget_tokens == 50000

    def test_local_endpoint_no_api_key(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            [northstack]
            name = "LocalCo"

            [[northstack.profiles]]
            name = "local"
            protocol = "openai_chat"
            base_url = "http://localhost:11434/v1"
            model = "llama3"
            max_concurrency = 4
        """)
        config_path = tmp_path / "local.toml"
        config_path.write_text(toml_content)
        config = NorthStackConfig.from_toml(config_path)
        assert config.profiles[0].api_key_env is None

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            NorthStackConfig.from_toml(tmp_path / "nope.toml")

    def test_invalid_toml_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.toml"
        bad.write_text("this is not [valid toml {{{")
        with pytest.raises(Exception):
            NorthStackConfig.from_toml(bad)

    def test_config_validation_rejects_bad_profile(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            [northstack]
            name = "BadCo"

            [[northstack.profiles]]
            name = "bad"
            protocol = "bogus_protocol"
            base_url = "http://x"
            model = "m"
            max_concurrency = 1
        """)
        config_path = tmp_path / "bad.toml"
        config_path.write_text(toml_content)
        with pytest.raises(Exception):
            NorthStackConfig.from_toml(config_path)

    def test_routing_table_parsed_and_role_map_built(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            [northstack]
            name = "RoutedCo"

            [[northstack.profiles]]
            name = "cheap-worker"
            protocol = "openai_chat"
            base_url = "http://x"
            model = "m"
            max_concurrency = 8
            roles = ["worker"]

            [[northstack.profiles]]
            name = "opus"
            protocol = "anthropic_messages"
            base_url = "http://y"
            model = "claude-opus"
            max_concurrency = 1
            roles = ["orchestrator"]

            [[northstack.routing]]
            role = "worker"
            profiles = ["cheap-worker"]

            [[northstack.routing]]
            role = "orchestrator"
            profiles = ["opus"]
        """)
        config_path = tmp_path / "routed.toml"
        config_path.write_text(toml_content)
        config = NorthStackConfig.from_toml(config_path)

        assert config.role_map()[Role.WORKER] == ["cheap-worker"]
        assert config.role_map()[Role.ORCHESTRATOR] == ["opus"]

    def test_routing_rejects_unknown_profile(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            [northstack]
            name = "BadRoute"

            [[northstack.profiles]]
            name = "cheap-worker"
            protocol = "openai_chat"
            base_url = "http://x"
            model = "m"
            max_concurrency = 8
            roles = ["worker"]

            [[northstack.routing]]
            role = "worker"
            profiles = ["does-not-exist"]
        """)
        config_path = tmp_path / "badroute.toml"
        config_path.write_text(toml_content)
        with pytest.raises(ValidationError):
            NorthStackConfig.from_toml(config_path)

    def test_routing_rejects_role_tag_mismatch(self, tmp_path: Path):
        toml_content = textwrap.dedent("""\
            [northstack]
            name = "Mismatch"

            [[northstack.profiles]]
            name = "cheap-worker"
            protocol = "openai_chat"
            base_url = "http://x"
            model = "m"
            max_concurrency = 8
            roles = ["worker"]

            [[northstack.routing]]
            role = "orchestrator"
            profiles = ["cheap-worker"]
        """)
        config_path = tmp_path / "mismatch.toml"
        config_path.write_text(toml_content)
        with pytest.raises(ValidationError):
            NorthStackConfig.from_toml(config_path)

    def test_routing_rejects_duplicate_role(self):
        cheap = ModelProfile(
            name="cheap-worker",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            max_concurrency=8,
            roles={Role.WORKER},
        )
        from northstack.config import RouteMapping

        with pytest.raises(ValidationError):
            NorthStackConfig(
                name="dup",
                profiles=[cheap],
                routing=[
                    RouteMapping(role=Role.WORKER, profiles=["cheap-worker"]),
                    RouteMapping(role=Role.WORKER, profiles=["cheap-worker"]),
                ],
            )


class TestCommandConfigEnvAllowlist:
    @pytest.mark.parametrize("name", ["../escape", "space name", "<script>", "a" * 61])
    def test_rejects_unsafe_command_names(self, name):
        with pytest.raises(ValidationError):
            CommandConfig(name=name, argv=["true"])


def test_config_names_are_unique_case_insensitively() -> None:
    profiles = [
        ModelProfile(
            name=name,
            protocol=Protocol.OPENAI_CHAT,
            base_url="https://example.com/v1",
            model="model",
            max_concurrency=1,
        )
        for name in ("Worker", "worker")
    ]
    with pytest.raises(ValidationError, match="Duplicate profile names"):
        NorthStackConfig(name="company", profiles=profiles)
    with pytest.raises(ValidationError, match="Duplicate command names"):
        NorthStackConfig(
            name="company",
            commands=[CommandConfig(name=name, argv=["true"]) for name in ("Lint", "lint")],
        )

    """CommandConfig must reject sensitive env names, mirroring CommandProfile."""

    @pytest.mark.parametrize("name", ["API_KEY", "MY_SECRET", "AUTH_TOKEN", "APIKEY"])
    def test_rejects_sensitive_env_name(self, name: str):
        with pytest.raises(ValidationError, match="sensitive"):
            CommandConfig(name="c", argv=["x"], env_allowlist=[name])

    def test_rejects_glued_secret_suffix(self):
        for name in ["APIKEY", "MYSECRET", "MYAUTH"]:
            with pytest.raises(ValidationError, match="sensitive"):
                CommandConfig(name="c", argv=["x"], env_allowlist=[name])

    def test_accepts_benign_names(self):
        cmd = CommandConfig(name="c", argv=["x"], env_allowlist=["PATH", "MY_CUSTOM_VAR"])
        assert cmd.env_allowlist == ["PATH", "MY_CUSTOM_VAR"]


class TestRunConfigBudget:
    """0 on a budget axis is how TOML spells the domain's ``None`` (unlimited)."""

    def test_finite_defaults_map_through(self):
        b = RunConfig().default_budget()
        assert (b.token_limit, b.cost_limit_usd) == (100_000, 5.0)

    @pytest.mark.parametrize(
        ("tokens", "cost", "expected"),
        [
            (0, 5.0, (None, 5.0)),
            (100_000, 0.0, (100_000, None)),
            (0, 0.0, (None, None)),
        ],
    )
    def test_zero_axis_means_unlimited(self, tokens, cost, expected):
        b = RunConfig(default_budget_tokens=tokens, default_budget_cost_usd=cost).default_budget()
        assert (b.token_limit, b.cost_limit_usd) == expected

    def test_unlimited_budget_never_exceeded(self):
        budget = RunConfig(default_budget_tokens=0, default_budget_cost_usd=0.0).default_budget()
        usage = BudgetUsage(total_tokens=10**12, total_cost_usd=10**6)
        assert usage.exceeds(budget) is False

    def test_zero_from_toml(self, tmp_path: Path):
        path = tmp_path / "unlimited.toml"
        path.write_text(
            textwrap.dedent("""\
                [northstack]
                name = "UnlimitedCo"

                [northstack.run]
                default_budget_tokens = 0
                default_budget_cost_usd = 0.0

                [[northstack.profiles]]
                name = "local"
                protocol = "openai_chat"
                base_url = "http://localhost:11434/v1"
                model = "llama3"
                max_concurrency = 1
            """)
        )
        budget = NorthStackConfig.from_toml(path).run.default_budget()
        assert budget.token_limit is None
        assert budget.cost_limit_usd is None

    def test_summary_names_the_unlimited_axes(self):
        assert RunConfig().budget_summary() == "100,000 tokens, $5.00"
        assert (
            RunConfig(default_budget_tokens=0, default_budget_cost_usd=0.0).budget_summary()
            == "unlimited tokens, unlimited cost"
        )

    def test_negative_still_rejected(self):
        with pytest.raises(ValidationError):
            RunConfig(default_budget_tokens=-1)
