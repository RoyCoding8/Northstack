"""FastAPI control-surface API tests (no live provider).

Exercises the real app end-to-end via fastapi.testclient.TestClient over
the in-process ASGI transport.  No network: run tests use a config with NO
worker-role profile, so build_company selects the DeterministicAnalysisRunner
(the pipeline's hermetic fixture path) and the company never opens a socket
to a model provider -- the run still produces a ledger of events + a
terminal outcome (abstained) for the history/compare/integrity/event-poll
assertions.

Live-API tests (a real provider on the wire) live at the bottom, gated on
the MC_LIVE_API env toggle via the live marker -- off by default, see
conftest.py.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from northstack.config import (
    ModelProfile,
    NorthStackConfig,
    Protocol,
    Role,
    RunConfig,
    SecretEnvRef,
)
from northstack.interfaces.web.server import create_app

# Config fixtures


def _profile(
    name: str, *, roles: set[Role] | None = None, key: str | None = "MY_KEY"
) -> ModelProfile:
    """A minimal profile.  No Role.WORKER by default -> deterministic path."""
    return ModelProfile(
        name=name,
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://127.0.0.1:65535/v1",  # intentionally dead; never hit
        model="m",
        api_key_env=SecretEnvRef(env_var=key) if key else None,
        roles=roles or set(),
        max_concurrency=1,
    )


def _deterministic_config(name: str = "TestCo") -> NorthStackConfig:
    """A config whose company takes the deterministic (no-provider) path.

    No profile declares Role.WORKER so build_company falls back to
    DeterministicAnalysisRunner; the pipeline abstains without any HTTP.
    """
    reviewer = _profile("reviewer-1", roles={Role.REVIEWER})
    planner = _profile("planner-1", roles={Role.PLANNER})
    return NorthStackConfig(
        name=name,
        profiles=[reviewer, planner],
        commands=[],
        run=RunConfig(),
        routing=[],
    )


def _write_config(tmp_path: Path, config: NorthStackConfig) -> Path:
    """Persist a config to a TOML file create_app can load via from_toml."""
    from northstack.adapters.config_toml import save_config_to_toml

    path = tmp_path / "northstack.toml"
    save_config_to_toml(config, path)
    return path


@pytest.fixture
def app(tmp_path: Path) -> object:
    """A FastAPI app whose ConfigStore backs a tmp TOML + a tmp base root."""
    config = _deterministic_config()
    config_path = _write_config(tmp_path, config)
    application = create_app(config_path)
    # Allow workspace discovery + file browsing rooted at the tmp area.
    application.state.files_base_root = str(tmp_path)
    return application


@pytest.fixture
def client(app: object) -> Iterator[TestClient]:
    with TestClient(app) as c:  # triggers lifespan (active_runs dict etc.)
        yield c


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An empty workspace dir for runs + file browsing."""
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture(autouse=True)
def _env_keys():
    """Resolve the test secret env-var so key_status reports OK."""
    os.environ["MY_KEY"] = "sk-dummy"
    yield
    os.environ.pop("MY_KEY", None)


def _wait_for_terminal(client: TestClient, run_id: str, *, timeout_s: float = 20.0) -> dict:
    """Poll /events until the run reaches a terminal status; return the snapshot.

    The deterministic run completes synchronously-ish; the in-process task is
    driven by the TestClient's event loop so we just need to let it run.
    """
    deadline = time.time() + timeout_s
    last_seq = 0
    while time.time() < deadline:
        r = client.get(f"/api/runs/{run_id}/events", params={"since": last_seq})
        assert r.status_code == 200, r.text
        body = r.json()
        for e in body["events"]:
            last_seq = max(last_seq, e["seq"])
            if e["kind"] == "outcome_emitted":
                return client.get(f"/api/runs/{run_id}").json()
        time.sleep(0.05)
    return client.get(f"/api/runs/{run_id}").json()


def _run_until_terminal(client: TestClient, workspace: Path, goal: str) -> str:
    """Start a run and wait for terminal state.  Returns the run_id."""
    r = client.post("/api/runs", json={"goal": goal, "workspace_root": str(workspace)})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    snap = _wait_for_terminal(client, run_id)
    assert snap["status"] in ("verified", "abstained", "failed"), snap
    return run_id


def _seed_history_run(
    client: TestClient, workspace: Path, run_id: str, status: str, outcome: str
) -> None:
    from northstack.adapters.sqlite_ledger import Ledger
    from northstack.domain import RunOutcome, RunStatus
    from northstack.events.catalog import OutcomeEmitted, RunCreated, StatusChanged

    workspace.mkdir()
    with Ledger(path=workspace / ".northstack" / "ledger.db") as ledger:
        ledger.append_next(run_id, RunCreated())
        statuses = (
            ["contracted", "planned", "executing", "verifying", "verified"]
            if status == "verified"
            else [status]
        )
        for value in statuses:
            ledger.append_next(run_id, StatusChanged(status=RunStatus(value)))
        ledger.append_next(run_id, OutcomeEmitted(outcome=RunOutcome(outcome)))
    client.app.state.run_index.register(run_id, str(workspace))


# Config reads


class TestConfigReads:
    def test_get_config_has_derived_tier_and_key_status(self, client: TestClient) -> None:
        r = client.get("/api/config")
        assert r.status_code == 200
        c = r.json()
        assert c["name"] == "TestCo"
        assert c["unsaved"] is False
        names = {p["name"] for p in c["profiles"]}
        assert {"reviewer-1", "planner-1"} <= names
        for p in c["profiles"]:
            # tier is derived + present (never None) -- it's a property
            assert "tier" in p and p["tier"] in (1, 2, 3, 4)
            # key_status is env:NAME OK|UNSET or 'no key', never a value
            assert p["key_status"].startswith(("env:", "no key"))
            # serialize the whole profile and confirm no secret VALUE leaked
            import json as _json

            assert "sk-dummy" not in _json.dumps(p)

    def test_secrets_status_never_value(self, client: TestClient) -> None:
        r = client.get("/api/secrets/status")
        assert r.status_code == 200
        body = r.json()
        names = {p["name"] for p in body["profiles"]}
        assert {"reviewer-1", "planner-1"} <= names
        for p in body["profiles"]:
            assert p["api_key_env"] == "MY_KEY"
            assert p["key_status"] == "env:MY_KEY OK"


# Config writes


class TestConfigName:
    def test_patch_name(self, client: TestClient) -> None:
        r = client.patch("/api/config/name", json={"name": "Renamed Co"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "Renamed Co"
        # dirty now
        assert body["unsaved"] is True

    def test_patch_name_empty_rejected(self, client: TestClient) -> None:
        r = client.patch("/api/config/name", json={"name": "   "})
        assert r.status_code == 400


class TestProfileCrud:
    def _body(self, name: str = "newp") -> dict:
        return {
            "name": name,
            "protocol": "openai_chat",
            "base_url": "http://localhost",
            "model": "m",
            "api_key_env": "MY_KEY",
            "roles": ["reviewer"],
            "capabilities": ["tool_use"],
            "max_concurrency": 2,
        }

    def test_add_profile(self, client: TestClient) -> None:
        r = client.post("/api/config/profiles", json=self._body())
        assert r.status_code == 200, r.text
        names = {p["name"] for p in r.json()["profiles"]}
        assert "newp" in names

    def test_remote_credentialed_http_requires_explicit_opt_in(self, client: TestClient) -> None:
        body = self._body("remote")
        body["base_url"] = "http://proxy.example.com/v1"

        rejected = client.post("/api/config/profiles", json=body)
        assert rejected.status_code == 400
        assert "must use https" in rejected.text

        body["allow_insecure_http"] = True
        accepted = client.post("/api/config/profiles", json=body)
        assert accepted.status_code == 200, accepted.text
        profile = next(p for p in accepted.json()["profiles"] if p["name"] == "remote")
        assert profile["allow_insecure_http"] is True

    def test_add_duplicate_profile_rejected(self, client: TestClient) -> None:
        assert client.post("/api/config/profiles", json=self._body("dup")).status_code == 200
        r = client.post("/api/config/profiles", json=self._body("dup"))
        assert r.status_code == 400

    def test_update_profile(self, client: TestClient) -> None:
        client.post("/api/config/profiles", json=self._body("p-up"))
        r = client.put(
            "/api/config/profiles/p-up", json={**self._body("p-up"), "max_concurrency": 8}
        )
        assert r.status_code == 200, r.text
        p = next(x for x in r.json()["profiles"] if x["name"] == "p-up")
        assert p["max_concurrency"] == 8

    def test_profile_transport_settings_round_trip_through_api(self, client: TestClient) -> None:
        body = {
            **self._body("transport"),
            "request_timeout_seconds": 17.5,
            "strict_stream_completion": False,
            "transport_retries": 0,
            "transport_retry_backoff_seconds": [0.25],
        }
        response = client.post("/api/config/profiles", json=body)
        assert response.status_code == 200, response.text
        profile = next(p for p in response.json()["profiles"] if p["name"] == "transport")
        assert {
            key: profile[key]
            for key in body
            if key.startswith(("request_", "strict_", "transport_"))
        } == {
            key: value
            for key, value in body.items()
            if key.startswith(("request_", "strict_", "transport_"))
        }

    def test_profile_validation_does_not_echo_rejected_secret(self, client: TestClient) -> None:
        secret = "plaintext-credential-that-must-not-echo"
        response = client.post(
            "/api/config/profiles",
            json={**self._body("unsafe"), "extra_headers": {"X-Client-Secret": secret}},
        )
        assert response.status_code == 400
        assert secret not in response.text

    def test_delete_profile(self, client: TestClient) -> None:
        client.post("/api/config/profiles", json=self._body("gone"))
        r = client.delete("/api/config/profiles/gone")
        assert r.status_code == 200
        assert "gone" not in {p["name"] for p in r.json()["profiles"]}

    def test_delete_routed_profile_rejected(self, client: TestClient) -> None:
        # Route reviewer-1 to the reviewer role, then deletion must 400.
        r = client.put(
            "/api/config/routing",
            json={"routing": [{"role": "reviewer", "profiles": ["reviewer-1"]}]},
        )
        assert r.status_code == 200, r.text
        r = client.delete("/api/config/profiles/reviewer-1")
        assert r.status_code == 400
        assert "routing" in r.json()["detail"].lower()

    def test_duplicate_profile(self, client: TestClient) -> None:
        r = client.post(
            "/api/config/profiles/reviewer-1/duplicate", json={"new_name": "reviewer-copy"}
        )
        assert r.status_code == 200, r.text
        names = {p["name"] for p in r.json()["profiles"]}
        assert "reviewer-copy" in names
        src = next(p for p in r.json()["profiles"] if p["name"] == "reviewer-1")
        clone = next(p for p in r.json()["profiles"] if p["name"] == "reviewer-copy")
        assert clone["model"] == src["model"]
        assert clone["protocol"] == src["protocol"]

    def test_delete_routed_profile_with_explicit_cascade(self, client: TestClient) -> None:
        r = client.put(
            "/api/config/routing",
            json={"routing": [{"role": "reviewer", "profiles": ["reviewer-1"]}]},
        )
        assert r.status_code == 200, r.text

        r = client.delete(
            "/api/config/profiles/reviewer-1",
            params={"remove_from_routing": "true"},
        )

        assert r.status_code == 200, r.text
        assert "reviewer-1" not in {p["name"] for p in r.json()["profiles"]}
        assert r.json()["routing"] == []


class TestCommandCrud:
    def _body(self, name: str = "echo") -> dict:
        return {
            "name": name,
            "argv": ["python", "-c", "print('hi')"],
            "timeout_seconds": 5.0,
            "max_output_bytes": 4096,
            "env_allowlist": ["PATH"],
        }

    def test_add_command(self, client: TestClient) -> None:
        r = client.post("/api/config/commands", json=self._body())
        assert r.status_code == 200, r.text
        assert any(c["name"] == "echo" for c in r.json()["commands"])

    def test_update_and_delete_command(self, client: TestClient) -> None:
        client.post("/api/config/commands", json=self._body("c1"))
        r = client.put("/api/config/commands/c1", json={**self._body("c1"), "timeout_seconds": 1.0})
        assert r.status_code == 200
        assert next(c for c in r.json()["commands"] if c["name"] == "c1")["timeout_seconds"] == 1.0
        r = client.delete("/api/config/commands/c1")
        assert r.status_code == 200
        assert "c1" not in {c["name"] for c in r.json()["commands"]}

    def test_command_test_runs_and_is_not_isolated(self, client: TestClient) -> None:
        client.post("/api/config/commands", json=self._body("ok"))
        r = client.post("/api/config/commands/ok/test")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "ok"
        assert body["exit_code"] == 0
        assert "hi" in body["stdout"]
        # Documents that this is NOT a sandbox:
        assert body["isolated"] is False
        assert "truncated" in body

    def test_command_test_does_not_inherit_sensitive_parent_env(
        self, client: TestClient, monkeypatch
    ) -> None:
        monkeypatch.setenv("NORTHSTACK_PROVIDER_TOKEN", "super-secret")
        body = {
            "name": "env-check",
            "argv": [
                "python",
                "-c",
                "import os; print(os.getenv('NORTHSTACK_PROVIDER_TOKEN', 'missing'))",
            ],
            "timeout_seconds": 5.0,
            "max_output_bytes": 4096,
            "env_allowlist": ["PATH"],
        }
        assert client.post("/api/config/commands", json=body).status_code == 200

        result = client.post("/api/config/commands/env-check/test").json()

        assert result["exit_code"] == 0
        assert result["stdout"].strip() == "missing"
        assert "super-secret" not in str(result)

    def test_command_test_unknown_404(self, client: TestClient) -> None:
        r = client.post("/api/config/commands/no-such/test")
        assert r.status_code == 404

    def test_command_test_unknown_protocol_rejected_on_profile(self, client: TestClient) -> None:
        # Invalid protocol -> Protocol() raises -> 400.
        r = client.post(
            "/api/config/profiles",
            json={
                "name": "bad-proto",
                "protocol": "not_a_protocol",
                "base_url": "x",
                "model": "m",
                "roles": ["reviewer"],
            },
        )
        assert r.status_code == 400


class TestPersistence:
    def test_save_round_trips_and_clears_unsaved(self, client: TestClient, tmp_path: Path) -> None:
        client.patch("/api/config/name", json={"name": "Saved Co"})
        assert client.get("/api/config").json()["unsaved"] is True
        r = client.post("/api/config/save")
        assert r.status_code == 200, r.text
        # saved is the toml path string
        assert r.json()["saved"].endswith("northstack.toml")
        # After save, unsaved flag clears.
        assert client.get("/api/config").json()["unsaved"] is False
        # The on-disk file reflects the new name.
        on_disk = (tmp_path / "northstack.toml").read_text(encoding="utf-8")
        assert "Saved Co" in on_disk

    def test_validate_is_no_op_on_valid_config(self, client: TestClient) -> None:
        r = client.post("/api/config/validate")
        assert r.status_code == 200
        assert r.json()["valid"] is True

    def test_reload_drops_in_memory_edits(self, client: TestClient) -> None:
        original_name = client.get("/api/config").json()["name"]
        client.patch("/api/config/name", json={"name": "Transient"})
        assert client.get("/api/config").json()["name"] == "Transient"
        r = client.post("/api/config/reload")
        assert r.status_code == 200
        assert r.json()["name"] == original_name
        assert r.json()["unsaved"] is False

    def test_reset_clears_to_name_only(self, client: TestClient) -> None:
        r = client.post("/api/config/reset")
        assert r.status_code == 200
        body = r.json()
        assert body["profiles"] == []
        assert body["commands"] == []
        assert body["routing"] == []
        assert body["unsaved"] is True


class TestConfigRun:
    def test_put_run_round_trips_all_fields(self, client: TestClient) -> None:
        body = {
            "default_budget_tokens": 0,
            "default_budget_cost_usd": 0.0,
            "stall_window_seconds": 30.0,
            "planner_mode": "model",
            "falsifier_mode": "model",
            "calibration_path": "cal.jsonl",
        }
        r = client.put("/api/config/run", json=body)
        assert r.status_code == 200, r.text
        run = r.json()["run"]
        assert run == {
            "default_budget_tokens": 0,
            "default_budget_cost_usd": 0.0,
            "stall_window_seconds": 30.0,
            "planner_mode": "model",
            "falsifier_mode": "model",
            "calibration_path": "cal.jsonl",
        }
        assert r.json()["unsaved"] is True
        # GET reflects the same run block
        assert client.get("/api/config").json()["run"] == run

    def test_put_run_bad_mode_rejected(self, client: TestClient) -> None:
        # Literal-typed body fields reject bogus modes at FastAPI's request
        # validation layer (422) before the config store is ever touched.
        r = client.put("/api/config/run", json={"planner_mode": "bogus"})
        assert r.status_code == 422
        assert client.get("/api/config").json()["run"]["planner_mode"] == "single"

    def test_put_run_negative_budget_rejected(self, client: TestClient) -> None:
        r = client.put("/api/config/run", json={"default_budget_tokens": -1})
        assert r.status_code == 400


class TestConfigToml:
    def test_get_toml_documents_current_config(self, client: TestClient) -> None:
        r = client.get("/api/config/toml")
        assert r.status_code == 200, r.text
        text = r.json()["text"]
        assert text.startswith("[northstack]")
        assert 'name = "TestCo"' in text

    def test_put_toml_applies_full_document(self, client: TestClient) -> None:
        text = (
            "[northstack]\n"
            'name = "TomlCo"\n'
            "[northstack.run]\n"
            "default_budget_tokens = 1\n"
            "stall_window_seconds = 5.0\n"
        )
        r = client.put("/api/config/toml", json={"text": text})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["name"] == "TomlCo"
        assert body["run"]["default_budget_tokens"] == 1
        assert body["run"]["stall_window_seconds"] == 5.0
        assert body["unsaved"] is True

    def test_put_toml_invalid_rejected_store_unchanged(self, client: TestClient) -> None:
        before = client.get("/api/config").json()["name"]
        for bad in ("not toml = = =", "[other]\nx = 1"):
            r = client.put("/api/config/toml", json={"text": bad})
            assert r.status_code == 400
        assert client.get("/api/config").json()["name"] == before

    def test_put_toml_preserves_unknown_sections_through_save(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        text = '[northstack]\nname = "TestCo"\n[northstack.workspace]\nmax_list_entries = 42\n'
        assert client.put("/api/config/toml", json={"text": text}).status_code == 200
        saved = client.post("/api/config/save")
        assert saved.status_code == 200, saved.text
        on_disk = (tmp_path / "northstack.toml").read_text(encoding="utf-8")
        assert "max_list_entries = 42" in on_disk
        # Reload from disk and the unknown section is still in the document.
        assert client.post("/api/config/reload").status_code == 200
        assert "northstack.workspace]" in client.get("/api/config/toml").json()["text"]


class TestRoutingPresets:
    def test_list_presets_reports_availability(self, client: TestClient) -> None:
        r = client.get("/api/config/routing/presets")
        assert r.status_code == 200
        presets = {p["id"]: p for p in r.json()["presets"]}
        assert {"single_expert", "cheap_fanout", "strong_integrator"} <= presets.keys()
        for preset in presets.values():
            assert {"label", "description", "routing", "available", "reason"} <= preset.keys()
        assert presets["single_expert"]["available"] is False
        assert "all five roles" in presets["single_expert"]["reason"]

    def test_apply_unavailable_preset_has_concise_error(self, client: TestClient) -> None:
        r = client.post("/api/config/routing/presets/single_expert/apply")
        assert r.status_code == 400
        assert r.json()["detail"] == "No profile declares all five roles."
        assert "NorthStackConfig" not in r.text

    def test_apply_preset_with_matching_profiles_succeeds(self, client: TestClient) -> None:
        # Define a single profile named "expert" carrying every role, then the
        # single_expert preset applies.
        r = client.post(
            "/api/config/profiles",
            json={
                "name": "expert",
                "protocol": "openai_chat",
                "base_url": "http://localhost",
                "model": "m",
                "api_key_env": "MY_KEY",
                "roles": [
                    "worker",
                    "reviewer",
                    "planner",
                    "specialist",
                    "orchestrator",
                ],
                "max_concurrency": 1,
            },
        )
        assert r.status_code == 200, r.text
        r = client.post("/api/config/routing/presets/single_expert/apply")
        assert r.status_code == 200, r.text
        routing = {e["role"]: e["profiles"] for e in r.json()["routing"]}
        for role in ("worker", "reviewer", "planner", "specialist", "orchestrator"):
            assert routing[role] == ["expert"]

    def test_apply_unknown_preset_404(self, client: TestClient) -> None:
        r = client.post("/api/config/routing/presets/no-such/apply")
        assert r.status_code == 404


# Runs (no live provider)


class TestRuns:
    def test_start_run_returns_run_id(self, client: TestClient, workspace: Path) -> None:
        r = client.post(
            "/api/runs",
            json={"goal": "make a file hello.txt", "workspace_root": str(workspace)},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["run_id"].startswith("run-")
        assert body["workspace_root"] == str(workspace.resolve())

    def test_start_run_bad_workspace_400(self, client: TestClient, tmp_path: Path) -> None:
        r = client.post(
            "/api/runs",
            json={"goal": "x", "workspace_root": str(tmp_path / "does-not-exist")},
        )
        assert r.status_code == 400

    @pytest.mark.parametrize(
        "overrides",
        [
            {"goal": ""},
            {"goal": "   "},
            {"goal": "x" * 65_537},
            {"max_waves": 0},
            {"max_waves": 101},
        ],
    )
    def test_start_run_rejects_invalid_request_bounds(
        self, client: TestClient, workspace: Path, overrides: dict[str, object]
    ) -> None:
        body = {"goal": "g", "workspace_root": str(workspace), **overrides}
        response = client.post("/api/runs", json=body)
        assert response.status_code == 422
        assert client.app.state.supervisors == {}

    def test_startup_failure_rolls_back_supervisor_and_ledger(
        self, client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from northstack.adapters.sqlite_ledger import Ledger
        from northstack.application.run_index import RunIndex

        closed: list[Ledger] = []
        original_close = Ledger.close

        def close(ledger: Ledger) -> None:
            closed.append(ledger)
            original_close(ledger)

        def reject_registration(index: RunIndex, run_id: str, root: str) -> None:
            raise RuntimeError("injected registration failure")

        monkeypatch.setattr(Ledger, "close", close)
        monkeypatch.setattr(RunIndex, "register", reject_registration)
        with pytest.raises(RuntimeError, match="injected registration failure"):
            client.post("/api/runs", json={"goal": "g", "workspace_root": str(workspace)})
        assert client.app.state.supervisors == {}
        assert len(closed) == 1

    def test_a_run_that_raises_records_its_cause(
        self,
        client: TestClient,
        workspace: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A run that dies before writing an outcome must not die silently."""
        from northstack.application.orchestrator import Company

        async def explode(self, request, *, run_id=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("injected run failure")

        monkeypatch.setattr(Company, "run_async", explode)
        with caplog.at_level(logging.ERROR, logger="northstack.interfaces.web.routes_runs"):
            r = client.post("/api/runs", json={"goal": "g", "workspace_root": str(workspace)})
            run_id = r.json()["run_id"]
            deadline = time.time() + 10.0
            while time.time() < deadline and "injected run failure" not in caplog.text:
                client.get(f"/api/runs/{run_id}/events", params={"since": 0})
                time.sleep(0.05)
        assert "injected run failure" in caplog.text
        assert f"run_id={run_id}" in caplog.text

    def test_duplicate_registration_does_not_forget_existing_run(
        self, client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from northstack.application.run_index import DuplicateRunIdError
        from northstack.interfaces.web import routes_runs

        run_id, prior = "run-0123456789ab", workspace / "prior"
        client.app.state.run_index.register(run_id, str(prior))
        monkeypatch.setattr(
            routes_runs.uuid, "uuid4", lambda: type("Id", (), {"hex": run_id[4:]})()
        )
        with pytest.raises(DuplicateRunIdError):
            client.post("/api/runs", json={"goal": "g", "workspace_root": str(workspace)})
        assert client.app.state.run_index.workspace_of(run_id) == str(prior.resolve())

    def test_ambiguous_historical_run_returns_conflict(
        self, client: TestClient, workspace: Path
    ) -> None:
        from northstack.adapters.sqlite_ledger import Ledger
        from northstack.application.run_index import RunIndex
        from northstack.events.catalog import RunCreated

        workspaces = [workspace / name for name in ("first", "second")]
        for root in workspaces:
            with Ledger(root / ".northstack" / "ledger.db") as ledger:
                ledger.append_next("run-duplicate", RunCreated())
        index = RunIndex()
        index.load_historical([str(root) for root in workspaces])
        client.app.state.run_index = index
        response = client.get("/api/runs/run-duplicate")
        assert response.status_code == 409
        assert response.json()["detail"]["category"] == "ambiguous_run_id"

    def test_start_run_workspace_outside_files_base_root_is_rejected(self, tmp_path: Path) -> None:
        # POST /api/runs must confine workspace_root to files_base_root, the
        # same invariant _open_workspace enforces for the file browser. A real
        # directory that exists but sits OUTSIDE the allowlist must be rejected
        # (403), not accepted -- otherwise an authenticated caller can start a
        # run against any host directory. apps.state.run_workspaces is the only
        # registration, so this is a pure containment check, no auth concern.
        config_path = _write_config(tmp_path, _deterministic_config())
        application = create_app(config_path, files_base_root=tmp_path)
        outside = tmp_path.parent / "northstack-runs-outside-allowlist"
        outside.mkdir(exist_ok=True)
        try:
            with TestClient(application) as c:
                r = c.post(
                    "/api/runs",
                    json={"goal": "x", "workspace_root": str(outside)},
                )
                assert r.status_code == 403, (r.status_code, r.text)
                assert "outside" in r.json().get("detail", "").lower()
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_unknown_run_404(self, client: TestClient) -> None:
        assert client.get("/api/runs/run-nope").status_code == 404
        assert client.get("/api/runs/run-nope/events").status_code == 404

    def test_single_axis_budget_keeps_other_axis_unlimited(
        self, client: TestClient, workspace: Path
    ) -> None:
        r = client.post(
            "/api/runs",
            json={
                "goal": "create hello.txt",
                "workspace_root": str(workspace),
                "budget_tokens": 1_000_000,
            },
        )
        run_id = r.json()["run_id"]
        snap = _wait_for_terminal(client, run_id)

        assert snap["budget"] == {
            "token_limit": 1_000_000,
            "cost_limit_usd": None,
        }

    def test_run_completes_and_events_poll(self, client: TestClient, workspace: Path) -> None:
        r = client.post(
            "/api/runs",
            json={"goal": "create hello.txt", "workspace_root": str(workspace)},
        )
        run_id = r.json()["run_id"]
        snap = _wait_for_terminal(client, run_id)
        # The deterministic path abstains (no worker model to verify against).
        assert snap["status"] in ("verified", "abstained", "failed")
        assert snap["outcome"] in ("verified", "abstained", "failed", None)
        ev = client.get(f"/api/runs/{run_id}/events", params={"since": 0}).json()
        assert len(ev["events"]) >= 1
        assert ev["next_seq"] >= 1

    def test_integrity_ok_after_run(self, client: TestClient, workspace: Path) -> None:
        r = client.post(
            "/api/runs",
            json={"goal": "g", "workspace_root": str(workspace)},
        )
        run_id = r.json()["run_id"]
        _wait_for_terminal(client, run_id)
        ir = client.get(f"/api/runs/{run_id}/integrity").json()
        assert ir["ok"] is True
        assert ir["events_checked"] >= 1

    def test_corrupt_ledger_returns_typed_conflict(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        workspace, run_id = tmp_path / "corrupt", "run-corrupt"
        _seed_history_run(client, workspace, run_id, "verified", "verified")
        with sqlite3.connect(workspace / ".northstack" / "ledger.db") as conn:
            conn.execute(
                "UPDATE events SET payload = ? WHERE run_id = ? AND seq = 1",
                ("{", run_id),
            )
        for suffix in ("", "/events", "/ledger.json"):
            response = client.get(f"/api/runs/{run_id}{suffix}")
            assert response.status_code == 409
            assert response.json()["category"] == "ledger_corruption"
        integrity = client.get(f"/api/runs/{run_id}/integrity")
        assert integrity.status_code == 200
        assert integrity.json()["error_category"] == "payload_json"

    def test_ledger_json_matches_events(self, client: TestClient, workspace: Path) -> None:
        r = client.post(
            "/api/runs",
            json={"goal": "g", "workspace_root": str(workspace)},
        )
        run_id = r.json()["run_id"]
        _wait_for_terminal(client, run_id)
        ledger = client.get(f"/api/runs/{run_id}/ledger.json").json()
        evs = client.get(f"/api/runs/{run_id}/events", params={"since": 0, "limit": 5000}).json()
        led_seqs = {e["seq"] for e in ledger["events"]}
        ev_seqs = {e["seq"] for e in evs["events"]}
        assert led_seqs == ev_seqs
        assert all("hash_chain" in e and "prev_hash" in e for e in ledger["events"])

    def test_ledger_json_export_is_paginated(self, client: TestClient, tmp_path: Path) -> None:
        workspace, run_id = tmp_path / "export", "run-export"
        _seed_history_run(client, workspace, run_id, "verified", "verified")
        first = client.get(f"/api/runs/{run_id}/ledger.json", params={"limit": 2}).json()
        assert len(first["events"]) == 2
        assert first["truncated"] is True
        second = client.get(
            f"/api/runs/{run_id}/ledger.json",
            params={"since": first["next_seq"], "limit": 2},
        ).json()
        assert second["events"]
        assert second["events"][0]["seq"] > first["next_seq"]

    def test_stop_run_404_when_not_active(self, client: TestClient) -> None:
        r = client.post("/api/runs/run-nope/stop")
        assert r.status_code == 404

    def test_active_runs_list_shape(self, client: TestClient) -> None:
        r = client.get("/api/runs/active")
        assert r.status_code == 200
        assert "active" in r.json()
        assert isinstance(r.json()["active"], list)

    def test_run_reads_never_borrow_supervisor_ledger(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from northstack.adapters.sqlite_ledger import Ledger
        from northstack.application.run_supervisor import RunSupervisor

        workspace, run_id = tmp_path / "history", "run-fresh-reader"
        _seed_history_run(client, workspace, run_id, "verified", "verified")
        live = Ledger(workspace / ".northstack" / "ledger.db")
        client.app.state.supervisors[run_id] = RunSupervisor(
            run_id=run_id,
            ledger=live,
            workspace=str(workspace),
            task=None,
        )

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("request borrowed the supervisor ledger")

        for method in ("events", "events_since", "verify_integrity"):
            monkeypatch.setattr(live, method, forbidden)
        for suffix in ("", "/events", "/integrity", "/ledger.json"):
            response = client.get(f"/api/runs/{run_id}{suffix}")
            assert response.status_code == 200

    def test_history_lists_completed_runs(self, client: TestClient, workspace: Path) -> None:
        # Two runs at the SAME workspace share one ledger db; run_workspaces
        # retains the entry after the run finishes so the list endpoint can
        # find it.
        ids = []
        for goal in ("a", "b"):
            ids.append(_run_until_terminal(client, workspace, goal))
        runs = client.get("/api/runs", params={"limit": 50}).json()["runs"]
        found = {r["run_id"]: r for r in runs}
        for rid in ids:
            assert rid in found
            assert found[rid]["event_count"] >= 1
            assert found[rid]["status"] in ("verified", "abstained", "failed")

    def test_history_filters(self, client: TestClient, workspace: Path) -> None:
        _run_until_terminal(client, workspace, "f")
        # status filter for a value that no run has -> empty list
        runs = client.get("/api/runs", params={"status": "intake"}).json()["runs"]
        assert runs == []

    def test_run_negative_budget_returns_422_not_500(
        self, client: TestClient, workspace: Path
    ) -> None:
        # Bug 3: budget_tokens / budget_cost_usd had no ge=0 on RunStartBody, so
        # a negative value passed request validation and only blew up inside
        # _budget()'s Budget(...) construction -> unhandled ValidationError ->
        # HTTP 500.  Must now be a 422 (validation) at the request boundary,
        # never a 500.  Cover both axes + an off-zero negative cost.
        for body in (
            {"goal": "g", "workspace_root": str(workspace), "budget_tokens": -1},
            {"goal": "g", "workspace_root": str(workspace), "budget_cost_usd": -0.5},
            {
                "goal": "g",
                "workspace_root": str(workspace),
                "budget_tokens": 0,
                "budget_cost_usd": -1.0,
            },
        ):
            r = client.post("/api/runs", json=body)
            assert r.status_code == 422, (body, r.status_code, r.text)

    def test_compare_two_runs(self, client: TestClient, workspace: Path) -> None:
        a = _run_until_terminal(client, workspace, "goal-a")
        b = _run_until_terminal(client, workspace, "goal-b")
        r = client.get("/api/runs/compare", params={"a": a, "b": b})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["a"] == a and body["b"] == b
        assert body["goal"] == {"a": "goal-a", "b": "goal-b"}
        assert "delta" in body
        assert "total_calls" in body["delta"]
        assert "usage" in body
        assert "status" in body
        # Regression for a copy-paste bug: total_output_tokens delta must be
        # computed from OUTPUT tokens of both runs (it once subtracted a's
        # INPUT tokens -> self-compare gave a spurious nonzero output delta).
        ua, ub = body["usage"]["a"], body["usage"]["b"]
        for k in ("total_calls", "total_cost_usd", "total_input_tokens", "total_output_tokens"):
            assert body["delta"][k] == ub[k] - ua[k], (k, body["delta"][k], ub[k], ua[k])
        # Usage values are numeric (the UI does arithmetic / .toLocaleString).
        for side in ("a", "b"):
            for k in ("total_calls", "total_cost_usd", "total_input_tokens", "total_output_tokens"):
                assert isinstance(body["usage"][side][k], (int, float))

    def test_compare_self_is_zero_delta(self, client: TestClient, workspace: Path) -> None:
        a = _run_until_terminal(client, workspace, "solo")
        r = client.get("/api/runs/compare", params={"a": a, "b": a})
        assert r.status_code == 200, r.text
        # Comparing a run to itself must yield all-zero deltas -- the hard
        # version of the output-token regression above.
        for k, v in r.json()["delta"].items():
            assert v == 0 or v == 0.0, (k, v)

    def test_compare_unknown_run_404(self, client: TestClient) -> None:
        r = client.get("/api/runs/compare", params={"a": "run-x", "b": "run-y"})
        # No ledger db exists -> 404 'no ledger found'
        assert r.status_code == 404

    def test_compare_does_not_relabel_internal_failure_as_unknown(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from northstack.interfaces.web import routes_runs

        def fail(*_args):
            raise RuntimeError("internal failure")

        monkeypatch.setattr(routes_runs, "_snapshot_for_run", fail)
        unchecked = TestClient(client.app, raise_server_exceptions=False)
        try:
            response = unchecked.get("/api/runs/compare", params={"a": "run-a", "b": "run-b"})
        finally:
            unchecked.close()
        assert response.status_code == 500
        body = response.json()
        assert body["category"] == "internal_error"
        assert body["correlation_id"] == response.headers["X-Correlation-ID"]
        assert "internal failure" not in response.text

    def test_compare_one_unknown_run_404_no_ghost(
        self, client: TestClient, workspace: Path
    ) -> None:
        # Regression for a live-found bug: replay_run(Ledger, ) returns an EMPTY
        # RunState (status=intake, zero events) for an unknown run_id instead of
        # raising, so compare of a REAL run vs a NONEXISTENT id used to return
        # 200 with a fabricated "ghost" snapshot (blank goal, zeroed counters)
        # on the unknown side.  The handler now 404s when a run has no events.
        real = _run_until_terminal(client, workspace, "ghost-guard")
        # unknown as b:
        rb = client.get("/api/runs/compare", params={"a": real, "b": "run-ghost"})
        assert rb.status_code == 404, rb.text
        # unknown as a (symmetric):
        ra = client.get("/api/runs/compare", params={"a": "run-ghost", "b": real})
        assert ra.status_code == 404, ra.text

    def test_export_csv(self, client: TestClient, workspace: Path) -> None:
        _run_until_terminal(client, workspace, "csv")
        r = client.get("/api/runs/export", params={"format": "csv"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        header = r.text.splitlines()[0]
        assert "run_id,status,outcome" in header

    def test_export_bad_format_400(self, client: TestClient) -> None:
        r = client.get("/api/runs/export", params={"format": "json"})
        assert r.status_code == 400

    def test_export_is_bounded_and_reports_continuation(
        self, client: TestClient, workspace: Path
    ) -> None:
        for goal in ("one", "two"):
            _run_until_terminal(client, workspace, goal)
        response = client.get("/api/runs/export", params={"limit": 1})
        assert response.status_code == 200
        assert len(response.text.splitlines()) == 2
        assert response.headers["x-northstack-truncated"] == "true"
        assert response.headers["x-northstack-next-offset"] == "1"
        assert client.get("/api/runs/export", params={"limit": 10_001}).status_code == 422

    def test_history_compare_and_export_span_workspaces(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _seed_history_run(client, tmp_path / "history-a", "run-a", "failed", "failed")
        _seed_history_run(client, tmp_path / "history-b", "run-b", "verified", "verified")
        listed = client.get("/api/runs", params={"limit": 10}).json()["runs"]
        assert {run["run_id"] for run in listed} >= {"run-a", "run-b"}
        compared = client.get("/api/runs/compare", params={"a": "run-a", "b": "run-b"})
        assert compared.status_code == 200, compared.text
        assert compared.json()["status"] == {"a": "failed", "b": "verified"}
        exported = client.get("/api/runs/export").text
        assert "run-a" in exported and "run-b" in exported

    def test_history_filters_before_global_pagination(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        _seed_history_run(client, tmp_path / "older", "run-failed", "failed", "failed")
        time.sleep(0.01)
        _seed_history_run(client, tmp_path / "newer", "run-verified", "verified", "verified")
        body = client.get("/api/runs", params={"limit": 1, "status": "failed"}).json()
        assert [run["run_id"] for run in body["runs"]] == ["run-failed"]
        assert body["total"] == 1
        assert body["truncated"] is False

    def test_history_filter_count_offset_and_empty_contract(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        for index, (run_id, status, outcome) in enumerate(
            [
                ("run-failed", "failed", "failed"),
                ("run-abstained", "failed", "abstained"),
                ("run-verified", "verified", "verified"),
            ]
        ):
            _seed_history_run(client, tmp_path / f"filter-{index}", run_id, status, outcome)
            time.sleep(0.01)
        cases = [
            ({"status": "failed"}, 2, {"run-failed", "run-abstained"}),
            ({"outcome": "verified"}, 1, {"run-verified"}),
            (
                {"status": "failed", "outcome": "failed"},
                1,
                {"run-failed"},
            ),
            ({"status": "missing"}, 0, set()),
        ]
        for params, total, expected in cases:
            body = client.get("/api/runs", params=params).json()
            assert body["total"] == total
            assert {run["run_id"] for run in body["runs"]} == expected
        page = client.get("/api/runs", params={"status": "failed", "limit": 1, "offset": 1}).json()
        assert page["total"] == 2
        assert len(page["runs"]) == 1
        assert page["next_offset"] is None


# Files (workspace-relative only; traversal rejected)


class TestApiAuthentication:
    def test_token_protects_all_api_routes_but_not_static(self, tmp_path: Path) -> None:
        config_path = _write_config(tmp_path, _deterministic_config())
        application = create_app(config_path, api_token="test-token", files_base_root=tmp_path)
        with TestClient(application) as protected:
            assert protected.get("/").status_code == 200
            assert protected.get("/api/config").status_code == 401
            assert protected.get("/api/docs").status_code == 401
            assert (
                protected.get("/api/config", headers={"Authorization": "Bearer wrong"}).status_code
                == 401
            )
            assert (
                protected.get(
                    "/api/config", headers={"Authorization": "Bearer test-token"}
                ).status_code
                == 200
            )

    def test_every_api_route_requires_token(self, tmp_path: Path) -> None:
        # Enumerate every /api/ route from the OpenAPI schema and assert each
        # returns 401 without a bearer token -- guards against a future route
        # being mounted outside the auth middleware's /api/ prefix match.
        config_path = _write_config(tmp_path, _deterministic_config())
        application = create_app(config_path, api_token="test-token", files_base_root=tmp_path)
        with TestClient(application) as protected:
            paths = [
                p for p in application.openapi()["paths"] if p.startswith("/api/") and "{" not in p
            ]
            assert paths, "expected at least one concrete /api/ route"
            for path in paths:
                # Both GET-only and POST routes: unauthenticated must 401.
                assert protected.get(path).status_code == 401, f"GET {path} was not token-protected"


class TestFiles:
    def test_workspaces_discovery(self, client: TestClient, workspace: Path) -> None:
        # Create a ledger.db in the workspace so discovery lists it.
        (workspace / ".northstack").mkdir()
        (workspace / ".northstack" / "ledger.db").write_bytes(b"")
        r = client.get("/api/files/workspaces", params={"base": str(workspace.parent)})
        assert r.status_code == 200
        names = {w["name"] for w in r.json()["workspaces"]}
        assert workspace.name in names

    def test_workspaces_discovery_is_paginated(self, client: TestClient, workspace: Path) -> None:
        base = workspace / "scan"
        for name in ("a", "b", "c"):
            ledger = base / name / ".northstack"
            ledger.mkdir(parents=True)
            (ledger / "ledger.db").write_bytes(b"")
        first = client.get(
            "/api/files/workspaces",
            params={"base": str(base), "limit": 2},
        ).json()
        second = client.get(
            "/api/files/workspaces",
            params={"base": str(base), "limit": 2, "offset": 2},
        ).json()
        assert [item["name"] for item in first["workspaces"]] == ["a", "b"]
        assert first["truncated"] is True
        assert first["next_offset"] == 2
        assert [item["name"] for item in second["workspaces"]] == ["c"]
        assert second["truncated"] is False

    def test_workspaces_unreadable_base_is_typed(
        self, client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def denied(_path):
            raise PermissionError("denied")

        monkeypatch.setattr("northstack.interfaces.web.routes_files.os.scandir", denied)
        response = client.get("/api/files/workspaces", params={"base": str(workspace.parent)})
        assert response.status_code == 403
        assert response.json()["detail"]["category"] == "workspace_base_unreadable"

    def test_workspaces_empty_base_is_not_an_error(
        self, client: TestClient, workspace: Path
    ) -> None:
        base = workspace / "empty"
        base.mkdir()
        response = client.get("/api/files/workspaces", params={"base": str(base)})
        assert response.status_code == 200
        assert response.json()["workspaces"] == []

    def test_workspaces_scan_has_a_hard_cap(
        self, client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = workspace / "many"
        for name in ("a", "b", "c"):
            (base / name).mkdir(parents=True)
        monkeypatch.setattr("northstack.interfaces.web.routes_files._MAX_WORKSPACE_SCAN", 2)
        body = client.get("/api/files/workspaces", params={"base": str(base)}).json()
        assert body["scanned"] == 2
        assert body["truncated"] is True

    def test_workspaces_base_outside_allowlist_is_rejected(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        # The module docstring promises "Traversal outside the base is
        # rejected" for the operator-picked base.  ``files_base_root`` (set
        # on the app fixture to ``tmp_path``) is the allowlist root; a
        # caller who passes ``?base=<dir outside that root>`` must NOT be
        # able to enumerate arbitrary host directories for ledger.db files.
        # Build a sibling dir OUTSIDE the allowlist root with a decoy
        # ledger.db and confirm the route refuses to scan it.
        outside = tmp_path.parent / "northstack-outside-allowlist"
        outside.mkdir(exist_ok=True)
        (outside / ".northstack").mkdir(exist_ok=True)
        (outside / ".northstack" / "ledger.db").write_bytes(b"")
        try:
            r = client.get("/api/files/workspaces", params={"base": str(outside)})
            # The module docstring promises "Traversal outside the base is
            # rejected"; a base resolving outside files_base_root must be 400,
            # not 200 with the decoy workspace enumerated.
            assert r.status_code == 400, (
                "list_workspaces accepted a base outside files_base_root; "
                "the docstring promises 'Traversal outside the base is "
                f"rejected'.  Got {r.status_code} with workspaces="
                f"{r.json().get('workspaces')!r}"
            )
            # And the rejection detail must point at the confinement failure,
            # not a different error (e.g. "not a directory").
            assert "outside the allowlist root" in r.json().get("detail", "")
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_tree_lists_files(self, client: TestClient, workspace: Path) -> None:
        (workspace / "hello.txt").write_text("hi", encoding="utf-8")
        (workspace / "sub").mkdir()
        r = client.get("/api/files/tree", params={"workspace": str(workspace), "path": "."})
        assert r.status_code == 200, r.text
        entries = r.json()["entries"]
        names = {e["name"] for e in entries}
        assert "hello.txt" in names
        types = {e["name"]: e["type"] for e in entries}
        assert types["sub"] == "dir"

    def test_tree_empty_path_treats_as_root(self, client: TestClient, workspace: Path) -> None:
        # Regression: an omitted/empty ``path`` must behave like the workspace
        # root (``.``), not 400 "Invalid or unsafe path".  The UI always sends
        # "." but a malformed URL or a client that omits path should still get
        # the root listing, not a confusing traversal error.
        (workspace / "hello.txt").write_text("hi", encoding="utf-8")
        r = client.get("/api/files/tree", params={"workspace": str(workspace), "path": ""})
        assert r.status_code == 200, r.text
        names = {e["name"] for e in r.json()["entries"]}
        assert "hello.txt" in names

    def test_read_file(self, client: TestClient, workspace: Path) -> None:
        (workspace / "hello.txt").write_text("NorthStack ran OK", encoding="utf-8")
        r = client.get(
            "/api/files/read",
            params={"workspace": str(workspace), "path": "hello.txt"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["content"] == "NorthStack ran OK"
        assert body["truncated"] is False
        assert body["total_bytes"] == len("NorthStack ran OK".encode())

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            ".env.local",
            "private.key",
            "cert.pem",
            ".git/config",
            ".northstack/ledger.db",
            ".ssh/id_rsa",
            ".ssh/id_ed25519",
            "id_dsa",
            "id_ecdsa",
        ],
    )
    def test_sensitive_file_reads_are_denied(
        self, client: TestClient, workspace: Path, path: str
    ) -> None:
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("secret", encoding="utf-8")

        response = client.get("/api/files/read", params={"workspace": str(workspace), "path": path})

        assert response.status_code == 403

    def test_workspace_root_outside_allowlist_is_rejected(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / "outside-workspace-root"
        outside.mkdir(exist_ok=True)
        try:
            response = client.get(
                "/api/files/tree",
                params={"workspace": str(outside), "path": "."},
            )
            assert response.status_code == 403
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_traversal_rejected(self, client: TestClient, workspace: Path) -> None:
        r = client.get(
            "/api/files/read",
            params={"workspace": str(workspace), "path": "../../etc/passwd"},
        )
        assert r.status_code == 400

    def test_artifacts_unknown_run_with_workspace_404(
        self, client: TestClient, workspace: Path, tmp_path: Path
    ) -> None:
        # Regression for a live-found bug: /files/artifacts stores blobs in a
        # per-WORKSPACE dir (not per-run).  A caller who passes a NONEXISTENT
        # run_id plus some real workspace that DOES have artifacts used to get
        # THAT workspace's artifacts back under the wrong run_id (200).  Now the
        # handler validates the run exists in the workspace's ledger, else 404.
        # Seed a real run so the workspace has a non-empty ledger, then make an
        # artifacts dir with a stray blob (as if a prior run produced it).
        _run_until_terminal(client, workspace, "seed")
        art_dir = workspace / ".northstack" / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "stray.bin").write_bytes(b"x")
        r = client.get(
            "/api/files/artifacts",
            params={"run_id": "run-ghost", "workspace": str(workspace)},
        )
        assert r.status_code == 404, r.text

    def test_artifacts_known_run_lists_files(self, client: TestClient, workspace: Path) -> None:
        rid = _run_until_terminal(client, workspace, "artifacts goal")
        # No artifacts dir yet -> empty list (not 404, not a stray dir).
        r = client.get("/api/files/artifacts", params={"run_id": rid, "workspace": str(workspace)})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["run_id"] == rid
        assert "artifacts" in body
        assert isinstance(body["artifacts"], list)

    def test_artifacts_rejects_run_from_different_workspace(
        self, client: TestClient, workspace: Path
    ) -> None:
        run_id = _run_until_terminal(client, workspace, "workspace ownership")
        other = workspace.parent / "other-workspace"
        art_dir = other / ".northstack" / "artifacts"
        art_dir.mkdir(parents=True)
        (art_dir / "foreign.bin").write_bytes(b"secret")
        response = client.get(
            "/api/files/artifacts",
            params={"run_id": run_id, "workspace": str(other)},
        )
        assert response.status_code == 404

    def test_artifacts_are_paginated_and_temporary_files_are_hidden(
        self, client: TestClient, workspace: Path
    ) -> None:
        run_id = _run_until_terminal(client, workspace, "artifact pagination")
        art_dir = workspace / ".northstack" / "artifacts"
        for relative in ["00/a", "00/b", "01/c"]:
            target = art_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(relative.encode())
        (art_dir / "00" / ".a.partial.tmp").write_bytes(b"temporary")
        first = client.get(
            "/api/files/artifacts",
            params={"run_id": run_id, "workspace": str(workspace), "limit": 2},
        ).json()
        assert [item["path"].replace("\\", "/") for item in first["artifacts"]] == [
            "00/a",
            "00/b",
        ]
        assert first["truncated"] is True and first["next_offset"] == 2
        assert first["next_cursor"].replace("\\", "/") == "00/b"
        (art_dir / "00" / "aa").write_bytes(b"inserted before cursor")
        second = client.get(
            "/api/files/artifacts",
            params={
                "run_id": run_id,
                "workspace": str(workspace),
                "limit": 2,
                "cursor": first["next_cursor"],
            },
        ).json()
        assert [item["path"].replace("\\", "/") for item in second["artifacts"]] == ["01/c"]
        assert second["truncated"] is False and second["next_offset"] is None

    def test_tree_bad_workspace_400(self, client: TestClient, tmp_path: Path) -> None:
        r = client.get(
            "/api/files/tree",
            params={"workspace": str(tmp_path / "nope"), "path": "."},
        )
        assert r.status_code == 400


# Static + root


class TestStaticAndRoot:
    def test_root_serves_index(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200
        assert "NorthStack Control Surface" in r.text
        assert "app.js" in r.text

    def test_static_js_reachable(self, client: TestClient) -> None:
        r = client.get("/static/js/app.js")
        assert r.status_code == 200
        # The compiled shell always has an import or router reference.
        body = r.text
        assert "import" in body or "router" in body.lower()


# Live provider tests (OFF by default; gated on MC_LIVE_API)


# Live provider config -- read from env, never a hardcoded endpoint.  The
# default (hermetic) suite is unaffected; these only matter when the operator
# runs `MC_LIVE_API=1 uv run pytest -m live`.
#   MC_LIVE_BASE_URL  base URL of the provider, e.g. http://127.0.0.1:8317/v1
#   MC_LIVE_MODEL     model name the endpoint serves, e.g. gpt-4o-mini
#   MC_LIVE_KEY_ENV   name of the env var holding the api key (NOT the value)
#                    e.g. CLIPROXY_API_KEY  -- the key itself must live in env
#   MC_LIVE_PROTOCOL  optional: openai_chat (default) | anthropic_messages
#                     | gemini_generate_content

_LIVE_BASE_URL = os.environ.get("MC_LIVE_BASE_URL", "").strip()
_LIVE_MODEL = os.environ.get("MC_LIVE_MODEL", "").strip()
_LIVE_KEY_ENV = os.environ.get("MC_LIVE_KEY_ENV", "").strip()
_LIVE_PROTOCOL = os.environ.get("MC_LIVE_PROTOCOL", "openai_chat").strip()


_LIVE_PROTOCOLS = {
    "anthropic_messages": Protocol.ANTHROPIC_MESSAGES,
    "gemini_generate_content": Protocol.GEMINI_GENERATE_CONTENT,
    "openai_chat": Protocol.OPENAI_CHAT,
}


def _live_configured() -> bool:
    """True iff the operator supplied a complete live provider via env."""
    if _LIVE_PROTOCOL not in _LIVE_PROTOCOLS:
        return False
    # The key must be exported under that name, not merely named.
    return bool(
        _LIVE_BASE_URL
        and _LIVE_MODEL
        and _LIVE_KEY_ENV
        and os.environ.get(_LIVE_KEY_ENV, "").strip()
    )


def _live_profile() -> ModelProfile:
    """A WORKER-role profile pointed at the env-configured live endpoint."""
    return ModelProfile(
        name="live-worker",
        protocol=_LIVE_PROTOCOLS[_LIVE_PROTOCOL],
        base_url=_LIVE_BASE_URL,
        model=_LIVE_MODEL,
        api_key_env=SecretEnvRef(env_var=_LIVE_KEY_ENV),
        roles={Role.WORKER, Role.REVIEWER, Role.PLANNER},
        max_concurrency=1,
    )


def _live_config(tmp_path: Path) -> tuple[NorthStackConfig, Path]:
    """A config with a real WORKER profile + routing, persisted to tmp TOML."""
    cfg = NorthStackConfig(
        name="LiveCo",
        profiles=[_live_profile()],
        commands=[],
        run=RunConfig(),
        routing=[
            {"role": Role.WORKER, "profiles": ["live-worker"]},
            {"role": Role.REVIEWER, "profiles": ["live-worker"]},
            {"role": Role.PLANNER, "profiles": ["live-worker"]},
        ],
    )
    path = tmp_path / "live-northstack.toml"
    from northstack.adapters.config_toml import save_config_to_toml

    save_config_to_toml(cfg, path)
    return cfg, path


@pytest.fixture
def live_app(tmp_path: Path) -> object:
    """A FastAPI app backed by a real live provider -- live tests only.

    Short-circuits to a skip when no live endpoint is configured via env, so a
    missing endpoint is a clean skip -- never a fixture-setup ERROR.
    """
    _skip_if_unconfigured()
    _cfg, path = _live_config(tmp_path)
    application = create_app(path)
    application.state.files_base_root = str(tmp_path)
    return application


@pytest.fixture
def live_client(live_app: object) -> Iterator[TestClient]:
    with TestClient(live_app) as c:
        yield c


def _skip_if_unconfigured() -> None:
    """Skip (not fail) when the operator enabled live tests but gave no endpoint."""
    if not _live_configured():
        pytest.skip(
            "live provider not configured -- set MC_LIVE_BASE_URL, MC_LIVE_MODEL, "
            "MC_LIVE_KEY_ENV (and export the key under that name) to run live tests"
        )


@pytest.mark.live
class TestLiveProvider:
    """Live provider on the wire -- only runs when MC_LIVE_API=1.

    Asserts the REAL end-to-end path: POST /runs -> run_async against a live
    model gateway -> a real ledger of events + a terminal outcome + integrity.
    The endpoint is fully env-driven so the default suite stays hermetic; with
    no endpoint configured the live tests skip, they never fail the suite.
    """

    def test_live_marker_is_honored_when_disabled(self, client: TestClient) -> None:
        """With MC_LIVE_API unset this whole class is skipped by conftest.

        Run only to prove the marker + gate wiring -- it is itself the toggle
        contract: it executes iff the toggle is on.  When the toggle is off it
        never runs (skipped by the autouse fixture), so this assertion holds by
        construction whenever the body is reached.
        """
        assert os.environ.get("MC_LIVE_API", "0").lower() in ("1", "true", "yes", "on")

    def test_live_config_resolves_key_ok(self, live_client: TestClient) -> None:
        """The live profile's api_key_env resolves OK before any run starts."""
        _skip_if_unconfigured()
        r = live_client.get("/api/config")
        assert r.status_code == 200
        prof = next(p for p in r.json()["profiles"] if p["name"] == "live-worker")
        assert prof["key_status"].startswith("env:")
        assert prof["key_status"].endswith("OK"), prof["key_status"]

    def test_live_run_reaches_terminal(self, live_client: TestClient, tmp_path: Path) -> None:
        """A real run against the live provider reaches a terminal status.

        A reachable model that can create a file should verify; weaker models
        may abstain -- both are acceptable as 'the live path works'.  A run
        stuck at 'executing' (provider never responded) is the only failure.
        """
        _skip_if_unconfigured()
        ws = tmp_path / "live-ws"
        ws.mkdir()
        goal = "Create a file called hello.txt containing the text 'NorthStack ran OK'."
        r = live_client.post("/api/runs", json={"goal": goal, "workspace_root": str(ws)})
        assert r.status_code == 200, r.text
        run_id = r.json()["run_id"]

        snap = _wait_for_terminal(live_client, run_id, timeout_s=120.0)
        assert snap["status"] in ("verified", "abstained", "failed"), snap
        # The terminal status must NOT be 'executing' -- that means the live
        # provider never answered and the run hung (a real regression).
        assert snap["status"] != "executing", "live run never reached a terminal state"

    def test_live_run_has_events_and_integrity(
        self, live_client: TestClient, tmp_path: Path
    ) -> None:
        """A live run produces a hash-chain ledger that verifies intact."""
        _skip_if_unconfigured()
        ws = tmp_path / "live-ws2"
        ws.mkdir()
        goal = "Create a file called hello.txt containing the text 'NorthStack ran OK'."
        r = live_client.post("/api/runs", json={"goal": goal, "workspace_root": str(ws)})
        assert r.status_code == 200, r.text
        run_id = r.json()["run_id"]
        _wait_for_terminal(live_client, run_id, timeout_s=120.0)

        # The run emitted events (a real ledger, not an empty stub).
        ev = live_client.get(f"/api/runs/{run_id}/events", params={"since": 0, "limit": 5000})
        assert ev.status_code == 200
        assert len(ev.json()["events"]) > 0

        # The hash chain verifies (integrity == ok for any terminal run).
        ig = live_client.get(f"/api/runs/{run_id}/integrity")
        assert ig.status_code == 200
        assert ig.json()["ok"] is True, ig.json()
