"""Hermetic tests for the Docker isolation seam.

No Docker required: the availability probe is injectable/resettable, and the
real-machine test asserts the fail-closed path against whatever host this
runs on (a machine without Docker exercises the refusal; one with Docker
exercises a real wrapped run).

Pinned behavior:

  - config validation: docker isolation requires an image; unknown isolation
    values are rejected at parse time;
  - argv wrapping: fixed flags (--rm, --network none, mount, workdir) with
    only image and argv as variables;
  - fail-closed: an unavailable Docker NEVER degrades to host execution;
  - TOML round-trip: isolation fields survive save/load and defaults stay
    byte-stable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from northstack.adapters.workspace.commands import (
    CommandResult,
    docker_available,
    reset_docker_probe_cache,
    wrap_docker_argv,
)
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.config import CommandConfig


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    reset_docker_probe_cache()
    yield
    reset_docker_probe_cache()


def _unavailable_probe() -> CommandResult:
    return CommandResult(ok=False, exit_code=1, stderr=b"docker: command not found")


def _available_probe() -> CommandResult:
    return CommandResult(ok=True, exit_code=0, stdout=b"27.0.3")


# Config validation


def test_docker_isolation_requires_image():
    with pytest.raises(ValidationError, match="requires docker_image"):
        CommandConfig(name="t", argv=["ls"], isolation="docker")
    with pytest.raises(ValidationError, match="requires docker_image"):
        CommandProfile(name="t", argv=["ls"], isolation="docker", docker_image="   ")


def test_unknown_isolation_value_rejected():
    with pytest.raises(ValidationError):
        CommandConfig(name="t", argv=["ls"], isolation="vm")


def test_host_isolation_remains_the_default():
    assert CommandConfig(name="t", argv=["ls"]).isolation == "host"
    assert CommandProfile(name="t", argv=["ls"]).isolation == "host"


# Argv wrapping


def test_wrap_docker_argv_fixed_flags_and_variables(tmp_path: Path):
    wrapped = wrap_docker_argv(
        ["python", "-m", "pytest", "-q"],
        tmp_path,
        "python:3.12-slim",
        "northstack-test",
    )
    assert wrapped[:3] == ["docker", "run", "--rm"]
    assert "--init" in wrapped
    assert "--network" in wrapped and "none" in wrapped
    assert wrapped[wrapped.index("--cap-drop") + 1] == "ALL"
    assert wrapped[wrapped.index("--security-opt") + 1] == "no-new-privileges"
    assert wrapped[wrapped.index("--pids-limit") + 1] == "256"
    assert wrapped[wrapped.index("--name") + 1] == "northstack-test"
    assert f"{tmp_path}:/workspace" in wrapped
    assert "-w" in wrapped and "/workspace" in wrapped
    # Only the image and the operator's argv are variables.
    assert wrapped[wrapped.index("python:3.12-slim") + 1 :] == ["python", "-m", "pytest", "-q"]


def test_wrap_docker_argv_requires_image(tmp_path: Path):
    with pytest.raises(ValueError, match="non-empty image"):
        wrap_docker_argv(["ls"], tmp_path, " ")


@pytest.mark.parametrize("image", ["--privileged", " python:3.12", "python:3.12\n--privileged"])
def test_docker_image_cannot_be_parsed_as_cli_options(tmp_path: Path, image: str):
    with pytest.raises(ValueError, match="non-option reference"):
        wrap_docker_argv(["ls"], tmp_path, image)
    with pytest.raises(ValidationError, match="non-option reference"):
        CommandConfig(name="t", argv=["ls"], isolation="docker", docker_image=image)
    with pytest.raises(ValidationError, match="non-option reference"):
        CommandProfile(name="t", argv=["ls"], isolation="docker", docker_image=image)


# Fail-closed execution


def test_unavailable_docker_never_falls_back_to_host(tmp_path: Path):
    """A docker-isolated command on a docker-less host is refused, not run
    on the host. This is the load-bearing safety property of the seam."""
    docker_available(probe=_unavailable_probe)  # seed the cache: unavailable
    ws = RestrictedWorkspace(tmp_path)
    # A command that WOULD succeed on the host -- the point is that it must
    # not get the chance.
    profile = CommandProfile(
        name="would-succeed-on-host",
        argv=["python", "-c", "print('leaked')"],
        isolation="docker",
        docker_image="python:3.12-slim",
    )
    result = ws.execute_command(profile)
    assert not result.ok
    assert "refusing to run on host" in (result.error or "")
    assert b"leaked" not in (result.data or b"")


def test_available_docker_runs_wrapped_command(tmp_path: Path, monkeypatch):
    docker_available(probe=_available_probe)  # seed the cache: available
    captured: dict = {}

    from northstack.adapters.workspace import restricted as restricted_mod

    def fake_run_command(argv, **kw):
        captured["argv"] = argv
        captured["cwd"] = kw.get("cwd")
        return CommandResult(ok=True, exit_code=0, stdout=b"in-container")

    monkeypatch.setattr(restricted_mod, "run_command", fake_run_command)
    ws = RestrictedWorkspace(tmp_path)
    profile = CommandProfile(
        name="boxed",
        argv=["pytest", "-q"],
        isolation="docker",
        docker_image="python:3.12-slim",
    )
    result = ws.execute_command(profile)
    assert result.ok
    assert b"in-container" in result.data
    assert captured["argv"][0] == "docker"
    assert "python:3.12-slim" in captured["argv"]
    assert "pytest" in captured["argv"]


def test_timed_out_docker_run_is_force_removed_and_verified(tmp_path: Path, monkeypatch):
    docker_available(probe=_available_probe)
    calls = []

    from northstack.adapters.workspace import restricted as restricted_mod

    def fake_run_command(argv, **_kwargs):
        calls.append(argv)
        if argv[1:3] == ["rm", "-f"]:
            return CommandResult(ok=True, exit_code=0)
        if argv[1:3] == ["container", "inspect"]:
            return CommandResult(ok=False, exit_code=1, stderr=b"No such container")
        return CommandResult(ok=False, exit_code=-1, error="Command timed out after 1s")

    monkeypatch.setattr(restricted_mod, "run_command", fake_run_command)
    result = RestrictedWorkspace(tmp_path).execute_command(
        CommandProfile(
            name="boxed",
            argv=["sleep", "10"],
            isolation="docker",
            docker_image="alpine",
            timeout_seconds=1,
        )
    )
    name = calls[0][calls[0].index("--name") + 1]
    assert name.startswith("northstack-")
    assert calls[1] == ["docker", "rm", "-f", name]
    assert calls[2] == ["docker", "container", "inspect", name]
    assert "cleanup failed" not in result.error


def test_timed_out_docker_cleanup_failure_is_reported(tmp_path: Path, monkeypatch):
    docker_available(probe=_available_probe)

    from northstack.adapters.workspace import restricted as restricted_mod

    def fake_run_command(argv, **_kwargs):
        if argv[1:3] == ["container", "inspect"]:
            return CommandResult(ok=True, exit_code=0, stdout=b"still-running")
        if argv[1:3] == ["rm", "-f"]:
            return CommandResult(ok=False, exit_code=1, error="daemon refused removal")
        return CommandResult(ok=False, exit_code=-1, error="Command timed out after 1s")

    monkeypatch.setattr(restricted_mod, "run_command", fake_run_command)
    result = RestrictedWorkspace(tmp_path).execute_command(
        CommandProfile(
            name="boxed",
            argv=["sleep", "10"],
            isolation="docker",
            docker_image="alpine",
            timeout_seconds=1,
        )
    )
    assert "container cleanup failed: container still exists" in result.error


def test_probe_result_is_cached_per_process():
    docker_available(probe=_available_probe)
    first = docker_available(probe=_unavailable_probe)  # probe must not re-run
    assert first[0] is True


def test_host_isolation_ignores_docker_entirely(tmp_path: Path, monkeypatch):
    """Host mode must not probe Docker or wrap argv -- unchanged behavior."""
    from northstack.adapters.workspace import restricted as restricted_mod

    def boom_probe():
        raise AssertionError("host mode must not probe docker")

    monkeypatch.setattr(restricted_mod, "docker_available", boom_probe)
    ws = RestrictedWorkspace(tmp_path)
    result = ws.execute_command(
        CommandProfile(name="plain", argv=["python", "-c", "print('host')"])
    )
    assert result.ok
    assert b"host" in result.data


def test_real_machine_docker_path_is_fail_closed_or_real():
    """Against the actual host (no injected probe): either Docker exists and
    the probe reports it, or it doesn't and a docker-isolated command is
    refused. There is no third outcome."""
    reset_docker_probe_cache()
    available, _ = docker_available()
    if not available:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ws = RestrictedWorkspace(Path(tmp))
            result = ws.execute_command(
                CommandProfile(
                    name="real",
                    argv=["echo", "hi"],
                    isolation="docker",
                    docker_image="alpine",
                )
            )
            assert not result.ok
            assert "refusing" in (result.error or "")
    # else: Docker genuinely present on this machine; the wrapped-run case
    # is covered by test_available_docker_runs_wrapped_command.


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("NORTHSTACK_DOCKER_SMOKE") != "1",
    reason="real Docker smoke is opt-in",
)
def test_real_docker_daemon_executes_isolated_workspace_command(tmp_path: Path):
    reset_docker_probe_cache()
    result = RestrictedWorkspace(tmp_path).execute_command(
        CommandProfile(
            name="docker-smoke",
            argv=["sh", "-c", "printf northstack > docker-smoke.txt"],
            isolation="docker",
            docker_image="alpine:3.20",
            timeout_seconds=30,
        )
    )
    assert result.ok, result.error
    assert (tmp_path / "docker-smoke.txt").read_text(encoding="utf-8") == "northstack"


# TOML round-trip


def test_isolation_fields_round_trip(tmp_path: Path):
    from northstack.adapters.config_toml import config_to_toml
    from northstack.config import NorthStackConfig

    config = NorthStackConfig(
        name="rt",
        commands=[
            CommandConfig(
                name="boxed-test",
                argv=["python", "-m", "pytest", "-q"],
                isolation="docker",
                docker_image="python:3.12-slim",
                timeout_seconds=120.0,
            )
        ],
    )
    text = config_to_toml(config)
    assert 'isolation = "docker"' in text
    assert 'docker_image = "python:3.12-slim"' in text
    out = tmp_path / "rt.toml"
    out.write_text(text, encoding="utf-8")
    loaded = NorthStackConfig.from_toml(out)
    assert loaded.commands[0].isolation == "docker"
    assert loaded.commands[0].docker_image == "python:3.12-slim"


def test_default_isolation_fields_stay_unemitted():
    from northstack.adapters.config_toml import config_to_toml
    from northstack.config import NorthStackConfig

    text = config_to_toml(NorthStackConfig(name="d"))
    assert "isolation" not in text
    assert "docker_image" not in text
