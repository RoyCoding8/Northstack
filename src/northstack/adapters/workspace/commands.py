"""Bounded non-shell subprocess execution."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from northstack.domain.container_policy import validate_docker_image


class CommandResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    stdout: bytes = b""
    stderr: bytes = b""
    exit_code: int | None = None
    truncated: bool = False
    total_bytes: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    error: str = ""


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt" and proc.pid is not None:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                timeout=5,
            )
            if result.returncode:
                proc.kill()
        else:
            kill_group = getattr(os, "killpg", None)
            if kill_group is None or proc.pid is None:
                proc.kill()
            else:
                try:
                    kill_group(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
                except OSError:
                    proc.kill()
    except (OSError, subprocess.TimeoutExpired):
        proc.kill()
    if proc.poll() is None:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def _capture_output(
    stdout_path: str, stderr_path: str, max_bytes: int
) -> tuple[bytes, bytes, int, int]:
    totals = os.path.getsize(stdout_path), os.path.getsize(stderr_path)
    budget = max(0, max_bytes)
    limits = [min(totals[0], budget // 2), min(totals[1], budget - budget // 2)]
    spare = budget - sum(limits)
    for index in (0, 1):
        donated = min(totals[index] - limits[index], spare)
        limits[index] += donated
        spare -= donated
    with open(stdout_path, "rb") as stream:
        stdout = stream.read(limits[0])
    with open(stderr_path, "rb") as stream:
        if limits[1] < totals[1]:
            stream.seek(-limits[1], os.SEEK_END)
        stderr = stream.read(limits[1])
    return stdout, stderr, totals[0], totals[1]


_DOCKER_WORKSPACE_TARGET = "/workspace"

_docker_probe_cache: tuple[bool, str] | None = None


def wrap_docker_argv(
    argv: list[str],
    workspace_root: Path | str,
    image: str,
    container_name: str | None = None,
) -> list[str]:
    """Wrap an exact argv for isolated execution in a throwaway container.

    The container is ``--rm`` (throwaway), ``--network none`` (no egress),
    runs as the image's default user, mounts the workspace read-write at
    ``/workspace``, and sets ``/workspace`` as cwd. Only these fixed flags
    are ever injected -- the image and the operator's argv are the variables.
    """
    image = validate_docker_image(image)
    mount = f"{Path(workspace_root)}:{_DOCKER_WORKSPACE_TARGET}"
    return [
        "docker",
        "run",
        "--rm",
        "--init",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        *(["--name", container_name] if container_name else []),
        "-v",
        mount,
        "-w",
        _DOCKER_WORKSPACE_TARGET,
        image,
        *argv,
    ]


def docker_available(probe: Callable[[], CommandResult] | None = None) -> tuple[bool, str]:
    """Probe the Docker CLI + daemon once per process. Returns (ok, detail).

    ``probe`` is injectable for tests; production asks the daemon for its
    version (``docker version --format ...``), which requires the CLI on PATH
    AND a reachable daemon -- the two things docker isolation needs.
    """
    global _docker_probe_cache
    if _docker_probe_cache is not None:
        return _docker_probe_cache
    if probe is None:

        def _probe() -> CommandResult:
            return run_command(
                ["docker", "version", "--format", "{{.Server.Version}}"],
                cwd=None,
                env_allowlist=["PATH"],
                timeout_seconds=10.0,
                max_output_bytes=4096,
            )

        probe = _probe
    result = probe()
    detail = (
        f"exit={result.exit_code} {result.stderr.decode(errors='replace').strip()[:120]}".strip()
        if not result.ok
        else result.stdout.decode(errors="replace").strip()[:60]
    )
    _docker_probe_cache = (result.ok, detail or "unavailable")
    return _docker_probe_cache


def reset_docker_probe_cache() -> None:
    """Test seam: forget the cached probe so a new probe runs."""
    global _docker_probe_cache
    _docker_probe_cache = None


_OS_BASELINE_VARS = (
    ("SystemRoot", "windir"),
    ("SystemDrive",),
    ("ComSpec",),
    ("PATHEXT",),
    ("TEMP", "TMP"),
)


def _baseline_env(source_env: Mapping[str, str]) -> dict[str, str]:
    lowered = {key.lower(): value for key, value in source_env.items()}
    env: dict[str, str] = {}
    for names in _OS_BASELINE_VARS:
        for name in names:
            if name.lower() in lowered:
                env[name] = lowered[name.lower()]
                break
    return env


def run_command(
    argv: list[str],
    *,
    cwd: Path | str | None,
    env_allowlist: list[str],
    timeout_seconds: float,
    max_output_bytes: int,
    environ: dict[str, str] | None = None,
    terminate: Callable[[subprocess.Popen[bytes]], None] = _terminate_process_tree,
) -> CommandResult:
    """Run exact argv with only explicitly allowed environment variables."""
    if not argv:
        raise ValueError("argv must not be empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout_seconds must be finite and nonnegative")
    if max_output_bytes < 0:
        raise ValueError("max_output_bytes must be nonnegative")
    start = time.perf_counter()
    source_env = os.environ if environ is None else environ
    allowed = {key: source_env[key] for key in env_allowlist if key in source_env}
    if sys.platform == "win32":
        env = {**_baseline_env(source_env), **allowed}
    else:
        env = allowed
    timeout = timeout_seconds if timeout_seconds > 0 else None
    stdout_fd: int | None = None
    stderr_fd: int | None = None
    stdout_path = ""
    stderr_path = ""
    proc: subprocess.Popen[bytes] | None = None
    stdout = b""
    stderr = b""
    total_stdout = 0
    total_stderr = 0
    timed_out = False
    cleanup_error = ""
    phase = "create command output files"
    try:
        stdout_fd, stdout_path = tempfile.mkstemp(prefix=".mc_stdout_")
        stderr_fd, stderr_path = tempfile.mkstemp(prefix=".mc_stderr_")
        phase = "start command"
        proc = subprocess.Popen(
            argv,
            stdout=stdout_fd,
            stderr=stderr_fd,
            env=env,
            cwd=str(cwd) if cwd is not None else None,
            shell=False,
            start_new_session=os.name != "nt",
        )
        os.close(stdout_fd)
        stdout_fd = None
        os.close(stderr_fd)
        stderr_fd = None
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                terminate(proc)
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_error = f"; cleanup failed: {exc}"
            if proc.returncode is None:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_error += f"; reap failed: {exc}"
        phase = "capture command output"
        stdout, stderr, total_stdout, total_stderr = _capture_output(
            stdout_path, stderr_path, max_output_bytes
        )
    except OSError as exc:
        return CommandResult(
            ok=False,
            error=f"Failed to {phase}: {exc}",
            duration_ms=int((time.perf_counter() - start) * 1000),
        )
    finally:
        for fd in (stdout_fd, stderr_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for temp_path in (stdout_path, stderr_path):
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    total_bytes = total_stdout + total_stderr
    exit_code = proc.returncode if proc is not None and proc.returncode is not None else -1
    return CommandResult(
        ok=exit_code == 0 and not timed_out,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        truncated=total_bytes > max_output_bytes,
        total_bytes=total_bytes,
        duration_ms=int((time.perf_counter() - start) * 1000),
        error=(
            f"Command timed out after {timeout_seconds}s{cleanup_error}"
            if timed_out
            else stderr.decode(errors="replace").strip()
            if exit_code != 0
            else ""
        ),
    )
