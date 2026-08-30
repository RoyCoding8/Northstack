from __future__ import annotations

import math
import sys
import time

import pytest

from northstack.adapters.workspace.commands import run_command


@pytest.mark.parametrize(
    ("stdout_size", "stderr_size", "budget"),
    [
        (750, 0, 1000),
        (0, 750, 1000),
        (400, 400, 1000),
        (500, 500, 1000),
        (900, 100, 1000),
        (100, 900, 1000),
        (750, 750, 1000),
    ],
)
def test_combined_output_uses_the_whole_budget(
    tmp_path, stdout_size: int, stderr_size: int, budget: int
) -> None:
    code = f"import os;os.write(1,b'a'*{stdout_size});os.write(2,b'b'*{stderr_size})"
    result = run_command(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=5,
        max_output_bytes=budget,
    )
    total = stdout_size + stderr_size
    assert result.ok
    assert result.total_bytes == total
    assert len(result.stdout) + len(result.stderr) == min(total, budget)
    assert result.truncated is (total > budget)


def test_truncated_stderr_keeps_the_tail(tmp_path) -> None:
    code = "import os;os.write(2,b'HEAD'+b'x'*1000+b'TAIL')"
    result = run_command(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=5,
        max_output_bytes=100,
    )
    assert result.stderr.endswith(b"TAIL")
    assert result.truncated


def test_timeout_retains_output_and_exit_status(tmp_path) -> None:
    code = "import os,time;os.write(2,b'ready');time.sleep(5)"
    result = run_command(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=0.05,
        max_output_bytes=100,
    )
    assert not result.ok
    assert result.stderr == b"ready"
    assert result.total_bytes == 5
    assert result.exit_code is not None
    assert "timed out" in result.error.lower()


def test_timeout_cleanup_failure_preserves_timeout_and_reaps(tmp_path) -> None:
    def terminate_then_fail(proc) -> None:
        proc.kill()
        raise OSError("injected cleanup failure")

    result = run_command(
        [sys.executable, "-c", "import time;time.sleep(5)"],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=0.05,
        max_output_bytes=100,
        terminate=terminate_then_fail,
    )
    assert not result.ok
    assert "timed out" in result.error.lower()
    assert "injected cleanup failure" in result.error
    assert result.exit_code is not None


@pytest.mark.parametrize(
    ("stdout_size", "stderr_size", "expected"),
    [(1, 0, (b"a", b"")), (0, 1, (b"", b"b")), (1, 1, (b"", b"b"))],
)
def test_one_byte_output_budget(
    tmp_path, stdout_size: int, stderr_size: int, expected: tuple[bytes, bytes]
) -> None:
    code = f"import os;os.write(1,b'a'*{stdout_size});os.write(2,b'b'*{stderr_size})"
    result = run_command(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=5,
        max_output_bytes=1,
    )
    assert (result.stdout, result.stderr) == expected
    assert result.truncated is (stdout_size + stderr_size > 1)


def test_timeout_during_heavy_stdout_and_stderr_is_bounded(tmp_path) -> None:
    code = "import os,time;os.write(1,b'a'*2000000);os.write(2,b'b'*2000000);time.sleep(5)"
    result = run_command(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=0.5,
        max_output_bytes=1025,
    )
    assert not result.ok
    assert result.truncated
    assert len(result.stdout) + len(result.stderr) == 1025
    assert result.total_bytes == 4_000_000
    assert "timed out" in result.error.lower()


@pytest.mark.parametrize(
    ("argv", "timeout", "cap"),
    [([], 1.0, 1), (["x"], math.nan, 1), (["x"], -1.0, 1), (["x"], 1.0, -1)],
)
def test_invalid_command_bounds_are_rejected(argv, timeout: float, cap: int, tmp_path) -> None:
    with pytest.raises(ValueError):
        run_command(
            argv,
            cwd=tmp_path,
            env_allowlist=[],
            timeout_seconds=timeout,
            max_output_bytes=cap,
        )


def test_timeout_kills_spawned_descendants(tmp_path) -> None:
    started, leaked = tmp_path / "started", tmp_path / "leaked"
    child = (
        "import pathlib,time;"
        f"pathlib.Path({str(started)!r}).write_text('started');"
        "time.sleep(1.5);"
        f"pathlib.Path({str(leaked)!r}).write_text('leaked')"
    )
    parent = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child!r}]);"
        "time.sleep(10)"
    )
    result = run_command(
        [sys.executable, "-c", parent],
        cwd=tmp_path,
        env_allowlist=[],
        timeout_seconds=1,
        max_output_bytes=100,
    )
    assert not result.ok
    assert started.exists()
    time.sleep(1)
    assert not leaked.exists()
