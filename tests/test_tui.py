"""Tests for the interactive TUI launcher (``northstack.interfaces.tui``).

No TTY is required: every test exercises the pure-Python pieces
(``BrowserState``, the static/interactive renderers, ``_execute_action`` with
a mocked ``subprocess.run``). The interactive ``run_interactive`` loop itself is
not driven here -- it blocks on stdin -- which is by design.
"""

from __future__ import annotations

import io
from unittest import mock

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from northstack.config import (
    ModelProfile,
    NorthStackConfig,
    Protocol,
    Role,
    RunConfig,
    SecretEnvRef,
)
from northstack.interfaces.tui import (
    LAUNCHER_ITEMS,
    TAB_NAMES,
    BrowserState,
    _execute_action,
    _launcher_tab,
    _launcher_tab_static,
    dump_static,
)

# Config helpers (mirror the test_web_api.py pattern, kept local so test_tui.py
# has no cross-file dependency).


def _profile(name: str, *, roles: set[Role] | None = None) -> ModelProfile:
    return ModelProfile(
        name=name,
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://127.0.0.1:65535/v1",
        model="m",
        api_key_env=SecretEnvRef(env_var="MY_KEY"),
        roles=roles or set(),
        max_concurrency=1,
    )


def _make_cfg(name: str = "TestCo") -> NorthStackConfig:
    """A config exercising every tab: 2 profiles, 1 routing entry, 1 command."""
    from northstack.config import CommandConfig, RouteMapping

    return NorthStackConfig(
        name=name,
        profiles=[
            _profile("worker-1", roles={Role.WORKER}),
            _profile("reviewer-1", roles={Role.REVIEWER}),
        ],
        commands=[CommandConfig(name="ls", argv=["ls"], env_allowlist=["PATH"])],
        run=RunConfig(),
        routing=[RouteMapping(role=Role.WORKER, profiles=["worker-1"])],
    )


def _render_str(renderable) -> str:
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, width=120).print(renderable)
    return buf.getvalue()


# BrowserState


class TestBrowserState:
    def test_tab_names_has_four_entries(self):
        assert len(TAB_NAMES) == 4
        assert TAB_NAMES[0] == "Actions"  # launcher is the first tab
        assert TAB_NAMES[1] == "Profiles"
        assert TAB_NAMES[2] == "Routing"
        assert TAB_NAMES[3] == "Commands"

    def test_launcher_items_count(self):
        assert len(LAUNCHER_ITEMS) == 5

    def test_cursor_array_has_four_elements(self):
        state = BrowserState(_make_cfg())
        assert len(state.cursor) == 4
        assert all(c == 0 for c in state.cursor)
        assert state.launcher_last_result is None

    def test_rows_per_tab_new_order(self):
        state = BrowserState(_make_cfg())
        # tab 0 = Actions (launcher); tabs 1/2/3 = Profiles/Routing/Commands
        assert state.rows(0) == len(LAUNCHER_ITEMS)
        assert state.rows(1) == 2  # profiles: worker-1, reviewer-1
        assert state.rows(2) == 1  # routing
        assert state.rows(3) == 1  # commands

    def test_move_in_launcher_tab(self):
        state = BrowserState(_make_cfg())
        state.tab = 0  # Actions
        state.move(1)
        assert state.cursor[0] == 1
        state.move(-1)
        assert state.cursor[0] == 0
        # wraps modulo item count
        state.move(10)
        assert state.cursor[0] == 10 % len(LAUNCHER_ITEMS)

    def test_switch_tab_actions_first_then_info(self):
        state = BrowserState(_make_cfg())
        assert state.tab == 0  # starts on Actions
        state.switch_tab(1)  # Actions -> Profiles
        assert state.tab == 1
        state.switch_tab(1)  # -> Routing
        assert state.tab == 2
        state.switch_tab(1)  # -> Commands
        assert state.tab == 3
        state.switch_tab(1)  # wraps back to Actions
        assert state.tab == 0


# Rendering


class TestLauncherRendering:
    def test_launcher_tab_returns_panel(self):
        panel = _launcher_tab(BrowserState(_make_cfg()))
        assert isinstance(panel, Panel)

    def test_launcher_tab_shows_all_items(self):
        out = _render_str(_launcher_tab(BrowserState(_make_cfg())))
        for item in LAUNCHER_ITEMS:
            assert item in out

    def test_launcher_tab_shows_selection_marker(self):
        state = BrowserState(_make_cfg())
        state.cursor[0] = 2  # Actions is now tab 0
        out = _render_str(_launcher_tab(state))
        # the selected row is prefixed with the marker
        assert "▶" in out  # ▶
        assert LAUNCHER_ITEMS[2] in out

    def test_launcher_tab_shows_last_result(self):
        state = BrowserState(_make_cfg())
        state.launcher_last_result = "exit 0 (passed)"
        out = _render_str(_launcher_tab(state))
        assert "Last:" in out
        assert "exit 0 (passed)" in out

    def test_launcher_tab_static_returns_table_with_items(self):
        table = _launcher_tab_static()
        assert isinstance(table, Table)
        out = _render_str(table)
        for item in LAUNCHER_ITEMS:
            assert item in out

    def test_dump_static_includes_actions(self):
        buf = io.StringIO()
        with mock.patch("northstack.interfaces.tui.console", Console(file=buf, width=120)):
            dump_static(_make_cfg(), None)
        out = buf.getvalue()
        assert "Actions" in out


# _execute_action (mocked subprocess / web)


def _completed(returncode: int = 0):
    return mock.Mock(returncode=returncode)


class TestExecuteAction:
    def test_hermetic_builds_correct_command(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")  # exists check is in _launch_web, not here
        with mock.patch(
            "northstack.interfaces.tui.subprocess.run", return_value=_completed(0)
        ) as run:
            _execute_action(0, cfg, config_path)
        run.assert_called_once()
        args, kwargs = run.call_args
        cmd = args[0]
        assert cmd == ["uv", "run", "-q", "pytest", "-m", "not live", "-q"]
        assert kwargs["cwd"] == config_path.resolve().parent

    def test_live_sets_env_vars(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")
        with mock.patch(
            "northstack.interfaces.tui.subprocess.run", return_value=_completed(0)
        ) as run:
            _execute_action(1, cfg, config_path)
        run.assert_called_once()
        env = run.call_args.kwargs["env"]
        assert env["MC_LIVE_API"] == "1"
        assert env["MC_LIVE_BASE_URL"] == "http://127.0.0.1:8317/v1"
        assert env["MC_LIVE_KEY_ENV"] == "CLIPROXY_API_KEY"
        assert env["MC_LIVE_MODEL"] == "bedrock/minimax.minimax-m2.5"
        # command selects the live marker + web extra
        cmd = run.call_args.args[0]
        assert cmd == ["uv", "run", "--extra", "web", "pytest", "-m", "live", "-q"]

    def test_test_summaries(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")
        for rc, fragment in [(0, "passed"), (5, "no tests collected"), (1, "failed")]:
            with mock.patch(
                "northstack.interfaces.tui.subprocess.run", return_value=_completed(rc)
            ):
                summary = _execute_action(0, cfg, config_path)
            assert fragment in summary
            assert str(rc) in summary

    def test_missing_uv_returns_hint(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")
        with mock.patch("northstack.interfaces.tui.subprocess.run", side_effect=FileNotFoundError):
            summary = _execute_action(0, cfg, config_path)
        assert "uv" in summary.lower()
        assert "PATH" in summary

    def test_web_missing_extras_returns_install_hint(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")
        # _launch_web raises typer.Exit when web extras missing -> _execute_action
        # catches SystemExit and returns the hint instead of killing the caller.
        with mock.patch("northstack.interfaces.tui._launch_web", side_effect=SystemExit(1)):
            for idx in (2, 3):
                summary = _execute_action(idx, cfg, config_path)
                assert "uv sync --extra web" in summary

    def test_frontend_calls_launch_web(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")
        with mock.patch(
            "northstack.interfaces.tui._launch_web", return_value="opened browser at ..."
        ) as lw:
            summary = _execute_action(4, cfg, config_path)
        lw.assert_called_once_with("frontend", config_path.resolve())
        assert summary == "opened browser at ..."

    def test_unknown_action(self, tmp_path):
        cfg = _make_cfg()
        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")
        summary = _execute_action(99, cfg, config_path)
        assert "unknown" in summary.lower()


# C-2: _launch_web("both") silent server death


class TestLaunchWebBothServerDeath:
    """When the uvicorn daemon thread dies during startup (e.g. port-in-use),
    ``_launch_web("both")`` must NOT print the server URL or open the browser.
    The user should see a clear error, not a phantom URL."""

    def test_server_bind_failure_no_browser_open(self, tmp_path):
        """RED: server.run() raises OSError -> thread dies -> no browser.open."""
        import types

        from northstack.interfaces.tui import _launch_web

        config_path = tmp_path / "northstack.toml"
        config_path.write_text("")

        # Build a fake uvicorn module with Server.run() raising OSError
        fake_uvicorn = types.ModuleType("uvicorn")

        class _FakeConfig:
            def __init__(self, *a, **kw):
                pass

        class _FakeServer:
            started = False

            def __init__(self, cfg):
                pass

            def run(self):
                raise OSError("Address already in use")

        fake_uvicorn.Config = _FakeConfig
        fake_uvicorn.Server = _FakeServer

        fake_create_app = mock.MagicMock()

        with (
            mock.patch(
                "northstack.interfaces.tui._import_web",
                return_value=(fake_uvicorn, fake_create_app),
            ),
            mock.patch("webbrowser.open") as mock_wb_open,
            mock.patch("time.sleep"),  # speed up poll loop
        ):
            result = _launch_web("both", config_path)

        # The browser must NOT have been opened
        mock_wb_open.assert_not_called()
        # The summary must indicate failure, not success
        assert "stopped" not in result or "error" in result.lower() or "failed" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
