"""Interactive full-screen config browser for northstack.

A real TUI (not a static dump): full-screen, keyboard-driven, stays open until
you quit. Built on ``rich`` for rendering and the stdlib for raw key reading
(``msvcrt`` on Windows, ``termios``/``tty`` on POSIX) -- no extra dependency.

It loads secrets from a local ``.env`` (python-dotenv) *before* constructing the
config, so a bare ``uv run python -m northstack.interfaces.tui`` resolves
``CLIPROXY_API_KEY`` etc. without prefixing every shell command.

Controls
--------
    ← / →, Tab / Shift+Tab   switch tabs (Actions -> Profiles -> Routing -> Commands)
    j / k, Up / Down          move the cursor within a tab
    Enter                     (Actions) launch the selected item; (Profiles) expand details
    q / Esc                   quit

The first tab is **Actions** -- the launcher (run tests, launch the web UI,
launch backend/frontend, quit). The info tabs (Profiles / Routing / Commands)
follow it.

Usage::

    uv run python -m northstack.interfaces.tui        # ./northstack.toml, ./.env
    python -m northstack.interfaces.tui --config path/to/x.toml
    ./run.bat (Windows) / ./run.sh (POSIX)             # repo launchers

When stdout is not a TTY (piped, redirected, tests), the interactive loop is
skipped and a static one-shot dump is printed instead -- so nothing breaks in
automation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import typer
from dotenv import load_dotenv
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from northstack.config import Capability, NorthStackConfig, Role
from pydantic import ValidationError

app = typer.Typer(help="northstack: interactive config browser TUI", add_completion=False)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

console = Console(force_terminal=True)


def load_secrets(config_path: Path) -> Path | None:
    """Load a .env file so SecretEnvRef.resolve() can read the keys.

    Real process environment takes precedence (override=False): an exported
    var on the shell always beats the file. Returns the path loaded, or None.
    """
    cfg_dir = config_path.resolve().parent
    for cand in (cfg_dir / ".env", Path.cwd() / ".env"):
        if cand.is_file():
            load_dotenv(cand, override=False)
            return cand
    return None


def _roles_cell(profile) -> str:
    order = [Role.WORKER, Role.REVIEWER, Role.PLANNER, Role.SPECIALIST, Role.ORCHESTRATOR]
    present = [r.value for r in order if r in profile.roles]
    return ", ".join(present) if present else "(none)"


def _caps_cell(profile) -> str:
    order = [
        Capability.TOOL_USE,
        Capability.NATIVE_JSON_SCHEMA,
        Capability.VISION,
        Capability.STREAMING,
    ]
    present = [c.value for c in order if c in profile.capabilities]
    return ", ".join(present) if present else "-"


def _key_style(key: str) -> str:
    return "green" if key.endswith("OK") else ("red" if "UNSET" in key else "dim")


_SPECIAL_KEYS = {("\t",): "tab", ("\r", "\n"): "enter", ("\x1b",): "esc"}
_ESC_SEQUENCES = {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "[Z": "shifttab"}
_WIN_CODES = {72: "up", 80: "down", 75: "left", 77: "right", 27: "esc"}


def _named_key(ch: str) -> str:
    for chars, name in _SPECIAL_KEYS.items():
        if ch in chars:
            return name
    if ch in ("\x03",):
        raise KeyboardInterrupt
    if ch in ("q", "Q"):
        return "quit"
    return ch


def _read_key() -> str:
    """Read one key press: 'up'/'down'/'left'/'right'/'tab'/'shifttab'/
    'enter'/'esc'/'quit', or a single printable character."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = ord(msvcrt.getwch())
            return _WIN_CODES.get(code, f"fn{code}")
        return _named_key(ch)

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return _named_key(ch)
        import select

        seq = ""
        for _ in range(2):
            if select.select([sys.stdin], [], [], 0.05)[0]:
                seq += sys.stdin.read(1)
            else:
                break
        return _ESC_SEQUENCES.get(seq, "esc")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


TAB_NAMES = ("Actions", "Profiles", "Routing", "Commands")

LAUNCHER_ITEMS = (
    "Run Hermetic Tests",
    "Run Live Tests",
    "Launch Web UI (backend + browser)",
    "Launch Backend Only",
    "Launch Frontend Only (browser)",
)


class BrowserState:
    def __init__(self, cfg: NorthStackConfig):
        self.cfg = cfg
        self.tab = 0
        self.cursor = [0, 0, 0, 0]
        self.expanded_profile: int | None = None
        self.launcher_last_result: str | None = None

    def rows(self, tab: int) -> int:
        return (
            len(LAUNCHER_ITEMS),
            len(self.cfg.profiles),
            len(self.cfg.routing),
            len(self.cfg.commands),
        )[tab]

    def move(self, delta: int) -> None:
        n = self.rows(self.tab)
        if n == 0:
            self.cursor[self.tab] = 0
            return
        self.cursor[self.tab] = (self.cursor[self.tab] + delta) % n

    def switch_tab(self, delta: int) -> None:
        self.tab = (self.tab + delta) % len(TAB_NAMES)
        self.cursor[self.tab] = min(self.cursor[self.tab], max(0, self.rows(self.tab) - 1))


def _header_table(cfg: NorthStackConfig, env_path: Path | None) -> Table:
    env_line = f"loaded {env_path}" if env_path else "no .env found"
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold cyan")
    t.add_column()
    t.add_row("northstack", cfg.name)
    t.add_row("budget", cfg.run.budget_summary())
    t.add_row("secrets", env_line)
    return t


def _profiles_tab(state: BrowserState) -> Panel:
    table = Table(border_style="cyan", show_lines=False, expand=True, show_header=True)
    table.add_column("name", style="bold", no_wrap=True)
    table.add_column("protocol", style="cyan", no_wrap=True)
    table.add_column("model", no_wrap=False)
    table.add_column("tier", justify="center", no_wrap=True)
    table.add_column("conc", justify="right", no_wrap=True)
    table.add_column("roles", no_wrap=False)
    table.add_column("api key", no_wrap=True)

    sel = state.cursor[1]
    for i, p in enumerate(state.cfg.profiles):
        marker = "▶ " if i == sel else "  "
        key = p.key_status()
        row_style = "on grey15" if i == sel else ""
        table.add_row(
            f"{marker}{p.name}",
            p.protocol.value,
            p.model,
            str(p.tier),
            str(p.max_concurrency),
            _roles_cell(p),
            Text(key, style=_key_style(key)),
            style=row_style,
        )
    title = "Profiles  (Enter = details, j/k = move, ←/→ = tab, q = quit)"
    body: Table | Group = table
    if state.expanded_profile is not None and 0 <= state.expanded_profile < len(state.cfg.profiles):
        body = Group(table, _profile_details(state.cfg.profiles[state.expanded_profile]))
    return Panel(body, title=title, border_style="cyan")


def _profile_details(p) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    t.add_row("model", p.model)
    t.add_row("protocol", p.protocol.value)
    t.add_row("base_url", p.base_url)
    t.add_row("api_key_env", p.key_status())
    t.add_row("roles", _roles_cell(p))
    t.add_row("capabilities", _caps_cell(p))
    t.add_row("max_concurrency", str(p.max_concurrency))
    t.add_row("requests_per_minute", str(p.requests_per_minute))
    t.add_row("context_window_tokens", f"{p.context_window_tokens:,}")
    t.add_row("max_output_tokens", f"{p.max_output_tokens:,}")
    t.add_row("input_price $/M", f"{p.input_price_per_million_usd}")
    t.add_row("output_price $/M", f"{p.output_price_per_million_usd}")
    return t


def _routing_tab(state: BrowserState) -> Panel:
    table = Table(border_style="cyan", expand=True)
    table.add_column("role", style="bold", no_wrap=True)
    table.add_column("profile fallback chain", style="green", no_wrap=False)
    sel = state.cursor[2]
    for i, entry in enumerate(state.cfg.routing):
        marker = "▶ " if i == sel else "  "
        chain = "  →  ".join(entry.profiles) if entry.profiles else "(empty)"
        table.add_row(f"{marker}{entry.role.value}", chain, style="on grey15" if i == sel else "")
    if not state.cfg.routing:
        return Panel(
            Text("No [northstack.routing] table -- legacy role-tag filtering."), title="Routing"
        )
    title = "Routing  (j/k = move, ←/→ = tab, q = quit)"
    return Panel(table, title=title, border_style="cyan")


def _commands_tab(state: BrowserState) -> Panel:
    if not state.cfg.commands:
        return Panel(Text("No command profiles."), title="Commands")
    table = Table(border_style="cyan", expand=True)
    table.add_column("name", style="bold", no_wrap=True)
    table.add_column("argv", no_wrap=False)
    table.add_column("timeout", justify="right", no_wrap=True)
    table.add_column("max bytes", justify="right", no_wrap=True)
    table.add_column("env allowlist", no_wrap=False)
    sel = state.cursor[3]
    for i, c in enumerate(state.cfg.commands):
        marker = "▶ " if i == sel else "  "
        argv = " ".join(c.argv)
        allow = ", ".join(c.env_allowlist) if c.env_allowlist else "-"
        table.add_row(
            f"{marker}{c.name}",
            argv,
            f"{c.timeout_seconds:g}s",
            str(c.max_output_bytes),
            allow,
            style="on grey15" if i == sel else "",
        )
    title = "Commands  (j/k = move, ←/→ = tab, q = quit)"
    return Panel(table, title=title, border_style="cyan")


def _launcher_tab(state: BrowserState) -> Panel:
    table = Table(border_style="cyan", expand=True, show_header=False)
    table.add_column("action", style="bold", no_wrap=True)
    sel = state.cursor[0]
    for i, item in enumerate(LAUNCHER_ITEMS):
        marker = "▶ " if i == sel else "  "
        table.add_row(f"{marker}{item}", style="on grey15" if i == sel else "")
    body: Table | Group = table
    if state.launcher_last_result:
        body = Group(
            table,
            Text(""),
            Text(f"Last: {state.launcher_last_result}", style="dim cyan"),
        )
    title = "Actions  (Enter = launch, j/k = move, ←/→ = tab, q = quit)"
    return Panel(body, title=title, border_style="cyan")


def _tab_bar(state: BrowserState) -> Align:
    parts = []
    for i, name in enumerate(TAB_NAMES):
        if i == state.tab:
            parts.append(Text(f" [{name}] ", style="bold white on cyan"))
        else:
            parts.append(Text(f"  {name}  ", style="grey50"))
        parts.append(Text(" "))
    bar = Text.assemble(*parts)
    return Align.center(bar)


def render_screen(state: BrowserState, env_path: Path | None) -> Group:
    tab_panels = [_launcher_tab, _profiles_tab, _routing_tab, _commands_tab]
    return Group(
        Panel(_header_table(state.cfg, env_path), title="northstack config", border_style="cyan"),
        _tab_bar(state),
        tab_panels[state.tab](state),
    )


def _try_web(mode: str, config_path: Path) -> str:
    try:
        return _launch_web(mode, config_path.resolve())
    except SystemExit:
        return "Web extras not installed. Run: uv sync --extra web"


def _execute_action(action_idx: int, cfg: NorthStackConfig, config_path: Path) -> str:
    """Run the selected launcher item and return a one-line summary.

    Called between ``Live`` blocks so output streams to the real terminal.
    Web-extras errors come back as hint strings instead of exiting the TUI.
    """
    cwd = config_path.resolve().parent
    arms = (
        lambda: _run_pytest(["uv", "run", "-q", "pytest", "-m", "not live", "-q"], cwd, {}),
        lambda: _run_pytest(
            ["uv", "run", "--extra", "web", "pytest", "-m", "live", "-q"],
            cwd,
            {
                "MC_LIVE_API": "1",
                "MC_LIVE_BASE_URL": "http://127.0.0.1:8317/v1",
                "MC_LIVE_KEY_ENV": "CLIPROXY_API_KEY",
                "MC_LIVE_MODEL": "bedrock/minimax.minimax-m2.5",
            },
        ),
        lambda: _try_web("both", config_path),
        lambda: _try_web("backend", config_path),
        lambda: _launch_web("frontend", config_path.resolve()),
    )
    return arms[action_idx]() if action_idx < len(arms) else f"unknown action {action_idx}"


def _run_pytest(cmd: list[str], cwd: Path, extra_env: dict[str, str]) -> str:
    """Run a pytest command (inherited env + extra_env), return a one-line summary."""
    env = {**os.environ, **extra_env}
    console.print(f"[dim]$ {' '.join(cmd)}  (cwd={cwd})[/dim]")
    try:
        result = subprocess.run(cmd, cwd=cwd, env=env, check=False)  # noqa: S603  (fixed argv list)
    except FileNotFoundError:
        return "'uv' not found on PATH -- is the project environment set up?"
    tail = _short_tail(result.returncode)
    return f"exit {result.returncode} ({tail})"


def _short_tail(returncode: int) -> str:
    if returncode == 0:
        return "passed"
    if returncode == 5:
        return "no tests collected / all deselected"
    return f"failed (rc={returncode})"


def _pause_return(cfg_subtitle: str) -> None:
    """On the real terminal: hold the streamed output visible until the operator
    presses a key, *then* the caller re-enters the Live alternate screen.

    Without this, ``Live`` recreates the alternate screen immediately and wipes
    the pytest/uvicorn output (and the test that just failed) before anyone can
    read it. Reads one keypress via the same ``_read_key`` primitive -- a real
    terminal is required (we only get here when ``_is_tty()`` was true).
    """
    console.print()
    console.print(f"[bold cyan]Done -- {cfg_subtitle}[/bold cyan]")
    console.print("[dim]Press any key to return to the TUI...[/dim]", end="")
    sys.stdout.flush()
    _read_key()
    console.print()


def _is_tty() -> bool:
    return sys.stdout.isatty() and sys.stdin.isatty()


_KEY_NAV = {
    "tab": ("switch_tab", 1),
    "right": ("switch_tab", 1),
    "shifttab": ("switch_tab", -1),
    "left": ("switch_tab", -1),
    "down": ("move", 1),
    "j": ("move", 1),
    "up": ("move", -1),
    "k": ("move", -1),
}


def run_interactive(
    cfg: NorthStackConfig, env_path: Path | None, config_path: Path | None = None
) -> None:
    state = BrowserState(cfg)
    config_path = config_path or _default_config_path()
    try:
        while True:
            pending_action: int | None = None
            with Live(console=console, screen=True, transient=True, refresh_per_second=30) as live:
                live.update(render_screen(state, env_path))
                while True:
                    key = _read_key()
                    if key in ("quit", "esc"):
                        return
                    if key in _KEY_NAV:
                        method, delta = _KEY_NAV[key]
                        getattr(state, method)(delta)
                    elif key == "enter" and state.tab == 1 and state.cfg.profiles:
                        cur = state.cursor[1]
                        state.expanded_profile = None if state.expanded_profile == cur else cur
                    elif key == "enter" and state.tab == 0:
                        pending_action = state.cursor[0]
                        break
                    live.update(render_screen(state, env_path))

            if pending_action is not None:
                state.launcher_last_result = _execute_action(pending_action, state.cfg, config_path)
                _pause_return(state.launcher_last_result)
    except KeyboardInterrupt:
        pass


def dump_static(cfg: NorthStackConfig, env_path: Path | None) -> None:
    console.print(
        Panel(_header_table(cfg, env_path), title="northstack config", border_style="cyan")
    )
    console.print()
    console.print(_launcher_tab_static())
    console.print()
    console.print(_profiles_tab_static(cfg))
    console.print()
    console.print(_routing_tab_static(cfg))
    console.print()
    console.print(_commands_tab_static(cfg))


def _profiles_tab_static(cfg: NorthStackConfig) -> Table:
    table = Table(title="Model profiles", border_style="cyan", show_lines=True)
    table.add_column("name", style="bold", no_wrap=True)
    table.add_column("protocol", style="cyan", no_wrap=True)
    table.add_column("model", no_wrap=False)
    table.add_column("tier", justify="center", no_wrap=True)
    table.add_column("conc", justify="right", no_wrap=True)
    table.add_column("roles", no_wrap=False)
    table.add_column("capabilities", no_wrap=False)
    table.add_column("api key", no_wrap=True)
    for p in cfg.profiles:
        key = p.key_status()
        table.add_row(
            p.name,
            p.protocol.value,
            p.model,
            str(p.tier),
            str(p.max_concurrency),
            _roles_cell(p),
            _caps_cell(p),
            Text(key, style=_key_style(key)),
        )
    return table


def _routing_tab_static(cfg: NorthStackConfig) -> Table | Text:
    if not cfg.routing:
        return Text("No [northstack.routing] table -- legacy role-tag filtering.")
    table = Table(title="Role routing", border_style="cyan")
    table.add_column("role", style="bold")
    table.add_column("profile fallback chain", style="green")
    for entry in cfg.routing:
        chain = "  →  ".join(entry.profiles) if entry.profiles else "(empty)"
        table.add_row(entry.role.value, chain)
    return table


def _commands_tab_static(cfg: NorthStackConfig) -> Table | Text:
    if not cfg.commands:
        return Text("No command profiles.")
    table = Table(title="Command profiles", border_style="cyan")
    table.add_column("name", style="bold")
    table.add_column("argv")
    table.add_column("timeout", justify="right")
    table.add_column("max bytes", justify="right")
    table.add_column("env allowlist")
    for c in cfg.commands:
        argv = " ".join(c.argv)
        allow = ", ".join(c.env_allowlist) if c.env_allowlist else "-"
        table.add_row(c.name, argv, f"{c.timeout_seconds:g}s", str(c.max_output_bytes), allow)
    return table


def _launcher_tab_static() -> Table:
    """Static Actions listing (non-interactive dump). Mirrors LAUNCHER_ITEMS."""
    table = Table(title="Actions (launcher)", border_style="cyan", show_header=False)
    table.add_column("action", style="bold")
    for item in LAUNCHER_ITEMS:
        table.add_row(item)
    return table


def _default_config_path() -> Path:
    return Path.cwd() / "northstack.toml"


@app.callback(invoke_without_command=True)
def browse(
    ctx: typer.Context,
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to TOML config (defaults to ./northstack.toml).",
    ),
) -> None:
    """Load + browse the company config (interactive TUI, or static dump if not a TTY)."""
    if ctx.invoked_subcommand is not None:
        return
    config_path = config or _default_config_path()
    env_path = load_secrets(config_path)
    try:
        cfg = NorthStackConfig.from_toml(config_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e
    except (tomllib.TOMLDecodeError, ValidationError, ValueError, KeyError, TypeError) as e:
        console.print(f"[red]Invalid config:[/red] {e}")
        raise typer.Exit(code=1) from e

    if _is_tty():
        run_interactive(cfg, env_path, config_path)
    else:
        dump_static(cfg, env_path)


web_app = typer.Typer(help="Launch the web control surface (localhost).")
app.add_typer(web_app, name="web")


def _import_web() -> tuple[ModuleType, Callable[[Path], Any]]:
    """Lazily import uvicorn + create_app, or exit with a clear install hint."""
    try:
        import uvicorn

        from northstack.interfaces.web.server import create_app
    except ImportError as e:  # pragma: no cover - environment-dependent
        console.print(
            f"[red]Web extras not installed:[/red] {e}\n"
            "Install them with:  [cyan]uv sync --extra web[/cyan]"
        )
        raise typer.Exit(code=1) from e
    return uvicorn, create_app


def _resolve_config(config: Path | None) -> Path:
    return (config or _default_config_path()).resolve()


def _launch_web(
    mode: str,
    config_path: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
) -> str:
    """Shared core of the ``web backend`` / ``web both`` commands and the TUI launcher.

    ``mode`` is one of ``"backend"`` (block on uvicorn.run), ``"both"`` (serve in
    a daemon thread + open the browser + spin until ctrl-C), or ``"frontend"``
    (just open the browser). Runs on the real terminal — caller must have
    dropped any ``Live`` alternate-screen context first. Returns a one-line
    summary string (suitable for the TUI's "Last:" line).
    """
    if mode == "frontend":
        import webbrowser

        url = f"http://{host}:{port}/"
        console.print(f"Opening [cyan]{url}[/cyan] in the default browser.")
        console.print(
            "  If nothing loads, start the backend first:  [cyan]northstack tui web backend[/cyan]"
        )
        webbrowser.open(url)
        return f"opened browser at {url}"

    uvicorn, create_app = _import_web()
    if not config_path.is_file():
        console.print(f"[red]Config not found:[/red] {config_path}")
        raise typer.Exit(code=1)

    app_obj = create_app(config_path)

    if mode == "backend":
        console.print(
            f"[cyan]northstack control surface[/cyan]  config={config_path}\n"
            f"  serving on http://{host}:{port}/  (API docs /api/docs, ctrl-C to stop)"
        )
        uvicorn.run(app_obj, host=host, port=port, reload=reload)
        return f"backend served at http://{host}:{port}/ (stopped)"

    import threading
    import time
    import webbrowser

    server = uvicorn.Server(uvicorn.Config(app_obj, host=host, port=port, log_level="warning"))
    _serve_error: list[BaseException | None] = [None]

    def _serve() -> None:
        try:
            server.run()
        except Exception as exc:  # noqa: BLE001
            _serve_error[0] = exc

    thread = threading.Thread(target=_serve, name="northstack-web", daemon=True)
    thread.start()

    for _ in range(40):
        if getattr(server, "started", False) or not thread.is_alive():
            break
        time.sleep(0.05)

    if not thread.is_alive() and _serve_error[0] is not None:
        console.print(
            f"[red]Server failed to start:[/red] {_serve_error[0]}\n"
            f"  Check if port {port} is already in use."
        )
        return f"server failed on port {port}: {_serve_error[0]}"

    console.print(
        f"[cyan]northstack control surface[/cyan]  http://{host}:{port}/\n"
        "  (API docs /api/docs, ctrl-C to stop)"
    )
    webbrowser.open(f"http://{host}:{port}/")
    try:
        while thread.is_alive():
            time.sleep(0.25)
    except KeyboardInterrupt:
        console.print("\n[dim]stopping...[/dim]")
        server.should_exit = True
        thread.join(timeout=5)
    return f"backend+browser http://{host}:{port}/ (stopped)"


@web_app.command("backend")
def web_backend(
    config: Path = typer.Option(
        None, "--config", "-c", help="TOML config (defaults to ./northstack.toml)."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only)."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes (dev)."),
) -> None:
    """Serve the control surface on 127.0.0.1 (no browser popup)."""
    _launch_web("backend", _resolve_config(config), host=host, port=port, reload=reload)


@web_app.command("frontend")
def web_frontend(
    host: str = typer.Option("127.0.0.1", "--host", help="Backend host."),
    port: int = typer.Option(8000, "--port", help="Backend port."),
) -> None:
    """Open the control surface in the browser (assumes backend is running)."""
    _launch_web("frontend", _resolve_config(None), host=host, port=port)


@web_app.command("both")
def web_both(
    config: Path = typer.Option(
        None, "--config", "-c", help="TOML config (defaults to ./northstack.toml)."
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only)."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
) -> None:
    """Launch backend + open the browser in one command (ctrl-C to stop)."""
    _launch_web("both", _resolve_config(config), host=host, port=port)


if __name__ == "__main__":
    app()
