@echo off
REM northstack TUI launcher (Windows).
REM Usage: run.bat              -> browses .\northstack.toml with .\.env loaded
REM        run.bat --config other.toml
REM Loads secrets from .env via the TUI itself; no inline env prefix needed.
REM The Actions tab runs the hermetic/live test suites and launches the web UI.
REM For the headless surface (run, benchmark --live --ablate, calibrate,
REM inspect, replay, config validate, ledger verify):
REM   uv run northstack --help
uv run python -m northstack.interfaces.tui %*
