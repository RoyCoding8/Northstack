"""Shared pytest config for the northstack suite.

Two test populations, deliberately separated:

  - **Default suite** (runs on every ``uv run pytest``): unit + FastAPI
    ``TestClient`` tests that exercise the real app end-to-end over the
    in-process ASGI transport -- but with NO live model provider on the wire.
    The web run tests use a config with no WORKER profile so the pipeline
    takes the ``DeterministicAnalysisRunner`` path and never opens a socket.

  - **Live-API tests** (marker ``live``): hit a REAL provider endpoint
    (the configured model gateway -- e.g. the local Mimo proxy).  These are
    OFF by default and gated on an environment toggle so the default suite
    stays hermetic and CI-safe.  Enable with::

        MC_LIVE_API=1 uv run pytest -m live

    (or run the whole suite including live with ``MC_LIVE_API=1``).  Any test
    marked ``live`` that runs without the toggle set is *skipped*, not failed.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``live`` marker so pytest -m 'not live' is valid."""
    config.addinivalue_line(
        "markers",
        "live: live provider/API integration test (gated on MC_LIVE_API env, off by default)",
    )
    config.addinivalue_line(
        "markers",
        "docker: real Docker daemon integration test (gated on NORTHSTACK_DOCKER_SMOKE)",
    )


def _live_enabled() -> bool:
    return os.environ.get("MC_LIVE_API", "0").lower() in ("1", "true", "yes", "on")


@pytest.fixture(autouse=True)
def _skip_live_unless_enabled(request: pytest.FixtureRequest) -> None:
    """Skip any ``live``-marked test unless MC_LIVE_API is set.

    Default-suite tests (no marker) always run.  This is the env-toggle the
    operator asked for: live API testings available on demand, never on by
    default.
    """
    if request.node.get_closest_marker("live") and not _live_enabled():
        pytest.skip("live API test -- set MC_LIVE_API=1 to run")
