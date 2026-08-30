"""Tool registry: one source of tool identity.

The control plane has three places that each spell out the tool set
independently -- a hardcoded name list in ``build.py``, ``_WORKSPACE_TOOL_DEFS``
in the orchestrator, and the ``if/elif`` name chain in ``worker._execute_tool``.
They drift: ``web_fetch`` is *dispatchable* (the worker has an
``elif name == "web_fetch"`` arm) but is *never advertised* to the model -- it is
absent from ``_WORKSPACE_TOOL_DEFS`` and from the build.py name list, and the
orchestrator never passes it via ``additional_tools``. That asymmetry is the
bug this step exists to catch.

A single ``ToolRegistry`` built once in the composition root declares every
tool once -- name, description, JSON schema, ``mutating`` flag and behaviour
together -- and the orchestrator, worker and contract validator all read
that one declaration. This file pins the invariant the registry must hold:

  **every advertised tool is dispatchable, and every dispatchable tool is
  advertised.**

Today (RED) this fails naming ``web_fetch``: the worker can execute it but no
``ToolDefinition`` advertises it.
"""

from __future__ import annotations

from northstack.adapters.providers.wire import ToolDefinition
from northstack.application.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    """The registry the composition root builds.

    Mirrors ``build.build_company``: workspace tools, ``cmd_*`` entries generated
    from the configured commands, and ``web_fetch``. The registry is the single
    declaration site; this helper is the one place tests repeat the construction
    so the parity test runs against the same object production builds.
    """
    return ToolRegistry.with_defaults(command_profiles={})


class TestToolRegistryParity:
    """Every advertised tool is dispatchable; every dispatchable tool advertised."""

    def test_every_advertised_tool_is_dispatchable(self) -> None:
        """No tool the model is offered can name a behaviour the worker can't
        execute -- a model call to an undispatchable tool is a guaranteed
        ``Unknown tool`` error at runtime."""
        registry = _registry()
        advertised = {t.name for t in registry.advertised()}
        dispatchable = registry.dispatchable_names()
        undispatchable = sorted(advertised - dispatchable)
        assert not undispatchable, (
            "advertised tools that the worker cannot dispatch: "
            f"{undispatchable}; every advertised tool must be dispatchable"
        )

    def test_every_dispatchable_tool_is_advertised(self) -> None:
        """No tool the worker can execute is silently unreachable from the model
        -- a dispatchable-but-unadvertised tool is dead code the contract never
        offers. This is the arm that catches ``web_fetch`` today: the worker has an
        ``elif name == "web_fetch"`` branch but no ``ToolDefinition`` advertises
        it, so the model can never ask for it."""
        registry = _registry()
        advertised = {t.name for t in registry.advertised()}
        dispatchable = registry.dispatchable_names()
        unadvertised = sorted(dispatchable - advertised)
        assert not unadvertised, (
            "dispatchable tools that are never advertised to the model: "
            f"{unadvertised}; every dispatchable tool must be advertised "
            "(web_fetch must be reachable)"
        )


class TestToolRegistryDeclaration:
    """The registry is the one place a tool's identity lives."""

    def test_advertised_are_tool_definitions(self) -> None:
        """``advertised()`` returns the wire ``ToolDefinition`` objects the model
        gateway consumes -- not bare names -- so the schema travels with the
        declaration and cannot drift from the behaviour."""
        registry = _registry()
        for td in registry.advertised():
            assert isinstance(td, ToolDefinition)
            assert td.name
            assert td.parameters.get("type") == "object"

    def test_mutating_flag_is_declared_not_derived(self) -> None:
        """``mutating`` is a declared attribute of the tool, not a hardcoded name
        list. ``write``/``create``/``replace`` are mutating; ``read``/``list``/
        ``search``/``web_fetch``/``cmd_*`` are not. Mutation leases are driven off
        this flag, not the ``("write","create","replace","patch")`` tuple --
        ``"patch"`` is not a tool and must not appear."""
        registry = _registry()
        by_name = {t.name: t for t in registry.advertised()}
        assert "patch" not in by_name, "'patch' is not a tool and must not be advertised"

        mutating = {t.name for t in registry.iter_tools() if t.mutating}
        assert {"write", "create", "replace"} <= mutating
        # read-only / fetch / command tools are NOT mutating.
        for non_mutating in ("read", "list", "search", "web_fetch"):
            assert non_mutating not in mutating, f"{non_mutating} must not be mutating"
