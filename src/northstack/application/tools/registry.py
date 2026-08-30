"""Single source of tool identity.

A :class:`Tool` carries the name, description, JSON-schema parameters,
``mutating`` flag and behaviour together; the :class:`ToolRegistry` is built
once in the composition root (:func:`ToolRegistry.with_defaults`). The
orchestrator asks the registry for the ``ToolDefinition`` list to advertise;
the worker asks it what is dispatchable and how to execute; the contract
validator asks it for the legal name set. Adding a tool means editing exactly
this file.

A single declaration keeps the advertised set and the dispatchable set from
drifting -- a tool the worker can execute (e.g. ``web_fetch``) is always also
offered to the model, and vice versa.

Design notes:
  - ``advertised()`` returns the wire ``ToolDefinition`` objects the model
    gateway consumes, so the parameter schema travels with the declaration and
    cannot drift from the behaviour.
  - ``dispatchable_names()`` is the set of names the worker can execute. With
    every tool registered (including ``web_fetch``) it equals the advertised
    set -- the parity invariant pinned by ``tests/test_tool_registry.py``.
  - The ``mutating`` flag is *declared* here, not derived from a hardcoded name
    list; mutation leases are driven off it. The old
    ``("write", "create", "replace", "patch")`` tuple (which named ``"patch"``
    -- not a tool) disappears.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from northstack.adapters.providers.wire import ToolDefinition
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace, ToolResult
from northstack.adapters.workspace.webfetch import WebReader


def _schema(*required: str, optional: str = "") -> dict[str, Any]:
    """A minimal object JSON-schema for a tool's parameters.

    Mirrors the orchestrator's former ``_schema`` helper so the advertised
    schemas are byte-for-byte what they were before the extraction: an object
    with string properties, the named required fields, and
    ``additionalProperties: False``.
    """
    props: dict[str, Any] = {f: {"type": "string"} for f in required}
    if optional:
        props[optional] = {"type": "string"}
    return {
        "type": "object",
        "properties": props,
        "required": list(required),
        "additionalProperties": False,
    }


_READ_PARAMS = _schema("path")
_WRITE_PARAMS = _schema("path", "content")
_REPLACE_PARAMS = _schema("path", "old", "new")
_SEARCH_PARAMS = _schema("pattern", optional="path")
_WEB_FETCH_PARAMS = _schema("url")
_CMD_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


class Tool(Protocol):
    """One tool the model may call: identity + behaviour together.

    ``execute`` is the worker's dispatch arm for this tool. The registry holds
    concrete ``Tool`` records; ``advertised()`` projects them to the wire
    ``ToolDefinition`` the gateway consumes. ``execute`` returns the workspace's
    typed :class:`ToolResult` so the worker can render it into a
    ``ToolResultMessage`` (success text, truncation marker, or error) without
    re-implementing any tool's behaviour.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    mutating: bool

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        """Run the tool against ``ctx`` with ``args``; return its typed result."""
        ...


class ToolContext(Protocol):
    """The collaborators a tool's ``execute`` reaches into.

    A narrow view of the worker's state so tools do not see the whole
    ``NativeWorker``. ``workspace`` is the mediated filesystem, ``web_reader``
    is the SSRF-protected fetch (may be ``None``), ``command_profiles`` maps
    ``cmd_*`` names to their profiles, and ``lease`` is the active mutation
    lease (``None`` when no lease is held).
    """

    workspace: RestrictedWorkspace
    web_reader: WebReader | None
    command_profiles: dict[str, CommandProfile]
    lease: str | None


class _BaseTool:
    """Shared storage for a tool's declared identity."""

    __slots__ = ("description", "mutating", "name", "parameters")

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        mutating: bool,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.mutating = mutating


class _ReadTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="read",
            description="Read a file at a workspace-relative path.",
            parameters=_READ_PARAMS,
            mutating=False,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ctx.workspace.read(args.get("path", "."))


class _ListTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="list",
            description="List entries under a workspace-relative directory path.",
            parameters=_READ_PARAMS,
            mutating=False,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ctx.workspace.list(args.get("path", "."))


class _SearchTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="search",
            description="Search file names for a pattern under a directory.",
            parameters=_SEARCH_PARAMS,
            mutating=False,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ctx.workspace.search(args.get("pattern", ""), args.get("path", "."))


class _WriteTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="write",
            description="Write bytes to an existing file at a workspace-relative path.",
            parameters=_WRITE_PARAMS,
            mutating=True,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        content = args.get("content", "").encode("utf-8")
        return ctx.workspace.write(args.get("path", ""), content, lease=ctx.lease)


class _CreateTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="create",
            description="Atomically create a new file at a workspace-relative path.",
            parameters=_WRITE_PARAMS,
            mutating=True,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        content = args.get("content", "").encode("utf-8")
        return ctx.workspace.create(args.get("path", ""), content, lease=ctx.lease)


class _ReplaceTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="replace",
            description="Replace all occurrences of ``old`` with ``new`` in a file.",
            parameters=_REPLACE_PARAMS,
            mutating=True,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ctx.workspace.replace(
            args.get("path", ""),
            args.get("old", ""),
            args.get("new", ""),
            lease=ctx.lease,
        )


class _WebFetchTool(_BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="web_fetch",
            description="Fetch a public URL with SSRF protection; returns the body text.",
            parameters=_WEB_FETCH_PARAMS,
            mutating=False,
        )

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.web_reader is None:
            return ToolResult(ok=False, operation="web_fetch", error="Web fetch not available")
        return ctx.web_reader.fetch(args.get("url", ""))


class _CommandTool(_BaseTool):
    """A ``cmd_<name>`` tool generated from a configured :class:`CommandProfile`.

    The parameter schema is an empty (but explicit) object with ``required: []``
    -- some upstreams reject a bare empty-properties object as
    ``invalid_request_error``, so ``required`` is an explicit empty list. The
    profile is resolved at ``execute`` time from the registry's name.
    """

    def __init__(self, cmd_name: str, profile: CommandProfile) -> None:
        argv_head = " ".join(profile.argv[:2])
        super().__init__(
            name=f"cmd_{cmd_name}",
            description=f"Execute configured command: {cmd_name} ({argv_head})",
            parameters=_CMD_PARAMS,
            mutating=True,
        )
        self._cmd_name = cmd_name

    async def execute(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        profile = ctx.command_profiles.get(self._cmd_name)
        if profile is None:
            return ToolResult(
                ok=False, operation="cmd", error=f"Unknown command profile: {self._cmd_name}"
            )
        return ctx.workspace.execute_command(profile)


class ToolRegistry:
    """The single declaration of every tool the control plane knows.

    Built once in the composition root (``build.build_company`` via
    :meth:`with_defaults`). Holds ordered ``Tool`` records; projects them to
    wire ``ToolDefinition`` objects for advertisement and answers
    dispatchability. The orchestrator, worker and contract validator all read
    this object and never re-spell a tool name.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: list[Tool] = list(tools)
        self._by_name: dict[str, Tool] = {t.name: t for t in self._tools}

    @classmethod
    def with_defaults(
        cls,
        *,
        command_profiles: dict[str, CommandProfile],
    ) -> ToolRegistry:
        """The registry the composition root builds.

        Mirrors the former ``build.py`` name list + the orchestrator's
        ``_WORKSPACE_TOOL_DEFS`` + the worker's dispatch arms, now unified:
        the six workspace tools, one ``cmd_<name>`` tool per configured
        command profile, and ``web_fetch``. ``web_fetch`` is advertised here
        for the first time, closing the dispatch/advertise gap.
        """
        tools: list[Tool] = [
            _ReadTool(),
            _WriteTool(),
            _CreateTool(),
            _ReplaceTool(),
            _ListTool(),
            _SearchTool(),
            _WebFetchTool(),
        ]
        for cmd_name, profile in command_profiles.items():
            tools.append(_CommandTool(cmd_name, profile))
        return cls(tools)

    def advertised(self) -> list[ToolDefinition]:
        """The wire ``ToolDefinition`` objects to advertise to the model.

        Order is stable (insertion order) so the advertised set is
        deterministic across runs.
        """
        return [
            ToolDefinition(name=t.name, description=t.description, parameters=t.parameters)
            for t in self._tools
        ]

    def dispatchable_names(self) -> set[str]:
        """Names the worker can execute. Equals the advertised set once every
        tool (including ``web_fetch``) is registered -- the parity invariant."""
        return set(self._by_name)

    def names(self) -> list[str]:
        """The ordered list of legal tool names (for the contract validator)."""
        return [t.name for t in self._tools]

    def iter_tools(self) -> list[Tool]:
        """The declared ``Tool`` records (identity + behaviour)."""
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        """The tool for ``name``, or ``None`` if it is not registered."""
        return self._by_name.get(name)

    async def execute(self, name: str, ctx: ToolContext, args: dict[str, Any]) -> ToolResult | None:
        """Execute the named tool against ``ctx``. Returns ``None`` if ``name``
        is not a registered (dispatchable) tool; otherwise the typed
        :class:`ToolResult` for the worker to render into a message."""
        tool = self._by_name.get(name)
        if tool is None:
            return None
        return await tool.execute(ctx, args)


BuildRegistry = Callable[[dict[str, CommandProfile]], ToolRegistry]
