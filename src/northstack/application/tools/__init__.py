"""Single source of tool identity.

Each tool is declared once -- name, description, JSON schema, ``mutating``
flag and behaviour together -- in :class:`ToolRegistry`. The orchestrator,
worker and contract validator all read that one declaration. See
:mod:`northstack.application.tools.registry`.
"""
