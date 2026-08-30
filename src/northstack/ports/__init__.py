"""Structural contracts between layers.

Adapters satisfy these by shape, not by inheritance, so nothing below this
package imports it.  The application layer depends on these Protocols rather
than on concrete adapters.
"""

from northstack.ports.protocols import (
    ArtifactSink,
    Clock,
    EventSink,
    GatewayPort,
    Sleeper,
    WorkerPort,
    WorkspacePort,
)

__all__ = [
    "ArtifactSink",
    "Clock",
    "EventSink",
    "GatewayPort",
    "Sleeper",
    "WorkerPort",
    "WorkspacePort",
]
