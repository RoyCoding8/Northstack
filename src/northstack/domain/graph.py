"""Rolling-wave execution graph: cells, edges, and versioned graphs.

``CellStatus`` is the single status vocabulary for a cell.  The legacy ``Cell``
model that reused ``RunStatus`` is gone -- two competing enums for one concept
forced every projection handler to update both in lockstep.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from northstack.domain.budget import Budget
from northstack.domain.contract import WorkContract


class CellMode(str, enum.Enum):
    """Read/write mode for a cell."""

    READ_ONLY = "read_only"
    MUTATING = "mutating"


class CellStatus(str, enum.Enum):
    """Execution status of a cell in the graph."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class GraphEdge(BaseModel):
    """Dependency edge between cells in the rolling-wave graph."""

    from_id: str
    to_id: str
    kind: str = Field(default="blocks", description="Edge type: blocks, informs, etc.")


class GraphCell(BaseModel):
    """A unit of work in the rolling-wave execution graph.

    Carries capability requirements, read/write mode, input/output schemas,
    profile requirements, dependencies, and acceptance links.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(default="")
    wave: int = Field(default=0, ge=0, description="Wave number for rolling-wave scheduling")
    required_capabilities: list[str] = Field(default_factory=list)
    mode: CellMode = Field(default=CellMode.READ_ONLY)
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON-Schema for expected input"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON-Schema for produced output"
    )
    required_profile_roles: list[str] = Field(
        default_factory=list, description="Roles a profile must have"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Cell IDs that must complete first"
    )
    acceptance_criterion_indices: list[int] = Field(
        default_factory=list, description="Linked contract criterion indices"
    )
    contract: WorkContract = Field(
        default_factory=lambda: WorkContract(
            id="placeholder", objective="", budget=Budget.default()
        )
    )
    status: CellStatus = Field(default=CellStatus.PENDING)


class GraphVersion(BaseModel):
    """Versioned execution graph with milestones and current horizon."""

    model_config = ConfigDict(frozen=True)

    version: int = Field(default=1, ge=1)
    cells: list[GraphCell] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    milestones: list[str] = Field(
        default_factory=list, description="Cell IDs that are stable boundaries"
    )
    current_horizon: int = Field(default=0, ge=0, description="Max wave currently planned")

    def with_cell_status(self, cell_id: str, status: CellStatus) -> GraphVersion:
        """Return a copy with one cell's status replaced; cells stay immutable."""
        return self.model_copy(
            update={
                "cells": [
                    c.model_copy(update={"status": status}) if c.id == cell_id else c
                    for c in self.cells
                ]
            }
        )
