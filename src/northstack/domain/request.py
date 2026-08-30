"""Entry point value: a request to run a project."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from northstack.domain.budget import Budget

GoalText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=65_536)]
WorkspaceRootText = Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
MAX_WAVES = 100


class ProjectRequest(BaseModel):
    """Entry point: a request to run a project."""

    model_config = ConfigDict(frozen=True)

    goal: GoalText = Field(description="What to achieve")
    workspace_root: WorkspaceRootText = Field(description="Absolute path to workspace root")
    constraints: list[str] = Field(default_factory=list)
    budget: Budget | None = Field(
        default=None, description="Override budget; None uses config defaults"
    )
    tool_policy: list[str] = Field(
        default_factory=list, description="Restrict allowed tools; empty = all allowed"
    )
    max_waves: int = Field(
        default=3, ge=1, le=MAX_WAVES, description="Max replanning waves before forced finish"
    )
