"""Work contracts and their acceptance criteria."""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from northstack.domain.budget import Budget


class CriterionKind(str, enum.Enum):
    """Allowed kinds of acceptance criteria.

    Hard criteria: command, file_diff, tree_digest, schema, policy.  Soft: soft_rubric.
    """

    COMMAND = "command"
    FILE_DIFF = "file_diff"
    TREE_DIGEST = "tree_digest"
    SCHEMA = "schema"
    POLICY = "policy"
    SOFT_RUBRIC = "soft_rubric"


class _CriterionBase(BaseModel):
    """Common fields shared by every criterion variant.

    ``kind`` is the discriminator (a fixed ``Literal`` per subclass, so an
    unknown kind fails at parse time); ``description`` is free-form prose.
    Satisfaction is tracked in the event stream, never mutated in-place, so no
    ``satisfied`` field lives here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = Field(default="")


class CommandCriterion(_CriterionBase):
    """Hard gate: run a configured command and assert its exit code."""

    kind: Literal[CriterionKind.COMMAND] = CriterionKind.COMMAND
    command_name: str = Field(min_length=1)
    exit_code: int = Field(default=0)


class FileDiffCriterion(_CriterionBase):
    """Hard gate: assert a file exists / matches content."""

    kind: Literal[CriterionKind.FILE_DIFF] = CriterionKind.FILE_DIFF
    path: str = Field(min_length=1)
    must_exist: bool = Field(default=True)
    content_hash: str | None = Field(default=None)
    content_contains: str | None = Field(default=None)
    content_equals: str | None = Field(default=None)


class SchemaCriterion(_CriterionBase):
    """Hard gate: validate a runtime artifact against a JSON schema."""

    kind: Literal[CriterionKind.SCHEMA] = CriterionKind.SCHEMA
    artifact_digest: str = Field(min_length=1)
    json_schema: dict[str, Any] = Field(default_factory=dict)


class PolicyCriterion(_CriterionBase):
    """Hard gate: check a tool / capability policy."""

    kind: Literal[CriterionKind.POLICY] = CriterionKind.POLICY
    check: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)


class TreeDigestCriterion(_CriterionBase):
    """Hard gate: a directory's whole file tree matches one digest.

    Covers every file under ``path`` (byte content + relative path set), so
    edits, deletions AND newly added files all fail the gate -- unlike
    per-file pins, which cannot see a file that did not exist at compile
    time. Cache/scratch directories are excluded from the digest.
    """

    kind: Literal[CriterionKind.TREE_DIGEST] = CriterionKind.TREE_DIGEST
    path: str = Field(min_length=1)
    tree_hash: str = Field(min_length=1)


class SoftRubricCriterion(_CriterionBase):
    """Soft rubric: a blinding-gated qualitative check."""

    kind: Literal[CriterionKind.SOFT_RUBRIC] = CriterionKind.SOFT_RUBRIC
    prompt: str = Field(default="")


AcceptanceCriterion = Annotated[
    CommandCriterion
    | FileDiffCriterion
    | TreeDigestCriterion
    | SchemaCriterion
    | PolicyCriterion
    | SoftRubricCriterion,
    Field(discriminator="kind"),
]


class WorkContract(BaseModel):
    """Executable work contract for a cell.

    Immutable: once the contract is agreed, its terms cannot change.
    Satisfaction of criteria and budget consumption are tracked via events.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, description="Unique contract identifier")
    version: int = Field(default=1, ge=1, description="Contract version number")
    objective: str = Field(description="What this contract aims to achieve")
    scope: str = Field(default="", description="In-scope work boundaries")
    deliverables: list[str] = Field(default_factory=list, description="Named deliverables expected")
    constraints: list[str] = Field(
        default_factory=list, description="Hard constraints (time, resource, policy)"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Assumptions that may be invalidated"
    )
    forbidden_outcomes: list[str] = Field(
        default_factory=list, description="Outcomes that must NOT occur"
    )
    allowed_tools: list[str] = Field(
        default_factory=list, description="Tool IDs the worker may use"
    )
    workspace_scope: str = Field(
        default="", description="Filesystem / repo scope for this contract"
    )
    budget: Budget
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    unresolved_ambiguity: list[str] = Field(
        default_factory=list, description="Open questions blocking confidence"
    )
    abstention_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Min confidence to proceed; below this the worker abstains",
    )
