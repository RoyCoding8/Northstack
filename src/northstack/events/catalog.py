"""The typed event catalog: one model per ``EventKind``.

Every payload is frozen and ``extra="forbid"``.  Together they form a
discriminated union on ``kind``, so:

  - a producer cannot emit a field the catalog does not declare;
  - a reader cannot invent a field the producer never wrote;
  - the projection can ``match`` on type and close with ``assert_never``, which
    turns "added an event kind but forgot to project it" into a *type* error.

Required data carries no default.  Optional data may, but only where absence is
genuinely meaningful -- never to paper over a field a producer forgot.
"""

from __future__ import annotations

import enum
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from northstack.domain.budget import Budget, BudgetUsage, Spend
from northstack.domain.contract import AcceptanceCriterion, CriterionKind
from northstack.domain.graph import CellStatus, GraphCell, GraphEdge
from northstack.domain.outcome import (
    ArtifactRef,
    EvidenceRecord,
    FailureType,
    HardGateFailure,
    RecoveryAction,
    RunOutcome,
)
from northstack.domain.status import RunStatus

CURRENT_SCHEMA_VERSION = 1


class EventKind(str, enum.Enum):
    RUN_CREATED = "run_created"
    STATUS_CHANGED = "status_changed"
    CELL_CREATED = "cell_created"
    CELL_ADVANCED = "cell_advanced"
    CLAIM_RECORDED = "claim_recorded"
    ARTIFACT_STORED = "artifact_stored"
    BUDGET_UPDATED = "budget_updated"

    REQUEST_ACCEPTED = "request_accepted"
    WORKSPACE_SNAPSHOT = "workspace_snapshot"
    ANALYSIS_REQUESTED = "analysis_requested"
    ANALYSIS_COMPLETED = "analysis_completed"
    CONTRACT_PROPOSED = "contract_proposed"
    CONTRACT_VALIDATED = "contract_validated"
    CONTRACT_AMENDED = "contract_amended"
    GRAPH_PROPOSED = "graph_proposed"
    GRAPH_ACCEPTED = "graph_accepted"
    ROUTE_SELECTED = "route_selected"
    CELL_STARTED = "cell_started"
    CELL_PROGRESS = "cell_progress"
    CELL_COMPLETED = "cell_completed"
    CELL_FAILED = "cell_failed"
    EVIDENCE_RECORDED = "evidence_recorded"
    VERIFICATION_CHECK = "verification_check"
    RECOVERY_TRANSITION = "recovery_transition"
    OUTCOME_EMITTED = "outcome_emitted"
    STALL_DETECTED = "stall_detected"


class PayloadBase(BaseModel):
    """Common shape for every event payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=CURRENT_SCHEMA_VERSION, ge=1)


class RunCreated(PayloadBase):
    kind: Literal[EventKind.RUN_CREATED] = EventKind.RUN_CREATED


class StatusChanged(PayloadBase):
    kind: Literal[EventKind.STATUS_CHANGED] = EventKind.STATUS_CHANGED
    status: RunStatus


class RequestAccepted(PayloadBase):
    kind: Literal[EventKind.REQUEST_ACCEPTED] = EventKind.REQUEST_ACCEPTED
    goal: str
    workspace_root: str
    budget: Budget | None = None


class WorkspaceSnapshot(PayloadBase):
    kind: Literal[EventKind.WORKSPACE_SNAPSHOT] = EventKind.WORKSPACE_SNAPSHOT
    root: str
    file_count: int = Field(ge=0)
    digest: str = ""


class AnalysisRequested(PayloadBase):
    kind: Literal[EventKind.ANALYSIS_REQUESTED] = EventKind.ANALYSIS_REQUESTED
    profile: str
    analysis: dict[str, Any] = Field(default_factory=dict)


class AnalysisCompleted(PayloadBase):
    kind: Literal[EventKind.ANALYSIS_COMPLETED] = EventKind.ANALYSIS_COMPLETED
    profile: str
    analysis: dict[str, Any] = Field(default_factory=dict)


class ContractProposed(PayloadBase):
    kind: Literal[EventKind.CONTRACT_PROPOSED] = EventKind.CONTRACT_PROPOSED
    id: str
    version: int = Field(ge=1)
    objective: str
    scope: str = ""
    deliverables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    workspace_scope: str = ""
    budget: Budget
    acceptance_criteria_count: int = Field(ge=0)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    unresolved_ambiguity: list[str] = Field(default_factory=list)


class ContractValidated(PayloadBase):
    kind: Literal[EventKind.CONTRACT_VALIDATED] = EventKind.CONTRACT_VALIDATED
    id: str
    version: int = Field(ge=1)


class ContractAmended(PayloadBase):
    kind: Literal[EventKind.CONTRACT_AMENDED] = EventKind.CONTRACT_AMENDED
    id: str
    version: int = Field(ge=1)
    reason: str = ""


class GraphProposed(PayloadBase):
    kind: Literal[EventKind.GRAPH_PROPOSED] = EventKind.GRAPH_PROPOSED
    version: int = Field(ge=1)
    errors: list[str] = Field(default_factory=list)
    rejected: bool = False


class GraphAccepted(PayloadBase):
    kind: Literal[EventKind.GRAPH_ACCEPTED] = EventKind.GRAPH_ACCEPTED
    version: int = Field(ge=1)
    cells: list[GraphCell] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    milestones: list[str] = Field(default_factory=list)


class CellCreated(PayloadBase):
    kind: Literal[EventKind.CELL_CREATED] = EventKind.CELL_CREATED
    cell: GraphCell


class CellAdvanced(PayloadBase):
    kind: Literal[EventKind.CELL_ADVANCED] = EventKind.CELL_ADVANCED
    cell_id: str
    status: CellStatus


class RouteSelected(PayloadBase):
    kind: Literal[EventKind.ROUTE_SELECTED] = EventKind.ROUTE_SELECTED
    cell_id: str
    profile_name: str
    reason: str = ""


class CellStarted(PayloadBase):
    kind: Literal[EventKind.CELL_STARTED] = EventKind.CELL_STARTED
    cell_id: str
    profile_name: str = ""


class CellProgress(PayloadBase):
    """One moment inside a cell's model/tool loop.

    Between ``cell_started`` and its terminal event a cell was opaque: a
    runaway turn loop, a context compaction and a slow tool were equally
    invisible.  One kind rather than one per moment keeps the wire and the
    projection small -- the worker names the moment in ``step``, and
    ``detail`` carries only scalars (never message content, which would put
    model output and file bytes into the ledger).
    """

    kind: Literal[EventKind.CELL_PROGRESS] = EventKind.CELL_PROGRESS
    cell_id: str
    step: str = Field(max_length=64)
    attempt: int = Field(default=0, ge=0)
    turn: int = Field(default=0, ge=0)
    detail: dict[str, str | int | float | bool] = Field(default_factory=dict)


class CellCompleted(PayloadBase):
    kind: Literal[EventKind.CELL_COMPLETED] = EventKind.CELL_COMPLETED
    cell_id: str
    output_artifact: ArtifactRef
    usage: Spend


class CellFailed(PayloadBase):
    """One shape for every cell failure.

    Previously emitted two ways -- ``{cell_id, reason}`` when routing abstained
    and ``{cell_id, error, error_kind}`` when the worker failed -- so every
    reader had to guess which keys were present.
    """

    kind: Literal[EventKind.CELL_FAILED] = EventKind.CELL_FAILED
    cell_id: str
    error: str
    error_kind: str


class ClaimRecorded(PayloadBase):
    kind: Literal[EventKind.CLAIM_RECORDED] = EventKind.CLAIM_RECORDED
    cell_id: str
    claim: str
    evidence_digest: str = ""


class ArtifactStored(PayloadBase):
    kind: Literal[EventKind.ARTIFACT_STORED] = EventKind.ARTIFACT_STORED
    artifact: ArtifactRef


class BudgetUpdated(PayloadBase):
    kind: Literal[EventKind.BUDGET_UPDATED] = EventKind.BUDGET_UPDATED
    usage: BudgetUsage
    exhausted: bool = False


class VerificationCheck(PayloadBase):
    kind: Literal[EventKind.VERIFICATION_CHECK] = EventKind.VERIFICATION_CHECK
    criterion_index: int = Field(ge=0)
    criterion_kind: CriterionKind
    passed: bool
    detail: str = ""
    evidence_artifact_digest: str = ""


class EvidenceRecorded(PayloadBase):
    kind: Literal[EventKind.EVIDENCE_RECORDED] = EventKind.EVIDENCE_RECORDED
    outcome: RunOutcome
    records: list[EvidenceRecord] = Field(default_factory=list)
    hard_gate_failures: list[HardGateFailure] = Field(default_factory=list)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    tools_used: list[str] = Field(default_factory=list)
    material_disagreement: bool = False


class RecoveryTransition(PayloadBase):
    kind: Literal[EventKind.RECOVERY_TRANSITION] = EventKind.RECOVERY_TRANSITION
    cell_id: str
    failure_type: FailureType
    action: RecoveryAction
    attempt_number: int = Field(ge=0)
    error_detail: str = Field(default="", max_length=2000)


class OutcomeEmitted(PayloadBase):
    kind: Literal[EventKind.OUTCOME_EMITTED] = EventKind.OUTCOME_EMITTED
    outcome: RunOutcome
    reason: str = ""


class StallDetected(PayloadBase):
    """A run that was alive but not progressing inside the configured window.

    A cell that hangs (no progress, no terminal event) would leave the run
    pinned forever; the stall detector emits this when the elapsed time since
    the last per-cell heartbeat exceeds the configured window, and the run
    abstains -- a stuck run is an unknown-outcome run, so abstention (not a
    guessed failure) is the honest terminal. ``cell_id`` is empty when the
    stall is run-wide rather than pinned to one cell.
    """

    kind: Literal[EventKind.STALL_DETECTED] = EventKind.STALL_DETECTED
    cell_id: str = ""


EventPayload = Annotated[
    RunCreated
    | StatusChanged
    | RequestAccepted
    | WorkspaceSnapshot
    | AnalysisRequested
    | AnalysisCompleted
    | ContractProposed
    | ContractValidated
    | ContractAmended
    | GraphProposed
    | GraphAccepted
    | CellCreated
    | CellAdvanced
    | RouteSelected
    | CellStarted
    | CellProgress
    | CellCompleted
    | CellFailed
    | ClaimRecorded
    | ArtifactStored
    | BudgetUpdated
    | VerificationCheck
    | EvidenceRecorded
    | RecoveryTransition
    | OutcomeEmitted
    | StallDetected,
    Field(discriminator="kind"),
]

PAYLOAD_MODELS: tuple[type[PayloadBase], ...] = get_args(get_args(EventPayload)[0])

PAYLOAD_BY_KIND: dict[EventKind, type[PayloadBase]] = {
    model.model_fields["kind"].default: model for model in PAYLOAD_MODELS
}

if set(PAYLOAD_BY_KIND) != set(EventKind):
    missing = sorted(k.value for k in set(EventKind) - set(PAYLOAD_BY_KIND))
    raise RuntimeError(f"event kinds without a payload model: {missing}")

PAYLOAD_ADAPTER: TypeAdapter[EventPayload] = TypeAdapter(EventPayload)


def parse_payload(data: Any) -> EventPayload:
    """Validate a raw payload mapping into its catalog model.

    Raises ``pydantic.ValidationError``; callers that know the ledger row wrap
    it in ``LedgerCorruption`` so the failure names the offending seq.
    """
    return PAYLOAD_ADAPTER.validate_python(data)
