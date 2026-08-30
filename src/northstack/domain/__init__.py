"""Pure domain values and invariants. Zero I/O, zero framework coupling.

This package is the bottom layer: nothing here may import from ``events``,
``adapters``, ``application`` or ``interfaces``.
"""

from northstack.domain.budget import Budget, BudgetUsage, RemainingBudget, Spend
from northstack.domain.contract import (
    AcceptanceCriterion,
    CommandCriterion,
    CriterionKind,
    FileDiffCriterion,
    PolicyCriterion,
    SchemaCriterion,
    SoftRubricCriterion,
    TreeDigestCriterion,
    WorkContract,
)
from northstack.domain.graph import CellMode, CellStatus, GraphCell, GraphEdge, GraphVersion
from northstack.domain.outcome import (
    ArtifactRef,
    AttemptSignature,
    CalibrationRecord,
    EvidenceManifest,
    EvidenceRecord,
    FailureType,
    HardGateFailure,
    RecoveryAction,
    RunOutcome,
)
from northstack.domain.request import ProjectRequest
from northstack.domain.run_state import RunState
from northstack.domain.status import RunStatus

__all__ = [
    "AcceptanceCriterion",
    "ArtifactRef",
    "AttemptSignature",
    "Budget",
    "BudgetUsage",
    "CalibrationRecord",
    "CellMode",
    "CellStatus",
    "CommandCriterion",
    "CriterionKind",
    "EvidenceManifest",
    "EvidenceRecord",
    "FailureType",
    "FileDiffCriterion",
    "GraphCell",
    "GraphEdge",
    "GraphVersion",
    "HardGateFailure",
    "PolicyCriterion",
    "ProjectRequest",
    "RecoveryAction",
    "RemainingBudget",
    "RunOutcome",
    "RunState",
    "RunStatus",
    "SchemaCriterion",
    "SoftRubricCriterion",
    "Spend",
    "TreeDigestCriterion",
    "WorkContract",
]
