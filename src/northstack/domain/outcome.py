"""Outcomes, evidence, failure classification, and recovery vocabulary."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field

from northstack.domain.budget import BudgetUsage
from northstack.domain.contract import CriterionKind


class RunOutcome(str, enum.Enum):
    """Terminal outcome of a run."""

    VERIFIED = "verified"
    ABSTAINED = "abstained"
    FAILED = "failed"


class FailureType(str, enum.Enum):
    """Classification of failure modes for recovery decisions."""

    TRANSIENT = "transient"
    SAMPLING = "sampling"
    CAPABILITY = "capability"
    DECOMPOSITION = "decomposition"
    SPECIFICATION = "specification"
    INTEGRATION = "integration"
    SAFETY = "safety"
    BUDGET = "budget"


class RecoveryAction(str, enum.Enum):
    """Allowed recovery actions per failure type."""

    BACKOFF_RETRY = "backoff_retry"
    CHANGED_STRATEGY_RETRY = "changed_strategy_retry"
    REROUTE_ESCALATE = "reroute_escalate"
    SPLIT_REPLAN = "split_replan"
    CONTRACT_AMENDMENT = "contract_amendment"
    ABSTAIN = "abstain"
    TERMINATE = "terminate"
    SCOPE_REDUCTION = "scope_reduction"
    FAIL = "fail"


class ArtifactRef(BaseModel):
    """Content-addressed blob reference."""

    digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$", description="sha256:hex digest")
    media_type: str
    size_bytes: int = Field(ge=0)


class EvidenceRecord(BaseModel):
    """Verdict and evidence for one criterion check."""

    model_config = ConfigDict(frozen=True)

    criterion_index: int = Field(ge=0)
    kind: CriterionKind
    passed: bool
    evidence_artifact_digest: str = Field(default="")
    detail: str = Field(default="")


class HardGateFailure(BaseModel):
    """A single hard-gate criterion that failed verification.

    Carries the criterion index and a human-readable detail so the control
    surface can show *which* gate failed and why -- a flat ``list[str]`` would
    lose the index.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    detail: str = Field(default="")


class EvidenceManifest(BaseModel):
    """Complete evidence manifest for a run outcome.

    Lists verdicts per criterion, capability/tool audit, usage, and final
    outcome.  States are only: VERIFIED, ABSTAINED, FAILED.
    """

    model_config = ConfigDict(frozen=True)

    outcome: RunOutcome
    records: list[EvidenceRecord] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    tools_audit: list[str] = Field(
        default_factory=list, description="All tools that were available"
    )
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    hard_gate_failures: list[HardGateFailure] = Field(default_factory=list)
    soft_verdict: str = Field(
        default="", description="Aggregated soft verdict if hard gates passed"
    )
    material_disagreement: bool = Field(
        default=False, description="True if blinded reviewers disagreed materially"
    )


class AttemptSignature(BaseModel):
    """Exact signature over a recovery attempt.

    Rejects duplicate attempts: same contract version + cell + profile +
    strategy must not be retried.
    """

    model_config = ConfigDict(frozen=True)

    contract_version: int = Field(ge=1)
    cell_id: str = Field(min_length=1)
    profile_name: str = Field(min_length=1)
    strategy_id: str = Field(default="", description="Prompt/strategy identifier")
    tool_plan: str = Field(default="", description="Sorted tool list as a fingerprint")
    evidence_digest: str = Field(
        default="", description="Digest of prior evidence that triggered retry"
    )


class CalibrationRecord(BaseModel):
    """Stored calibration for a soft-rubric criterion.

    Tracks reviewer agreement history to set conservative thresholds.
    """

    model_config = ConfigDict(frozen=True)

    criterion_index: int = Field(ge=0)
    reviewer_agreement_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Historical agreement rate among blinded reviewers",
    )
    min_reviewers: int = Field(default=2, ge=1, description="Minimum reviewers required")
    agreement_threshold: float = Field(
        default=0.67,
        ge=0.0,
        le=1.0,
        description="Min agreement rate to accept; below -> ABSTAINED",
    )
    sample_count: int = Field(default=0, ge=0, description="Number of calibration samples")
