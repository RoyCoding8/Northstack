"""Soft-rubric review: blinded reviewers with calibration-gated aggregation.

The review seam is async and evidence-aware: a reviewer receives the *content*
of the evidence it must judge (resolved from the artifact store by the caller),
the criterion description, and the contract objective -- and nothing else.
Blinding is structural rather than procedural: executor identity, profile
names, tool trails, and conversation history simply do not exist in the
reviewer's input, so they cannot leak into a verdict.

Every reviewer failure is a fail-closed ``ReviewVerdict`` (not passed), which
the aggregation treats as disagreement -- preserving the release law: a run
with soft rubrics abstains rather than verifying on broken evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from northstack.domain.contract import CriterionKind, WorkContract
from northstack.domain.outcome import CalibrationRecord

MIN_BLINDED_REVIEWERS = 2


class ReviewVerdict(BaseModel):
    """One blinded reviewer's typed verdict on one soft criterion.

    ``passed`` is the reviewer's judgment; ``confidence`` in [0, 1] accompanies
    it for calibration analysis; ``rationale_digest`` is a short digest of the
    reviewer's reasoning (never free text, so the ledger stays bounded and the
    reviewer's chain-of-thought is not persisted verbatim).
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale_digest: str = Field(default="", max_length=128)


class BlindedReviewer(Protocol):
    """Protocol for a blinded reviewer cell.

    The reviewer sees: the criterion's kind and description, the contract's
    objective, and the resolved evidence content for that criterion. It does
    not see -- and cannot be shown -- the executor, the routing decision, the
    tool trail, or any conversation: those fields do not exist in this
    signature. Failures must surface as a fail-closed verdict, not an
    exception, so one broken reviewer degrades to disagreement (abstention)
    instead of crashing verification.
    """

    async def review(
        self,
        *,
        criterion_index: int,
        criterion_kind: str,
        description: str,
        objective: str,
        evidence_content: str,
    ) -> ReviewVerdict: ...


class DeterministicReviewer:
    """Deterministic reviewer that returns a fixed verdict per criterion.

    Test seam only: production wires :class:`ModelBackedReviewer` instances.
    """

    def __init__(self, verdicts: dict[int, bool] | None = None) -> None:
        self._verdicts = verdicts or {}

    async def review(
        self,
        *,
        criterion_index: int,
        criterion_kind: str,
        description: str,
        objective: str,
        evidence_content: str,
    ) -> ReviewVerdict:
        return ReviewVerdict(passed=self._verdicts.get(criterion_index, True))


class SoftRubricChecker:
    """Runs soft-rubric criteria through blinded reviewers.

    Requires at least two blinded reviewers. With no calibration, or with
    material disagreement, the outcome is ABSTAINED -- never VERIFIED.
    """

    def __init__(
        self,
        reviewers: list[BlindedReviewer] | None = None,
        calibration_records: list[CalibrationRecord] | None = None,
    ) -> None:
        self._reviewers = reviewers or []
        self._calibration = {r.criterion_index: r for r in (calibration_records or [])}

    async def check(
        self,
        contract: WorkContract,
        *,
        evidence_contents: Mapping[int, str] | None = None,
    ) -> tuple[dict[int, bool], bool]:
        """Aggregate blinded verdicts for every soft criterion.

        ``evidence_contents`` maps criterion index to resolved artifact
        content; reviewers judge content, not digests. Indices absent from the
        map pass the empty string -- a reviewer that cannot see evidence fails
        closed through its own judgment (and uncalibrated/missing-evidence
        disagreement keeps the run honest).
        """
        contents = evidence_contents or {}
        soft_indices = [
            i
            for i, c in enumerate(contract.acceptance_criteria)
            if c.kind == CriterionKind.SOFT_RUBRIC.value
        ]

        if len(self._reviewers) < MIN_BLINDED_REVIEWERS:
            return dict.fromkeys(soft_indices, False), bool(soft_indices)

        verdicts: dict[int, bool] = {}
        material_disagreement = False
        for i in soft_indices:
            criterion = contract.acceptance_criteria[i]
            reviewer_verdicts = [
                await reviewer.review(
                    criterion_index=i,
                    criterion_kind=criterion.kind,
                    description=criterion.description,
                    objective=contract.objective,
                    evidence_content=contents.get(i, ""),
                )
                for reviewer in self._reviewers
            ]
            agreement_rate = sum(v.passed for v in reviewer_verdicts) / len(reviewer_verdicts)

            cal = self._calibration.get(i)
            accepted = (
                cal is not None
                and cal.sample_count > 0
                and len(self._reviewers) >= cal.min_reviewers
                and agreement_rate >= cal.agreement_threshold
            )
            verdicts[i] = accepted
            material_disagreement = material_disagreement or not accepted

        return verdicts, material_disagreement
