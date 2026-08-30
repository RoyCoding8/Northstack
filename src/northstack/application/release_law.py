"""The release law: the sole authority that decides a run's outcome.

Nothing else in the control plane constructs ``RunOutcome``.  The orchestrator
gathers the inputs -- hard-gate results, an optional soft review, budget usage,
the tools that were used -- and hands them to ``ReleaseLaw.decide``, which
returns the terminal ``Verdict``.  Centralising the decision here makes the law
inspectable and testable as a pure function of its inputs.

Invariants the law upholds (pinned by a matrix test in ``test_release_law.py``):

  1. Every acceptance criterion in the contract maps to exactly one result.
     A hard criterion needs a ``HardCheckResult`` at its index; a soft criterion
     needs an entry in ``soft.verdicts`` at its index.  An unevaluated criterion
     is never silently skipped to VERIFIED -- it yields FAILED, naming the index.
  2. A failed hard gate dominates every other signal: the verdict is FAILED
     regardless of the soft review or the remaining budget.
  3. With all hard gates passing, an exhausted budget means the run could not
     finish verification -- the verdict is ABSTAINED, never VERIFIED.
  4. With all hard gates passing and budget remaining, a soft review that
     records material disagreement (or was absent when soft rubrics exist)
     means the verdict is ABSTAINED, never VERIFIED.
  5. A criterion whose kind the law does not recognise cannot pass by claim --
     an unknown kind is an unevaluated criterion, so the verdict is FAILED.

Order of evaluation: hard failure -> unevaluated criterion -> budget exhausted
-> soft disagreement/absence -> VERIFIED.  Hard failure is checked first because
it dominates; the unevaluated check then guarantees no criterion is skipped
before budget and soft signals are consulted.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from northstack.application.verification.hard_gates import HardCheckResult
from northstack.domain.budget import BudgetUsage
from northstack.domain.contract import AcceptanceCriterion, CriterionKind, WorkContract
from northstack.domain.outcome import RunOutcome

_HARD_KINDS: frozenset[CriterionKind] = frozenset(
    {
        CriterionKind.COMMAND,
        CriterionKind.FILE_DIFF,
        CriterionKind.TREE_DIGEST,
        CriterionKind.SCHEMA,
        CriterionKind.POLICY,
    }
)


class SoftReview(BaseModel):
    """Aggregated verdict of the blinded soft-rubric review.

    ``verdicts`` maps each soft criterion index to whether the reviewers
    accepted it; ``material_disagreement`` is True when the reviewers disagreed
    enough that the run must abstain rather than claim verification.  A soft
    review is optional: ``None`` means no soft rubrics were evaluated.
    """

    model_config = ConfigDict(frozen=True)

    verdicts: dict[int, bool] = Field(default_factory=dict)
    material_disagreement: bool = Field(default=False)


class Verdict(BaseModel):
    """The terminal decision of the release law for one run.

    ``outcome`` is the sole ``RunOutcome`` the control plane may emit; ``reason``
    is a human-readable audit string explaining how the law reached it.
    """

    model_config = ConfigDict(frozen=True)

    outcome: RunOutcome
    reason: str = Field(default="")


class ReleaseLaw:
    """Decide a run's outcome from its evidence -- the sole ``RunOutcome`` author.

    Stateless: every decision is a pure function of the inputs passed to
    ``decide``, so the law can be exercised exhaustively against a matrix
    without any run machinery.
    """

    def decide(
        self,
        contract: WorkContract,
        hard: Sequence[HardCheckResult],
        soft: SoftReview | None,
        usage: BudgetUsage,
        tools_used: Sequence[str],
    ) -> Verdict:
        """Return the terminal ``Verdict`` for a run's gathered evidence.

        See the module docstring for the invariants and their evaluation order.
        """
        criteria: Sequence[AcceptanceCriterion] = contract.acceptance_criteria

        hard_by_index: dict[int, HardCheckResult] = {r.criterion_index: r for r in hard}
        soft_verdicts: dict[int, bool] = soft.verdicts if soft is not None else {}

        for index, criterion in enumerate(criteria):
            kind = self._kind_of(criterion)
            if kind is None:
                return Verdict(
                    outcome=RunOutcome.FAILED,
                    reason=f"criterion {index} has unknown kind '{criterion.kind}'",
                )
            if kind in _HARD_KINDS:
                result = hard_by_index.get(index)
                if result is None:
                    return Verdict(
                        outcome=RunOutcome.FAILED,
                        reason=f"unevaluated criterion {index} (hard, {kind.value})",
                    )
            else:
                if index not in soft_verdicts:
                    return Verdict(
                        outcome=RunOutcome.FAILED,
                        reason=f"unevaluated criterion {index} (soft rubric)",
                    )

        hard_failures = [r for r in hard if r.criterion_index < len(criteria) and not r.passed]
        if hard_failures:
            first = hard_failures[0]
            return Verdict(
                outcome=RunOutcome.FAILED,
                reason=(
                    f"hard gate failed at criterion {first.criterion_index} "
                    f"({first.kind.value}): {first.detail}"
                ),
            )

        if usage.exceeds(contract.budget):
            return Verdict(
                outcome=RunOutcome.ABSTAINED,
                reason="budget exhausted before verification completed",
            )

        soft_indices = [
            i for i, c in enumerate(criteria) if self._kind_of(c) == CriterionKind.SOFT_RUBRIC
        ]
        if soft_indices:
            if soft is None or soft.material_disagreement:
                return Verdict(
                    outcome=RunOutcome.ABSTAINED,
                    reason="soft review recorded material disagreement",
                )
            if not all(soft_verdicts.get(i, False) for i in soft_indices):
                return Verdict(
                    outcome=RunOutcome.ABSTAINED,
                    reason="soft rubric not satisfied by blinded reviewers",
                )

        return Verdict(outcome=RunOutcome.VERIFIED, reason="all acceptance criteria satisfied")

    @staticmethod
    def _kind_of(criterion: AcceptanceCriterion) -> CriterionKind | None:
        """Return the recognised ``CriterionKind`` for a criterion, or None.

        ``AcceptanceCriterion.kind`` is stored as a raw string so it can carry
        kinds the law does not yet know; this centralises the "is this a kind we
        evaluate?" question in one place.
        """
        try:
            return CriterionKind(criterion.kind)
        except ValueError:
            return None
