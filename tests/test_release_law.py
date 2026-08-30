"""Release law: the sole authority that decides a run's outcome.

The matrix is the contract the module must keep:

  hard pass/fail  x  soft pass/abstain/absent  x  budget ok/exhausted

plus the row that matters most: a criterion with *no* result at all must yield
``FAILED``.  An unevaluated criterion is never silently skipped to VERIFIED.
"""

from __future__ import annotations

import pytest

from northstack.application.release_law import ReleaseLaw, SoftReview, Verdict
from northstack.application.verification.hard_gates import HardCheckResult
from northstack.domain import Budget, WorkContract
from northstack.domain.budget import BudgetUsage
from northstack.domain.contract import (
    AcceptanceCriterion,
    CommandCriterion,
    CriterionKind,
    SoftRubricCriterion,
)
from northstack.domain.outcome import RunOutcome

# This test pins the matrix the sole RunOutcome authority must uphold.  Any
# change to the imports above is a regression in the contract.


# Fixtures: contracts with a known mix of hard and soft criteria


def _criterion(kind: CriterionKind, index: int) -> AcceptanceCriterion:
    """Build a typed criterion for the matrix fixtures.

    The discriminated union is not callable, so each kind is constructed via its
    variant class. Only the kinds the matrix uses are supported; the bogus-kind
    case is exercised separately (a bogus kind cannot be built at all now).
    """
    desc = f"criterion-{index}"
    if kind is CriterionKind.COMMAND:
        return CommandCriterion(description=desc, command_name="probe")
    if kind is CriterionKind.SOFT_RUBRIC:
        return SoftRubricCriterion(description=desc)
    raise AssertionError(f"_criterion does not fixture kind {kind!r}")


def _contract(
    criteria: list[AcceptanceCriterion],
    *,
    cost_limit: float | None = 5.0,
) -> WorkContract:
    """A minimal contract carrying exactly the given criteria."""
    return WorkContract(
        id="wc-1",
        version=1,
        objective="o",
        budget=Budget.default(),
        acceptance_criteria=criteria,
    )


def _hard(index: int, kind: CriterionKind, *, passed: bool) -> HardCheckResult:
    return HardCheckResult(criterion_index=index, kind=kind, passed=passed)


def _soft(passed_indices: list[int], *, disagree: bool = False) -> SoftReview:
    return SoftReview(
        verdicts={i: True for i in passed_indices},
        material_disagreement=disagree,
    )


def _usage(*, cost: float) -> BudgetUsage:
    return BudgetUsage(total_cost_usd=cost)


# The matrix: (hard, soft, budget) -> expected RunOutcome

# A contract with one hard (COMMAND, index 0) and one soft (SOFT_RUBRIC, index 1).
_MIXED_CONTRACT = _contract(
    [_criterion(CriterionKind.COMMAND, 0), _criterion(CriterionKind.SOFT_RUBRIC, 1)]
)
# A contract with a single hard criterion and no soft criteria: ``soft=None`` is
# the honest "no soft rubrics exist" case, not an unevaluated criterion.
_HARD_ONLY_CONTRACT = _contract([_criterion(CriterionKind.COMMAND, 0)])


MATRIX = [
    # hard passes, budget ok
    pytest.param(
        _MIXED_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=True)],
        _soft([1]),
        _usage(cost=0.0),
        RunOutcome.VERIFIED,
        id="hard-pass-soft-pass-budget-ok",
    ),
    pytest.param(
        _HARD_ONLY_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=True)],
        None,
        _usage(cost=0.0),
        RunOutcome.VERIFIED,
        id="hard-pass-soft-absent-budget-ok",
    ),
    pytest.param(
        _MIXED_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=True)],
        _soft([1], disagree=True),
        _usage(cost=0.0),
        RunOutcome.ABSTAINED,
        id="hard-pass-soft-disagree-budget-ok",
    ),
    # hard fails -- dominates everything
    pytest.param(
        _MIXED_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=False)],
        _soft([1]),
        _usage(cost=0.0),
        RunOutcome.FAILED,
        id="hard-fail-soft-pass",
    ),
    pytest.param(
        _MIXED_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=False)],
        _soft([1], disagree=True),
        _usage(cost=0.0),
        RunOutcome.FAILED,
        id="hard-fail-soft-disagree",
    ),
    pytest.param(
        _HARD_ONLY_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=False)],
        None,
        _usage(cost=0.0),
        RunOutcome.FAILED,
        id="hard-fail-soft-absent",
    ),
    # hard passes, budget exhausted -> cannot finish verification
    pytest.param(
        _MIXED_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=True)],
        _soft([1]),
        _usage(cost=100.0),
        RunOutcome.ABSTAINED,
        id="hard-pass-soft-pass-budget-exhausted",
    ),
    pytest.param(
        _HARD_ONLY_CONTRACT,
        [_hard(0, CriterionKind.COMMAND, passed=True)],
        None,
        _usage(cost=100.0),
        RunOutcome.ABSTAINED,
        id="hard-pass-soft-absent-budget-exhausted",
    ),
]


@pytest.mark.parametrize("contract, hard, soft, usage, expected", MATRIX)
def test_release_law_matrix(contract, hard, soft, usage, expected):
    law = ReleaseLaw()
    verdict = law.decide(
        contract=contract,
        hard=hard,
        soft=soft,
        usage=usage,
        tools_used=[],
    )
    assert isinstance(verdict, Verdict)
    assert verdict.outcome == expected


# The row that matters most: an unevaluated criterion -> FAILED


def test_unevaluated_criterion_yields_failed():
    """A criterion with no result at all must not be skipped to VERIFIED.

    The contract has a hard criterion at index 0 and a soft criterion at
    index 1, but the caller supplies a result for index 0 only.  Index 1 is
    unevaluated -- the verdict must be FAILED, naming the unevaluated criterion.
    """
    contract = _MIXED_CONTRACT
    law = ReleaseLaw()
    verdict = law.decide(
        contract=contract,
        hard=[_hard(0, CriterionKind.COMMAND, passed=True)],
        # soft is None: the soft criterion at index 1 has no result.
        soft=None,
        usage=_usage(cost=0.0),
        tools_used=[],
    )
    assert verdict.outcome == RunOutcome.FAILED
    assert "unevaluated" in verdict.reason.lower()


def test_unknown_criterion_kind_cannot_reach_the_law():
    """A bogus criterion kind is rejected by the union before the law sees it.

    ``AcceptanceCriterion`` is a discriminated union, so a contract carrying an
    unknown kind cannot be constructed -- the release law never has to decide
    whether an unrecognised hard criterion was satisfied, because no such
    contract exists.  This pins that the gate moved earlier, to parse time.
    """
    from pydantic import TypeAdapter, ValidationError

    with pytest.raises(ValidationError):
        TypeAdapter(AcceptanceCriterion).validate_python({"kind": "bogus_kind", "description": "x"})


def test_soft_only_consulted_when_all_hard_pass():
    """Soft disagreement must not flip a hard-fail verdict to ABSTAINED.

    If any hard gate fails, soft review is irrelevant -- the verdict is FAILED
    regardless of how the soft rubric landed.  This stops a disagreeing soft
    review from masking a hard failure.
    """
    contract = _MIXED_CONTRACT
    law = ReleaseLaw()
    verdict = law.decide(
        contract=contract,
        hard=[_hard(0, CriterionKind.COMMAND, passed=False)],
        soft=_soft([1], disagree=True),
        usage=_usage(cost=0.0),
        tools_used=[],
    )
    assert verdict.outcome == RunOutcome.FAILED


def test_tree_digest_is_a_hard_kind():
    """Regression: TREE_DIGEST must classify as hard, not fall into the soft
    branch where its missing verdict reads as 'unevaluated criterion (soft
    rubric)' and fails an otherwise fully-verified run."""
    from northstack.domain import TreeDigestCriterion

    contract = _contract(
        [TreeDigestCriterion(description="suite pinned", path="tests", tree_hash="a" * 64)]
    )
    verdict = ReleaseLaw().decide(
        contract=contract,
        hard=[_hard(0, CriterionKind.TREE_DIGEST, passed=True)],
        soft=None,
        usage=_usage(cost=0.0),
        tools_used=[],
    )
    assert verdict.outcome == RunOutcome.VERIFIED


# re-export so a missing import surfaces as ModuleNotFoundError at collection
__all__ = [
    "ReleaseLaw",
    "SoftReview",
    "Verdict",
]
