"""Run status and the authoritative transition table.

  intake -> contracted -> planned -> executing -> verifying
    -> verified | abstained | failed

Terminal outcomes (verified, abstained, failed) cannot transition.

ADR 0001 invariant: ``EXECUTING -> ABSTAINED`` is illegal. A run that
abstains while executing (budget exhaustion, a routing abstention, a
short-circuit) must route through ``VERIFYING`` first::

    EXECUTING -> VERIFYING -> ABSTAINED

That detour lives in exactly one place -- :meth:`RunStateMachine.route` --
instead of being patched inline at each emit site.
"""

from __future__ import annotations

import enum
from itertools import pairwise
from typing import ClassVar


class RunStatus(str, enum.Enum):
    INTAKE = "intake"
    CONTRACTED = "contracted"
    PLANNED = "planned"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    ABSTAINED = "abstained"
    FAILED = "failed"

    @classmethod
    def can_transition(cls, from_status: RunStatus, to_status: RunStatus) -> bool:
        """Return True if the transition is legal."""
        return RunStateMachine.can_transition(from_status, to_status)

    @classmethod
    def is_terminal(cls, status: RunStatus) -> bool:
        """Return True if no transitions are allowed out."""
        return RunStateMachine.is_terminal(status)


def _build_transitions() -> dict[RunStatus, frozenset[RunStatus]]:
    """Build the total transition table.

    Every non-terminal phase advances to its happy-path successor and may
    escape to ``FAILED``. ``ABSTAINED`` is reachable from every non-terminal
    phase *except* ``EXECUTING`` -- an execution-time abstention routes through
    ``VERIFYING`` (ADR 0001), so ``EXECUTING``'s escape set deliberately omits
    ``ABSTAINED``. Terminal phases map to an empty set.
    """
    phases = (
        RunStatus.INTAKE,
        RunStatus.CONTRACTED,
        RunStatus.PLANNED,
        RunStatus.EXECUTING,
        RunStatus.VERIFYING,
        RunStatus.VERIFIED,
    )
    can_abstain_from = {p for p in phases[:-1] if p is not RunStatus.EXECUTING}
    transitions: dict[RunStatus, frozenset[RunStatus]] = {}
    for frm, to in pairwise(phases):
        escapes = (
            frozenset({RunStatus.ABSTAINED, RunStatus.FAILED})
            if frm in can_abstain_from
            else frozenset({RunStatus.FAILED})
        )
        transitions[frm] = escapes | {to}
    for terminal in (RunStatus.VERIFIED, RunStatus.ABSTAINED, RunStatus.FAILED):
        transitions[terminal] = frozenset()
    return transitions


class RunStateMachine:
    """Sole owner of the status transition table and the execution detour.

    The table is one nested dict of allowed transitions. ``can_transition``
    answers legality queries; ``route`` returns the concrete status path an
    emit site must walk to reach a target legally -- emitting the
    ``EXECUTING -> VERIFYING -> ABSTAINED`` detour for an execution-time
    abstention rather than an illegal direct edge.
    """

    _TRANSITIONS: ClassVar[dict[RunStatus, frozenset[RunStatus]]] = _build_transitions()

    _DETOURS: ClassVar[dict[tuple[RunStatus, RunStatus], list[RunStatus]]] = {
        (RunStatus.EXECUTING, RunStatus.ABSTAINED): [
            RunStatus.VERIFYING,
            RunStatus.ABSTAINED,
        ],
    }

    @classmethod
    def transitions(cls) -> dict[RunStatus, frozenset[RunStatus]]:
        """Return the authoritative transition table."""
        return cls._TRANSITIONS

    @classmethod
    def can_transition(cls, from_status: RunStatus, to_status: RunStatus) -> bool:
        """Return True if the (direct) transition is legal."""
        return to_status in cls._TRANSITIONS[from_status]

    @classmethod
    def is_terminal(cls, status: RunStatus) -> bool:
        """Return True if no transitions are allowed out."""
        return not cls._TRANSITIONS[status]

    @classmethod
    def route(cls, from_status: RunStatus, to_status: RunStatus) -> list[RunStatus]:
        """Return the legal status path from ``from_status`` to ``to_status``.

        A legal direct edge routes as ``[to_status]``. The single exception is
        an execution-time abstention: ``EXECUTING -> ABSTAINED`` is illegal, so
        the path detours through ``VERIFYING`` (ADR 0001) and returns
        ``[VERIFYING, ABSTAINED]``.

        Raises ``ValueError`` for a transition that is neither a legal direct
        edge nor a known detour.
        """
        detour = cls._DETOURS.get((from_status, to_status))
        if detour is not None:
            return list(detour)
        if cls.can_transition(from_status, to_status):
            return [to_status]
        raise ValueError(f"Illegal status transition: {from_status.value} -> {to_status.value}")
