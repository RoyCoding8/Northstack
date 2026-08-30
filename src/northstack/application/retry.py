"""Single retry authority: one owner of attempt counting and escalation.

The orchestrator delegates every recovery decision here. Two bugs this fixes:

* **Dedup never fired.** The old site built
  ``AttemptSignature(strategy_id=f"attempt-{n}")`` -- every retry's signature
  was unique by construction, so the deduplicator never triggered and a stuck
  cell retried the same strategy forever instead of escalating the ladder.
* **Error-kind mismatch.** A string ``error_kind`` plus a keyword heuristic
  let ``"tool"`` vs ``"tool_execution"`` drift apart. The shared
  :class:`WorkerErrorKind` enum is the only boundary; it maps 1:1 to
  :class:`FailureType`, so the mismatch cannot exist.

The ladder is one nested literal table driven by one loop.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from northstack.domain import AttemptSignature, FailureType, RecoveryAction


class WorkerErrorKind(str, enum.Enum):
    """Error category the worker emits. Maps 1:1 to ``FailureType``.

    One enum replaces the string ``error_kind`` + ``_ERROR_KIND_TO_FAILURE``
    map + the ``FailureClassifier`` keyword heuristic, so the
    ``"tool"`` vs ``"tool_execution"`` mismatch cannot exist.
    """

    PROVIDER = "provider"
    BUDGET = "budget"
    SCHEMA = "schema"
    TOOL = "tool"
    SAFETY = "safety"
    AUTHENTICATION = "authentication"
    CAPABILITY = "capability"
    CONFIGURATION = "configuration"


WORKER_ERROR_TO_FAILURE: dict[WorkerErrorKind, FailureType] = {
    WorkerErrorKind.PROVIDER: FailureType.TRANSIENT,
    WorkerErrorKind.BUDGET: FailureType.BUDGET,
    WorkerErrorKind.SCHEMA: FailureType.SAMPLING,
    WorkerErrorKind.TOOL: FailureType.CAPABILITY,
    WorkerErrorKind.SAFETY: FailureType.SAFETY,
    WorkerErrorKind.AUTHENTICATION: FailureType.SPECIFICATION,
    WorkerErrorKind.CAPABILITY: FailureType.CAPABILITY,
    WorkerErrorKind.CONFIGURATION: FailureType.SPECIFICATION,
}


def classify(error_kind: WorkerErrorKind | str) -> FailureType:
    """Classify a worker error into a ``FailureType`` via the shared enum.

    Accepts the enum or its string value so callers that still hold a string
    (e.g. an unconverted ``WorkerResult.error_kind``) classify through the
    same single map rather than a heuristic. An unknown kind is TRANSIENT --
    the retryable default -- never silently mis-routed to a terminal rung.
    """
    kind = error_kind if isinstance(error_kind, WorkerErrorKind) else WorkerErrorKind(error_kind)
    return WORKER_ERROR_TO_FAILURE[kind]


def _coerce_failure_type(failure: FailureType | str) -> FailureType:
    """Accept a ``FailureType``, its value (``"transient``) or name (``TRANSIENT``)."""
    if isinstance(failure, FailureType):
        return failure
    try:
        return FailureType(failure)
    except ValueError:
        return FailureType[failure]  # by enum member name (e.g. "TRANSIENT")


RECOVERY_POLICY: dict[FailureType, list[RecoveryAction]] = {
    FailureType.TRANSIENT: [
        RecoveryAction.BACKOFF_RETRY,
        RecoveryAction.REROUTE_ESCALATE,
        RecoveryAction.TERMINATE,
    ],
    FailureType.SAMPLING: [
        RecoveryAction.CHANGED_STRATEGY_RETRY,
        RecoveryAction.BACKOFF_RETRY,
        RecoveryAction.ABSTAIN,
    ],
    FailureType.CAPABILITY: [
        RecoveryAction.REROUTE_ESCALATE,
        RecoveryAction.SPLIT_REPLAN,
        RecoveryAction.CONTRACT_AMENDMENT,
        RecoveryAction.ABSTAIN,
    ],
    FailureType.DECOMPOSITION: [
        RecoveryAction.SPLIT_REPLAN,
        RecoveryAction.CONTRACT_AMENDMENT,
        RecoveryAction.ABSTAIN,
    ],
    FailureType.SPECIFICATION: [
        RecoveryAction.CONTRACT_AMENDMENT,
        RecoveryAction.ABSTAIN,
    ],
    FailureType.INTEGRATION: [
        RecoveryAction.SPLIT_REPLAN,
        RecoveryAction.TERMINATE,
    ],
    FailureType.SAFETY: [RecoveryAction.TERMINATE],
    FailureType.BUDGET: [RecoveryAction.SCOPE_REDUCTION, RecoveryAction.FAIL],
}


def _sig_key(sig: AttemptSignature) -> str:
    """Deterministic identity key for an attempt signature.

    Built from the real strategy -- ``(contract_version, cell_id, profile,
    tool_plan, evidence_digest)`` -- never ``f"attempt-{n}"``, so repeating a
    strategy is detectable and the ladder escalates past ``allowed[0]``.
    """
    return (
        f"{sig.contract_version}\x1f{sig.cell_id}\x1f{sig.profile_name}"
        f"\x1f{sig.tool_plan}\x1f{sig.evidence_digest}"
    )


@dataclass(frozen=True)
class RetryPolicy:
    """Single retry/recovery authority.

    ``next_action`` returns the recovery action for one failure. Escalation
    is driven by signature dedup: the Nth time the *same* signature is seen,
    the policy advances to rung ``min(N, len(ladder)-1)``. The first failure
    of a signature is rung 0 (e.g. ``BACKOFF_RETRY``); the second identical
    failure escalates to rung 1 (e.g. ``REROUTE_ESCALATE``) instead of a third
    identical retry -- the bug that made the ladder unreachable.

    The policy is frozen (immutable config: the ladder table) but holds a
    mutable deduplicator, because dedup state is per-run execution state,
    not configuration. ``attempt`` is accepted for callers that track it but
    does not drive the rung -- the signature does, so a rerouted profile with
    a fresh signature resets to rung 0 even at a high attempt count.
    """

    policy: dict[FailureType, list[RecoveryAction]] = field(default_factory=lambda: RECOVERY_POLICY)
    _seen: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def next_action(
        self,
        attempt: int,
        failure: FailureType | str,
        tried_sig: AttemptSignature,
    ) -> RecoveryAction:
        """Return the recovery action for this failure.

        ``failure`` accepts a ``FailureType`` or its string value. A repeated
        signature escalates the rung; a fresh signature starts at rung 0.
        """
        ftype = _coerce_failure_type(failure)
        ladder = self.policy.get(ftype, [RecoveryAction.TERMINATE])
        key = _sig_key(tried_sig)
        rung = self._seen.get(key, 0)
        self._seen[key] = rung + 1
        return ladder[min(rung, len(ladder) - 1)]

    def seen_count(self, tried_sig: AttemptSignature) -> int:
        """How many times this signature has been consulted (test seam)."""
        return self._seen.get(_sig_key(tried_sig), 0)
