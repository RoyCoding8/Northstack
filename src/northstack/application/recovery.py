"""Typed bounded recovery.

Public seam:
  - FailureClassifier: rule-based classification of worker errors
  - RecoveryPolicy: table-driven allowed actions per failure type
  - AttemptDeduplicator: exact AttemptSignature-based duplicate rejection
  - RecoveryManager: orchestrates classify -> policy lookup -> dedup -> action

Retry ownership: ``RecoveryManager`` is NOT a retry-limit owner. The
per-failed-cell ``Budget.max_retries`` is enforced by the orchestration
per-cell loop before selecting/emitting the next recovery action, so the
ledger records the action actually taken. ``RecoveryManager.decide`` returns
the policy-selected action for a single failure regardless of how many
retries the calling cell has already consumed; it has no run-wide counter
that could leak retry slots across cells.

Safety failure cannot retry or circumvent.
"""

from __future__ import annotations

from northstack.application.retry import RECOVERY_POLICY, WorkerErrorKind, classify
from northstack.domain import AttemptSignature, Budget, BudgetUsage, FailureType, RecoveryAction


class FailureClassifier:
    """Classify a worker error kind into a ``FailureType``.

    A thin, injectable wrapper over :func:`retry.classify` kept for existing
    call sites and tests. Classification now goes through the single shared
    ``WorkerErrorKind`` map; the keyword heuristic is gone.
    """

    def __init__(
        self,
        custom_rules: dict[str, FailureType] | None = None,
    ) -> None:
        self._custom = custom_rules or {}

    def classify(self, error_kind: str, detail: str = "") -> FailureType:
        """Classify via the shared enum; honour per-instance overrides."""
        if error_kind in self._custom:
            return self._custom[error_kind]
        try:
            return classify(WorkerErrorKind(error_kind))
        except ValueError:
            return FailureType.TRANSIENT


class AttemptDeduplicator:
    """Rejects duplicate recovery attempts via AttemptSignature.

    Tracks seen signatures within a run. Exact signature match = duplicate.
    """

    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}  # run_id -> set of signature hashes

    def _sig_hash(self, sig: AttemptSignature) -> str:
        """Deterministic hash of an attempt signature."""
        import hashlib
        import json

        canonical = json.dumps(sig.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def is_duplicate(self, run_id: str, sig: AttemptSignature) -> bool:
        """Return True if this exact signature has been seen before."""
        sig_hash = self._sig_hash(sig)
        run_sigs = self._seen.setdefault(run_id, set())
        return sig_hash in run_sigs

    def record(self, run_id: str, sig: AttemptSignature) -> None:
        """Record a signature as seen."""
        sig_hash = self._sig_hash(sig)
        self._seen.setdefault(run_id, set()).add(sig_hash)

    def clear(self, run_id: str) -> None:
        """Clear all signatures for a run."""
        self._seen.pop(run_id, None)


class RecoveryManager:
    """Orchestrates the recovery decision pipeline.

    Flow:
      1. Classify failure
      2. Budget exhaustion check (terminal for BUDGET failures)
      3. Deduplicate attempt signatures
      4. Select best allowed action from policy table

    Retry ownership: this class deliberately holds NO run-wide retry counter.
    The per-failed-cell ``Budget.max_retries`` cap is owned and enforced by
    the orchestration per-cell loop (see ``Company._run_cell``), which checks
    ``attempt_number`` against ``contract.budget.max_retries`` BEFORE
    selecting/emitting the next recovery action. ``decide`` therefore returns
    the policy-selected action for a single failure independent of how many
    retries the calling cell has already consumed, so a buggy hidden counter
    cannot leak retry slots across cells of the same run.
    """

    def __init__(
        self,
        classifier: FailureClassifier | None = None,
        deduplicator: AttemptDeduplicator | None = None,
        policy: dict[FailureType, list[RecoveryAction]] | None = None,
    ) -> None:
        self._classifier = classifier or FailureClassifier()
        self._deduplicator = deduplicator or AttemptDeduplicator()
        self._policy = policy or RECOVERY_POLICY

    def decide(
        self,
        run_id: str,
        error_kind: str,
        error_detail: str,
        attempt_signature: AttemptSignature | None = None,
        contract_budget: Budget | None = None,
        usage: BudgetUsage | None = None,
    ) -> RecoveryAction:
        """Decide on a recovery action for a failed cell.

        Returns the selected RecoveryAction. Safety failures always -> TERMINATE.
        Does NOT consult or record any retry count; the caller owns the cap.
        """
        failure_type = self._classifier.classify(error_kind, error_detail)

        if failure_type == FailureType.SAFETY:
            return RecoveryAction.TERMINATE

        if failure_type == FailureType.BUDGET and usage and contract_budget:
            cost_headroom = (
                contract_budget.cost_limit_usd is None
                or usage.total_cost_usd < contract_budget.cost_limit_usd * 0.5
            )
            token_headroom = (
                contract_budget.token_limit is None
                or usage.total_tokens < contract_budget.token_limit * 0.5
            )
            if cost_headroom or token_headroom:
                return RecoveryAction.SCOPE_REDUCTION
        if failure_type == FailureType.BUDGET:
            return RecoveryAction.FAIL

        if attempt_signature and self._deduplicator.is_duplicate(run_id, attempt_signature):
            return RecoveryAction.TERMINATE

        allowed = self._policy.get(failure_type, [RecoveryAction.TERMINATE])

        if allowed:
            action = allowed[0]
            if attempt_signature:
                self._deduplicator.record(run_id, attempt_signature)
            return action

        return RecoveryAction.TERMINATE

    def classify(self, error_kind: str, detail: str = "") -> FailureType:
        """Expose classification for external use."""
        return self._classifier.classify(error_kind, detail)

    def allowed_actions(self, failure_type: FailureType) -> list[RecoveryAction]:
        """Return allowed actions for a failure type."""
        return list(self._policy.get(failure_type, [RecoveryAction.TERMINATE]))
