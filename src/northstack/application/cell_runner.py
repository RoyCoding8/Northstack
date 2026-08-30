"""The per-cell attempt loop.

``CellRunner`` owns the per-cell retry loop: the attempt-by-attempt driver
that runs a ``WorkerProtocol`` against one cell, records spend, selects a
recovery action on failure, reroutes on a repeated strategy signature, and
carries the in-flight conversation across retries via the
``resume_from_messages`` seam (ADR 0001).

It is the sole owner of ``contract.budget.max_retries``: ``max_retries == 0``
means no configured cap; otherwise the cap is enforced before the recovery
action is selected so the ledger records the action actually taken.

It never touches the ledger directly. Every authoritative transition is
emitted through the ``emit`` callback the orchestrator binds to
``EventStream`` -- preserving the "Company is the only component allowed to
append authoritative state/outcome transitions" invariant. The
``resume_from_messages`` conversation is carried in-memory across retries
and never serialised into a ledger payload (ADR 0001); for run-level resume
it IS persisted to disk under the workspace's ``.northstack/resume`` tree and
cleared on cell success.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.wire import ModelMessage, ToolDefinition
from northstack.application.budget_authority import BudgetAuthority
from northstack.application.recovery import RecoveryManager
from northstack.application.retry import RetryPolicy, WorkerErrorKind, classify
from northstack.application.routing import RouteDecision, Router
from northstack.config import Capability
from northstack.domain import ArtifactRef
from northstack.domain.budget import BudgetUsage, Spend
from northstack.domain.contract import WorkContract
from northstack.domain.graph import GraphCell
from northstack.domain.outcome import (
    AttemptSignature,
    FailureType,
    RecoveryAction,
    RunOutcome,
)
from northstack.application.tracing import cell_trace, fanout
from northstack.application.worker import WorkerEvent
from northstack.ports.protocols import MemoryPort
from northstack.events.catalog import (
    CellCompleted,
    CellFailed,
    CellProgress,
    CellStarted,
    EventPayload,
    RecoveryTransition,
    RouteSelected,
)

logger = logging.getLogger(__name__)

_RETRY_ACTIONS = frozenset({RecoveryAction.BACKOFF_RETRY, RecoveryAction.CHANGED_STRATEGY_RETRY})
_ABSTAIN_ACTIONS = frozenset(
    {
        RecoveryAction.ABSTAIN,
        RecoveryAction.CONTRACT_AMENDMENT,
        RecoveryAction.SPLIT_REPLAN,
        RecoveryAction.SCOPE_REDUCTION,
    }
)


def _recovery_backoff_delay(attempt: int) -> float:
    """Exponential delay between recovery rounds, capped at 8s."""
    delay = min(0.5 * (2 ** (attempt - 1)), 8.0)
    return float(delay)


def _progress_sink(
    emit: _Emit, run_id: str, cell_id: str, attempt: int
) -> Callable[[WorkerEvent], Awaitable[None]]:
    """Project the worker's loop events onto the ledger.

    Scalars only: a detail value that is not a scalar is dropped rather than
    stringified, so message content can never reach the ledger by accident.
    """

    async def sink(event: WorkerEvent) -> None:
        await emit(
            run_id,
            CellProgress(
                cell_id=cell_id,
                step=event.kind.value,
                attempt=attempt,
                turn=event.turn,
                detail={
                    k: v for k, v in event.detail.items() if isinstance(v, str | int | float | bool)
                },
            ),
        )

    return sink


async def _recovery_sleep(delay: float) -> None:
    """Sleep seam between recovery rounds -- monkeypatched in tests to skip.

    ``await asyncio.sleep`` yields the event loop, so a cell in its backoff
    window does not block a concurrently-gathered read-only sibling.
    """
    await asyncio.sleep(delay)


_LEGACY_ERROR_KINDS: dict[str, WorkerErrorKind] = {
    "rate_limit": WorkerErrorKind.PROVIDER,
    "timeout": WorkerErrorKind.PROVIDER,
}


def _coerce_error_kind(error_kind: str) -> WorkerErrorKind:
    """Coerce a (possibly legacy) error_kind string to the shared enum."""
    try:
        return WorkerErrorKind(error_kind)
    except ValueError:
        return _LEGACY_ERROR_KINDS.get(error_kind, WorkerErrorKind.PROVIDER)


def _tool_plan_fingerprint(cell: GraphCell, decision: RouteDecision) -> str:
    """Stable strategy fingerprint for the attempt signature.

    A rerouted profile with the same tools yields the same plan, so a stuck
    strategy is detectable. Kept coarse deliberately: a strategy fingerprint,
    not a per-call log.
    """
    return f"{cell.mode}:{decision.selected_profile}"


def _evidence_digest_so_far(cell: GraphCell) -> str:
    """Digest of prior evidence that triggered the retry.

    Empty until a real evidence artifact exists; a fresh artifact after a
    changed strategy resets escalation. Retries of the same strategy dedup
    against each other.
    """
    return ""


async def _store_evidence(
    artifact_store: ArtifactStore, content: bytes, media_type: str = "application/json"
) -> ArtifactRef:
    return await asyncio.to_thread(artifact_store.write, content, media_type=media_type)


class _Emit(Protocol):
    async def __call__(self, run_id: str, payload: EventPayload) -> None: ...


class CellRunner:
    """Owns the per-cell attempt loop.

    Driven per cell via :meth:`run_cell`; emits through ``emit`` only and
    never opens the ledger itself.
    """

    def __init__(
        self,
        *,
        worker: Any,
        router: Router,
        retry_policy: RetryPolicy,
        recovery: RecoveryManager,
        artifact_store: ArtifactStore,
        resume_dir: Path | None = None,
        memory: MemoryPort | None = None,
        memory_namespace: str = "default",
    ) -> None:
        self._worker = worker
        self._router = router
        self._retry_policy = retry_policy
        self._recovery = recovery
        self._artifact_store = artifact_store
        self._resume_dir = resume_dir
        self._memory = memory
        self._memory_namespace = memory_namespace

    def _recalled_prompt(self, cell: GraphCell) -> str:
        """Prior knowledge relevant to this cell, as a system-prompt section.

        Advisory, never authoritative: a memory is what an earlier run
        believed, and the verification gates still decide what is true.  Say
        so in the prompt, or a stale lesson reads as a specification.
        """
        if self._memory is None:
            return ""
        try:
            found = self._memory.recall(self._memory_namespace, cell.contract.objective)
        except Exception:
            logger.warning("memory recall failed for cell %s", cell.id, exc_info=True)
            return ""
        if not found:
            return ""
        lines = "\n".join(f"- {m.text}" for m in found)
        return (
            f"Prior knowledge from earlier runs (advisory -- verify before relying on it):\n{lines}"
        )

    def _remember_outcome(self, cell: GraphCell, run_id: str, result: Any) -> None:
        if self._memory is None:
            return
        try:
            self._memory.remember(
                self._memory_namespace,
                f"{cell.name}: {cell.contract.objective}\nOutcome: {result.text}",
                source=run_id,
            )
        except Exception:
            logger.warning("memory write failed for cell %s", cell.id, exc_info=True)

    def _checkpoint_path(self, run_id: str, cell_id: str) -> Path | None:
        if self._resume_dir is None:
            return None
        return self._resume_dir / f"{cell_id}.json"

    def _write_checkpoint(self, run_id: str, cell_id: str, messages: list[Any]) -> None:
        """Persist the conversation after a tool round (best effort).

        A failed checkpoint write must never kill the cell that is actually
        running; the worst case is losing resume state for one round.
        """
        path = self._checkpoint_path(run_id, cell_id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"messages": [m.model_dump(mode="json") for m in messages]}),
                encoding="utf-8",
            )
            tmp.replace(path)
        except Exception:
            logger.debug("checkpoint write failed cell=%s", cell_id, exc_info=True)

    def _load_checkpoint(self, run_id: str, cell_id: str) -> list[Any] | None:
        path = self._checkpoint_path(run_id, cell_id)
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [ModelMessage.model_validate(m) for m in data["messages"]]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _clear_checkpoint(self, run_id: str, cell_id: str) -> None:
        path = self._checkpoint_path(run_id, cell_id)
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    async def run_cell(
        self,
        *,
        run_id: str,
        cell: GraphCell,
        contract: WorkContract,
        tool_defs: list[ToolDefinition],
        usage: BudgetUsage,
        evidence_digests: dict[int, str],
        tools_used: list[str],
        budget_auth: BudgetAuthority,
        emit: _Emit,
        heartbeat: Callable[[], None] | None = None,
    ) -> RunOutcome | bool:
        usage_before = usage.model_copy()
        needs = {Capability.TOOL_USE} if tool_defs else set()
        decision = self._router.route(cell, contract, require_capabilities=needs)
        await emit(
            run_id,
            RouteSelected(
                cell_id=cell.id, profile_name=decision.selected_profile, reason=decision.reason
            ),
        )

        if decision.abstained:
            await emit(
                run_id,
                CellFailed(cell_id=cell.id, error=decision.reason, error_kind="routing"),
            )
            return RunOutcome.ABSTAINED

        await emit(run_id, CellStarted(cell_id=cell.id, profile_name=decision.selected_profile))

        attempt_number = 0
        resume_messages: list[Any] | None = self._load_checkpoint(run_id, cell.id)
        recalled = self._recalled_prompt(cell)
        next_decision: RouteDecision | None = None
        result: Any = None
        while True:
            if heartbeat is not None:
                heartbeat()
            try:
                with cell_trace(run_id, cell.id, decision.selected_profile) as span_sink:
                    progress: dict[str, Any] = {
                        "on_event": fanout(
                            _progress_sink(emit, run_id, cell.id, attempt_number), span_sink
                        )
                    }
                    if heartbeat is not None:
                        progress["on_progress"] = heartbeat
                    if self._resume_dir is not None:
                        progress["on_checkpoint"] = lambda msgs: self._write_checkpoint(
                            run_id, cell.id, msgs
                        )
                    result = await self._worker.run(
                        cell,
                        decision.selected_profile,
                        tool_defs,
                        system_prompt=recalled,
                        output_json_schema=cell.output_schema or None,
                        resume_from_messages=resume_messages,
                        **progress,
                    )
            except Exception as e:  # noqa: BLE001
                error_kind = "provider"
                error_detail = str(e)
                result = None
            else:
                budget_auth.record(
                    Spend(
                        input_tokens=result.total_input_tokens,
                        output_tokens=result.total_output_tokens,
                        cost_usd=result.total_cost_usd,
                        calls=result.api_calls,
                    )
                )
                usage.total_input_tokens += result.total_input_tokens
                usage.total_output_tokens += result.total_output_tokens
                usage.total_cost_usd += result.total_cost_usd
                usage.total_calls += result.api_calls
                if result.ok:
                    break
                error_kind = result.error_kind
                error_detail = result.error
                if contract.budget.max_calls and usage.total_calls >= contract.budget.max_calls:
                    error_kind = "budget"
                    error_detail = "API call limit exceeded"

            capped = (
                contract.budget.max_retries != 0 and attempt_number >= contract.budget.max_retries
            )
            if capped:
                action = RecoveryAction.TERMINATE
                await _emit_recovery(
                    emit,
                    run_id,
                    cell.id,
                    error_kind,
                    error_detail,
                    action,
                    attempt_number,
                    self._recovery,
                )
            else:
                action = await self._handle_cell_failure(
                    run_id,
                    cell,
                    contract,
                    error_kind,
                    error_detail,
                    decision,
                    attempt_number,
                    usage,
                    emit,
                )
            attempt_number += 1

            carried = list(result.messages) if result is not None and result.messages else None

            if action in _RETRY_ACTIONS:
                await _recovery_sleep(_recovery_backoff_delay(attempt_number))
                resume_messages = carried
                continue
            if action == RecoveryAction.REROUTE_ESCALATE:
                next_decision = self._router.route(
                    cell,
                    contract,
                    remaining_budget=usage.remaining(contract.budget),
                    excluded_profiles={decision.selected_profile},
                    require_capabilities=needs,
                )
                if not next_decision.abstained:
                    decision = next_decision
                    await emit(
                        run_id,
                        RouteSelected(
                            cell_id=cell.id,
                            profile_name=decision.selected_profile,
                            reason=f"recovery reroute: {decision.reason}",
                        ),
                    )
                    resume_messages = carried
                    continue

                if classify(_coerce_error_kind(error_kind)) == FailureType.TRANSIENT:
                    await _emit_recovery(
                        emit,
                        run_id,
                        cell.id,
                        error_kind,
                        error_detail,
                        RecoveryAction.BACKOFF_RETRY,
                        attempt_number,
                        self._recovery,
                    )
                    await _recovery_sleep(_recovery_backoff_delay(attempt_number))
                    resume_messages = carried
                    continue

            if (
                action == RecoveryAction.REROUTE_ESCALATE
                and next_decision is not None
                and next_decision.abstained
            ):
                effective_error = f"{next_decision.reason} (original failure: {error_detail})"
            else:
                effective_error = error_detail

            await emit(
                run_id,
                CellFailed(cell_id=cell.id, error=effective_error[:500], error_kind=error_kind),
            )
            return RunOutcome.ABSTAINED if action in _ABSTAIN_ACTIONS else RunOutcome.FAILED

        if result is None:  # pragma: no cover
            raise RuntimeError("cell loop exited without a result")

        output_bytes = result.text.encode("utf-8") if result.text else b"{}"
        ref = await _store_evidence(self._artifact_store, output_bytes)
        self._clear_checkpoint(run_id, cell.id)
        self._remember_outcome(cell, run_id, result)
        tools_used.extend(self._tool_names_used(result))

        for ci in cell.acceptance_criterion_indices:
            evidence_digests[ci] = ref.digest

        await emit(
            run_id,
            CellCompleted(
                cell_id=cell.id,
                output_artifact=ref,
                usage=Spend(
                    input_tokens=usage.total_input_tokens - usage_before.total_input_tokens,
                    output_tokens=usage.total_output_tokens - usage_before.total_output_tokens,
                    cost_usd=usage.total_cost_usd - usage_before.total_cost_usd,
                    calls=usage.total_calls - usage_before.total_calls,
                ),
            ),
        )
        return True

    async def _handle_cell_failure(
        self,
        run_id: str,
        cell: GraphCell,
        contract: WorkContract,
        error_kind: str,
        error_detail: str,
        decision: RouteDecision,
        attempt_number: int,
        usage: BudgetUsage | None,
        emit: _Emit,
    ) -> RecoveryAction:
        failure_type = classify(_coerce_error_kind(error_kind))
        if failure_type == FailureType.SAFETY:
            action = RecoveryAction.TERMINATE
            await _emit_recovery(
                emit,
                run_id,
                cell.id,
                error_kind,
                error_detail,
                action,
                attempt_number,
                self._recovery,
            )
            return action
        if failure_type == FailureType.BUDGET:
            action = self._recovery.decide(
                run_id=run_id,
                error_kind=error_kind,
                error_detail=error_detail,
                attempt_signature=None,
                contract_budget=contract.budget,
                usage=usage,
            )
            await _emit_recovery(
                emit,
                run_id,
                cell.id,
                error_kind,
                error_detail,
                action,
                attempt_number,
                self._recovery,
            )
            return action

        sig = AttemptSignature(
            contract_version=contract.version,
            cell_id=cell.id,
            profile_name=decision.selected_profile,
            tool_plan=_tool_plan_fingerprint(cell, decision),
            evidence_digest=_evidence_digest_so_far(cell),
        )
        action = self._retry_policy.next_action(
            attempt=attempt_number,
            failure=failure_type,
            tried_sig=sig,
        )
        await _emit_recovery(
            emit, run_id, cell.id, error_kind, error_detail, action, attempt_number, self._recovery
        )
        return action

    def _tool_names_used(self, result: Any) -> list[str]:
        """Extract authoritative tool names from assistant tool-call messages."""
        return [call.name for message in result.messages for call in message.tool_calls]


async def _emit_recovery(
    emit: _Emit,
    run_id: str,
    cell_id: str,
    error_kind: str,
    error_detail: str,
    action: RecoveryAction,
    attempt_number: int,
    recovery: RecoveryManager,
) -> None:
    await emit(
        run_id,
        RecoveryTransition(
            cell_id=cell_id,
            failure_type=recovery.classify(error_kind, error_detail),
            action=action,
            attempt_number=attempt_number,
            error_detail=error_detail[:2000],
        ),
    )
