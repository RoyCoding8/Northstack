"""Pipeline orchestration: Company.run public seam.

Public seam:
  - Company(config, ledger, workspace, gateway, ...) -> RunOutcome
  - Company.run_async(request) -> RunOutcome

Coordinates: request -> contract -> plan -> ready cells -> worker(s) ->
checkpoint/replan -> verify/recover -> outcome.

Company is the ONLY component allowed to append authoritative state/outcome
transitions. Model output is proposed data validated before eventing.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import uuid
from collections.abc import Callable, Coroutine, Sequence
from time import monotonic
from typing import Any, Protocol

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.wire import ToolDefinition
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.application.budget_authority import BudgetAuthority
from northstack.application.cell_runner import CellRunner
from northstack.application.contracting import ContractCompiler
from northstack.application.planning import GraphPlanner, single_cell_graph
from northstack.application.recovery import RecoveryManager
from northstack.application.release_law import ReleaseLaw, SoftReview, Verdict
from northstack.application.replay import replay_run
from northstack.application.retry import RetryPolicy
from northstack.application.routing import Router
from northstack.application.scheduling import Scheduler
from northstack.application.stall_detector import StallDetector
from northstack.application.tools.registry import ToolRegistry
from northstack.application.verification.hard_gates import HardCheckResult, HardGateVerifier
from northstack.application.verification.soft_rubric import BlindedReviewer, SoftRubricChecker
from northstack.application.worker import WorkerResult
from northstack.config import NorthStackConfig
from northstack.domain.budget import Budget, BudgetUsage, Spend
from northstack.domain.contract import CriterionKind, WorkContract
from northstack.domain.graph import CellMode, CellStatus, GraphCell, GraphVersion
from northstack.domain.outcome import (
    CalibrationRecord,
    EvidenceRecord,
    HardGateFailure,
    RunOutcome,
)
from northstack.domain.request import ProjectRequest
from northstack.domain.run_state import RunState
from northstack.domain.status import RunStateMachine, RunStatus
from northstack.events.catalog import (
    BudgetUpdated,
    CellCompleted,
    EventKind,
    EventPayload,
    EvidenceRecorded,
    GraphAccepted,
    GraphProposed,
    OutcomeEmitted,
    RequestAccepted,
    StallDetected,
    StatusChanged,
)
from northstack.events.stream import EventStream
from northstack.ports.protocols import MemoryPort

logger = logging.getLogger(__name__)

_RESUME_REPLAY_KINDS = frozenset(
    {
        EventKind.ANALYSIS_REQUESTED,
        EventKind.ANALYSIS_COMPLETED,
        EventKind.CONTRACT_PROPOSED,
        EventKind.CONTRACT_VALIDATED,
        EventKind.GRAPH_ACCEPTED,
    }
)

_OUTCOME_TO_STATUS: dict[RunOutcome, RunStatus] = {
    RunOutcome.VERIFIED: RunStatus.VERIFIED,
    RunOutcome.ABSTAINED: RunStatus.ABSTAINED,
    RunOutcome.FAILED: RunStatus.FAILED,
}


async def _finish_despite_cancellation(task: asyncio.Task[None]) -> asyncio.CancelledError | None:
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = exc
    task.result()
    return cancelled


def _budget_exhausted(auth: BudgetAuthority) -> bool:
    """True when the spend authority reports a set axis has run dry.

    The wave boundary asks ``BudgetAuthority`` -- the sole owner of spend --
    whether remaining headroom is zero on a *set* axis. Unlimited axes
    (``None``) never trip, mirroring ``BudgetUsage.exceeds(None)``. Token and
    cost axes are independent; either hitting zero means the run must abstain.
    """
    remaining = auth.remaining()
    if remaining.tokens is not None and remaining.tokens <= 0:
        return True
    return remaining.cost_usd is not None and remaining.cost_usd <= 1e-9


class _StalledSentinel:
    """Single-instance sentinel for ``is _STALLED`` checks."""

    __slots__ = ()


_STALLED = _StalledSentinel()

STALL_WATCHDOG_POLL_SECONDS = 0.1


class WorkerProtocol(Protocol):
    async def run(
        self,
        cell: GraphCell,
        profile_name: str,
        tool_defs: list[ToolDefinition],
        *,
        system_prompt: str = "",
        output_json_schema: dict[str, Any] | None = None,
        resume_from_messages: list[Any] | None = None,
    ) -> WorkerResult: ...


class WorkerFactory(Protocol):
    """Creates a worker bound to a workspace and gateway."""

    def create(self, workspace: RestrictedWorkspace) -> WorkerProtocol: ...


class GatewayTeardown(Protocol):
    """The gateway surface Company owns: only the teardown ``close``.

    Company never calls the gateway itself (workers/analysers do, through their
    own protocols); it only owns the close-on-the-same-loop lifecycle, so the
    honest contract is this single-method Protocol -- not the concrete adapter
    (importing the adapter would also violate the layer direction). Tests pass
    ``None`` (no gateway to close), so the parameter is ``GatewayTeardown | None``.
    """

    async def close(self) -> None: ...


class Company:
    """Pipeline orchestrator.

    Company is the ONLY component allowed to append authoritative state/outcome
    transitions. Model output is proposed data validated before eventing.

    Uses one ModelGateway (composition root) so profile limits are shared.
    """

    def __init__(
        self,
        *,
        config: NorthStackConfig,
        ledger: Ledger,
        artifact_store: ArtifactStore,
        workspace: RestrictedWorkspace,
        gateway: GatewayTeardown | None,
        worker_factory: WorkerFactory,
        compiler: ContractCompiler,
        command_profiles: dict[str, CommandProfile] | None = None,
        reviewers: list[BlindedReviewer] | None = None,
        calibration_records: list[CalibrationRecord] | None = None,
        tool_registry: ToolRegistry | None = None,
        planner: GraphPlanner | None = None,
        memory: MemoryPort | None = None,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._artifact_store = artifact_store
        self._workspace = workspace
        self._resume_dir = workspace.root / ".northstack" / "resume"
        self._memory = memory
        self._gateway = gateway
        self._worker_factory = worker_factory
        self._compiler = compiler
        self._command_profiles = command_profiles or {}
        self._router = Router(config)
        self._planner = planner or GraphPlanner(self._config.role_map())
        self._scheduler = Scheduler()
        self._hard_verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
            command_profiles=self._command_profiles,
        )
        self._soft_checker = SoftRubricChecker(
            reviewers=reviewers or [],
            calibration_records=calibration_records,
        )
        self._recovery = RecoveryManager()
        self._retry_policy = RetryPolicy()
        self._tool_registry = tool_registry or ToolRegistry.with_defaults(
            command_profiles=self._command_profiles
        )
        self._release_law = ReleaseLaw()
        self._stall_clock: Callable[[], float] = monotonic

    def run(self, request: ProjectRequest) -> RunOutcome:
        """Execute the full pipeline synchronously.

        An explicit sync entrypoint: it creates and owns its own event loop via
        ``asyncio.run``. It refuses to run when a loop is already running --
        callers inside a live loop must use :meth:`run_async` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(request))
        raise RuntimeError(
            "Company.run() is a synchronous entrypoint that owns its own event "
            "loop; it cannot run inside an already-running event loop. "
            "Use await company.run_async(...) instead."
        )

    async def run_async(self, request: ProjectRequest, *, run_id: str | None = None) -> RunOutcome:
        run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        current_status = RunStatus.INTAKE
        terminal_outcome: RunOutcome | None = None

        try:
            await self._emit(
                run_id,
                RequestAccepted(
                    goal=request.goal,
                    workspace_root=request.workspace_root,
                    budget=(
                        Budget(
                            token_limit=request.budget.token_limit,
                            cost_limit_usd=request.budget.cost_limit_usd,
                        )
                        if request.budget is not None
                        else None
                    ),
                ),
            )

            contract = await self._compiler.compile(request, self._ledger, self._config, run_id)
            await self._emit_status(run_id, RunStatus.CONTRACTED, current_status)
            current_status = RunStatus.CONTRACTED

            plan = await self._planner.plan(contract, run_id)
            graph = plan if isinstance(plan, GraphVersion) else single_cell_graph(plan)

            graph_errors = self._planner.validate(
                graph, run_budget=contract.budget, max_waves=request.max_waves
            )
            if graph_errors:
                await self._emit(
                    run_id,
                    GraphProposed(version=graph.version, errors=graph_errors, rejected=True),
                )
                cancelled = await self._emit_terminal_outcome(
                    run_id, RunOutcome.ABSTAINED, "graph validation failed"
                )
                terminal_outcome = RunOutcome.ABSTAINED
                if cancelled is not None:
                    raise cancelled
                await self._emit_status(run_id, RunStatus.ABSTAINED, current_status)
                return RunOutcome.ABSTAINED

            await self._emit(
                run_id,
                GraphAccepted(
                    version=graph.version,
                    cells=graph.cells,
                    edges=graph.edges,
                    milestones=graph.milestones,
                ),
            )
            await self._emit_status(run_id, RunStatus.PLANNED, current_status)
            current_status = RunStatus.PLANNED

            await self._emit_status(run_id, RunStatus.EXECUTING, current_status)
            current_status = RunStatus.EXECUTING

            usage = BudgetUsage()
            evidence_digests: dict[int, str] = {}
            tools_used: list[str] = []
            worker = self._worker_factory.create(self._workspace)

            budget_auth = BudgetAuthority(
                request.budget if request.budget is not None else Budget()
            )

            stall_detector = StallDetector(
                window_seconds=self._config.run.stall_window_seconds,
                clock=self._stall_clock,
            )

            cell_outcome = await self._execute_cells(
                run_id,
                graph,
                contract,
                worker,
                usage,
                evidence_digests,
                tools_used,
                request,
                budget_auth,
                stall_detector,
            )
            if cell_outcome is not None:
                cell_verdict = Verdict(
                    outcome=cell_outcome,
                    reason="cell short-circuited the run before verification",
                )
                await self._emit_evidence(
                    run_id,
                    cell_verdict,
                    [],
                    SoftReview(),
                    usage,
                    tools_used,
                )
                cancelled = await self._emit_terminal_outcome(
                    run_id, cell_outcome, cell_verdict.reason
                )
                terminal_outcome = cell_outcome
                if cancelled is not None:
                    raise cancelled
                final_status = _OUTCOME_TO_STATUS[cell_outcome]
                for nxt in RunStateMachine.route(current_status, final_status):
                    await self._emit_status(run_id, nxt, current_status)
                    current_status = nxt
                return cell_outcome

            final_outcome, cancelled = await self._verify_and_finish(
                run_id, contract, usage, evidence_digests, tools_used, current_status
            )
            terminal_outcome = final_outcome
            if cancelled is not None:
                raise cancelled
            await self._emit_status(run_id, _OUTCOME_TO_STATUS[final_outcome], RunStatus.VERIFYING)
            return final_outcome

        except asyncio.CancelledError:
            await self._emit_failure(
                run_id, current_status, "cancelled", terminal_outcome=terminal_outcome
            )
            raise
        except Exception as e:
            logger.exception("Company.run failed")
            cancelled = await self._emit_failure(
                run_id, current_status, str(e)[:500], terminal_outcome=terminal_outcome
            )
            if cancelled is not None:
                raise cancelled from e
            return terminal_outcome or RunOutcome.FAILED
        finally:
            if self._gateway is not None:
                try:
                    await self._gateway.close()
                except Exception:
                    logger.warning("gateway.close() failed during teardown", exc_info=True)

    async def _emit_budget_exhausted(self, run_id: str, usage: BudgetUsage) -> None:
        """Record that the run-level budget was crossed before abstaining.

        Emits ``BUDGET_UPDATED`` flagged ``exhausted=True`` so the timeline
        shows why the run abstained -- the bare ``OUTCOME_EMITTED=abstained``
        from the short-circuit path carries no reason.
        """
        await self._emit(run_id, BudgetUpdated(usage=usage, exhausted=True))

    async def _verify_and_finish(
        self,
        run_id: str,
        contract: WorkContract,
        usage: BudgetUsage,
        evidence_digests: dict[int, str],
        tools_used: list[str],
        current_status: RunStatus,
    ) -> tuple[RunOutcome, asyncio.CancelledError | None]:
        """Verification phase through the terminal outcome emission.

        Shared by ``run_async`` and ``resume_async``. Returns the decided
        outcome and the cancellation sentinel from the terminal emission; the
        caller owns the raise and the final status transition.
        """
        await self._emit(run_id, BudgetUpdated(usage=usage))

        await self._emit_status(run_id, RunStatus.VERIFYING, current_status)

        hard_results = await self._hard_verifier.verify(contract, evidence_digests, tools_used)

        soft_verdicts, material_disagreement = await self._soft_checker.check(
            contract,
            evidence_contents=await self._resolve_evidence_contents(evidence_digests),
        )
        soft_review = SoftReview(
            verdicts=soft_verdicts,
            material_disagreement=material_disagreement,
        )

        verdict = self._release_law.decide(
            contract=contract,
            hard=hard_results,
            soft=soft_review,
            usage=usage,
            tools_used=tools_used,
        )
        final_outcome = verdict.outcome

        await self._emit_evidence(
            run_id,
            verdict,
            hard_results,
            soft_review,
            usage,
            tools_used,
        )
        cancelled = await self._emit_terminal_outcome(run_id, final_outcome, verdict.reason)
        return final_outcome, cancelled

    async def resume_async(self, source_run_id: str, *, run_id: str | None = None) -> RunOutcome:
        """Continue a stopped or failed run under a fresh run id.

        Terminal statuses cannot transition, so a resume is a NEW run: the
        source's intake/contract/graph events are re-emitted under the new id
        (replay rebuilds an identical contract and graph), completed cells are
        seeded with their evidence, and only the remaining cells execute.
        Mid-turn conversations come from the on-disk cell checkpoints the
        ``CellRunner`` maintains under the workspace's ``.northstack/resume``
        tree.
        """
        state = replay_run(self._ledger, source_run_id)
        if state.current_contract is None or state.graph is None:
            raise ValueError(f"run {source_run_id} has no contract/graph to resume from")
        if state.status == RunStatus.VERIFIED:
            raise ValueError(f"run {source_run_id} is already verified")
        new_run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        request = ProjectRequest(
            goal=state.goal,
            workspace_root=state.workspace_root,
            budget=state.run_budget,
        )
        current_status = RunStatus.INTAKE
        terminal_outcome: RunOutcome | None = None
        try:
            await self._emit(
                new_run_id,
                RequestAccepted(
                    goal=state.goal,
                    workspace_root=state.workspace_root,
                    budget=state.run_budget,
                ),
            )
            source_completed = {
                event.payload.cell_id: event.payload
                for event in self._ledger.events(source_run_id)
                if isinstance(event.payload, CellCompleted)
            }
            completed_ids = {
                c.id for c in state.graph.cells if c.status == CellStatus.COMPLETED
            } | set(source_completed)
            for event in self._ledger.events(source_run_id):
                if event.payload.kind not in _RESUME_REPLAY_KINDS:
                    continue
                payload = event.payload
                if isinstance(payload, GraphAccepted):
                    payload = payload.model_copy(
                        update={
                            "cells": [
                                c.model_copy(update={"status": CellStatus.COMPLETED})
                                if c.id in completed_ids
                                else c
                                for c in payload.cells
                            ]
                        }
                    )
                await self._emit(new_run_id, payload)
            for event in self._ledger.events(source_run_id):
                if isinstance(event.payload, CellCompleted):
                    await self._emit(new_run_id, event.payload)
            for status in (RunStatus.CONTRACTED, RunStatus.PLANNED, RunStatus.EXECUTING):
                await self._emit_status(new_run_id, status, current_status)
                current_status = status

            usage = state.usage.model_copy()
            evidence_digests: dict[int, str] = {}
            tools_used: list[str] = []
            worker = self._worker_factory.create(self._workspace)
            budget_auth = BudgetAuthority(
                state.run_budget if state.run_budget is not None else Budget()
            )
            if usage.total_input_tokens or usage.total_output_tokens or usage.total_calls:
                budget_auth.record(
                    Spend(
                        input_tokens=usage.total_input_tokens,
                        output_tokens=usage.total_output_tokens,
                        cost_usd=usage.total_cost_usd,
                        calls=usage.total_calls,
                    )
                )
            for cell in state.graph.cells:
                ref = source_completed.get(cell.id)
                if cell.id in completed_ids and ref is not None:
                    for ci in cell.acceptance_criterion_indices:
                        evidence_digests[ci] = ref.output_artifact.digest

            stall_detector = StallDetector(
                window_seconds=self._config.run.stall_window_seconds,
                clock=self._stall_clock,
            )
            cell_outcome = await self._execute_cells(
                new_run_id,
                state.graph,
                state.current_contract,
                worker,
                usage,
                evidence_digests,
                tools_used,
                request,
                budget_auth,
                stall_detector,
                initially_completed=completed_ids,
            )
            if cell_outcome is not None:
                cell_verdict = Verdict(
                    outcome=cell_outcome,
                    reason="cell short-circuited the run before verification",
                )
                await self._emit_evidence(
                    new_run_id,
                    cell_verdict,
                    [],
                    SoftReview(),
                    usage,
                    tools_used,
                )
                cancelled = await self._emit_terminal_outcome(
                    new_run_id, cell_outcome, cell_verdict.reason
                )
                terminal_outcome = cell_outcome
                if cancelled is not None:
                    raise cancelled
                for nxt in RunStateMachine.route(current_status, _OUTCOME_TO_STATUS[cell_outcome]):
                    await self._emit_status(new_run_id, nxt, current_status)
                    current_status = nxt
                return cell_outcome

            final_outcome, cancelled = await self._verify_and_finish(
                new_run_id,
                state.current_contract,
                usage,
                evidence_digests,
                tools_used,
                current_status,
            )
            terminal_outcome = final_outcome
            if cancelled is not None:
                raise cancelled
            await self._emit_status(
                new_run_id, _OUTCOME_TO_STATUS[final_outcome], RunStatus.VERIFYING
            )
            return final_outcome
        except asyncio.CancelledError:
            await self._emit_failure(
                new_run_id, current_status, "cancelled", terminal_outcome=terminal_outcome
            )
            raise
        except Exception as e:
            logger.exception("Company.resume failed")
            cancelled = await self._emit_failure(
                new_run_id, current_status, str(e)[:500], terminal_outcome=terminal_outcome
            )
            if cancelled is not None:
                raise cancelled from e
            return terminal_outcome or RunOutcome.FAILED

    async def _execute_cells(
        self,
        run_id: str,
        graph: GraphVersion,
        contract: WorkContract,
        worker: WorkerProtocol,
        usage: BudgetUsage,
        evidence_digests: dict[int, str],
        tools_used: list[str],
        request: ProjectRequest,
        budget_auth: BudgetAuthority,
        stall_detector: StallDetector,
        initially_completed: set[str] | None = None,
    ) -> RunOutcome | None:
        completed_ids: set[str] = set(initially_completed or ())
        failed_ids: set[str] = set()

        for _wave in range(request.max_waves + 1):
            ready = self._scheduler.ready_cells(graph, completed_ids, failed_ids)
            if not ready:
                if not failed_ids and len(completed_ids) == len(graph.cells):
                    return None
                return RunOutcome.FAILED

            read_only_cells = [c for c in ready if c.mode != CellMode.MUTATING]
            mutating_cells = [c for c in ready if c.mode == CellMode.MUTATING]

            if read_only_cells:
                results = await self._run_with_stall_watchdog(
                    stall_detector,
                    asyncio.gather(
                        *(
                            self._run_cell(
                                run_id,
                                c,
                                contract,
                                worker,
                                usage,
                                evidence_digests,
                                tools_used,
                                budget_auth,
                                stall_detector,
                            )
                            for c in read_only_cells
                        ),
                        return_exceptions=True,
                    ),
                )
                if results is _STALLED:
                    await self._emit(run_id, StallDetected())
                    return RunOutcome.ABSTAINED
                for cell_obj, result in zip(read_only_cells, results, strict=True):
                    if isinstance(result, Exception):
                        logger.error(
                            "cell raised before completing cell_id=%s",
                            cell_obj.id,
                            exc_info=result,
                        )
                        failed_ids.add(cell_obj.id)
                    elif isinstance(result, RunOutcome):
                        return result
                    else:
                        completed_ids.add(cell_obj.id)
                if _budget_exhausted(budget_auth):
                    await self._emit_budget_exhausted(run_id, usage)
                    return RunOutcome.ABSTAINED

            for mc in mutating_cells:
                result = await self._run_with_stall_watchdog(
                    stall_detector,
                    self._run_cell(
                        run_id,
                        mc,
                        contract,
                        worker,
                        usage,
                        evidence_digests,
                        tools_used,
                        budget_auth,
                        stall_detector,
                    ),
                )
                if result is _STALLED:
                    await self._emit(run_id, StallDetected(cell_id=mc.id))
                    return RunOutcome.ABSTAINED
                if isinstance(result, RunOutcome):
                    return result
                completed_ids.add(mc.id)
                if _budget_exhausted(budget_auth):
                    await self._emit_budget_exhausted(run_id, usage)
                    return RunOutcome.ABSTAINED

        if len(completed_ids) != len(graph.cells):
            return RunOutcome.FAILED
        return None

    async def _run_cell(
        self,
        run_id: str,
        cell: GraphCell,
        contract: WorkContract,
        worker: WorkerProtocol,
        usage: BudgetUsage,
        evidence_digests: dict[int, str],
        tools_used: list[str],
        budget_auth: BudgetAuthority,
        stall_detector: StallDetector,
    ) -> RunOutcome | bool:
        runner = CellRunner(
            worker=worker,
            router=self._router,
            retry_policy=self._retry_policy,
            recovery=self._recovery,
            artifact_store=self._artifact_store,
            resume_dir=self._resume_dir,
            memory=self._memory,
            memory_namespace=self._config.name,
        )
        return await runner.run_cell(
            run_id=run_id,
            cell=cell,
            contract=contract,
            tool_defs=self._build_tool_defs(contract),
            usage=usage,
            evidence_digests=evidence_digests,
            tools_used=tools_used,
            budget_auth=budget_auth,
            emit=self._emit,
            heartbeat=stall_detector.heartbeat,
        )

    async def _run_with_stall_watchdog(
        self,
        stall_detector: StallDetector,
        coro: Coroutine[Any, Any, Any] | asyncio.Future[Any],
    ) -> Any:
        """Race ``coro`` against the stall detector.

        A concurrent watchdog polls ``is_stalled()`` and cancels the guarded
        coroutine when the run has been alive without a progress beat for longer
        than the configured window -- so a cell that hangs (no further
        heartbeat) is interrupted rather than pinning the run forever. Returns
        the coroutine's result, or the ``_STALLED`` sentinel when the watchdog
        tripped.

        The watchdog yields control on each poll (``await asyncio.sleep``), so
        even a hung cell that never yields (``await event.wait()``) is cancelled
        once the window elapses -- the watchdog runs on the same loop. With an
        injected clock the window is read off that clock, not wall time, so the
        detector is deterministic in tests. A window of 0 disables the detector:
        ``is_stalled()`` never returns True, the watchdog never cancels, and the
        guarded coroutine runs to completion (the operator opted out).
        """
        task = asyncio.ensure_future(coro)
        if stall_detector.is_stalled():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return _STALLED

        async def _watch() -> None:
            while not task.done():
                if stall_detector.is_stalled():
                    task.cancel()
                    return
                await asyncio.sleep(STALL_WATCHDOG_POLL_SECONDS)

        watcher = asyncio.ensure_future(_watch())
        try:
            return await task
        except asyncio.CancelledError:
            if not stall_detector.is_stalled():
                raise
            return _STALLED
        finally:
            if not watcher.done():
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher

    def _build_tool_defs(self, contract: WorkContract) -> list[ToolDefinition]:
        """Advertise the registry's tool definitions to the model, filtered to
        what the contract permits.

        The single declaration of every tool -- name, description, JSON schema,
        ``mutating`` flag and behaviour -- lives in :class:`ToolRegistry`. The
        orchestrator advertises what the registry declares; ``cmd_*``
        command-profile tools and ``web_fetch`` are advertised here, closing
        the dispatch/advertise gap.

        ``contract.allowed_tools`` is honoured at this seam: a tool absent from
        the allow-list is never advertised. An empty allow-list means "no
        restriction", so the full registry set is advertised. The worker does
        not re-filter -- advertisement is the single enforcement point, so a
        tool the contract did not grant cannot reach the model.
        """
        advertised = self._tool_registry.advertised()
        if not contract.allowed_tools:
            return advertised
        allowed = set(contract.allowed_tools)
        return [td for td in advertised if td.name in allowed]

    async def _resolve_evidence_contents(self, evidence_digests: dict[int, str]) -> dict[int, str]:
        """Resolve each criterion's evidence artifact to reviewable text.

        A missing artifact resolves to the empty string with a logged warning:
        the reviewer then judges absent evidence and fails closed through its
        own verdict, which the aggregation records as disagreement -- the
        abstention law stays intact and the gap stays visible in the log.
        """

        def _resolve(digest: str) -> str:
            try:
                return self._artifact_store.read_by_digest(digest).decode("utf-8", errors="replace")
            except FileNotFoundError:
                logger.warning("soft-review evidence artifact missing: %s", digest)
                return ""

        return {
            i: await asyncio.to_thread(_resolve, digest) for i, digest in evidence_digests.items()
        }

    async def _emit_evidence(
        self,
        run_id: str,
        verdict: Verdict,
        hard_results: Sequence[HardCheckResult],
        soft_review: SoftReview,
        usage: BudgetUsage,
        tools_used: list[str],
    ) -> None:
        hard_failures = [
            r for r in hard_results if not r.passed and r.kind != CriterionKind.SOFT_RUBRIC
        ]
        await self._emit(
            run_id,
            EvidenceRecorded(
                outcome=verdict.outcome,
                records=[
                    EvidenceRecord(
                        criterion_index=r.criterion_index,
                        kind=r.kind,
                        passed=r.passed,
                        evidence_artifact_digest=r.evidence_ref.digest if r.evidence_ref else "",
                        detail=r.detail,
                    )
                    for r in hard_results
                ],
                hard_gate_failures=[
                    HardGateFailure(index=r.criterion_index, detail=r.detail) for r in hard_failures
                ],
                usage=usage,
                tools_used=tools_used,
                material_disagreement=soft_review.material_disagreement,
            ),
        )

    async def _emit_status(self, run_id: str, status: RunStatus, current_status: RunStatus) -> None:
        if not RunStatus.can_transition(current_status, status):
            raise ValueError(f"Illegal status transition: {current_status.value} -> {status.value}")
        await self._emit(run_id, StatusChanged(status=status))

    async def _emit_failure(
        self,
        run_id: str,
        current_status: RunStatus,
        reason: str,
        *,
        terminal_outcome: RunOutcome | None = None,
    ) -> asyncio.CancelledError | None:
        """Best-effort terminal outcome+status; return cancellation after persistence."""

        async def persist() -> None:
            try:
                outcome = terminal_outcome or RunOutcome.FAILED
                if terminal_outcome is None:
                    await self._emit(run_id, OutcomeEmitted(outcome=outcome, reason=reason))
                await self._emit_status(run_id, _OUTCOME_TO_STATUS[outcome], current_status)
            except (ValueError, KeyError, OSError, RuntimeError, sqlite3.Error):
                logger.warning("failed to persist terminal run state", exc_info=True)

        return await _finish_despite_cancellation(asyncio.create_task(persist()))

    async def _emit(self, run_id: str, payload: EventPayload) -> None:
        await EventStream(self._ledger, run_id).emit_async(payload)

    async def _emit_terminal_outcome(
        self, run_id: str, outcome: RunOutcome, reason: str
    ) -> asyncio.CancelledError | None:
        return await _finish_despite_cancellation(
            asyncio.create_task(self._emit(run_id, OutcomeEmitted(outcome=outcome, reason=reason)))
        )


def inspect_run(ledger: Ledger, run_id: str) -> RunState:
    return replay_run(ledger, run_id)
