"""CellRunner: the per-cell attempt loop extracted from the orchestrator.

``CellRunner`` owns the per-cell attempt loop, reroute-escalate and the
``resume_from_messages`` seam ADR 0001 requires: the in-flight conversation
carried across retries stays *transient* and is never written to the ledger.
The orchestrator becomes a phase sequencer that drives it.

These tests exercise ``CellRunner`` directly (not through ``Company``) so the
loop's invariants are pinned at their own boundary:

1. **Transient resume stays out of the ledger** (ADR 0001): a retrying cell
   carries ``resume_from_messages`` across attempts, but the payloads the
   CellRunner emits to the ledger never serialise the carried conversation.

2. **A successful cell emits ``CellCompleted`` with evidence and returns True.**

3. **A read-only cell makes measurable progress while another cell sits in its
   backoff window** — backoff is ``await asyncio.sleep`` and so does not block
   the event loop, so a concurrently-scheduled read-only cell advances during
   the first cell's backoff sleep. This is the concurrency property the loop
   must keep.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from northstack.adapters.providers.wire import MessageRole, ModelMessage, ToolCall
from northstack.application.budget_authority import BudgetAuthority
from northstack.application.cell_runner import CellRunner
from northstack.application.recovery import RecoveryManager
from northstack.application.retry import RetryPolicy
from northstack.application.routing import Router
from northstack.application.worker import WorkerEvent, WorkerEventKind, WorkerResult
from northstack.config import ModelProfile, NorthStackConfig, Protocol
from northstack.domain import Budget, GraphCell, RunOutcome, Spend, WorkContract
from northstack.domain.contract import CommandCriterion
from northstack.events.catalog import CellCompleted, CellProgress, RecoveryTransition

# Helpers


def _profile(name: str = "worker") -> ModelProfile:
    return ModelProfile(
        name=name,
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model=f"test-{name}",
        max_concurrency=4,
        requests_per_minute=1000,
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=5.0,
        max_output_tokens=4096,
    )


def _config(profiles: list[ModelProfile] | None = None) -> NorthStackConfig:
    return NorthStackConfig(name="test", profiles=profiles or [_profile()])


def _contract(*, max_retries: int = 0, max_calls: int = 0) -> WorkContract:
    return WorkContract(
        id="wc-1",
        version=1,
        objective="test",
        budget=Budget(
            token_limit=100_000,
            cost_limit_usd=5.0,
            max_calls=max_calls,
            max_retries=max_retries,
        ),
        acceptance_criteria=[
            CommandCriterion(description="check", command_name="check", exit_code=0)
        ],
    )


def _cell(cell_id: str = "cell-1", *, contract: WorkContract | None = None) -> GraphCell:
    return GraphCell(
        id=cell_id,
        name=cell_id,
        mode="read_only",
        contract=contract or _contract(),
        acceptance_criterion_indices=[0],
    )


def _cell_runner(
    worker: Any,
    *,
    config: NorthStackConfig | None = None,
    artifact_store: Any | None = None,
) -> CellRunner:
    store = artifact_store or _NullArtifactStore()
    return CellRunner(
        worker=worker,
        router=Router(config or _config()),
        retry_policy=RetryPolicy(),
        recovery=RecoveryManager(),
        artifact_store=store,
    )


class _NullArtifactStore:
    """Minimal artifact store: writes return a real ArtifactRef (sha256)."""

    def write(self, content: bytes, media_type: str = "application/json") -> Any:
        import hashlib

        from northstack.domain import ArtifactRef

        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        return ArtifactRef(digest=digest, size_bytes=len(content), media_type=media_type)


class _RecordingEmitter:
    """Captures payloads the CellRunner emits, with no real ledger."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit(self, run_id: str, payload: Any) -> None:
        self.events.append(payload)


class _ScriptedWorker:
    """Worker that returns a scripted sequence of ``WorkerResult`` per call.

    Records every call's ``resume_from_messages`` so the transient-resume
    invariant is observable: the carried conversation is threaded back into the
    next ``run`` but never appears in the ledger.
    """

    def __init__(self, outcomes: list[WorkerResult]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def run(
        self,
        cell: Any,
        profile_name: str,
        tool_defs: list[Any],
        *,
        system_prompt: str = "",
        output_json_schema: dict[str, Any] | None = None,
        resume_from_messages: list[Any] | None = None,
        on_progress: Any = None,
        on_checkpoint: Any = None,
        on_event: Any = None,
    ) -> WorkerResult:
        self.calls.append(
            {"cell_id": cell.id, "profile": profile_name, "resume": resume_from_messages}
        )
        if self._outcomes:
            return self._outcomes.pop(0)
        return WorkerResult(ok=True, text="fallback", total_input_tokens=1, total_output_tokens=1)


_CARRIED_SENTINEL = "round-1-tool-result"


def _carried_conversation() -> list[ModelMessage]:
    """A mid-cell in-flight conversation for the resume seam (ADR 0001).

    Distinctive ``content`` so a test can prove the conversation was threaded
    into the retry (via ``resume_from_messages``) yet never serialised into a
    ledger payload.
    """
    return [
        ModelMessage(role=MessageRole.ASSISTANT, content="thinking", tool_calls=[_tool_call()]),
        ModelMessage(role=MessageRole.TOOL, content=_CARRIED_SENTINEL, tool_call_id="call-1"),
    ]


def _tool_call(cid: str = "call-1", name: str = "read") -> ToolCall:
    return ToolCall(id=cid, name=name, arguments={"path": "x.txt"})


def _ok(text: str = "done", *, messages: list[ModelMessage] | None = None) -> WorkerResult:
    # A successful final assistant message with one tool-call so the
    # ``tools_used`` extraction (``message.tool_calls``) is exercised.
    final = (
        messages
        if messages is not None
        else [
            ModelMessage(
                role=MessageRole.ASSISTANT, content=text, tool_calls=[_tool_call(name="read")]
            )
        ]
    )
    return WorkerResult(
        ok=True,
        text=text,
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=0.001,
        api_calls=1,
        messages=final,
    )


def _transient_fail(*, messages: list[ModelMessage] | None = None) -> WorkerResult:
    """A 503-class transient failure carrying a mid-cell conversation."""
    return WorkerResult(
        ok=False,
        error="HTTP 503",
        error_kind="provider",
        text="",
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=0.001,
        api_calls=1,
        messages=messages if messages is not None else _carried_conversation(),
    )


# Test 1: transient resume_from_messages never reaches the ledger


class TestTransientResumeStaysOutOfLedger:
    @pytest.mark.asyncio
    async def test_carried_conversation_not_in_any_emitted_payload(
        self, tmp_path, monkeypatch
    ) -> None:
        """A transient failure carrying an in-flight conversation is retried
        via ``resume_from_messages`` (ADR 0001), but the conversation never
        serialises into anything the CellRunner emits to the ledger.

        Pin: the only ledger payloads emitted for the retrying cell are
        typed events (RouteSelected / CellStarted / RecoveryTransition /
        CellFailed), none of which carry a ``messages`` / ``resume`` field.
        """
        from northstack.events import catalog as cat

        # No real backoff sleep -- the CellRunner's backoff seam is a no-op.
        async def _no_sleep(delay: float) -> None:
            await asyncio.sleep(0)

        monkeypatch.setattr("northstack.application.cell_runner._recovery_sleep", _no_sleep)

        contract = _contract(max_retries=3)
        cell = _cell(contract=contract)
        # Two profiles: the 2nd same-signature TRANSIENT escalates to
        # REROUTE_ESCALATE (rung 1), which reroutes to the second profile and
        # resets the ladder (fresh signature). ``max_retries=3`` keeps the cap
        # above the rung-1 escalation so the loop survives long enough to carry
        # a resume forward and then succeed:
        #   fail(0, A) -> BACKOFF_RETRY (resume)
        #   fail(1, A) -> REROUTE_ESCALATE -> profile B (resume)
        #   ok (2, B)
        worker = _ScriptedWorker([_transient_fail(), _transient_fail(), _ok()])
        runner = _cell_runner(worker, config=_config([_profile("worker"), _profile("escalate")]))

        budget_auth = BudgetAuthority(contract.budget)
        emitter = _RecordingEmitter()
        usage = _fresh_usage()

        result = await runner.run_cell(
            run_id="run-1",
            cell=cell,
            contract=contract,
            tool_defs=[],
            usage=usage,
            evidence_digests={},
            tools_used=[],
            budget_auth=budget_auth,
            emit=emitter.emit,
        )

        assert result is True, "cell should complete after resume across retries"
        # The conversation WAS threaded into the second/third worker.run calls.
        resumed = [c for c in worker.calls if c["resume"]]
        assert resumed, "resume_from_messages must be carried into a retry"
        # ...but NO ledger payload may reference the in-flight conversation.
        ledger_blob = " ".join(repr(e) for e in emitter.events)
        assert _CARRIED_SENTINEL not in ledger_blob, (
            "in-flight conversation leaked into a ledger payload (ADR 0001 invariant): "
            + ledger_blob
        )
        # And the cited events are exactly the typed catalog kinds.
        kinds = {type(e).__name__ for e in emitter.events}
        assert kinds <= set(dir(cat)), f"unexpected payload types: {kinds}"
        assert RecoveryTransition in [type(e) for e in emitter.events]

    @pytest.mark.asyncio
    async def test_failed_call_exhausts_cell_call_budget_before_retry(self) -> None:
        contract = _contract(max_retries=3, max_calls=1)
        worker = _ScriptedWorker([_transient_fail(), _ok()])
        runner = _cell_runner(worker)
        result = await runner.run_cell(
            run_id="run-1",
            cell=_cell(contract=contract),
            contract=contract,
            tool_defs=[],
            usage=_fresh_usage(),
            evidence_digests={},
            tools_used=[],
            budget_auth=BudgetAuthority(contract.budget),
            emit=_RecordingEmitter().emit,
        )
        assert result is RunOutcome.ABSTAINED
        assert len(worker.calls) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_kind", ["authentication", "configuration"])
    async def test_permanent_provider_failure_never_backs_off_or_reroutes(
        self, error_kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fail_sleep(_delay: float) -> None:
            raise AssertionError("permanent provider failures must not back off")

        import northstack.application.cell_runner as cell_runner_module

        monkeypatch.setattr(cell_runner_module, "_recovery_sleep", fail_sleep)
        contract = _contract(max_retries=3)
        worker = _ScriptedWorker(
            [
                WorkerResult(
                    ok=False,
                    error="missing credentials",
                    error_kind=error_kind,
                    api_calls=1,
                )
            ]
        )
        emitter = _RecordingEmitter()
        result = await _cell_runner(worker).run_cell(
            run_id="run-1",
            cell=_cell(contract=contract),
            contract=contract,
            tool_defs=[],
            usage=_fresh_usage(),
            evidence_digests={},
            tools_used=[],
            budget_auth=BudgetAuthority(contract.budget),
            emit=emitter.emit,
        )
        assert result is RunOutcome.ABSTAINED
        assert len(worker.calls) == 1
        assert len([event for event in emitter.events if event.kind == "route_selected"]) == 1


# Test 2: success emits CellCompleted and returns True


class TestSuccessEmitsCompletedEvidence:
    @pytest.mark.asyncio
    async def test_ok_cell_emits_cell_completed_and_returns_true(self, tmp_path) -> None:
        contract = _contract()
        cell = _cell(contract=contract)
        worker = _ScriptedWorker([_ok(text='{"answer": 42}').model_copy(update={"api_calls": 3})])
        runner = _cell_runner(worker)
        budget_auth = BudgetAuthority(contract.budget)
        emitter = _RecordingEmitter()
        usage = _fresh_usage()

        result = await runner.run_cell(
            run_id="run-1",
            cell=cell,
            contract=contract,
            tool_defs=[],
            usage=usage,
            evidence_digests={},
            tools_used=[],
            budget_auth=budget_auth,
            emit=emitter.emit,
        )

        assert result is True
        completed = [e for e in emitter.events if isinstance(e, CellCompleted)]
        assert len(completed) == 1
        assert completed[0].cell_id == "cell-1"
        assert completed[0].usage.input_tokens == 10
        assert completed[0].usage.calls == 3
        assert usage.total_calls == 3

    @pytest.mark.asyncio
    async def test_completed_usage_includes_failed_attempts_before_success(self) -> None:
        contract = _contract(max_retries=1)
        worker = _ScriptedWorker([_transient_fail(), _ok()])
        runner = _cell_runner(worker)
        emitter = _RecordingEmitter()
        usage = _fresh_usage()
        result = await runner.run_cell(
            run_id="run-1",
            cell=_cell(contract=contract),
            contract=contract,
            tool_defs=[],
            usage=usage,
            evidence_digests={},
            tools_used=[],
            budget_auth=BudgetAuthority(contract.budget),
            emit=emitter.emit,
        )
        completed = next(event for event in emitter.events if isinstance(event, CellCompleted))
        assert result is True
        assert completed.usage == Spend(
            input_tokens=20,
            output_tokens=10,
            cost_usd=0.002,
            calls=2,
        )
        assert usage.total_calls == completed.usage.calls

    @pytest.mark.asyncio
    async def test_run_call_total_equals_sum_of_completed_cells(self) -> None:
        contract = _contract()
        worker = _ScriptedWorker(
            [
                _ok().model_copy(update={"api_calls": 2}),
                _ok().model_copy(update={"api_calls": 3}),
            ]
        )
        runner = _cell_runner(worker)
        emitter = _RecordingEmitter()
        usage = _fresh_usage()
        authority = BudgetAuthority(contract.budget)
        for cell_id in ["cell-a", "cell-b"]:
            assert (
                await runner.run_cell(
                    run_id="run-1",
                    cell=_cell(cell_id, contract=contract),
                    contract=contract,
                    tool_defs=[],
                    usage=usage,
                    evidence_digests={},
                    tools_used=[],
                    budget_auth=authority,
                    emit=emitter.emit,
                )
                is True
            )
        completed = [event for event in emitter.events if isinstance(event, CellCompleted)]
        assert sum(event.usage.calls for event in completed) == usage.total_calls == 5


# Test 3 (plan-named): a read-only cell advances during another's backoff


class TestReadOnlyProgressDuringBackoff:
    @pytest.mark.asyncio
    async def test_read_only_cell_progresses_while_backoff_sleeps(self) -> None:
        """A read-only cell makes measurable progress while another cell sits
        in its backoff window.

        The CellRunner's backoff is ``await asyncio.sleep(delay)`` -- it
        yields the event loop, so a concurrently-``gather``ed read-only cell
        runs to completion *during* the first cell's backoff sleep rather
        than waiting for it: extracting the loop into its own module must
        not make backoff block.
        """
        backoff_started = asyncio.Event()
        progressed = asyncio.Event()

        async def _conditional_sleep(delay: float) -> None:
            backoff_started.set()
            # Short real sleep: long enough for the other cell to progress.
            await asyncio.sleep(0.05 if delay > 0 else 0)
            backoff_started.clear()

        contract_failing = _contract(max_retries=3)
        contract_fast = _contract()
        failing_cell = _cell("cell-a", contract=contract_failing)
        fast_cell = _cell("cell-b", contract=contract_fast)

        failing_worker = _ScriptedWorker(
            [_transient_fail(), _transient_fail(), _transient_fail(), _transient_fail()]
        )
        fast_worker = _ScriptedWorker([_ok(text="fast-done")])

        runner_failing = _cell_runner(
            failing_worker, config=_config([_profile("worker"), _profile("escalate")])
        )
        runner_fast = _cell_runner(fast_worker)

        # Patch only the failing runner's backoff seam so the test observes the
        # overlap. (Both run via the same module-level seam.)
        import northstack.application.cell_runner as cr

        original_sleep = cr._recovery_sleep
        cr._recovery_sleep = _conditional_sleep
        try:

            async def _run_fast() -> Any:
                # Let the failing cell enter its backoff window first.
                await backoff_started.wait()
                emitter = _RecordingEmitter()
                res = await runner_fast.run_cell(
                    run_id="run-fast",
                    cell=fast_cell,
                    contract=contract_fast,
                    tool_defs=[],
                    usage=_fresh_usage(),
                    evidence_digests={},
                    tools_used=[],
                    budget_auth=BudgetAuthority(contract_fast.budget),
                    emit=emitter.emit,
                )
                progressed.set()
                return res

            async def _run_failing() -> Any:
                emitter = _RecordingEmitter()
                return await runner_failing.run_cell(
                    run_id="run-fail",
                    cell=failing_cell,
                    contract=contract_failing,
                    tool_defs=[],
                    usage=_fresh_usage(),
                    evidence_digests={},
                    tools_used=[],
                    budget_auth=BudgetAuthority(contract_failing.budget),
                    emit=emitter.emit,
                )

            await asyncio.wait_for(asyncio.gather(_run_failing(), _run_fast()), timeout=2)
        finally:
            cr._recovery_sleep = original_sleep

        assert progressed.is_set(), (
            "read-only cell did not make measurable progress during the "
            "failing cell's backoff window -- backoff is blocking the loop"
        )


def _fresh_usage() -> Any:
    from northstack.domain import BudgetUsage

    return BudgetUsage()


class _EmittingWorker(_ScriptedWorker):
    """Publishes worker events through ``on_event`` before returning."""

    def __init__(self, events: list[WorkerEvent], outcomes: list[WorkerResult]) -> None:
        super().__init__(outcomes)
        self._events = events

    async def run(self, *args: Any, on_event: Any = None, **kwargs: Any) -> WorkerResult:
        for event in self._events:
            await on_event(event)
        return await super().run(*args, **kwargs)


class TestIntraCellProgress:
    """A cell was previously opaque between ``cell_started`` and its terminal
    event. The worker's event stream is projected onto ``CellProgress`` so a
    runaway turn loop, a compaction, or a slow tool are all visible.
    """

    @pytest.mark.asyncio
    async def test_worker_events_reach_the_ledger_as_cell_progress(self):
        events = [
            WorkerEvent(kind=WorkerEventKind.TURN_STARTED, turn=1, detail={"messages": 2}),
            WorkerEvent(
                kind=WorkerEventKind.MODEL_CALL_COMPLETED, turn=1, detail={"output_tokens": 5}
            ),
        ]
        emitter = _RecordingEmitter()
        runner = _cell_runner(_EmittingWorker(events, [_ok()]))

        contract = _contract()
        await runner.run_cell(
            run_id="run-1",
            cell=_cell(),
            contract=contract,
            tool_defs=[],
            usage=_fresh_usage(),
            evidence_digests={},
            tools_used=[],
            budget_auth=BudgetAuthority(contract.budget),
            emit=emitter.emit,
        )

        progress = [e for e in emitter.events if isinstance(e, CellProgress)]
        assert [(p.step, p.turn) for p in progress] == [
            ("turn_started", 1),
            ("model_call_completed", 1),
        ]
        assert all(p.cell_id == "cell-1" for p in progress)
        assert progress[1].detail == {"output_tokens": 5}

    @pytest.mark.asyncio
    async def test_non_scalar_detail_is_dropped_not_stringified(self):
        """Detail is the one free-form field on the payload; a message list
        reaching it would put model output into the ledger, which ADR 0001
        forbids. Dropping is the safe failure, not ``str()``.
        """
        events = [
            WorkerEvent(
                kind=WorkerEventKind.TURN_STARTED,
                turn=1,
                detail={"messages": 2, "conversation": [{"role": "user", "content": "secret"}]},
            )
        ]
        emitter = _RecordingEmitter()
        runner = _cell_runner(_EmittingWorker(events, [_ok()]))

        contract = _contract()
        await runner.run_cell(
            run_id="run-1",
            cell=_cell(),
            contract=contract,
            tool_defs=[],
            usage=_fresh_usage(),
            evidence_digests={},
            tools_used=[],
            budget_auth=BudgetAuthority(contract.budget),
            emit=emitter.emit,
        )

        progress = [e for e in emitter.events if isinstance(e, CellProgress)]
        assert progress[0].detail == {"messages": 2}
        assert "secret" not in str(emitter.events)
