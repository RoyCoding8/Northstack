from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import ToolResult
from northstack.application.cell_runner import _recovery_sleep
from northstack.application.orchestrator import inspect_run
from northstack.application.projection_cache import ProjectionCache
from northstack.application.recovery import (
    AttemptDeduplicator,
    RecoveryManager,
)
from northstack.application.retry import RECOVERY_POLICY
from northstack.application.run_supervisor import RunSupervisor
from northstack.application.tools.registry import (
    _ListTool,
    _ReplaceTool,
    _SearchTool,
    _WebFetchTool,
    ToolRegistry,
)
from northstack.domain import AttemptSignature, Budget, FailureType, RecoveryAction
from northstack.domain.run_state import RunState
from northstack.events.catalog import (
    AnalysisRequested,
    ContractProposed,
    GraphAccepted,
    RequestAccepted,
    RunCreated,
    WorkspaceSnapshot,
)
from northstack.events.envelope import EventEnvelope

from tests.helpers.events import env


# GROUP P -- ProjectionCache


def _mixed_events(run_id: str, n: int) -> list[EventEnvelope]:
    """A list of n events mixing several catalog payload types."""
    payloads = [
        RequestAccepted(goal="g", workspace_root="/ws"),
        WorkspaceSnapshot(root="/ws", file_count=2),
        RunCreated(),
        ContractProposed(
            id="c", version=1, objective="o", budget=Budget(), acceptance_criteria_count=0
        ),
        GraphAccepted(version=2),
        AnalysisRequested(profile="p"),
        ContractProposed(
            id="c2", version=3, objective="o2", budget=Budget(), acceptance_criteria_count=0
        ),
    ]
    return [env(seq=i + 1, payload=payloads[i % len(payloads)], run_id=run_id) for i in range(n)]


def test_p1_cold_cache_state_is_none() -> None:
    cache = ProjectionCache()
    assert cache.state("r1") is None


def test_p2_state_returns_last_projected_object() -> None:
    cache = ProjectionCache()
    projected = cache.project("r1", None, [env(seq=1, run_id="r1")])
    assert cache.state("r1") is projected


def test_p3_incremental_fold_equals_full_fold() -> None:
    events = _mixed_events("r1", 7)

    inc = ProjectionCache()
    incremental_state = None
    for event in events:
        incremental_state = inc.project("r1", None, [event])

    full = ProjectionCache()
    full_state = full.project("r1", None, events)

    assert incremental_state == full_state


def test_p4_cursor_advances_by_tail_len_and_equals_n() -> None:
    events = _mixed_events("r1", 7)
    cache = ProjectionCache()

    cache.project("r1", None, events[:2])
    assert cache.cursor("r1") == 2
    cache.project("r1", None, events[2:5])
    assert cache.cursor("r1") == 5
    cache.project("r1", None, events[5:])
    assert cache.cursor("r1") == 7


def test_p5_empty_tail_is_noop_on_state_and_cursor() -> None:
    events = _mixed_events("r1", 3)
    cache = ProjectionCache()
    before = cache.project("r1", None, events)
    assert cache.cursor("r1") == 3

    after = cache.project("r1", None, [])
    assert after == before
    assert cache.cursor("r1") == 3


def test_p6_invalidate_resets_to_cold() -> None:
    events = _mixed_events("r1", 3)
    cache = ProjectionCache()
    cache.project("r1", None, events)
    cache.invalidate("r1")
    assert cache.state("r1") is None
    assert cache.cursor("r1") == 0


def test_p7_invalidate_unknown_run_id_does_not_raise() -> None:
    cache = ProjectionCache()
    cache.invalidate("never-seen")


def test_p8_two_runs_do_not_disturb_each_other() -> None:
    events = _mixed_events("shared", 4)
    cache = ProjectionCache()
    cache.project("a", None, events[:2])
    cache.project("b", None, events[:3])
    assert cache.cursor("a") == 2
    assert cache.cursor("b") == 3

    cache.invalidate("a")
    assert cache.state("a") is None
    assert cache.cursor("b") == 3


# GROUP S -- RunSupervisorHandle


def test_s1_run_id_property() -> None:
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=None)
    assert sup.run_id == "r1"


def test_s2_workspace_property() -> None:
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=None)
    assert sup.workspace == "/ws"


def test_s3_workspace_retained_after_release() -> None:
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=None)
    sup.release()
    assert sup.workspace == "/ws"


def test_s4_is_active_false_when_task_none() -> None:
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=None)
    assert sup.is_active is False


async def test_s5_is_active_true_for_unfinished_task() -> None:
    async def _t() -> None:
        await asyncio.sleep(0.05)

    task = asyncio.create_task(_t())
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=task)
    assert sup.is_active is True
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_s6_is_active_false_once_task_done() -> None:
    async def _t() -> int:
        return 1

    task = asyncio.create_task(_t())
    await task
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=task)
    assert sup.is_active is False


async def test_s7_is_active_false_after_release() -> None:
    async def _t() -> None:
        await asyncio.sleep(0.05)

    task = asyncio.create_task(_t())
    await asyncio.sleep(0)
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=task)
    sup.release()
    assert sup.is_active is False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_s8_heartbeat_calls_callback_once_per_call() -> None:
    calls: list[None] = []
    sup = RunSupervisor(
        run_id="r1", ledger=None, workspace="/ws", task=None, heartbeat=lambda: calls.append(None)
    )
    sup.heartbeat()
    sup.heartbeat()
    assert len(calls) == 2


def test_s9_heartbeat_with_none_does_not_raise() -> None:
    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=None)
    sup.heartbeat()


def test_s10_release_is_idempotent_closes_ledger_once() -> None:
    class _StubLedger:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    ledger = _StubLedger()
    sup = RunSupervisor(run_id="r1", ledger=ledger, workspace="/ws", task=None)

    sup.release()
    sup.release()
    sup.release()

    assert ledger.close_calls == 1
    assert sup.ledger is None
    assert sup.task is None


def test_s11_release_completes_when_ledger_close_raises() -> None:
    class _RaisingLedger:
        def close(self) -> None:
            raise RuntimeError("disk gone")

    sup = RunSupervisor(run_id="r1", ledger=_RaisingLedger(), workspace="/ws", task=None)
    sup.release()
    assert sup.released is True


async def test_s12_bind_task_on_released_handle_cancels_task() -> None:
    async def _noop() -> None:
        return

    sup = RunSupervisor(run_id="r1", ledger=None, workspace="/ws", task=None)
    sup.release()
    assert sup.released is True

    new_task = asyncio.create_task(_noop())
    sup.bind_task(new_task)
    try:
        await new_task
    except asyncio.CancelledError:
        pass
    assert new_task.cancelled()


# GROUP T -- tool registry


class _RecordingWorkspace:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[object, ...]] = {}

    def read(self, rel_path: str) -> ToolResult:
        self.calls["read"] = (rel_path,)
        return ToolResult(ok=True, operation="read")

    def list(self, rel_path: str) -> ToolResult:
        self.calls["list"] = (rel_path,)
        return ToolResult(ok=True, operation="list")

    def search(self, pattern: str, path: str = ".") -> ToolResult:
        self.calls["search"] = (pattern, path)
        return ToolResult(ok=True, operation="search")

    def write(self, rel_path: str, content: bytes, lease: str | None = None) -> ToolResult:
        self.calls["write"] = (rel_path, content, lease)
        return ToolResult(ok=True, operation="write")

    def create(self, rel_path: str, content: bytes, lease: str | None = None) -> ToolResult:
        self.calls["create"] = (rel_path, content, lease)
        return ToolResult(ok=True, operation="create")

    def replace(self, rel_path: str, old: str, new: str, lease: str | None = None) -> ToolResult:
        self.calls["replace"] = (rel_path, old, new, lease)
        return ToolResult(ok=True, operation="replace")


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        workspace=_RecordingWorkspace(),
        web_reader=None,
        command_profiles={},
        lease="LEASE-123",
    )


async def test_t1_list_tool_delegates_to_workspace_list() -> None:
    ctx = _ctx()
    tool = _ListTool()
    await tool.execute(ctx, {"path": "/sub"})
    assert ctx.workspace.calls["list"] == ("/sub",)


async def test_t2_list_tool_defaults_path_to_dot() -> None:
    ctx = _ctx()
    tool = _ListTool()
    await tool.execute(ctx, {})
    assert ctx.workspace.calls["list"] == (".",)


async def test_t3_search_tool_delegates_and_defaults() -> None:
    ctx = _ctx()
    tool = _SearchTool()
    await tool.execute(ctx, {"pattern": "*.py", "path": "/d"})
    assert ctx.workspace.calls["search"] == ("*.py", "/d")

    await tool.execute(ctx, {})
    assert ctx.workspace.calls["search"] == ("", ".")


async def test_t4_replace_tool_forwards_lease() -> None:
    ctx = _ctx()
    tool = _ReplaceTool()
    await tool.execute(ctx, {"path": "/f", "old": "a", "new": "b"})
    assert ctx.workspace.calls["replace"] == ("/f", "a", "b", "LEASE-123")


async def test_t5_web_fetch_without_reader_returns_error() -> None:
    ctx = _ctx()
    ctx.web_reader = None
    tool = _WebFetchTool()
    result = await tool.execute(ctx, {"url": "http://x"})
    assert result.ok is False
    assert result.operation == "web_fetch"
    assert result.error


async def test_t6_web_fetch_delegates_to_reader() -> None:
    class _Reader:
        def fetch(self, url: str, method: str = "GET") -> ToolResult:
            self.called = (url,)
            return ToolResult(ok=True, operation="web_fetch")

    ctx = _ctx()
    reader = _Reader()
    ctx.web_reader = reader
    tool = _WebFetchTool()
    await tool.execute(ctx, {"url": "http://y"})
    assert reader.called == ("http://y",)


def test_t7_names_match_advertised_order() -> None:
    registry = ToolRegistry.with_defaults(command_profiles={})
    assert registry.names() == [td.name for td in registry.advertised()]


def test_t8_parity_invariant_holds() -> None:
    registry = ToolRegistry.with_defaults(command_profiles={})
    assert (
        set(registry.names())
        == registry.dispatchable_names()
        == {d.name for d in registry.advertised()}
    )


# GROUP R -- recovery


def _sig() -> AttemptSignature:
    return AttemptSignature(
        contract_version=1,
        cell_id="c1",
        profile_name="p",
        strategy_id="s",
        tool_plan="tp",
        evidence_digest="",
    )


def test_r1_clear_makes_signature_not_duplicate() -> None:
    dedup = AttemptDeduplicator()
    sig = _sig()
    dedup.record("r1", sig)
    assert dedup.is_duplicate("r1", sig) is True
    dedup.clear("r1")
    assert dedup.is_duplicate("r1", sig) is False


def test_r2_clear_is_scoped_per_run() -> None:
    dedup = AttemptDeduplicator()
    sig = _sig()
    dedup.record("a", sig)
    dedup.record("b", sig)
    dedup.clear("a")
    assert dedup.is_duplicate("a", sig) is False
    assert dedup.is_duplicate("b", sig) is True


def test_r3_clear_unknown_run_id_does_not_raise() -> None:
    dedup = AttemptDeduplicator()
    dedup.clear("never-seen")


def test_r4_allowed_actions_returns_policy_list() -> None:
    manager = RecoveryManager()
    assert manager.allowed_actions(FailureType.TRANSIENT) == RECOVERY_POLICY[FailureType.TRANSIENT]


def test_r5_allowed_actions_returns_a_copy() -> None:
    """The snapshot is taken before the mutation, so aliasing cannot hide behind it."""
    manager = RecoveryManager()
    expected = list(manager.allowed_actions(FailureType.TRANSIENT))
    manager.allowed_actions(FailureType.TRANSIENT).append(RecoveryAction.TERMINATE)
    assert manager.allowed_actions(FailureType.TRANSIENT) == expected
    assert RECOVERY_POLICY[FailureType.TRANSIENT] == expected


def test_r6_allowed_actions_absent_type_returns_terminate() -> None:
    manager = RecoveryManager(policy={FailureType.SAFETY: [RecoveryAction.ABSTAIN]})
    assert manager.allowed_actions(FailureType.TRANSIENT) == [RecoveryAction.TERMINATE]


# GROUP C -- cell_runner._recovery_sleep


async def test_c1_recovery_sleep_completes_and_returns_none() -> None:
    result = await _recovery_sleep(0)
    assert result is None


async def test_c2_recovery_sleep_yields_the_loop() -> None:
    """Ordering only holds if the sleep suspends: a blocking sleep would finish first."""
    events: list[str] = []

    async def _sleeper() -> None:
        await _recovery_sleep(0.05)
        events.append("sleep")

    async def _sibling() -> None:
        events.append("sibling")

    await asyncio.gather(_sleeper(), _sibling())
    assert events == ["sibling", "sleep"]


# GROUP O -- orchestrator.inspect_run


def test_o1_inspect_run_returns_run_state_with_matching_run_id(tmp_path: Path) -> None:
    ledger = Ledger(path=tmp_path / "ledger.db")
    try:
        ledger.append_next("r1", RunCreated())
        state = inspect_run(ledger, "r1")
    finally:
        ledger.close()
    assert isinstance(state, RunState)
    assert state.run_id == "r1"
