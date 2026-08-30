"""Run resume: mid-turn conversation checkpoints and resume_async.

Covers the two recoverability seams:
- the CellRunner persists the worker conversation to disk after each tool
  round and clears it on cell success, so a dead process leaves a checkpoint
  behind;
- ``Company.resume_async`` continues a failed run under a fresh run id:
  re-emitted contract/graph events rebuild the same plan, completed cells are
  seeded (never re-executed), and the interrupted cell resumes from its
  checkpoint instead of starting over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.wire import MessageRole, ModelMessage
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.application.contracting import (
    AcceptanceAnalysis,
    ContractCompiler,
    DeterministicAnalysisRunner,
    RequirementsAnalysis,
)
from northstack.application.orchestrator import Company
from northstack.application.replay import replay_run
from northstack.events.catalog import EventKind
from northstack.application.worker import WorkerResult
from northstack.config import ModelProfile, NorthStackConfig, Protocol, Role
from northstack.domain import (
    Budget,
    CellStatus,
    GraphCell,
    GraphVersion,
    ProjectRequest,
    RunOutcome,
)

from tests.test_orchestrator import CleanSynthesizer


class ScriptedWorker:
    """Worker that plays a per-cell script of steps on each ``run`` call.

    Steps:
        {"checkpoint": [messages]}  -> invoke on_checkpoint (a tool round
                                       completed; the CellRunner persists it)
        {"raise": "msg"}            -> crash the attempt mid-cell
        {"ok": "text"}              -> succeed; messages return for carry
    """

    def __init__(self, script: dict[str, list[dict[str, Any]]]) -> None:
        self._script = {k: list(v) for k, v in script.items()}
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
    ) -> Any:
        self.calls.append({"cell": cell.id, "resumed": resume_from_messages})
        # Consume steps until one resolves the attempt: a checkpoint step
        # only reports a completed tool round, it does not end the attempt.
        steps = self._script.get(cell.id, [])
        while True:
            # Exhausted script = crash: an unexpected worker call must not
            # silently succeed (that would mask a resume bug as a green run).
            step = steps.pop(0) if steps else {"raise": "unexpected worker call"}
            if "checkpoint" in step and on_checkpoint is not None:
                on_checkpoint(step["checkpoint"])
            if "raise" in step:
                raise RuntimeError(step["raise"])
            if "ok" in step:
                return WorkerResult(
                    ok=True,
                    text=step["ok"],
                    messages=step.get("messages", []),
                    total_input_tokens=10,
                    total_output_tokens=5,
                    total_cost_usd=0.0001,
                )


class _SharedWorkerFactory:
    def __init__(self, worker: ScriptedWorker) -> None:
        self._worker = worker

    def create(self, workspace: Any) -> ScriptedWorker:
        return self._worker


def _company(tmp_path: Path, worker: ScriptedWorker, planner: Any) -> Company:
    config = NorthStackConfig(
        name="test",
        profiles=[
            ModelProfile(
                name="worker",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost",
                model="test",
                roles={Role.WORKER},
                max_concurrency=1,
            )
        ],
    )
    runner = DeterministicAnalysisRunner(
        requirements=RequirementsAnalysis(scope="test", deliverables=["out.py"]),
        acceptance=AcceptanceAnalysis(
            criteria=[
                {
                    "kind": "command",
                    "description": "echo test",
                    "parameters": {"command_name": "echo_ok", "exit_code": 0},
                },
            ]
        ),
    )
    return Company(
        config=config,
        ledger=Ledger(path=tmp_path / "test.db"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        workspace=RestrictedWorkspace(tmp_path / "workspace"),
        gateway=None,
        worker_factory=_SharedWorkerFactory(worker),
        compiler=ContractCompiler(analysis_runner=runner, synthesizer=CleanSynthesizer()),
        command_profiles={
            "echo_ok": CommandProfile(name="echo_ok", argv=["python", "-c", "print('ok')"])
        },
        planner=planner,
    )


class _StubPlanner:
    """Plan seam stub returning one fixed graph."""

    def __init__(self, graph: GraphVersion) -> None:
        self._graph = graph

    async def plan(self, contract: Any, run_id: str) -> GraphVersion:
        return self._graph

    def validate(
        self, graph: GraphVersion, run_budget: Any = None, max_waves: int = 3
    ) -> list[str]:
        return []


def _single_cell_graph() -> GraphVersion:
    return GraphVersion(
        version=1,
        cells=[
            GraphCell(id="cell-a", name="a", wave=0, acceptance_criterion_indices=[0]),
        ],
        edges=[],
    )


def _two_cell_graph() -> GraphVersion:
    return GraphVersion(
        version=1,
        cells=[
            GraphCell(id="cell-a", name="a", wave=0, acceptance_criterion_indices=[0]),
            GraphCell(id="cell-b", name="b", wave=1, acceptance_criterion_indices=[0]),
        ],
        edges=[],
    )


def _msgs(*contents: str) -> list[Any]:
    return [ModelMessage(role=MessageRole.ASSISTANT, content=c) for c in contents]


def test_checkpoint_survives_process_restart_and_resume_mid_turn(tmp_path: Path) -> None:
    # Two crashing attempts (retry cap 1) -> FAILED with the conversation on disk.
    crash_script = {
        "cell-a": [
            {"checkpoint": _msgs("turn-1", "turn-2")},
            {"raise": "process died"},
            {"checkpoint": _msgs("turn-1", "turn-2", "turn-3")},
            {"raise": "process died again"},
        ]
    }
    worker_a = ScriptedWorker(crash_script)
    company_a = _company(tmp_path, worker_a, _StubPlanner(_single_cell_graph()))
    outcome = company_a.run(
        ProjectRequest(
            goal="Build feature", workspace_root=str(tmp_path), budget=Budget(max_retries=1)
        )
    )
    assert outcome == RunOutcome.FAILED

    run_id = company_a._ledger.list_runs()[0].run_id
    state = replay_run(company_a._ledger, run_id)
    assert state.status.value == "failed"

    # The crashed attempt left its conversation on disk.
    checkpoint = tmp_path / "workspace" / ".northstack" / "resume" / "cell-a.json"
    assert checkpoint.is_file()

    # A fresh process (new Company) resumes: the worker receives the
    # checkpointed conversation instead of a fresh prompt.
    worker_b = ScriptedWorker({"cell-a": [{"ok": "recovered"}]})
    company_b = _company(tmp_path, worker_b, _StubPlanner(_single_cell_graph()))
    import asyncio

    outcome = asyncio.run(company_b.resume_async(run_id))
    assert outcome == RunOutcome.VERIFIED
    assert worker_b.calls[0]["cell"] == "cell-a"
    assert [m.content for m in worker_b.calls[0]["resumed"]] == [
        "turn-1",
        "turn-2",
        "turn-3",
    ]

    # Success cleared the checkpoint.
    assert not checkpoint.exists()


def test_resume_skips_completed_cells(tmp_path: Path) -> None:
    fail_on_b = {"cell-a": [{"ok": "a done"}], "cell-b": [{"raise": "b crashed"}]}
    worker_a = ScriptedWorker(fail_on_b)
    company_a = _company(tmp_path, worker_a, _StubPlanner(_two_cell_graph()))
    outcome = company_a.run(
        ProjectRequest(
            goal="Build feature", workspace_root=str(tmp_path), budget=Budget(max_retries=1)
        )
    )
    assert outcome == RunOutcome.FAILED
    assert [c["cell"] for c in worker_a.calls] == ["cell-a", "cell-b", "cell-b"]

    run_id = company_a._ledger.list_runs()[0].run_id
    worker_b = ScriptedWorker({"cell-b": [{"ok": "b recovered"}]})
    company_b = _company(tmp_path, worker_b, _StubPlanner(_two_cell_graph()))
    import asyncio

    outcome = asyncio.run(company_b.resume_async(run_id))
    assert outcome == RunOutcome.VERIFIED

    # cell-a was completed in the source run: never re-executed, and the
    # replayed graph stamps it completed so projection (and UI) agree.
    assert [c["cell"] for c in worker_b.calls] == ["cell-b"]
    new_run_id = company_b._ledger.list_runs()[0].run_id
    graph_event = next(
        e
        for e in company_b._ledger.events(new_run_id)
        if e.payload.kind == EventKind.GRAPH_ACCEPTED
    )
    statuses = {c.id: c.status for c in graph_event.payload.cells}
    assert statuses["cell-a"] == CellStatus.COMPLETED
    completed_events = {
        e.payload.cell_id
        for e in company_b._ledger.events(new_run_id)
        if e.payload.kind == EventKind.CELL_COMPLETED
    }
    assert "cell-a" in completed_events
