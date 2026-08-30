"""Event model, ledger append_next, and replay reconstruction.

Split from test_control_plane.py along the events/projection seam.
Moved verbatim; no test rewritten or renamed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import RestrictedWorkspace
from northstack.application.contracting import (
    _criterion_from_dict,
)
from northstack.application.planning import GraphPlanner
from northstack.application.replay import replay_run
from northstack.config import (
    Capability,
    ModelProfile,
    NorthStackConfig,
    Protocol,
    Role,
    RunConfig,
)
from northstack.domain import (
    ArtifactRef,
    Budget,
    CommandCriterion,
    CriterionKind,
    FailureType,
    GraphCell,
    GraphVersion,
    ProjectRequest,
    RecoveryAction,
    RunOutcome,
    RunStatus,
    SoftRubricCriterion,
    WorkContract,
)
from northstack.domain.budget import Spend
from northstack.events.catalog import (
    CellCompleted,
    CellCreated,
    ContractProposed,
    EventKind,
    EvidenceRecorded,
    GraphAccepted,
    OutcomeEmitted,
    RecoveryTransition,
    RequestAccepted,
    RouteSelected,
    RunCreated,
    StatusChanged,
)

# CleanSynthesizer: passes through criteria/deliverables without defaults


class CleanSynthesizer:
    """Synthesizer that uses criteria and deliverables exactly as provided."""

    def synthesize(self, request, req_analysis, repo_analysis, acc_analysis, budget):
        # The typed union carries parameters as flat fields; flatten the legacy
        # ``{kind, description, parameters}`` dict the analysis emits before
        # building each criterion, exactly as the production synthesizer does.
        criteria = [_criterion_from_dict(c, i) for i, c in enumerate(acc_analysis.criteria)]
        deliverables = list(req_analysis.deliverables)
        return WorkContract(
            id=f"wc-clean-{int(time.time() * 1000)}",
            version=1,
            objective=request.goal,
            scope=req_analysis.scope,
            deliverables=deliverables,
            constraints=req_analysis.constraints + repo_analysis.conventions,
            assumptions=req_analysis.assumptions,
            allowed_tools=request.tool_policy,
            workspace_scope=repo_analysis.workspace_scope or request.workspace_root,
            budget=budget,
            acceptance_criteria=criteria,
            unresolved_ambiguity=req_analysis.ambiguities + acc_analysis.ambiguities,
            abstention_threshold=acc_analysis.recommended_abstention_threshold,
        )


# Fake test doubles


class FakeWorker:
    """Deterministic worker that returns predetermined results."""

    def __init__(
        self,
        results: dict[str, Any] | None = None,
        default_ok: bool = True,
        default_text: str = "output",
    ) -> None:
        self._results = results or {}
        self._default_ok = default_ok
        self._default_text = default_text
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
        self.calls.append(
            {
                "cell_id": cell.id,
                "profile_name": profile_name,
            }
        )

        key = f"{cell.id}:{profile_name}"
        if key in self._results:
            return self._results[key]

        from northstack.application.worker import WorkerResult

        return WorkerResult(
            ok=self._default_ok,
            text=self._default_text,
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost_usd=0.001,
        )


class FakeWorkerFactory:
    """Factory that returns a shared FakeWorker."""

    def __init__(self, worker: FakeWorker | None = None) -> None:
        self._worker = worker or FakeWorker()

    def create(self, workspace: Any) -> FakeWorker:
        return self._worker


class SequenceWorker(FakeWorker):
    """Returns a sequence of results, then fails if called unexpectedly."""

    def __init__(self, results: list[Any]) -> None:
        super().__init__()
        self._sequence = list(results)

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
        self.calls.append({"cell_id": cell.id, "profile_name": profile_name})
        if not self._sequence:
            raise AssertionError("worker called more times than expected")
        return self._sequence.pop(0)


class StaticGraphPlanner(GraphPlanner):
    """Returns a prebuilt graph for Company seam tests."""

    def __init__(self, graph: GraphVersion) -> None:
        self._graph = graph

    async def plan(self, contract: WorkContract, run_id: str) -> GraphVersion:
        return self._graph


class FailingPlanner(GraphPlanner):
    """Planner whose validate() always returns errors, triggering graph-validation abstain."""

    def __init__(self, graph: GraphVersion) -> None:
        self._graph = graph

    async def plan(self, contract: WorkContract, run_id: str) -> GraphVersion:
        return self._graph

    def validate(self, graph: GraphVersion) -> list[str]:
        return ["deliberate validation failure for testing"]


# Fixtures


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def ledger(tmp_db: Path) -> Ledger:
    return Ledger(path=tmp_db)


@pytest.fixture
def artifact_store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def workspace(tmp_path: Path) -> RestrictedWorkspace:
    return RestrictedWorkspace(tmp_path / "workspace")


@pytest.fixture
def sample_request() -> ProjectRequest:
    return ProjectRequest(
        goal="Implement a greeting function",
        workspace_root="/tmp/workspace",
    )


@pytest.fixture
def sample_config() -> NorthStackConfig:
    return NorthStackConfig(
        name="test",
        profiles=[
            ModelProfile(
                name="cheap-worker",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost:8080",
                model="test-cheap",
                roles={Role.WORKER},
                capabilities={Capability.TOOL_USE},
                max_concurrency=4,
                input_price_per_million_usd=0.5,
                output_price_per_million_usd=1.5,
            ),
            ModelProfile(
                name="expert-synth",
                protocol=Protocol.ANTHROPIC_MESSAGES,
                base_url="http://localhost:8081",
                model="test-expert",
                roles={Role.WORKER, Role.PLANNER, Role.REVIEWER},
                capabilities={Capability.TOOL_USE, Capability.NATIVE_JSON_SCHEMA},
                max_concurrency=1,
                input_price_per_million_usd=15.0,
                output_price_per_million_usd=75.0,
            ),
        ],
        run=RunConfig(
            default_budget_tokens=50_000,
            default_budget_cost_usd=10.0,
        ),
    )


@pytest.fixture
def sample_contract() -> WorkContract:
    return WorkContract(
        id="wc-test-1",
        version=1,
        objective="Implement a greeting function",
        scope="functions/ directory",
        deliverables=["greeting.py", "test_greeting.py"],
        constraints=["Python 3.12+"],
        allowed_tools=["read_file", "write_file", "run_command"],
        workspace_scope="/tmp/workspace",
        budget=Budget(token_limit=10_000, cost_limit_usd=2.0),
        acceptance_criteria=[
            CommandCriterion(
                description="pytest|tests pass",
                command_name="pytest",
                exit_code=0,
            ),
            SoftRubricCriterion(description="Code quality review"),
        ],
    )


# A) Event model and Ledger.append_next


class TestEventModel:
    def test_all_new_event_kinds_exist(self):
        expected = [
            "request_accepted",
            "workspace_snapshot",
            "analysis_requested",
            "analysis_completed",
            "contract_proposed",
            "contract_validated",
            "contract_amended",
            "graph_proposed",
            "graph_accepted",
            "route_selected",
            "cell_started",
            "cell_completed",
            "cell_failed",
            "evidence_recorded",
            "verification_check",
            "recovery_transition",
            "outcome_emitted",
        ]
        for name in expected:
            assert EventKind(name) is not None

    def test_criterion_kind_values(self):
        values = {k.value for k in CriterionKind}
        assert values == {
            "command",
            "file_diff",
            "tree_digest",
            "schema",
            "policy",
            "soft_rubric",
        }

    def test_acceptance_criterion_carries_typed_fields(self):
        c = CommandCriterion(description="run tests", command_name="pytest", exit_code=0)
        assert c.command_name == "pytest"
        assert c.exit_code == 0


class TestLedgerAppendNext:
    def test_append_next_genesis(self, ledger: Ledger):
        ev = ledger.append_next("run-1", RunCreated())
        assert ev.seq == 1
        assert ev.prev_hash == ""
        assert ev.hash_chain != ""

    def test_append_next_monotonic(self, ledger: Ledger):
        ev1 = ledger.append_next("run-1", RunCreated())
        ev2 = ledger.append_next("run-1", StatusChanged(status=RunStatus.INTAKE))
        ev3 = ledger.append_next(
            "run-1",
            ContractProposed(
                id="wc-1",
                version=1,
                objective="test",
                budget=Budget(),
                acceptance_criteria_count=0,
            ),
        )
        assert ev2.seq == 2
        assert ev2.prev_hash == ev1.hash_chain
        assert ev3.seq == 3
        assert ev3.prev_hash == ev2.hash_chain

    def test_append_next_hash_chain_integrity(self, ledger: Ledger):
        for _ in range(5):
            ledger.append_next("run-1", RunCreated())
        result = ledger.verify_integrity("run-1")
        assert result.ok
        assert result.events_checked == 5


# Replay state reconstruction


class TestReplayReconstruction:
    def test_replay_request_accepted(self, ledger: Ledger):
        ledger.append_next(
            "run-1",
            RequestAccepted(goal="Build API", workspace_root="/src"),
        )
        state = replay_run(ledger, "run-1")
        assert state.goal == "Build API"
        assert state.workspace_root == "/src"

    def test_replay_contract_proposed(self, ledger: Ledger):
        ledger.append_next(
            "run-1",
            ContractProposed(
                id="wc-1",
                version=2,
                objective="test",
                budget=Budget(token_limit=1000, cost_limit_usd=1.0),
                acceptance_criteria_count=0,
            ),
        )
        state = replay_run(ledger, "run-1")
        assert state.contract_version == 2

    def test_replay_graph_accepted(self, ledger: Ledger):
        ledger.append_next(
            "run-1",
            GraphAccepted(version=1, cells=[], edges=[], milestones=[]),
        )
        state = replay_run(ledger, "run-1")
        assert state.graph_version == 1

    def test_replay_route_selected(self, ledger: Ledger):
        ledger.append_next(
            "run-1",
            RouteSelected(cell_id="cell-1", profile_name="cheap-worker"),
        )
        state = replay_run(ledger, "run-1")
        assert state.routes["cell-1"] == "cheap-worker"

    def test_replay_outcome_emitted(self, ledger: Ledger):
        ledger.append_next("run-1", OutcomeEmitted(outcome=RunOutcome.VERIFIED))
        state = replay_run(ledger, "run-1")
        assert state.outcome == RunOutcome.VERIFIED

    def test_replay_recovery_transition(self, ledger: Ledger):
        ledger.append_next(
            "run-1",
            RecoveryTransition(
                cell_id="cell-1",
                failure_type=FailureType.TRANSIENT,
                action=RecoveryAction.BACKOFF_RETRY,
                attempt_number=0,
            ),
        )
        state = replay_run(ledger, "run-1")
        assert state.failure_type == FailureType.TRANSIENT
        assert any(e["action"] == "backoff_retry" for e in state.recovery_events)

    def test_replay_evidence_records_verified_flips_cell_status(self, ledger: Ledger):
        # A cell that completes then passes hard gates ends up at RunStatus.VERIFIED
        # in the legacy per-cell projection, not stuck at VERIFYING forever.
        ledger.append_next(
            "run-1",
            CellCreated(cell=GraphCell(id="cell-1", name="build")),
        )
        ledger.append_next(
            "run-1",
            CellCompleted(
                cell_id="cell-1",
                output_artifact=ArtifactRef(
                    digest="sha256:" + "a" * 64,
                    media_type="application/json",
                    size_bytes=1,
                ),
                usage=Spend(),
            ),
        )
        ledger.append_next(
            "run-1",
            EvidenceRecorded(outcome=RunOutcome.VERIFIED),
        )
        state = replay_run(ledger, "run-1")
        assert state.cells[0].status == RunStatus.VERIFIED

    def test_replay_recovery_events_lossless(self, ledger: Ledger):
        # Per-run recovery summary must carry cell + failure_type + attempt so
        # an operator can audit WHICH cell's WHICH attempt fired WHICH action.
        # The bare action-string list loses that correlation. Source of truth =
        # the literal payloads appended here; the projection must reproduce
        # them losslessly, not recompute the shape the way the emitter does.
        ledger.append_next(
            "run-1",
            RecoveryTransition(
                cell_id="cell-1",
                failure_type=FailureType.TRANSIENT,
                action=RecoveryAction.BACKOFF_RETRY,
                attempt_number=1,
            ),
        )
        ledger.append_next(
            "run-1",
            RecoveryTransition(
                cell_id="cell-1",
                failure_type=FailureType.BUDGET,
                action=RecoveryAction.SCOPE_REDUCTION,
                attempt_number=2,
            ),
        )
        state = replay_run(ledger, "run-1")
        assert state.recovery_events == [
            {
                "cell_id": "cell-1",
                "failure_type": "transient",
                "action": "backoff_retry",
                "attempt_number": 1,
            },
            {
                "cell_id": "cell-1",
                "failure_type": "budget",
                "action": "scope_reduction",
                "attempt_number": 2,
            },
        ]
