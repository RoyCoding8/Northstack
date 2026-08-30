"""Company pipeline, run supervisor, budget enforcement, and stall seam.

Split from test_control_plane.py along the orchestrator seam.
Moved verbatim; no test rewritten or renamed.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.application.contracting import (
    AcceptanceAnalysis,
    ContractCompiler,
    DeterministicAnalysisRunner,
    RepoAnalysis,
    RequirementsAnalysis,
    _criterion_from_dict,
)
from northstack.application.orchestrator import Company
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
    Budget,
    CommandCriterion,
    GraphCell,
    GraphEdge,
    GraphVersion,
    ProjectRequest,
    RunOutcome,
    RunStatus,
    SoftRubricCriterion,
    WorkContract,
)
from northstack.events.catalog import (
    EventKind,
    OutcomeEmitted,
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

    def validate(
        self,
        graph: GraphVersion,
        *,
        run_budget: Budget | None = None,
        max_waves: int | None = None,
    ) -> list[str]:
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


# G) Pipeline end-to-end


class TestCompanyPipeline:
    def test_executes_dependent_graph_cells_once(
        self, tmp_path: Path, sample_contract: WorkContract
    ):
        """Completing one immutable graph cell unlocks its dependent cell."""
        from northstack.application.worker import WorkerResult

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
        first = GraphCell(
            id="first",
            name="first",
            mode="read_only",
            contract=sample_contract,
            acceptance_criterion_indices=[0],
        )
        second = GraphCell(
            id="second",
            name="second",
            wave=1,
            mode="mutating",
            dependencies=["first"],
            contract=sample_contract,
            acceptance_criterion_indices=[0],
        )
        graph = GraphVersion(
            version=1,
            cells=[first, second],
            edges=[GraphEdge(from_id="first", to_id="second")],
            milestones=["second"],
            current_horizon=1,
        )
        worker = SequenceWorker(
            [
                WorkerResult(ok=True, text="first done"),
                WorkerResult(ok=True, text="second done"),
            ]
        )
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")
        company = Company(
            config=config,
            ledger=ledger,
            artifact_store=store,
            workspace=ws,
            gateway=None,
            worker_factory=FakeWorkerFactory(worker),
            compiler=ContractCompiler(analysis_runner=DeterministicAnalysisRunner()),
            command_profiles={
                "pytest": CommandProfile(name="pytest", argv=["python", "-c", "print('ok')"])
            },
        )
        company._planner = StaticGraphPlanner(graph)

        try:
            outcome = company.run(
                ProjectRequest(goal="two cells", workspace_root=str(tmp_path), max_waves=2)
            )

            assert outcome == RunOutcome.ABSTAINED
            assert [call["cell_id"] for call in worker.calls] == ["first", "second"]
        finally:
            ledger.close()

    def test_transient_failure_retries_with_new_signature(self, tmp_path: Path, monkeypatch):
        """A bounded transient recovery action actually re-invokes the cell."""
        import asyncio

        from northstack.application.worker import WorkerResult

        # Skip the real asyncio.sleep backoff between recovery rounds.
        monkeypatch.setattr(
            "northstack.application.cell_runner._recovery_sleep",
            lambda delay: asyncio.sleep(0),
        )

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
                        "description": "check",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    }
                ]
            ),
        )
        worker = SequenceWorker(
            [
                WorkerResult(ok=False, error="temporary 429", error_kind="rate_limit"),
                WorkerResult(ok=True, text="recovered"),
            ]
        )
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(name="check", argv=["python", "-c", "print('ok')"])
                },
            )
            outcome = company.run(
                ProjectRequest(
                    goal="retry",
                    workspace_root=str(tmp_path),
                    budget=Budget(
                        token_limit=1000,
                        cost_limit_usd=1.0,
                        max_retries=1,
                    ),
                )
            )

            assert outcome == RunOutcome.VERIFIED
            assert len(worker.calls) == 2
        finally:
            ledger.close()

    def test_capability_failure_reroutes_to_another_profile(self, tmp_path: Path):
        """Capability recovery structurally selects a different eligible profile."""
        from northstack.application.worker import WorkerResult

        config = NorthStackConfig(
            name="test",
            profiles=[
                ModelProfile(
                    name="cheap",
                    protocol=Protocol.OPENAI_CHAT,
                    base_url="http://localhost",
                    model="cheap",
                    roles={Role.WORKER},
                    max_concurrency=4,
                ),
                ModelProfile(
                    name="expert",
                    protocol=Protocol.OPENAI_CHAT,
                    base_url="http://localhost",
                    model="expert",
                    roles={Role.WORKER},
                    max_concurrency=1,
                    input_price_per_million_usd=20.0,
                    output_price_per_million_usd=20.0,
                ),
            ],
        )
        runner = DeterministicAnalysisRunner(
            requirements=RequirementsAnalysis(scope="test", deliverables=["out.py"]),
            acceptance=AcceptanceAnalysis(
                criteria=[
                    {
                        "kind": "command",
                        "description": "check",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    }
                ]
            ),
        )
        worker = SequenceWorker(
            [
                WorkerResult(ok=False, error="tool unavailable", error_kind="tool"),
                WorkerResult(ok=True, text="recovered"),
            ]
        )
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(name="check", argv=["python", "-c", "print('ok')"])
                },
            )
            outcome = company.run(
                ProjectRequest(
                    goal="reroute",
                    workspace_root=str(tmp_path),
                    budget=Budget(
                        token_limit=1000,
                        cost_limit_usd=1.0,
                        max_retries=1,
                    ),
                )
            )

            assert outcome == RunOutcome.VERIFIED
            assert [call["profile_name"] for call in worker.calls] == ["expert", "cheap"]
        finally:
            ledger.close()

    def test_reroute_abstain_emits_reroute_failure_reason(self, tmp_path: Path):
        """When reroute has no eligible profile, CELL_FAILED carries the
        reroute reason ('no eligible profile found') AND the original worker
        error -- an abstaining escalation must not erase the root cause."""
        from northstack.application.worker import WorkerResult

        config = NorthStackConfig(
            name="test",
            profiles=[
                ModelProfile(
                    name="only",
                    protocol=Protocol.OPENAI_CHAT,
                    base_url="http://localhost",
                    model="only",
                    roles={Role.WORKER},
                    max_concurrency=4,
                ),
            ],
        )
        runner = DeterministicAnalysisRunner(
            requirements=RequirementsAnalysis(scope="test", deliverables=["out.py"]),
            acceptance=AcceptanceAnalysis(
                criteria=[
                    {
                        "kind": "command",
                        "description": "check",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    }
                ]
            ),
        )
        worker = SequenceWorker(
            [
                WorkerResult(ok=False, error="tool unavailable", error_kind="tool"),
            ]
        )
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(name="check", argv=["python", "-c", "print('ok')"])
                },
            )
            outcome = company.run(
                ProjectRequest(
                    goal="reroute-abstain",
                    workspace_root=str(tmp_path),
                    budget=Budget(
                        token_limit=1000,
                        cost_limit_usd=1.0,
                        max_retries=1,
                    ),
                )
            )

            assert outcome == RunOutcome.FAILED

            # Inspect ledger events for the terminal CELL_FAILED payload.
            run_id = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchone()["run_id"]
            events = ledger.events(run_id)
            cell_failed = [e for e in events if e.kind == EventKind.CELL_FAILED]
            assert len(cell_failed) >= 1
            error = cell_failed[-1].payload.error

            # The error names both the reroute failure reason and the
            # original worker error that triggered recovery.
            assert "no eligible" in error, (
                f"expected reroute abstain reason in CELL_FAILED error, got: {error!r}"
            )
            assert "tool unavailable" in error, (
                f"original worker error dropped from CELL_FAILED: {error!r}"
            )
        finally:
            ledger.close()

    def test_reroute_keeps_tier3_when_budget_ample(self, tmp_path: Path):
        """A reroute must not exclude a tier-3 profile when ample budget remains.

        _score_profile reads its remaining_budget arg as REMAINING; the reroute
        caller used to pass cumulative spent, so <$1 spent excluded every tier-3
        profile early in a run. With $9.50 of $10 left, the other tier-3 profile
        must be reached and the run must VERIFY.
        """
        from northstack.application.worker import WorkerResult

        # Two tier-3 (opus-class) profiles: avg price $20/M -> tier 3.
        # The first-listed scores higher and is picked first; on its failure
        # the reroute must reach the second (the only remaining candidate).
        config = NorthStackConfig(
            name="t3-only",
            profiles=[
                ModelProfile(
                    name="opus-a",
                    protocol=Protocol.OPENAI_CHAT,
                    base_url="http://localhost",
                    model="opus",
                    roles={Role.WORKER},
                    max_concurrency=1,
                    input_price_per_million_usd=20.0,
                    output_price_per_million_usd=20.0,
                ),
                ModelProfile(
                    name="opus-b",
                    protocol=Protocol.OPENAI_CHAT,
                    base_url="http://localhost",
                    model="opus",
                    roles={Role.WORKER},
                    max_concurrency=1,
                    input_price_per_million_usd=20.0,
                    output_price_per_million_usd=20.0,
                ),
            ],
        )
        runner = DeterministicAnalysisRunner(
            requirements=RequirementsAnalysis(scope="test", deliverables=["out.py"]),
            acceptance=AcceptanceAnalysis(
                criteria=[
                    {
                        "kind": "command",
                        "description": "check",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    }
                ]
            ),
        )
        # First call fails with a $0.50 spend (the (0, $1) window the inverted
        # guard mis-read); the second (other tier-3) succeeds.
        worker = SequenceWorker(
            [
                WorkerResult(
                    ok=False,
                    error="tool unavailable",
                    error_kind="tool",
                    total_cost_usd=0.50,
                ),
                WorkerResult(ok=True, text="recovered"),
            ]
        )
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(name="check", argv=["python", "-c", "print('ok')"])
                },
            )
            outcome = company.run(
                ProjectRequest(
                    goal="reroute-t3",
                    workspace_root=str(tmp_path),
                    # Under the graph-validation $100 cap; $9.50 left after
                    # the $0.50 spend -- ample for a tier-3 call.
                    budget=Budget(
                        token_limit=100_000,
                        cost_limit_usd=10.0,
                        max_retries=1,
                    ),
                )
            )

            assert outcome == RunOutcome.VERIFIED, (
                f"tier-3 reroute excluded despite ample budget; outcome={outcome}"
            )
            # The reroute reached the other tier-3 profile, not a retry.
            called = {c["profile_name"] for c in worker.calls}
            assert called == {"opus-a", "opus-b"}, (
                f"expected reroute to reach both tier-3 profiles, got calls={worker.calls!r}"
            )
        finally:
            ledger.close()

    def test_verified_outcome(self, tmp_path: Path):
        """Hard criterion with real command execution -> VERIFIED."""
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

        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")
        worker = FakeWorker(default_ok=True, default_text="done")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    # ``["echo", "ok"]`` only resolves when a Unix ``echo.exe``
                    # (Git for Windows / MSYS2) is on PATH -- on a vanilla Windows
                    # install ``echo`` is ``cmd.exe``'s builtin (not on disk), so
                    # ``shell=False`` raises FileNotFoundError and the gate fails
                    # for the wrong reason. Use the project interpreter instead,
                    # which is cross-platform and always resolvable.
                    "echo_ok": CommandProfile(name="echo_ok", argv=["python", "-c", "print('ok')"]),
                },
            )
            request = ProjectRequest(
                goal="Build feature",
                workspace_root=str(tmp_path),
            )
            outcome = company.run(request)
            assert outcome == RunOutcome.VERIFIED
        finally:
            ledger.close()

    def test_hard_failed_outcome(self, tmp_path: Path):
        """Hard gate failure (missing command profile) -> FAILED."""
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
                        "description": "will fail",
                        "parameters": {"command_name": "nonexistent", "exit_code": 0},
                    },
                ]
            ),
        )

        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={},  # no command profiles
            )
            request = ProjectRequest(
                goal="Build feature",
                workspace_root=str(tmp_path),
            )
            outcome = company.run(request)
            assert outcome == RunOutcome.FAILED
        finally:
            ledger.close()

    def test_abstained_uncalibrated_soft_rubric(self, tmp_path: Path):
        """Soft rubric with no calibration -> ABSTAINED."""
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
                    {"kind": "soft_rubric", "description": "quality review"},
                ]
            ),
        )

        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                calibration_records=[],
            )
            request = ProjectRequest(
                goal="Review code",
                workspace_root=str(tmp_path),
            )
            outcome = company.run(request)
            assert outcome == RunOutcome.ABSTAINED
        finally:
            ledger.close()

    def test_exception_converts_to_failed(self, tmp_path: Path):
        """Unhandled exception -> FAILED (failure-safe)."""
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

        class BrokenRunner:
            async def run_requirements(self, req, profile):
                raise RuntimeError("analysis exploded")

            async def run_repo(self, req, profile):
                return RepoAnalysis()

            async def run_acceptance(self, req, profile):
                return AcceptanceAnalysis()

        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(),
                compiler=ContractCompiler(
                    analysis_runner=BrokenRunner(),
                    synthesizer=CleanSynthesizer(),
                ),
            )
            request = ProjectRequest(
                goal="test",
                workspace_root=str(tmp_path),
            )
            outcome = company.run(request)
            assert outcome == RunOutcome.FAILED
        finally:
            ledger.close()

    def test_worker_failure_enters_recovery(self, tmp_path: Path, monkeypatch):
        """Worker failure triggers recovery -> FAILED."""
        import asyncio

        # Skip the real asyncio.sleep backoff between recovery rounds.
        monkeypatch.setattr(
            "northstack.application.cell_runner._recovery_sleep",
            lambda delay: asyncio.sleep(0),
        )

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
                        "description": "check",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    },
                ]
            ),
        )

        worker = FakeWorker(default_ok=False, default_text="")
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(
                        name="check",
                        argv=["python", "-c", "print('ok')"],
                    ),
                },
            )
            request = ProjectRequest(
                goal="test",
                workspace_root=str(tmp_path),
            )
            outcome = company.run(request)
            assert outcome == RunOutcome.FAILED
            # Verify recovery event was emitted
            events = ledger.events(
                ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchone()["run_id"]
            )
            recovery_events = [e for e in events if e.kind == EventKind.RECOVERY_TRANSITION]
            assert len(recovery_events) > 0
        finally:
            ledger.close()

    def test_audit_replay_equivalence(self, tmp_path: Path):
        """Replaying events produces identical state to live execution."""
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
                        "description": "verify",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    },
                ]
            ),
        )

        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")

        try:
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(
                        name="check",
                        argv=["python", "-c", "print('ok')"],
                    ),
                },
            )
            request = ProjectRequest(
                goal="Audit test",
                workspace_root=str(tmp_path),
            )
            outcome = company.run(request)
            assert outcome == RunOutcome.VERIFIED

            # Find run_id and verify integrity
            row = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchone()
            run_id = row["run_id"]
            integrity = ledger.verify_integrity(run_id)
            assert integrity.ok

            state = replay_run(ledger, run_id)
            assert state.outcome == RunOutcome.VERIFIED
            assert state.status == RunStatus.VERIFIED

            snap = state.snapshot()
            assert snap["outcome"] == "verified"
            assert snap["status"] == "verified"
        finally:
            ledger.close()


# Cancellation: a cancelled run_async must still reach a TERMINAL ledger state.
# Regression for a live-found bug: ``asyncio.CancelledError`` is a
# ``BaseException`` and bypassed ``run_async``'s ``except Exception``, so a
# run cancelled via the web "Stop" (``task.cancel()``) emitted no terminal
# OUTCOME_EMITTED + no FAILED status -- it sat at ``status=intake`` forever in
# history while the run-detail page polled ``outcome_emitted`` indefinitely.
# The fix added a dedicated ``except asyncio.CancelledError`` clause in
# ``Company.run_async`` that emits a FAILED terminal event then re-raises.


class _SleepingAnalysisRunner:
    """Deterministic runner that ``await``s for a long time so a cancel lands
    mid-run (the real DeterministicAnalysisRunner has no await-yield seam)."""

    def __init__(self, delay: float = 30.0) -> None:
        self._delay = delay
        self._requirements = RequirementsAnalysis(
            scope="default scope", deliverables=["deliverable_1"]
        )
        self._repo = RepoAnalysis()
        self._acceptance = AcceptanceAnalysis()

    async def run_requirements(self, request, profile):
        await asyncio.sleep(self._delay)  # the seam the cancel interrupts
        return self._requirements

    async def run_repo(self, request, profile):
        return self._repo

    async def run_acceptance(self, request, profile):
        return self._acceptance


class TestRunAsyncCancellation:
    def test_cancelled_run_emits_terminal_failed(self, tmp_path: Path) -> None:
        async def scenario() -> None:
            ledger = Ledger(path=tmp_path / "cancel.db")
            store = ArtifactStore(tmp_path / "artifacts")
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
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=RestrictedWorkspace(tmp_path / "ws"),
                gateway=None,
                worker_factory=None,  # type: ignore[arg-type]
                compiler=ContractCompiler(
                    analysis_runner=_SleepingAnalysisRunner(),
                    tool_registry=["read", "write", "create", "replace", "list"],
                ),
            )
            request = ProjectRequest(goal="slow goal", workspace_root=str(tmp_path / "ws"))
            task = asyncio.create_task(company.run_async(request, run_id="run-cancel"))
            # Let it reach the sleeping seam: REQUEST_ACCEPTED is emitted
            # synchronously before the first await inside compile().
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            ledger.close()

            # Re-open and assert a TERMINAL event was written (the bug: none).
            ledger = Ledger(path=tmp_path / "cancel.db")
            events = ledger.events("run-cancel")
            outcomes = [e for e in events if e.kind == EventKind.OUTCOME_EMITTED]
            statuses = [
                e.payload.status.value for e in events if e.kind == EventKind.STATUS_CHANGED
            ]
            ledger.close()
            # A terminal OUTCOME_EMITTED must exist (the bug: zero outcomes).
            assert outcomes, "cancelled run emitted no OUTCOME_EMITTED"
            assert outcomes[-1].payload.outcome.value == "failed"
            # The final status must be terminal 'failed', not stuck at 'intake'.
            assert "failed" in statuses, f"no failed status; statuses={statuses}"

        asyncio.run(scenario())


class TestRunAsyncAppendsOffTheEventLoop:
    """Ledger appends on the async path run on a worker thread, not the
    event-loop thread. ``Ledger.append_next`` is blocking sqlite I/O; if it
    runs in the loop thread a slow disk stalls every concurrently-gathered
    coroutine (including a sibling read-only cell). The append is offloaded
    via ``asyncio.to_thread`` so its executing thread id differs from the
    loop's."""

    def test_async_emits_append_on_a_worker_thread(self, tmp_path: Path) -> None:
        async def scenario() -> None:
            loop_thread = threading.get_ident()
            append_threads: list[int] = []

            class _ThreadSpyLedger(Ledger):
                """Records the thread id of every append_next, then delegates."""

                def append_next(self, run_id, payload):
                    append_threads.append(threading.get_ident())
                    return super().append_next(run_id, payload)

            ledger = _ThreadSpyLedger(path=tmp_path / "thread.db")
            store = ArtifactStore(tmp_path / "artifacts")
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
                            "description": "check",
                            "parameters": {"command_name": "check", "exit_code": 0},
                        }
                    ]
                ),
            )
            worker = FakeWorker(default_ok=True, default_text="output")
            company = Company(
                config=config,
                ledger=ledger,
                artifact_store=store,
                workspace=RestrictedWorkspace(tmp_path / "ws"),
                gateway=None,
                worker_factory=FakeWorkerFactory(worker),
                compiler=ContractCompiler(
                    analysis_runner=runner,
                    synthesizer=CleanSynthesizer(),
                ),
                command_profiles={
                    "check": CommandProfile(
                        name="check",
                        argv=["python", "-c", "print('ok')"],
                    ),
                },
            )
            request = ProjectRequest(goal="test", workspace_root=str(tmp_path / "ws"))
            try:
                await company.run_async(request, run_id="run-thread")
            finally:
                ledger.close()

            assert append_threads, "run_async made no ledger appends"
            loop_thread_appends = [t for t in append_threads if t == loop_thread]
            assert not loop_thread_appends, (
                f"{len(loop_thread_appends)} of {len(append_threads)} appends ran on "
                "the event-loop thread; the async path must offload append_next "
                "to a worker thread (asyncio.to_thread)"
            )

        asyncio.run(scenario())


class TestRunBudgetEnforcement:
    """Lock in the run-level budget fix (3 coupled bugs):

    1. ABSENT enforcement: ``Budget.can_spend`` existed but was never called --
       usage accumulated in ``_run_cell`` but was never compared to the limit,
       so ``budget_tokens=100`` ran to completion.  Now exceeding the run
       budget short-circuits to ABSTAINED.
    2. Missing from snapshot: the run-level budget was never emitted to the
       ledger (REQUEST_ACCEPTED carried only goal+workspace_root), so the UI's
       budget bullet had no target.  Now REQUEST_ACCEPTED carries it and the
       replay snapshot exposes ``budget``.
    3. Negative -> 422 (this is the web layer; see test_web_api.py
       test_run_negative_budget_422).
    """

    def _company(self, tmp_path: Path, worker: Any, budget: Budget, graph: GraphVersion) -> Company:
        ledger = Ledger(path=tmp_path / "budget.db")
        store = ArtifactStore(tmp_path / "artifacts")
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
        company = Company(
            config=config,
            ledger=ledger,
            artifact_store=store,
            workspace=RestrictedWorkspace(tmp_path / "ws"),
            gateway=None,
            worker_factory=FakeWorkerFactory(worker),
            compiler=ContractCompiler(
                analysis_runner=DeterministicAnalysisRunner(),
                tool_registry=["read", "write", "create", "replace", "list"],
            ),
            command_profiles={
                "pytest": CommandProfile(name="pytest", argv=["python", "-c", "print('ok')"])
            },
        )
        company._planner = StaticGraphPlanner(graph)
        # Stash for teardown; run() is sync here (Company.run wraps asyncio.run).
        company._test_ledger = ledger  # type: ignore[attr-defined]
        return company

    def test_a_raising_read_only_cell_records_its_cause(
        self, tmp_path: Path, sample_contract: WorkContract, caplog
    ) -> None:
        """A cell that raises fails the run; the cause must not vanish.

        The gather collects the exception, the run reports FAILED a wave later,
        and the cell emitted nothing -- so this log line is the only record of
        why the run died.
        """
        from northstack.application.worker import WorkerResult

        def cell(cid: str) -> GraphCell:
            return GraphCell(
                id=cid,
                name=cid,
                mode="read_only",
                contract=sample_contract,
                acceptance_criterion_indices=[0],
            )

        graph = GraphVersion(
            version=1,
            cells=[cell("a"), cell("b")],
            edges=[],
            milestones=["a", "b"],
            current_horizon=0,
        )
        worker = SequenceWorker([WorkerResult(ok=True, text="a done")])
        company = self._company(
            tmp_path, worker, Budget(token_limit=100000, cost_limit_usd=1000.0), graph
        )
        ledger = company._test_ledger  # type: ignore[attr-defined]
        original = company._run_cell

        async def run_cell(run_id, cell_obj, *args, **kwargs):
            if cell_obj.id == "b":
                raise RuntimeError("cell b exploded")
            return await original(run_id, cell_obj, *args, **kwargs)

        company._run_cell = run_cell  # type: ignore[assignment]

        async def scenario() -> RunOutcome:
            return await company.run_async(
                ProjectRequest(
                    goal="raise",
                    workspace_root=str(tmp_path / "ws"),
                    budget=Budget(token_limit=100000, cost_limit_usd=1000.0),
                    max_waves=2,
                ),
                run_id="run-raises",
            )

        try:
            with caplog.at_level(logging.ERROR):
                outcome = asyncio.run(scenario())
        finally:
            ledger.close()

        assert outcome == RunOutcome.FAILED
        assert "cell b exploded" in caplog.text
        assert "cell_id=b" in caplog.text

    def test_budget_exceeded_abstains_and_marks_exhausted(
        self, tmp_path: Path, sample_contract: WorkContract
    ) -> None:
        from northstack.application.worker import WorkerResult

        # Two read_only cells; each burns 1000 in+1000 out tokens = 2000 tokens.
        # Run budget token_limit=1500 -> the FIRST cell alone (2000 tokens)
        # already exceeds it, so the run abstains after wave 1 with an
        # exhausted marker.  (A mutating cell would also work, but read_only
        # cells run concurrently and exercise the post-batch budget check.)
        def cell(cid: str) -> GraphCell:
            return GraphCell(
                id=cid,
                name=cid,
                mode="read_only",
                contract=sample_contract,
                acceptance_criterion_indices=[0],
            )

        graph = GraphVersion(
            version=1,
            cells=[cell("a"), cell("b")],
            edges=[],
            milestones=["a", "b"],
            current_horizon=0,
        )
        worker = SequenceWorker(
            [
                WorkerResult(
                    ok=True,
                    text="a done",
                    total_input_tokens=1000,
                    total_output_tokens=1000,
                ),
                WorkerResult(
                    ok=True,
                    text="b done",
                    total_input_tokens=1000,
                    total_output_tokens=1000,
                ),
            ]
        )
        company = self._company(
            tmp_path, worker, Budget(token_limit=1500, cost_limit_usd=1000.0), graph
        )
        ledger = company._test_ledger  # type: ignore[attr-defined]
        run_id = "run-budget"

        async def scenario() -> RunOutcome:
            return await company.run_async(
                ProjectRequest(
                    goal="burn budget",
                    workspace_root=str(tmp_path / "ws"),
                    budget=Budget(token_limit=1500, cost_limit_usd=1000.0),
                    max_waves=2,
                ),
                run_id=run_id,
            )

        try:
            outcome = asyncio.run(scenario())
        finally:
            ledger.close()

        # Bug 1: the run must abstain instead of running to completion.
        assert outcome == RunOutcome.ABSTAINED, f"expected abstained, got {outcome}"

        ledger = Ledger(path=tmp_path / "budget.db")
        try:
            events = ledger.events(run_id)
            snapshot = replay_run(ledger, run_id).snapshot()
        finally:
            ledger.close()

        # A BUDGET_UPDATED event carrying exhausted=True must precede the
        # outcome -- this is the "why" the run abstained (the bare
        # OUTCOME_EMITTED=abstained carries no reason).  Without Bug 1's fix
        # there is NO exhausted marker at all and the run runs to completion.
        budget_events = [e for e in events if e.kind == EventKind.BUDGET_UPDATED]
        assert any(e.payload.exhausted for e in budget_events), (
            f"no exhausted=True BUDGET_UPDATED event; "
            f"budget_events={[e.payload for e in budget_events]}"
        )
        outcomes = [e for e in events if e.kind == EventKind.OUTCOME_EMITTED]
        assert outcomes, "no OUTCOME_EMITTED"
        assert outcomes[-1].payload.outcome == RunOutcome.ABSTAINED
        statuses = [e.payload.status.value for e in events if e.kind == EventKind.STATUS_CHANGED]
        assert "abstained" in statuses, f"no abstained status; statuses={statuses}"

        # Bug 2: the replay snapshot must carry the run-level budget so the
        # UI budget bullet has a target.  Before the fix the snapshot had usage
        # but no budget (None even though the operator set token_limit=1500).
        assert snapshot["budget"] == {
            "token_limit": 1500,
            "cost_limit_usd": 1000.0,
        }, f"snapshot budget wrong / missing: {snapshot.get('budget')}"
        # And the REQUEST_ACCEPTED event must now carry the budget too (the
        # the ledger-only audit path -- an auditor replaying ledger.json sees
        # the limit the run was started under).
        accepted = [e for e in events if e.kind == EventKind.REQUEST_ACCEPTED]
        assert accepted, "no REQUEST_ACCEPTED event"
        accepted_budget = accepted[0].payload.budget
        assert accepted_budget is not None, "REQUEST_ACCEPTED budget missing"
        assert accepted_budget.token_limit == 1500
        assert accepted_budget.cost_limit_usd == 1000.0

    def test_no_budget_runs_unlimited_and_snapshot_is_none(
        self, tmp_path: Path, sample_contract: WorkContract
    ) -> None:
        """Complement: a run with NO budget (budget=None) must NOT abstain on
        usage and the snapshot budget must be None (UI renders 'unlimited').
        Guards against an over-eager fix that would trip on a default-zero axis
        or fabricate a budget when none was set."""
        from northstack.application.worker import WorkerResult

        cell = GraphCell(
            id="only",
            name="only",
            mode="mutating",
            contract=sample_contract,
            acceptance_criterion_indices=[0],
        )
        graph = GraphVersion(
            version=1,
            cells=[cell],
            edges=[],
            milestones=["only"],
            current_horizon=0,
        )
        worker = SequenceWorker(
            [
                WorkerResult(
                    ok=True,
                    text="done",
                    total_input_tokens=9_000_000,
                    total_output_tokens=9_000_000,
                )
            ]
        )
        company = self._company(tmp_path, worker, Budget(token_limit=1, cost_limit_usd=1.0), graph)
        # No budget on the request -> unlimited.
        ledger = company._test_ledger  # type: ignore[attr-defined]
        run_id = "run-nobudget"

        async def scenario() -> RunOutcome:
            return await company.run_async(
                ProjectRequest(
                    goal="unlimited",
                    workspace_root=str(tmp_path / "ws"),
                    max_waves=1,
                ),
                run_id=run_id,
            )

        try:
            outcome = asyncio.run(scenario())
        finally:
            ledger.close()
        # The run must reach a terminal outcome (verified/abstained/failed) and
        # NOT have been killed by a spurious budget trip.  It does abstain on
        # the deterministic contract's hard gate, but for the gate -- not usage.
        assert outcome in (
            RunOutcome.VERIFIED,
            RunOutcome.ABSTAINED,
            RunOutcome.FAILED,
        ), f"unexpected outcome: {outcome}"

        # Must NOT have abstained on budget (it abstains/verifies on the hard
        # gate per the deterministic contract, NOT because usage tripped -- 18M
        # tokens with no budget must sail right through).  The point: budget=None
        # never trips regardless of usage magnitude.
        ledger = Ledger(path=tmp_path / "budget.db")
        try:
            events = ledger.events(run_id)
            snapshot = replay_run(ledger, run_id).snapshot()
        finally:
            ledger.close()
        # No exhausted marker when there was no budget to exhaust.
        assert not any(
            e.kind == EventKind.BUDGET_UPDATED and e.payload.exhausted for e in events
        ), "budget exhausted emitted despite no run-level budget"
        # Snapshot budget is None for an unlimited run.
        assert snapshot["budget"] is None, f"unlimited run has a budget: {snapshot['budget']}"

    def test_graph_validation_failure_emits_terminal_status_changed(
        self, tmp_path: Path, sample_contract: WorkContract
    ):
        """L-a regression: graph-validation abstain must emit a terminal STATUS_CHANGED.

        Before the fix, _emit_status(ABSTAINED, CONTRACTED) silently did nothing
        because CONTRACTED->ABSTAINED was not in RunStatus._TRANSITIONS.  The run
        emitted OUTCOME_EMITTED=abstained but no terminal STATUS_CHANGED, leaving
        it stuck at status=contracted in the UI forever.

        Fix choice C: ABSTAINED added to all non-terminal phases' transition sets,
        matching the FAILED pattern.  Design intent (models.py comment) already
        states "any non-terminal phase may transition to FAILED"; ABSTAINED has
        the same rationale -- a pre-execution abstain is a legitimate terminal.
        """
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")
        try:
            company = Company(
                config=NorthStackConfig(
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
                ),
                ledger=ledger,
                artifact_store=store,
                workspace=ws,
                gateway=None,
                worker_factory=FakeWorkerFactory(),
                compiler=ContractCompiler(
                    analysis_runner=DeterministicAnalysisRunner(),
                ),
                command_profiles={},
            )
            # Inject a planner that always fails validation, triggering the
            # graph-validation abstain branch at orchestration.py ~line 518.
            company._planner = FailingPlanner(
                GraphVersion(
                    version=1,
                    cells=[
                        GraphCell(
                            id="cell-1",
                            contract=sample_contract,
                            wave=0,
                            mode="read_only",
                        ),
                    ],
                    edges=[],
                    milestones=["cell-1"],
                    current_horizon=0,
                )
            )

            outcome = company.run(
                ProjectRequest(
                    goal="graph validation failure",
                    workspace_root=str(tmp_path),
                )
            )
            assert outcome == RunOutcome.ABSTAINED

            # The run MUST have a terminal STATUS_CHANGED event as its most
            # recent status change.  Without the fix, the run gets stuck at
            # status=contracted with outcome=abstained -- non-terminal forever.
            run_ids = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchall()
            assert run_ids, "no run persisted"
            events = ledger.events(run_ids[0]["run_id"])
            status_events = [e for e in events if e.kind == EventKind.STATUS_CHANGED]
            assert status_events, (
                "no STATUS_CHANGED event emitted after graph validation failure; "
                "the run is stuck at a non-terminal status"
            )
            last_status = status_events[-1].payload.status.value
            assert last_status in {
                RunStatus.ABSTAINED.value,
                RunStatus.FAILED.value,
            }, (
                f"most recent STATUS_CHANGED is '{last_status}', not terminal; "
                f"all status changes: "
                f"{[e.payload.status.value for e in status_events]}"
            )
        finally:
            ledger.close()


# H) RunStatus transition table unit tests


class TestRunStatusTransitions:
    async def test_cancelled_terminal_outcome_is_tracked_without_duplication(
        self, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from northstack.events.stream import EventStream

        company = object.__new__(Company)
        company._ledger = ledger
        entered, release = asyncio.Event(), asyncio.Event()
        original = EventStream.emit_async

        async def delayed_emit(stream: EventStream, payload):
            if isinstance(payload, OutcomeEmitted):
                entered.set()
                await release.wait()
            return await original(stream, payload)

        monkeypatch.setattr(EventStream, "emit_async", delayed_emit)
        task = asyncio.create_task(
            company._emit_terminal_outcome("r1", RunOutcome.VERIFIED, "done")
        )
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        cancelled = await task
        assert isinstance(cancelled, asyncio.CancelledError)
        await company._emit_failure(
            "r1", RunStatus.VERIFYING, "cancelled", terminal_outcome=RunOutcome.VERIFIED
        )
        events = ledger.events("r1")
        assert sum(isinstance(event.payload, OutcomeEmitted) for event in events) == 1
        assert events[-1].payload.status is RunStatus.VERIFIED

    async def test_failure_recovery_finishes_terminal_pair_when_cancelled(
        self, ledger: Ledger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from northstack.events.stream import EventStream

        company = object.__new__(Company)
        company._ledger = ledger
        entered, release = asyncio.Event(), asyncio.Event()
        original = EventStream.emit_async

        async def delayed_emit(stream: EventStream, payload):
            if isinstance(payload, OutcomeEmitted):
                entered.set()
                await release.wait()
            return await original(stream, payload)

        monkeypatch.setattr(EventStream, "emit_async", delayed_emit)
        task = asyncio.create_task(company._emit_failure("r1", RunStatus.INTAKE, "cancelled"))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        assert isinstance(await task, asyncio.CancelledError)
        events = ledger.events("r1")
        assert [type(event.payload) for event in events] == [OutcomeEmitted, StatusChanged]
        assert events[-1].payload.status is RunStatus.FAILED

    async def test_failure_recovery_does_not_duplicate_existing_terminal_outcome(
        self, ledger: Ledger
    ) -> None:
        company = object.__new__(Company)
        company._ledger = ledger
        ledger.append_next("r1", RunCreated())
        ledger.append_next("r1", OutcomeEmitted(outcome=RunOutcome.VERIFIED))
        await company._emit_failure(
            "r1",
            RunStatus.VERIFYING,
            "late failure",
            terminal_outcome=RunOutcome.VERIFIED,
        )
        events = ledger.events("r1")
        assert sum(isinstance(event.payload, OutcomeEmitted) for event in events) == 1
        assert isinstance(events[-1].payload, StatusChanged)
        assert events[-1].payload.status is RunStatus.VERIFIED

    """Pin the RunStatus._TRANSITIONS table so state-machine widenings are explicit."""

    def test_contracted_can_transition_to_abstained(self):
        """L-a fix: CONTRACTED -> ABSTAINED must be allowed for graph-validation abstain."""
        assert RunStatus.can_transition(RunStatus.CONTRACTED, RunStatus.ABSTAINED)

    def test_contracted_can_transition_to_planned(self):
        assert RunStatus.can_transition(RunStatus.CONTRACTED, RunStatus.PLANNED)

    def test_contracted_can_transition_to_failed(self):
        assert RunStatus.can_transition(RunStatus.CONTRACTED, RunStatus.FAILED)

    def test_planned_can_transition_to_abstained(self):
        """ABSTAINED reachable from any non-terminal phase, matching FAILED."""
        assert RunStatus.can_transition(RunStatus.PLANNED, RunStatus.ABSTAINED)

    def test_executing_routes_to_abstained_through_verifying(self):
        """ADR 0001: EXECUTING -> ABSTAINED is illegal; it detours via VERIFYING,
        owned in one place by RunStateMachine.route."""
        from northstack.domain.status import RunStateMachine

        assert not RunStatus.can_transition(RunStatus.EXECUTING, RunStatus.ABSTAINED)
        assert RunStateMachine.route(RunStatus.EXECUTING, RunStatus.ABSTAINED) == [
            RunStatus.VERIFYING,
            RunStatus.ABSTAINED,
        ]

    def test_intake_can_transition_to_abstained(self):
        assert RunStatus.can_transition(RunStatus.INTAKE, RunStatus.ABSTAINED)

    def test_terminal_statuses_have_no_transitions(self):
        from northstack.domain.status import RunStateMachine

        table = RunStateMachine.transitions()
        for status in (RunStatus.VERIFIED, RunStatus.ABSTAINED, RunStatus.FAILED):
            assert table[status] == set(), (
                f"{status} should be terminal with no outgoing transitions"
            )


# Stall detector
# A run that is alive but not progressing inside the configured window
# abstains with a typed ``StallDetected`` event. Per-cell heartbeats are the
# progress signal; the detector compares "now" against the last beat. A window
# of 0 disables the detector (no configured cap).


class TestStallDetector:
    """``StallDetector`` owns the "alive but not progressing" signal.

    A clock-injectable unit so the test advances time deterministically rather
    than sleeping. ``heartbeat()`` records a progress beat; ``is_stalled()``
    is True only when the window is configured (>0) AND the elapsed time since
    the last beat exceeds it.
    """

    def test_not_stalled_within_window(self):
        from northstack.application.stall_detector import StallDetector

        now = [0.0]

        def clock() -> float:
            return now[0]

        detector = StallDetector(window_seconds=10.0, clock=clock)
        detector.heartbeat()
        now[0] = 5.0  # within the window
        assert not detector.is_stalled()

    def test_stalled_when_no_progress_past_window(self):
        from northstack.application.stall_detector import StallDetector

        now = [0.0]

        def clock() -> float:
            return now[0]

        detector = StallDetector(window_seconds=10.0, clock=clock)
        detector.heartbeat()
        now[0] = 11.0  # past the window, no further beat
        assert detector.is_stalled()

    def test_heartbeat_resets_the_window(self):
        from northstack.application.stall_detector import StallDetector

        now = [0.0]

        def clock() -> float:
            return now[0]

        detector = StallDetector(window_seconds=10.0, clock=clock)
        detector.heartbeat()
        now[0] = 9.0
        detector.heartbeat()  # progress just before the window expires
        now[0] = 18.0  # 9s since the last beat -> within the window
        assert not detector.is_stalled()
        now[0] = 21.0  # 12s since the last beat -> stalled
        assert detector.is_stalled()

    def test_window_zero_disables_detector(self):
        from northstack.application.stall_detector import StallDetector

        now = [0.0]

        def clock() -> float:
            return now[0]

        # 0 means no configured cap: the detector never trips, mirroring the
        # BudgetAuthority "None == unlimited" semantics.
        detector = StallDetector(window_seconds=0.0, clock=clock)
        detector.heartbeat()
        now[0] = 10_000.0
        assert not detector.is_stalled()


class TestStallDetectedProjection:
    """Adding ``StallDetected`` to the catalog forces a fold handler (the
    projection's ``match`` closes with ``assert_never``); the handler abstains.
    """

    def test_stall_detected_folds_to_abstained(self):
        from northstack.domain import RunState
        from northstack.events.catalog import StallDetected
        from northstack.events.envelope import EventEnvelope
        from northstack.events.projection import fold

        state = RunState(run_id="run-1")
        env = EventEnvelope(seq=1, run_id="run-1", payload=StallDetected(cell_id="cell-1"))
        folded = fold(state, env)
        assert folded.outcome == RunOutcome.ABSTAINED


class TestStalledRunAbstains:
    """A run whose cell hangs (alive but not progressing) abstains with a
    ``StallDetected`` event in the ledger (ADR 0001).

    The stall watchdog races the in-flight cell against the stall window. The
    cell here hangs on an ``asyncio.Event`` that never fires; the test advances
    the stall clock past the window so the watchdog trips, cancels the hung
    cell, and the run abstains. No real sleep -- the clock is injected.
    """

    def test_hanging_cell_abstains_with_stall_detected(self, tmp_path: Path) -> None:
        import asyncio

        from northstack.adapters.artifacts import ArtifactStore
        from northstack.adapters.sqlite_ledger import Ledger
        from northstack.adapters.workspace.restricted import (
            CommandProfile,
            RestrictedWorkspace,
        )
        from northstack.application.contracting import (
            AcceptanceAnalysis,
            ContractCompiler,
            DeterministicAnalysisRunner,
            RequirementsAnalysis,
        )
        from northstack.application.orchestrator import Company
        from northstack.application.worker import WorkerResult
        from northstack.config import ModelProfile, NorthStackConfig, Protocol, Role

        hang = asyncio.Event()

        class HangingWorker:
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
                await hang.wait()  # never set -> hangs forever
                return WorkerResult(ok=True, text="x", total_input_tokens=1, total_output_tokens=1)

        class HangingFactory:
            def create(self, workspace: Any) -> HangingWorker:
                return HangingWorker()

        # Injected stall clock: starts at 0, advanced past the window mid-run.
        now = [0.0]

        def clock() -> float:
            return now[0]

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
            run=__import__("northstack.config", fromlist=["RunConfig"]).RunConfig(
                stall_window_seconds=10.0
            ),
        )
        runner = DeterministicAnalysisRunner(
            requirements=RequirementsAnalysis(scope="test", deliverables=["out.py"]),
            acceptance=AcceptanceAnalysis(
                criteria=[
                    {
                        "kind": "command",
                        "description": "check",
                        "parameters": {"command_name": "check", "exit_code": 0},
                    }
                ]
            ),
        )
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        ws = RestrictedWorkspace(tmp_path / "workspace")
        company = Company(
            config=config,
            ledger=ledger,
            artifact_store=store,
            workspace=ws,
            gateway=None,
            worker_factory=HangingFactory(),
            compiler=ContractCompiler(analysis_runner=runner, synthesizer=CleanSynthesizer()),
            command_profiles={
                "check": CommandProfile(name="check", argv=["python", "-c", "print('ok')"]),
            },
        )
        company._stall_clock = clock

        # Pre-generate stable run_id so the ledger can be re-opened and replayed
        # after the run closes it.
        run_id = "stall-run"
        request = ProjectRequest(goal="test", workspace_root=str(tmp_path))

        async def _drive() -> Any:
            task = asyncio.create_task(company.run_async(request, run_id=run_id))
            # Let the run reach the hanging cell. A single sleep(0) yields once
            # but the run crosses several await points (contracting, planning,
            # routing) before the worker hangs; a small real sleep gives those
            # awaits enough turns to land. The assertion uses no real sleep --
            # the watchdog trips on the injected clock, not wall time.
            await asyncio.sleep(0.2)
            now[0] = 100.0  # past the 10s window -> watchdog trips
            # A bounded wait: with no watchdog the hang would block forever; the
            # timeout surfaces that as a test failure rather than a stuck suite.
            return await asyncio.wait_for(task, timeout=5)

        try:
            outcome = asyncio.run(_drive())
        finally:
            ledger.close()

        assert outcome == RunOutcome.ABSTAINED
        # Re-open a fresh read handle to inspect the ledger (the run closed it).
        read_ledger = Ledger(path=tmp_path / "test.db")
        try:
            rows = read_ledger.events(run_id)
            kinds = [row.kind.value for row in rows]
        finally:
            read_ledger.close()
        assert "stall_detected" in kinds, f"no StallDetected event in ledger: {kinds}"
