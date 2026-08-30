"""Comprehensive tests for the control plane.

Covers:
  A) Event model, Ledger.append_next, replay state reconstruction
  B) Contract compilation pipeline (async parallel fan-out, synthesis, validation)
  C) Rolling-wave graph (planner, validator, scheduler)
  D) Inspectable rule-based router
  E) Verification/release law (real command execution, soft rubric, evidence manifest)
  F) Typed bounded recovery (classification, dedup, budget enforcement)
  G) Pipeline and CLI (Company.run with fake seams, end-to-end scenarios)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.application.contracting import (
    AcceptanceAnalysis,
    ContractCompiler,
    ContractValidator,
    DeterministicAnalysisRunner,
    RequirementsAnalysis,
    _criterion_from_dict,
)
from northstack.application.orchestrator import Company
from northstack.application.planning import GraphPlanner
from northstack.application.recovery import AttemptDeduplicator, FailureClassifier, RecoveryManager
from northstack.application.routing import Router
from northstack.application.scheduling import Scheduler
from northstack.application.tools.registry import ToolRegistry
from northstack.application.verification.hard_gates import (
    CommandEvidence,
    HardCheckResult,
    HardGateVerifier,
)
from northstack.application.verification.soft_rubric import DeterministicReviewer, SoftRubricChecker
from northstack.config import (
    Capability,
    ModelProfile,
    NorthStackConfig,
    Protocol,
    Role,
    RouteMapping,
    RunConfig,
)
from northstack.domain import (
    AcceptanceCriterion,
    AttemptSignature,
    Budget,
    BudgetUsage,
    CalibrationRecord,
    CommandCriterion,
    CriterionKind,
    FailureType,
    FileDiffCriterion,
    GraphCell,
    GraphEdge,
    GraphVersion,
    PolicyCriterion,
    ProjectRequest,
    RecoveryAction,
    RunOutcome,
    RunStatus,
    SchemaCriterion,
    SoftRubricCriterion,
    TreeDigestCriterion,
    WorkContract,
)
from northstack.events.catalog import (
    EventKind,
    OutcomeEmitted,
    RequestAccepted,
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


# B) Contract compilation pipeline


class TestContractCompilation:
    async def test_parallel_fan_out_distinct_profiles(self, sample_request, sample_config):
        runner = DeterministicAnalysisRunner()
        compiler = ContractCompiler(
            analysis_runner=runner,
            tool_registry=["read_file", "write_file"],
        )
        contract = await compiler.compile(sample_request, config=sample_config)
        assert contract.objective == "Implement a greeting function"
        assert contract.version == 1
        assert len(contract.deliverables) > 0
        assert len(contract.acceptance_criteria) > 0

    async def test_default_synthesizer_preserves_criterion_parameters(self, sample_request):
        runner = DeterministicAnalysisRunner(
            acceptance=AcceptanceAnalysis(
                criteria=[
                    {
                        "kind": "command",
                        "description": "test command",
                        "parameters": {"command_name": "pytest", "exit_code": 0},
                    }
                ]
            )
        )
        compiler = ContractCompiler(analysis_runner=runner)

        contract = await compiler.compile(sample_request)

        assert isinstance(contract.acceptance_criteria[0], CommandCriterion)
        assert contract.acceptance_criteria[0].command_name == "pytest"
        assert contract.acceptance_criteria[0].exit_code == 0

    def test_validator_rejects_empty_objective(self, sample_request):
        validator = ContractValidator()
        bad = WorkContract(
            id="wc-bad",
            objective="",
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
        )
        errors = validator.validate(bad, sample_request)
        assert any("objective" in e for e in errors)

    def test_validator_rejects_empty_deliverables(self, sample_request):
        validator = ContractValidator()
        bad = WorkContract(
            id="wc-bad",
            objective="Do stuff",
            deliverables=[],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
        )
        errors = validator.validate(bad, sample_request)
        assert any("deliverables" in e for e in errors)

    def test_validator_rejects_unknown_tools(self, sample_request):
        validator = ContractValidator(tool_registry=["read", "write"])
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            allowed_tools=["read", "write", "hacking_tool"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
        )
        errors = validator.validate(contract, sample_request)
        assert any("hacking_tool" in e for e in errors)

    def test_validator_reads_registry_dispatchable_names(self, sample_request):
        """The legal-name set comes from the single ``ToolRegistry``: the
        validator asks the registry what is dispatchable, so no second,
        hand-maintained list of tool names can drift. ``web_fetch`` is
        dispatchable in the default registry, so a contract allowing it is
        accepted; a name the registry does NOT dispatch (``hacking_tool``) is
        rejected even though no separate name list was passed."""
        validator = ContractValidator(tool_registry=ToolRegistry.with_defaults(command_profiles={}))
        allowed = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            allowed_tools=["read", "write", "web_fetch"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                CommandCriterion(description="run tests", command_name="pytest", exit_code=0),
            ],
        )
        assert validator.validate(allowed, sample_request) == []

        rejected = WorkContract(
            id="wc-2",
            objective="test",
            deliverables=["x"],
            allowed_tools=["read", "hacking_tool"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                CommandCriterion(description="run tests", command_name="pytest", exit_code=0),
            ],
        )
        errors = validator.validate(rejected, sample_request)
        assert any("hacking_tool" in e for e in errors)

    def test_invalid_criterion_kind_rejected_at_parse_time(self):
        """A bogus criterion kind cannot reach the validator at all.

        The discriminated union rejects unknown kinds when the contract is
        built, so the gate that once lived in ContractValidator now fires at
        parse time -- the contract carrying the bad criterion never exists.
        """
        from pydantic import TypeAdapter, ValidationError

        with pytest.raises(ValidationError):
            TypeAdapter(AcceptanceCriterion).validate_python(
                {"kind": "invalid_kind", "description": "bad"}
            )

    async def test_falsifier_can_reject(self, sample_request):
        class RejectingFalsifier:
            async def check(self, contract, request):
                return "wrong interpretation"

        runner = DeterministicAnalysisRunner()
        compiler = ContractCompiler(
            analysis_runner=runner,
            falsifier=RejectingFalsifier(),
        )
        with pytest.raises(ValueError, match="Falsifier"):
            await compiler.compile(sample_request)


# C) Graph planning


class TestGraphPlanning:
    async def test_planner_creates_default_graph(self, sample_contract):
        planner = GraphPlanner()
        graph = await planner.plan(sample_contract, "run-1")
        assert graph.version == 1
        assert len(graph.cells) == 1
        assert graph.cells[0].wave == 0

    def test_graph_validator_rejects_an_empty_graph(self):
        assert "graph contains no cells" in GraphPlanner().validate(GraphVersion())

    def test_graph_validator_rejects_cycles(self, sample_contract):
        planner = GraphPlanner()
        graph = GraphVersion(
            version=1,
            cells=[
                GraphCell(id="a", contract=sample_contract, mode="read_only", dependencies=["b"]),
                GraphCell(id="b", contract=sample_contract, mode="read_only", dependencies=["a"]),
            ],
            edges=[
                GraphEdge(from_id="a", to_id="b"),
                GraphEdge(from_id="b", to_id="a"),
            ],
        )
        errors = planner.validate(graph)
        assert any("cycle" in e for e in errors)

    def test_graph_validator_rejects_dependency_only_cycles(self, sample_contract):
        graph = GraphVersion(
            cells=[
                GraphCell(id="a", contract=sample_contract, dependencies=["b"]),
                GraphCell(id="b", contract=sample_contract, dependencies=["a"]),
            ]
        )

        assert any("cycle" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_unknown_dependencies(self, sample_contract):
        graph = GraphVersion(
            cells=[GraphCell(id="a", contract=sample_contract, dependencies=["missing"])]
        )

        assert any("unknown cell 'missing'" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_self_dependencies(self, sample_contract):
        graph = GraphVersion(
            cells=[GraphCell(id="a", contract=sample_contract, dependencies=["a"])],
            edges=[GraphEdge(from_id="a", to_id="a")],
        )

        assert any("depends on itself" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_duplicate_dependencies(self, sample_contract):
        graph = GraphVersion(
            cells=[
                GraphCell(id="a", contract=sample_contract),
                GraphCell(id="b", contract=sample_contract, dependencies=["a", "a"]),
            ],
            edges=[GraphEdge(from_id="a", to_id="b")],
        )

        assert any("duplicate dependencies" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_blocking_edges_that_disagree_with_dependencies(
        self, sample_contract
    ):
        graph = GraphVersion(
            cells=[
                GraphCell(id="a", contract=sample_contract),
                GraphCell(id="b", contract=sample_contract),
            ],
            edges=[GraphEdge(from_id="a", to_id="b")],
        )

        assert any("blocking edges disagree" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_duplicate_edges(self, sample_contract):
        graph = GraphVersion(
            cells=[
                GraphCell(id="a", contract=sample_contract),
                GraphCell(id="b", contract=sample_contract, dependencies=["a"]),
            ],
            edges=[
                GraphEdge(from_id="a", to_id="b"),
                GraphEdge(from_id="a", to_id="b"),
            ],
        )

        assert any("duplicate edges" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_unknown_edge_endpoints(self, sample_contract):
        graph = GraphVersion(
            cells=[GraphCell(id="a", contract=sample_contract)],
            edges=[GraphEdge(from_id="a", to_id="missing", kind="informs")],
        )

        assert any("unknown edge endpoint" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_unknown_and_duplicate_milestones(self, sample_contract):
        graph = GraphVersion(
            cells=[GraphCell(id="a", contract=sample_contract)],
            milestones=["missing", "missing"],
        )

        errors = GraphPlanner().validate(graph)
        assert "duplicate milestones in graph" in errors
        assert "unknown milestone 'missing'" in errors

    def test_graph_validator_rejects_duplicate_and_out_of_range_criteria(self, sample_contract):
        graph = GraphVersion(
            cells=[
                GraphCell(
                    id="a",
                    contract=sample_contract,
                    acceptance_criterion_indices=[0, 0, -1, 999],
                )
            ]
        )

        errors = GraphPlanner().validate(graph)
        assert any("duplicate criterion indices" in error for error in errors)
        assert any("invalid criterion indices" in error for error in errors)

    def test_graph_validator_rejects_a_horizon_that_omits_planned_waves(self, sample_contract):
        graph = GraphVersion(
            cells=[GraphCell(id="a", contract=sample_contract, wave=2)],
            current_horizon=1,
        )

        assert any("current horizon" in error for error in GraphPlanner().validate(graph))

    def test_graph_validator_rejects_dependencies_in_the_same_or_later_wave(self, sample_contract):
        graph = GraphVersion(
            cells=[
                GraphCell(id="a", contract=sample_contract, wave=1),
                GraphCell(id="b", contract=sample_contract, wave=1, dependencies=["a"]),
            ],
            edges=[GraphEdge(from_id="a", to_id="b")],
            current_horizon=1,
        )

        assert any(
            "must be in an earlier wave" in error for error in GraphPlanner().validate(graph)
        )

    def test_graph_validator_rejects_multiple_mutating_same_wave(self, sample_contract):
        planner = GraphPlanner()
        graph = GraphVersion(
            version=1,
            cells=[
                GraphCell(id="a", contract=sample_contract, wave=0, mode="mutating"),
                GraphCell(id="b", contract=sample_contract, wave=0, mode="mutating"),
            ],
            edges=[],
        )
        errors = planner.validate(graph)
        assert any("mutating" in e for e in errors)

    def test_graph_validator_rejects_more_mutating_cells_than_waves(self, sample_contract):
        # The wave loop serializes mutating cells one per pass, so a graph
        # with m mutating cells cannot finish inside max_waves < m however
        # its wave numbers are arranged. This must fail at planning time,
        # not starve cells mid-run.
        planner = GraphPlanner()
        graph = GraphVersion(
            version=1,
            cells=[
                GraphCell(id=f"m{i}", contract=sample_contract, wave=i, mode="mutating")
                for i in range(4)
            ],
            edges=[],
        )
        errors = planner.validate(graph, max_waves=3)
        assert any("max_waves=3" in e for e in errors)
        assert not any("max_waves" in e for e in planner.validate(graph, max_waves=4))
        assert not any("max_waves" in e for e in planner.validate(graph))

    def test_unlimited_cells_allowed_under_unlimited_run(self, sample_contract):
        # Config semantics: 0 == unlimited on a run axis. An unlimited run
        # axis makes unlimited cell budgets the operator's explicit choice,
        # not a validation failure.
        planner = GraphPlanner()
        unlimited = sample_contract.model_copy(
            update={"budget": Budget(token_limit=None, cost_limit_usd=None)}, deep=True
        )
        graph = GraphVersion(version=1, cells=[GraphCell(id="a", contract=unlimited)], edges=[])
        assert planner.validate(graph, run_budget=unlimited.budget) == []

    def test_unlimited_cells_rejected_under_finite_run(self, sample_contract):
        planner = GraphPlanner()
        unlimited = sample_contract.model_copy(
            update={"budget": Budget(token_limit=None, cost_limit_usd=None)}, deep=True
        )
        graph = GraphVersion(version=1, cells=[GraphCell(id="a", contract=unlimited)], edges=[])
        errors = planner.validate(graph, run_budget=Budget.default())
        assert any("unlimited token budget" in e for e in errors)
        assert any("unlimited cost budget" in e for e in errors)

    async def test_planner_stamps_worker_role_when_routed(self, sample_contract):
        # Regression for the routing->selection wiring: when the operator routes
        # WORKER to a non-empty chain, the planner must stamp required_profile_roles
        # with the WORKER role so Router._score_profile actually consults the
        # ordered chain.  Without the stamp the chain is plumbed-but-unconsulted.
        p = ModelProfile(
            name="p",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        config = NorthStackConfig(
            name="routed",
            profiles=[p],
            routing=[RouteMapping(role=Role.WORKER, profiles=["p"])],
        )
        planner = GraphPlanner(config.role_map())
        graph = await planner.plan(sample_contract, "run-1")
        assert graph.cells[0].required_profile_roles == ["worker"]

    async def test_planner_requires_worker_role_even_when_not_routed(self, sample_contract):
        # Role mappings rank eligible workers. They must not make an
        # orchestrator/reviewer eligible to execute a mutating worker cell.
        orchestrator = ModelProfile(
            name="opus-orch",
            protocol=Protocol.ANTHROPIC_MESSAGES,
            base_url="http://y",
            model="claude-opus",
            roles={Role.ORCHESTRATOR},
            max_concurrency=1,
        )
        config = NorthStackConfig(
            name="no-worker",
            profiles=[orchestrator],
            routing=[RouteMapping(role=Role.ORCHESTRATOR, profiles=["opus-orch"])],
        )
        planner = GraphPlanner(config.role_map())
        graph = await planner.plan(sample_contract, "run-1")
        assert graph.cells[0].required_profile_roles == ["worker"]

    async def test_plan_then_route_honours_ordered_routing_chain(self, sample_contract):
        # End-to-end wiring: GraphPlanner(config.role_map()).plan() produces a cell
        # the Router then routes to the FIRST-listed profile in the ordered WORKER
        # chain -- proving the operator's ordering genuinely drives selection,
        # not just tier-based generic scoring of any eligible WORKER profile.
        primary = ModelProfile(
            name="primary",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        secondary = ModelProfile(
            name="secondary",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        config = NorthStackConfig(
            name="chain-order",
            profiles=[primary, secondary],
            # Order deliberately avoids "primary" being first alphabetically or by
            # insertion: the planner+router must pick the first LISTED entry.
            routing=[RouteMapping(role=Role.WORKER, profiles=["secondary", "primary"])],
        )
        planner = GraphPlanner(config.role_map())
        router = Router(config)
        graph = await planner.plan(sample_contract, "run-order")
        cell = graph.cells[0]
        assert cell.required_profile_roles == ["worker"]
        decision = router.route(cell, sample_contract)
        assert not decision.abstained
        assert decision.selected_profile == "secondary"

    def test_scheduler_ready_cells(self, sample_contract):
        scheduler = Scheduler()
        graph = GraphVersion(
            version=1,
            cells=[
                GraphCell(id="a", contract=sample_contract, mode="read_only", status="completed"),
                GraphCell(
                    id="b",
                    contract=sample_contract,
                    mode="read_only",
                    dependencies=["a"],
                    status="pending",
                ),
                GraphCell(
                    id="c",
                    contract=sample_contract,
                    mode="mutating",
                    dependencies=["a"],
                    status="pending",
                ),
            ],
            edges=[
                GraphEdge(from_id="a", to_id="b"),
                GraphEdge(from_id="a", to_id="c"),
            ],
        )
        ready = scheduler.ready_cells(graph)
        ids = {c.id for c in ready}
        assert "b" in ids
        assert "c" in ids


# D) Router


class TestRouter:
    def test_route_selects_profile(self, sample_config, sample_contract):
        router = Router(sample_config)
        cell = GraphCell(
            id="cell-1",
            contract=sample_contract,
            mode="read_only",
            required_profile_roles=["worker"],
        )
        decision = router.route(cell, sample_contract)
        assert not decision.abstained
        assert decision.selected_profile != ""

    def test_route_abstains_when_no_match(self, sample_contract):
        config = NorthStackConfig(name="empty", profiles=[])
        router = Router(config)
        cell = GraphCell(id="cell-1", contract=sample_contract, mode="read_only")
        decision = router.route(cell, sample_contract)
        assert decision.abstained

    def test_role_map_routes_orchestrator_to_opus_profile(self, sample_contract):
        cheap = ModelProfile(
            name="haiku-worker",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="haiku",
            roles={Role.WORKER},
            max_concurrency=8,
            input_price_per_million_usd=0.5,
            output_price_per_million_usd=1.5,
        )
        opus = ModelProfile(
            name="opus-orch",
            protocol=Protocol.ANTHROPIC_MESSAGES,
            base_url="http://y",
            model="claude-opus",
            roles={Role.ORCHESTRATOR},
            max_concurrency=1,
            input_price_per_million_usd=15.0,
            output_price_per_million_usd=75.0,
        )
        config = NorthStackConfig(
            name="routed",
            profiles=[cheap, opus],
            routing=[
                RouteMapping(role=Role.WORKER, profiles=["haiku-worker"]),
                RouteMapping(role=Role.ORCHESTRATOR, profiles=["opus-orch"]),
            ],
        )
        router = Router(config)

        worker_cell = GraphCell(
            id="c-w",
            contract=sample_contract,
            mode="mutating",
            required_profile_roles=["worker"],
        )
        assert router.route(worker_cell, sample_contract).selected_profile == "haiku-worker"

        orch_cell = GraphCell(
            id="c-o",
            contract=sample_contract,
            mode="mutating",
            required_profile_roles=["orchestrator"],
        )
        assert router.route(orch_cell, sample_contract).selected_profile == "opus-orch"

    def test_role_map_abstains_when_role_unmapped(self, sample_contract):
        cheap = ModelProfile(
            name="haiku-worker",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="haiku",
            roles={Role.WORKER},
            max_concurrency=8,
        )
        config = NorthStackConfig(
            name="partial",
            profiles=[cheap],
            routing=[RouteMapping(role=Role.WORKER, profiles=["haiku-worker"])],
        )
        router = Router(config)
        cell = GraphCell(
            id="c-o",
            contract=sample_contract,
            mode="read_only",
            required_profile_roles=["orchestrator"],
        )
        assert router.route(cell, sample_contract).abstained

    def test_role_map_fallback_chain_honours_order(self, sample_contract):
        primary = ModelProfile(
            name="p",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        secondary = ModelProfile(
            name="s",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        config = NorthStackConfig(
            name="chain",
            profiles=[primary, secondary],
            routing=[RouteMapping(role=Role.WORKER, profiles=["s", "p"])],
        )
        router = Router(config)
        cell = GraphCell(
            id="c",
            contract=sample_contract,
            mode="read_only",
            required_profile_roles=["worker"],
        )
        # Written order wins even though profiles are otherwise identical.
        assert router.route(cell, sample_contract).selected_profile == "s"

    def test_role_map_excludes_failed_profile_then_uses_chain(self, sample_contract):
        primary = ModelProfile(
            name="p",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        secondary = ModelProfile(
            name="s",
            protocol=Protocol.OPENAI_CHAT,
            base_url="http://x",
            model="m",
            roles={Role.WORKER},
            max_concurrency=1,
        )
        config = NorthStackConfig(
            name="chain2",
            profiles=[primary, secondary],
            routing=[RouteMapping(role=Role.WORKER, profiles=["p", "s"])],
        )
        router = Router(config)
        cell = GraphCell(
            id="c",
            contract=sample_contract,
            mode="read_only",
            required_profile_roles=["worker"],
        )
        assert router.route(cell, sample_contract).selected_profile == "p"
        assert router.route(cell, sample_contract, excluded_profiles={"p"}).selected_profile == "s"

    def test_route_accepts_remaining_budget_type(self, sample_config, sample_contract):
        """Cumulative usage produces a distinct remaining-budget value."""
        router = Router(sample_config)
        cell = GraphCell(
            id="cell-1",
            contract=sample_contract,
            mode="read_only",
            required_profile_roles=["worker"],
        )
        budget = Budget(token_limit=1_000_000, cost_limit_usd=10.0, max_calls=10)
        spent = BudgetUsage(
            total_input_tokens=500,
            total_output_tokens=300,
            total_cost_usd=0.50,
            total_calls=2,
        )
        remaining = spent.remaining(budget)
        assert remaining.tokens == 999_200
        assert remaining.cost_usd == 9.5
        decision = router.route(cell, sample_contract, remaining_budget=remaining)
        assert decision is not None


class TestStatusEmissionInvariant:
    def test_illegal_status_transition_is_observable(self):
        company = object.__new__(Company)
        company._ledger = None

        async def _call() -> None:
            # ``_emit_status`` is async (the append is offloaded to a worker
            # thread). The legality guard still raises BEFORE any offload.
            await company._emit_status("run", RunStatus.VERIFIED, RunStatus.INTAKE)

        with pytest.raises(ValueError, match="Illegal status transition"):
            asyncio.run(_call())


# E) Verification


class TestHardGateVerifier:
    async def test_tree_digest_passes_unchanged_and_fails_tampered(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """The whole-tree gate sees edits AND newly added files."""
        from northstack.application.verification.hard_gates import compute_tree_digest

        tests = workspace.root / "tests"
        tests.mkdir()
        (tests / "test_x.py").write_text("x = 1\n")
        digest = compute_tree_digest(workspace.root, "tests")

        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract = WorkContract(
            id="wc-tree",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                TreeDigestCriterion(description="suite pinned", path="tests", tree_hash=digest),
            ],
        )
        results = await verifier.verify(contract)
        assert results[0].passed

        # A new file collected before test_x (the falsifier's attack) trips it.
        (tests / "aaa_patch.py").write_text("import app\n")
        results = await verifier.verify(contract)
        assert not results[0].passed
        assert "changed since compile" in results[0].detail

    async def test_command_executes_real_command(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """HardGateVerifier executes real commands via workspace."""
        verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
            command_profiles={
                "echo_test": CommandProfile(
                    name="echo_test",
                    argv=["python", "-c", "print('hello')"],
                ),
            },
        )
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                CommandCriterion(
                    description="echo test",
                    command_name="echo_test",
                    exit_code=0,
                ),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1
        assert results[0].passed
        assert results[0].evidence_ref is not None
        # The stored artifact is valid JSON that round-trips through the
        # command-evidence model (no f-string quoting of bytes).
        content = artifact_store.read(results[0].evidence_ref)
        payload = json.loads(content)
        assert payload["stdout"].strip() == "hello"
        assert payload["ok"] is True
        CommandEvidence.model_validate(payload)

    async def test_command_evidence_is_valid_json_with_tricky_output(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """Output containing quotes, braces and newlines must still parse.

        The old f-string ``{tool_result.data!r}`` quoted the ``bytes`` repr,
        producing JSON no ``json.loads`` would accept once output held quotes or
        braces. The pydantic model serialisation makes the artifact valid JSON
        regardless of the command's stdout.
        """
        verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
            command_profiles={
                "tricky": CommandProfile(
                    name="tricky",
                    argv=["python", "-c", 'print(\'{"a": 1}\\nquote"end\')'],
                ),
            },
        )
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                CommandCriterion(description="tricky output", command_name="tricky", exit_code=0),
            ],
        )

        results = await verifier.verify(contract)
        assert results[0].passed
        content = artifact_store.read(results[0].evidence_ref)
        # The regression: this used to raise json.JSONDecodeError.
        payload = json.loads(content)
        CommandEvidence.model_validate(payload)
        assert "quote" in payload["stdout"]

    async def test_command_fails_on_bad_command(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """Failing real command produces failed check with evidence."""
        verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
            command_profiles={
                "fail_cmd": CommandProfile(
                    name="fail_cmd",
                    argv=["python", "-c", "import sys; sys.exit(1)"],
                ),
            },
        )
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                CommandCriterion(
                    description="should fail",
                    command_name="fail_cmd",
                    exit_code=0,
                ),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].evidence_ref is not None

    async def test_file_diff_checks_real_file(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """File check verifies actual file existence via workspace."""
        # Create a test file in the workspace
        ws_root = tmp_path / "workspace"
        ws_root.mkdir(exist_ok=True)
        (ws_root / "test.txt").write_text("hello world")

        verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
        )
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                FileDiffCriterion(
                    description="file exists",
                    path="test.txt",
                    must_exist=True,
                ),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1
        assert results[0].passed

    async def test_file_diff_content_contains_passes(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """content_contains verifies the file holds the exact required text."""
        ws_root = tmp_path / "workspace"
        ws_root.mkdir(exist_ok=True)
        (ws_root / "hello.txt").write_text("NorthStack ran OK")

        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["hello.txt"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                FileDiffCriterion(
                    description="hello.txt has required text",
                    path="hello.txt",
                    must_exist=True,
                    content_contains="NorthStack ran OK",
                ),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1
        assert results[0].passed

    async def test_file_diff_content_contains_fails_on_missing_text(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """A file that exists but lacks the required content must fail hard."""
        ws_root = tmp_path / "workspace"
        ws_root.mkdir(exist_ok=True)
        (ws_root / "hello.txt").write_text("something else entirely")

        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["hello.txt"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                FileDiffCriterion(
                    description="hello.txt has required text",
                    path="hello.txt",
                    must_exist=True,
                    content_contains="NorthStack ran OK",
                ),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1
        assert not results[0].passed

    async def test_file_diff_content_equals_exact_match(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """content_equals requires the full file body to match exactly."""
        ws_root = tmp_path / "workspace"
        ws_root.mkdir(exist_ok=True)
        (ws_root / "exact.txt").write_text("exactly this")

        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract_pass = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["exact.txt"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                FileDiffCriterion(
                    description="exact match",
                    path="exact.txt",
                    must_exist=True,
                    content_equals="exactly this",
                ),
            ],
        )
        results = await verifier.verify(contract_pass)
        assert results[0].passed

        # Trailing content breaks the exact match.
        contract_fail = contract_pass.model_copy(
            update={
                "id": "wc-2",
                "acceptance_criteria": [
                    FileDiffCriterion(
                        description="exact match",
                        path="exact.txt",
                        must_exist=True,
                        content_equals="exactly this\n",
                    )
                ],
            }
        )
        results = await verifier.verify(contract_fail)
        assert not results[0].passed

    async def test_forged_digest_cannot_pass(
        self,
        tmp_path: Path,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """Schema check with forged digest fails."""
        verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
        )
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SchemaCriterion(
                    description="schema check",
                    artifact_digest="sha256:" + "a" * 64,
                    json_schema={"type": "object"},
                ),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1
        assert not results[0].passed  # forged digest fails

    async def test_command_honors_expected_exit_code(
        self,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """A command criterion may explicitly require a non-zero exit code."""
        verifier = HardGateVerifier(
            workspace=workspace,
            artifact_store=artifact_store,
            command_profiles={
                "expected_two": CommandProfile(
                    name="expected_two",
                    argv=["python", "-c", "import sys; sys.exit(2)"],
                ),
            },
        )
        contract = WorkContract(
            id="wc-expected-exit",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                CommandCriterion(
                    description="exit two",
                    command_name="expected_two",
                    exit_code=2,
                ),
            ],
        )

        results = await verifier.verify(contract)

        assert results[0].passed

    async def test_schema_uses_runtime_evidence_digest(
        self,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """Schema checks bind to the cell output recorded for that criterion."""
        ref = artifact_store.write(b'{"answer": 42}', media_type="application/json")
        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract = WorkContract(
            id="wc-runtime-schema",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SchemaCriterion(
                    description="answer schema",
                    # The runtime evidence digest (passed below) overrides this
                    # placeholder; the criterion field is required, so a non-
                    # empty sentinel keeps construction valid while the real
                    # binding comes from the authoritative evidence map.
                    artifact_digest="runtime",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"const": 42}},
                        "required": ["answer"],
                    },
                ),
            ],
        )

        results = await verifier.verify(contract, evidence_digests={0: ref.digest})

        assert results[0].passed

    async def test_policy_uses_authoritative_tool_audit(
        self,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """Policy verification ignores model-provided tools_used parameters."""
        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract = WorkContract(
            id="wc-policy-audit",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                PolicyCriterion(
                    description="no writes",
                    check="forbidden_tools",
                    tools=["write"],
                ),
            ],
        )

        results = await verifier.verify(contract, tools_used=["write"])

        assert not results[0].passed

    async def test_unknown_policy_check_fails_closed(
        self,
        workspace: RestrictedWorkspace,
        artifact_store: ArtifactStore,
    ):
        """Unknown policy checks cannot silently pass a release gate."""
        verifier = HardGateVerifier(workspace=workspace, artifact_store=artifact_store)
        contract = WorkContract(
            id="wc-unknown-policy",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                PolicyCriterion(
                    description="unsupported",
                    check="future_policy",
                ),
            ],
        )

        results = await verifier.verify(contract)

        assert not results[0].passed

    async def test_soft_criteria_get_a_result_not_silently_skipped(self):
        """A soft rubric criterion is not a hard gate, but it must not be
        silently dropped: the verifier returns a result for *every* criterion
        (``continue`` on an unhandled kind is forbidden). The soft result is
        marked passed and labelled so it cannot be mistaken for a hard failure.
        """
        ws = RestrictedWorkspace("/tmp/nonexistent")
        store = ArtifactStore("/tmp/nonexistent_artifacts")
        verifier = HardGateVerifier(workspace=ws, artifact_store=store)
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SoftRubricCriterion(description="quality"),
            ],
        )

        results = await verifier.verify(contract)
        assert len(results) == 1, "every criterion must yield a result, none skipped"
        assert results[0].kind == CriterionKind.SOFT_RUBRIC
        assert results[0].passed is True
        assert "not a hard gate" in results[0].detail


class TestSoftRubricChecker:
    async def test_requires_two_reviewers(self):
        checker = SoftRubricChecker(reviewers=[])
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SoftRubricCriterion(description="quality"),
            ],
        )
        verdicts, disagreement = await checker.check(contract)
        assert disagreement is True
        assert all(not v for v in verdicts.values())

    async def test_calibrated_reviewers_agree(self):
        reviewers = [
            DeterministicReviewer(verdicts={0: True}),
            DeterministicReviewer(verdicts={0: True}),
        ]
        calibration = [
            CalibrationRecord(
                criterion_index=0,
                reviewer_agreement_rate=0.9,
                sample_count=10,
                agreement_threshold=0.67,
            ),
        ]
        checker = SoftRubricChecker(reviewers=reviewers, calibration_records=calibration)
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SoftRubricCriterion(description="quality"),
            ],
        )
        verdicts, disagreement = await checker.check(contract)
        assert not disagreement
        assert verdicts.get(0) is True

    async def test_uncalibrated_abstains(self):
        reviewers = [
            DeterministicReviewer(verdicts={0: True}),
            DeterministicReviewer(verdicts={0: True}),
        ]
        checker = SoftRubricChecker(reviewers=reviewers, calibration_records=[])
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SoftRubricCriterion(description="quality"),
            ],
        )
        verdicts, disagreement = await checker.check(contract)
        assert disagreement is True
        assert verdicts.get(0) is False

    def test_check_schema_invalid_digest_returns_hard_check_result_not_raises(self):
        """Mal digest (missing sha256: prefix) triggers ValidationError which
        should be caught and turned into a failed HardCheckResult, not raised."""
        verifier = HardGateVerifier(
            workspace=MagicMock(),
            artifact_store=MagicMock(),
        )
        criterion = SchemaCriterion(
            description="schema check",
            artifact_digest="abc123",
            json_schema={"type": "object"},
        )
        result = verifier._check_schema(criterion, 0)
        assert isinstance(result, HardCheckResult)
        assert result.passed is False
        assert "artifact" in result.detail.lower()

    async def test_min_reviewers_enforced_per_criterion(self):
        """When CalibrationRecord.min_reviewers exceeds the number of available
        reviewers the criterion should fail even if all reviewers agree."""
        reviewers = [
            DeterministicReviewer(verdicts={0: True}),
            DeterministicReviewer(verdicts={0: True}),
        ]
        calibration = [
            CalibrationRecord(
                criterion_index=0,
                min_reviewers=5,
                sample_count=10,
                agreement_threshold=0.67,
            ),
        ]
        checker = SoftRubricChecker(reviewers=reviewers, calibration_records=calibration)
        contract = WorkContract(
            id="wc-1",
            objective="test",
            deliverables=["x"],
            budget=Budget(token_limit=100, cost_limit_usd=0.1),
            acceptance_criteria=[
                SoftRubricCriterion(description="quality"),
            ],
        )
        verdicts, disagreement = await checker.check(contract)
        assert disagreement is True
        assert verdicts.get(0) is False


# F) Recovery


class TestFailureClassifier:
    def test_classifies_provider_error(self):
        classifier = FailureClassifier()
        assert classifier.classify("provider") == FailureType.TRANSIENT

    def test_classifies_budget_error(self):
        classifier = FailureClassifier()
        assert classifier.classify("budget") == FailureType.BUDGET

    def test_classifies_safety_error(self):
        classifier = FailureClassifier()
        assert classifier.classify("safety") == FailureType.SAFETY


class TestAttemptDeduplicator:
    def test_first_attempt_not_duplicate(self):
        dedup = AttemptDeduplicator()
        sig = AttemptSignature(
            contract_version=1,
            cell_id="c1",
            profile_name="p1",
            strategy_id="s1",
        )
        assert not dedup.is_duplicate("run-1", sig)

    def test_second_identical_is_duplicate(self):
        dedup = AttemptDeduplicator()
        sig = AttemptSignature(
            contract_version=1,
            cell_id="c1",
            profile_name="p1",
            strategy_id="s1",
        )
        dedup.record("run-1", sig)
        assert dedup.is_duplicate("run-1", sig)

    def test_different_strategy_not_duplicate(self):
        dedup = AttemptDeduplicator()
        sig1 = AttemptSignature(
            contract_version=1,
            cell_id="c1",
            profile_name="p1",
            strategy_id="s1",
        )
        sig2 = AttemptSignature(
            contract_version=1,
            cell_id="c1",
            profile_name="p1",
            strategy_id="s2",
        )
        dedup.record("run-1", sig1)
        assert not dedup.is_duplicate("run-1", sig2)


class TestRecoveryManager:
    def test_safety_always_terminates(self):
        manager = RecoveryManager()
        action = manager.decide(
            run_id="run-1",
            error_kind="safety",
            error_detail="refused",
        )
        assert action == RecoveryAction.TERMINATE

    def test_transient_retries(self):
        manager = RecoveryManager()
        action = manager.decide(
            run_id="run-1",
            error_kind="provider",
            error_detail="timeout",
        )
        assert action == RecoveryAction.BACKOFF_RETRY

    def test_duplicate_signature_terminates(self):
        manager = RecoveryManager()
        sig = AttemptSignature(
            contract_version=1,
            cell_id="c1",
            profile_name="p1",
        )
        manager.decide(
            run_id="run-1",
            error_kind="provider",
            error_detail="timeout",
            attempt_signature=sig,
        )
        action = manager.decide(
            run_id="run-1",
            error_kind="provider",
            error_detail="timeout",
            attempt_signature=sig,
        )
        assert action == RecoveryAction.TERMINATE

    def test_decide_without_usage_returns_fail_even_with_headroom(self):
        """Pins the ``decide()``-seam contract: when ``usage`` is omitted a
        budget failure returns FAIL regardless of contract headroom.

        The SCOPE_REDUCTION branch in ``decide()`` (recovery.py:240-252) is
        gated on ``usage and contract_budget``; with ``usage=None`` it is
        skipped and the failure falls through to FAIL.  This is the contract
        the orchestration caller depends on -- if it ever stops threading
        ``usage`` into ``_handle_cell_failure`` (orchestration.py), budget
        failures with ample headroom will silently regress from
        SCOPE_REDUCTION back to FAIL.

        (This test was previously named
        ``test_budget_scope_reduction_branch_is_dead_in_production`` and
        documented the branch as unreachable.  It is now reachable: the
        orchestration caller threads ``usage`` into ``decide()``.  The
        branch's liveness is covered at the orchestration seam by
        ``test_budget_failure_with_headroom_routes_to_scope_reduction``;
        this test stays as the ``decide()``-seam guard.)
        """
        manager = RecoveryManager()
        contract_budget = Budget(token_limit=100_000, cost_limit_usd=100.0)
        # With usage omitted, even a huge-headroom contract and zero usage
        # cannot enter the SCOPE_REDUCTION branch -- a budget failure is FAIL.
        action = manager.decide(
            run_id="run-2",
            error_kind="budget",
            error_detail="exceeded",
            contract_budget=contract_budget,
            # usage intentionally omitted -- pins the decide()-seam contract.
        )
        assert action == RecoveryAction.FAIL

    def test_budget_failure_with_headroom_routes_to_scope_reduction(
        self, tmp_path: Path, sample_contract: WorkContract
    ):
        """A budget failure reached via the orchestration seam (not via
        ``decide()`` directly) must hit the SCOPE_REDUCTION branch when the
        run has spent less than half its contract budget -- the recovery
        policy's intended "shrink scope rather than hard-fail" path.

        This is the live-behavior counterpart to
        ``test_budget_scope_reduction_branch_is_dead_in_production``: that
        test documents the dead branch at the ``decide()`` seam (no caller
        passed ``usage``); this test asserts the *fixed* orchestration seam
        threads cumulative ``usage`` into ``decide()`` so the branch fires.

        Independent source of truth for the expected action: the threshold
        logic in ``RecoveryManager.decide`` (cost<50% OR tokens<50% -> SCOPE).
        We set a generous contract budget and a worker that returns a budget
        failure after spending a few tokens, so cumulative usage is far below
        the 50% threshold on both axes.
        """
        from northstack.adapters.artifacts import ArtifactStore
        from northstack.adapters.sqlite_ledger import Ledger
        from northstack.adapters.workspace.restricted import RestrictedWorkspace
        from northstack.application.contracting import ContractCompiler
        from northstack.application.orchestrator import Company
        from northstack.application.worker import WorkerResult
        from northstack.domain import Budget

        # Generous budget: a budget failure after a few tokens is well under
        # 50% on both axes -> SCOPE_REDUCTION, not FAIL.
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
        # Tighten the contract budget so we control the 50% threshold
        # precisely.  A budget that allows 100k tokens means 50% = 50k; the
        # scripted failure spends ~150 tokens, comfortably under.
        budget = Budget(token_limit=100_000, cost_limit_usd=10.0, max_retries=0)
        contract = sample_contract.model_copy(update={"budget": budget})
        cell = GraphCell(
            id="c1",
            name="c1",
            mode="read_only",
            contract=contract,
            acceptance_criterion_indices=[0],
        )
        graph = GraphVersion(
            version=1,
            cells=[cell],
            edges=[],
            milestones=["c1"],
            current_horizon=0,
        )

        # Worker returns a budget failure (not an exception -- so the
        # orchestration's `else` branch accumulates usage from the result).
        failing = WorkerResult(
            ok=False,
            error="token limit exceeded",
            error_kind="budget",
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost_usd=0.001,
        )
        worker = SequenceWorker([failing])
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
                    analysis_runner=DeterministicAnalysisRunner(),
                ),
                command_profiles={},
            )
            # Inject the graph directly so the planner doesn't re-derive it.
            company._planner = StaticGraphPlanner(graph)

            request = ProjectRequest(
                goal="test",
                workspace_root=str(tmp_path),
                budget=budget,
            )
            outcome = company.run(request)
            # A budget failure with no retry budget is terminal for the run.
            # SCOPE_REDUCTION is a terminal ABSTAIN-class action, so the run
            # outcome must be ABSTAINED -- the recovery policy's "shrink scope
            # rather than hard-fail" verdict.  A bug at orchestration.py's
            # terminal block (next_decision only bound on the REROUTE branch)
            # turns this into FAILED via an UnboundLocalError that masks the
            # scope-reduction decision, so we pin the whole run-level contract
            # here, not just the recovery action.
            run_ids = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchall()
            assert run_ids, "no run persisted"
            events = ledger.events(run_ids[0]["run_id"])
            recovery = [e for e in events if e.kind == EventKind.RECOVERY_TRANSITION]
            assert recovery, "no RECOVERY_TRANSITION emitted for the budget failure"
            actions = [e.payload.action.value for e in recovery]
            assert "scope_reduction" in actions, (
                "budget failure with <50% usage did not route to SCOPE_REDUCTION; "
                f"recovery actions emitted: {actions}.  The orchestration seam "
                "is dropping cumulative `usage` when calling decide(), so the "
                "scope-reduction branch stays dead."
            )
            # The terminal block must complete: a CELL_FAILED event carries
            # the budget cause to the ledger (so the UI shows the right error
            # kind), and the run returns ABSTAINED.  If the terminal block
            # raises (e.g. UnboundLocalError on next_decision for a non-reroute
            # action), CellFailed is never emitted and outcome is FAILED.
            assert outcome == RunOutcome.ABSTAINED, (
                "SCOPE_REDUCTION is a terminal ABSTAIN-class action; the run "
                f"outcome must be ABSTAINED, got {outcome}.  The terminal "
                "recovery block likely raised before emitting CELL_FAILED/"
                "returning."
            )
            cell_failed = [e for e in events if e.kind == EventKind.CELL_FAILED]
            assert cell_failed, (
                "no CELL_FAILED emitted for the terminal budget failure -- the "
                "terminal recovery block did not complete."
            )
            assert cell_failed[0].payload.error_kind == "budget", (
                "CELL_FAILED must carry error_kind=budget for a budget failure; "
                f"got {cell_failed[0].payload.error_kind!r}."
            )
        finally:
            ledger.close()


# F2) Retry ownership and recovery accounting
# The per-cell contract's ``Budget.max_retries`` is the ONLY retry cap; the
# hidden run-wide ``BudgetEnforcer`` has been removed from
# ``RecoveryManager``.  ``max_retries == 0`` means "no configured orchestration
# cap" (compatibility), and the cap is enforced BEFORE ``RECOVERY_TRANSITION``
# is emitted so the ledger records the action actually taken.


class TestRetryOwnershipAndRecoveryAccounting:
    """Red-green slices for the retry-ownership fix (plan section 1).

    Seams (pre-agreed in the approved plan):
      - ``RecoveryManager.decide`` -- no run-wide retry counter; a transient
        failure always returns ``BACKOFF_RETRY`` regardless of how many times
        ``decide`` is called for the same run.
      - ``Company.run`` -- ``contract.budget.max_retries`` is the sole per-cell
        cap, enforced before ``RECOVERY_TRANSITION`` so the ledger records the
        actual action (``TERMINATE`` once capped, not the would-be retry).
      - ``Budget.max_retries == 0`` -- no configured orchestration cap.
    """

    def test_recovery_manager_has_no_budget_enforcer(self):
        """``RecoveryManager`` must not carry a run-wide ``BudgetEnforcer``.

        It was a hidden run-wide counter that conflicted with the contract's
        per-failed-cell ``Budget.max_retries`` meaning and leaked retry slots
        across cells.  The orchestration per-cell loop is now the sole owner.
        """
        import northstack.application.recovery as recovery_mod

        assert not hasattr(recovery_mod, "BudgetEnforcer"), (
            "BudgetEnforcer must be removed from northstack.recovery; it was a "
            "hidden run-wide counter conflicting with per-cell max_retries"
        )
        manager = RecoveryManager()
        assert not hasattr(manager, "_budget_enforcer"), (
            "RecoveryManager must not hold an internal BudgetEnforcer"
        )

    def test_decide_transient_never_terminates_via_run_wide_cap(self):
        """A transient failure always returns ``BACKOFF_RETRY`` no matter how
        many times ``decide`` is called for the same run -- there is no
        hidden run-wide retry counter in ``RecoveryManager``.

        Independent source of truth: the RECOVERY_POLICY table maps TRANSIENT
        -> [BACKOFF_RETRY, ...]; nothing in ``decide`` consults an attempt
        count.
        """
        manager = RecoveryManager()
        for _ in range(7):
            assert (
                manager.decide(
                    run_id="run-1",
                    error_kind="provider",
                    error_detail="timeout",
                )
                == RecoveryAction.BACKOFF_RETRY
            ), "RecoveryManager must not terminate transient failures via a run-wide cap"

    def test_max_retries_zero_means_no_orchestration_cap(self, tmp_path: Path, monkeypatch):
        """``max_retries == 0`` means no configured orchestration cap: a
        transient failure that recovers on the next attempt is retried and
        the run VERIFIES, rather than terminating on the first failure.

        Compatibility: the old run-wide ``BudgetEnforcer`` defaulted to 3 and
        ``max_retries == 0`` was the "no cap" sentinel; this pins that the
        removal of the hidden cap did not replace it with another constant.
        """
        import asyncio

        from northstack.application.worker import WorkerResult

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
        # First call transient-fails, second succeeds: with no cap, the retry
        # is allowed and the run verifies. The failure count (1) would trip a
        # cap of 1, but max_retries=0 means no cap at all.
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
                    goal="retry-unlimited",
                    workspace_root=str(tmp_path),
                    budget=Budget(
                        token_limit=1000,
                        cost_limit_usd=1.0,
                        max_retries=0,
                    ),
                )
            )
            assert outcome == RunOutcome.VERIFIED, (
                "max_retries=0 means no configured cap; a single transient "
                f"failure must be retried and the run verified, got {outcome}"
            )
            assert len(worker.calls) == 2
        finally:
            ledger.close()

    def test_retry_cap_enforced_before_recovery_transition(self, tmp_path: Path, monkeypatch):
        """With ``max_retries=1`` the cap is enforced BEFORE the
        ``RECOVERY_TRANSITION`` event is emitted, so the ledger records the
        action actually taken (``terminate``), not the would-be retry.

        Sequence: attempt 0 fails -> RECOVERY_TRANSITION action=backoff_retry,
        retry; attempt 1 fails -> cap says TERMINATE -> RECOVERY_TRANSITION
        action=terminate (NOT backoff_retry).  We assert the last recovery
        event's action is ``terminate`` exactly, exercising boundary accuracy
        and ledger/event fidelity.
        """
        import asyncio

        from northstack.application.worker import WorkerResult

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
        # Both attempts fail transiently. max_retries=1 allows exactly one
        # retry; the second failure is capped to TERMINATE before emission.
        worker = SequenceWorker(
            [
                WorkerResult(ok=False, error="temporary 429", error_kind="rate_limit"),
                WorkerResult(ok=False, error="temporary 429", error_kind="rate_limit"),
                WorkerResult(ok=False, error="temporary 429", error_kind="rate_limit"),
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
                    goal="cap-then-terminate",
                    workspace_root=str(tmp_path),
                    budget=Budget(
                        token_limit=1000,
                        cost_limit_usd=1.0,
                        max_retries=1,
                    ),
                )
            )
            assert outcome == RunOutcome.FAILED
            run_id = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchone()["run_id"]
            events = ledger.events(run_id)
            recovery = [e for e in events if e.kind == EventKind.RECOVERY_TRANSITION]
            assert recovery, "no RECOVERY_TRANSITION emitted"
            actions = [e.payload.action.value for e in recovery]
            # First failure: within cap -> backoff_retry.
            assert actions[0] == "backoff_retry", (
                f"first recovery action must be backoff_retry, got {actions[0]!r}"
            )
            # Second failure: cap exhausted -> TERMINATE recorded (the actual
            # action taken), NOT backoff_retry. This is the core fix: the
            # ledger records the action actually taken, not the uncapped one.
            assert actions[-1] == "terminate", (
                "RECOVERY_TRANSITION must record action=terminate once the "
                f"retry cap is hit; got {actions[-1]!r}, full actions={actions}. "
                "The cap must be enforced BEFORE emitting RECOVERY_TRANSITION."
            )
            # Boundary: exactly one retry was attempted (two worker calls
            # total: the initial attempt + one retry).
            assert len(worker.calls) == 2, (
                f"max_retries=1 allows exactly one retry (2 worker calls); got {len(worker.calls)}"
            )
        finally:
            ledger.close()

    def test_retries_used_by_one_cell_do_not_leak_to_another(self, tmp_path: Path, monkeypatch):
        """Recovery escalation state must not leak across cells -- the
        ``RetryPolicy`` dedup counter is keyed by the real strategy signature
        (contract_version, cell_id, profile, tool_plan, evidence_digest), so
        cell 1 escalating its ladder must not advance cell 2's rung.

        Two cells share one profile and an identical (empty-strategy)
        signature per cell. Each fails with a TRANSIENT (``rate_limit``)
        error. Under the single-owner ``RetryPolicy``, each cell independently
        walks its ladder: failure 0 -> ``BACKOFF_RETRY`` (retry same profile),
        failure 1 -> ``REROUTE_ESCALATE`` (reroute excludes the only profile
        and abstains, which for a TRANSIENT failure degrades to one more
        ``BACKOFF_RETRY`` rather than killing the cell), failure 2 ->
        ``TERMINATE`` on the exhausted cap. Both cells must walk the identical
        ladder -- proving cell 1's escalation did not pre-consume cell 2's
        rung. Under a buggy run-wide counter, cell 2 would start at rung 1+
        and terminate early.
        """
        import asyncio

        from northstack.application.worker import WorkerResult

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
        contract = WorkContract(
            id="wc-multi",
            version=1,
            objective="two cells",
            scope="test",
            deliverables=["out.py"],
            budget=Budget(token_limit=10_000, cost_limit_usd=5.0, max_retries=2),
            acceptance_criteria=[
                CommandCriterion(
                    description="check",
                    command_name="check",
                    exit_code=0,
                )
            ],
        )
        first = GraphCell(
            id="first",
            name="first",
            mode="read_only",
            contract=contract,
            acceptance_criterion_indices=[0],
        )
        # Independent cells in the same wave (no dependency) so BOTH run even
        # when 'first' fails -- the no-leak claim is about per-cell escalation
        # state, not dependency ordering. Both are read_only so they execute
        # concurrently in one gather, each emitting its own recovery events.
        second = GraphCell(
            id="second",
            name="second",
            mode="read_only",
            contract=contract,
            acceptance_criterion_indices=[0],
        )
        graph = GraphVersion(
            version=1,
            cells=[first, second],
            edges=[],
            milestones=["first", "second"],
            current_horizon=0,
        )
        # Each cell burns its own max_retries=2 cap: 3 worker calls apiece,
        # walking rung 0 -> 1 -> exhausted INDEPENDENTLY of the other cell.
        worker = SequenceWorker(
            [WorkerResult(ok=False, error="fail", error_kind="rate_limit") for _ in range(6)]
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
            company._planner = StaticGraphPlanner(graph)
            outcome = company.run(
                ProjectRequest(
                    goal="two cells",
                    workspace_root=str(tmp_path),
                    max_waves=2,
                    # The compiled contract's budget is sourced from the
                    # request; the per-cell contract budget on the graph cells
                    # is NOT what _run_cell consults. Set max_retries here so
                    # the compiled (run) contract carries the per-cell cap.
                    budget=Budget(token_limit=10_000, cost_limit_usd=5.0, max_retries=2),
                )
            )
            assert outcome == RunOutcome.FAILED
            run_id = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchone()["run_id"]
            events = ledger.events(run_id)
            recovery = [e for e in events if e.kind == EventKind.RECOVERY_TRANSITION]

            # Group recovery actions by cell.
            by_cell: dict[str, list[str]] = {}
            for e in recovery:
                cid = e.payload.cell_id
                by_cell.setdefault(cid, []).append(e.payload.action.value)

            # Each cell independently walks its ladder to rung 1:
            # failure 0 -> backoff_retry, failure 1 -> reroute_escalate (which,
            # with the only profile excluded, abstains). Cell 1 reaching rung 1
            # must NOT have advanced cell 2's rung -- the signature-keyed dedup
            # counter is per-cell, not run-wide.
            ladder = ["backoff_retry", "reroute_escalate", "backoff_retry", "terminate"]
            for cid in ("first", "second"):
                assert by_cell.get(cid) == ladder, (
                    f"cell {cid!r} must walk {ladder} independently -- proving one "
                    f"cell's escalation did not leak across cells; got {by_cell.get(cid)!r}"
                )
            assert len(worker.calls) == 6, (
                f"each cell must get its own worker calls; got {len(worker.calls)}"
            )
        finally:
            ledger.close()

    def test_zero_retries_boundary_with_explicit_one_cap(self, tmp_path: Path, monkeypatch):
        """Boundary accuracy for the retry-cap decision: ``max_retries=1``
        allows exactly one retry. The first failure emits
        ``action=backoff_retry`` (within cap); the second failure emits
        ``action=terminate`` (cap exhausted). The cap is enforced before
        emission, so the event sequence is exact and not off-by-one.
        """
        import asyncio

        from northstack.application.worker import WorkerResult

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
                WorkerResult(ok=False, error="fail", error_kind="rate_limit"),
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
                    goal="boundary",
                    workspace_root=str(tmp_path),
                    budget=Budget(
                        token_limit=1000,
                        cost_limit_usd=1.0,
                        max_retries=1,
                    ),
                )
            )
            assert outcome == RunOutcome.VERIFIED
            run_id = ledger._conn.execute("SELECT run_id FROM events LIMIT 1").fetchone()["run_id"]
            events = ledger.events(run_id)
            recovery = [e for e in events if e.kind == EventKind.RECOVERY_TRANSITION]
            # Exactly one recovery event, within the cap: backoff_retry.
            assert [e.payload.action.value for e in recovery] == ["backoff_retry"], (
                "with max_retries=1 and a single failure, exactly one "
                f"RECOVERY_TRANSITION backoff_retry must be recorded; got "
                f"{[e.payload.action.value for e in recovery]!r}"
            )
        finally:
            ledger.close()


# CLI integration tests


class TestCLI:
    def test_inspect_command(self, tmp_path: Path):
        from typer.testing import CliRunner

        from northstack.interfaces.cli import app

        runner = CliRunner()
        db_path = tmp_path / "test.db"

        ledger = Ledger(path=db_path)
        ledger.append_next(
            "run-1",
            RequestAccepted(goal="test", workspace_root="/tmp"),
        )
        ledger.append_next("run-1", OutcomeEmitted(outcome=RunOutcome.VERIFIED))
        ledger.close()

        result = runner.invoke(
            app,
            [
                "inspect",
                "--db",
                str(db_path),
                "--run-id",
                "run-1",
            ],
        )
        assert result.exit_code == 0
        assert "test" in result.output
        assert "verified" in result.output

    def test_replay_command(self, tmp_path: Path):
        from typer.testing import CliRunner

        from northstack.interfaces.cli import app

        runner = CliRunner()
        db_path = tmp_path / "test.db"

        ledger = Ledger(path=db_path)
        ledger.append_next("run-1", RunCreated())
        ledger.append_next("run-1", StatusChanged(status=RunStatus.CONTRACTED))
        ledger.close()

        result = runner.invoke(
            app,
            [
                "replay",
                "--db",
                str(db_path),
                "--run-id",
                "run-1",
            ],
        )
        assert result.exit_code == 0
        assert "Integrity: OK" in result.output
        assert "contracted" in result.output


# Tool-definition contracts (regression for the live 400 bug)


class TestBuildToolDefs:
    """The live CLI run used to HTTP 400 on every worker call because:

    1. Workplace tool parameters were empty (``{"type":"object","properties":{}}``)
       -- rejected by the OpenAI-compatible proxy upstream as
       ``invalid_request_error``.
    2. ``cmd_*`` command-profile tools were built BOTH in
       ``Company._build_tool_defs`` AND appended again by ``NativeWorker`` --
       duplicate tool names, also rejected by the upstream.

    These tests pin the fix: real per-tool schemas and a single source of
    cmd tools.
    """

    def test_allowed_tools_filter_at_advertisement(self, tmp_path: Path):
        """``contract.allowed_tools`` is honoured at the orchestrator seam: a
        tool absent from ``allowed_tools`` is never advertised to the model.
        The orchestrator filters the registry's advertised set down to the
        contract's allow-list, so the worker never has to re-filter and a tool
        the contract did not grant cannot reach the model at all."""
        company = self._company_with_commands(tmp_path)
        contract = self._minimal_contract(tmp_path)
        contract = contract.model_copy(update={"allowed_tools": ["read", "list", "cmd_lint"]})
        defs = company._build_tool_defs(contract)
        names = {d.name for d in defs}
        # Exactly the allowed set is advertised -- nothing more, nothing less.
        assert names == {"read", "list", "cmd_lint"}, names
        # A disallowed mutating tool is never offered.
        assert "write" not in names
        assert "cmd_test" not in names
        assert "web_fetch" not in names

    def test_empty_allowed_tools_advertises_everything(self, tmp_path: Path):
        """An empty ``allowed_tools`` (the default, meaning "no restriction") must
        advertise the full registry set, not zero tools -- otherwise the
        contract's "no restriction" intent would silently strip every tool."""
        company = self._company_with_commands(tmp_path)
        contract = self._minimal_contract(tmp_path)  # allowed_tools=[]
        defs = company._build_tool_defs(contract)
        names = {d.name for d in defs}
        # The full registry set (workspace tools + web_fetch + cmd_*).
        assert {"read", "write", "create", "replace", "list", "search", "web_fetch"} <= names
        assert {"cmd_lint", "cmd_test", "cmd_format"} <= names

    def _company_with_commands(self, tmp_path: Path) -> Company:
        from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace

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
        ledger = Ledger(path=tmp_path / "test.db")
        store = ArtifactStore(tmp_path / "artifacts")
        return Company(
            config=config,
            ledger=ledger,
            artifact_store=store,
            workspace=RestrictedWorkspace(tmp_path / "workspace"),
            gateway=None,
            worker_factory=None,  # type: ignore[arg-type]
            compiler=ContractCompiler(
                analysis_runner=DeterministicAnalysisRunner(),
                tool_registry=[
                    "read",
                    "write",
                    "create",
                    "replace",
                    "list",
                    "search",
                    "cmd_lint",
                    "cmd_test",
                    "cmd_format",
                ],
            ),
            command_profiles={
                "lint": CommandProfile(name="lint", argv=["ruff", "check", "."]),
                "test": CommandProfile(name="test", argv=["pytest", "-q"]),
                "format": CommandProfile(name="format", argv=["ruff", "format", "."]),
            },
            tool_registry=ToolRegistry.with_defaults(
                command_profiles={
                    "lint": CommandProfile(name="lint", argv=["ruff", "check", "."]),
                    "test": CommandProfile(name="test", argv=["pytest", "-q"]),
                    "format": CommandProfile(name="format", argv=["ruff", "format", "."]),
                }
            ),
        )

    def _minimal_contract(self, tmp_path: Path) -> WorkContract:
        return WorkContract(
            id="wc-1",
            objective="x",
            scope="s",
            deliverables=["deliverable_1"],
            budget=Budget(token_limit=100_000, cost_limit_usd=5.0),
            allowed_tools=[],
            acceptance_criteria=[
                CommandCriterion(
                    description="check",
                    command_name="check",
                    exit_code=0,
                )
            ],
        )

    def test_no_duplicate_tool_names(self, tmp_path: Path):
        company = self._company_with_commands(tmp_path)
        defs = company._build_tool_defs(self._minimal_contract(tmp_path))
        names = [d.name for d in defs]
        # No tool may be advertised twice. The live CLI once 400'd because BOTH
        # orchestration and the worker built cmd_* tools, producing duplicate
        # names the upstream rejected; now the single registry advertises each
        # tool exactly once (orchestration advertises, the worker no longer
        # rebuilds), so there can be no duplicates.
        assert len(names) == len(set(names)), f"duplicate tool names: {names}"
        # Each configured command profile yields exactly one cmd_* tool.
        for cmd in ("cmd_lint", "cmd_test", "cmd_format"):
            assert names.count(cmd) == 1, f"{cmd} advertised {names.count(cmd)} times"
        # web_fetch is advertised (closing the former dispatch/advertise gap).
        assert "web_fetch" in names, names
        # The six workspace tools are all present.
        assert {"read", "write", "create", "replace", "list", "search"} <= set(names)

    def test_workspace_tools_have_real_parameter_schemas(self, tmp_path: Path):
        company = self._company_with_commands(tmp_path)
        defs = company._build_tool_defs(self._minimal_contract(tmp_path))
        by_name = {d.name: d for d in defs}
        # Every advertised tool carries a real JSON-schema object with an
        # explicit ``required`` key. ``cmd_*`` tools legitimately advertises an
        # empty properties object (with ``required: []``) -- an empty properties
        # object *alone* is the 400 trigger some upstreams reject, so the
        # registry keeps the explicit ``required`` list; that is what we assert
        # here for every tool. The data-bearing workspace tools must additionally
        # declare their real argument properties (the 400 trigger for them would
        # be empty args the worker then cannot read).
        for name, d in by_name.items():
            assert d.parameters.get("type") == "object", f"{name}: {d.parameters}"
            assert "required" in d.parameters, f"{name} missing 'required': {d.parameters}"
            # cmd_* take no args -> empty properties is correct; everything else
            # must declare at least one real argument property.
            if not name.startswith("cmd_"):
                props = d.parameters.get("properties", {})
                assert props, f"{name} has empty properties: {d.parameters}"
        # Spot-check arg names match what the registry's tools read at execute.
        assert "path" in by_name["read"].parameters["properties"]
        assert set(by_name["write"].parameters["properties"]) == {"path", "content"}
        assert set(by_name["replace"].parameters["properties"]) == {"path", "old", "new"}
        assert "pattern" in by_name["search"].parameters["properties"]
        assert "url" in by_name["web_fetch"].parameters["properties"]
