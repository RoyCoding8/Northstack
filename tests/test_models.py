"""Tests for domain models at the public seam."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

from northstack.adapters.providers.wire import (
    FinishReason,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
)
from northstack.domain import (
    AcceptanceCriterion,
    ArtifactRef,
    Budget,
    BudgetUsage,
    CommandCriterion,
    CriterionKind,
    GraphCell,
    GraphVersion,
    RunState,
    RunStatus,
    WorkContract,
)
from northstack.events.catalog import StatusChanged
from tests.helpers.events import env


def _usage(*, tokens: int, cost: float) -> BudgetUsage:
    return BudgetUsage(total_input_tokens=tokens, total_cost_usd=cost)


# RunStatus: authoritative state machine


class TestRunStatus:
    def test_all_expected_statuses_exist(self):
        expected = {
            "intake",
            "contracted",
            "planned",
            "executing",
            "verifying",
            "verified",
            "abstained",
            "failed",
        }
        actual = {s.value for s in RunStatus}
        assert expected == actual

    @pytest.mark.parametrize(
        "from_s,to_s",
        [
            (RunStatus.INTAKE, RunStatus.CONTRACTED),
            (RunStatus.CONTRACTED, RunStatus.PLANNED),
            (RunStatus.PLANNED, RunStatus.EXECUTING),
            (RunStatus.EXECUTING, RunStatus.VERIFYING),
            (RunStatus.EXECUTING, RunStatus.FAILED),
            (RunStatus.VERIFYING, RunStatus.VERIFIED),
            (RunStatus.VERIFYING, RunStatus.ABSTAINED),
            (RunStatus.VERIFYING, RunStatus.FAILED),
        ],
    )
    def test_legal_transitions_accepted(self, from_s, to_s):
        assert RunStatus.can_transition(from_s, to_s) is True

    @pytest.mark.parametrize(
        "from_s,to_s",
        [
            (RunStatus.VERIFIED, RunStatus.EXECUTING),
            (RunStatus.INTAKE, RunStatus.VERIFIED),
            (RunStatus.FAILED, RunStatus.VERIFIED),
            (RunStatus.PLANNED, RunStatus.INTAKE),
            (RunStatus.ABSTAINED, RunStatus.EXECUTING),
            (RunStatus.EXECUTING, RunStatus.INTAKE),
            (RunStatus.EXECUTING, RunStatus.ABSTAINED),
        ],
    )
    def test_illegal_transitions_rejected(self, from_s, to_s):
        assert RunStatus.can_transition(from_s, to_s) is False

    def test_verified_is_terminal(self):
        for s in RunStatus:
            if s == RunStatus.VERIFIED:
                continue
            assert RunStatus.can_transition(RunStatus.VERIFIED, s) is False

    def test_abstained_is_terminal(self):
        for s in RunStatus:
            if s == RunStatus.ABSTAINED:
                continue
            assert RunStatus.can_transition(RunStatus.ABSTAINED, s) is False

    def test_failed_is_terminal(self):
        for s in RunStatus:
            if s == RunStatus.FAILED:
                continue
            assert RunStatus.can_transition(RunStatus.FAILED, s) is False

    def test_is_terminal(self):
        assert RunStatus.is_terminal(RunStatus.VERIFIED) is True
        assert RunStatus.is_terminal(RunStatus.ABSTAINED) is True
        assert RunStatus.is_terminal(RunStatus.FAILED) is True
        assert RunStatus.is_terminal(RunStatus.INTAKE) is False
        assert RunStatus.is_terminal(RunStatus.EXECUTING) is False


# RunStateMachine: the single owner of the transition table and the detour


class TestRunStateMachine:
    """The transition table and the EXECUTING->VERIFYING->ABSTAINED detour live
    in exactly one place (``RunStateMachine``), not patched inline in the
    orchestrator.  ADR 0001: ``EXECUTING -> ABSTAINED`` is illegal; budget
    exhaustion routes through ``VERIFYING``.
    """

    def test_executing_to_abstained_is_illegal_direct_edge(self):
        """ADR 0001: the direct EXECUTING -> ABSTAINED edge is forbidden."""
        from northstack.domain.status import RunStateMachine

        assert RunStateMachine.can_transition(RunStatus.EXECUTING, RunStatus.ABSTAINED) is False

    def test_executing_can_still_reach_verifying(self):
        from northstack.domain.status import RunStateMachine

        assert RunStateMachine.can_transition(RunStatus.EXECUTING, RunStatus.VERIFYING) is True

    def test_route_executing_to_abstained_goes_through_verifying(self):
        """The detour: EXECUTING -> VERIFYING -> ABSTAINED, in one place."""
        from northstack.domain.status import RunStateMachine

        path = RunStateMachine.route(RunStatus.EXECUTING, RunStatus.ABSTAINED)
        assert path == [RunStatus.VERIFYING, RunStatus.ABSTAINED]

    def test_route_direct_edge_is_a_single_step(self):
        """A legal direct edge routes as just the target (no detour)."""
        from northstack.domain.status import RunStateMachine

        assert RunStateMachine.route(RunStatus.INTAKE, RunStatus.CONTRACTED) == [
            RunStatus.CONTRACTED
        ]
        assert RunStateMachine.route(RunStatus.VERIFYING, RunStatus.VERIFIED) == [
            RunStatus.VERIFIED
        ]
        assert RunStateMachine.route(RunStatus.VERIFYING, RunStatus.FAILED) == [RunStatus.FAILED]

    def test_route_executing_to_failed_is_direct(self):
        """FAILED is reachable from EXECUTING directly (no detour needed)."""
        from northstack.domain.status import RunStateMachine

        assert RunStateMachine.route(RunStatus.EXECUTING, RunStatus.FAILED) == [RunStatus.FAILED]

    def test_route_illegal_transition_raises(self):
        """Routing an illegal (non-detourable) transition raises ValueError."""
        from northstack.domain.status import RunStateMachine

        # EXECUTING -> VERIFIED is illegal and not a detour case.
        with pytest.raises(ValueError):
            RunStateMachine.route(RunStatus.EXECUTING, RunStatus.VERIFIED)
        # Terminal -> anything is illegal.
        with pytest.raises(ValueError):
            RunStateMachine.route(RunStatus.VERIFIED, RunStatus.EXECUTING)

    def test_table_is_a_single_nested_dict(self):
        """The transition table is one nested dict owned by the state machine,
        so the EXECUTING -> VERIFYING -> ABSTAINED detour is encoded in exactly
        one place instead of being patched inline.
        """
        from northstack.domain.status import RunStateMachine

        table = RunStateMachine.transitions()
        assert isinstance(table, dict)
        # EXECUTING's neighbours do NOT include the illegal direct ABSTAINED.
        assert RunStatus.ABSTAINED not in table[RunStatus.EXECUTING]
        assert RunStatus.VERIFYING in table[RunStatus.EXECUTING]
        # Every non-terminal phase may still escape to ABSTAINED except EXECUTING.
        for frm in (RunStatus.INTAKE, RunStatus.CONTRACTED, RunStatus.PLANNED):
            assert RunStatus.ABSTAINED in table[frm]
        # Terminals have no outgoing edges.
        for terminal in (RunStatus.VERIFIED, RunStatus.ABSTAINED, RunStatus.FAILED):
            assert table[terminal] == frozenset()


# Budget (immutable)


class TestBudget:
    @pytest.mark.parametrize(
        ("model", "values"),
        [
            (Budget, {"cost_limit_usd": float("inf")}),
            (Budget, {"max_wall_time_seconds": float("inf")}),
            (BudgetUsage, {"total_cost_usd": float("inf")}),
        ],
    )
    def test_rejects_non_finite_resource_values(self, model, values):
        with pytest.raises(ValidationError):
            model(**values)

    def test_create_with_positive_limits(self):
        b = Budget(token_limit=100_000, cost_limit_usd=5.0)
        assert b.token_limit == 100_000
        assert b.cost_limit_usd == 5.0

    def test_negative_limits_rejected(self):
        with pytest.raises(ValidationError):
            Budget(token_limit=-1, cost_limit_usd=5.0)

    def test_extended_limits(self):
        b = Budget(
            token_limit=1000,
            cost_limit_usd=1.0,
            max_calls=50,
            max_tool_rounds=10,
            max_wall_time_seconds=300.0,
            max_retries=3,
        )
        assert b.max_calls == 50
        assert b.max_tool_rounds == 10
        assert b.max_wall_time_seconds == 300.0
        assert b.max_retries == 3

    def test_budget_is_frozen(self):
        b = Budget(token_limit=1000, cost_limit_usd=1.0)
        with pytest.raises(ValidationError):
            b.token_limit = 2000  # type: ignore[misc]

    def test_usage_within_limits_does_not_exceed(self):
        b = Budget(token_limit=1000, cost_limit_usd=1.0)
        assert _usage(tokens=500, cost=0.5).exceeds(b) is False

    def test_usage_exceeds_tokens(self):
        b = Budget(token_limit=1000, cost_limit_usd=1.0)
        assert _usage(tokens=1001, cost=0.5).exceeds(b) is True

    def test_usage_exceeds_cost(self):
        b = Budget(token_limit=1000, cost_limit_usd=1.0)
        assert _usage(tokens=100, cost=1.1).exceeds(b) is True

    def test_float_epsilon_does_not_fabricate_exhaustion(self):
        b = Budget(token_limit=None, cost_limit_usd=0.3)
        assert _usage(tokens=0, cost=0.1 + 0.2).exceeds(b) is False

    def test_none_axes_are_unlimited(self):
        b = Budget(token_limit=None, cost_limit_usd=None)

        assert _usage(tokens=10_000_000, cost=10_000.0).exceeds(b) is False

    def test_no_budget_never_exceeds(self):
        assert _usage(tokens=10_000_000, cost=10_000.0).exceeds(None) is False

    def test_remaining_preserves_unlimited_axes(self):
        from northstack.domain import BudgetUsage

        remaining = BudgetUsage(
            total_input_tokens=10,
            total_output_tokens=5,
            total_cost_usd=2.0,
        ).remaining(Budget(token_limit=None, cost_limit_usd=None))

        assert remaining.tokens is None
        assert remaining.cost_usd is None


# AcceptanceCriterion (immutable)


class TestAcceptanceCriterion:
    def test_command_criterion_round_trips(self):
        ac = CommandCriterion(
            description="lint clean",
            command_name="ruff",
            exit_code=0,
        )
        assert ac.kind == CriterionKind.COMMAND
        assert ac.command_name == "ruff"
        assert ac.exit_code == 0

    def test_criterion_is_frozen(self):
        ac = CommandCriterion(description="lint", command_name="ruff")
        with pytest.raises(ValidationError):
            ac.command_name = "mypy"  # type: ignore[misc]

    def test_extra_field_is_forbidden(self):
        with pytest.raises(ValidationError):
            CommandCriterion(command_name="ruff", satisfies=True)  # type: ignore[call-arg]

    def test_unknown_kind_cannot_be_constructed(self):
        """A bogus criterion kind must fail at parse time, never reach VERIFIED.

        The discriminated union rejects any kind outside the five known ones, so
        a malformed contract cannot slip through to the release law pretending
        one of its criteria was satisfied.
        """
        with pytest.raises(ValidationError):
            TypeAdapter(AcceptanceCriterion).validate_python(
                {"kind": "bogus_kind", "description": "x"}
            )


# WorkContract (immutable, expanded)


class TestWorkContract:
    def test_create_contract(self):
        wc = WorkContract(
            id="wc-1",
            objective="Implement feature X",
            deliverables=["code", "tests"],
            budget=Budget(token_limit=50_000, cost_limit_usd=2.0),
        )
        assert wc.id == "wc-1"
        assert wc.version == 1
        assert wc.objective == "Implement feature X"

    def test_contract_is_frozen(self):
        wc = WorkContract(
            id="wc-1",
            objective="Do thing",
            budget=Budget(token_limit=1000, cost_limit_usd=0.5),
        )
        with pytest.raises(ValidationError):
            wc.objective = "Changed"  # type: ignore[misc]

    def test_contract_full_fields(self):
        wc = WorkContract(
            id="wc-full",
            version=3,
            objective="Build API",
            scope="REST endpoints only",
            deliverables=["endpoints", "docs"],
            constraints=["no breaking changes"],
            assumptions=["existing DB schema"],
            forbidden_outcomes=["data loss"],
            allowed_tools=["ruff", "pytest"],
            workspace_scope="src/api/",
            budget=Budget(
                token_limit=100_000,
                cost_limit_usd=5.0,
                max_calls=200,
                max_wall_time_seconds=600,
                max_retries=5,
            ),
            acceptance_criteria=[
                CommandCriterion(description="All tests green", command_name="pytest"),
                CommandCriterion(description="No lint errors", command_name="ruff"),
            ],
            unresolved_ambiguity=["Auth mechanism TBD"],
            abstention_threshold=0.7,
        )
        assert wc.version == 3
        assert len(wc.acceptance_criteria) == 2
        assert wc.abstention_threshold == 0.7
        assert wc.budget.max_calls == 200

    def test_contract_requires_id_and_objective(self):
        with pytest.raises(ValidationError):
            WorkContract(budget=Budget(token_limit=100, cost_limit_usd=0.1))
        with pytest.raises(ValidationError):
            WorkContract(id="x", budget=Budget(token_limit=100, cost_limit_usd=0.1))


# EventEnvelope


class TestEventEnvelope:
    def test_create_event(self):
        ev = env(1, run_id="run-1")
        assert ev.run_id == "run-1"
        assert ev.seq == 1
        assert ev.schema_version == 1
        assert ev.hash_chain == ""  # ledger fills this in

    def test_event_with_empty_prev_hash_is_genesis(self):
        ev = env(1, prev_hash="")
        assert ev.prev_hash == ""

    def test_event_with_explicit_prev_hash(self):
        ev = env(2, StatusChanged(status=RunStatus.INTAKE), prev_hash="abc123")
        assert ev.prev_hash == "abc123"


# ArtifactRef


class TestArtifactRef:
    def test_create_artifact_ref(self):
        ar = ArtifactRef(
            digest="sha256:" + "a" * 64,
            media_type="application/json",
            size_bytes=1024,
        )
        assert ar.size_bytes == 1024

    def test_digest_format_validated(self):
        with pytest.raises(ValidationError):
            ArtifactRef(digest="not-a-digest", media_type="text/plain", size_bytes=1)


# RunState (projected)


class TestRunState:
    def test_create_empty_run_state(self):
        rs = RunState(run_id="run-1")
        assert rs.run_id == "run-1"
        assert rs.status == RunStatus.INTAKE
        assert rs.cells == []
        assert rs.events_replayed == 0

    def test_run_state_snapshot(self):
        rs = RunState(run_id="run-1")
        snap = rs.snapshot()
        assert snap["run_id"] == "run-1"
        assert snap["status"] == "intake"
        assert snap["events_replayed"] == 0


# Property-based: graph acyclicity on well-formed inputs


@given(
    cells=st.lists(
        st.fixed_dictionaries(
            {
                "id": st.text(
                    min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))
                ),
                "name": st.text(min_size=1, max_size=20),
            }
        ),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x["id"],
    ),
)
@settings(max_examples=50)
def test_graph_can_be_constructed_with_unique_ids(cells):
    """A graph with unique cell IDs should always construct successfully."""
    cell_objs = [
        GraphCell(
            id=c["id"],
            name=c["name"],
            contract=WorkContract(
                id=f"wc-{c['id']}",
                objective="t",
                budget=Budget(token_limit=100, cost_limit_usd=0.1),
            ),
        )
        for c in cells
    ]
    g = GraphVersion(cells=cell_objs, edges=[])
    assert len(g.cells) == len(cells)


def test_graph_cell_rejects_negative_wave() -> None:
    with pytest.raises(ValidationError):
        GraphCell(
            id="cell",
            name="Cell",
            wave=-1,
            contract=WorkContract(
                id="wc-cell",
                objective="test",
                budget=Budget(token_limit=100, cost_limit_usd=0.1),
            ),
        )


# MessageRole


class TestMessageRole:
    def test_all_roles_exist(self):
        expected = {"system", "user", "assistant", "tool"}
        actual = {r.value for r in MessageRole}
        assert expected == actual


# ToolDefinition


class TestToolDefinition:
    def test_create_tool_definition(self):
        td = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert td.name == "get_weather"
        assert td.description == "Get weather for a city"

    def test_tool_definition_frozen(self):
        td = ToolDefinition(name="test", description="test")
        with pytest.raises(ValidationError):
            td.name = "other"  # type: ignore[misc]

    def test_tool_definition_default_parameters(self):
        td = ToolDefinition(name="test")
        assert td.parameters == {"type": "object", "properties": {}}


# ToolCall


class TestToolCall:
    def test_create_tool_call(self):
        tc = ToolCall(id="tc-1", name="read", arguments={"path": "file.txt"})
        assert tc.id == "tc-1"
        assert tc.name == "read"
        assert tc.arguments == {"path": "file.txt"}

    def test_tool_call_frozen(self):
        tc = ToolCall(id="tc-1", name="read")
        with pytest.raises(ValidationError):
            tc.id = "other"  # type: ignore[misc]

    def test_tool_call_default_arguments(self):
        tc = ToolCall(id="tc-1", name="test")
        assert tc.arguments == {}


# ToolResultMessage


class TestToolResultMessage:
    def test_create_tool_result(self):
        tr = ToolResultMessage(tool_call_id="tc-1", content="result data")
        assert tr.tool_call_id == "tc-1"
        assert tr.content == "result data"
        assert tr.is_error is False

    def test_tool_result_error(self):
        tr = ToolResultMessage(tool_call_id="tc-1", content="error", is_error=True)
        assert tr.is_error is True


# ModelMessage


class TestModelMessage:
    def test_user_message(self):
        msg = ModelMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.tool_calls == []

    def test_assistant_with_tool_calls(self):
        msg = ModelMessage(
            role=MessageRole.ASSISTANT,
            content="Let me check",
            tool_calls=[ToolCall(id="tc1", name="read", arguments={"path": "."})],
        )
        assert len(msg.tool_calls) == 1

    def test_tool_result_message(self):
        msg = ModelMessage(
            role=MessageRole.TOOL,
            content="file contents",
            tool_call_id="tc1",
        )
        assert msg.tool_call_id == "tc1"

    def test_model_message_frozen(self):
        msg = ModelMessage(role=MessageRole.USER, content="Hi")
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]


# FinishReason


class TestFinishReason:
    def test_all_reasons_exist(self):
        expected = {"end_turn", "tool_use", "max_tokens", "stop_sequence", "error"}
        actual = {r.value for r in FinishReason}
        assert expected == actual


# Usage


class TestUsage:
    def test_usage_defaults(self):
        u = Usage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0

    def test_usage_total(self):
        u = Usage(input_tokens=100, output_tokens=50, cache_creation_tokens=10, cache_read_tokens=5)
        assert u.total_tokens == 150

    def test_usage_frozen(self):
        u = Usage(input_tokens=100)
        with pytest.raises(ValidationError):
            u.input_tokens = 200  # type: ignore[misc]


# ModelRequest


class TestModelRequest:
    def test_create_request(self):
        req = ModelRequest(
            profile_name="test-worker",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
        )
        assert req.profile_name == "test-worker"
        assert len(req.messages) == 1
        assert req.system == ""
        assert req.tools == []

    def test_request_with_tools(self):
        req = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            tools=[ToolDefinition(name="read", description="Read")],
        )
        assert len(req.tools) == 1

    def test_request_get_max_tokens_explicit(self):
        req = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
            max_output_tokens=2048,
        )
        assert req.get_max_tokens(4096) == 2048

    def test_request_get_max_tokens_fallback(self):
        req = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
        )
        assert req.get_max_tokens(4096) == 4096

    def test_request_frozen(self):
        req = ModelRequest(
            profile_name="test",
            messages=[ModelMessage(role=MessageRole.USER, content="Hi")],
        )
        with pytest.raises(ValidationError):
            req.profile_name = "other"  # type: ignore[misc]

    def test_request_requires_messages(self):
        with pytest.raises(ValidationError):
            ModelRequest(profile_name="test", messages=[])


# ModelResponse


class TestModelResponse:
    def test_create_response(self):
        resp = ModelResponse(
            text="Hello",
            finish_reason=FinishReason.END_TURN,
            provider="openai",
            model="gpt-4",
        )
        assert resp.text == "Hello"
        assert resp.finish_reason == FinishReason.END_TURN
        assert resp.tool_calls == []
        assert resp.response_artifact_id is None

    def test_response_with_tool_calls(self):
        resp = ModelResponse(
            text="",
            tool_calls=[ToolCall(id="tc1", name="read", arguments={"path": "."})],
            finish_reason=FinishReason.TOOL_USE,
            provider="anthropic",
            model="claude-3",
        )
        assert len(resp.tool_calls) == 1
        assert resp.finish_reason == FinishReason.TOOL_USE

    def test_response_with_artifact(self):
        resp = ModelResponse(
            text="done",
            finish_reason=FinishReason.END_TURN,
            provider="openai",
            model="gpt-4",
            response_artifact_id="sha256:" + "a" * 64,
        )
        assert resp.response_artifact_id is not None

    def test_response_frozen(self):
        resp = ModelResponse(
            text="hello",
            finish_reason=FinishReason.END_TURN,
            provider="openai",
            model="gpt-4",
        )
        with pytest.raises(ValidationError):
            resp.text = "changed"  # type: ignore[misc]


# type: ignore[misc]
