"""Property-based tests for the pure domain layer (budget, status, graph)."""

from __future__ import annotations

from itertools import pairwise, product

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from northstack.domain.budget import Budget, BudgetUsage
from northstack.domain.graph import CellMode, CellStatus, GraphCell, GraphVersion
from northstack.domain.status import RunStateMachine, RunStatus


budget_strategy = st.builds(
    Budget,
    token_limit=st.one_of(st.none(), st.integers(min_value=0, max_value=10**9)),
    cost_limit_usd=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=10**6, allow_nan=False, allow_infinity=False),
    ),
    max_calls=st.integers(min_value=0, max_value=10**6),
    max_tool_rounds=st.integers(min_value=0, max_value=10**6),
    max_wall_time_seconds=st.floats(
        min_value=0.0, max_value=10**6, allow_nan=False, allow_infinity=False
    ),
    max_retries=st.integers(min_value=0, max_value=10**6),
)

usage_strategy = st.builds(
    BudgetUsage,
    total_input_tokens=st.integers(min_value=0, max_value=10**9),
    total_output_tokens=st.integers(min_value=0, max_value=10**9),
    total_cost_usd=st.floats(min_value=0.0, max_value=10**6, allow_nan=False, allow_infinity=False),
    total_calls=st.integers(min_value=0, max_value=10**6),
)

_id = st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N")))

cell_strategy = st.builds(
    GraphCell,
    id=_id,
    name=st.text(max_size=20),
    wave=st.integers(min_value=0, max_value=100),
    mode=st.sampled_from(CellMode),
    status=st.sampled_from(CellStatus),
    dependencies=st.lists(_id, max_size=5),
    acceptance_criterion_indices=st.lists(st.integers(min_value=0, max_value=20), max_size=5),
)

graph_strategy = st.builds(
    GraphVersion,
    version=st.integers(min_value=1, max_value=100),
    cells=st.lists(cell_strategy, min_size=1, max_size=10, unique_by=lambda c: c.id),
)


# Budget


@given(budget=budget_strategy, usage=usage_strategy)
@settings(max_examples=200)
def test_b1_remaining_never_negative(budget, usage):
    """No field of remaining is ever negative (unlimited axes stay None)."""
    remaining = usage.remaining(budget)
    assert remaining.tokens is None or remaining.tokens >= 0
    assert remaining.cost_usd is None or remaining.cost_usd >= 0.0
    assert remaining.calls is None or remaining.calls >= 0


@given(usage=usage_strategy)
@settings(max_examples=200)
def test_b2_exceeds_none_false(usage):
    """exceeds(None) is always False."""
    assert usage.exceeds(None) is False


@given(
    budget=budget_strategy,
    u1=usage_strategy,
    d_in=st.integers(min_value=0, max_value=10**6),
    d_out=st.integers(min_value=0, max_value=10**6),
    d_cost=st.floats(min_value=0.0, max_value=10**6, allow_nan=False, allow_infinity=False),
    d_calls=st.integers(min_value=0, max_value=10**5),
)
@settings(max_examples=200)
def test_b3_monotonicity(budget, u1, d_in, d_out, d_cost, d_calls):
    """Increasing totals never makes an exceeded budget become unexceeded."""
    u2 = BudgetUsage(
        total_input_tokens=u1.total_input_tokens + d_in,
        total_output_tokens=u1.total_output_tokens + d_out,
        total_cost_usd=u1.total_cost_usd + d_cost,
        total_calls=u1.total_calls + d_calls,
    )
    if u1.exceeds(budget):
        assert u2.exceeds(budget)


@given(usage=usage_strategy)
@settings(max_examples=200)
def test_b4_open_budget_never_exceeded(usage):
    """An open budget (no token/cost limit, no call cap) is never exceeded."""
    b = Budget(token_limit=None, cost_limit_usd=None, max_calls=0)
    assert usage.exceeds(b) is False


@given(usage=usage_strategy)
@settings(max_examples=200)
def test_b5_total_tokens_sum(usage):
    """total_tokens is always the sum of input and output tokens."""
    assert usage.total_tokens == usage.total_input_tokens + usage.total_output_tokens


@given(token_limit=st.integers(min_value=0, max_value=10**9), usage=usage_strategy)
@settings(max_examples=200)
def test_b6_remaining_tokens_under_limit(token_limit, usage):
    """Under the token limit, remaining.tokens equals the headroom."""
    b = Budget(token_limit=token_limit, cost_limit_usd=None, max_calls=0)
    if usage.total_tokens <= token_limit:
        remaining = usage.remaining(b)
        assert remaining.tokens is not None
        assert remaining.tokens == token_limit - usage.total_tokens


def test_b7_frozen_and_default():
    """Budget is frozen and from_payload(None) equals default()."""
    b = Budget(token_limit=1000, cost_limit_usd=1.0)
    with pytest.raises(ValidationError):
        b.token_limit = 2000
    with pytest.raises(ValidationError):
        b.cost_limit_usd = 2.0
    assert Budget.from_payload(None) == Budget.default()


@given(budget=budget_strategy)
@settings(max_examples=200)
def test_b8_round_trip(budget):
    """Budget round-trips through model_dump -> from_payload."""
    assert Budget.from_payload(budget.model_dump()) == budget


# Status


def test_s1_predicates_agree():
    """RunStatus and RunStateMachine expose identical predicates over all pairs."""
    for a, b in product(RunStatus, repeat=2):
        assert RunStatus.can_transition(a, b) == RunStateMachine.can_transition(a, b)
    for s in RunStatus:
        assert RunStatus.is_terminal(s) == RunStateMachine.is_terminal(s)


def test_s2_terminals_have_no_outgoing():
    """A terminal status has no outgoing transitions to any status."""
    for t in RunStatus:
        if not RunStateMachine.is_terminal(t):
            continue
        for s in RunStatus:
            assert RunStateMachine.can_transition(t, s) is False


@given(
    a=st.sampled_from(list(RunStatus)),
    b=st.sampled_from(list(RunStatus)),
)
@settings(max_examples=300)
def test_s3_route_respects_transitions(a, b):
    """Every returned route walks only legal edges and ends at the target."""
    try:
        path = RunStateMachine.route(a, b)
    except ValueError:
        return
    assert path
    full = [a] + path
    for x, y in pairwise(full):
        assert RunStateMachine.can_transition(x, y)
    assert path[-1] is b


@given(
    a=st.sampled_from(list(RunStatus)),
    b=st.sampled_from(list(RunStatus)),
)
@settings(max_examples=300)
def test_s4_route_matches_can_transition(a, b):
    """A legal direct edge routes as exactly [b]."""
    if RunStateMachine.can_transition(a, b):
        assert RunStateMachine.route(a, b) == [b]


def test_s5_executing_to_abstained_detour():
    """The documented EXECUTING -> VERIFYING -> ABSTAINED detour holds."""
    assert RunStateMachine.route(RunStatus.EXECUTING, RunStatus.ABSTAINED) == [
        RunStatus.VERIFYING,
        RunStatus.ABSTAINED,
    ]


def test_s6_string_round_trip():
    """Every RunStatus round-trips through its string value."""
    for s in RunStatus:
        assert RunStatus(s.value) is s


# Graph


@given(graph=graph_strategy, new_status=st.sampled_from(CellStatus))
@settings(max_examples=100)
def test_g1_returns_new_object(graph, new_status):
    """with_cell_status returns a new object and leaves the original untouched."""
    target = graph.cells[0]
    original_status = target.status
    result = graph.with_cell_status(target.id, new_status)
    assert result is not graph
    assert graph.cells[0].status == original_status


@given(graph=graph_strategy, new_status=st.sampled_from(CellStatus))
@settings(max_examples=100)
def test_g2_only_targeted_changes(graph, new_status):
    """Exactly the targeted cell's status changes; every other cell is equal."""
    target = graph.cells[0]
    if new_status == target.status:
        return
    result = graph.with_cell_status(target.id, new_status)
    for c in result.cells:
        orig = next(o for o in graph.cells if o.id == c.id)
        if c.id == target.id:
            assert c.status == new_status
        else:
            assert c == orig


@given(graph=graph_strategy, new_status=st.sampled_from(CellStatus))
@settings(max_examples=100)
def test_g3_non_status_fields_unchanged(graph, new_status):
    """Every non-status field of the targeted cell is preserved."""
    target = graph.cells[0]
    result = graph.with_cell_status(target.id, new_status)
    res = next(c for c in result.cells if c.id == target.id)
    for field in GraphCell.model_fields:
        if field == "status":
            continue
        assert getattr(target, field) == getattr(res, field)


@given(graph=graph_strategy, new_status=st.sampled_from(CellStatus))
@settings(max_examples=100)
def test_g4_idempotent(graph, new_status):
    """Applying with_cell_status twice with the same status is idempotent."""
    target = graph.cells[0]
    once = graph.with_cell_status(target.id, new_status)
    twice = once.with_cell_status(target.id, new_status)
    assert twice == once


@given(graph=graph_strategy, new_status=st.sampled_from(CellStatus))
@settings(max_examples=100)
def test_g5_unknown_cell_id(graph, new_status):
    """An unknown cell id silently returns an unchanged copy (no raise)."""
    result = graph.with_cell_status("definitely-not-a-cell-id", new_status)
    assert result == graph


def test_g6_empty_id_rejected():
    """GraphCell requires a non-empty id."""
    with pytest.raises(ValidationError):
        GraphCell(id="")
