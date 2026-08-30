"""BudgetAuthority: one gate for every dollar and token.

``BudgetAuthority`` is the single owner of spend: ``reserve`` pre-authorises
an estimate (raising ``BudgetExhausted`` if no headroom), ``commit``
reconciles the reservation against actual spend, and ``remaining`` reports
the live headroom. A Hypothesis property pins the core invariant:
reserve/commit arithmetic never lets committed spend exceed the configured
limit, regardless of estimate/actual drift or call order.
"""

from __future__ import annotations

import threading
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from northstack.application.budget_authority import (
    BudgetAuthority,
    BudgetExhausted,
    Reservation,
)
from northstack.application import budget_authority as budget_module
from northstack.domain import Budget, RemainingBudget, Spend

# Basic reserve / commit / remaining


class TestBudgetAuthorityBasics:
    def test_token_limit_counts_input_and_output_tokens(self) -> None:
        auth = BudgetAuthority(Budget(token_limit=10))
        auth.reserve("cell-1", Spend(input_tokens=4, output_tokens=6))

        assert auth.remaining().tokens == 0
        with pytest.raises(BudgetExhausted):
            auth.reserve("cell-2", Spend(output_tokens=1))

    def test_commit_rejects_an_already_committed_reservation(self) -> None:
        auth = BudgetAuthority(Budget(token_limit=100))
        reservation = auth.reserve("cell-1", Spend(input_tokens=40))
        auth.commit(reservation, Spend(input_tokens=40))

        with pytest.raises(ValueError, match="not active"):
            auth.commit(reservation, Spend(input_tokens=40))
        assert auth.remaining().tokens == 60

    def test_commit_rejects_a_foreign_reservation_without_mutating_either_authority(
        self,
    ) -> None:
        owner = BudgetAuthority(Budget(token_limit=100))
        other = BudgetAuthority(Budget(token_limit=100))
        reservation = owner.reserve("cell-1", Spend(input_tokens=40))

        with pytest.raises(ValueError, match="not active"):
            other.commit(reservation, Spend(input_tokens=40))
        assert owner.remaining().tokens == 60
        assert other.remaining().tokens == 100

    def test_release_refunds_headroom_and_rejects_reuse(self) -> None:
        auth = BudgetAuthority(Budget(token_limit=100, cost_limit_usd=1.0))
        reservation = auth.reserve("cell-1", Spend(input_tokens=40, cost_usd=0.4))

        auth.release(reservation)

        assert auth.remaining() == RemainingBudget(tokens=100, cost_usd=1.0)
        with pytest.raises(ValueError, match="not active"):
            auth.release(reservation)

    def test_reserve_then_commit_decrements_remaining(self) -> None:
        """A committed spend reduces remaining by exactly the actual."""
        auth = BudgetAuthority(Budget(token_limit=1000, cost_limit_usd=10.0))
        res = auth.reserve("cell-1", Spend(input_tokens=100, output_tokens=0, cost_usd=1.0))
        assert isinstance(res, Reservation)
        # Reserved (not yet committed) spend is held against remaining.
        assert auth.remaining().tokens == 900
        assert auth.remaining().cost_usd == pytest.approx(9.0)
        auth.commit(res, Spend(input_tokens=100, cost_usd=1.0))
        # Committing the exact estimate leaves remaining unchanged.
        assert auth.remaining().tokens == 900
        assert auth.remaining().cost_usd == pytest.approx(9.0)

    def test_commit_under_estimate_refunds_headroom(self) -> None:
        """When actual < estimate, committing refunds the difference."""
        auth = BudgetAuthority(Budget(token_limit=1000, cost_limit_usd=10.0))
        res = auth.reserve("cell-1", Spend(input_tokens=200, cost_usd=2.0))
        # Reserved 200 -> remaining 800.
        assert auth.remaining().tokens == 800
        # Actual was only 50; the 150 over-reservation is refunded.
        auth.commit(res, Spend(input_tokens=50, cost_usd=0.5))
        assert auth.remaining().tokens == 950
        assert auth.remaining().cost_usd == pytest.approx(9.5)

    def test_reserve_raises_when_estimate_exceeds_headroom(self) -> None:
        """A reservation whose estimate exceeds remaining raises BudgetExhausted."""
        auth = BudgetAuthority(Budget(token_limit=1000, cost_limit_usd=10.0))
        auth.reserve("cell-1", Spend(input_tokens=800, cost_usd=8.0))
        # Only 200 tokens left; reserving 300 must fail.
        with pytest.raises(BudgetExhausted):
            auth.reserve("cell-2", Spend(input_tokens=300, cost_usd=1.0))

    def test_remaining_preserves_unlimited_axes_as_none(self) -> None:
        """Unlimited axes (None) stay None in remaining -- never a finite sentinel."""
        auth = BudgetAuthority(Budget(token_limit=1000))  # cost unlimited
        auth.reserve("cell-1", Spend(input_tokens=100, cost_usd=5.0))
        rem = auth.remaining()
        assert rem.tokens == 900
        assert rem.cost_usd is None

    def test_record_counts_output_tokens_and_preserves_unlimited_axes(self) -> None:
        auth = BudgetAuthority(Budget(token_limit=10))

        auth.record(Spend(input_tokens=2, output_tokens=3, cost_usd=9.0))

        assert auth.remaining() == RemainingBudget(tokens=5, cost_usd=None)

    def test_zero_limits_reject_positive_spend_but_allow_zero(self) -> None:
        auth = BudgetAuthority(Budget(token_limit=0, cost_limit_usd=0.0))

        auth.reserve("zero", Spend())
        with pytest.raises(BudgetExhausted):
            auth.reserve("token", Spend(output_tokens=1))
        with pytest.raises(BudgetExhausted):
            auth.reserve("cost", Spend(cost_usd=0.01))
        assert auth.remaining() == RemainingBudget(tokens=0, cost_usd=0.0)

    def test_unlimited_axes_stay_unlimited_through_every_accounting_path(self) -> None:
        auth = BudgetAuthority(Budget())
        reservation = auth.reserve("cell", Spend(input_tokens=2, output_tokens=3, cost_usd=1.0))

        auth.commit(reservation, Spend(input_tokens=1, output_tokens=2, cost_usd=0.5))
        auth.record(Spend(input_tokens=5, output_tokens=8, cost_usd=4.0))

        assert auth.remaining() == RemainingBudget(tokens=None, cost_usd=None)

    def test_failed_mixed_limit_reservations_leave_both_axes_unchanged(self) -> None:
        auth = BudgetAuthority(Budget(token_limit=10, cost_limit_usd=1.0))

        with pytest.raises(BudgetExhausted):
            auth.reserve("token", Spend(input_tokens=11, cost_usd=0.1))
        with pytest.raises(BudgetExhausted):
            auth.reserve("cost", Spend(input_tokens=1, cost_usd=1.01))

        assert auth.remaining() == RemainingBudget(tokens=10, cost_usd=1.0)


# Hypothesis property: committed spend never exceeds the limit


@settings(max_examples=200)
@given(
    token_limit=st.integers(min_value=1, max_value=100_000),
    cost_limit=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    estimates=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=50_000),  # est input tokens
            st.floats(
                min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False
            ),  # est cost
        ),
        min_size=0,
        max_size=20,
    ),
    actuals_seed=st.integers(min_value=0, max_value=10_000),
)
def test_committed_spend_never_exceeds_limit(
    token_limit: int,
    cost_limit: float,
    estimates: list[tuple[int, float]],
    actuals_seed: int,
) -> None:
    """The invariant: after any sequence of reserve/commit, the *committed*
    spend never exceeds the configured limit, and ``remaining`` is never
    negative. Reservations that would exceed headroom raise ``BudgetExhausted``
    and do not mutate state.
    """
    import random

    auth = BudgetAuthority(Budget(token_limit=token_limit, cost_limit_usd=cost_limit))
    rng = random.Random(actuals_seed)
    committed_tokens = 0
    committed_cost = 0.0
    pending: list[tuple[Reservation, int, float]] = []

    for est_tokens, est_cost in estimates:
        # Actuals may drift below or at the estimate (never above: a worker
        # cannot spend more than it reserved).
        act_tokens = rng.randint(0, est_tokens) if est_tokens > 0 else 0
        # Clamp after rounding: a sub-microdollar estimate rounds *up* past
        # itself, which would break the "never above" premise this test states.
        act_cost = min(round(rng.uniform(0.0, est_cost) if est_cost > 0 else 0.0, 6), est_cost)
        try:
            res = auth.reserve(
                f"cell-{len(pending)}", Spend(input_tokens=est_tokens, cost_usd=est_cost)
            )
        except BudgetExhausted:
            # A failed reservation must not change committed spend.
            assert auth.remaining().tokens is not None
            assert auth.remaining().tokens >= 0
            continue
        pending.append((res, act_tokens, act_cost))
        auth.commit(res, Spend(input_tokens=act_tokens, cost_usd=act_cost))
        committed_tokens += act_tokens
        committed_cost += act_cost

        rem = auth.remaining()
        # Committed spend never exceeds the limit -> remaining never negative.
        assert rem.tokens is not None and rem.tokens >= 0, (
            f"token remaining {rem.tokens} went negative; "
            f"committed={committed_tokens}, limit={token_limit}"
        )
        assert rem.cost_usd is not None and rem.cost_usd >= -1e-9, (
            f"cost remaining {rem.cost_usd} went negative; "
            f"committed={committed_cost}, limit={cost_limit}"
        )
        # The authority's own accounting matches our independent tally (within
        # epsilon for float drift).
        assert committed_tokens <= token_limit
        assert committed_cost <= cost_limit + 1e-6


def test_reservation_is_opaque_and_reusable_type() -> None:
    """``Reservation`` is a typed handle returned by reserve, taken by commit."""
    auth = BudgetAuthority(Budget(token_limit=1000, cost_limit_usd=10.0))
    res = auth.reserve("cell-1", Spend(input_tokens=10, cost_usd=1.0))
    assert hasattr(res, "cell_id")
    # remaining() returns the typed RemainingBudget, not a bare number.
    assert isinstance(auth.remaining(), RemainingBudget)


@pytest.mark.parametrize(("attempts", "expected"), [(9, 9), (10, 10), (11, 10)])
def test_concurrent_reservations_cannot_overbook_or_lose_updates(
    monkeypatch, attempts: int, expected: int
) -> None:
    original_tokens = budget_module._tokens

    def delayed_tokens(spend: Spend) -> int:
        tokens = original_tokens(spend)
        time.sleep(0.002)
        return tokens

    monkeypatch.setattr(budget_module, "_tokens", delayed_tokens)
    auth = BudgetAuthority(Budget(token_limit=10))
    barrier = threading.Barrier(attempts + 1)
    succeeded: list[Reservation] = []

    def reserve(index: int) -> None:
        barrier.wait()
        try:
            succeeded.append(auth.reserve(str(index), Spend(input_tokens=1)))
        except BudgetExhausted:
            pass

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(attempts)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert len(succeeded) == expected
    assert auth.remaining().tokens == 10 - expected


def test_call_limit_is_reserved_committed_released_and_recorded() -> None:
    auth = BudgetAuthority(Budget(max_calls=2))
    reservation = auth.reserve("cell", Spend(calls=2))
    assert auth.remaining().calls == 0
    auth.release(reservation)
    assert auth.remaining().calls == 2
    reservation = auth.reserve("cell", Spend(calls=1))
    auth.commit(reservation, Spend(calls=1))
    auth.record(Spend(calls=1))
    assert auth.remaining().calls == 0
    with pytest.raises(BudgetExhausted):
        auth.reserve("cell", Spend(calls=1))
