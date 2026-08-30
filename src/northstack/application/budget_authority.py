"""One gate for every dollar and token.

``BudgetAuthority`` is the single owner of spend: remaining budget is a
distinct type from cumulative spend, and the authority speaks
``RemainingBudget`` / ``Spend``, never a bare number. ``reserve`` pre-
authorises an estimate (raising :class:`BudgetExhausted` if no headroom),
``commit`` reconciles the reservation against actual spend, and
``remaining`` reports live headroom.

The invariant (pinned by a Hypothesis property): reserve/commit arithmetic
never lets committed spend exceed the configured limit. A failed reservation
does not mutate state.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from threading import RLock

from northstack.domain import Budget, RemainingBudget, Spend

_ids = itertools.count(1)


def _tokens(spend: Spend) -> int:
    return spend.input_tokens + spend.output_tokens


class BudgetExhausted(Exception):
    """Raised when a reservation estimate exceeds remaining headroom."""


class InvalidReservation(ValueError):
    """Raised when a reservation is foreign, forged, or no longer active."""


@dataclass(frozen=True)
class Reservation:
    """Opaque handle returned by :meth:`BudgetAuthority.reserve`.

    Carries the cell the reservation is for and the estimate it was authorised
    against, so :meth:`BudgetAuthority.commit` can release the held headroom
    and reconcile it against actual spend.
    """

    reservation_id: int
    cell_id: str
    estimate: Spend


@dataclass
class BudgetAuthority:
    """Single owner of spend against one :class:`Budget`.

    ``remaining()`` reports ``limit - (committed + reserved)``: committed spend
    plus the estimates held by outstanding reservations, so a reservation that
    has not yet committed still counts against headroom (you cannot double-book
    the same budget). On commit the held estimate is released and the actual
    spend is added to the committed total; if actual exceeds estimate the
    excess is refused (a worker cannot spend more than it reserved).
    """

    budget: Budget
    _committed_tokens: int = field(default=0, init=False)
    _committed_cost: float = field(default=0.0, init=False)
    _reserved_tokens: int = field(default=0, init=False)
    _reserved_cost: float = field(default=0.0, init=False)
    _committed_calls: int = field(default=0, init=False)
    _reserved_calls: int = field(default=0, init=False)
    _reservations: dict[int, Reservation] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def _require_active(self, reservation: Reservation) -> Spend:
        if self._reservations.get(reservation.reservation_id) is not reservation:
            raise InvalidReservation(f"reservation {reservation.reservation_id} is not active")
        return reservation.estimate

    def reserve(self, cell_id: str, estimate: Spend) -> Reservation:
        """Pre-authorise ``estimate`` against remaining headroom.

        Raises :class:`BudgetExhausted` if the estimate would push committed +
        reserved spend over a set limit. A failed reservation leaves state
        unchanged.
        """
        with self._lock:
            if self.budget.token_limit is not None and (
                self._committed_tokens + self._reserved_tokens + _tokens(estimate)
                > self.budget.token_limit
            ):
                raise BudgetExhausted(
                    f"token budget exhausted: reserved+committed="
                    f"{self._committed_tokens + self._reserved_tokens}, "
                    f"estimate={_tokens(estimate)}, limit={self.budget.token_limit}"
                )
            if self.budget.cost_limit_usd is not None and (
                self._committed_cost + self._reserved_cost + estimate.cost_usd
                > self.budget.cost_limit_usd + 1e-9
            ):
                raise BudgetExhausted(
                    f"cost budget exhausted: reserved+committed="
                    f"{self._committed_cost + self._reserved_cost:.6f}, "
                    f"estimate={estimate.cost_usd:.6f}, limit={self.budget.cost_limit_usd}"
                )
            if self.budget.max_calls and (
                self._committed_calls + self._reserved_calls + estimate.calls
                > self.budget.max_calls
            ):
                raise BudgetExhausted(
                    f"call budget exhausted: reserved+committed="
                    f"{self._committed_calls + self._reserved_calls}, "
                    f"estimate={estimate.calls}, limit={self.budget.max_calls}"
                )
            reservation = Reservation(reservation_id=next(_ids), cell_id=cell_id, estimate=estimate)
            self._reserved_tokens += _tokens(estimate)
            self._reserved_cost += estimate.cost_usd
            self._reserved_calls += estimate.calls
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def commit(self, reservation: Reservation, actual: Spend) -> None:
        """Reconcile ``reservation`` against ``actual`` spend.

        Releases the held estimate and records the actual. A worker cannot
        spend more than it reserved: actual exceeding the estimate raises
        :class:`BudgetExhausted` without mutating state, so the over-spend is
        rejected rather than silently absorbed.
        """
        with self._lock:
            est = self._require_active(reservation)
            if (
                _tokens(actual) > _tokens(est)
                or actual.cost_usd > est.cost_usd + 1e-9
                or actual.calls > est.calls
            ):
                raise BudgetExhausted(
                    f"actual spend exceeds reservation for {reservation.cell_id}: "
                    f"actual=(tokens={_tokens(actual)}, cost={actual.cost_usd:.6f}), "
                    f"estimate=(tokens={_tokens(est)}, cost={est.cost_usd:.6f})"
                )
            self._reserved_tokens -= _tokens(est)
            self._reserved_cost -= est.cost_usd
            self._reserved_calls -= est.calls
            self._committed_tokens += _tokens(actual)
            self._committed_cost += actual.cost_usd
            self._committed_calls += actual.calls
            del self._reservations[reservation.reservation_id]

    def release(self, reservation: Reservation) -> None:
        """Cancel one active reservation and refund all held headroom."""
        with self._lock:
            estimate = self._require_active(reservation)
            self._reserved_tokens -= _tokens(estimate)
            self._reserved_cost -= estimate.cost_usd
            self._reserved_calls -= estimate.calls
            del self._reservations[reservation.reservation_id]

    def record(self, actual: Spend) -> None:
        """Record post-hoc actual spend with no prior reservation.

        The orchestrator learns spend *after* the worker has already made the
        call (the worker reports actuals, not estimates), so there is no
        estimate to reserve against. ``record`` is that post-hoc seam: it adds
        ``actual`` straight to committed spend without holding or releasing a
        reservation, and never raises -- a single call can overshoot a set
        limit (the worker does not pre-flight), so enforcement is the
        wave-boundary ``remaining()`` check that abstains once headroom is
        gone, not a refusal here. ``reserve``/``commit`` remain the pre-flight
        path with their stricter ``actual <= estimate`` invariant.
        """
        with self._lock:
            self._committed_tokens += _tokens(actual)
            self._committed_cost += actual.cost_usd
            self._committed_calls += actual.calls

    def remaining(self) -> RemainingBudget:
        """Live headroom: ``limit - (committed + reserved)``.

        Unlimited axes (``None``) stay ``None`` -- never a finite sentinel.
        """
        with self._lock:
            return RemainingBudget(
                tokens=(
                    max(
                        0,
                        self.budget.token_limit - self._committed_tokens - self._reserved_tokens,
                    )
                    if self.budget.token_limit is not None
                    else None
                ),
                cost_usd=(
                    max(
                        0.0,
                        self.budget.cost_limit_usd - self._committed_cost - self._reserved_cost,
                    )
                    if self.budget.cost_limit_usd is not None
                    else None
                ),
                calls=(
                    max(0, self.budget.max_calls - self._committed_calls - self._reserved_calls)
                    if self.budget.max_calls
                    else None
                ),
            )
