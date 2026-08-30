"""Budget limits, cumulative usage, and remaining headroom.

``Budget`` is the operator-set limit, ``BudgetUsage`` is cumulative spend, and
``RemainingBudget`` is what is left.  Keeping remaining structurally distinct
from cumulative spend stops a caller inverting a low-budget guard.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Budget(BaseModel):
    """Resource budgets for a cell or run.

    Immutable: once set, limits cannot be changed.  Usage tracking lives in the
    ledger/event stream, not in the model.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    token_limit: int | None = Field(
        default=None, ge=0, description="Max total tokens allowed; None = unlimited"
    )
    cost_limit_usd: float | None = Field(
        default=None, ge=0.0, description="Max total USD cost allowed; None = unlimited"
    )
    max_calls: int = Field(default=0, ge=0, description="Max API calls (0 = unlimited)")
    max_tool_rounds: int = Field(
        default=0, ge=0, description="Max tool-call rounds (0 = unlimited)"
    )
    max_wall_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Max wall-clock seconds (0 = unlimited)"
    )
    max_retries: int = Field(default=0, ge=0, description="Max retry attempts per failed step")

    @classmethod
    def default(cls) -> Budget:
        """The fallback budget (matches RunConfig defaults). Single source."""
        return cls(token_limit=100_000, cost_limit_usd=5.0)

    @classmethod
    def from_payload(cls, data: Mapping[str, Any] | None) -> Budget:
        """Build from a ledger payload, falling back to the default.

        Validates rather than splatting: a payload carrying an unknown or
        mistyped field must fail here, not silently produce a wrong limit.
        """
        return cls.model_validate(dict(data)) if data else cls.default()


class Spend(BaseModel):
    """Resources consumed by a single attempt or call.

    Distinct from ``BudgetUsage`` (cumulative, run-scoped) so a per-cell figure
    can never be mistaken for a running total.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    calls: int = Field(default=0, ge=0)


class RemainingBudget(BaseModel):
    """Remaining limits, preserving unlimited axes as ``None``."""

    model_config = ConfigDict(allow_inf_nan=False)

    tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)
    calls: int | None = Field(default=None, ge=0)


class BudgetUsage(BaseModel):
    """Cumulative resource usage for tracking against budget."""

    model_config = ConfigDict(allow_inf_nan=False)

    total_input_tokens: int = Field(default=0, ge=0)
    total_output_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_calls: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def remaining(self, budget: Budget) -> RemainingBudget:
        """Compute remaining limits without inventing finite sentinels."""
        return RemainingBudget(
            tokens=(
                max(0, budget.token_limit - self.total_tokens)
                if budget.token_limit is not None
                else None
            ),
            cost_usd=(
                max(0.0, budget.cost_limit_usd - self.total_cost_usd)
                if budget.cost_limit_usd is not None
                else None
            ),
            calls=(max(0, budget.max_calls - self.total_calls) if budget.max_calls else None),
        )

    def exceeds(self, budget: Budget | None) -> bool:
        """True if cumulative usage has crossed a limit that was actually set.

        Unlimited axes (``None``) never trip.  Float comparison carries an
        epsilon so accumulated rounding cannot fabricate an exhaustion.
        """
        if budget is None:
            return False
        if budget.token_limit is not None and self.total_tokens > budget.token_limit:
            return True
        return bool(
            budget.cost_limit_usd is not None and self.total_cost_usd > budget.cost_limit_usd + 1e-9
        )
