"""Inspectable, rule-based routing of a cell onto a model profile."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from northstack.config import Capability, ModelProfile, NorthStackConfig
from northstack.domain.budget import RemainingBudget
from northstack.domain.contract import WorkContract
from northstack.domain.graph import CellMode, GraphCell

_ROLE_ORDER_WEIGHT = 10.0
_ROLE_ORDER_BASE = 1000.0
_CONCURRENCY_BONUS = 0.5
_TIER_SCORE: dict[CellMode, dict[int, float]] = {
    CellMode.MUTATING: {1: 1.0, 2: 2.0, 3: 3.0},
    CellMode.READ_ONLY: {1: 3.0, 2: 2.0, 3: 1.0},
}
_TIER3_MIN_REMAINING_USD = 1.0


class RouteDecision(BaseModel):
    """Output of routing: candidate scores, reasons, and selection."""

    model_config = ConfigDict(frozen=True)

    cell_id: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    selected_profile: str = Field(default="")
    reason: str = Field(default="")
    abstained: bool = Field(default=False)


class Router:
    """Inspectable rule-based router."""

    def __init__(self, config: NorthStackConfig) -> None:
        self._config = config
        self._role_map: dict[str, list[str]] = {
            role.value: names for role, names in config.role_map().items()
        }

    def route(
        self,
        cell: GraphCell,
        contract: WorkContract,
        remaining_budget: RemainingBudget | None = None,
        excluded_profiles: set[str] | None = None,
        require_capabilities: set[Capability] | None = None,
    ) -> RouteDecision:
        excluded_profiles = excluded_profiles or set()
        required_caps = set(require_capabilities or ())
        for cap in cell.required_capabilities:
            try:
                required_caps.add(Capability(cap))
            except ValueError:
                pass
        candidates = [
            score
            for profile in self._config.profiles
            if profile.name not in excluded_profiles
            for score in (self._score_profile(profile, cell, remaining_budget, required_caps),)
            if score is not None
        ]

        if not candidates:
            return RouteDecision(
                cell_id=cell.id, reason="no eligible profile found", abstained=True
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        return RouteDecision(
            cell_id=cell.id,
            candidates=candidates,
            selected_profile=best["profile_name"],
            reason=best["reason"],
            abstained=False,
        )

    def _score_profile(
        self,
        profile: ModelProfile,
        cell: GraphCell,
        remaining_budget: RemainingBudget | None,
        required_caps: set[Capability],
    ) -> dict[str, Any] | None:
        required_roles = set(cell.required_profile_roles)

        role_order: dict[str, int] = {}
        if self._role_map and required_roles:
            for role_name in required_roles:
                ordered = self._role_map.get(role_name) or []
                if profile.name in ordered:
                    role_order[role_name] = ordered.index(profile.name)
            if not role_order:
                return None
        elif required_roles and not (required_roles & {r.value for r in profile.roles}):
            return None

        if not required_caps <= set(profile.capabilities):
            return None

        if (
            remaining_budget
            and profile.tier >= 3
            and remaining_budget.cost_usd is not None
            and 0 < remaining_budget.cost_usd < _TIER3_MIN_REMAINING_USD
        ):
            return None

        score = 0.0
        reasons: list[str] = []

        if role_order:
            best_order = min(role_order.values())
            score += (_ROLE_ORDER_BASE - best_order) * _ROLE_ORDER_WEIGHT
            reasons.append(f"role order {best_order} for {sorted(role_order)}")
        else:
            mode = cell.mode if cell.mode in _TIER_SCORE else CellMode.READ_ONLY
            score += _TIER_SCORE[mode].get(profile.tier, 1.0)
            reasons.append(f"tier-{profile.tier} for {mode.value}")

        if profile.max_concurrency > 1:
            score += _CONCURRENCY_BONUS
            reasons.append("high concurrency")

        return {
            "profile_name": profile.name,
            "score": score,
            "reason": "; ".join(reasons),
            "tier": profile.tier,
        }
