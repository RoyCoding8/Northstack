"""Model-backed contract falsification via the SPECIALIST role.

The falsifier searches for a *passing-but-wrong* interpretation of a
contract: a reading under which every stated criterion could be satisfied
while the request's actual intent is missed. A concrete counter-interpretation
rejects the contract at compile time (``ContractCompiler`` raises), forcing a
revised contract rather than a subtly wrong run.

Fail-open by design: an unavailable or unparseable falsifier returns ``None``
(no objection). A falsifier outage must not kill every run -- the hard gates
and the release law remain the safety properties; falsification is an added
quality check the operator opts into (``falsifier_mode = "model"``). The
asymmetry is the point: a *finding* is actionable, an *absence* caused by an
outage is indistinguishable from silence and is treated as such.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict

from northstack.adapters.providers.wire import MessageRole, ModelMessage, ModelRequest
from northstack.domain.contract import WorkContract
from northstack.domain.request import ProjectRequest
from northstack.ports.protocols import GatewayPort

logger = logging.getLogger(__name__)

_MAX_COUNTER_CHARS = 500

_SYSTEM_PROMPT = (
    "You are an adversarial contract reviewer. Your ONLY job is to find a "
    "concrete passing-but-wrong interpretation: a way the stated acceptance "
    "criteria could all pass while the requester's actual intent is missed. "
    "Be specific -- name the wrong deliverable or behavior that would still "
    "pass. If no such interpretation exists (the criteria pin the intent), "
    'respond with JSON {"counter_interpretation": "none"}. Respond with JSON '
    "only.\n\n"
    "A counter-interpretation must survive EVERY criterion, including any "
    "soft_rubric judged by independent blinded reviewers reading the actual "
    "workspace evidence. A scenario whose survival depends on reviewers being "
    "fooled (e.g. hardcoded outputs for the exact asserted inputs -- the thing "
    "the rubric exists to catch) is not a valid counter-interpretation; object "
    "only on wrongness no post-hoc reading of the work can distinguish.\n\n"
    "Execution-surface facts you must respect when judging reachability:\n"
    "- A 'command' criterion executes an operator-configured argv preset "
    "(e.g. the full pytest invocation). The executor CANNOT redefine, skip, "
    "or wrap named commands -- do not invent scenarios where it does.\n"
    "- Object only to wrongness reachable through ordinary changes to the "
    "workspace files the executor controls (product code, new files)."
)

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "counter_interpretation": {"type": "string", "maxLength": 1000},
    },
    "required": ["counter_interpretation"],
    "additionalProperties": False,
}


class _Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    counter_interpretation: str = "none"


class ModelBackedFalsifier:
    """Asks a SPECIALIST-role profile for a passing-but-wrong reading."""

    def __init__(
        self,
        gateway: GatewayPort,
        profile_name: str,
        *,
        max_output_tokens: int = 1024,
    ) -> None:
        self._gateway = gateway
        self._profile_name = profile_name
        self._max_output_tokens = max_output_tokens

    @property
    def profile_name(self) -> str:
        return self._profile_name

    async def check(self, contract: WorkContract, request: ProjectRequest) -> str | None:
        criteria_lines = "\n".join(
            f"  [{i}] {c.kind}: {c.description}" for i, c in enumerate(contract.acceptance_criteria)
        )
        prompt = (
            f"Original request: {request.goal}\n\n"
            f"Proposed contract objective: {contract.objective}\n"
            f"Deliverables: {', '.join(contract.deliverables)}\n"
            f"Constraints: {', '.join(contract.constraints) or '(none)'}\n"
            f"Acceptance criteria:\n{criteria_lines}\n\n"
            "Find a passing-but-wrong interpretation, or say none."
        )
        model_request = ModelRequest(
            profile_name=self._profile_name,
            system=_SYSTEM_PROMPT,
            messages=[ModelMessage(role=MessageRole.USER, content=prompt)],
            output_json_schema=_VERDICT_SCHEMA,
            max_output_tokens=self._max_output_tokens,
            temperature=0.0,
        )
        try:
            response = await self._gateway.complete(model_request)
        except Exception:  # noqa: BLE001 - falsifier outage fails open
            logger.warning("model falsifier gateway failure profile=%s", self._profile_name)
            return None
        return self._parse(getattr(response, "text", "") or "")

    @staticmethod
    def _parse(text: str) -> str | None:
        """Extract the counter-interpretation, or None when sound/unparseable."""
        stripped = text.strip()
        if not stripped:
            return None
        candidate = stripped
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            if len(lines) > 1:
                candidate = "\n".join(lines[1:]).removesuffix("```").strip()
        start = candidate.find("{")
        end = candidate.rfind("}")
        counter = ""
        if 0 <= start < end:
            try:
                parsed = json.loads(candidate[start : end + 1])
                counter = (
                    str(parsed.get("counter_interpretation", ""))
                    if isinstance(parsed, dict)
                    else ""
                )
            except json.JSONDecodeError:
                counter = ""
        if not counter:
            counter = candidate
        normalized = counter.strip()
        if not normalized or normalized.lower() in {"none", "n/a", "null", "no"}:
            return None
        return normalized[:_MAX_COUNTER_CHARS]
