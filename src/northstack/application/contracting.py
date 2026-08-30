"""Contract compilation pipeline.

Public seam:
  - ContractCompiler.compile(request, workspace, ledger, config) -> WorkContract
  - ContractValidator.validate(contract, tool_registry) -> list[str]
  - AnalysisRunner protocol for injectable analysis (tests use deterministic fakes)

Flow:
  1. Three blinded analyses fan out in parallel (requirements, repo constraints,
     acceptance/risk/ambiguity).
  2. Expert synthesis produces an immutable WorkContract.
  3. Independent falsifier searches for a passing-but-wrong interpretation.
  4. ContractValidator enforces structural invariants before the contract is accepted.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import keyword
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.intake_scan import conventions_from_scan, scan_workspace
from northstack.application.json_extraction import extract_first_json_object
from northstack.application.tools.registry import ToolRegistry
from northstack.application.verification.hard_gates import compute_tree_digest
from northstack.config import ModelProfile, NorthStackConfig, Role
from northstack.config import Protocol as ConfigProtocol
from northstack.domain import (
    AcceptanceCriterion,
    Budget,
    CommandCriterion,
    CriterionKind,
    ProjectRequest,
    SoftRubricCriterion,
    WorkContract,
)
from northstack.ports.protocols import GatewayPort
from northstack.events.catalog import (
    AnalysisCompleted,
    AnalysisRequested,
    ContractProposed,
    ContractValidated,
)
from northstack.events.stream import EventStream

logger = logging.getLogger(__name__)

_JSON_RETRY_FACTOR = 4

_CRITERION_ADAPTER: TypeAdapter[AcceptanceCriterion] = TypeAdapter(AcceptanceCriterion)


def _criterion_from_dict(c: dict[str, Any], index: int) -> AcceptanceCriterion:
    """Build one typed criterion from a legacy analysis dict.

    The model analysis emits ``{kind, description, parameters: {...}}``; the
    typed union carries those as flat fields, so the nested ``parameters`` dict
    is merged into the top level before validation. A bogus kind raises a
    ``ValidationError`` here rather than reaching the release law.
    """
    kind_str = c.get("kind", CriterionKind.SOFT_RUBRIC.value)
    try:
        CriterionKind(kind_str)
    except ValueError as exc:
        raise ValueError(f"criterion {index} has unknown kind '{kind_str}'") from exc

    flat: dict[str, Any] = {"kind": kind_str}
    flat["description"] = c.get("description", f"criterion_{index}")
    params = c.get("parameters", {})
    if not isinstance(params, dict):
        raise TypeError(f"criterion {index} parameters must be a dict, got {type(params)!r}")
    if kind_str == CriterionKind.COMMAND.value and "command_name" not in params:
        for alias in ("command", "argv", "cmd"):
            if isinstance(params.get(alias), str):
                params = {
                    "command_name": params[alias],
                    **{k: v for k, v in params.items() if k != alias},
                }
                break
    flat.update(params)
    try:
        return _CRITERION_ADAPTER.validate_python(flat)
    except ValidationError as exc:
        extras = {str(e["loc"][-1]) for e in exc.errors() if e["type"] == "extra_forbidden"}
        if extras:
            try:
                return _CRITERION_ADAPTER.validate_python(
                    {k: v for k, v in flat.items() if k not in extras}
                )
            except ValidationError:
                pass
        raise ValueError(f"criterion {index} could not be parsed: {exc}") from exc


_FALLBACK_PROFILE = ModelProfile(
    name="default",
    protocol=ConfigProtocol.OPENAI_CHAT,
    base_url="http://localhost",
    model="default",
    max_concurrency=1,
)


class RequirementsAnalysis(BaseModel):
    """Output of requirements/scope analysis."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    scope: str = Field(default="")
    deliverables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)


class RepoAnalysis(BaseModel):
    """Output of repository constraints/conventions analysis."""

    model_config = ConfigDict(frozen=True)

    workspace_scope: str = Field(default="")
    conventions: list[str] = Field(default_factory=list)
    existing_patterns: list[str] = Field(default_factory=list)
    tool_restrictions: list[str] = Field(default_factory=list)
    scan_digest: str = Field(
        default="",
        description="Digest of the deterministic intake workspace scan, when one ran",
    )


class AcceptanceAnalysis(BaseModel):
    """Output of acceptance/risk/ambiguity analysis."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    criteria: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    recommended_abstention_threshold: float = Field(default=0.5)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _drop_ungrounded_content_checks(
    criteria: list[dict[str, Any]], goal: str, root: Path
) -> list[dict[str, Any]]:
    """Strip ``content_contains`` needles the goal never asked for.

    The analyzer routinely invents identifiers -- an exact test function name,
    one particular spelling of a guard -- and a hard gate on an invented name
    fails a worker whose output is correct but named differently. A needle
    survives only when every identifier in it already appears in the goal or in
    the file the criterion targets; a criterion left asserting nothing but an
    existence check another criterion already makes is dropped outright.
    """
    out: list[dict[str, Any]] = []
    asserted_exists: set[str] = set()
    for c in criteria:
        params = c["parameters"]
        path = str(params.get("path", ""))
        needle = params.get("content_contains")
        if c["kind"] == CriterionKind.FILE_DIFF.value and isinstance(needle, str) and needle:
            corpus = goal
            with contextlib.suppress(OSError):
                corpus += (root / path).read_text(encoding="utf-8", errors="ignore")
            if any(t not in corpus for t in _IDENT_RE.findall(needle) if not keyword.iskeyword(t)):
                params = {k: v for k, v in params.items() if k != "content_contains"}
                if path in asserted_exists and not (
                    params.get("content_hash") or params.get("content_equals")
                ):
                    continue
                c = {**c, "parameters": params}
        if c["kind"] == CriterionKind.FILE_DIFF.value and params.get("must_exist"):
            asserted_exists.add(path)
        out.append(c)
    return out


class AnalysisRunner(Protocol):
    """Protocol for injectable analysis execution.

    All methods are async to support real model-backed runners.
    Tests provide a DeterministicAnalysisRunner that returns fixture data.
    Production uses a ModelBackedAnalysisRunner that calls the gateway.
    """

    async def run_requirements(
        self, request: ProjectRequest, profile: ModelProfile
    ) -> RequirementsAnalysis: ...

    async def run_repo(self, request: ProjectRequest, profile: ModelProfile) -> RepoAnalysis: ...

    async def run_acceptance(
        self, request: ProjectRequest, profile: ModelProfile
    ) -> AcceptanceAnalysis: ...


class DeterministicAnalysisRunner:
    """Returns fixed fixture data for deterministic testing (async)."""

    def __init__(
        self,
        requirements: RequirementsAnalysis | None = None,
        repo: RepoAnalysis | None = None,
        acceptance: AcceptanceAnalysis | None = None,
    ) -> None:
        self._requirements = requirements or RequirementsAnalysis(
            scope="default scope",
            deliverables=["deliverable_1"],
        )
        self._repo = repo or RepoAnalysis()
        self._acceptance = acceptance or AcceptanceAnalysis()

    async def run_requirements(
        self, request: ProjectRequest, profile: ModelProfile
    ) -> RequirementsAnalysis:
        return self._requirements

    async def run_repo(self, request: ProjectRequest, profile: ModelProfile) -> RepoAnalysis:
        return self._repo

    async def run_acceptance(
        self, request: ProjectRequest, profile: ModelProfile
    ) -> AcceptanceAnalysis:
        return self._acceptance


_ACCEPTANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["command", "file_diff", "policy", "soft_rubric"],
                    },
                    "description": {"type": "string"},
                    "parameters": {"type": "object"},
                },
                "required": ["kind", "description", "parameters"],
                "additionalProperties": False,
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "recommended_abstention_threshold": {"type": "number"},
    },
    "required": ["criteria", "risks", "ambiguities"],
    "additionalProperties": False,
}

_REQUIREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scope": {"type": "string"},
        "deliverables": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "ambiguities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["scope", "deliverables"],
    "additionalProperties": False,
}


_FILE_CONTENT_GOAL = re.compile(
    r"""(?ix)
    \b(?:create|write|make)\b          # verb
    (?:\s+a)?\s+file
    (?:\s+(?:called|named|at))?\s+
    ['"`]?
    (?P<path>[\w./\\-]+\.\w+)          # path with an extension
    ['"`]?
    .*?
    \b(?:contain(?:ing|s)?|with(?:\s+the)?\s+(?:text|content|string))\b
    \s*(?:the\s+text\s*)?
    ['"`]
    (?P<text>.+?)
    ['"`]
    """,
)


def _deterministic_file_content_analysis(goal: str) -> tuple[str, str] | None:
    """If the goal is an explicit 'create file X with text Y' instruction,
    return ``(path, text)`` so the runner can synthesize a ``file_diff``
    criterion without calling a model. Returns None for any other goal -- the
    model-backed path (or the soft-rubric fallback) then takes over.
    """
    m = _FILE_CONTENT_GOAL.search(goal)
    if not m:
        return None
    path = m.group("path").strip()
    text = m.group("text")
    if not path or not text:
        return None
    return path, text


_SENT = r"(?:[^\n.]|\.\S)"
_TEST_GOAL_RE = re.compile(
    rf"\b(?:tests?|suite)\b{_SENT}{{0,80}}\bpass(?:es|ing)?\b"
    rf"|\bpass(?:es|ing)?\b{_SENT}{{0,40}}\b(?:pytest|suite)\b"
    r"|\brun(?:ning)?\s+(?:the\s+)?tests?\b",
    re.IGNORECASE,
)

_PYTEST_TAMPER_FILES = ("conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml")


def _pytest_command_name(config: NorthStackConfig) -> str | None:
    """The configured command profile that runs pytest, if any."""
    for cmd in config.commands:
        if any("pytest" in part for part in cmd.argv):
            return cmd.name
    return None


class ModelBackedAnalysisRunner:
    """Production AnalysisRunner that calls the model gateway.

    Unlike the deterministic fixture, this asks the model to emit concrete,
    hard-checkable acceptance criteria (command / file_diff) for the goal.
    For simple deliverable-creation tasks the verifier can then reach
    ``verified`` purely from hard gates -- no soft rubric, no reviewers, no
    abstention. When the model cannot produce executable criteria it returns a
    single ``soft_rubric`` criterion, which preserves the abstention law
    (abstain rather than fake confidence).

    For goals that require no model judgment to specify acceptance -- e.g.
    "create a file called X containing the text 'Y'" -- a deterministic
    heuristic supplies the criterion without a model call. This follows the
    project's routing law ("deterministic operation when no model judgment is
    required") and makes trivial deliverable-creation runs reliably reach
    ``verified`` instead of depending on a weak model following a structured-
    output instruction.

    The runner is provider-neutral: it builds a ModelRequest and delegates to
    the injected gateway. It requests native JSON output when the profile
    advertises ``NATIVE_JSON_SCHEMA``; otherwise it asks for plain JSON in the
    text and parses it leniently.
    """

    def __init__(
        self,
        gateway: GatewayPort,
        profile_name: str,
        *,
        max_output_tokens: int = 1024,
        command_names: Sequence[str] = (),
    ) -> None:
        self._gateway = gateway
        self._profile_name = profile_name
        self._max_output_tokens = max_output_tokens
        self._command_names = list(command_names)

    async def run_requirements(
        self, request: ProjectRequest, profile: ModelProfile
    ) -> RequirementsAnalysis:
        det = _deterministic_file_content_analysis(request.goal)
        if det is not None:
            path, _text = det
            return RequirementsAnalysis(
                scope=request.goal,
                deliverables=[path],
            )
        prompt = (
            "You are a requirements analyst. Given the project goal, return a "
            "compact JSON object with keys: scope, deliverables (list of named "
            "deliverables), constraints, assumptions, ambiguities. Be concrete "
            "and minimal.\n\nGoal:\n"
            f"{request.goal}"
        )
        data = await self._complete_json(prompt, _REQUIREMENTS_SCHEMA)
        if not isinstance(data, dict):
            return RequirementsAnalysis(scope=request.goal)
        try:
            return RequirementsAnalysis(
                scope=data.get("scope", "") or request.goal,
                deliverables=data.get("deliverables") or [],
                constraints=data.get("constraints", []),
                assumptions=data.get("assumptions", []),
                ambiguities=data.get("ambiguities", []),
            )
        except Exception:  # noqa: BLE001
            return RequirementsAnalysis(scope=request.goal)

    async def run_repo(self, request: ProjectRequest, profile: ModelProfile) -> RepoAnalysis:
        scan = await asyncio.to_thread(scan_workspace, request.workspace_root, max_entries=400)
        conventions, patterns = conventions_from_scan(scan)
        return RepoAnalysis(
            workspace_scope=request.workspace_root,
            conventions=conventions,
            existing_patterns=patterns,
            tool_restrictions=[],
            scan_digest=scan.digest(),
        )

    async def run_acceptance(
        self, request: ProjectRequest, profile: ModelProfile
    ) -> AcceptanceAnalysis:
        det = _deterministic_file_content_analysis(request.goal)
        if det is not None:
            path, text = det
            return AcceptanceAnalysis(
                criteria=[
                    {
                        "kind": CriterionKind.FILE_DIFF.value,
                        "description": f"file '{path}' contains the required text",
                        "parameters": {
                            "path": path,
                            "must_exist": True,
                            "content_contains": text,
                        },
                    }
                ],
                risks=[],
                ambiguities=[],
                recommended_abstention_threshold=0.5,
            )
        scan = await asyncio.to_thread(scan_workspace, request.workspace_root, max_entries=200)
        stack_line = f"\n\nWorkspace facts: {scan.summary()}\n"
        command_line = (
            "one of " + ", ".join(repr(n) for n in self._command_names)
            if self._command_names
            else "<a known command profile>"
        )
        prompt = (
            "You are an acceptance-criteria engineer. Produce executable, "
            "hard-checkable acceptance criteria for the goal below. Prefer "
            "'file_diff' criteria (parameters: path, must_exist=true, "
            "content_contains=<exact substring the deliverable must contain>) "
            f"and 'command' criteria (parameters: command_name={command_line}, "
            "exit_code=0 -- a criterion asserting any other exit code is never "
            "useful). A 'soft_rubric' costs the run its verdict unless the "
            "reviewer panel was calibrated for that criterion's position, so "
            "reach for one only when no command or file check can express the "
            "check at all. In particular, 'the code/tests handle X' is a "
            "file_diff on that file with content_contains=<the token X appears "
            "as>, not a rubric. Return JSON: "
            "{criteria:[{kind,description,parameters}], risks:[], ambiguities:[], "
            "recommended_abstention_threshold:0.5}. For a file whose contents "
            "matter, always set content_contains to the exact required text."
            f"{stack_line}\nGoal:\n{request.goal}"
        )
        data = await self._complete_json(prompt, _ACCEPTANCE_SCHEMA)
        criteria = _drop_ungrounded_content_checks(
            self._coerce_criteria(data), request.goal, Path(request.workspace_root)
        )
        if not criteria:
            criteria = [
                {
                    "kind": CriterionKind.SOFT_RUBRIC.value,
                    "description": "default quality rubric",
                    "parameters": {},
                }
            ]
        return AcceptanceAnalysis(
            criteria=criteria,
            risks=(data or {}).get("risks", []) if isinstance(data, dict) else [],
            ambiguities=(data or {}).get("ambiguities", []) if isinstance(data, dict) else [],
            recommended_abstention_threshold=(
                (data or {}).get("recommended_abstention_threshold", 0.5)
                if isinstance(data, dict)
                else 0.5
            ),
        )

    async def _complete_json(self, prompt: str, schema: dict[str, Any]) -> Any:
        """Call the gateway and return parsed JSON, or None on any failure."""
        from northstack.adapters.providers.wire import (
            FinishReason,
            MessageRole,
            ModelMessage,
            ModelRequest,
        )

        for budget in (self._max_output_tokens, self._max_output_tokens * _JSON_RETRY_FACTOR):
            request = ModelRequest(
                profile_name=self._profile_name,
                messages=[ModelMessage(role=MessageRole.USER, content=prompt)],
                output_json_schema=schema,
                max_output_tokens=budget,
                temperature=0.0,
            )
            try:
                response = await self._gateway.complete(request)
            except Exception as exc:  # noqa: BLE001
                self._warn_json_fallback(f"gateway_failure:{type(exc).__name__}")
                return None

            text = (response.text or "").strip()
            candidate = extract_first_json_object(text)
            if candidate is not None:
                return candidate
            self._warn_json_fallback(
                "candidate_decode_failure" if "{" in text else "no_json_object", text
            )
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                self._warn_json_fallback("raw_text_parse_failure", text)

            if text and response.finish_reason is not FinishReason.MAX_TOKENS:
                break
        return None

    @staticmethod
    def _warn_json_fallback(stage: str, text: str = "") -> None:
        """Record a bounded parse stage without exposing model output."""
        logger.warning(
            "model JSON fallback stage=%s response_chars=%d",
            stage,
            len(text),
        )

    @staticmethod
    def _coerce_criteria(data: Any) -> list[dict[str, Any]]:
        """Validate the model's criteria list against CriterionKind.

        Drops any criterion whose kind is not a known CriterionKind or whose
        parameters is not a dict. Never raises.
        """
        if not isinstance(data, dict):
            return []
        raw = data.get("criteria")
        if not isinstance(raw, list):
            return []
        valid_kinds = {k.value for k in CriterionKind}
        out: list[dict[str, Any]] = []
        for c in raw:
            if not isinstance(c, dict):
                continue
            kind = c.get("kind")
            params = c.get("parameters")
            if kind not in valid_kinds or not isinstance(params, dict):
                continue
            out.append(
                {
                    "kind": kind,
                    "description": str(c.get("description", "")),
                    "parameters": params,
                }
            )
        return out


class ContractSynthesizer(Protocol):
    """Protocol for synthesizing a WorkContract from analysis results."""

    def synthesize(
        self,
        request: ProjectRequest,
        req_analysis: RequirementsAnalysis,
        repo_analysis: RepoAnalysis,
        acc_analysis: AcceptanceAnalysis,
        budget: Budget,
    ) -> WorkContract:
        """Produce a WorkContract from the three analyses."""
        ...


class DeterministicSynthesizer:
    """Produces a deterministic WorkContract from analyses."""

    def synthesize(
        self,
        request: ProjectRequest,
        req_analysis: RequirementsAnalysis,
        repo_analysis: RepoAnalysis,
        acc_analysis: AcceptanceAnalysis,
        budget: Budget,
    ) -> WorkContract:
        criteria = []
        for i, c in enumerate(acc_analysis.criteria):
            try:
                criteria.append(_criterion_from_dict(c, i))
            except (ValueError, TypeError) as exc:
                logger.warning("dropping unparsable acceptance criterion: %s", exc)

        if not criteria:
            criteria.append(
                SoftRubricCriterion(description="default quality rubric"),
            )

        deliverables = req_analysis.deliverables or ["deliverable_1"]

        return WorkContract(
            id=f"wc-{int(time.time() * 1000)}",
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


class Falsifier(Protocol):
    """Searches for a passing-but-wrong interpretation of the contract.

    Async: a model-backed falsifier (SPECIALIST role) asks an endpoint; a
    deterministic one returns without awaiting anything.
    """

    async def check(self, contract: WorkContract, request: ProjectRequest) -> str | None:
        """Return a counter-interpretation string, or None if contract is sound."""
        ...


class DeterministicFalsifier:
    """Always returns None (no counter-interpretation found)."""

    async def check(self, contract: WorkContract, request: ProjectRequest) -> str | None:
        return None


class ContractValidator:
    """Deterministic validator for WorkContract structural invariants.

    Requires:
      - Nonempty objective and deliverables
      - Every deliverable linked to at least one criterion
      - Allowed tools subset of actual tool registry
      - Workspace scope is config-contained
      - Budgets not exceeding request
      - Criterion kinds restricted to CriterionKind enum
      - Material ambiguity either resolved or triggers abstention
    """

    VALID_CRITERION_KINDS: frozenset[str] = frozenset(k.value for k in CriterionKind)

    def __init__(
        self,
        tool_registry: ToolRegistry | list[str] | None = None,
        command_names: list[str] | None = None,
    ) -> None:
        if isinstance(tool_registry, ToolRegistry):
            self._tool_registry = tool_registry.dispatchable_names()
        else:
            self._tool_registry = set(tool_registry or [])
        self._command_names = set(command_names or [])

    def validate(
        self,
        contract: WorkContract,
        request: ProjectRequest | None = None,
    ) -> list[str]:
        """Return list of validation error strings. Empty = valid."""
        errors: list[str] = []

        if not contract.objective.strip():
            errors.append("objective must be nonempty")

        if not contract.deliverables:
            errors.append("deliverables must be nonempty")

        if contract.deliverables and not contract.acceptance_criteria:
            errors.append("deliverables exist but no acceptance criteria defined")

        if self._tool_registry and contract.allowed_tools:
            unknown = set(contract.allowed_tools) - self._tool_registry
            if unknown:
                errors.append(f"allowed_tools references unknown tools: {sorted(unknown)}")

        if self._command_names:
            unknown_cmds = sorted(
                {
                    c.command_name
                    for c in contract.acceptance_criteria
                    if isinstance(c, CommandCriterion) and c.command_name not in self._command_names
                }
            )
            if unknown_cmds:
                errors.append(
                    f"command criteria reference unknown command profiles: {unknown_cmds}"
                )

        if request and request.budget:
            rb = request.budget
            cb = contract.budget
            if rb.token_limit is not None and (
                cb.token_limit is None or cb.token_limit > rb.token_limit
            ):
                errors.append(
                    f"contract token_limit {cb.token_limit} exceeds request {rb.token_limit}"
                )
            if rb.cost_limit_usd is not None and (
                cb.cost_limit_usd is None or cb.cost_limit_usd > rb.cost_limit_usd
            ):
                errors.append(
                    f"contract cost_limit_usd {cb.cost_limit_usd} exceeds "
                    f"request {rb.cost_limit_usd}"
                )

        for i, c in enumerate(contract.acceptance_criteria):
            if c.kind not in self.VALID_CRITERION_KINDS:
                errors.append(
                    f"criterion {i} has invalid kind '{c.kind}'; "
                    f"allowed: {sorted(self.VALID_CRITERION_KINDS)}"
                )

        if contract.unresolved_ambiguity and contract.abstention_threshold <= 0.0:
            errors.append(
                "unresolved ambiguities exist but abstention_threshold is 0; "
                "set threshold > 0 or resolve ambiguities"
            )

        return errors


class ContractCompiler:
    """Compiles a ProjectRequest into a validated WorkContract.

    Flow:
      1. Fan out three analyses in parallel (via runner)
      2. Synthesize contract from analysis results
      3. Falsifier checks for misinterpretation
      4. Validate structural invariants
      5. Emit events to ledger
    """

    def __init__(
        self,
        analysis_runner: AnalysisRunner,
        synthesizer: ContractSynthesizer | None = None,
        falsifier: Falsifier | None = None,
        tool_registry: ToolRegistry | list[str] | None = None,
        command_names: list[str] | None = None,
    ) -> None:
        self._runner = analysis_runner
        self._synthesizer = synthesizer or DeterministicSynthesizer()
        self._falsifier = falsifier or DeterministicFalsifier()
        self._validator = ContractValidator(
            tool_registry=tool_registry, command_names=command_names
        )

    async def compile(
        self,
        request: ProjectRequest,
        ledger: Ledger | None = None,
        config: NorthStackConfig | None = None,
        run_id: str = "",
    ) -> WorkContract:
        """Compile a ProjectRequest into a WorkContract.

        Uses asyncio.gather for parallel fan-out of the three analyses.
        If ledger is provided, events are emitted for each pipeline step.
        Raises ValueError if validation fails.
        """
        import asyncio

        budget = request.budget
        if budget is None and config:
            budget = config.run.default_budget()
        if budget is None:
            budget = Budget.default()

        req_profile = repo_profile = acc_profile = self._pick_profile(config, "worker")

        if ledger and run_id:
            stream = EventStream(ledger, run_id)
            for analysis_name, profile in [
                ("requirements", req_profile),
                ("repo_constraints", repo_profile),
                ("acceptance", acc_profile),
            ]:
                await stream.emit_async(
                    AnalysisRequested(
                        profile=profile.name,
                        analysis={"name": analysis_name},
                    ),
                )

        req_analysis, repo_analysis, acc_analysis = await asyncio.gather(
            self._runner.run_requirements(request, req_profile),
            self._runner.run_repo(request, repo_profile),
            self._runner.run_acceptance(request, acc_profile),
        )
        acc_analysis = self._normalize_command_criteria(config, acc_analysis)
        acc_analysis = self._upgrade_test_goal_criterion(request, config, acc_analysis)

        if ledger and run_id:
            stream = EventStream(ledger, run_id)
            for analysis_name, analysis_result in [
                ("requirements", req_analysis),
                ("repo_constraints", repo_analysis),
                ("acceptance", acc_analysis),
            ]:
                detail: dict[str, Any] = {"name": analysis_name}
                scan_digest = getattr(analysis_result, "scan_digest", "")
                if scan_digest:
                    detail["scan_digest"] = scan_digest
                await stream.emit_async(
                    AnalysisCompleted(
                        profile=repo_profile.name,
                        analysis=detail,
                    ),
                )

        contract = self._synthesizer.synthesize(
            request,
            req_analysis,
            repo_analysis,
            acc_analysis,
            budget,
        )

        if ledger and run_id:
            await EventStream(ledger, run_id).emit_async(
                ContractProposed(
                    id=contract.id,
                    version=contract.version,
                    objective=contract.objective,
                    scope=contract.scope,
                    deliverables=contract.deliverables,
                    constraints=contract.constraints,
                    allowed_tools=contract.allowed_tools,
                    workspace_scope=contract.workspace_scope,
                    budget=contract.budget,
                    acceptance_criteria_count=len(contract.acceptance_criteria),
                    acceptance_criteria=contract.acceptance_criteria,
                    unresolved_ambiguity=contract.unresolved_ambiguity,
                ),
            )

        counter = await self._falsifier.check(contract, request)
        if counter:
            raise ValueError(f"Falsifier found counter-interpretation: {counter}")

        errors = self._validator.validate(contract, request)
        if errors:
            raise ValueError(f"Contract validation failed: {'; '.join(errors)}")

        if ledger and run_id:
            await EventStream(ledger, run_id).emit_async(
                ContractValidated(id=contract.id, version=contract.version),
            )

        return contract

    def _pick_profile(
        self,
        config: NorthStackConfig | None,
        role: str,
    ) -> ModelProfile:
        """Pick a profile matching the given role from config."""
        if config is None or not config.profiles:
            return _FALLBACK_PROFILE
        return next((p for p in config.profiles if Role(role) in p.roles), config.profiles[0])

    @staticmethod
    def _normalize_command_criteria(
        config: NorthStackConfig | None,
        acc_analysis: AcceptanceAnalysis,
    ) -> AcceptanceAnalysis:
        """Repair model-proposed command criteria against configured profiles.

        Models name the tool ("pytest"); the hard gate executes a named command
        profile. When the proposed name is unknown but exactly one configured
        profile runs pytest, rewrite it deterministically; otherwise drop the
        criterion -- an unexecutable criterion must not reach verification.
        """
        if config is None or not config.commands:
            return acc_analysis
        known = {c.name for c in config.commands}
        pytest_profiles = [
            c.name for c in config.commands if any("pytest" in part for part in c.argv)
        ]
        changed = False
        out: list[dict[str, Any]] = []
        for c in acc_analysis.criteria:
            params = c.get("parameters", {})
            if (
                c.get("kind") == CriterionKind.COMMAND.value
                and params.get("command_name") not in known
            ):
                if len(pytest_profiles) == 1:
                    out.append(
                        {
                            **c,
                            "parameters": {**params, "command_name": pytest_profiles[0]},
                        }
                    )
                    logger.info(
                        "normalized command criterion %r -> profile %r",
                        params.get("command_name"),
                        pytest_profiles[0],
                    )
                else:
                    logger.warning(
                        "dropped command criterion referencing unknown profile %r",
                        params.get("command_name"),
                    )
                changed = True
                continue
            out.append(c)
        if not changed:
            return acc_analysis
        return acc_analysis.model_copy(update={"criteria": out})

    @staticmethod
    def _upgrade_test_goal_criterion(
        request: ProjectRequest,
        config: NorthStackConfig | None,
        acc_analysis: AcceptanceAnalysis,
    ) -> AcceptanceAnalysis:
        """Deterministic acceptance upgrade for make-the-tests-pass goals.

        When the goal explicitly demands passing tests, the workspace holds a
        test suite, and a pytest command profile is configured, acceptance of
        the suite itself is an observable fact -- no model judgment required.
        Appended on top of the model's criteria:
          - the ``command`` hard gate (suite must run green);
          - a whole-tree ``tree_digest`` pin over ``tests/`` (or per-file
            hashes in a root-level layout), so neutralizing, deleting, or
            ADDING test files all fail verification;
          - ``must_exist=False`` tamper guards for pytest config files that do
            not exist yet (a root conftest.py can otherwise deselect or rewrite
            failures -- reachable through ordinary workspace writes).
        The soft rubric stays: it is the only guard against a fix overfitted
        to the exact asserted inputs, which no hard gate here can see.
        """
        if config is None or not _TEST_GOAL_RE.search(request.goal):
            return acc_analysis
        cmd = _pytest_command_name(config)
        root = Path(request.workspace_root)
        tests_dir = root / "tests"
        has_tests = tests_dir.is_dir() or bool(list(root.glob("test_*.py")))
        if cmd is None or not has_tests:
            return acc_analysis

        criteria = [dict(c) for c in acc_analysis.criteria]
        have = {
            (
                c.get("kind"),
                c.get("parameters", {}).get("command_name")
                or c.get("parameters", {}).get("path", ""),
                c.get("parameters", {}).get("exit_code", 0),
            )
            for c in criteria
        }
        if ("command", cmd, 0) not in have:
            criteria.append(
                {
                    "kind": CriterionKind.COMMAND.value,
                    "description": f"command '{cmd}' exits 0 -- full test suite passes",
                    "parameters": {"command_name": cmd, "exit_code": 0},
                }
            )

        if not any(c.get("kind") == CriterionKind.SOFT_RUBRIC.value for c in criteria):
            criteria.append(
                {
                    "kind": CriterionKind.SOFT_RUBRIC.value,
                    "description": (
                        "genuine fix: the product code addresses the reported "
                        "defect generally, not by hardcoding outputs for the "
                        "exact inputs the tests assert"
                    ),
                }
            )

        def pin(path: Path, *, must_exist: bool) -> None:
            rel = path.relative_to(root).as_posix()
            if ("file_diff", rel, 0) in have:
                return
            criteria.append(
                {
                    "kind": CriterionKind.FILE_DIFF.value,
                    "description": (
                        f"'{rel}' must stay byte-identical"
                        if must_exist
                        else f"pytest tamper guard: '{rel}' must not be created"
                    ),
                    "parameters": {
                        "path": rel,
                        "must_exist": must_exist,
                        **(
                            {"content_hash": hashlib.sha256(path.read_bytes()).hexdigest()}
                            if must_exist
                            else {}
                        ),
                    },
                }
            )

        writes_tests = any(
            str(c.get("parameters", {}).get("path", "")).startswith("tests/") for c in criteria
        )
        if tests_dir.is_dir() and not writes_tests:
            if not any(c.get("kind") == "tree_digest" for c in criteria):
                criteria.append(
                    {
                        "kind": CriterionKind.TREE_DIGEST.value,
                        "description": (
                            "'tests/' tree is byte-identical to compile time "
                            "(no edits, deletions, or added files)"
                        ),
                        "parameters": {
                            "path": "tests",
                            "tree_hash": compute_tree_digest(root, "tests"),
                        },
                    }
                )
        elif tests_dir.is_dir():
            for f in sorted(tests_dir.rglob("test_*.py"))[:50]:
                pin(f, must_exist=True)
            conftest = tests_dir / "conftest.py"
            pin(conftest, must_exist=conftest.exists())
        else:
            for f in sorted(root.glob("test_*.py"))[:50]:
                pin(f, must_exist=True)

        for name in _PYTEST_TAMPER_FILES:
            if ("file_diff", name, 0) in have:
                continue
            guard = root / name
            exists = guard.exists()
            criteria.append(
                {
                    "kind": CriterionKind.FILE_DIFF.value,
                    "description": (
                        f"pytest config '{name}' must stay byte-identical"
                        if exists
                        else f"pytest tamper guard: '{name}' must not be created"
                    ),
                    "parameters": {
                        "path": name,
                        "must_exist": exists,
                        **(
                            {"content_hash": hashlib.sha256(guard.read_bytes()).hexdigest()}
                            if exists
                            else {}
                        ),
                    },
                }
            )

        return acc_analysis.model_copy(update={"criteria": criteria})
