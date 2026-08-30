"""Live benchmark strategies: real endpoints, real workspaces, hidden checks.

Every configuration runs the same raw request against a clean per-run copy of
the task's immutable workspace template (the SWE-bench reproducibility model),
under the same tool policy and resource ceilings, and is scored post-hoc by
executable hidden checks the system under test never sees.

Configurations:

  - ``company`` -- the full control plane (contract analyses, routing, cells,
    hard gates, soft review, bounded recovery) via :func:`build_company`.
  - ``strong_single`` / ``singleton_expert`` / ``cheap_best_of_n`` -- honest
    simpler agent configurations: a bare :class:`NativeWorker` model/tool loop
    pinned to one profile, no contract machinery, no routing, no recovery.
    Their claimed outcome is worker completion under the ceilings; the hidden
    checks then decide verified score, false acceptance, and false rejection.

Retained-outcome law: the hidden checks decide the retained ``outcome``
(evaluation truth); the system's own claim is audited against it as
``false_acceptance`` / ``false_rejection``. Best-of-N selection uses the
hidden-check verified score -- an oracle-based selector that makes the
baseline *stronger* than any deployable verifier could, so a company win
against it is conservative. All candidates' resources are accounted.

Baseline profile selection is deterministic and inspectable: the operator's
routing chains win when present (worker -> cheap, specialist -> expert,
orchestrator -> strong); otherwise tier scoring with price tie-breaks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.gateway import ModelGateway
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.application.benchmark import (
    BenchmarkMetrics,
    BenchmarkStrategy,
    BenchmarkTask,
    Configuration,
    HiddenCheck,
    RunResult,
)
from northstack.application.build import build_company
from northstack.application.contracting import AnalysisRunner, DeterministicAnalysisRunner
from northstack.application.tools.registry import ToolRegistry
from northstack.application.worker import NativeWorker
from northstack.config import ModelProfile, NorthStackConfig, Role
from northstack.domain.budget import Budget, BudgetUsage
from northstack.domain.contract import SoftRubricCriterion, WorkContract
from northstack.domain.graph import CellMode, CellStatus, GraphCell
from northstack.domain.outcome import RunOutcome
from northstack.domain.request import ProjectRequest
from northstack.events.catalog import BudgetUpdated, EvidenceRecorded, RecoveryTransition
from northstack.events.envelope import EventEnvelope

logger = logging.getLogger(__name__)

_SNAPSHOT_SKIP = frozenset(
    {
        ".git",
        ".northstack",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)
_SNAPSHOT_MAX_ENTRIES = 20_000
_SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024
_SNAPSHOT_MAX_DEPTH = 64

_BASELINE_MAX_TOOL_ROUNDS = 24
_BASELINE_MAX_WALL_SECONDS = 900.0


def _company_budget(task: BenchmarkTask, max_retries_override: int | None = None) -> Budget:
    budget = Budget(
        token_limit=task.token_limit,
        cost_limit_usd=task.cost_limit_usd,
        max_wall_time_seconds=task.wall_time_limit_seconds,
    )
    if max_retries_override is not None:
        budget = budget.model_copy(update={"max_retries": max_retries_override})
    return budget


class CheckOutcome(BaseModel):
    """Result of executing the hidden checks against a finished workspace."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    failures: list[str] = Field(default_factory=list)


class ScoringGate:
    """Executes a task's hidden checks post-hoc through the workspace chokepoint.

    Command checks run exact-argv subprocesses (``shell=False``, workspace
    cwd, PATH-only env) and compare exit codes; file checks assert presence
    and optional content. No check result ever feeds back into the system
    under test.
    """

    def __init__(self, checks: list[HiddenCheck]) -> None:
        self._checks = checks

    async def score(self, workspace_root: Path) -> CheckOutcome:
        if not self._checks:
            return CheckOutcome(score=0.0, failures=["no hidden checks defined"])
        workspace = RestrictedWorkspace(workspace_root)
        failures: list[str] = []
        for check in self._checks:
            passed = await asyncio.to_thread(self._run_check, workspace, check)
            if not passed:
                failures.append(check.name)
        score = (len(self._checks) - len(failures)) / len(self._checks)
        return CheckOutcome(score=score, failures=failures)

    @staticmethod
    def _run_check(workspace: RestrictedWorkspace, check: HiddenCheck) -> bool:
        if check.argv:
            profile = CommandProfile(
                name=check.name,
                argv=check.argv,
                timeout_seconds=check.timeout_seconds,
                max_output_bytes=check.max_output_bytes,
                env_allowlist=["PATH"],
            )
            result = workspace.execute_command(profile)
            return result.exit_code is not None and result.exit_code == check.expect_exit_code
        if check.path:
            result = workspace.read(check.path)
            if not result.ok or result.truncated:
                return False
            if check.content_contains:
                return check.content_contains.encode() in result.data
            return True
        logger.warning("hidden check '%s' has neither argv nor path; failing", check.name)
        return False


def snapshot_workspace(template: Path, destination: Path) -> Path:
    """Copy the immutable task template into a fresh per-run workspace."""
    if not template.is_dir():
        raise FileNotFoundError(f"task workspace template not found: {template}")
    if destination.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    staged = scratch / "payload"
    try:
        expected = tree_digest(template)
        shutil.copytree(
            template,
            staged,
            ignore=_ignore_snapshot_entries,
            symlinks=True,
        )
        if tree_digest(template) != expected or tree_digest(staged) != expected:
            raise OSError("workspace template changed during snapshot")
        os.replace(staged, destination)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return destination


def _ignore_snapshot_entries(directory: str, entries: list[str]) -> list[str]:
    return [e for e in entries if e in _SNAPSHOT_SKIP or e.endswith(".pyc")]


def tree_digest(root: Path) -> str:
    """Deterministic content digest over a template tree (sorted paths)."""
    root = root.resolve(strict=True)
    hasher = hashlib.sha256()
    for path in _template_files(root):
        rel = path.relative_to(root).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        hasher.update(digest.digest())
    return f"sha256:{hasher.hexdigest()}"


def _template_files(root: Path) -> list[Path]:
    root = root.resolve(strict=True)
    files, stack, entries, total = [], [(root, 0)], 0, 0
    while stack:
        directory, depth = stack.pop()
        with os.scandir(directory) as scanned:
            children = sorted(scanned, key=lambda entry: entry.name)
        for entry in children:
            if entry.name in _SNAPSHOT_SKIP or entry.name.endswith(".pyc"):
                continue
            entries += 1
            if entries > _SNAPSHOT_MAX_ENTRIES:
                raise ValueError(f"snapshot entry limit exceeded: {_SNAPSHOT_MAX_ENTRIES}")
            if entry.is_symlink():
                raise ValueError(f"snapshot template contains link: {entry.path}")
            path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if depth >= _SNAPSHOT_MAX_DEPTH:
                    raise ValueError(f"snapshot depth limit exceeded: {_SNAPSHOT_MAX_DEPTH}")
                stack.append((path, depth + 1))
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
                if total > _SNAPSHOT_MAX_BYTES:
                    raise ValueError(f"snapshot byte limit exceeded: {_SNAPSHOT_MAX_BYTES}")
                files.append(path)
            else:
                raise ValueError(f"snapshot template contains unsupported entry: {entry.path}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _write_meta(
    run_dir: Path,
    task: BenchmarkTask,
    configuration: Configuration,
    index: int,
    template_digest: str,
    snapshot: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "task_id": task.id,
                "configuration": configuration.value,
                "index": index,
                "template_digest": template_digest,
                "snapshot": str(snapshot),
                "recorded_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _assemble_run_result(
    *,
    task: BenchmarkTask,
    configuration: Configuration,
    index: int,
    claimed_verified: bool,
    metrics: BenchmarkMetrics,
    check_outcome: CheckOutcome,
    error: str = "",
    ablation: str = "",
) -> RunResult:
    """Apply the retained-outcome law: hidden checks decide, claims audit."""
    if check_outcome.score >= 1.0:
        outcome = RunOutcome.VERIFIED
    elif check_outcome.score <= 0.0:
        outcome = RunOutcome.FAILED
    else:
        outcome = RunOutcome.ABSTAINED
    return RunResult(
        task_id=task.id,
        configuration=configuration,
        repeat_index=index,
        outcome=outcome,
        verified_score=check_outcome.score,
        false_acceptance=claimed_verified and check_outcome.score < 1.0,
        false_rejection=(not claimed_verified) and check_outcome.score >= 1.0,
        error=error,
        metrics=metrics,
        ablation=ablation,
    )


def select_baseline_profiles(config: NorthStackConfig) -> dict[str, str]:
    """Pick the three baseline profiles deterministically.

    Routing chains are authoritative when present (worker/specialist/
    orchestrator heads); otherwise tier+price scoring. Raises when the config
    has no profiles at all.
    """
    if not config.profiles:
        raise ValueError("live benchmark requires at least one configured profile")
    role_map = config.role_map()
    known = {p.name for p in config.profiles}

    def _chain_first(role: Role) -> str | None:
        for name in role_map.get(role) or []:
            if name in known:
                return name
        return None

    def _by_tier(highest: bool) -> ModelProfile:
        return sorted(
            config.profiles,
            key=lambda p: (p.tier, p.output_price_per_million_usd, p.name),
            reverse=highest,
        )[0]

    strong = _chain_first(Role.ORCHESTRATOR) or _by_tier(highest=True).name
    expert = _chain_first(Role.SPECIALIST) or strong
    cheap = _chain_first(Role.WORKER) or _by_tier(highest=False).name
    return {"strong_single": strong, "singleton_expert": expert, "cheap": cheap}


class LiveCompanyStrategy:
    """The full control-plane configuration, end to end, on a fresh snapshot.

    Ablation parameters build company-minus-one-mechanism variants for the
    protocol's component ablations: ``label`` names the variant (run dirs and
    retained results carry it), ``max_retries_override`` pins the recovery
    cap, and ``analysis_runner`` swaps intake for the deterministic runner.
    Mechanism removals that live in config (routing, planner mode,
    falsifier mode) arrive as an already-modified ``config`` copy from
    :func:`ablation_strategies`.
    """

    def __init__(
        self,
        config: NorthStackConfig,
        *,
        runs_dir: Path,
        suite_dir: Path,
        label: str = "",
        max_retries_override: int | None = None,
        analysis_runner: AnalysisRunner | None = None,
    ) -> None:
        self._config = config
        self._runs_dir = runs_dir
        self._suite_dir = suite_dir
        self._label = label
        self._max_retries_override = max_retries_override
        self._analysis_runner = analysis_runner

    async def run(self, task: BenchmarkTask, repeat_index: int) -> RunResult:
        tag = self._label or Configuration.COMPANY.value
        run_dir = self._runs_dir / f"{task.id}-{repeat_index}-{tag}"
        template = self._template_for(task)
        template_digest = await asyncio.to_thread(tree_digest, template)
        snapshot = await asyncio.to_thread(snapshot_workspace, template, run_dir / "workspace")
        _write_meta(run_dir, task, Configuration.COMPANY, repeat_index, template_digest, snapshot)

        run_id = f"bench-{task.id}-{repeat_index}-{tag}"
        budget = _company_budget(task, self._max_retries_override)
        request = ProjectRequest(
            goal=task.request,
            workspace_root=str(snapshot),
            budget=budget,
        )
        components = build_company(
            self._config,
            snapshot,
            db_path=run_dir / "ledger.db",
            analysis_runner=self._analysis_runner,
        )
        try:
            outcome = await components.company.run_async(request, run_id=run_id)
            claimed_verified = outcome == RunOutcome.VERIFIED
            metrics, reason = _company_metrics(components.ledger.events(run_id))
        finally:
            components.close()

        check_outcome = await ScoringGate(task.checks).score(snapshot)
        return _assemble_run_result(
            task=task,
            configuration=Configuration.COMPANY,
            index=repeat_index,
            claimed_verified=claimed_verified,
            metrics=metrics,
            check_outcome=check_outcome,
            error=reason,
            ablation=self._label,
        )

    def _template_for(self, task: BenchmarkTask) -> Path:
        candidate = Path(task.workspace)
        if not candidate.is_absolute():
            candidate = self._suite_dir / candidate
        return candidate.resolve()


class LiveWorkerStrategy:
    """A simpler agent configuration: one bare worker loop, one pinned profile."""

    def __init__(
        self,
        config: NorthStackConfig,
        *,
        configuration: Configuration,
        profile_name: str,
        runs_dir: Path,
        suite_dir: Path,
    ) -> None:
        self._config = config
        self._configuration = configuration
        self._profile_name = profile_name
        self._runs_dir = runs_dir
        self._suite_dir = suite_dir

    async def run(self, task: BenchmarkTask, repeat_index: int) -> RunResult:
        run_dir = self._runs_dir / f"{task.id}-{repeat_index}-{self._configuration.value}"
        template = self._template_for(task)
        template_digest = await asyncio.to_thread(tree_digest, template)
        snapshot = await asyncio.to_thread(snapshot_workspace, template, run_dir / "workspace")
        _write_meta(run_dir, task, self._configuration, repeat_index, template_digest, snapshot)

        command_profiles = {
            command.name: CommandProfile.from_config(command) for command in self._config.commands
        }
        tool_registry = ToolRegistry.with_defaults(command_profiles=command_profiles)
        gateway = ModelGateway(self._config, artifact_store=ArtifactStore(run_dir / "artifacts"))
        worker = NativeWorker(
            gateway,
            RestrictedWorkspace(snapshot),
            command_profiles=command_profiles,
            tool_registry=tool_registry,
        )
        cell = GraphCell(
            id=f"{task.id}-{repeat_index}",
            name=task.id,
            mode=CellMode.MUTATING,
            status=CellStatus.PENDING,
            contract=WorkContract(
                id=f"wc-{task.id}-{repeat_index}",
                objective=task.request,
                deliverables=["workspace deliverables"],
                workspace_scope=str(snapshot),
                budget=Budget(
                    token_limit=task.token_limit,
                    cost_limit_usd=task.cost_limit_usd,
                    max_tool_rounds=_BASELINE_MAX_TOOL_ROUNDS,
                    max_wall_time_seconds=_BASELINE_MAX_WALL_SECONDS,
                ),
                acceptance_criteria=[SoftRubricCriterion(description="baseline worker completion")],
            ),
        )
        error = ""
        claimed_verified = False
        metrics = BenchmarkMetrics()
        try:
            try:
                result = await worker.run(cell, self._profile_name, tool_registry.advertised())
            except Exception as exc:  # noqa: BLE001
                error = f"worker crashed: {exc}"[:300]
                result = None
            if result is not None:
                claimed_verified = bool(result.ok)
                metrics = BenchmarkMetrics(
                    input_tokens=result.total_input_tokens,
                    output_tokens=result.total_output_tokens,
                    calls=max(result.api_calls, 1),
                    wall_time_ms=result.wall_time_ms,
                    retries=0,
                    tool_operations=result.tool_calls_made,
                    configured_cost_usd=result.total_cost_usd,
                )
                if claimed_verified and result.total_cost_usd > task.cost_limit_usd:
                    claimed_verified = False
                    error = f"cost ceiling breached: ${result.total_cost_usd:.4f}"
                if not result.ok:
                    error = f"{result.error_kind}: {result.error[:200]}"
        finally:
            await gateway.close()

        check_outcome = await ScoringGate(task.checks).score(snapshot)
        return _assemble_run_result(
            task=task,
            configuration=self._configuration,
            index=repeat_index,
            claimed_verified=claimed_verified,
            metrics=metrics,
            check_outcome=check_outcome,
            error=error,
        )

    def _template_for(self, task: BenchmarkTask) -> Path:
        candidate = Path(task.workspace)
        if not candidate.is_absolute():
            candidate = self._suite_dir / candidate
        return candidate.resolve()


def _company_metrics(events: list[EventEnvelope]) -> tuple[BenchmarkMetrics, str]:
    """Derive provider-neutral metrics from the run's own ledger events."""
    usage = BudgetUsage()
    retries = 0
    tool_operations = 0
    for envelope in events:
        payload = envelope.payload
        if isinstance(payload, BudgetUpdated):
            usage = payload.usage
        elif isinstance(payload, EvidenceRecorded):
            usage = payload.usage
            tool_operations = len(payload.tools_used)
        elif isinstance(payload, RecoveryTransition):
            retries += 1
    wall_ms = int((events[-1].timestamp - events[0].timestamp) * 1000) if len(events) >= 2 else 0
    metrics = BenchmarkMetrics(
        input_tokens=usage.total_input_tokens,
        output_tokens=usage.total_output_tokens,
        calls=usage.total_calls,
        wall_time_ms=wall_ms,
        retries=retries,
        tool_operations=tool_operations,
        configured_cost_usd=usage.total_cost_usd,
    )
    return metrics, f"ledger events: {len(events)}"


def live_strategies(
    config: NorthStackConfig,
    *,
    runs_dir: Path,
    suite_dir: Path,
) -> dict[Configuration, BenchmarkStrategy]:
    """Build the four preregistered live configurations."""
    picks = select_baseline_profiles(config)
    logger.info(
        "live baseline profiles: strong=%s expert=%s cheap=%s",
        picks["strong_single"],
        picks["singleton_expert"],
        picks["cheap"],
    )
    return {
        Configuration.STRONG_SINGLE: LiveWorkerStrategy(
            config,
            configuration=Configuration.STRONG_SINGLE,
            profile_name=picks["strong_single"],
            runs_dir=runs_dir,
            suite_dir=suite_dir,
        ),
        Configuration.CHEAP_BEST_OF_N: LiveWorkerStrategy(
            config,
            configuration=Configuration.CHEAP_BEST_OF_N,
            profile_name=picks["cheap"],
            runs_dir=runs_dir,
            suite_dir=suite_dir,
        ),
        Configuration.SINGLETON_EXPERT: LiveWorkerStrategy(
            config,
            configuration=Configuration.SINGLETON_EXPERT,
            profile_name=picks["singleton_expert"],
            runs_dir=runs_dir,
            suite_dir=suite_dir,
        ),
        Configuration.COMPANY: LiveCompanyStrategy(config, runs_dir=runs_dir, suite_dir=suite_dir),
    }


def ablation_strategies(
    config: NorthStackConfig,
    *,
    runs_dir: Path,
    suite_dir: Path,
) -> dict[str, BenchmarkStrategy]:
    """Build the protocol's component ablations: company minus one mechanism.

    Each ablation modifies exactly one thing and retains everything else:

      - ``no_routing``: the routing table is stripped; the Router falls back
        to tier/price scoring (no role chains).
      - ``single_cell``: ``planner_mode`` forced to the deterministic
        single-cell planner (no model-proposed decomposition).
      - ``deterministic_intake``: contract analyses use the deterministic
        fixture runner (no model-backed requirements/acceptance analysis;
        the repo scan still runs inside it).
      - ``minimal_recovery``: the per-cell recovery cap pinned to 1 (the
        smallest legal cap -- ``0`` means *unlimited*, so single-attempt is
        not expressible; this is the closest removal and is labelled
        honestly).
      - ``no_falsifier``: ``falsifier_mode`` forced off (diagnostic when the
        base configuration runs model-backed falsification).
    """
    return {
        "no_routing": LiveCompanyStrategy(
            config.model_copy(update={"routing": []}),
            runs_dir=runs_dir / "no_routing",
            suite_dir=suite_dir,
            label="no_routing",
        ),
        "single_cell": LiveCompanyStrategy(
            config.model_copy(
                update={"run": config.run.model_copy(update={"planner_mode": "single"})}
            ),
            runs_dir=runs_dir / "single_cell",
            suite_dir=suite_dir,
            label="single_cell",
        ),
        "deterministic_intake": LiveCompanyStrategy(
            config,
            runs_dir=runs_dir / "deterministic_intake",
            suite_dir=suite_dir,
            label="deterministic_intake",
            analysis_runner=DeterministicAnalysisRunner(),
        ),
        "minimal_recovery": LiveCompanyStrategy(
            config,
            runs_dir=runs_dir / "minimal_recovery",
            suite_dir=suite_dir,
            label="minimal_recovery",
            max_retries_override=1,
        ),
        "no_falsifier": LiveCompanyStrategy(
            config.model_copy(
                update={"run": config.run.model_copy(update={"falsifier_mode": "off"})}
            ),
            runs_dir=runs_dir / "no_falsifier",
            suite_dir=suite_dir,
            label="no_falsifier",
        ),
    }
