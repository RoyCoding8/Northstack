"""Shared composition root for building a runnable ``Company``.

Extracted from ``cli.py run_project`` so both the CLI and the web control
surface compose a ``Company`` the same way.  The caller owns the lifecycle:

  - CLI calls ``Company.run(request)`` (sync wrapper, creates/destroys its own
    event loop) and closes the ledger in a finally.
  - The web server wraps ``company.run_async(request, run_id=...)`` in an
    ``asyncio.Task`` on the server's event loop and lets ``run_async`` close
    the (loop-bound) gateway in its own finally.

Lifecycle note (do NOT regress): the ``ModelGateway``'s httpx ``AsyncClient``
is bound to whatever event loop first uses it.  ``run_async`` closes the
gateway on the *same* loop it ran on.  Never share a gateway across runs and
never close it from a different loop -- each ``build_company`` call creates a
fresh gateway owned by the single run it serves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.gateway import ModelGateway
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.sqlite_memory import SqliteMemory
from northstack.adapters.workspace.restricted import CommandProfile, RestrictedWorkspace
from northstack.application.calibration import load_calibration_records
from northstack.application.contracting import (
    AnalysisRunner,
    ContractCompiler,
    DeterministicAnalysisRunner,
    Falsifier,
    ModelBackedAnalysisRunner,
)
from northstack.application.falsification import ModelBackedFalsifier
from northstack.application.orchestrator import Company
from northstack.application.planning import GraphPlanner
from northstack.application.planning_model import ModelBackedPlanner
from northstack.application.tools.registry import ToolRegistry
from northstack.application.verification.model_review import ModelBackedReviewer
from northstack.application.verification.soft_rubric import (
    MIN_BLINDED_REVIEWERS,
    BlindedReviewer,
)
from northstack.application.worker import NativeWorker
from northstack.config import Capability, NorthStackConfig, Role
from northstack.domain.outcome import CalibrationRecord

logger = logging.getLogger(__name__)


@dataclass
class CompanyComponents:
    """Everything the caller must hold/clean up returned by build_company.

    ``company`` is ready to run; ``ledger`` and ``memory`` must be closed by
    the caller when done (``close()`` handles both).  The gateway is closed by
    ``run_async`` itself (same loop), so the caller must NOT close it (see
    module docstring).
    """

    company: Company
    ledger: Ledger
    artifact_store: ArtifactStore
    gateway: ModelGateway
    workspace: RestrictedWorkspace
    memory: SqliteMemory | None = None

    def close(self) -> None:
        """Release every handle this build opened, in reverse order."""
        if self.memory is not None:
            self.memory.close()
        self.ledger.close()


class NativeWorkerFactory:
    """Builds a ``NativeWorker`` bound to a fresh per-run gateway.

    The gateway is captured at build_company time (one fresh gateway per run).
    The workspace passed to ``create`` is forwarded as-is to ``NativeWorker``
    rather than hard-coding an outer capture, so the caller's chosen workspace
    wins (mirrors how the orchestrator invokes workers on the run workspace).
    The single :class:`ToolRegistry` is threaded through so the worker
    dispatches every tool through the one declaration instead of an
    ``if/elif`` name chain.
    """

    def __init__(
        self,
        gateway: ModelGateway,
        command_profiles: dict[str, CommandProfile],
        tool_registry: ToolRegistry,
    ) -> None:
        self._gateway = gateway
        self._command_profiles = command_profiles
        self._tool_registry = tool_registry

    def create(self, workspace: RestrictedWorkspace) -> NativeWorker:
        return NativeWorker(
            self._gateway,
            workspace,
            command_profiles=self._command_profiles,
            tool_registry=self._tool_registry,
        )


def build_company(
    config: NorthStackConfig,
    workspace: Path,
    *,
    db_path: Path | None = None,
    analysis_runner: AnalysisRunner | None = None,
) -> CompanyComponents:
    """Compose a runnable ``Company`` plus its lifecycle handles.

    Does NOT create a ``ProjectRequest`` or start a run; the caller builds the
    request and drives ``run`` / ``run_async`` itself. ``analysis_runner``
    overrides the default runner selection (used by the benchmark's
    deterministic-intake ablation); None keeps the configured behavior.
    """
    resolved_db_path = db_path or Path(workspace) / ".northstack" / "ledger.db"
    artifact_path = resolved_db_path.parent / "artifacts"
    artifact_store = ArtifactStore(artifact_path)
    restricted_workspace = RestrictedWorkspace(workspace)
    gateway = ModelGateway(config, artifact_store=artifact_store)
    command_profiles = {
        command.name: CommandProfile.from_config(command) for command in config.commands
    }

    tool_registry = ToolRegistry.with_defaults(command_profiles=command_profiles)

    profiles_by_name = {p.name: p for p in config.profiles}

    if analysis_runner is None:
        workers = [p for p in config.profiles if Role.WORKER in p.roles]
        analysis_profile_name = next(
            (p.name for p in workers if Capability.NATIVE_JSON_SCHEMA in p.capabilities),
            workers[0].name if workers else "",
        )
        if analysis_profile_name:
            analysis_runner = ModelBackedAnalysisRunner(
                gateway,
                analysis_profile_name,
                max_output_tokens=profiles_by_name[analysis_profile_name].max_output_tokens,
                command_names=[c.name for c in config.commands],
            )
        else:
            analysis_runner = DeterministicAnalysisRunner()

    reviewer_chain = config.role_map().get(Role.REVIEWER, [])
    reviewers: list[BlindedReviewer] = []
    for profile_name in reviewer_chain[:MIN_BLINDED_REVIEWERS]:
        reviewers.append(
            ModelBackedReviewer(
                gateway,
                profile_name,
                max_output_tokens=profiles_by_name[profile_name].max_output_tokens,
            )
        )

    calibration_records: list[CalibrationRecord] | None = None
    if config.run.calibration_path:
        calibration_records = load_calibration_records(Path(config.run.calibration_path))

    planner: GraphPlanner | None = None
    if config.run.planner_mode == "model":
        planner_chain = config.role_map().get(Role.PLANNER, [])
        planner_profile = next((n for n in planner_chain if n in profiles_by_name), None)
        if planner_profile is not None:
            planner = ModelBackedPlanner(
                gateway,
                planner_profile,
                config.role_map(),
                max_output_tokens=profiles_by_name[planner_profile].max_output_tokens,
            )
        else:
            logger.warning(
                "planner_mode='model' but no PLANNER-role profile is routed; "
                "using the default single-cell planner"
            )

    falsifier: Falsifier | None = None
    if config.run.falsifier_mode == "model":
        specialist_chain = config.role_map().get(Role.SPECIALIST, [])
        falsifier_profile = next((n for n in specialist_chain if n in profiles_by_name), None)
        if falsifier_profile is not None:
            falsifier = ModelBackedFalsifier(
                gateway,
                falsifier_profile,
                max_output_tokens=profiles_by_name[falsifier_profile].max_output_tokens,
            )
        else:
            logger.warning(
                "falsifier_mode='model' but no SPECIALIST-role profile is routed; "
                "falsification stays off"
            )

    ledger = Ledger(path=resolved_db_path)
    memory = (
        SqliteMemory(resolved_db_path.parent / "memory.db") if config.run.memory_enabled else None
    )
    try:
        company = Company(
            config=config,
            ledger=ledger,
            artifact_store=artifact_store,
            workspace=restricted_workspace,
            gateway=gateway,
            worker_factory=NativeWorkerFactory(gateway, command_profiles, tool_registry),
            compiler=ContractCompiler(
                analysis_runner=analysis_runner,
                falsifier=falsifier,
                tool_registry=tool_registry,
                command_names=[c.name for c in config.commands],
            ),
            command_profiles=command_profiles,
            reviewers=reviewers,
            calibration_records=calibration_records,
            tool_registry=tool_registry,
            planner=planner,
            memory=memory,
        )
    except BaseException:
        ledger.close()
        if memory is not None:
            memory.close()
        raise

    return CompanyComponents(
        company=company,
        ledger=ledger,
        artifact_store=artifact_store,
        gateway=gateway,
        workspace=restricted_workspace,
        memory=memory,
    )
