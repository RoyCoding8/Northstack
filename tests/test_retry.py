"""Retry policy: one owner of attempt counting and recovery escalation.

Two invariants this file pins:

1. ``max_retries`` is a *single* ceiling, not a squared one: with
   ``max_retries=3`` and a provider that always fails, the provider is called
   **exactly 4 times** (1 initial + 3 retries). The worker makes one attempt
   per call and does not retry internally; the orchestrator's per-cell loop
   is the only layer that bounds the count.

2. Two ``TRANSIENT`` failures with the *same* strategy signature escalate to
   the next rung of ``RECOVERY_POLICY`` instead of a third ``BACKOFF_RETRY``.
   The signature carries the real strategy id (not ``f"attempt-{n}"``), so a
   repeated strategy is recognised and the deduplicator fires -- a stuck cell
   escalates rather than retrying the same strategy forever.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from northstack.adapters.artifacts import ArtifactStore
from northstack.adapters.providers.gateway import (
    HTTPProviderError,
    ModelGateway,
)
from northstack.adapters.sqlite_ledger import Ledger
from northstack.adapters.workspace.restricted import RestrictedWorkspace
from northstack.application.build import NativeWorkerFactory
from northstack.application.contracting import ContractCompiler, DeterministicAnalysisRunner
from northstack.application.orchestrator import Company
from northstack.application.planning import GraphPlanner
from northstack.application.release_law import ReleaseLaw  # noqa: F401  (seam sanity)
from northstack.application.retry import RECOVERY_POLICY, RetryPolicy
from northstack.application.tools.registry import ToolRegistry
from northstack.config import ModelProfile, NorthStackConfig, Protocol
from northstack.domain import (
    AttemptSignature,
    Budget,
    FailureType,
    GraphCell,
    GraphVersion,
    RecoveryAction,
    RunOutcome,
    WorkContract,
)
from northstack.domain.contract import CommandCriterion

# Helpers


def _profile() -> ModelProfile:
    return ModelProfile(
        name="worker",
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test-model",
        max_concurrency=4,
        requests_per_minute=1000,
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=5.0,
        max_output_tokens=4096,
    )


def _profile_escalate() -> ModelProfile:
    """A second, distinct profile so REROUTE_ESCALATE can reroute to it.

    ``max_retries`` is a *ceiling* the dedup escalation ladder can terminate
    before reaching: a same-signature TRANSIENT failure escalates to
    REROUTE_ESCALATE after the 2nd attempt (Test 2 pins this). With a single
    profile the reroute abstains and the run stops at 2 calls, so the ceiling
    is never exercised. A second profile gives the reroute somewhere to go;
    the new profile is a *fresh* signature (signature includes ``profile_name``),
    so the ladder resets to rung 0 and the loop lives long enough for the cap
    to bound it at exactly 4. Both profiles use the same always-failing adapter.
    """
    return ModelProfile(
        name="worker-escalate",
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost:8080/v1",
        model="test-model-escalate",
        max_concurrency=4,
        requests_per_minute=1000,
        input_price_per_million_usd=1.0,
        output_price_per_million_usd=5.0,
        max_output_tokens=4096,
    )


def _config() -> NorthStackConfig:
    return NorthStackConfig(name="test", profiles=[_profile(), _profile_escalate()])


def _wire_always_failing_503(gateway: ModelGateway, calls: dict[str, int]) -> None:
    """Wire an adapter that always raises a retryable 503, counting calls."""

    async def fn(req, prof, client, api_key):
        calls["n"] += 1
        raise HTTPProviderError(
            "Provider returned HTTP 503", status_code=503, provider="openai", model="test-model"
        )

    mock_adapter = MagicMock()
    mock_adapter.complete = fn
    gateway._adapters[Protocol.OPENAI_CHAT] = mock_adapter
    gateway._client = AsyncMock()


class _StaticPlanner(GraphPlanner):
    def __init__(self, graph: GraphVersion) -> None:
        self._graph = graph

    async def plan(self, contract: WorkContract, run_id: str) -> GraphVersion:
        return self._graph


def _single_cell_graph(contract: WorkContract) -> GraphVersion:
    cell = GraphCell(
        id="cell-1",
        name="cell-1",
        mode="read_only",
        contract=contract,
        acceptance_criterion_indices=[0],
    )
    return GraphVersion(
        version=1,
        cells=[cell],
        edges=[],
        milestones=["cell-1"],
        current_horizon=0,
    )


# Test 1: the cap is squared -- exactly 4 provider calls, not ~16


class TestRetryCapIsNotSquared:
    @pytest.mark.asyncio
    async def test_max_retries_3_always_failing_calls_provider_exactly_four_times(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """``max_retries=3`` caps total provider calls at exactly 4 (not ~16).

        The worker runs one attempt per call and does not retry internally;
        ``max_retries`` is a *ceiling* enforced by the orchestrator's per-cell
        loop, so the count is 1 + max_retries, not max_retries squared.

        The dedup escalation ladder sits *under* the ceiling: a
        same-signature TRANSIENT failure escalates to REROUTE_ESCALATE after the
        2nd attempt (pinned by Test 2). With a single profile that reroute
        abstains and the run stops at 2 calls, so the ceiling is never reached.
        This config therefore carries TWO profiles (``worker`` and
        ``worker-escalate``) so the reroute finds an alternative. The new
        profile is a fresh attempt signature (signature includes
        ``profile_name``), so the ladder resets to rung 0 and the loop lives
        long enough for the cap to bound it. The four calls are then:

          call 1 (profile A, sig A, rung 0)   -> BACKOFF_RETRY
          call 2 (profile A, sig A, rung 1)   -> REROUTE_ESCALATE -> profile B
          call 3 (profile B, sig B, rung 0)  -> BACKOFF_RETRY
          call 4 (attempt 3 == max_retries)  -> TERMINATE          (the cap)

        The ceiling is the sole thing that stops the run -- it is not squared
        across two layers, and it is reached (4, not 16) because the reroute
        to a fresh-profile signature lets the loop survive the escalation.
        """

        # No real waiting during retries -- the orchestrator's backoff seam
        # is a no-op. The worker does not retry internally, so there is no
        # worker-side backoff seam to patch.
        async def _no_sleep(delay: float) -> None:
            await asyncio.sleep(0)

        monkeypatch.setattr("northstack.application.cell_runner._recovery_sleep", _no_sleep)

        calls: dict[str, int] = {"n": 0}
        gw = ModelGateway(_config())
        _wire_always_failing_503(gw, calls)

        contract = WorkContract(
            id="wc-1",
            version=1,
            objective="fail forever",
            budget=Budget(token_limit=100_000, cost_limit_usd=5.0, max_retries=3),
            acceptance_criteria=[
                CommandCriterion(description="check", command_name="check", exit_code=0)
            ],
        )
        graph = _single_cell_graph(contract)

        ledger = Ledger(path=tmp_path / "retry.db")
        store = ArtifactStore(tmp_path / "artifacts")
        company = Company(
            config=_config(),
            ledger=ledger,
            artifact_store=store,
            workspace=RestrictedWorkspace(tmp_path / "ws"),
            gateway=gw,
            worker_factory=NativeWorkerFactory(
                gw, {}, ToolRegistry.with_defaults(command_profiles={})
            ),
            compiler=ContractCompiler(analysis_runner=DeterministicAnalysisRunner()),
            command_profiles={},
        )
        company._planner = _StaticPlanner(graph)  # type: ignore[attr-defined]
        company._test_ledger = ledger  # type: ignore[attr-defined]

        try:
            outcome = await company.run_async(
                _request(tmp_path, max_retries=3),
                run_id="run-retry",
            )
        finally:
            ledger.close()

        # The run must terminate (fail) -- the provider never succeeds.
        assert outcome in {RunOutcome.FAILED, RunOutcome.ABSTAINED}

        # Exactly 4 provider calls: the ceiling (max_retries=3) is the sole
        # bound, reached because the 2nd-call reroute finds a fresh-profile
        # signature. Not ~16 -- the cap is not squared across two layers.
        assert calls["n"] == 4, (
            f"expected exactly 4 provider calls (the max_retries=3 ceiling), "
            f"got {calls['n']}; the retry cap is being squared across two layers "
            "or the reroute dead-ended before the ceiling was reached"
        )


def _request(tmp_path: Path, *, max_retries: int) -> Any:
    from northstack.domain import ProjectRequest

    return ProjectRequest(
        goal="fail forever",
        workspace_root=str(tmp_path / "ws"),
        budget=Budget(token_limit=100_000, cost_limit_usd=5.0, max_retries=max_retries),
        max_waves=1,
    )


# Test 2: two same-signature TRANSIENT failures escalate the rung


class TestSameSignatureEscalatesRung:
    def test_two_same_signature_transient_failures_escalate_not_third_backoff(self) -> None:
        """Two TRANSIENT failures with the same strategy signature must move to
        the next rung of RECOVERY_POLICY (REROUTE_ESCALATE), not a third
        BACKOFF_RETRY.

        Today ``AttemptSignature(strategy_id=f"attempt-{n}")`` makes every
        retry's signature unique, so the deduplicator never fires and a stuck
        cell keeps issuing BACKOFF_RETRY for the same strategy.
        """
        from northstack.application.retry import (
            RetryPolicy,
        )
        from northstack.domain import AttemptSignature

        policy = RetryPolicy()
        sig = AttemptSignature(
            contract_version=1,
            cell_id="cell-1",
            profile_name="worker",
            strategy_id="strategy-A",
            tool_plan="read",
            evidence_digest="d-1",
        )

        # First TRANSIENT failure: BACKOFF_RETRY.
        first = policy.next_action(attempt=0, failure="TRANSIENT", tried_sig=sig)
        assert first.value == "backoff_retry", f"first failure should backoff, got {first}"

        # Second TRANSIENT failure with the SAME signature: dedup fires, so we
        # escalate to the next rung (REROUTE_ESCALATE), not a third BACKOFF_RETRY.
        second = policy.next_action(attempt=1, failure="TRANSIENT", tried_sig=sig)
        assert second.value == "reroute_escalate", (
            f"second same-signature failure must escalate to reroute_escalate, "
            f"got {second} (dedup did not fire -- strategy_id is per-attempt)"
        )


# Test 3: the recovery ladder is the single table, table-tested over every
# (FailureType, attempt) pair.
# The escalation walk is: the Nth time the SAME signature is seen, the policy
# returns rung ``min(N, len(ladder)-1)``. This parametrises every FailureType
# against every attempt index from a fresh signature (rung 0) through and past
# the terminal rung (the clamp), asserting each returned action equals the
# ladder table -- so the ladder is the *only* source of recovery decisions
# and no FailureType silently routes to the wrong terminal action.


def _sig(seed: int) -> AttemptSignature:
    """A distinct signature per parametrised run so dedup state is isolated."""
    return AttemptSignature(
        contract_version=1,
        cell_id=f"cell-{seed}",
        profile_name="worker",
        strategy_id=f"strategy-{seed}",
        tool_plan="read",
        evidence_digest=f"d-{seed}",
    )


@pytest.mark.parametrize(
    "ftype",
    list(RECOVERY_POLICY.keys()),
    ids=[f.name for f in RECOVERY_POLICY],
)
@pytest.mark.parametrize(
    "attempt",
    # Per-FailureType attempt ranges expanded inline below; parametrize over the
    # union and skip out-of-range so each FailureType is probed to its own
    # terminal-rung clamp (lengths differ across ladders).
    list(range(max(len(ladder) for ladder in RECOVERY_POLICY.values()) + 1)),
)
def test_recovery_ladder_matches_table_for_every_failure_and_attempt(
    ftype: FailureType, attempt: int
) -> None:
    """For every (FailureType, attempt) pair, ``next_action`` returns exactly
    the rung the ``RECOVERY_POLICY`` table prescribes: rung
    ``min(attempt, len(ladder)-1)``. The ladder is the single source of
    recovery decisions.
    """
    ladder = RECOVERY_POLICY[ftype]
    if attempt >= len(ladder):
        pytest.skip(f"attempt {attempt} out of range for {ftype.name} (ladder len {len(ladder)})")
    policy = RetryPolicy()
    sig = _sig(hash((ftype, attempt)))
    expected = ladder[min(attempt, len(ladder) - 1)]
    # Replay the same signature `attempt + 1` times so the Nth sighting is rung N.
    last: RecoveryAction | None = None
    for i in range(attempt + 1):
        last = policy.next_action(attempt=i, failure=ftype, tried_sig=sig)
    assert last == expected, (
        f"{ftype.name} attempt={attempt}: expected {expected.value}, got {last.value}"
    )


@pytest.mark.parametrize(
    "ftype",
    list(RECOVERY_POLICY.keys()),
    ids=[f.name for f in RECOVERY_POLICY],
)
def test_recovery_ladder_clamps_at_terminal_rung(ftype: FailureType) -> None:
    """Once a signature has been seen more times than its ladder has rungs, the
    policy keeps returning the terminal rung -- never an index error, never a
    phantom action past the table. (Clamp property.)"""
    ladder = RECOVERY_POLICY[ftype]
    policy = RetryPolicy()
    sig = _sig(hash(("clamp", ftype)))
    # Drive one sighting past the terminal rung.
    for i in range(len(ladder) + 1):
        action = policy.next_action(attempt=i, failure=ftype, tried_sig=sig)
    assert action == ladder[-1], (
        f"{ftype.name} past terminal rung: expected {ladder[-1].value} (clamp), got {action.value}"
    )
