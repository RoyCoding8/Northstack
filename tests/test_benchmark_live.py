"""Hermetic tests for the live benchmark machinery.

No network: the model gateway is replaced with a factory that wires a fake
adapter (the same seam test_worker.py uses). These tests pin:

  - the scoring gate's pass/fail math on real subprocesses and files;
  - snapshot isolation (template untouched, deterministic digest);
  - deterministic baseline profile selection (role chains, then tiers);
  - the retained-outcome law: hidden checks decide, claims audit
    (false acceptance / false rejection);
  - the bare-worker baseline and the full company strategy, end to end,
    from template snapshot through scored RunResult.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import northstack.application.benchmark_live as benchmark_live
from northstack.adapters.providers.gateway import ModelGateway
from northstack.adapters.providers.wire import (
    FinishReason,
    ModelResponse,
    ToolCall,
    Usage,
)
from northstack.application.benchmark import BenchmarkTask, Configuration, HiddenCheck
from northstack.application.benchmark_live import (
    LiveWorkerStrategy,
    ScoringGate,
    live_strategies,
    select_baseline_profiles,
    snapshot_workspace,
    tree_digest,
)
from northstack.config import ModelProfile, NorthStackConfig, Protocol, Role
from northstack.domain import RunOutcome


# Fixtures: a tiny task template and a fake-gateway factory


@pytest.fixture
def template(tmp_path: Path) -> Path:
    root = tmp_path / "template"
    root.mkdir()
    (root / "README.md").write_text("# task\n")
    (root / "app.py").write_text("def greet():\n    return 'hi'\n")
    git = root / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    return root


def _task(template: Path, **kw) -> BenchmarkTask:
    defaults: dict = {
        "id": "t1",
        "category": "bug_fix",
        "request": "do the thing",
        "workspace": str(template),
        "token_limit": 10_000,
        "cost_limit_usd": 1.0,
    }
    defaults.update(kw)
    return BenchmarkTask(**defaults)


def _ok(text: str = "Done!") -> ModelResponse:
    return ModelResponse(
        text=text,
        finish_reason=FinishReason.END_TURN,
        usage=Usage(input_tokens=10, output_tokens=5),
        provider="openai",
        model="test-model",
    )


def _tool(name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(
        text="",
        tool_calls=[ToolCall(id="call-1", name=name, arguments=arguments)],
        finish_reason=FinishReason.TOOL_USE,
        usage=Usage(input_tokens=10, output_tokens=5),
        provider="openai",
        model="test-model",
    )


class GatewayFactory:
    """Builds real gateways whose provider adapter plays scripted responses."""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.built: list[ModelGateway] = []

    def __call__(self, config: NorthStackConfig, artifact_store=None) -> ModelGateway:
        gateway = ModelGateway(config, artifact_store=artifact_store)
        # Each built gateway plays its own copy of the script: one factory may
        # build several gateways (one per benchmark configuration).
        responses = list(self._responses)

        async def fake_complete(request, profile, client, api_key):
            if responses:
                return responses.pop(0)
            return _ok()

        adapter = MagicMock()
        adapter.complete = fake_complete
        gateway._adapters[Protocol.OPENAI_CHAT] = adapter
        self.built.append(gateway)
        return gateway


def _config() -> NorthStackConfig:
    return NorthStackConfig(
        name="bench-test",
        profiles=[
            ModelProfile(
                name="cheap",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost:8080/v1",
                model="m",
                roles=[Role.WORKER],
                capabilities=["tool_use"],
                max_concurrency=4,
            ),
        ],
    )


# Scoring gate


async def test_scoring_gate_command_checks_pass_and_fail(tmp_path: Path):
    (tmp_path / "f.txt").write_text("x")
    gate = ScoringGate(
        [
            HiddenCheck(name="ok", argv=["python", "-c", "print('fine')"]),
            HiddenCheck(name="boom", argv=["python", "-c", "raise SystemExit(3)"]),
        ]
    )
    outcome = await gate.score(tmp_path)
    assert outcome.score == 0.5
    assert outcome.failures == ["boom"]


async def test_scoring_gate_nonzero_expected_exit_code(tmp_path: Path):
    gate = ScoringGate(
        [
            HiddenCheck(
                name="expects-3", argv=["python", "-c", "raise SystemExit(3)"], expect_exit_code=3
            )
        ]
    )
    assert (await gate.score(tmp_path)).score == 1.0


async def test_scoring_gate_file_checks(tmp_path: Path):
    (tmp_path / "out.txt").write_text("hello world")
    gate = ScoringGate(
        [
            HiddenCheck(name="present", path="out.txt"),
            HiddenCheck(name="contains", path="out.txt", content_contains="world"),
            HiddenCheck(name="wrong-content", path="out.txt", content_contains="nope"),
            HiddenCheck(name="missing", path="absent.txt"),
        ]
    )
    outcome = await gate.score(tmp_path)
    assert outcome.score == 0.5
    assert outcome.failures == ["wrong-content", "missing"]


async def test_scoring_gate_accepts_empty_file_existence_check(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_bytes(b"")
    outcome = await ScoringGate([HiddenCheck(name="empty", path="empty.txt")]).score(tmp_path)
    assert outcome.score == 1.0


async def test_scoring_gate_rejects_truncated_prefix_match(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_bytes(b"required" + b"x" * 1_048_576)
    outcome = await ScoringGate(
        [HiddenCheck(name="large", path="large.txt", content_contains="required")]
    ).score(tmp_path)
    assert outcome.score == 0.0


async def test_scoring_gate_without_checks_is_loud_zero(tmp_path: Path):
    outcome = await ScoringGate([]).score(tmp_path)
    assert outcome.score == 0.0
    assert outcome.failures == ["no hidden checks defined"]


# Snapshots and digests


def test_snapshot_copies_template_and_excludes_derived_state(template: Path, tmp_path: Path):
    dest = tmp_path / "run" / "workspace"
    snapshot_workspace(template, dest)
    assert (dest / "app.py").read_text().startswith("def greet")
    assert not (dest / ".git").exists()
    # The immutable template is untouched by the copy.
    assert (template / ".git" / "HEAD").exists()


def test_snapshot_and_digest_reject_links(template: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        os.symlink(outside, template / "linked.txt")
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    with pytest.raises(ValueError, match="link"):
        tree_digest(template)
    with pytest.raises(ValueError, match="link"):
        snapshot_workspace(template, tmp_path / "snapshot")


def test_failed_snapshot_removes_staging_and_destination(
    template: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "snapshot"

    def fail_copy(source: Path, staged: Path, **kwargs: object) -> None:
        staged.mkdir(parents=True)
        (staged / "partial.txt").write_text("partial", encoding="utf-8")
        raise OSError("injected copy failure")

    monkeypatch.setattr(benchmark_live.shutil, "copytree", fail_copy)
    with pytest.raises(OSError, match="injected copy failure"):
        snapshot_workspace(template, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".snapshot-*"))


def test_tree_digest_is_stable_and_content_sensitive(template: Path):
    first = tree_digest(template)
    assert first == tree_digest(template)
    assert first.startswith("sha256:")
    (template / "app.py").write_text("def greet():\n    return 'changed'\n")
    assert tree_digest(template) != first


def test_tree_digest_ignores_derived_state(template: Path):
    before = tree_digest(template)
    cache = template / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-312.pyc").write_bytes(b"\x00\x01")
    assert tree_digest(template) == before


# Baseline profile selection


def _profile(name: str, tier_hint_price: float, roles: list[Role]) -> ModelProfile:
    return ModelProfile(
        name=name,
        protocol=Protocol.OPENAI_CHAT,
        base_url="http://localhost/v1",
        model=name,
        roles=roles,
        input_price_per_million_usd=tier_hint_price,
        output_price_per_million_usd=tier_hint_price * 3,
        max_concurrency=2,
    )


def test_select_baseline_profiles_prefers_routing_chains():
    config = NorthStackConfig(
        name="sel",
        profiles=[
            _profile("strong", 15.0, [Role.ORCHESTRATOR]),
            _profile("expert", 10.0, [Role.SPECIALIST]),
            _profile("cheap", 0.3, [Role.WORKER]),
        ],
        routing=[
            {"role": "worker", "profiles": ["cheap"]},
            {"role": "specialist", "profiles": ["expert"]},
            {"role": "orchestrator", "profiles": ["strong"]},
        ],
    )
    picks = select_baseline_profiles(config)
    assert picks == {
        "strong_single": "strong",
        "singleton_expert": "expert",
        "cheap": "cheap",
    }


def test_select_baseline_profiles_falls_back_to_tier_scoring():
    config = NorthStackConfig(
        name="sel",
        profiles=[
            _profile("big", 20.0, []),
            _profile("small", 0.1, []),
            _profile("mid", 5.0, []),
        ],
    )
    picks = select_baseline_profiles(config)
    assert picks["strong_single"] == "big"
    assert picks["cheap"] == "small"
    # No specialist configured: the expert baseline degrades to the strong pick.
    assert picks["singleton_expert"] == "big"


def test_select_baseline_profiles_requires_profiles():
    with pytest.raises(ValueError, match="at least one configured profile"):
        select_baseline_profiles(NorthStackConfig(name="empty", profiles=[]))


# LiveWorkerStrategy end to end (fake gateway)


async def test_live_worker_strategy_success_is_verified(
    template: Path, tmp_path: Path, monkeypatch
):
    factory = GatewayFactory(
        [
            _tool("create", {"path": "answer.txt", "content": "the answer is 42"}),
            _ok("Created the file."),
        ]
    )
    monkeypatch.setattr("northstack.application.benchmark_live.ModelGateway", factory)
    task = _task(
        template,
        request="create a file called answer.txt containing the text 'the answer is 42'",
        checks=[
            HiddenCheck(name="answer-file", path="answer.txt", content_contains="the answer is 42")
        ],
    )
    strategy = LiveWorkerStrategy(
        _config(),
        configuration=Configuration.STRONG_SINGLE,
        profile_name="cheap",
        runs_dir=tmp_path / "runs",
        suite_dir=template.parent,
    )
    result = await strategy.run(task, 0)

    assert result.outcome == RunOutcome.VERIFIED
    assert result.verified_score == 1.0
    assert not result.false_acceptance
    assert not result.false_rejection
    assert result.metrics.input_tokens > 0
    assert result.metrics.calls >= 1
    assert result.metrics.tool_operations == 1
    # The deliverable lives in the snapshot, not the immutable template.
    run_dir = tmp_path / "runs" / "t1-0-strong_single"
    assert (run_dir / "workspace" / "answer.txt").exists()
    assert not (template / "answer.txt").exists()
    assert (run_dir / "meta.json").exists()


async def test_live_worker_strategy_claim_without_deliverable_is_false_acceptance(
    template: Path, tmp_path: Path, monkeypatch
):
    factory = GatewayFactory([_ok("I definitely did it.")])
    monkeypatch.setattr("northstack.application.benchmark_live.ModelGateway", factory)
    task = _task(
        template,
        checks=[HiddenCheck(name="answer-file", path="answer.txt", content_contains="42")],
    )
    strategy = LiveWorkerStrategy(
        _config(),
        configuration=Configuration.CHEAP_BEST_OF_N,
        profile_name="cheap",
        runs_dir=tmp_path / "runs",
        suite_dir=template.parent,
    )
    result = await strategy.run(task, 0)

    # The worker claimed completion; the hidden check disagrees.
    assert result.outcome == RunOutcome.FAILED
    assert result.verified_score == 0.0
    assert result.false_acceptance is True


async def test_live_worker_strategy_failed_worker_is_false_rejection_when_checks_pass(
    template: Path, tmp_path: Path, monkeypatch
):
    # A deliverable already in the template + a failing worker (provider error
    # surfaces as ok=False): hidden checks pass, the claim failed.
    (template / "answer.txt").write_text("the answer is 42")

    class _BrokenFactory(GatewayFactory):
        def __call__(self, config, artifact_store=None):
            gateway = super().__call__(config, artifact_store)

            async def broken(request, profile, client, api_key):
                raise RuntimeError("endpoint down")

            gateway._adapters[Protocol.OPENAI_CHAT].complete = broken
            return gateway

    monkeypatch.setattr(
        "northstack.application.benchmark_live.ModelGateway",
        _BrokenFactory([]),
    )
    task = _task(
        template,
        checks=[HiddenCheck(name="answer-file", path="answer.txt", content_contains="42")],
    )
    strategy = LiveWorkerStrategy(
        _config(),
        configuration=Configuration.SINGLETON_EXPERT,
        profile_name="cheap",
        runs_dir=tmp_path / "runs",
        suite_dir=template.parent,
    )
    result = await strategy.run(task, 0)

    assert result.outcome == RunOutcome.VERIFIED  # hidden truth
    assert result.false_rejection is True  # the claim failed anyway
    assert result.error  # the failure is retained, not dropped


def test_live_strategies_builds_all_four_configurations(tmp_path: Path):
    strategies = live_strategies(_config(), runs_dir=tmp_path, suite_dir=tmp_path)
    assert set(strategies) == set(Configuration)
    assert isinstance(strategies[Configuration.COMPANY], object)


async def test_live_company_strategy_end_to_end(template: Path, tmp_path: Path, monkeypatch):
    """Full pipeline: snapshot -> company run (fake gateway) -> ledger metrics
    -> hidden-check scoring. The goal matches the deterministic file-content
    intake shortcut, so the contract synthesizes a hard file_diff criterion
    without any model call; the fake worker then writes the deliverable."""
    from northstack.application.benchmark_live import LiveCompanyStrategy

    factory = GatewayFactory(
        [
            _tool("create", {"path": "hello.txt", "content": "hello live"}),
            _ok("Created the file."),
        ]
    )
    monkeypatch.setattr("northstack.application.build.ModelGateway", factory)
    task = _task(
        template,
        id="hello",
        request="create a file called hello.txt containing the text 'hello live'",
        checks=[HiddenCheck(name="hello-file", path="hello.txt", content_contains="hello live")],
    )
    strategy = LiveCompanyStrategy(_config(), runs_dir=tmp_path / "runs", suite_dir=tmp_path)
    result = await strategy.run(task, 0)

    assert result.outcome == RunOutcome.VERIFIED
    assert result.verified_score == 1.0
    assert not result.false_acceptance and not result.false_rejection
    assert result.metrics.input_tokens > 0
    assert result.metrics.calls >= 1
    run_dir = tmp_path / "runs" / "hello-0-northstack"
    assert (run_dir / "workspace" / "hello.txt").read_text() == "hello live"
    assert (run_dir / "ledger.db").exists()
    assert (run_dir / "meta.json").exists()
    assert not (template / "hello.txt").exists()


async def test_live_company_strategy_failure_is_retained_not_dropped(
    template: Path, tmp_path: Path, monkeypatch
):
    """A worker that claims success without delivering fails the company's own
    hard gate: the claimed outcome is FAILED, and the hidden check agrees."""
    from northstack.application.benchmark_live import LiveCompanyStrategy

    factory = GatewayFactory([_ok("Consider it done.")])
    monkeypatch.setattr("northstack.application.build.ModelGateway", factory)
    task = _task(
        template,
        id="nope",
        request="create a file called hello.txt containing the text 'hello live'",
        checks=[HiddenCheck(name="hello-file", path="hello.txt", content_contains="hello live")],
    )
    strategy = LiveCompanyStrategy(_config(), runs_dir=tmp_path / "runs", suite_dir=tmp_path)
    result = await strategy.run(task, 0)

    # The company's own hard gate failed the run (file absent)...
    assert result.outcome == RunOutcome.FAILED
    assert result.verified_score == 0.0
    # ...and since the company did not claim VERIFIED, no false acceptance.
    assert not result.false_acceptance


async def test_full_live_benchmark_runner_all_four_configurations(
    template: Path, tmp_path: Path, monkeypatch
):
    """The exact path the CLI drives: BenchmarkRunner over live_strategies.

    The goal matches the deterministic file-content intake shortcut, so the
    company's contract synthesizes a hard criterion without model analysis;
    every baseline is a bare worker loop. All four configurations must produce
    retained results with paired bootstrap intervals in the report."""
    from northstack.application.benchmark import BenchmarkConfig, BenchmarkRunner
    from northstack.application.benchmark_live import live_strategies

    worker_factory = GatewayFactory(
        [
            _tool("create", {"path": "hello.txt", "content": "hello live"}),
            _ok("Created the file."),
        ]
    )
    company_factory = GatewayFactory(
        [
            _tool("create", {"path": "hello.txt", "content": "hello live"}),
            _ok("Created the file."),
        ]
    )
    monkeypatch.setattr("northstack.application.benchmark_live.ModelGateway", worker_factory)
    monkeypatch.setattr("northstack.application.build.ModelGateway", company_factory)

    task = _task(
        template,
        id="hello",
        request="create a file called hello.txt containing the text 'hello live'",
        checks=[HiddenCheck(name="hello-file", path="hello.txt", content_contains="hello live")],
    )
    report = await BenchmarkRunner(
        strategies=live_strategies(_config(), runs_dir=tmp_path / "runs", suite_dir=tmp_path),
        config=BenchmarkConfig(repeats=1, cheap_candidates=1, bootstrap_samples=100, seed=5),
    ).run([task])

    assert len(report.results) == 4
    assert all(r.outcome == RunOutcome.VERIFIED for r in report.results)
    assert all(r.verified_score == 1.0 for r in report.results)
    assert not any(r.false_acceptance or r.false_rejection for r in report.results)
    # Paired company-minus-baseline intervals exist for the three baselines.
    assert set(report.paired_differences) == {
        Configuration.STRONG_SINGLE,
        Configuration.CHEAP_BEST_OF_N,
        Configuration.SINGLETON_EXPERT,
    }
    # Every configuration consumed real (fake-gateway) resources.
    assert all(r.metrics.input_tokens > 0 for r in report.results)
    # The template survived four live runs untouched.
    assert not (template / "hello.txt").exists()


def test_company_budget_carries_all_ceilings(template):
    """The company budget must bound wall time, not just tokens/cost.

    A degraded pool returning near-empty responses accrues ~0 tokens per call,
    so a token-only ceiling lets a worker spin indefinitely -- baselines are
    wall-capped; the company strategy was not.
    """
    from northstack.application.benchmark_live import _company_budget

    task = _task(template)
    budget = _company_budget(task)
    assert budget.token_limit == 10_000
    assert budget.cost_limit_usd == 1.0
    assert budget.max_wall_time_seconds == task.wall_time_limit_seconds > 0
    override = _company_budget(task, max_retries_override=2)
    assert override.max_retries == 2
