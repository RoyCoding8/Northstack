"""Tests for component ablations, the Pareto frontier, and the held-out guard.

Hermetic: scripted strategies and fake gateways only. Pinned:

  - the protocol's dominance rule (quality no worse, every resource no
    worse, at least one strictly better) and frontier membership;
  - ablation results are retained and summarized separately, excluded from
    the primary configuration summaries and paired analysis;
  - ``ablation_strategies`` removes exactly one mechanism per variant;
  - a cohort=heldout suite refuses to run without --allow-heldout.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from northstack.application.benchmark import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    Configuration,
    RunResult,
    analyze_frontier,
)
from northstack.domain import RunOutcome
from northstack.interfaces.cli import app


def _summary(configuration: Configuration, score: float, **resources: int) -> dict:
    """A ConfigurationSummary-shaped dict with resource overrides."""
    from northstack.application.benchmark import ConfigurationSummary

    base = dict(
        total_tokens=1000,
        calls=10,
        wall_time_ms=1000,
        retries=0,
        tool_operations=10,
        configured_cost_usd=1.0,
    )
    base.update(resources)
    return ConfigurationSummary(
        configuration=configuration,
        runs=1,
        verified=1,
        failed=0,
        abstained=0,
        false_acceptance=0,
        false_rejection=0,
        mean_verified_score=score,
        **base,
    ).model_dump()


# Frontier dominance


def test_dominance_requires_better_everywhere_and_strict_somewhere():
    summaries = _load_summaries(
        {
            Configuration.COMPANY: _summary(Configuration.COMPANY, 0.9, total_tokens=500),
            Configuration.STRONG_SINGLE: _summary(
                Configuration.STRONG_SINGLE, 0.9, total_tokens=800
            ),
        }
    )
    frontier = analyze_frontier(summaries)
    company = next(e for e in frontier.entries if e.configuration == Configuration.COMPANY)
    strong = next(e for e in frontier.entries if e.configuration == Configuration.STRONG_SINGLE)
    # Equal quality, fewer tokens: company dominates strong_single.
    assert company.pareto_efficient is True
    assert company.dominated_by == []
    assert strong.pareto_efficient is False
    assert strong.dominated_by == ["northstack"]


def _load_summaries(raw: dict) -> dict:
    from northstack.application.benchmark import ConfigurationSummary

    return {c: ConfigurationSummary.model_validate(d) for c, d in raw.items()}


def test_higher_quality_but_worse_resource_is_not_dominance():
    summaries = _load_summaries(
        {
            Configuration.COMPANY: _summary(Configuration.COMPANY, 0.9, total_tokens=2000),
            Configuration.STRONG_SINGLE: _summary(
                Configuration.STRONG_SINGLE, 0.8, total_tokens=1000
            ),
        }
    )
    frontier = analyze_frontier(summaries)
    assert all(e.pareto_efficient for e in frontier.entries)
    assert all(e.dominated_by == [] for e in frontier.entries)


def test_lower_quality_with_better_resources_is_not_dominance():
    summaries = _load_summaries(
        {
            Configuration.COMPANY: _summary(Configuration.COMPANY, 0.7, total_tokens=500),
            Configuration.STRONG_SINGLE: _summary(
                Configuration.STRONG_SINGLE, 0.8, total_tokens=800
            ),
        }
    )
    frontier = analyze_frontier(summaries)
    # Neither dominates: quality and resources point in opposite directions.
    assert all(e.dominated_by == [] for e in frontier.entries)


def test_frontier_markdown_renders_all_axes():
    summaries = _load_summaries(
        {
            Configuration.COMPANY: _summary(Configuration.COMPANY, 0.9),
            Configuration.STRONG_SINGLE: _summary(
                Configuration.STRONG_SINGLE, 0.8, total_tokens=5000
            ),
        }
    )
    text = analyze_frontier(summaries).to_markdown()
    assert "Quality-resource frontier" in text
    assert "northstack" in text
    assert "strong_single" in text
    assert "frontier" in text


# Runner ablations


class ScriptedStrategy:
    def __init__(self, result: RunResult) -> None:
        self._result = result

    async def run(self, task: BenchmarkTask, repeat_index: int) -> RunResult:
        return self._result.model_copy(update={"task_id": task.id, "repeat_index": repeat_index})


def _result(configuration: Configuration, score: float, tokens: int) -> RunResult:
    return RunResult(
        task_id="t",
        configuration=configuration,
        repeat_index=0,
        outcome=RunOutcome.VERIFIED if score >= 1.0 else RunOutcome.FAILED,
        verified_score=score,
        metrics=BenchmarkMetrics(input_tokens=tokens),
    )


async def test_ablations_retained_separately_and_excluded_from_primaries():
    task = BenchmarkTask(id="t", category="x", request="r", workspace="w")
    strategies = {c: ScriptedStrategy(_result(c, 0.5, 100)) for c in Configuration}
    ablations = {"no_routing": ScriptedStrategy(_result(Configuration.COMPANY, 0.25, 100))}
    report = await BenchmarkRunner(
        strategies=strategies,
        config=BenchmarkConfig(repeats=1, cheap_candidates=1, bootstrap_samples=50, seed=1),
        ablations=ablations,
    ).run([task])

    # Primary summaries unchanged by ablation presence: 4 runs, 100 tokens.
    assert len(report.results) == 4
    assert all(s.runs == 1 for s in report.summaries.values())
    assert report.summaries[Configuration.COMPANY].total_tokens == 100
    # The ablation run is retained under its own label.
    assert len(report.ablation_results) == 1
    assert report.ablation_results[0].ablation == "no_routing"
    assert report.ablation_results[0].configuration == Configuration.COMPANY
    assert report.ablation_summaries["no_routing"].mean_verified_score == 0.25
    # Frontier present and primary-only.
    assert report.frontier is not None
    assert {e.configuration for e in report.frontier.entries} == set(Configuration)
    # Markdown carries both sections.
    md = report.to_markdown()
    assert "Component ablations" in md
    assert "no_routing" in md
    assert "Quality-resource frontier" in md


async def test_report_without_ablations_has_no_ablation_section():
    task = BenchmarkTask(id="t", category="x", request="r", workspace="w")
    strategies = {c: ScriptedStrategy(_result(c, 0.5, 100)) for c in Configuration}
    report = await BenchmarkRunner(
        strategies=strategies,
        config=BenchmarkConfig(repeats=1, cheap_candidates=1, bootstrap_samples=50, seed=1),
    ).run([task])
    assert report.ablation_results == []
    assert report.ablation_summaries == {}
    assert "Component ablations" not in report.to_markdown()


# ablation_strategies: one mechanism removed per variant


def test_ablation_strategies_remove_exactly_one_mechanism_each(tmp_path: Path):
    from northstack.application.benchmark_live import ablation_strategies
    from northstack.config import (
        ModelProfile,
        NorthStackConfig,
        Protocol,
        Role,
        RouteMapping,
        RunConfig,
    )

    config = NorthStackConfig(
        name="ab",
        profiles=[
            ModelProfile(
                name="cheap",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost/v1",
                model="m",
                roles=[Role.WORKER],
                max_concurrency=2,
            ),
            ModelProfile(
                name="plan",
                protocol=Protocol.OPENAI_CHAT,
                base_url="http://localhost/v1",
                model="m",
                roles=[Role.PLANNER],
                max_concurrency=2,
            ),
        ],
        routing=[
            RouteMapping(role=Role.WORKER, profiles=["cheap"]),
            RouteMapping(role=Role.PLANNER, profiles=["plan"]),
        ],
        run=RunConfig(planner_mode="model", falsifier_mode="model"),
    )
    ablations = ablation_strategies(config, runs_dir=tmp_path, suite_dir=tmp_path)
    assert set(ablations) == {
        "no_routing",
        "single_cell",
        "deterministic_intake",
        "minimal_recovery",
        "no_falsifier",
    }

    no_routing = ablations["no_routing"]
    assert no_routing._config.routing == []
    assert no_routing._config.run.planner_mode == "model"  # untouched

    single_cell = ablations["single_cell"]
    assert single_cell._config.run.planner_mode == "single"
    assert single_cell._config.routing  # untouched

    deterministic = ablations["deterministic_intake"]
    assert deterministic._analysis_runner is not None
    assert deterministic._config.run.planner_mode == "model"  # untouched

    minimal = ablations["minimal_recovery"]
    assert minimal._max_retries_override == 1

    no_falsifier = ablations["no_falsifier"]
    assert no_falsifier._config.run.falsifier_mode == "off"
    assert config.run.falsifier_mode == "model"  # base config untouched (copies)


async def test_live_ablation_run_end_to_end(tmp_path: Path, monkeypatch):
    """A deterministic-intake ablation run through the fake gateway: the
    label rides the RunResult and the run directory; the retained outcome
    still follows the hidden checks."""
    from tests.test_benchmark_live import GatewayFactory, _config, _task, _tool, _ok
    from northstack.application.benchmark import HiddenCheck
    from northstack.application.benchmark_live import LiveCompanyStrategy
    from northstack.application.contracting import DeterministicAnalysisRunner

    template = tmp_path / "template"
    template.mkdir()
    (template / "README.md").write_text("# t\n")

    factory = GatewayFactory(
        [
            _tool("create", {"path": "hello.txt", "content": "hi"}),
            _ok("done"),
        ]
    )
    monkeypatch.setattr("northstack.application.build.ModelGateway", factory)
    task = _task(
        template,
        id="abl",
        request="create a file called hello.txt containing the text 'hi'",
        checks=[HiddenCheck(name="hf", path="hello.txt", content_contains="hi")],
    )
    strategy = LiveCompanyStrategy(
        _config(),
        runs_dir=tmp_path / "runs",
        suite_dir=tmp_path,
        label="deterministic_intake",
        analysis_runner=DeterministicAnalysisRunner(),
    )
    result = await strategy.run(task, 0)
    assert result.ablation == "deterministic_intake"
    assert result.outcome == RunOutcome.VERIFIED
    assert result.verified_score == 1.0
    assert (tmp_path / "runs" / "abl-0-deterministic_intake" / "meta.json").exists()


# Held-out suite guard


def _suite_file(tmp_path: Path, cohort: str | None) -> Path:
    suite: dict = {
        "name": "s",
        "tasks": [
            {
                "id": "t",
                "category": "x",
                "request": "r",
                "workspace": "w",
            }
        ],
    }
    if cohort is not None:
        suite["cohort"] = cohort
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    return path


def test_heldout_suite_refuses_without_flag(tmp_path: Path):
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--suite",
            str(_suite_file(tmp_path, "heldout")),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "heldout" in result.output
    assert "--allow-heldout" in result.output


def test_heldout_suite_runs_with_explicit_flag(tmp_path: Path):
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--suite",
            str(_suite_file(tmp_path, "heldout")),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-heldout",
        ],
    )
    assert result.exit_code == 0


def test_default_cohort_is_dev_and_needs_no_flag():
    assert BenchmarkSuite.model_validate({"name": "s", "tasks": []}).cohort == "dev"
