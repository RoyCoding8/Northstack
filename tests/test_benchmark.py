"""Behavior tests for the scientific benchmark seam."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from northstack.application.benchmark import (
    BenchmarkConfig,
    BenchmarkMetrics,
    BenchmarkRunner,
    BenchmarkSuite,
    BenchmarkTask,
    Configuration,
    RunResult,
    bootstrap_mean_ci,
    load_task_suite,
)
from northstack.domain import RunOutcome
from northstack.interfaces.cli import app


@pytest.mark.parametrize("task_id", ["../escape", "a/b", "a\\b", "two words", "CON", ".hidden"])
def test_benchmark_task_rejects_unsafe_id(task_id: str) -> None:
    with pytest.raises(ValidationError, match="portable slug"):
        BenchmarkTask(id=task_id, category="x", request="x", workspace=".")


def test_benchmark_suite_rejects_duplicate_task_ids() -> None:
    task = BenchmarkTask(id="duplicate", category="x", request="x", workspace=".")
    with pytest.raises(ValidationError, match="duplicate task ids"):
        BenchmarkSuite(name="suite", tasks=[task, task])


@pytest.mark.parametrize(
    ("field", "value"),
    [("repeats", 101), ("cheap_candidates", 101), ("bootstrap_samples", 1_000_001)],
)
def test_benchmark_config_rejects_excessive_resource_counts(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        BenchmarkConfig(**{field: value})


class ScriptedStrategy:
    """Returns configured results and records resource ceilings."""

    def __init__(self, results: dict[tuple[str, int], RunResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, int]] = []

    async def run(self, task: BenchmarkTask, repeat_index: int) -> RunResult:
        self.calls.append((task.id, repeat_index, task.token_limit))
        return self._results[(task.id, repeat_index)]


def result(
    task_id: str,
    configuration: Configuration,
    outcome: RunOutcome,
    *,
    verified_score: float,
    tokens: int,
    calls: int = 1,
    wall_time_ms: int = 10,
    retries: int = 0,
    false_acceptance: bool = False,
) -> RunResult:
    return RunResult(
        task_id=task_id,
        configuration=configuration,
        repeat_index=0,
        outcome=outcome,
        verified_score=verified_score,
        false_acceptance=false_acceptance,
        metrics=BenchmarkMetrics(
            input_tokens=tokens,
            output_tokens=0,
            calls=calls,
            wall_time_ms=wall_time_ms,
            retries=retries,
            tool_operations=0,
            configured_cost_usd=0.0,
        ),
    )


def test_load_task_suite_preserves_hidden_checks_and_budget(tmp_path: Path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "pilot",
                "tasks": [
                    {
                        "id": "bug-1",
                        "category": "bug_fix",
                        "request": "Fix add()",
                        "workspace": "fixtures/bug-1",
                        "hidden_checks": ["python -m pytest -q"],
                        "token_limit": 5000,
                        "cost_limit_usd": 1.0,
                    }
                ],
            }
        )
    )

    suite = load_task_suite(suite_path)

    assert suite.name == "pilot"
    assert suite.tasks[0].hidden_checks == ["python -m pytest -q"]
    assert suite.tasks[0].token_limit == 5000


async def test_runner_keeps_failures_abstentions_and_equal_resource_ceiling():
    task = BenchmarkTask(
        id="task-1",
        category="bug_fix",
        request="Fix it",
        workspace="fixture",
        hidden_checks=["pytest"],
        token_limit=1000,
        cost_limit_usd=1.0,
    )
    strategies = {
        Configuration.STRONG_SINGLE: ScriptedStrategy(
            {
                ("task-1", 0): result(
                    "task-1",
                    Configuration.STRONG_SINGLE,
                    RunOutcome.FAILED,
                    verified_score=0,
                    tokens=100,
                )
            }
        ),
        Configuration.CHEAP_BEST_OF_N: ScriptedStrategy(
            {
                ("task-1", 0): result(
                    "task-1",
                    Configuration.CHEAP_BEST_OF_N,
                    RunOutcome.ABSTAINED,
                    verified_score=0,
                    tokens=90,
                )
            }
        ),
        Configuration.SINGLETON_EXPERT: ScriptedStrategy(
            {
                ("task-1", 0): result(
                    "task-1",
                    Configuration.SINGLETON_EXPERT,
                    RunOutcome.VERIFIED,
                    verified_score=1,
                    tokens=120,
                )
            }
        ),
        Configuration.COMPANY: ScriptedStrategy(
            {
                ("task-1", 0): result(
                    "task-1", Configuration.COMPANY, RunOutcome.FAILED, verified_score=0, tokens=80
                )
            }
        ),
    }
    runner = BenchmarkRunner(
        strategies=strategies,
        config=BenchmarkConfig(repeats=1, cheap_candidates=1, bootstrap_samples=100, seed=7),
    )

    report = await runner.run([task])

    assert len(report.results) == 4
    assert {r.outcome for r in report.results} == {
        RunOutcome.FAILED,
        RunOutcome.ABSTAINED,
        RunOutcome.VERIFIED,
    }
    for strategy in strategies.values():
        assert strategy.calls == [("task-1", 0, 1000)]


async def test_runner_selects_best_of_n_cheap_candidate_with_verification_law():
    task = BenchmarkTask(
        id="task-1",
        category="feature",
        request="Add it",
        workspace="fixture",
        hidden_checks=["pytest"],
        token_limit=1000,
        cost_limit_usd=1.0,
    )
    cheap = ScriptedStrategy(
        {
            ("task-1", 0): result(
                "task-1",
                Configuration.CHEAP_BEST_OF_N,
                RunOutcome.FAILED,
                verified_score=0,
                tokens=20,
            ),
            ("task-1", 1): result(
                "task-1",
                Configuration.CHEAP_BEST_OF_N,
                RunOutcome.VERIFIED,
                verified_score=1,
                tokens=30,
            ),
            ("task-1", 2): result(
                "task-1",
                Configuration.CHEAP_BEST_OF_N,
                RunOutcome.ABSTAINED,
                verified_score=0,
                tokens=25,
            ),
        }
    )
    strategies = {
        c: ScriptedStrategy(
            {("task-1", 0): result("task-1", c, RunOutcome.FAILED, verified_score=0, tokens=10)}
        )
        for c in Configuration
        if c != Configuration.CHEAP_BEST_OF_N
    }
    strategies[Configuration.CHEAP_BEST_OF_N] = cheap
    runner = BenchmarkRunner(
        strategies=strategies,
        config=BenchmarkConfig(repeats=1, cheap_candidates=3, bootstrap_samples=100, seed=3),
    )

    report = await runner.run([task])
    selected = next(r for r in report.results if r.configuration == Configuration.CHEAP_BEST_OF_N)

    assert selected.outcome == RunOutcome.VERIFIED
    assert selected.metrics.calls == 3
    assert selected.metrics.total_tokens == 75
    assert [call[1] for call in cheap.calls] == [0, 1, 2]


def test_bootstrap_mean_ci_is_seeded_and_contains_observed_mean():
    first = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], samples=500, seed=11)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], samples=500, seed=11)

    assert first == second
    assert first.low <= 2.5 <= first.high


async def test_report_contains_paired_company_differences_and_serializes(tmp_path: Path):
    tasks = [
        BenchmarkTask(
            id="a",
            category="bug_fix",
            request="a",
            workspace="a",
            token_limit=100,
            cost_limit_usd=1,
        ),
        BenchmarkTask(
            id="b",
            category="refactor",
            request="b",
            workspace="b",
            token_limit=100,
            cost_limit_usd=1,
        ),
    ]
    strategies = {}
    for configuration in Configuration:
        score = 1.0 if configuration == Configuration.COMPANY else 0.0
        strategies[configuration] = ScriptedStrategy(
            {
                (task.id, 0): result(
                    task.id,
                    configuration,
                    RunOutcome.VERIFIED if score else RunOutcome.FAILED,
                    verified_score=score,
                    tokens=50,
                )
                for task in tasks
            }
        )
    runner = BenchmarkRunner(
        strategies=strategies,
        config=BenchmarkConfig(repeats=1, cheap_candidates=1, bootstrap_samples=100, seed=5),
    )

    report = await runner.run(tasks)
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    report.write(json_path, markdown_path)

    assert report.paired_differences[Configuration.STRONG_SINGLE].mean == 1.0
    assert len(json.loads(json_path.read_text())["results"]) == 8
    markdown = markdown_path.read_text()
    assert "False acceptance" in markdown
    assert "Abstained" in markdown
    assert "northstack" in markdown


def test_benchmark_cli_runs_fixture_strategies(tmp_path: Path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "cli-pilot",
                "tasks": [
                    {
                        "id": "doc-1",
                        "category": "documentation",
                        "request": "Update docs",
                        "workspace": "fixture",
                        "token_limit": 100,
                        "cost_limit_usd": 1.0,
                        "fixture_results": {
                            configuration.value: {
                                "outcome": "verified"
                                if configuration == Configuration.COMPANY
                                else "failed",
                                "verified_score": 1.0
                                if configuration == Configuration.COMPANY
                                else 0.0,
                                "input_tokens": 10,
                            }
                            for configuration in Configuration
                        },
                    }
                ],
            }
        )
    )
    output_dir = tmp_path / "out"

    response = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--suite",
            str(suite_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "100",
        ],
    )

    assert response.exit_code == 0, response.output
    assert (output_dir / "benchmark-report.json").exists()
    assert (output_dir / "benchmark-report.md").exists()
