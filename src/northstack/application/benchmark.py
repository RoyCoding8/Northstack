"""Scientific benchmark runner and machine-readable reporting.

Public seam:
  - load_task_suite(path) -> BenchmarkSuite
  - BenchmarkRunner(strategies, config).run(tasks) -> BenchmarkReport
  - bootstrap_mean_ci(values, samples, seed) -> ConfidenceInterval

The runner retains every failed and abstained run. It compares configurations
under each task's same declared resource ceiling and computes paired company
minus baseline differences with a deterministic seeded bootstrap.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from northstack.domain import RunOutcome

_RUN_OUTCOME_BY_LABEL = {
    RunOutcome.VERIFIED.value: RunOutcome.VERIFIED,
    RunOutcome.ABSTAINED.value: RunOutcome.ABSTAINED,
    RunOutcome.FAILED.value: RunOutcome.FAILED,
}
_TASK_ID = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?\Z")
_WINDOWS_RESERVED = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)


class Configuration(StrEnum):
    """End-to-end configurations in the scientific comparison."""

    STRONG_SINGLE = "strong_single"
    CHEAP_BEST_OF_N = "cheap_best_of_n"
    SINGLETON_EXPERT = "singleton_expert"
    COMPANY = "northstack"


class HiddenCheck(BaseModel):
    """One post-hoc executable check the system under test never sees.

    A command check (``argv`` non-empty) passes when the exact-argv subprocess
    exits with ``expect_exit_code``; a file check (``path`` set, ``argv``
    empty) passes when the path exists and optionally contains
    ``content_contains``. Checks run against the finished workspace snapshot
    through the restricted-workspace chokepoint (``shell=False``).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    argv: list[str] = Field(default_factory=list)
    expect_exit_code: int = 0
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_output_bytes: int = Field(default=131_072, ge=1)
    path: str = ""
    content_contains: str = ""


class BenchmarkTask(BaseModel):
    """One reproducible task and its common resource ceiling."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    request: str = Field(min_length=1)
    workspace: str = Field(min_length=1)
    hidden_checks: list[str] = Field(default_factory=list)
    checks: list[HiddenCheck] = Field(default_factory=list)
    token_limit: int = Field(default=100_000, ge=1)
    cost_limit_usd: float = Field(default=5.0, ge=0.0)
    wall_time_limit_seconds: float = Field(default=1800.0, gt=0.0)
    fixture_results: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not _TASK_ID.fullmatch(value) or value.casefold() in _WINDOWS_RESERVED:
            raise ValueError("task id must be a portable slug")
        return value


class BenchmarkSuite(BaseModel):
    """Named benchmark task collection.

    ``cohort`` implements the protocol's dev/held-out discipline: a
    ``heldout`` suite is frozen -- it must not be run during development or
    tuning, only after the protocol is frozen. The CLI enforces this with an
    explicit ``--allow-heldout`` override so accidental inclusion fails
    loudly instead of silently contaminating the held-out set.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    name: str = Field(min_length=1)
    cohort: Literal["dev", "heldout"] = Field(default="dev")
    tasks: list[BenchmarkTask]

    @model_validator(mode="after")
    def unique_task_ids(self) -> Self:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task ids")
        return self


class BenchmarkConfig(BaseModel):
    """Frozen benchmark execution and analysis settings."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    repeats: int = Field(default=1, ge=1, le=100)
    cheap_candidates: int = Field(default=3, ge=1, le=100)
    bootstrap_samples: int = Field(default=2_000, ge=1, le=1_000_000)
    seed: int = 1


class BenchmarkMetrics(BaseModel):
    """Provider-neutral resource usage for one retained result."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    calls: int = Field(default=0, ge=0)
    wall_time_ms: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    tool_operations: int = Field(default=0, ge=0)
    configured_cost_usd: float = Field(default=0.0, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(self, other: BenchmarkMetrics) -> BenchmarkMetrics:
        """Add resource usage without hiding rejected candidates."""
        return BenchmarkMetrics(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            calls=self.calls + other.calls,
            wall_time_ms=self.wall_time_ms + other.wall_time_ms,
            retries=self.retries + other.retries,
            tool_operations=self.tool_operations + other.tool_operations,
            configured_cost_usd=self.configured_cost_usd + other.configured_cost_usd,
        )


class RunResult(BaseModel):
    """One retained task/configuration/repeat result.

    ``ablation`` is non-empty for component-ablation runs: the run is a
    company variant with one mechanism removed, retained in the report's
    ``ablation_results`` and excluded from the primary configuration
    summaries and paired analysis.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    configuration: Configuration
    repeat_index: int = Field(ge=0)
    outcome: RunOutcome
    verified_score: float = Field(ge=0.0, le=1.0)
    false_acceptance: bool = False
    false_rejection: bool = False
    error: str = ""
    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)
    ablation: str = ""


class ConfidenceInterval(BaseModel):
    """Observed mean and percentile confidence interval."""

    model_config = ConfigDict(frozen=True)

    mean: float
    low: float
    high: float
    samples: int = Field(ge=1)


class ConfigurationSummary(BaseModel):
    """Aggregate outcomes and resources for one configuration."""

    model_config = ConfigDict(frozen=True)

    configuration: Configuration
    runs: int
    verified: int
    failed: int
    abstained: int
    false_acceptance: int
    false_rejection: int
    mean_verified_score: float
    total_tokens: int
    calls: int
    wall_time_ms: int
    retries: int
    tool_operations: int
    configured_cost_usd: float


class AblationSummary(BaseModel):
    """Aggregate outcomes and resources for one component ablation.

    Mirrors :class:`ConfigurationSummary` keyed by ablation label instead of
    a preregistered configuration.
    """

    model_config = ConfigDict(frozen=True)

    ablation: str
    runs: int
    verified: int
    failed: int
    abstained: int
    false_acceptance: int
    false_rejection: int
    mean_verified_score: float
    total_tokens: int
    calls: int
    wall_time_ms: int
    retries: int
    tool_operations: int
    configured_cost_usd: float


RESOURCE_AXES: tuple[tuple[str, str, str], ...] = (
    ("total_tokens", "Tokens", "tokens"),
    ("calls", "Calls", "calls"),
    ("wall_time_ms", "Wall time", "ms"),
    ("retries", "Retries", "retries"),
    ("tool_operations", "Tool ops", "ops"),
    ("configured_cost_usd", "Cost", "USD"),
)


class FrontierEntry(BaseModel):
    """One configuration's position on the quality-resource frontier."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    configuration: Configuration
    mean_verified_score: float
    resources: dict[str, int | float] = Field(description="axis attr -> total")
    dominated_by: list[str] = Field(default_factory=list)
    pareto_efficient: bool = True


class FrontierAnalysis(BaseModel):
    """The complete frontier over the preregistered configurations.

    A dominates B when A's quality is no worse, every resource axis is no
    worse, and at least one is strictly better (the protocol's dominance
    rule). Frontier membership -- not a winner -- is the claim the data
    supports.
    """

    model_config = ConfigDict(frozen=True)

    entries: list[FrontierEntry]
    resource_axes: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## Quality-resource frontier (Pareto)",
            "",
        ]
        if not self.entries:
            lines.append("(no configurations)")
            return "\n".join(lines) + "\n"
        lines.extend(
            [
                "Dominance: quality no worse AND every resource axis no worse AND",
                "at least one strictly better. Frontier membership is the claim",
                "the data supports -- not a winner.",
                "",
            ]
        )
        header = (
            "| Configuration | Quality | "
            + " | ".join(label for _, label, _ in RESOURCE_AXES)
            + " | Dominated by |"
        )
        lines.append(header)
        lines.append("|---" * (2 + len(RESOURCE_AXES) + 1) + "|")
        for entry in self.entries:
            values = " | ".join(
                _fmt_resource(entry.resources[attr]) for attr, _, _ in RESOURCE_AXES
            )
            dominated = ", ".join(entry.dominated_by) if entry.dominated_by else "— (frontier)"
            lines.append(
                f"| {entry.configuration.value} | {entry.mean_verified_score:.3f} | "
                f"{values} | {dominated} |"
            )
        return "\n".join(lines) + "\n"


def _fmt_resource(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return f"{value}"


def _dominates(a: ConfigurationSummary, b: ConfigurationSummary) -> bool:
    """The protocol's dominance rule: A over B."""
    if a.mean_verified_score < b.mean_verified_score:
        return False
    strictly_better = False
    for attr, _, _ in RESOURCE_AXES:
        if getattr(a, attr) > getattr(b, attr):
            return False
        if getattr(a, attr) < getattr(b, attr):
            strictly_better = True
    return a.mean_verified_score > b.mean_verified_score or strictly_better


def analyze_frontier(
    summaries: Mapping[Configuration, ConfigurationSummary],
) -> FrontierAnalysis:
    """Compute Pareto dominance over the preregistered configurations."""
    entries: list[FrontierEntry] = []
    for configuration, summary in summaries.items():
        dominated_by = sorted(
            other.configuration.value
            for other in summaries.values()
            if other.configuration != configuration and _dominates(other, summary)
        )
        entries.append(
            FrontierEntry(
                configuration=configuration,
                mean_verified_score=summary.mean_verified_score,
                resources={attr: getattr(summary, attr) for attr, _, _ in RESOURCE_AXES},
                dominated_by=dominated_by,
                pareto_efficient=not dominated_by,
            )
        )
    order = {c: i for i, c in enumerate(Configuration)}
    entries.sort(key=lambda e: order.get(e.configuration, len(order)))
    return FrontierAnalysis(
        entries=entries,
        resource_axes=[attr for attr, _, _ in RESOURCE_AXES],
    )


class BenchmarkReport(BaseModel):
    """Complete report; no run is discarded from ``results``."""

    model_config = ConfigDict(frozen=True)

    config: BenchmarkConfig
    results: list[RunResult]
    summaries: dict[Configuration, ConfigurationSummary]
    paired_differences: dict[Configuration, ConfidenceInterval]
    ablation_results: list[RunResult] = Field(default_factory=list)
    ablation_summaries: dict[str, AblationSummary] = Field(default_factory=dict)
    frontier: FrontierAnalysis | None = None

    def write(self, json_path: Path, markdown_path: Path) -> None:
        """Write the full JSON report and concise Markdown summary."""
        json_path = Path(json_path)
        markdown_path = Path(markdown_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(self.model_dump_json(indent=2))
        markdown_path.write_text(self.to_markdown())

    def to_markdown(self) -> str:
        lines = [
            "# Mini-company benchmark report",
            "",
            "All failures and abstentions are retained. Resource totals include "
            "rejected cheap candidates.",
            "",
            "| Configuration | Runs | Verified | Failed | Abstained | False acceptance | False rejection | Mean score | Tokens | Calls | Wall ms | Retries | Tool ops | Cost USD |",  # noqa: E501
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for configuration in Configuration:
            summary = self.summaries[configuration]
            lines.append(
                f"| {configuration.value} | {summary.runs} | {summary.verified} | "
                f"{summary.failed} | {summary.abstained} | {summary.false_acceptance} | "
                f"{summary.false_rejection} | {summary.mean_verified_score:.3f} | "
                f"{summary.total_tokens} | {summary.calls} | {summary.wall_time_ms} | "
                f"{summary.retries} | {summary.tool_operations} | "
                f"{summary.configured_cost_usd:.6f} |"
            )
        lines.extend(["", "## Paired company minus baseline verified-score differences", ""])
        for baseline, interval in self.paired_differences.items():
            lines.append(
                f"- **{baseline.value}**: mean {interval.mean:.3f}, "
                f"95% bootstrap CI [{interval.low:.3f}, {interval.high:.3f}]"
            )
        if self.frontier is not None:
            lines.extend(["", self.frontier.to_markdown()])
        if self.ablation_summaries:
            lines.extend(
                [
                    "",
                    "## Component ablations (company minus one mechanism)",
                    "",
                    "| Ablation | Runs | Verified | Failed | Abstained | "
                    "False acc | False rej | Mean score | Tokens | Calls | "
                    "Wall ms | Retries | Tool ops | Cost USD |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for label in sorted(self.ablation_summaries):
                abl_summary = self.ablation_summaries[label]
                lines.append(
                    f"| {label} | {abl_summary.runs} | {abl_summary.verified} | "
                    f"{abl_summary.failed} | {abl_summary.abstained} | "
                    f"{abl_summary.false_acceptance} | {abl_summary.false_rejection} | "
                    f"{abl_summary.mean_verified_score:.3f} | {abl_summary.total_tokens} | "
                    f"{abl_summary.calls} | {abl_summary.wall_time_ms} | "
                    f"{abl_summary.retries} | {abl_summary.tool_operations} | "
                    f"{abl_summary.configured_cost_usd:.6f} |"
                )
        return "\n".join(lines) + "\n"


class BenchmarkStrategy(Protocol):
    """Configuration adapter evaluated by the benchmark runner.

    Async so live strategies can drive real model endpoints, workers, and
    ledgers; fixture strategies simply return without awaiting anything.
    """

    async def run(self, task: BenchmarkTask, repeat_index: int) -> RunResult: ...


def load_task_suite(path: Path) -> BenchmarkSuite:
    """Load a JSON suite without interpreting hidden checks."""
    return BenchmarkSuite.model_validate_json(Path(path).read_text())


def bootstrap_mean_ci(values: list[float], *, samples: int, seed: int) -> ConfidenceInterval:
    """Deterministic percentile bootstrap for a paired mean."""
    if not values:
        return ConfidenceInterval(mean=0.0, low=0.0, high=0.0, samples=samples)
    rng = random.Random(seed)  # noqa: S311
    n = len(values)
    boot_means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    boot_means.sort()
    low_index = max(0, int(0.025 * samples))
    high_index = min(samples - 1, int(0.975 * samples))
    return ConfidenceInterval(
        mean=sum(values) / n,
        low=boot_means[low_index],
        high=boot_means[high_index],
        samples=samples,
    )


class BenchmarkRunner:
    """Run all configurations with shared tasks and retain every outcome.

    ``ablations`` adds company-minus-one-mechanism variants: their results are
    retained in ``ablation_results`` and summarized separately, excluded from
    the primary configuration summaries and paired analysis (the protocol
    treats ablations as diagnostic, not as competing configurations).
    """

    def __init__(
        self,
        *,
        strategies: Mapping[Configuration, BenchmarkStrategy],
        config: BenchmarkConfig | None = None,
        ablations: Mapping[str, BenchmarkStrategy] | None = None,
    ) -> None:
        missing = set(Configuration) - set(strategies)
        if missing:
            raise ValueError(f"missing benchmark strategies: {sorted(c.value for c in missing)}")
        self._strategies: Mapping[Configuration, BenchmarkStrategy] = strategies
        self._config = config or BenchmarkConfig()
        self._ablations: Mapping[str, BenchmarkStrategy] = ablations or {}

    async def run(self, tasks: list[BenchmarkTask]) -> BenchmarkReport:
        results: list[RunResult] = []
        ablation_results: list[RunResult] = []
        for task in tasks:
            for repeat_index in range(self._config.repeats):
                for configuration in Configuration:
                    if configuration == Configuration.CHEAP_BEST_OF_N:
                        retained = await self._run_best_of_n(task, repeat_index)
                    else:
                        retained = self._normalize_result(
                            await self._strategies[configuration].run(task, repeat_index),
                            task,
                            configuration,
                            repeat_index,
                        )
                    results.append(retained)
                for label, strategy in self._ablations.items():
                    ablated = await strategy.run(task, repeat_index)
                    if ablated.task_id != task.id:
                        raise ValueError(
                            f"ablation {label!r} returned task {ablated.task_id!r}; "
                            f"expected {task.id!r}"
                        )
                    ablation_results.append(
                        ablated.model_copy(
                            update={
                                "configuration": Configuration.COMPANY,
                                "repeat_index": repeat_index,
                                "ablation": label,
                            }
                        )
                    )

        summaries = {
            configuration: self._summarize(configuration, results)
            for configuration in Configuration
        }
        paired = self._paired_differences(results)
        return BenchmarkReport(
            config=self._config,
            results=results,
            summaries=summaries,
            paired_differences=paired,
            ablation_results=ablation_results,
            ablation_summaries={
                label: self._summarize_ablation(label, ablation_results)
                for label in sorted(self._ablations)
            },
            frontier=analyze_frontier(summaries),
        )

    async def _run_best_of_n(self, task: BenchmarkTask, repeat_index: int) -> RunResult:
        strategy = self._strategies[Configuration.CHEAP_BEST_OF_N]
        candidates = [
            self._normalize_result(
                await strategy.run(task, candidate_index),
                task,
                Configuration.CHEAP_BEST_OF_N,
                repeat_index,
            )
            for candidate_index in range(self._config.cheap_candidates)
        ]
        selected = max(
            candidates,
            key=lambda r: (
                r.outcome == RunOutcome.VERIFIED,
                not r.false_acceptance,
                r.verified_score,
                -r.metrics.total_tokens,
            ),
        )
        total_metrics = BenchmarkMetrics()
        for candidate in candidates:
            total_metrics = total_metrics.plus(candidate.metrics)
        return selected.model_copy(update={"metrics": total_metrics, "repeat_index": repeat_index})

    def _normalize_result(
        self,
        result: RunResult,
        task: BenchmarkTask,
        configuration: Configuration,
        repeat_index: int,
    ) -> RunResult:
        if result.task_id != task.id:
            raise ValueError(f"strategy returned task {result.task_id!r}; expected {task.id!r}")
        return result.model_copy(
            update={"configuration": configuration, "repeat_index": repeat_index}
        )

    def _summarize(
        self,
        configuration: Configuration,
        results: list[RunResult],
    ) -> ConfigurationSummary:
        selected = [r for r in results if r.configuration == configuration]
        count = len(selected)
        return ConfigurationSummary(
            configuration=configuration,
            runs=count,
            verified=sum(r.outcome == RunOutcome.VERIFIED for r in selected),
            failed=sum(r.outcome == RunOutcome.FAILED for r in selected),
            abstained=sum(r.outcome == RunOutcome.ABSTAINED for r in selected),
            false_acceptance=sum(r.false_acceptance for r in selected),
            false_rejection=sum(r.false_rejection for r in selected),
            mean_verified_score=(sum(r.verified_score for r in selected) / count if count else 0.0),
            total_tokens=sum(r.metrics.total_tokens for r in selected),
            calls=sum(r.metrics.calls for r in selected),
            wall_time_ms=sum(r.metrics.wall_time_ms for r in selected),
            retries=sum(r.metrics.retries for r in selected),
            tool_operations=sum(r.metrics.tool_operations for r in selected),
            configured_cost_usd=sum(r.metrics.configured_cost_usd for r in selected),
        )

    def _summarize_ablation(
        self,
        label: str,
        results: list[RunResult],
    ) -> AblationSummary:
        selected = [r for r in results if r.ablation == label]
        count = len(selected)
        return AblationSummary(
            ablation=label,
            runs=count,
            verified=sum(r.outcome == RunOutcome.VERIFIED for r in selected),
            failed=sum(r.outcome == RunOutcome.FAILED for r in selected),
            abstained=sum(r.outcome == RunOutcome.ABSTAINED for r in selected),
            false_acceptance=sum(r.false_acceptance for r in selected),
            false_rejection=sum(r.false_rejection for r in selected),
            mean_verified_score=(sum(r.verified_score for r in selected) / count if count else 0.0),
            total_tokens=sum(r.metrics.total_tokens for r in selected),
            calls=sum(r.metrics.calls for r in selected),
            wall_time_ms=sum(r.metrics.wall_time_ms for r in selected),
            retries=sum(r.metrics.retries for r in selected),
            tool_operations=sum(r.metrics.tool_operations for r in selected),
            configured_cost_usd=sum(r.metrics.configured_cost_usd for r in selected),
        )

    def _paired_differences(
        self,
        results: list[RunResult],
    ) -> dict[Configuration, ConfidenceInterval]:
        indexed = {(r.task_id, r.repeat_index, r.configuration): r.verified_score for r in results}
        pairs = sorted({(r.task_id, r.repeat_index) for r in results})
        intervals: dict[Configuration, ConfidenceInterval] = {}
        for baseline in Configuration:
            if baseline == Configuration.COMPANY:
                continue
            differences = [
                indexed[(task_id, repeat_index, Configuration.COMPANY)]
                - indexed[(task_id, repeat_index, baseline)]
                for task_id, repeat_index in pairs
            ]
            intervals[baseline] = bootstrap_mean_ci(
                differences,
                samples=self._config.bootstrap_samples,
                seed=self._config.seed + list(Configuration).index(baseline),
            )
        return intervals


class FixtureStrategy:
    """Explicit fixture-only strategy used by the local pilot CLI."""

    def __init__(self, configuration: Configuration) -> None:
        self._configuration = configuration

    async def run(self, task: BenchmarkTask, repeat_index: int) -> RunResult:
        fixture = task.fixture_results.get(self._configuration.value)
        if fixture is None:
            return RunResult(
                task_id=task.id,
                configuration=self._configuration,
                repeat_index=repeat_index,
                outcome=RunOutcome.ABSTAINED,
                verified_score=0.0,
                error="fixture result not provided",
            )
        return RunResult(
            task_id=task.id,
            configuration=self._configuration,
            repeat_index=repeat_index,
            outcome=_RUN_OUTCOME_BY_LABEL.get(
                fixture.get("outcome", "abstained"), RunOutcome.ABSTAINED
            ),
            verified_score=float(fixture.get("verified_score", 0.0)),
            false_acceptance=bool(fixture.get("false_acceptance", False)),
            false_rejection=bool(fixture.get("false_rejection", False)),
            error=str(fixture.get("error", "")),
            metrics=BenchmarkMetrics(
                input_tokens=int(fixture.get("input_tokens", 0)),
                output_tokens=int(fixture.get("output_tokens", 0)),
                calls=int(fixture.get("calls", 1)),
                wall_time_ms=int(fixture.get("wall_time_ms", 0)),
                retries=int(fixture.get("retries", 0)),
                tool_operations=int(fixture.get("tool_operations", 0)),
                configured_cost_usd=float(fixture.get("configured_cost_usd", 0.0)),
            ),
        )


def fixture_strategies() -> dict[Configuration, BenchmarkStrategy]:
    """Create the explicit offline pilot strategy set."""
    return {configuration: FixtureStrategy(configuration) for configuration in Configuration}
