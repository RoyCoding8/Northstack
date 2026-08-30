"""CLI for northstack: config validation, ledger inspection, and replay.

Public seam: typer app with subcommands:
  - config validate <path> -- load and validate a TOML config file
  - ledger events <db_path> <run_id> -- list events for a run
  - ledger replay <db_path> <run_id> -- replay events to reconstruct RunState
  - ledger verify <db_path> <run_id> -- verify hash chain integrity
  - run --config --workspace --goal -- run a project
  - inspect --db --run-id -- inspect a run's reconstructed state
  - replay --db --run-id -- replay and verify a run's event chain
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import typer

from northstack.adapters.sqlite_ledger import Ledger
from northstack.application.replay import replay_run
from northstack.config import NorthStackConfig
from northstack.domain import Budget, ProjectRequest
from northstack.interfaces.tui import app as tui_app
from northstack.interfaces.tui import load_secrets

app = typer.Typer(help="northstack: typed event-sourced orchestrator CLI")


def _load_config(path: Path) -> NorthStackConfig:
    """Load+validate a TOML config; exit(1) with a friendly message on failure."""
    load_secrets(path)
    try:
        return NorthStackConfig.from_toml(path)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
    except (ValueError, KeyError, tomllib.TOMLDecodeError) as e:
        typer.echo(f"Invalid config: {e}", err=True)
    raise typer.Exit(code=1)


config_app = typer.Typer(help="Configuration management")
app.add_typer(config_app, name="config")

ledger_app = typer.Typer(help="Ledger inspection and replay")
app.add_typer(ledger_app, name="ledger")

app.add_typer(tui_app, name="tui")


@config_app.command("validate")
def config_validate(
    path: Path = typer.Argument(..., help="Path to TOML config file"),
) -> None:
    """Load and validate a company configuration file."""
    config = _load_config(path)
    typer.echo(f"Config valid: {config.name}")
    for p in config.profiles:
        typer.echo(
            f"  {p.name}: {p.protocol.value} {p.model} "
            f"(tier={p.tier}, conc={p.max_concurrency}, {p.key_status()})"
        )
    typer.echo(f"  Default budget: {config.run.budget_summary()}")


@ledger_app.command("events")
def ledger_events(
    db_path: Path = typer.Argument(..., help="Path to SQLite database"),
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
) -> None:
    """List events for a run."""
    with Ledger(path=db_path) as ledger:
        events = ledger.events(run_id)
        if not events:
            typer.echo(f"No events found for run: {run_id}")
            return

        typer.echo(f"Events for run {run_id}: {len(events)} total")
        for ev in events:
            payload_preview = str(ev.payload.model_dump(mode="json"))[:80]
            typer.echo(f"  [{ev.seq}] {ev.kind.value}: {payload_preview}")


@ledger_app.command("replay")
def ledger_replay(
    db_path: Path = typer.Argument(..., help="Path to SQLite database"),
    run_id: str = typer.Argument(..., help="Run ID to replay"),
) -> None:
    """Replay events to reconstruct RunState."""
    with Ledger(path=db_path) as ledger:
        state = replay_run(ledger, run_id)
        snap = state.snapshot()
        typer.echo(f"Run: {snap['run_id']}")
        typer.echo(f"Status: {snap['status']}")
        typer.echo(f"Events replayed: {snap['events_replayed']}")
        typer.echo(f"Cells: {len(snap['cells'])}")
        if snap["cells"]:
            for c in snap["cells"]:
                typer.echo(f"  - {c['id']}: {c['name']} ({c['status']})")


@ledger_app.command("verify")
def ledger_verify(
    db_path: Path = typer.Argument(..., help="Path to SQLite database"),
    run_id: str = typer.Argument(..., help="Run ID to verify"),
) -> None:
    """Verify hash chain integrity for a run."""
    with Ledger(path=db_path) as ledger:
        result = ledger.verify_integrity(run_id)
        if result.ok:
            typer.echo(f"Integrity OK: {result.events_checked} events verified")
        else:
            typer.echo(
                f"Integrity FAILED at seq {result.error_at_seq}: {result.error_message}", err=True
            )
            raise typer.Exit(code=1)


_DEFAULT_MAX_WAVES = ProjectRequest.model_fields["max_waves"].default


@app.command("run")
def run_project(
    config_path: Path = typer.Option(..., "--config", "-c", help="TOML config file"),
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace root"),
    goal: str = typer.Option(..., "--goal", "-g", help="Project goal"),
    db_path: Path | None = typer.Option(None, "--db", "-d", help="Ledger DB path"),
    token_limit: int = typer.Option(0, "--token-limit", help="Budget token limit (0=default)"),
    cost_limit: float = typer.Option(0.0, "--cost-limit", help="Budget cost limit (0=default)"),
    max_waves: int = typer.Option(0, "--max-waves", help="Wave budget (0=default)"),
) -> None:
    """Run a project through the full pipeline."""
    from northstack.application.build import CompanyComponents, build_company

    config = _load_config(config_path)

    budget = None
    if token_limit > 0 or cost_limit > 0:
        fallback = config.run.default_budget()
        budget = Budget(
            token_limit=token_limit or fallback.token_limit,
            cost_limit_usd=cost_limit or fallback.cost_limit_usd,
        )

    request = ProjectRequest(
        goal=goal,
        workspace_root=str(workspace.resolve()),
        budget=budget,
        max_waves=max_waves if max_waves > 0 else _DEFAULT_MAX_WAVES,
    )

    if not workspace.exists() or not workspace.is_dir():
        typer.echo(f"Error: workspace does not exist or is not a directory: {workspace}", err=True)
        raise typer.Exit(code=1)

    components: CompanyComponents | None = None
    try:
        components = build_company(config, workspace, db_path=db_path)
        outcome = components.company.run(request)
        typer.echo(f"Outcome: {outcome.value}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from e
    finally:
        if components is not None:
            components.close()


@app.command("benchmark")
def benchmark_suite(
    suite_path: Path = typer.Option(..., "--suite", help="JSON benchmark suite"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Report output directory"),
    config_path: Path | None = typer.Option(
        None, "--config", "-c", help="TOML config (required for --live)"
    ),
    live: bool = typer.Option(
        False, "--live", help="Run live strategies against real configured endpoints"
    ),
    ablate: bool = typer.Option(
        False,
        "--ablate",
        help="With --live: also run the component ablations (company minus one mechanism)",
    ),
    allow_heldout: bool = typer.Option(
        False,
        "--allow-heldout",
        help="Explicitly permit running a cohort='heldout' suite (frozen-set discipline)",
    ),
    repeats: int = typer.Option(1, "--repeats", min=1),
    cheap_candidates: int = typer.Option(3, "--cheap-candidates", min=1),
    bootstrap_samples: int = typer.Option(2000, "--bootstrap-samples", min=1),
    seed: int = typer.Option(1, "--seed"),
) -> None:
    """Run the fixture pilot (default) or the live benchmark suite."""
    import asyncio

    from northstack.application.benchmark import (
        BenchmarkConfig,
        BenchmarkRunner,
        fixture_strategies,
        load_task_suite,
    )

    try:
        suite = load_task_suite(suite_path)
        if suite.cohort == "heldout" and not allow_heldout:
            typer.echo(
                f"Error: suite '{suite.name}' is cohort=heldout -- the frozen set. "
                "Running it during development or tuning contaminates it. Pass "
                "--allow-heldout only after the protocol is frozen.",
                err=True,
            )
            raise typer.Exit(code=1)
        ablations = None
        if live:
            if config_path is None:
                typer.echo(
                    "Error: --live requires --config pointing at a northstack TOML config",
                    err=True,
                )
                raise typer.Exit(code=1)
            config = _load_config(config_path)
            from northstack.application.benchmark_live import (
                ablation_strategies,
                live_strategies,
            )

            strategies = live_strategies(
                config, runs_dir=output_dir / "runs", suite_dir=suite_path.parent
            )
            if ablate:
                ablations = ablation_strategies(
                    config, runs_dir=output_dir / "runs" / "ablations", suite_dir=suite_path.parent
                )
            typer.echo(
                "WARNING: live benchmark calls every configured provider endpoint "
                "and consumes real budget under each task's ceilings."
            )
        else:
            strategies = fixture_strategies()

        report = asyncio.run(
            BenchmarkRunner(
                strategies=strategies,
                config=BenchmarkConfig(
                    repeats=repeats,
                    cheap_candidates=cheap_candidates,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed,
                ),
                ablations=ablations,
            ).run(suite.tasks)
        )
        json_path = output_dir / "benchmark-report.json"
        markdown_path = output_dir / "benchmark-report.md"
        report.write(json_path, markdown_path)
        typer.echo(f"Benchmark: {suite.name} ({'live' if live else 'fixture'})")
        typer.echo(f"Runs retained: {len(report.results)}")
        if report.ablation_results:
            typer.echo(f"Ablation runs retained: {len(report.ablation_results)}")
        frontier_members = [
            e.configuration.value
            for e in (report.frontier.entries if report.frontier else [])
            if e.pareto_efficient
        ]
        if frontier_members:
            typer.echo(f"Pareto frontier: {', '.join(frontier_members)}")
        typer.echo(f"JSON: {json_path}")
        typer.echo(f"Markdown: {markdown_path}")
    except (OSError, ValueError) as e:
        typer.echo(f"Benchmark error: {e}", err=True)
        raise typer.Exit(code=1) from e


@app.command("calibrate")
def calibrate_reviewers(
    config_path: Path = typer.Option(..., "--config", "-c", help="TOML config file"),
    samples_path: Path = typer.Option(
        ..., "--samples", help="JSON file of labeled calibration samples"
    ),
    output_dir: Path = typer.Option(..., "--output-dir", help="Where to write the report"),
) -> None:
    """Measure blinded-reviewer agreement/accuracy; emit CalibrationRecords.

    Calls every reviewer-role endpoint once per sample (real budget). Point
    ``[northstack.run] calibration_path`` at the emitted calibration.json to
    feed the records into future runs' soft-review panel.
    """
    import asyncio
    import json

    from northstack.adapters.providers.gateway import ModelGateway
    from northstack.application.calibration import (
        CalibrationRunner,
        load_calibration_suite,
    )
    from northstack.config import Role

    config = _load_config(config_path)
    reviewer_chain = [
        name
        for name in config.role_map().get(Role.REVIEWER, [])
        if name in {p.name for p in config.profiles}
    ][:2]
    if len(reviewer_chain) < 2:
        typer.echo(
            "Error: calibration needs two routed reviewer-role profiles "
            "(the blinded panel minimum); configure [northstack.routing] "
            "role='reviewer' with two profiles.",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        suite = load_calibration_suite(samples_path)
        typer.echo(
            f"WARNING: calibrate calls each reviewer endpoint once per sample "
            f"({len(suite.samples)} samples x {len(reviewer_chain)} reviewers)."
        )

        async def _measure():
            gateway = ModelGateway(config)
            try:
                runner = CalibrationRunner(gateway, reviewer_chain)
                return await runner.measure(suite)
            finally:
                await gateway.close()

        report = asyncio.run(_measure())
    except (OSError, ValueError) as e:
        typer.echo(f"Calibration error: {e}", err=True)
        raise typer.Exit(code=1) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calibration.json"
    md_path = output_dir / "calibration-report.md"
    json_path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    typer.echo(f"Suite: {report.suite_name}")
    for measurement in report.per_criterion:
        typer.echo(
            f"  criterion {measurement.criterion_index}: "
            f"agreement={measurement.agreement_rate:.3f} "
            f"accuracy={measurement.majority_accuracy:.3f} "
            f"suggested_threshold={measurement.suggested_threshold:.3f}"
        )
    typer.echo(f"JSON: {json_path}")
    typer.echo(f"Markdown: {md_path}")


@app.command("inspect")
def inspect_run(
    db_path: Path = typer.Option(..., "--db", "-d", help="Ledger DB path"),
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to inspect"),
) -> None:
    """Inspect a run's reconstructed state from the event ledger."""
    with Ledger(path=db_path) as ledger:
        state = replay_run(ledger, run_id)
        snap = state.snapshot()
        typer.echo(json.dumps(snap, indent=2, default=str))


@app.command("replay")
def replay_command(
    db_path: Path = typer.Option(..., "--db", "-d", help="Ledger DB path"),
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to replay"),
) -> None:
    """Replay a run's events and verify integrity."""
    with Ledger(path=db_path) as ledger:
        integrity = ledger.verify_integrity(run_id)
        if not integrity.ok:
            typer.echo(
                f"Integrity FAILED at seq {integrity.error_at_seq}: {integrity.error_message}",
                err=True,
            )
            raise typer.Exit(code=1)

        state = replay_run(ledger, run_id)
        snap = state.snapshot()
        typer.echo(f"Run: {snap['run_id']}")
        typer.echo(f"Status: {snap['status']}")
        typer.echo(f"Events replayed: {snap['events_replayed']}")
        typer.echo(f"Contract version: {snap.get('contract_version', 0)}")
        typer.echo(f"Graph version: {snap.get('graph_version', 0)}")
        typer.echo(f"Outcome: {snap.get('outcome', 'none')}")
        typer.echo(f"Integrity: OK ({integrity.events_checked} events verified)")


if __name__ == "__main__":
    app()
