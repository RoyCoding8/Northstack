# Experiment protocol

## Research question

Does the full company control plane improve the verified quality–resource frontier over simpler endpoint-backed agent configurations?

A positive result requires either:

- higher verified quality under an equal resource ceiling, or
- statistically comparable verified quality with lower provider-neutral resource use.

A successful demonstration alone is not evidence of superiority.

## Preregistered configurations

Each task is run under the same raw request, clean workspace snapshot, tool policy, hidden checks, token ceiling, configured cost ceiling, and stopping rule.

1. **Strong single** — strongest eligible configured model, one end-to-end worker.
2. **Cheap best-of-N** — N independent cheap candidates; the same verification law selects the result. Resource accounting includes all candidates.
3. **Singleton expert** — expert profile alone, structurally single-flight.
4. **Company** — contract analyses, synthesis seam, graph, routing, cells, verification, and bounded recovery.

Component ablations should additionally fix the contract and remove one mechanism at a time: routing, independent analysis, falsification, rolling graph, or recovery.

## Task suites

The included `benchmarks/pilot.json` is an offline fixture-mode operability test. Its numbers are illustrative inputs, not measured model performance.

The included `benchmarks/live-pilot.json` is the live **development** pilot: eight reproducible local repository tasks (two bug fixes, two feature additions, two behavior-preserving refactors, two documentation/configuration tasks) under `benchmarks/tasks/`, each with stdlib-`unittest` hidden checks so no snapshot needs extra tooling. Its integrity is CI-gated: pristine templates must be unsolved and intended solutions must score 1.0. Reports include the Pareto frontier over all resource axes, and `--ablate` runs the five component ablations (routing, model planning, model intake, recovery, falsification), retained separately from the primary comparison (ADR 0010).

Suites carry a `cohort` field (`dev` or `heldout`). A heldout suite refuses to run without an explicit `--allow-heldout` — the frozen set is mechanically protected from tuning-phase contamination.

A larger study should grow toward 20–30 tasks, with the development cohort for tuning and a held-out cohort used only after freezing the protocol.

Each task records:

- immutable clean workspace source and digest;
- raw request and constraints;
- hidden executable checks;
- task category;
- token, call, wall-time, retry, and configured-cost ceilings;
- labels needed to identify false acceptance and false rejection.

## Primary outcomes

For every retained run, report:

- final outcome: verified, failed, or abstained;
- verified score under hidden checks;
- false acceptance and false rejection where ground truth exists;
- input and output tokens;
- calls by model profile;
- wall time;
- retries;
- tool operations;
- configured monetary cost.

Free endpoints still consume tokens, calls, RPM, and latency. USD is not the sole resource axis.

## Release and scoring law

A run scores as verified only when all hard gates pass. A model or integrator cannot override command, file, schema, policy, provenance, safety, or budget failures.

Non-executable criteria require blinded reviewers with preregistered rubrics and measured calibration. Missing calibration or material disagreement produces abstention.

All failures and abstentions are retained in the report. No run may be dropped because it encountered provider errors, exhausted budget, or produced an inconvenient trace.

## Repetition and randomness

Before held-out evaluation, freeze:

- model profile configuration and endpoint versions where discoverable;
- prompts and schemas;
- N for cheap best-of-N;
- seeds where providers support them;
- retry and stopping rules;
- task order or randomized order seed;
- bootstrap seed and sample count;
- exclusion rules, ideally none beyond corrupted infrastructure runs defined in advance.

Repeat stochastic configurations enough times to estimate within-task variance. The initial runner defaults to one repetition for local fixture checks; this is insufficient for scientific claims.

## Paired analysis

The primary comparison is paired by task and repeat. For each baseline, compute:

`company verified score - baseline verified score`

Report the observed mean and a seeded percentile bootstrap confidence interval over paired task/repeat differences. Also report raw per-task results so the interval does not hide heterogeneous failure modes.

The implementation's bootstrap is intentionally simple and auditable. A publication-grade study should add sensitivity analyses such as task-stratified or hierarchical resampling when the suite is large enough.

## Frontier analysis

Do not collapse all resource use into one arbitrary scalar. Present quality against at least:

- total tokens;
- total calls;
- wall time;
- retries;
- configured cost.

A configuration dominates another only when quality is no worse and one or more resource dimensions are strictly better, or resource use is no worse and verified quality is strictly better. Report non-dominated configurations rather than declaring a winner from one metric.

## Calibration and false acceptance

Reviewer calibration data must be separate from evaluation tasks. Record sample count, agreement threshold, and labeled accuracy by rubric or task family.

False acceptance is especially costly: the system emitted `verified` but hidden or authoritative checks show the work was wrong. Report it separately and never merge it into ordinary failure.

## Operational checks before a live pilot

1. Run the full non-live test suite and formatting/lint checks.
2. Validate the target configuration and environment-variable references.
3. Run opt-in endpoint smoke tests.
4. Verify observed call intervals show no overlap for singleton expert profiles.
5. Execute each task once in a disposable copy to validate hidden checks.
6. Freeze config, task digests, prompts, N, seeds, and stopping rules.
7. Run all four configurations under equal ceilings.
8. Preserve ledgers, artifacts, diffs, failures, and abstentions.
9. Generate JSON and Markdown reports.
10. Review traces before making any causal or superiority claim.

## Current limitations

The vertical slice now provides model-backed contract analysis, model-backed blinded reviewers (async, evidence-aware, fail-closed), live benchmark strategies with hidden-check scoring, Docker isolation for command profiles (fail-closed), model-proposed graph planning and falsification under the propose-harden-fallback law, reviewer calibration measurement, component ablations, Pareto frontier reporting, and held-out cohort enforcement. It does not yet provide:

- split/replan and contract-amendment execution (a decomposition failure falls back to the single-cell plan; it does not yet re-plan mid-run);
- cross-process distributed rate limiting;
- a measured held-out benchmark (the live pilot is a development suite; its numbers must not be presented as held-out evidence);
- Docker isolation for the live benchmark's hidden checks (they run on the host; they are operator-authored, not model-authored).

These limitations must appear alongside any pilot result.

## Baseline operationalization

The baselines are bare worker loops: one `NativeWorker` pinned to a single profile, the same mediated tool policy as the company configuration, a single-cell contract carrying only the raw request and ceilings, and bounded loop limits. Their claimed outcome is worker completion; the token ceiling is enforced in-loop and the cost ceiling post-hoc (a bare loop has no wave boundary, so a baseline may overshoot the cost ceiling by at most one worker turn — an asymmetry in the baseline's favor, documented here). Best-of-N selects by hidden-check score, an oracle-based selector that strengthens the baseline beyond any deployable verifier. See [ADR 0007](adr/0007-live-benchmark-law.md).

