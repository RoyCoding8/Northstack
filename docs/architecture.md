# Architecture

## Design objective

`northstack` is a control-plane experiment for autonomous project delivery. Its hypothesis is not that company language helps models. Its hypothesis is that explicit contracts, task-dependent decomposition, selective model routing, independent evidence, hard verification, and bounded recovery may improve verified output per unit resource.

The control plane, not an LLM, is authoritative.

## Package layout and enforced layering

The package is layered, and the direction is machine-checked rather than documented. `lint-imports` runs in CI; a dependency pointing upwards fails the build.

```
interfaces -> application -> ports -> adapters -> events -> domain
```

| Layer | Holds | May import |
|---|---|---|
| `domain/` | `RunStatus` and its transition table, `Budget`/`BudgetUsage`/`RemainingBudget`, `WorkContract`, the graph (`GraphCell`, `GraphVersion`, `CellMode`, `CellStatus`), outcomes and evidence, `RunState` | nothing in `northstack` except itself |
| `events/` | `EventKind`, `EventEnvelope`, the projection that folds events into `RunState` | `domain` |
| `adapters/` | `sqlite_ledger`, `artifacts`, `providers/` (gateway, wire format, pricing, limiter), `workspace/` (restricted filesystem, commands, web fetch), `atomic_io`, `config_toml` | `events`, `domain` |
| `ports/` | Protocols the application talks through: `EventSink`, `ArtifactSink`, `GatewayPort`, `WorkspacePort`, `WorkerPort`, `Clock`, `Sleeper` | `adapters`, `events`, `domain` |
| `application/` | `orchestrator` (phase sequencing), `routing`, `planning`, `scheduling`, `worker`, `contracting`, `recovery`, `replay`, `verification/`, `cell_runner` (sole retry owner), `budget_authority` (sole spend owner), `release_law` (sole `RunOutcome` authority), `retry` (recovery policy + dedup) | everything below |
| `interfaces/` | `cli`, `tui`, `web/` (FastAPI control surface and the SPA) | everything below |

Two invariants this encodes, both of which had been violated:

- Storage does not depend on a read model. The SQLite ledger yields envelopes; folding them into a `RunState` lives in `application/replay.py`.
- Cost accounting is public. `adapters/providers/pricing.py` exposes `compute_cost_usd`, so the worker no longer imports a private symbol from the gateway.

`config.py` is cross-cutting and is contractually forbidden from importing any layer above the domain.

## Exclusions

The architecture deliberately excludes:

- persistent employee or department personas;
- salaries, morale, careers, politics, or office simulation;
- simulated customers as evidence of product value;
- claims that LLM behavior is a valid proxy for human organizations;
- a manager model that can waive hard tests or policy checks.

## Run state machine

A run follows:

`intake -> contracted -> planned -> executing -> verifying -> verified | abstained | failed`

Models may propose contracts, plans, tool calls, evidence, reviews, or recovery actions. Only `Company` emits authoritative state and outcome transitions. Terminal states cannot transition further.

## Executable contract

`WorkContract` is immutable and contains the objective, scope, deliverables, constraints, assumptions, forbidden outcomes, workspace scope, allowed tools, budgets, typed acceptance criteria, unresolved ambiguity, and abstention threshold.

Intake fans out requirements, repository, and acceptance/risk analyses; the repository analysis is fed by a deterministic, bounded workspace scan whose summary is digest-recorded. Contract falsification is opt-in and model-backed (`falsifier_mode = "model"`): a SPECIALIST-role profile hunts a reading of the contract that would pass its own gates while being wrong — a finding rejects the contract at compile time, an outage fails open (ADR 0009).

Acceptance criteria are compiled before execution:

- `command`: named exact-argv command and expected exit code;
- `file_diff`: file presence or absence and optional digest;
- `schema`: runtime artifact checked against JSON Schema;
- `policy`: authoritative tool audit checked against a supported policy;
- `soft_rubric`: blinded calibrated review after hard gates pass.

Unsupported hard policy checks fail closed.

## Dynamic task cells and graph

Cells are data, not personalities. A `GraphCell` carries capability requirements, model-role requirements, tool and mutation mode, dependencies, input/output schemas, budget-bearing contract, and acceptance links.

The graph is immutable. The scheduler tracks execution progress separately, runs ready read-only cells concurrently, and permits at most one mutating cell in a wave. Graph validation rejects cycles, duplicate IDs, excessive aggregate budgets, and concurrent mutating cells in one wave.

The default planner creates a single deep cell. With `planner_mode = "model"`, a PLANNER-role profile proposes a decomposition under the propose-harden-fallback law (ADR 0009): the hardener rewrites ids, recomputes waves topologically, enforces one mutating cell per wave, clamps budget shares to the contract, and requires full criterion coverage; any failure falls back to the canonical single-cell graph. Future replanning must append a new `GraphVersion`; it must not silently rewrite an accepted graph.

## Model gateway and routing

A profile configures an arbitrary compatible endpoint:

- OpenAI Chat Completions or Anthropic Messages protocol;
- base URL, model, and optional API-key environment reference;
- roles and capabilities;
- concurrency and requests/minute;
- context/output limits and configured prices.

One process uses one `ModelGateway`. The gateway owns shared per-profile semaphores and RPM windows, so a singleton expert profile is structurally single-flight. Separate processes do not yet share a distributed limiter.

The inspectable router filters capability and role mismatches, then scores eligible profiles using cell mutation mode, tier, concurrency, and remaining budget. Capability recovery excludes the failed profile and reroutes to another eligible profile. No eligible profile produces abstention.

## Native worker

`NativeWorker` implements a bounded model/tool loop. It receives an explicit router-selected profile name, exposes only contract-allowed tool definitions, validates tool arguments, acquires a mutation lease before mutation, validates structured output with JSON Schema, permits one bounded repair request, and accounts for calls, tokens, wall time, and configured cost.

The worker cannot declare the project verified. Its final text and tool trail are evidence inputs only.

## Restricted workspace

The native executor restricts operations to a configured root. It rejects absolute paths, traversal, symlinks, junctions, and Windows reparse points. Writes are atomic and require a mutation lease. Reads, listings, searches, and outputs are bounded.

Named commands use exact argv arrays, `shell=False`, a workspace cwd, timeout and output limits, and a validated environment allowlist that rejects names associated with credentials. A command may opt into Docker isolation (`isolation = "docker"`): the argv runs in a throwaway `--network none` container with the workspace bind-mounted at `/workspace`, and an unavailable Docker fails closed — the command is refused, never silently run on the host (ADR 0008).

This is not a security sandbox. Executed repository code has the host process's authority. Docker remains the intended isolation seam.

## Verification and release law

Hard gates run real workspace operations and cannot be waived:

1. command checks inspect actual exit codes;
2. file checks inspect actual workspace content;
3. schema checks read content-addressed runtime artifacts;
4. policy checks use the authoritative tool-call audit.

Only after hard gates pass may soft rubrics run. Soft review requires at least two blinded reviewers and criterion-specific calibration. Missing calibration or material disagreement causes abstention. The review seam is async and evidence-aware: reviewers judge resolved artifact content, blinding is structural (executor identity does not exist in the reviewer's inputs), and every reviewer failure fails closed to disagreement (ADR 0006). Calibration is measured, not assumed: `northstack calibrate` scores each routed reviewer on labeled samples, and the emitted `CalibrationRecord`s feed the panel's agreement thresholds via `[northstack.run] calibration_path`; a malformed calibration file is a hard error, never a silently uncalibrated panel.

The final outcome is `verified`, `abstained`, or `failed` and is appended with evidence and resource usage.

## Recovery

Failures are typed as transient, sampling, capability, decomposition, specification, integration, safety, or budget. Exact `AttemptSignature` values prevent repeating an identical strategy.

Implemented recovery includes bounded retry with a new strategy identifier and capability rerouting to a different profile. Safety terminates. Actions that require missing machinery—such as split/replan or contract amendment—produce abstention or failure rather than pretending recovery occurred.

## Ledger and artifacts

SQLite WAL is the authoritative event ledger. A serialized `BEGIN IMMEDIATE` append allocates a monotonic sequence per run. Each event hashes the exact redacted persisted representation and links to the previous event hash. Integrity verification recomputes the full chain.

Large content is stored by SHA-256 in the artifact store. `RunState` is reconstructed by audit replay from events. Audit replay is distinct from fixture replay and from stochastic provider re-execution.

## Benchmark architecture

`BenchmarkRunner` accepts injected end-to-end strategies for:

- `strong_single`;
- `cheap_best_of_n`;
- `singleton_expert`;
- `company`.

Every configuration receives the same `BenchmarkTask`, including request, hidden checks, and resource ceiling. The result schema retains failures, abstentions, false acceptance/rejection labels, tokens, calls, wall time, retries, tool operations, and configured cost.

Best-of-N accounts for every candidate's resources even though only the best verified candidate is retained as the configuration outcome. Reports include paired company-minus-baseline verified-score differences and deterministic seeded percentile bootstrap confidence intervals.

Live strategies (`application/benchmark_live.py`) run the four configurations for real: every run starts from a clean copy of the task's immutable workspace template (digest recorded), executes under the shared ceilings and tool policy, and is scored post-hoc by executable hidden checks the system under test never sees. The retained outcome follows the hidden checks; the system's own claim is audited as false acceptance/false rejection (ADR 0007).

Reports carry a Pareto frontier over quality versus every resource axis under the protocol's dominance rule — frontier membership only, never a declared winner. `--ablate` additionally runs five single-mechanism company ablations (no routing, single cell, deterministic intake, minimal recovery, no falsification), retained and summarized separately from the primary comparison. Suites tagged `cohort = "heldout"` are the frozen set: the CLI refuses to run them without an explicit `--allow-heldout` (ADR 0010).

