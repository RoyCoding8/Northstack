# ADR 0009: Propose-Harden-Fallback for Model-Backed Planning and Falsification

Accepted.
## Context

Three seams invite model judgment into control-plane decisions:

1. Graph planning — the single-cell planner never decomposes; rolling-wave multi-cell graphs existed as machinery (validation, scheduling, wave laws) with no producer.
2. Contract falsification — the falsifier seam existed with only a deterministic no-op implementation.
3. Reviewer calibration — the soft-review law requires measured `CalibrationRecord`s, but nothing could measure them.

The shared question: how does a stochastic, unreliable proposer participate in a deterministic control plane whose entire value proposition is that code owns contracts, budgets, state, and outcomes?

## Decision

One law for all three seams: propose → harden → fallback.

The model proposes (a decomposition, a counter-interpretation, a verdict). Deterministic code hardens the proposal into legal domain objects, owning every structural fact. When the proposal cannot be hardened, the system falls back to the deterministic default — never to an exception, never to a half-trusted proposal.

### ModelBackedPlanner (`planner_mode = "model"`)

The PLANNER-role profile proposes cells (name, mode, dependencies, criterion links, budget share). The hardener owns all structure:

- Cell ids are rewritten deterministically; dependencies resolve only to earlier cells. A forward reference is dropped, making cycles unconstructible rather than detected.
- Waves are recomputed as topological levels — the model's ordering beliefs are never consulted.
- At most one mutating cell per wave is enforced by bumping surplus mutating cells into fresh waves, bounded by a max-wave cap.
- Budget shares are normalized so the per-cell budgets can never sum above the contract's budget; the per-cell budget is copied wholesale so the `max_retries` single-owner invariant (ADR 0004) is untouched.
- Criterion links are validated in-range and must cover every criterion.
- Every cell is stamped `worker`-role, keeping routing authoritative.
- The finished graph must pass the production `GraphPlanner.validate()`.

Any failure — gateway error, unparseable JSON, schema violation, illegal graph, uncovered criterion, wave explosion — logs a reason and returns the canonical single-cell graph. Decomposition is an optimization, not a safety property: a run must never die because a planner endpoint hiccuped. The fallback is auditable: the accepted graph in the ledger has one cell.

### ModelBackedFalsifier (`falsifier_mode = "model"`)

The SPECIALIST-role profile hunts a passing-but-wrong interpretation of the contract. A concrete finding is actionable: compilation raises and the run fails loudly rather than proceeding on a misread contract. "none"/empty means sound.

Fail-open on outage: a falsifier that cannot answer returns no objection — an outage is indistinguishable from silence and is treated as such. The asymmetry is deliberate: a finding costs one run; a dead falsifier must not cost every run. Hard gates and the release law remain the safety properties.

### Calibration (`northstack calibrate`)

Reviewer-panel agreement and accuracy are measured on labeled samples through the production reviewer path (`ModelBackedReviewer`), producing per-criterion `CalibrationRecord`s. Setting `[northstack.run] calibration_path` loads them into the panel; a malformed file is a hard error — the operator asked for calibration, and silently running uncalibrated would violate the release law's honesty. The suggested `agreement_threshold` is `max(0.5, 1 - false-acceptance-rate)`: a panel that falsely accepts 10% of bad evidence must then agree on ≥90% of what it accepts. It is a floor for the operator to inspect, not a law.

## Why

- Authority stays where ADRs 0001–0005 put it: the model's contribution is always reducible to a validated domain object or discarded.
- Both failure directions are honest. Planning fails closed to the default (the single cell always runs); falsification fails open (no objection) — and each direction is the one whose silent failure is cheap.
- The seams are async (mirroring ADR 0006's review seam) and injectable, so every behavior is hermetically testable without endpoints.

## Consequences

- `GraphPlanner.plan` and `Falsifier.check` became async; the orchestrator awaits both. Test fakes updated accordingly.
- `RunConfig` grows `planner_mode`, `falsifier_mode`, `calibration_path`; the TOML writer emits them only when non-default.
- Modes without a routed matching role chain degrade with a logged warning (single-cell planner, falsifier off) rather than failing configuration.
- Split/replan execution (appending a new `GraphVersion` mid-run after a decomposition failure) remains future work; this ADR covers initial decomposition only.
