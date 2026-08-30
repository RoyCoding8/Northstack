# ADR 0010: Component Ablations, Pareto Frontier, Held-Out Discipline

Accepted.
## Context

The experiment protocol demands three things the runner did not yet do:

1. Ablations — "fix the contract and remove one mechanism at a time: routing, independent analysis, falsification, rolling graph, or recovery." Without them, a company win says nothing about which mechanism earned it.
2. Frontier analysis — "Do not collapse all resource use into one arbitrary scalar... Report non-dominated configurations rather than declaring a winner from one metric."
3. Held-out discipline — "a development suite for tuning and a held-out suite used only after freezing the protocol." Nothing enforced the separation; an accidental run during tuning would silently contaminate the frozen set.

## Decision

### Ablations are company variants, retained separately

`BenchmarkRunner` accepts an optional `ablations` mapping of label → strategy. Each ablation is a full `LiveCompanyStrategy` with exactly one mechanism removed — a modified config copy (the base config is never mutated):

| Label | Removed mechanism | How |
|---|---|---|
| `no_routing` | operator routing chains | routing table stripped; Router falls back to tier/price scoring |
| `single_cell` | model-proposed decomposition | `planner_mode` forced to `single` |
| `deterministic_intake` | model-backed analyses | `DeterministicAnalysisRunner` injected via `build_company`'s override seam |
| `minimal_recovery` | bounded recovery | per-cell retry cap pinned to 1 — the smallest legal cap, because `max_retries = 0` means unlimited; labelled honestly rather than "no recovery" |
| `no_falsifier` | model falsification | `falsifier_mode` forced `off` (diagnostic when the base runs it) |

Ablation results carry `ablation=<label>`, land in `ablation_results`, are summarized in `ablation_summaries`, render in their own report section — and are excluded from the primary configuration summaries, the paired bootstrap analysis, and the frontier: ablations are diagnostics, not competing configurations. `--ablate` on the CLI opts in (live mode).

### The frontier states dominance, never a winner

`analyze_frontier` applies the protocol's rule exactly: A dominates B when A's mean verified score is no worse AND every resource axis (tokens, calls, wall time, retries, tool ops, configured cost) is no worse AND at least one is strictly better. Each configuration's report row shows its quality, every resource axis, and who dominates it (`— (frontier)` when nobody does). Frontier membership is the claim the data supports. The CLI prints the frontier members; the Markdown carries the table.

### Held-out suites refuse to run

`BenchmarkSuite.cohort` is `"dev"` (default) or `"heldout"`. The benchmark CLI exits with an explicit error when asked to run a heldout suite without `--allow-heldout`, so tuning-phase contamination requires a typed, deliberate override rather than a stray command.

### The dev pilot grows to eight tasks

Four new hermetic tasks round out the protocol's 4–8 pilot range across all four categories: `fix-two-bugs` (two independent defects), `add-validation` (exception contract), `extract-helper` (behavior-preserving refactor that starts green), and `write-config` (structured JSON deliverable verified by an executable command check, not substring matching). All are held to the same integrity gate: pristine templates unsolved, intended solutions score 1.0, templates immutable.

## Why

- Ablations reuse the company strategy and the composition root's override seams, so a variant differs from production by exactly one knob — the ablation is the claim.
- The frontier's strict dominance rule is conservative in both directions: it cannot crown a configuration that is worse on any axis, and it never hides which axes trade off against each other.
- The held-out guard converts a discipline rule into a mechanical refusal — the same philosophy as fail-closed Docker (ADR 0008) and unsupported hard-policy checks.

## Consequences

- `RunResult` grows `ablation`, and the report grows `ablation_results`, `ablation_summaries`, and `frontier` — all default-valued, so previously emitted report JSON still loads.
- `minimal_recovery` pins the cap at 1 (two attempts); a true single-attempt mode is not expressible under the current budget semantics and would require changing `max_retries = 0`'s meaning — not worth breaking ADR 0004's invariant for.
- `build_company` gains an `analysis_runner` override, also usable by future ablations of other mechanisms.
