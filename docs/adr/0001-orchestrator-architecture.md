# ADR 0001: Orchestrator Architecture — Decisions to Preserve

Accepted, verified against the source (`orchestration.py`, `ledger.py`, `models.py`, `worker.py`, `recovery.py`, `providers.py`). This record exists so a future edit that doesn't know *why* the orchestrator is shaped this way doesn't undo the parts that are load-bearing.

## Context

NorthStack's orchestrator (`Company` in `orchestration.py`) drives a run through plan → execute → verify → recover, persisted via a hash-chain event-sourced `Ledger` (`ledger.py`) projected into a `RunState` (`models.py`). The worker (`worker.py`) runs a bounded model/tool loop per cell. Recovery (`recovery.py`) classifies failures and picks an action from the `RECOVERY_POLICY` ladders.

## Decisions to preserve (do not change without re-verifying)

### 1. Hash-chain event sourcing, not checkpoints

The ledger is the single source of truth; `RunState` is a projection of replayed events, never a mutable store. `CELL_COMPLETED` emits only on success.

Event-sourced audit logs beat checkpoints: you can reconstruct any past state, not just the last write. The hash chain (`Ledger.append_next` → `application/replay.py` → `events/projection.py:fold`) is the architecturally stronger choice, and it isn't up for trade. (The projection used to be a `Ledger._apply_event` method; it now lives in the `events` package as `fold`, driven by `replay_run` — see ADR 0002.)

A crashed attempt used to re-run the whole cell. `Worker.run` now exposes a `resume_from_messages` seam: a crashed attempt resumes from the last succeeded tool round without re-running it. The conversation history crosses the retry in memory only — it is never duplicated into the ledger, which stays the source of truth for committed state. In-flight conversation is a transient, not a durable fact.

### 2. Per-failure-type recovery ladders

`RECOVERY_POLICY` maps each of 8 `FailureType`s to an ordered ladder of `RecoveryAction`s. Escalation walks the ladder: the Nth time the same failure signature appears, the policy returns rung `min(N, len-1)`. The ten actions are `BACKOFF_RETRY`, `CHANGED_STRATEGY_RETRY`, `REROUTE_ESCALATE`, `SPLIT_REPLAN`, `CONTRACT_AMENDMENT`, `ABSTAIN`, `INTEGRATION_DIAGNOSIS`, `TERMINATE`, `SCOPE_REDUCTION`, and `FAIL`. `SAFETY` and `BUDGET` are terminal ladders — no retry, no circumvention.

This granularity is the contribution. OpenAI ships two recovery modes; Magentic-One has one (full reset). Escalate-the-profile, amend-the-contract, split-the-cell, shrink-the-scope is finer than either, and the taxonomy should not be collapsed. The map is per-`FailureType`; the four named strategic actions are the interesting rungs, not the whole table. `CellRunner`, not the worker or the orchestrator, owns retry and reroute (ADR 0004).

Recovery decisions used to be emitted per cell but never summarized per run, which made them hard to audit. The `RECOVERY_TRANSITION` → `RunState.recovery_events` projection is lossless — each event carries `{cell_id, failure_type, action, attempt_number}` — so an operator can audit which attempt on which cell fired which action. Pure observability; the policy's behavior didn't change.

### 3. `EXECUTING → ABSTAINED` is illegal

`RunStatus._TRANSITIONS` forbids it. Budget exhaustion goes `EXECUTING → VERIFYING → ABSTAINED`.

Forcing budget exhaustion through `VERIFYING` keeps the verify-then-decide gate on every terminal path: a run can't silently abstain without the verifier having had a say. This is load-bearing for the shrink-scope-rather-than-hard-fail contract (`SCOPE_REDUCTION` is a terminal ABSTAIN-class action reached via `VERIFYING`). Don't relax this transition.

### 4. Spend has one owner; routing stays out of accounting

`BudgetAuthority` is the single owner of spend: it reserves an estimate, commits the actual, and answers `remaining()` as a snapshot. `Router.route` reads `remaining_budget` read-only and is kept out of the accounting path. Routing and budget accounting must not co-locate — the thing that self-validates spend is `BudgetAuthority`, not the router.

No SDK tracks spent-vs-remaining natively (LiteLLM included); budget enforcement is universally the operator's job. Centralising it in one authority means it cannot disagree with itself. Details in ADR 0004.

### 5. Cache-aware cost accounting uses disjoint buckets

`_compute_cost_usd` bills `input_tokens` (non-cached) at 1.0x, `cache_creation_tokens` at 1.25x, `cache_read_tokens` at 0.1x of the input price. Adapters normalize to this contract — OpenAI's `prompt_tokens` is the *total* input inclusive of cached tokens, so the adapter subtracts `cached_tokens` out.

Cache reads are cheaper input; on providers that bill cache creation separately, the write is charged at a premium. The disjoint-bucket formula avoids double-counting the cache. Per-profile cache pricing is deferred (see open risks).

## Open risks (deferred, flagged — not hidden)

- Stall detection: there is no detection for a run that is alive but not progressing (a worker looping without producing evidence). Needs a "what does stall mean here" design decision before a test can pin it.
- State leakage into the verifier: not yet confirmed what the verifier consumes from worker partial output. A wrong-scoped fix is worse than the deferral.
- Per-profile cache pricing: the fixed 1.25x/0.1x rates approximate the economics of providers that bill a separate cache-creation bucket. Gemini bills cache reads only, so the creation bucket stays empty there. A `cache_creation_rate`/`cache_read_rate` on `ModelProfile` is the follow-up when a profile's cache economics diverge enough to matter.
- Resume accounting: on a resumed `worker.run`, `WorkerResult.tool_calls_made`/`tool_rounds` count only this run's work, not the resumed prior rounds. Token accumulation adds each attempt once (no double-count), but tool-call totals undercount the cell's real activity. A reporting-accuracy gap, not a correctness bug.

## Do not cite

FrugalGPT's "98% savings" is a cherry-picked best benchmark; the realistic range is 50–73%. Never present 98% as an expected saving.
