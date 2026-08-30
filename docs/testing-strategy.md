# Testing strategy

Where the suite stands and which directions still have unexplored fault-detection.

## Where we are

840+ tests, ~19.8k test LoC against ~16.6k src LoC, 88% line coverage, `fail_under = 85` as a regression floor. ruff, mypy, import-linter and pip-audit in CI. hypothesis is already a dependency.

Coverage is not the constraint. Two things learned while closing the last gaps:

1. **Line coverage hid the real holes.** Decoding `.coverage` and mapping executed lines onto the AST found 84 functions no test executed even once, while the file-level numbers looked healthy. That metric — never-executed functions, excluding `Protocol` stubs — is worth keeping.
2. **Green does not mean proving.** Of 65 tests written in the last five lanes, 5 passed for the wrong reason: a secret-leak test that asserted an invented string was absent, an aliasing test whose expected value mutated in lockstep with the thing under test, a yield-ordering test that held whether or not the coroutine yielded, and a provider error test that never touched the provider path. All four were in the security- or invariant-critical set.

Point 2 is the reason mutation testing leads the list below. Nothing else measures whether a test can fail.

## Tier 1 — highest fault-detection per unit of effort

### 1. Mutation testing

**Catches:** tests that cannot fail. Exactly the class of defect found manually above, but automatically and across the whole suite.

**Why here:** we have direct evidence the suite contains hollow assertions, found only by reading them one at a time. That does not scale.

**First action:** `mutmut` (or `cosmic-ray`) scoped to `src/northstack/domain/` and `src/northstack/application/budget_authority.py` — small, pure, high-consequence. Record a baseline mutation score before widening. Too slow for per-PR CI; run nightly or on demand.

**Watch for:** equivalent mutants in the redaction and policy code will need an ignore list.

### 2. Stateful / model-based testing of the ledger and projection

**Catches:** invariant violations reachable only through a specific sequence of operations — precisely what example-based tests miss.

**Why here:** the event ledger is the system's source of truth, it is append-only and hash-chained, and `ProjectionCache` maintains a cursor that must stay consistent with a full replay. Cursor arithmetic is exactly the kind of thing that breaks on an unusual interleaving. `test_cold_paths.py::test_p3_incremental_fold_equals_full_fold` is the single-case version of this; the general version is a state machine.

**First action:** `hypothesis.stateful.RuleBasedStateMachine` over the orchestrator's operations (create run, append event, project, invalidate, stop, replay). Invariants asserted after *every* rule:

- seq is strictly monotonic with no gaps
- the hash chain verifies end to end
- incremental projection equals a full `replay_run`
- budget remaining never goes negative
- a terminal cell never transitions again

### 3. Secret-redaction sweep

**Catches:** credential leakage into logs, errors, artifacts and API responses.

**Why here:** `secrets_policy.is_sensitive_env_name` already has a confirmed hole — plurals (`API_KEYS`, `TOKENS`, `PASSWORDS`, `SECRETS`) pass `validate_env_allowlist` because neither regex branch can match a trailing `S`. It is pinned as `xfail` in `test_domain_policies.py`. One confirmed hole in a fail-safe predicate justifies auditing the whole surface rather than patching the one case.

**First action:** a property test that generates env names from the sensitive stems with realistic decorations (plural, prefix, suffix, mixed case, separators) and asserts `is_sensitive_env_name` rejects all of them. Then a second sweep asserting a canary secret never appears in: log records, exception `str`/`repr`, stored artifacts, API response bodies, or TUI output.

## Tier 2 — needs the seams we now have

### 4. Fault injection through the provider seam

**Catches:** error-handling paths that only run when the outside world misbehaves.

**Why here:** `tests/helpers/fake_gateway.py` makes this cheap now — `httpx.MockTransport` drives the real `_execute_request` status pipeline without a network. Before it existed, tests mocked above the HTTP call and those paths never ran.

**First action:** scripted adverse sequences rather than single responses — 429 storms with `Retry-After`, a timeout mid-response, truncated JSON, a 200 with an empty `choices` array, a provider returning a different model than requested. Assert every run still reaches a terminal state and the ledger stays replayable.

### 5. Crash consistency

**Catches:** corruption from interruption at the wrong moment.

**Why here:** the ledger claims append-only durability and the supervisor claims release-exactly- once. Both are only interesting under abrupt termination.

**First action:** kill a run mid-flight (cancel the asyncio task at randomized points; separately, truncate the ledger file mid-record), then reopen and replay. The ledger must either replay cleanly or raise `LedgerCorruption` — never silently return a wrong state. Assert no orphaned tasks and no leaked file handles after `release()`.

### 6. Schema-evolution corpus

**Catches:** silent breakage of old runs when event shapes change.

**Why here:** `events/upcast.py` has the ladder machinery but `UPCASTERS` is empty at v1, so the mechanism is untested against real drift. The failure mode is invisible until someone opens an old run.

**First action:** freeze a corpus of serialized ledgers under `tests/fixtures/` tagged by schema version, and assert each still replays to an expected `RunState` on every commit. Add a new frozen ledger whenever `CURRENT_SCHEMA_VERSION` increments. This is regression insurance, not a bug hunt.

## Tier 3 — worth doing, lower yield right now

### 7. Adversarial input / agent-specific security

The orchestrator runs commands, writes files and fetches URLs on a model's instruction. Distinct from redaction:

- **Prompt injection through tool results** — a fetched page or file whose content instructs the model. Assert tool output is contained (delimited, labelled untrusted) and cannot alter the contract.
- **SSRF beyond the current check** — `url_policy` handles loopback; test redirect chains landing on loopback, DNS rebinding, IPv6 forms, decimal/octal IP encodings, and `0.0.0.0`.
- **Workspace escape** — path traversal, symlinks, absolute paths, Windows device names, and writes attempted without a valid lease.

### 8. Determinism and concurrency

Cells are gathered concurrently and the budget authority is a shared mutable resource. Test that the same inputs plus the same fakes produce a byte-identical ledger, and stress the budget path for double-spend under concurrent reservation. Add `pytest-randomly` (order independence) and `pytest-timeout` (deadlock detection) to CI — both are cheap and catch real coupling.

### 9. Characterization / golden tests

Snapshot the projected `RunState` and the rendered API payloads for a canned run. Catches unintended behaviour drift that no unit test asserts. Keep the golden files small and reviewable or they become noise.

### 10. Fake-vs-real contract drift

The fakes in `fake_gateway.py` encode an assumption about provider wire format. Validate the canned payloads against the providers' published response schemas, or capture cassettes once against a real endpoint, so the fakes cannot drift into fiction.

### 11. Performance guards

`ProjectionCache` exists specifically so a live view does not re-fold history on every poll. Assert that property as a complexity bound — projection cost scales with new events, not run length — rather than as a wall-clock benchmark.

## Suggested order

1. `pytest-randomly` + `pytest-timeout` in CI (one afternoon, immediate signal)
2. Secret-redaction property sweep, and decide the plural fix
3. Mutation baseline on `domain/`
4. Ledger state machine
5. Fault injection through the provider seam
6. Schema-evolution corpus

## Open source decisions

- **`secrets_policy` plurals** — currently `xfail`. Fixing the regex is a two-character change (`(?:S?)(?:_|$)`), but it widens what the sandbox rejects, so it is a policy call.
- **`GraphVersion.with_cell_status` with an unknown `cell_id`** — silently returns an unchanged copy. Raising would surface caller bugs; staying silent is safer for replay of unknown cells.

