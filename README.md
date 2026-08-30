# NorthStack

NorthStack is a provider-neutral control plane for autonomous AI work. It tests one hypothesis: do temporary, typed task cells — planned, budgeted, and verified by deterministic code — beat simpler agent configurations on verified quality per token, call, and wall-clock?

Models propose work; the control plane owns everything else. Contracts, permissions, budgets, state transitions, verification, and the release decision are all code, not model output. It is not an employee simulator — there are no personas, salaries, or a simulated org.

## What it does

Runs against OpenAI-compatible, Anthropic-compatible, and Gemini-compatible endpoints:

- Configurable model profiles — roles, capabilities, concurrency, RPM limits, prices — with structural single-flight for `max_concurrency = 1`.
- Immutable executable contracts and rolling-wave graph cells.
- Mediated workspace tools and exact named command profiles, with opt-in Docker isolation (`isolation = "docker"`): throwaway no-network containers, failing closed when Docker is unavailable (ADR 0008).
- SSE streaming with retry-until-first-byte, normalized into the same `ModelResponse` as non-streaming calls — the ledger cannot tell the transports apart. Workers emit a heartbeat per streamed delta so long generations never look stalled.
- SQLite WAL event ledger with a per-run SHA-256 hash chain, content-addressed artifacts, and audit replay.
- Command, file, JSON Schema, and policy hard gates that run real workspace operations and cannot be waived.
- Calibration-gated blinded soft review with abstention: reviewers judge resolved artifact content (never digests), executor identity is structurally absent from their inputs rather than withheld, and every reviewer failure fails closed to disagreement.
- Typed bounded retry and capability rerouting, with one owner per authority: `CellRunner` owns the retry cap and reroute, `BudgetAuthority` owns spend, and `ReleaseLaw` is the sole constructor of a run's `RunOutcome`.
- Model-proposed graph planning and contract falsification under one propose-harden-fallback law: the model's decomposition is deterministically hardened (rewritten ids, recomputed waves, clamped budgets, enforced criterion coverage) or discarded for the canonical single-cell plan; a falsifier finding rejects the contract while an outage fails open (ADR 0009).
- `northstack calibrate`, which measures the review panel and feeds the measured `CalibrationRecord`s back into it.
- Machine-checked layer direction (import-linter contracts) and an 85% coverage floor.
- A four-configuration benchmark runner with paired bootstrap intervals, runnable live against real endpoints over clean per-run workspace snapshots and scored post-hoc by hidden checks the system under test never sees, plus an eight-task hermetic live-pilot suite whose integrity is CI-gated.

## Status

Alpha. The safety boundaries and release decisions are explicit and tested, but provider-backed runs consume real credentials and budget, and host-mode commands execute repository code on the host.

## Install from source

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```powershell
git clone https://git.sr.ht/~roycoding8/northstack
Set-Location northstack
uv sync --extra dev --extra web
uv run northstack --help
Copy-Item northstack.example.toml northstack.toml
```

Edit `northstack.toml`, then set each configured API-key environment variable. Endpoints without keys may omit `api_key_env`. Provider URLs must use HTTP or HTTPS and cannot contain credentials, query strings, or fragments. Credentialed non-loopback endpoints must use HTTPS; `allow_insecure_http = true` is an explicit dangerous override intended only for a trusted plaintext proxy.

## Run a project

```powershell
uv run northstack run `
  --config northstack.toml `
  --workspace C:\path\to\repository `
  --goal "Fix the failing parser tests without changing the public API"
```

The ledger and artifact store are written under `<workspace>/.northstack/`. A custom SQLite path can be supplied with `--db`.

Intake is model-backed where judgment is required and deterministic where it is not: a bounded, digest-recorded workspace scan feeds the repo-constraints analysis (statements of fact about manifests and structure, not model guesses), and the acceptance analysis proposes hard-checkable criteria with stack awareness. Configure two `reviewer`-role profiles to enable the blinded soft-review panel; with fewer than two, soft rubrics abstain by law.

## Inspect and replay

```powershell
uv run northstack inspect --db .northstack\ledger.db --run-id <run-id>
uv run northstack replay --db .northstack\ledger.db --run-id <run-id>
```

Audit replay reconstructs state from recorded events. It does not claim deterministic re-execution of stochastic provider calls. The ledger uses SQLite WAL with `synchronous=NORMAL` — throughput over durability, so a power loss can discard the most recently committed tail. The hash chain verifies the events that survive; it cannot prove a missing tail once existed. `northstack ledger events` / `northstack ledger verify` dump and hash-chain-check a ledger directly, and `northstack config validate` checks a TOML config without starting a run.

## Interactive TUI

```powershell
uv run python -m northstack.interfaces.tui     # ./northstack.toml + ./.env (config required -- see Install)
uv run python -m northstack.interfaces.tui -c path\to\northstack.toml
```

A full-screen, keyboard-driven control panel. It opens on the Actions tab — the launcher — so you can drive the system from one place; the info tabs come after it.

| Tab (order) | Content |
|---|---|
| **Actions** | Run Hermetic Tests, Run Live Tests, Launch Web UI (backend + browser), Launch Backend Only, Launch Frontend Only. Selecting a test drops out of the TUI, streams pytest on the real terminal, then holds the output with a "Press any key to return" prompt so you can read which test failed, and returns with a one-line summary on the tab. |
| **Profiles** | Model profiles: name, protocol, model, tier, concurrency, roles, API-key status (env-var name + OK/UNSET — never the value). `Enter` expands full details. |
| **Routing** | Role → profile fallback chains. |
| **Commands** | Named subprocess profiles: argv, timeout, max output, env allowlist. |

| Keys | Action |
|---|---|
| `←` / `→` (or `Tab` / `Shift+Tab`) | switch tabs |
| `j` / `k` (or `↑` / `↓`) | move the cursor within a tab |
| `Enter` | (Actions) launch the selected item; (Profiles) expand/collapse details |
| `q` / `Esc` | quit |

When stdout is not a TTY (piped, redirected, CI), the interactive loop is skipped and a static one-shot dump prints instead. The same module drives the web surface without the full-screen UI: `uv run python -m northstack.interfaces.tui web backend|frontend|both`.

Live tests require the `web` extra and a reachable provider:

```powershell
uv sync --extra web
$env:CLIPROXY_API_KEY = "sk-..."            # the key named by MC_LIVE_KEY_ENV
uv run python -m northstack.interfaces.tui   # Actions -> Run Live Tests
```

## Web control surface

```powershell
uv sync --extra web
uv run northstack-web                       # http://127.0.0.1:8000 (localhost only)
uv run northstack-web --config northstack.toml --port 8000
```

A Material 3 control-room SPA served from `src/northstack/interfaces/web/static/`. The styled UI needs the Tailwind bundle built once: `npm ci && npm run build:css` in the repository root (`northstack-web` also attempts this itself at startup when the bundle is missing). Docker and CI build it as part of their pipelines. The REST API lives at `/api/*` with interactive docs at `/api/docs`.

Loopback binding is the backwards-compatible no-auth local mode. Exposing the server on a non-loopback interface requires both an explicit flag and a bearer token from an environment variable:

```powershell
$env:NORTHSTACK_WEB_TOKEN = "replace-with-a-long-random-token"
uv run northstack-web --host 0.0.0.0 --dangerous-allow-non-loopback
```

Use `--token-env OTHER_ENV_NAME` to choose another environment variable and `--files-base-root C:\allowed\root` to constrain workspace browsing. When authentication is on, every `/api/*` route — including docs and OpenAPI — requires `Authorization: Bearer <token>`. The SPA can attach the operator token from `mc.apiToken` in session or local storage; the server never returns or logs it.

From the browser, end to end:

- **Dashboard** — live KPIs, the intake→verified pipeline funnel, per-run budget bullets, and a secret-status grid (env-var name + OK/●UNSET — never a value).
- **Profiles / Routing / Commands** — full CRUD over the config, round-tripped through the frozen pydantic models; edits apply to the next started run. Routing rules are the actual tuning lever: swap a role→profile chain and re-run to compare. Commands can be test-run dry (read-only preview, not a sandbox).
- **Runs** — start a run live (`POST /api/runs` with a goal + workspace_root + optional budgets), watch the Run Detail page poll events (`?since=` cursor, ~700ms) as the phase tracker advances and the token sparkline ticks, browse history, compare two runs side by side (token/cost/call deltas), and export the raw ledger JSON for audit.
- **Files** — browse any workspace's deliverables and `.northstack/` artifacts.
- **Settings** — theme, polling cadence, reload-from-TOML with dirty-config confirm.

Config writes round-trip through the frozen pydantic models so validators catch bad edits before save; persisting to `northstack.toml` is an explicit Save to TOML action. Secrets are env-var references only — the UI, API, and TOML writer never display or accept a value. Runs build a fresh `Company` + `ModelGateway` per call (no shared loop state), so two concurrent runs both reach a terminal state cleanly.

The Files API resolves requested workspaces under the configured `files_base_root` before opening them and rejects symlink-resolved escapes. Direct reads also deny `.env*`, `.git` internals, private key/certificate extensions (`.pem`, `.key`, `.p12`, `.pfx`), and `.northstack/ledger.db`; run artifacts remain available through the dedicated artifact endpoint. These controls reduce accidental exposure but do not make the web process a hardened sandbox.

## Docker deployment

```bash
NORTHSTACK_WEB_TOKEN=$(openssl rand -hex 24) docker compose up --build
```

The image is a multi-stage build: a Node stage compiles the Tailwind bundle (`npm run build:css`), then the uv stage (`--locked --no-dev --extra web`) bakes it into the non-root runtime image. Compose binds port 8000, bind-mounts `runs/` and `sandboxes/`, health-checks `/`, and refuses to start without `NORTHSTACK_WEB_TOKEN` in `.env` — the entrypoint passes `--dangerous-allow-non-loopback` (container networking is non-loopback by definition), and the server fails closed without the token.

## Run the offline pilot benchmark

```powershell
uv run northstack benchmark `
  --suite benchmarks\pilot.json `
  --output-dir benchmark-output `
  --cheap-candidates 1 `
  --bootstrap-samples 2000 `
  --seed 1
```

The pilot is a fixture-mode operability check that emits `benchmark-report.json` and `benchmark-report.md`, retaining failures and abstentions. It is not evidence that the architecture is superior; live strategies must use identical task inputs, hidden checks, and resource ceilings before comparison.

## Run the live pilot benchmark

```powershell
uv run northstack benchmark `
  --live `
  --config northstack.toml `
  --suite benchmarks\live-pilot.json `
  --output-dir benchmark-output `
  --cheap-candidates 3 `
  --bootstrap-samples 2000 `
  --seed 1
```

This calls every configured provider endpoint and consumes real budget under each task's ceilings. Each of the four preregistered configurations (strong single, cheap best-of-N, singleton expert, company) runs the same raw request against a clean per-run copy of the task's immutable workspace template, under the same tool policy and ceilings, and is scored post-hoc by executable hidden checks the system under test never sees. The retained outcome is what the hidden checks say; the system's own claim is audited as false acceptance / false rejection. Best-of-N selection uses the hidden-check score — an oracle-based selector, so a company win is conservative — with all candidates' resources accounted. Runs, snapshots, ledgers, and per-run `meta.json` sidecars land under `<output-dir>/runs/`; a crashed run is retained as a failure, never dropped. See [ADR 0007](docs/adr/0007-live-benchmark-law.md).

The report includes a Pareto frontier over quality versus every resource axis (tokens, calls, wall time, retries, tool ops, configured cost) under the protocol's dominance rule — frontier membership, never a declared winner. Add `--ablate` to also run the five component ablations (company minus one mechanism: routing, model planning, model intake, recovery, falsification), retained separately from the primary comparison ([ADR 0010](docs/adr/0010-ablations-frontier-heldout.md)).

A suite may declare `"cohort": "heldout"` — the frozen set. The CLI refuses to run it without an explicit `--allow-heldout`, so tuning-phase contamination cannot happen by accident. The bundled `benchmarks/live-pilot.json` is the eight-task development cohort.

## Restricted execution and Docker isolation

The native workspace boundary is restricted execution, not a hardened sandbox. Models never receive arbitrary shell access: mutations go through mediated file tools and subprocesses use predefined exact argv arrays with `shell=False`. Still, repository code executed by a test command can affect the host.

For a real isolation boundary, opt individual commands into Docker:

```toml
[[northstack.commands]]
name = "test"
argv = ["python", "-m", "pytest", "tests/", "-q"]
isolation = "docker"
docker_image = "python:3.12-slim"
```

The command then runs in a uniquely named throwaway container with `--rm`, `--init`, no network, all capabilities dropped, no-new-privileges, a 256-process limit, and the workspace bind-mounted read-write at `/workspace`. A timed-out Docker client triggers a force-remove plus container inspection before NorthStack returns. Docker mode fails closed: if the Docker CLI or daemon is unavailable, the command is refused with an explicit error — it never silently falls back to host execution. Host mode remains the default.

The read-only web tool blocks obvious private, loopback, link-local, reserved, and redirected SSRF targets, but it does not pin the TCP connection to the validated DNS address and therefore does not claim complete DNS-rebinding protection.

## Calibrate the review panel

```powershell
uv run northstack calibrate --config northstack.toml --samples samples.json --output-dir calibration
```

Measures each routed reviewer profile's agreement and accuracy on labeled samples (one real endpoint call per reviewer per sample) and emits `calibration.json` plus a Markdown report. Point `[northstack.run] calibration_path` at the JSON so future runs load the measured `CalibrationRecord`s — soft rubrics can then verify under measured thresholds instead of abstaining for lack of calibration. A malformed calibration file is a hard error, never a silently uncalibrated panel.

Optional model-backed planning and falsification are similarly opt-in:

```toml
[northstack.run]
planner_mode = "model"     # PLANNER-role profile proposes; hardener owns structure
falsifier_mode = "model"   # SPECIALIST-role profile hunts passing-but-wrong readings
```

## Scientific claim standard

The system earns its complexity only if held-out experiments show either better verified quality at equal resource use, or equal verified quality with fewer tokens, calls, time, retries, or configured cost.

See [Architecture](docs/architecture.md), [Experiment protocol](docs/experiment-protocol.md), and the [ADR index](docs/adr/).

## Development

The `web` and `dev` extras are required for the release gates:

```powershell
uv run pytest -m "not live" -q                    # 1. tests — live-marked suites excluded by default
uv run ruff format . ; uv run ruff check .        # 2. format + lint
uv run mypy                                      # 3. whole-package type check
uv run lint-imports                              # 4. enforced layer direction (import-linter)
uv run coverage run -m pytest -m "not live" -q ; uv run coverage report   # 5. 85% coverage floor
npm ci ; npm test ; npm run build:css                  # 6. web unit tests + generated CSS
uv build                                               # 7. wheel + source distribution
```

Live-marked tests (which hit a real provider) are excluded from the default suite by `-m "not live"`. The real Docker daemon smoke is opt-in through the CI workflow dispatch input. Coverage is enforced at an 85% floor via `[tool.coverage.report] fail_under = 85`; CI runs the suite once under coverage (steps 1 and 5 together), runs filesystem/process contracts on Linux and Windows, checks the live-suite fixtures, and audits dependencies.
