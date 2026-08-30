# ADR 0008: Docker Isolation Seam for Command Profiles

Accepted.
## Context

The native workspace boundary was documented as "restricted execution, not a hardened sandbox": subprocesses use exact argv and `shell=False`, but repository code executed by a test command runs with the host process's authority. The README's answer was "use Docker or another real isolation boundary" — but there was no seam to actually configure that. Meanwhile the live benchmark executes model-written code inside workspace snapshots, which is exactly the threat model an isolation boundary exists for.

## Decision

Command profiles declare their isolation:

```toml
[[northstack.commands]]
name = "test"
argv = ["python", "-m", "pytest", "tests/", "-q"]
isolation = "docker"          # default: "host"
docker_image = "python:3.12-slim"
```

Host mode is the unchanged default. Nothing about existing behavior changes unless an operator opts a command into Docker.

Docker mode wraps, never interprets. The command's exact argv becomes:

```
docker run --rm --network none -v <workspace>:/workspace -w /workspace <image> <argv...>
```

Fixed flags only: throwaway container, no network egress, the workspace bind-mounted read-write at `/workspace` with it as cwd, and only the image and the operator's argv as variables. The wrapped argv still runs through the same `run_command` boundary (`shell=False`, bounded capture, timeout with process-tree termination, env allowlist applied to the docker CLI process — the container receives no passthrough environment at all).

Fail-closed availability law. Docker availability is probed once per process (CLI plus daemon, via `docker version`), with an injectable probe for tests. Unavailable Docker makes every docker-isolated command fail with an explicit error — `refusing to run on host` — and never a silent fallback to host execution. The refusal is the safety property: the operator asked for a boundary, and a boundary that quietly disappears under failure is not one.

## Why

- The escalation from "restricted" to "isolated" is per-command and explicit; commands that need host context, or operators without Docker, lose nothing.
- One fixed wrapper shape keeps the audit story simple: the ledger's command evidence describes the same argv either way, and the container cannot smuggle in network access or environment secrets by construction.
- Fail-closed mirrors the project's existing law for unsupported hard-policy checks: absence of the required machinery is a failure, not a downgrade.

## Consequences

- `--network none` means test commands requiring network (e.g. `pip install`) will not work under Docker isolation. That is intentional — such commands belong in image preparation, not in the run.
- The availability probe is cached per process; a daemon started mid-run is picked up only by a new process.
- Hidden checks in the live benchmark still execute on the host (they are operator-authored, not model-authored). Giving `HiddenCheck` an isolation option is future work if benchmark tasks ever need it.
- TOML round-trips emit the new fields only when non-default, so existing configs are byte-stable.
