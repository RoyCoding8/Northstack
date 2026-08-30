# ADR 0005: Enforced Layer Direction (KD5)

Accepted.
## Context

The architecture diagram always said the package is layered downwards: `interfaces -> application -> ports -> adapters -> events -> domain`, with `config` cross-cutting and forbidden from importing above the domain. For most of the project that direction was a comment in `docs/architecture.md` — and a comment nobody reads is a comment nobody honours. Two violations had crept in:

- Storage (`sqlite_ledger`) reached into a read model, coupling the append seam to how events are interpreted.
- The worker reached into a provider's private symbol instead of going through the public pricing seam.

Both were silent. They compiled, they passed, and they encoded a dependency the architecture explicitly forbids.

## Decision

The layer direction is machine-checked, not documented. `[tool.importlinter]` in `pyproject.toml` declares three contracts, and `uv run lint-imports` is a CI gate alongside the coverage floor:

1. "Layers point downwards only" — a `layers` contract over `interfaces`, `application`, `ports`, `adapters`, `events`, `domain`. An import pointing upwards is a CI failure.
2. "Config is cross-cutting and depends on nothing above the domain" — a `forbidden` contract on `northstack.config` against `adapters`, `application`, `events`, `interfaces`, `ports`.
3. "The domain is pure: no framework or I/O adapters" — a `forbidden` contract on `northstack.domain` against `httpx`, `sqlite3`, `fastapi`, `typer`, `jsonschema`.

## Why

- A violation is now a red build, not a review comment. The two prior violations are structurally impossible to reintroduce: the import contract rejects them at CI time.
- The domain-pure contract keeps the decision-theoretic core free of I/O, so it can be reasoned about and tested without a process, a socket, or a framework. An accidental `import sqlite3` in `domain/` is now a failing gate, not a latent coupling.
- Config being contractually below the domain means a load-bearing default cannot secretly depend on an adapter it is meant to configure.

## Consequences

- Refactoring must keep the arrow pointing down: moving a read model into storage, or a framework import into the domain, fails the build before it ships.
- The contracts are deliberately minimal (three). They encode the direction the diagram already promised; they don't invent invariants the code doesn't already assume.
