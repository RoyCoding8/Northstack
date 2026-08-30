FROM node:22-slim AS web
WORKDIR /web
COPY package.json package-lock.json tailwind.config.cjs tailwind-src.css ./
RUN npm ci
RUN mkdir -p src/northstack/interfaces/web/static && npm run build:css

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev --extra web

COPY README.md ./
COPY src ./src
COPY --from=web /web/src/northstack/interfaces/web/static/components.css \
     src/northstack/interfaces/web/static/components.css
RUN uv sync --locked --no-dev --extra web --no-editable

FROM python:3.12-slim

RUN groupadd -r northstack && useradd -r -g northstack -m northstack

WORKDIR /app
COPY --from=builder --chown=northstack:northstack /app/.venv /app/.venv
COPY --chown=northstack:northstack northstack.toml ./northstack.toml
COPY --chown=northstack:northstack sandboxes ./sandboxes
ENV PATH="/app/.venv/bin:$PATH" VIRTUAL_ENV=/app/.venv PORT=8080

USER northstack
EXPOSE 8080

# Cloud Run injects $PORT and terminates TLS in front of a public URL; the
# instance ships no provider credentials, so it can browse past runs but not
# start one.
ENTRYPOINT ["sh", "-c", "exec northstack-web --host 0.0.0.0 --port \"$PORT\" --dangerous-allow-non-loopback --dangerous-no-auth --config /app/northstack.toml --files-base-root /app"]
