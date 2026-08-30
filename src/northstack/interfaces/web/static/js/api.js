/*
  Tiny HTTP client + reactive store for the northstack control surface.

  No framework.  Plain modules.  The store is an object with subscribe(); views
  re-render from snapshots.  Polling is centralized here so cadence/pause live
  in one place.
*/

export const API = "/api";

const token = "NorthStack/1";
const AUTH_TOKEN_KEY = "mc.apiToken";
const DEFAULT_SETTINGS = Object.freeze({
  theme: "dark",
  eventsPollMs: 700,
  historyPollMs: 3000,
  landingPage: "#/dashboard",
});
const POLL_RANGES = { eventsPollMs: [200, 2000], historyPollMs: [1000, 10000] };

export class ApiError extends Error {
  constructor(message, { status, path, cause } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    if (cause) this.cause = cause;
  }
}

function requestHeaders(extra = {}) {
  const value = sessionStorage.getItem(AUTH_TOKEN_KEY) || localStorage.getItem(AUTH_TOKEN_KEY);
  return { ...extra, ...(value ? { Authorization: `Bearer ${value}` } : {}) };
}

export function setApiToken(value, { persistent = false } = {}) {
  sessionStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_TOKEN_KEY);
  if (value) (persistent ? localStorage : sessionStorage).setItem(AUTH_TOKEN_KEY, value);
}

async function jget(path, signal) {
  const r = await fetch(`${API}${path}`, { signal, headers: requestHeaders({ Accept: "application/json" }) });
  if (!r.ok) throw await asError(r, path);
  return r.status === 204 ? null : parseJson(r, path);
}
async function jsend(method, path, body, signal) {
  const r = await fetch(`${API}${path}`, {
    method,
    signal,
    headers: requestHeaders({ "Content-Type": "application/json", Accept: "application/json" }),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw await asError(r, path);
  return r.status === 204 ? null : parseJson(r, path);
}

async function parseJson(response, path) {
  const type = response.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() || "";
  const body = await response.text();
  if (!body) return null;
  if (type !== "application/json" && !type.endsWith("+json")) {
    throw new ApiError(`expected JSON from ${path}, received ${type || "unknown content type"}`, {
      status: response.status,
      path,
    });
  }
  try {
    return JSON.parse(body);
  } catch (cause) {
    throw new ApiError(`invalid JSON response from ${path}`, {
      status: response.status,
      path,
      cause,
    });
  }
}

async function asError(r, path) {
  let detail = `${r.status} ${r.statusText}`;
  try {
    const j = await r.json();
    detail = j.detail || JSON.stringify(j);
  } catch { /* keep status */ }
  return new ApiError(detail, { status: r.status, path });
}

export const http = {
  fetch: (path, options = {}) => fetch(`${API}${path}`, {
    ...options,
    headers: requestHeaders(options.headers || {}),
  }),
  get: jget,
  post: (p, b, s) => jsend("POST", p, b, s),
  put: (p, b, s) => jsend("PUT", p, b, s),
  patch: (p, b, s) => jsend("PATCH", p, b, s),
  del: (p, s) => jsend("DELETE", p, undefined, s),
};

/* ---------- reactive store ---------- */
export const store = {
  _subs: new Set(),
  _notifying: false,
  state: {
    config: null,        // GET /config (with key_status + tier + unsaved)
    secrets: null,       // GET /secrets/status
    runs: [],            // GET /runs
    activeRuns: [],      // GET /runs/active
    settings: loadSettings(),
    live: {},            // run_id -> { snapshot, events, since, poll }
  },
  get() { return this.state; },
  set(patch) {
    Object.assign(this.state, patch);
    this._notify();
  },
  setRun(runId, patch) {
    const cur = this.state.live[runId] || { events: [], since: 0, paused: false };
    this.state.live[runId] = { ...cur, ...patch };
    this._notify();
  },
  _notify() {
    // Guards synchronous re-entry only (a subscriber that calls set() inline
    // would otherwise recurse until the stack blows).  It cannot catch a
    // subscriber that awaits before writing -- that loop is unbounded and no
    // store can detect it, so never subscribe an effect that fetches and sets.
    if (this._notifying) return;
    this._notifying = true;
    try {
      for (const rec of [...this._subs]) {
        if (rec.owner && !rec.owner.isConnected) { this._subs.delete(rec); continue; }
        rec.fn(this.state);
      }
    } finally { this._notifying = false; }
  },
  /* `owner` is the view root. The router swaps #main without telling anyone,
     so a view's subscription is released when its root leaves the document
     rather than by a teardown call no caller makes. */
  subscribe(fn, owner) {
    const rec = { fn, owner };
    this._subs.add(rec);
    fn(this.state);
    return () => this._subs.delete(rec);
  },
};

export function loadSettings() {
  try {
    return normalizeSettings(JSON.parse(localStorage.getItem("mc.settings") || "{}"));
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(s) {
  store.state.settings = normalizeSettings({ ...store.state.settings, ...s });
  localStorage.setItem("mc.settings", JSON.stringify(store.state.settings));
}

function normalizeSettings(settings) {
  const normalized = { ...DEFAULT_SETTINGS, ...settings };
  for (const [key, [minimum, maximum]] of Object.entries(POLL_RANGES)) {
    const value = settings[key];
    normalized[key] = Number.isFinite(value) && value >= minimum && value <= maximum
      ? value
      : DEFAULT_SETTINGS[key];
  }
  return normalized;
}
