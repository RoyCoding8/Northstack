/*
  Polling manager.  Centralized so cadence/pause/resume live in one place.

  - History poll (runs list + active) at historyPollMs.
  - Per-run event poll at eventsPollMs with a since=<seq> cursor.
  - Pauses on document.hidden (visibilitychange); resumes with a one-shot
    immediate catch-up but caps a runaway backlog by clamping since.
  - Debounced to avoid stacking; one in-flight request at a time per lane.
*/
import { http, store } from "./api.js";

const PHASES = ["intake", "contracted", "planned", "executing", "verifying", "verified"];

export const poll = {
  _timers: {},
  _inflight: { history: false, runs: new Set() },
  _watched: new Set(),
  _enabled: false,

  start() {
    this.stop();
    this._enabled = true;
    this._history();
    for (const runId of this._watched) this._pollRun(runId, true);
  },

  stop() {
    this._enabled = false;
    for (const id of Object.keys(this._timers)) {
      clearTimeout(this._timers[id]);
      delete this._timers[id];
    }
    // _inflight is deliberately left alone: each lane clears its own flag in a
    // finally block, so resetting here would only let a second request into a
    // lane that already has one airborne.
  },

  refreshNow() {
    // fire an immediate round without waiting for the timer
    this._history(true);
  },

  _schedule(key, fn, ms) {
    this._timers[key] = setTimeout(() => fn(), ms);
  },

  async _history(immediate = false) {
    if (this._inflight.history) return;
    this._inflight.history = true;
    try {
      const [runs, active] = await Promise.all([http.get("/runs?limit=50"), http.get("/runs/active")]);
      store.set({ runs: runs.runs || [], activeRuns: active.active || [] });
    } catch (e) {
      if (e.status !== 404) console.warn("history poll", e.message);
    } finally {
      this._inflight.history = false;
      const ms = store.state.settings.historyPollMs;
      if (this._enabled && !immediate) this._schedule("history", () => this._history(), ms);
    }
  },

  watchRun(runId) {
    if (!runId) return;
    this._watched.add(runId);
    if (!this._enabled) return;
    // start polling this run's events if not already
    if (this._timers["run-" + runId]) return;
    this._pollRun(runId, true);
  },

  unwatchRun(runId) {
    this._watched.delete(runId);
    clearTimeout(this._timers["run-" + runId]);
    delete this._timers["run-" + runId];
  },

  /** One-shot fetch for manually refreshed views; never reschedules. */
  refreshRun(runId) {
    if (runId) return this._pollRun(runId);
  },

  async _pollRun(runId, immediate = false) {
    const key = "run-" + runId;
    if (this._inflight.runs.has(runId)) return;
    this._inflight.runs.add(runId);
    const live = store.state.live[runId];
    try {
      const snap = await http.get(`/runs/${runId}`);
      const since = live?.since ?? 0;
      const ev = await http.get(`/runs/${runId}/events?since=${since}&limit=500`);
      const merged = (live?.events || []).concat(ev.events || []);
      // cap backlog to avoid unbounded growth while paused/hidden
      const capped = merged.slice(-5000);
      const nextSince = ev.next_seq ?? since;
      store.setRun(runId, {
        snapshot: snap,
        events: capped,
        since: nextSince,
        active: snap.active === true,
      });
    } catch (e) {
      if (e.status !== 404) console.warn("run poll", runId, e.message);
    } finally {
      this._inflight.runs.delete(runId);
      const snap = store.state.live[runId]?.snapshot;
      const terminal = snap && (snap.status === "verified" || snap.status === "abstained" || snap.status === "failed");
      // terminal runs: one final poll then stop polling (keep data)
      if (terminal) this._watched.delete(runId);
      if (this._enabled && this._watched.has(runId)) {
        const ms = store.state.settings.eventsPollMs;
        this._timers[key] = setTimeout(() => this._pollRun(runId), ms);
      } else delete this._timers[key];
    }
  },
};

/* Registered once, at module scope.  `start()` is re-entered on every resume
   and on every mc:poll-rate change, so registering from inside it made each
   listener add another listener that itself calls start() -- the count doubled
   per tab switch until the re-render storm made the UI unclickable. */
document.addEventListener("visibilitychange", () => {
  if (document.hidden) poll.stop();
  else poll.start();
});

export { PHASES };
