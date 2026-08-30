/* Run Detail — live per-run page: phase tracker, stats, sparkline, budget,
   virtualized-ish event timeline, integrity, stop, re-run, export. */
import { el, toast, fmtInt, fmtUsd, fmtTime, fmtElapsed, fmtSpan, copyText, statusBadge, confirmDialog, dialog } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";
import { poll, PHASES } from "../poll.js";

const PHASE_ORDER = ["intake", "contracted", "planned", "executing", "verifying"];
const KIND_GROUPS = { all: null, status: ["status_changed", "request_accepted", "workspace_snapshot"], cell: ["cell_created", "cell_started", "cell_completed", "cell_failed", "cell_advanced", "claim_recorded", "route_selected"], budget: ["budget_updated"], evidence: ["evidence_recorded", "verification_check", "analysis_requested", "analysis_completed", "contract_proposed", "contract_validated", "contract_amended", "graph_proposed", "graph_accepted", "artifact_stored"], outcome: ["outcome_emitted", "recovery_transition"] };
export function runDetailView(runId) {
  const root = el("div");
  let kindFilter = "all", paused = true, autoScroll = true;
  let execTab = "graph", selectedCellId = null, lastGraphKey = "";
  const graphHost = el("div");
  const graphDetail = el("div");
  const redrawGraph = () => {
    const live = store.state.live[runId];
    if (!live?.snapshot) return;
    renderGraph(live.snapshot, live.events || [], graphCells(live.snapshot, live.events || []));
  };
  poll.refreshRun(runId);

  // header
  const head = el("div", { class: "page-head" });
  root.appendChild(head);
  const hero = el("div", { class: "row" });
  head.appendChild(hero);
  hero.appendChild(el("button", { class: "icon-btn", "aria-label": "Back to runs", onclick: () => { location.hash = "#/runs"; } }, Icon("chevronLeft")));
  hero.appendChild(el("h1", { style: { margin: 0 }, text: `Run ${runId.slice(0, 14)}` }));
  const badgeHost = el("span");
  hero.appendChild(badgeHost);
  hero.appendChild(el("button", { class: "icon-btn", "aria-label": "Copy run id", onclick: () => copyText(runId, "Run id copied") }, Icon("copy")));

  const meta = el("div", { class: "muted", style: { marginTop: "var(--p-space-2)" } });
  head.appendChild(meta);

  // phase tracker
  const tracker = el("div", { class: "stepper", style: { marginTop: "var(--p-space-4)" } });
  head.appendChild(tracker);

  // stat tiles
  const stats = el("div", { class: "grid grid--stats", style: { marginTop: "var(--p-space-4)" } });
  head.appendChild(stats);

  // sparkline + budget
  const chartRow = el("div", { class: "grid grid--2", style: { marginTop: "var(--p-space-4)" } });
  const sparkCard = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Cumulative tokens" }), el("canvas", { id: "spark", width: 600, height: 120, style: { width: "100%" } }));
  const budgetCard = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Budget" }));
  chartRow.appendChild(sparkCard);
  chartRow.appendChild(budgetCard);
  head.appendChild(chartRow);

  // failure / recovery panels (only when relevant)
  const failHost = el("div", { class: "stack", style: { marginTop: "var(--p-space-4)" } });
  head.appendChild(failHost);

  // execution: graph view (default) + raw event timeline
  const tlSection = el("section", { class: "card", style: { marginTop: "var(--p-space-4)" } });
  const tlHead = el("div", { class: "row" });
  tlHead.appendChild(el("h2", { class: "section-title", text: "Execution" }));
  const tabRow = el("div", { class: "row", style: { gap: "4px" } });
  for (const tab of ["graph", "events"]) {
    tabRow.appendChild(el("button", { class: "chip chip--filter", "aria-pressed": tab === "graph" ? "true" : "false", text: tab, onclick: (e) => { execTab = tab; tabRow.querySelectorAll(".chip--filter").forEach(c => c.setAttribute("aria-pressed", c === e.currentTarget ? "true" : "false")); graphWrap.classList.toggle("hidden", tab !== "graph"); tl.classList.toggle("hidden", tab !== "events"); filterRow.classList.toggle("hidden", tab !== "events"); render(); } }));
  }
  tlHead.appendChild(tabRow);
  const filterRow = el("div", { class: "row", style: { gap: "4px" } });
  for (const k of Object.keys(KIND_GROUPS)) {
    filterRow.appendChild(el("button", { class: "chip chip--filter", "aria-pressed": k === "all" ? "true" : "false", text: k, onclick: (e) => { kindFilter = k; filterRow.querySelectorAll(".chip--filter").forEach(c => c.setAttribute("aria-pressed", c === e.currentTarget ? "true" : "false")); render(); } }));
  }
  tlHead.appendChild(el("div", { class: "toolbar__spacer" }));
  const pauseButton = el("button", { class: "icon-btn", "aria-label": "Resume event polling", "aria-pressed": "true", onclick: () => { paused = !paused; pauseButton.setAttribute("aria-pressed", String(paused)); pauseButton.setAttribute("aria-label", paused ? "Resume event polling" : "Pause event polling"); pauseButton.replaceChildren(Icon(paused ? "play" : "pause")); if (paused) poll.unwatchRun(runId); else poll.watchRun(runId); } }, Icon("play"));
  tlHead.appendChild(pauseButton);
  tlHead.appendChild(el("button", { class: "icon-btn", "aria-label": "Refresh events", onclick: () => poll.refreshRun(runId) }, Icon("reload")));
  tlHead.appendChild(el("button", { class: "icon-btn", "aria-label": "Jump to latest", onclick: () => { autoScroll = true; render(); document.getElementById("tl")?.scrollTo({ top: 1e9 }); } }, Icon("arrowDown")));
  tlSection.appendChild(tlHead);
  const graphWrap = el("div", { class: "stack" },
    el("div", { class: "run-graph" }, graphHost),
    graphDetail,
  );
  tlSection.appendChild(graphWrap);
  tlSection.appendChild(filterRow);
  filterRow.classList.add("hidden");
  const tl = el("div", { id: "tl", class: "timeline hidden", style: { maxHeight: "60vh", overflow: "auto", marginTop: "var(--p-space-3)" }, "aria-live": "polite" });
  tlSection.appendChild(tl);
  root.appendChild(tlSection);

  // actions row
  const acts = el("div", { class: "row", style: { marginTop: "var(--p-space-4)" } });
  acts.appendChild(el("button", { class: "btn btn--tonal", onclick: checkIntegrity, "aria-label": "Integrity check" }, Icon("shield"), el("span", { text: "Integrity" })));
  acts.appendChild(el("button", { class: "btn btn--tonal", onclick: exportLedger, "aria-label": "Export ledger" }, Icon("download"), el("span", { text: "Export ledger" })));
  acts.appendChild(el("div", { class: "toolbar__spacer" }));
  acts.appendChild(el("button", { class: "btn btn--tonal", onclick: rerun, "aria-label": "Re-run with same goal" }, Icon("reload"), el("span", { text: "Re-run" })));
  const resumeButton = el("button", { class: "btn btn--filled", onclick: resumeRun, "aria-label": "Resume this run" }, Icon("play"), el("span", { text: "Resume" }));
  acts.appendChild(resumeButton);
  const stopButton = el("button", { class: "btn btn--danger", onclick: stopRun, "aria-label": "Stop run" }, Icon("stop"), el("span", { text: "Stop" }));
  acts.appendChild(stopButton);
  root.appendChild(acts);

  function render() {
    const live = store.state.live[runId];
    const snap = live?.snapshot;
    if (!snap) {
      tl.innerHTML = "";
      tl.appendChild(el("div", { class: "empty" }, el("div", { class: "skeleton", style: { height: "60px" } })));
      return;
    }
    stopButton.disabled = ["verified", "abstained", "failed"].includes(snap.status);
    resumeButton.classList.toggle("hidden", !["failed", "abstained"].includes(snap.status) || snap.active === true);
    // badge
    badgeHost.replaceChildren(statusBadge(snap.status, snap.outcome));
    // meta — snapshot() has no start_time; derive from first event timestamp.
    const evs = live.events || [];
    const startTime = evs.length ? evs[0].timestamp : null;
    const endTime = evs.length ? evs[evs.length - 1].timestamp : null;
    const timing = !startTime ? null
      : snap.active ? ` · ${fmtElapsed(startTime)}`
      : endTime ? ` · took ${fmtSpan(endTime - startTime)}` : null;
    // replaceChildren stringifies non-Nodes, so a bare null renders as "null".
    meta.replaceChildren(...[
      el("span", { class: "mono", text: runId }),
      el("span", { class: "muted", text: " · workspace " }),
      el("code", { text: snap.workspace_root || "—" }),
      el("span", { class: "muted", text: " · started " }),
      el("span", { text: startTime ? fmtTime(startTime) : "—" }),
      timing ? el("span", { class: "muted nowrap", text: timing }) : null,
    ].filter(Boolean));
    // phase tracker
    tracker.innerHTML = "";
    const curIdx = PHASE_ORDER.indexOf(snap.status);
    PHASE_ORDER.forEach((ph, i) => {
      const cls = snap.status === "verified" ? "stepper__step--terminal verified" :
        snap.status === "failed" || snap.status === "abstained" ? "stepper__step--terminal failed" :
        i < curIdx ? "stepper__step--done" : i === curIdx ? "stepper__step--current" : "";
      tracker.appendChild(el("div", { class: "stepper__step " + cls },
        el("div", { class: "stepper__dot" }), el("span", { text: ph })));
      if (i < PHASE_ORDER.length - 1) tracker.appendChild(el("div", { class: "stepper__sep" }));
    });
    // stats
    const u = snap.usage || {};
    stats.replaceChildren(
      stat("In tokens", fmtInt(u.total_input_tokens)),
      stat("Out tokens", fmtInt(u.total_output_tokens)),
      stat("Cost", fmtUsd(u.total_cost_usd)),
      stat("Calls", fmtInt(u.total_calls)),
      stat("Cells", fmtInt(snap.cells?.length || 0)),
      stat("Events", fmtInt(snap.events_replayed || 0)),
    );
    // sparkline
    drawSpark(live.events || []);
    budgetCard.replaceChildren(el("h3", { class: "section-title", text: "Budget" }));
    const budget = budgetProgress(snap), tokenTarget = budget.token_limit == null ? "unlimited" : fmtInt(budget.token_limit);
    budgetCard.appendChild(el("div", { class: "row" }, el("span", { class: "muted", text: `${fmtInt(budget.tokens)} / ${tokenTarget} tokens` }), budget.token_percent == null ? null : el("span", { class: "muted", style: { marginLeft: "auto" }, text: `${budget.token_percent.toFixed(0)}%` })));
    if (budget.token_percent != null) budgetCard.appendChild(el("div", { class: "budget-bar", style: { marginTop: "var(--p-space-2)" } }, el("div", { class: "budget-bar__fill" + (budget.token_percent >= 100 ? "--over" : budget.token_percent >= 70 ? "--warn" : "--ok"), style: { width: budget.token_percent + "%" } })));
    if (budget.cost || budget.cost_limit != null) {
      const target = budget.cost_limit == null ? "" : ` / ${fmtUsd(budget.cost_limit)}`;
      budgetCard.appendChild(el("div", { class: "muted", style: { marginTop: "var(--p-space-2)" }, text: `${fmtUsd(budget.cost)}${target} spent` }));
    }

    // failure / recovery
    failHost.replaceChildren();
    if (snap.status === "failed" || snap.status === "abstained") {
      failHost.appendChild(el("div", { class: "alert alert--error", role: "alert" }, Icon("warning", 20), el("div", {}, el("strong", { text: snap.outcome || snap.status }), snap.failure_type ? el("div", { class: "muted", text: `failure_type: ${snap.failure_type}` }) : null)));
    }
    if (snap.recovery_events && snap.recovery_events.length) {
      const rec = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Recovery actions" }));
      for (const e of snap.recovery_events) {
        const cell = e.cell_id ? `${e.cell_id}` : "?";
        const att = e.attempt_number != null ? `#${e.attempt_number}` : "#?";
        rec.appendChild(el("div", { class: "mono", text: `• ${e.action} · ${cell} ${att} (${e.failure_type || "?"})` }));
      }
      failHost.appendChild(rec);
    }

    // graph + events. The graph is only rebuilt when data actually changed:
    // the history poll re-renders every view on every tick, and rebuilding
    // the SVG under the cursor eats node clicks (mousedown/up race).
    const gCells = graphCells(snap, live.events || []);
    if (execTab === "graph") {
      const key = `${(live.events || []).length}:${snap.status}:${selectedCellId ?? ""}`;
      if (key !== lastGraphKey) {
        lastGraphKey = key;
        renderGraph(snap, live.events || [], gCells);
      }
    } else {
      lastGraphKey = "";
      renderDetail(live.events || [], gCells);
    }
    const kinds = KIND_GROUPS[kindFilter];
    const fevs = (live.events || []).filter(e => !kinds || kinds.includes(e.kind));
    tl.replaceChildren();
    if (fevs.length === 0) {
      tl.appendChild(el("div", { class: "empty" }, el("div", { text: "No events match." })));
    } else {
      // virtualize: render last 400 to keep DOM bounded
      const visible = fevs.slice(-400);
      for (const e of visible) tl.appendChild(eventRow(e));
    }
  }

  function svgEl(tag, attrs = {}) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    return e;
  }

  // The projection keeps cell status on the graph (state.cells is only
  // fed by cell_created, which the graph path skips), so merge statuses
  // from cell events onto the graph_accepted payload client-side.
  function graphCells(snap, events) {
    const ga = events.find(e => e.kind === "graph_accepted");
    let cells = snap.cells || [];
    if (!cells.length && ga?.payload?.cells) {
      const statusByCell = {};
      for (const e of events) {
        const p = e.payload || {};
        if (e.kind === "cell_started") statusByCell[p.cell_id] = "running";
        else if (e.kind === "cell_completed") statusByCell[p.cell_id] = "completed";
        else if (e.kind === "cell_failed") statusByCell[p.cell_id] = "failed";
      }
      cells = ga.payload.cells.map(c => ({ ...c, status: statusByCell[c.id] || c.status || "pending" }));
    }
    return cells;
  }

  function renderGraph(snap, events, cells) {
    graphHost.replaceChildren();
    if (!cells.length) {
      graphHost.appendChild(el("div", { class: "empty" }, el("div", { text: "No execution graph yet." })));
      graphDetail.replaceChildren();
      return;
    }
    const graphEvent = events.find(e => e.kind === "graph_accepted");
    const edges = graphEvent?.payload?.edges ?? [];
    const byWave = new Map();
    for (const c of cells) {
      if (!byWave.has(c.wave)) byWave.set(c.wave, []);
      byWave.get(c.wave).push(c);
    }
    const waves = [...byWave.keys()].sort((a, b) => a - b);
    const NODE_W = 188, NODE_H = 58, COL_W = 226, ROW_H = 76, PAD = 20;
    const rows = Math.max(...waves.map(w => byWave.get(w).length));
    const width = PAD * 2 + waves.length * COL_W;
    const height = PAD * 2 + rows * ROW_H;
    const pos = new Map();
    waves.forEach((w, wi) => byWave.get(w).forEach((c, ri) => pos.set(c.id, {
      x: PAD + wi * COL_W,
      y: PAD + ri * ROW_H,
    })));

    const svg = svgEl("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
    const defs = svgEl("defs");
    const marker = svgEl("marker", { id: "garrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
    marker.appendChild(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "gedge__arrow" }));
    defs.appendChild(marker);
    svg.appendChild(defs);

    const statusOf = (id) => cells.find(c => c.id === id)?.status ?? "pending";
    for (const edge of edges) {
      const a = pos.get(edge.from_id), b = pos.get(edge.to_id);
      if (!a || !b) continue;
      const x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2, x2 = b.x, y2 = b.y + NODE_H / 2;
      const mid = (x1 + x2) / 2;
      svg.appendChild(svgEl("path", {
        d: `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`,
        class: "gedge" + (statusOf(edge.from_id) === "completed" ? " gedge--done" : ""),
        "marker-end": "url(#garrow)",
      }));
    }

    for (const c of cells) {
      const { x, y } = pos.get(c.id);
      const g = svgEl("g", {
        class: "gnode",
        role: "button", tabindex: "0", "aria-label": `${c.name || c.id} (${c.status})`,
      });
      g.classList.add("gnode--" + c.status);
      if (selectedCellId === c.id) g.classList.add("gnode--selected");
      g.appendChild(svgEl("rect", { x, y, width: NODE_W, height: NODE_H, rx: "10" }));
      const label = c.name && c.name !== c.id ? c.name : c.id.replace(/^cell-[^-]*-/, "cell ");
      const t1 = svgEl("text", { x: x + 12, y: y + 22, class: "gnode__title" });
      t1.textContent = label.length > 26 ? label.slice(0, 25) + "…" : label;
      const profile = snap.routes?.[c.id];
      const t2 = svgEl("text", { x: x + 12, y: y + 42, class: "gnode__sub" });
      t2.textContent = `${c.status}${profile ? " · " + profile : ""}`;
      g.appendChild(t1);
      g.appendChild(t2);
      g.addEventListener("click", () => { selectedCellId = selectedCellId === c.id ? null : c.id; redrawGraph(); });
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectedCellId = selectedCellId === c.id ? null : c.id; redrawGraph(); } });
      svg.appendChild(g);
    }
    graphHost.appendChild(svg);
    renderDetail(events, cells);
  }

  function renderDetail(events, cells) {
    graphDetail.replaceChildren();
    const cell = (cells || []).find(c => c.id === selectedCellId);
    if (!cell) return;
    const evs = events.filter(e => e.payload?.cell_id === cell.id);
    const find = (kind) => [...evs].reverse().find(e => e.kind === kind);
    const route = find("route_selected"), started = find("cell_started"),
      completed = find("cell_completed"), failed = find("cell_failed");
    const usage = completed?.payload?.usage;
    const row = (label, value) => el("div", { class: "gdetail__row" },
      el("span", { class: "gdetail__label", text: label }),
      value instanceof Node ? value : el("span", { class: "gdetail__value", text: String(value ?? "—") }),
    );
    const panel = el("div", { class: "card gdetail" },
      el("div", { class: "row" },
        el("h3", { class: "card__title", text: cell.name && cell.name !== cell.id ? cell.name : cell.id }),
        statusBadge(cell.status, null),
      ),
      row("Cell ID", el("code", { text: cell.id })),
      row("Profile", route?.payload?.profile_name ?? "—"),
      row("Wave", cell.wave),
      row("Mode", cell.mode),
      row("Started", started ? fmtTime(started.timestamp) : "—"),
      row("Finished", completed ? fmtTime(completed.timestamp) : failed ? fmtTime(failed.timestamp) : "—"),
      usage ? row("Tokens", `${fmtInt(usage.input_tokens ?? 0)} in · ${fmtInt(usage.output_tokens ?? 0)} out`) : null,
      usage ? row("Cost", fmtUsd(usage.cost_usd ?? 0)) : null,
      completed?.payload?.output_artifact?.digest ? row("Artifact", el("code", { text: String(completed.payload.output_artifact.digest).slice(0, 24) + "…" })) : null,
      failed ? row("Error", el("span", { class: "gdetail__error", text: failed.payload.error || "—" })) : null,
      evs.length ? el("details", { class: "raw-json" },
        el("summary", { text: "Raw JSON" }),
        el("pre", { text: JSON.stringify(evs.map(e => ({ seq: e.seq, kind: e.kind, timestamp: e.timestamp, payload: e.payload })), null, 2) }),
      ) : null,
    );
    graphDetail.appendChild(panel);
  }

  function stat(label, value) { return el("div", { class: "stat" }, el("div", { class: "stat__label", text: label }), el("div", { class: "stat__value stat__value--mono", text: value })); }
  function eventRow(e) {
    const row = el("div", { class: "timeline__row", dataset: { kind: e.kind } });
    const head = el("div", { class: "timeline__head", onclick: () => { const b = row.querySelector(".timeline__body"); b.classList.toggle("hidden"); } });
    head.appendChild(el("span", { class: "timeline__seq", text: String(e.seq) }));
    head.appendChild(el("span", { class: "badge badge--neutral", text: e.kind }));
    if (e.payload?.profile_name) head.appendChild(el("span", { class: "chip chip--role", text: e.payload.profile_name }));
    head.appendChild(el("span", { class: "muted nowrap", style: { marginLeft: "auto" }, text: fmtTime(e.timestamp) }));
    row.appendChild(head);
    const body = el("div", { class: "timeline__body hidden" });
    body.textContent = JSON.stringify(e.payload, null, 2);
    row.appendChild(body);
    return row;
  }

  function drawSpark(events) {
    const canvas = document.getElementById("spark");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    // accumulate tokens from budget_updated events; fallback to event count
    const pts = [];
    let cum = 0;
    for (const e of events) {
      if (e.kind === "budget_updated") {
        const u = e.payload?.usage || {};
        cum = (u.total_input_tokens || 0) + (u.total_output_tokens || 0);
      }
      pts.push(cum);
    }
    if (pts.length < 2) pts.push(...Array.from({ length: Math.max(0, 2 - pts.length) }, () => 0));
    // grid
    ctx.strokeStyle = getCss("--chart-grid");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h - 1); ctx.lineTo(w, h - 1); ctx.stroke();
    const max = Math.max(...pts, 1);
    // Soft green area fill like Figma Growth chart — clean, not rainbow.
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, getCss("--chart-growth") || getCss("--chart-2"));
    grad.addColorStop(1, getCss("--chart-growth-soft") || getCss("--chart-fill"));
    ctx.strokeStyle = getCss("--chart-growth") || getCss("--chart-2") || getCss("--chart-stroke");
    ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((v, i) => { const x = (i / (pts.length - 1)) * w; const y = h - (v / max) * (h - 4) - 2; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = grad;
    ctx.globalAlpha = 0.22; ctx.fill(); ctx.globalAlpha = 1;
  }
  function getCss(v) { return (getComputedStyle(document.documentElement).getPropertyValue(v) || "").trim() || "currentColor"; }

  async function checkIntegrity() {
    try { const r = await http.get(`/runs/${runId}/integrity`); dialog({ title: "Integrity check", builder: (host, done) => { host.appendChild(el("div", { class: "alert " + (r.ok ? "alert--info" : "alert--error") }, Icon(r.ok ? "shield" : "warning", 18), el("div", {}, el("strong", { text: r.ok ? "Chain OK" : "Chain BROKEN" }), el("div", { class: "muted", text: `${r.events_checked} events checked${r.error_message ? `: ${r.error_message}` : ""}` })))); host.appendChild(el("div", { class: "dialog__actions" }, el("button", { class: "btn btn--tonal", text: "Close", onclick: () => done() }))); } }); } catch (e) { toast(e.message, { error: true }); }
  }
  async function exportLedger() {
    try {
      const ledger = await collectLedger(runId);
      const url = URL.createObjectURL(new Blob([JSON.stringify(ledger, null, 2)], { type: "application/json" }));
      const a = el("a", { href: url, download: `${runId}.ledger.json` });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) { toast(e.message, { error: true }); }
  }
  async function stopRun() {
    confirmDialog({ title: "Stop this run?", body: "Cancels the in-process task. The run will record a failed/abstained outcome.", confirmLabel: "Stop", onConfirm: async () => { try { await http.post(`/runs/${runId}/stop`); toast("Stopping"); } catch (e) { toast(e.message, { error: true }); } } });
  }
  function rerun() {
    const snap = store.state.live[runId]?.snapshot;
    const goal = snap?.goal || "";
    const ws = snap?.workspace_root || "";
    // hand off to runs view's new-run dialog via hash + a query param
    sessionStorage.setItem("mc.rerun", JSON.stringify({ goal, workspace_root: ws }));
    location.hash = "#/runs?rerun=1";
  }

  async function resumeRun() {
    try {
      const r = await http.post(`/runs/${encodeURIComponent(runId)}/resume`);
      toast(`Resumed as ${r.run_id}`);
      location.hash = `#/runs/${r.run_id}`;
    } catch (e) { toast(e.message, { error: true }); }
  }

  const unsubscribe = store.subscribe(render, root);
  root.dispose = () => { poll.unwatchRun(runId); unsubscribe(); };
  render();
  return root;
}

export async function collectLedger(runId, get = http.get, maxEvents = 100_000) {
  const events = [];
  let since = 0;
  while (true) {
    const page = await get(`/runs/${runId}/ledger.json?since=${since}&limit=5000`);
    if (!Array.isArray(page?.events)) throw new Error("invalid ledger export page");
    if (events.length + page.events.length > maxEvents) throw new Error(`ledger export exceeds ${maxEvents} events`);
    events.push(...page.events);
    if (!page.truncated) return { run_id: page.run_id || runId, events };
    if (!Number.isInteger(page.next_seq) || page.next_seq <= since) throw new Error("ledger export pagination made no progress");
    since = page.next_seq;
  }
}

export function budgetProgress(snapshot) {
  const usage = snapshot?.usage || {}, limits = snapshot?.budget || {};
  const tokens = (usage.total_input_tokens || 0) + (usage.total_output_tokens || 0);
  const token_limit = Number.isFinite(limits.token_limit) && limits.token_limit > 0 ? limits.token_limit : null;
  return {
    tokens,
    token_limit,
    token_percent: token_limit == null ? null : Math.min(100, tokens / token_limit * 100),
    cost: usage.total_cost_usd || 0,
    cost_limit: Number.isFinite(limits.cost_limit_usd) ? limits.cost_limit_usd : null,
  };
}
