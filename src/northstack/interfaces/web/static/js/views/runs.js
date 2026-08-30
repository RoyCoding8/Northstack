/* Runs — history table (filter/sort/CSV), New Run dialog, Compare two runs. */
import { el, toast, fmtTime, fmtInt, statusBadge, dialog, confirmDialog } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";

let pendingGoal = "";
let pendingWs = "";
let pendingBudget = "";
let pendingMaxWaves = 3;

export function runsView() {
  let sortKey = "start_time", sortDir = -1, statusF = null, outcomeF = null;
  const selected = new Set();
  const root = el("div");
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Runs" }),
    el("p", { text: "Run history and launch new runs." }),
  ));

  const tb = el("div", { class: "toolbar" });
  tb.appendChild(el("button", { class: "chip chip--filter", "aria-pressed": "false", text: "active only", onclick: (e) => { statusF = statusF === "active" ? null : "active"; updateFilters(); } }));
  tb.appendChild(el("button", { class: "chip chip--filter", "aria-pressed": "false", text: "verified", onclick: (e) => { outcomeF = outcomeF === "verified" ? null : "verified"; updateFilters(); } }));
  tb.appendChild(el("button", { class: "chip chip--filter", "aria-pressed": "false", text: "failed", onclick: (e) => { outcomeF = outcomeF === "failed" ? null : "failed"; updateFilters(); } }));
  tb.appendChild(el("div", { class: "toolbar__spacer" }));
  tb.appendChild(el("button", { class: "btn btn--tonal", onclick: exportCsv, "aria-label": "Export CSV" }, Icon("download"), el("span", { text: "CSV" })));
  tb.appendChild(el("button", { class: "btn btn--tonal", onclick: compareDialog, disabled: true, "aria-label": "Compare", dataset: { cmp: "1" } }, Icon("compare"), el("span", { text: "Compare" })));
  tb.appendChild(el("button", { class: "btn btn--danger", onclick: deleteSelected, disabled: true, "aria-label": "Delete selected runs", dataset: { del: "1" } }, Icon("trash"), el("span", { text: "Delete" })));
  tb.appendChild(el("button", { class: "btn btn--filled", onclick: newRunDialog, "aria-label": "New run" }, Icon("plus"), el("span", { text: "New run" })));
  root.appendChild(tb);

  const wrap = el("div", { class: "card", style: { padding: 0, overflow: "auto" } });
  root.appendChild(wrap);

  function updateFilters() {
    root.querySelectorAll(".chip--filter").forEach((c, i) => {
      const labels = ["active only", "verified", "failed"];
      const on = (labels[i] === "active only" && statusF === "active") || (labels[i] === "verified" && outcomeF === "verified") || (labels[i] === "failed" && outcomeF === "failed");
      c.setAttribute("aria-pressed", on ? "true" : "false");
    });
    render();
  }

  function render() {
    let runs = store.state.runs || [];
    if (statusF === "active") {
      const active = new Set(store.state.activeRuns || []);
      runs = runs.filter(r => active.has(r.run_id));
    }
    if (outcomeF) runs = runs.filter(r => r.outcome === outcomeF);
    runs = [...runs].sort((a, b) => {
      if (sortKey === "start_time") return (a.start_time - b.start_time) * sortDir;
      if (sortKey === "event_count") return (a.event_count - b.event_count) * sortDir;
      return String(a[sortKey]).localeCompare(String(b[sortKey])) * sortDir;
    });
    wrap.innerHTML = "";
    const cmpBtn = root.querySelector("[data-cmp]");
    if (cmpBtn) cmpBtn.disabled = selected.size !== 2;
    const delBtn = root.querySelector("[data-del]");
    if (delBtn) delBtn.disabled = selected.size === 0;

    if (runs.length === 0) {
      wrap.appendChild(el("div", { class: "empty" }, Icon("runs", 24), el("div", { text: "No runs match." })));
      return;
    }
    const table = el("table", { class: "table" });
    table.appendChild(el("thead", {}, el("tr", {},
      el("th", { text: "" }),
      colHead("run_id", "Run"), colHead("status", "Status"), colHead("outcome", "Outcome"),
      colHead("event_count", "Events"), colHead("start_time", "Started"), colHead("last_event_time", "Last"),
    )));
    const body = el("tbody");
    for (const r of runs) body.appendChild(rowTr(r));
    table.appendChild(body);
    wrap.appendChild(table);
  }
  function colHead(key, label) {
    const th = el("th", { onclick: () => { if (sortKey === key) sortDir = -sortDir; else { sortKey = key; sortDir = 1; } render(); }, "aria-sort": sortKey === key ? (sortDir === 1 ? "ascending" : "descending") : "none" });
    th.textContent = label;
    return th;
  }
  function rowTr(r) {
    const tr = el("tr", { role: "link", tabindex: 0, onclick: () => { location.hash = `#/runs/${r.run_id}`; }, onkeydown: (e) => { if (e.key === "Enter") location.hash = `#/runs/${r.run_id}`; } });
    const cb = el("input", { type: "checkbox", "aria-label": `Compare ${r.run_id}`, checked: selected.has(r.run_id), onclick: (e) => { e.stopPropagation(); toggleCompareSelection(selected, r.run_id, e.target.checked); render(); } });
    tr.appendChild(el("td", {}, cb));
    tr.appendChild(el("td", { class: "mono nowrap", text: r.run_id.slice(0, 14) }));
    tr.appendChild(el("td", {}, statusBadge(r.status, null)));
    tr.appendChild(el("td", {}, statusBadge(null, r.outcome)));
    tr.appendChild(el("td", { class: "num nowrap", text: fmtInt(r.event_count) }));
    tr.appendChild(el("td", { class: "mono nowrap", text: fmtTime(r.start_time) }));
    tr.appendChild(el("td", { class: "mono nowrap", text: fmtTime(r.last_event_time) }));
    return tr;
  }

  function newRunDialog(prefill) {
    if (prefill) { pendingGoal = prefill.goal || ""; pendingWs = prefill.workspace_root || ""; pendingBudget = prefill.budget || ""; pendingMaxWaves = prefill.max_waves || 3; }
    dialog({
      title: "New run",
      builder: (host, done) => {
        const form = el("div", { class: "stack" });
        form.appendChild(f("Goal", el("textarea", { class: "field__textarea", "aria-label": "Goal", required: true, maxlength: 65536, oninput: (e) => pendingGoal = e.target.value }, pendingGoal)));
        form.appendChild(f("Workspace root", el("input", { class: "field__input", "aria-label": "Workspace root", required: true, maxlength: 4096, value: pendingWs, oninput: (e) => pendingWs = e.target.value })));
        const grid = el("div", { class: "grid grid--2" });
        grid.appendChild(f("Budget tokens (optional)", el("input", { class: "field__input", type: "number", min: 0, step: 1, "aria-label": "Budget tokens", value: pendingBudget, oninput: (e) => pendingBudget = e.target.value })));
        grid.appendChild(f("Max waves", el("input", { class: "field__input", type: "number", min: 1, max: 100, step: 1, required: true, "aria-label": "Max waves", value: pendingMaxWaves, oninput: (e) => pendingMaxWaves = parseInt(e.target.value, 10) || 3 })));
        form.appendChild(grid);
        host.appendChild(form);
        host.appendChild(el("div", { class: "dialog__actions" },
          el("button", { class: "btn btn--tonal", text: "Cancel", onclick: () => done() }),
          el("button", { class: "btn btn--filled", text: "Start run", onclick: async () => {
            if (!pendingGoal.trim() || !pendingWs) { toast("Goal and workspace required", { error: true }); return; }
            try {
              const body = { goal: pendingGoal, workspace_root: pendingWs, max_waves: pendingMaxWaves };
              if (pendingBudget) body.budget_tokens = parseInt(pendingBudget, 10);
              const { run_id } = await http.post("/runs", body);
              toast(`Run started: ${run_id}`);
              done();
              location.hash = `#/runs/${run_id}`;
            } catch (e) { toast(e.message, { error: true }); }
          } }),
        ));
      },
    });
  }

  function deleteSelected() {
    const ids = [...selected];
    if (!ids.length) return;
    confirmDialog({
      title: ids.length === 1 ? `Delete run ${ids[0].slice(0, 14)}?` : `Delete ${ids.length} runs?`,
      body: "The runs disappear from this list. Their ledger events stay on disk, so audit and replay keep working.",
      confirmLabel: "Delete",
      onConfirm: async () => {
        try {
          await Promise.all(ids.map((id) => http.del(`/runs/${encodeURIComponent(id)}`)));
          selected.clear();
          toast(ids.length === 1 ? "Run deleted" : `${ids.length} runs deleted`);
          render();
        } catch (e) { toast(e.message, { error: true }); }
      },
    });
  }

  function compareDialog() {
    const ids = [...selected];
    if (ids.length !== 2) { toast("Select exactly two runs", { error: true }); return; }
    dialog({
      title: "Compare runs",
      builder: (host, done) => {
        host.appendChild(el("div", { class: "alert alert--info" }, Icon("compare", 18), el("div", { text: `${ids[0].slice(0, 14)}  vs  ${ids[1].slice(0, 14)}` })));
        const out = el("div", { class: "card" }, el("div", { class: "skeleton", style: { height: "120px" } }));
        host.appendChild(out);
        http.get(`/runs/compare?a=${encodeURIComponent(ids[0])}&b=${encodeURIComponent(ids[1])}`).then((d) => {
          out.innerHTML = "";
          out.appendChild(el("h3", { class: "section-title", text: "Outcome" }));
          out.appendChild(el("div", { class: "row" }, mini(d.outcome.a), Icon("chevronRight"), mini(d.outcome.b)));
          out.appendChild(el("h3", { class: "section-title", style: { marginTop: "var(--p-space-3)" }, text: "Usage delta (a → b)" }));
          const u = d.usage || {};
          out.appendChild(el("div", { class: "mono" }, `calls ${fmtInt((u.a?.total_calls||0))} → ${fmtInt((u.b?.total_calls||0))} (Δ ${fmtInt(d.delta?.total_calls||0)})`));
          out.appendChild(el("div", { class: "mono" }, `cost  $${(u.a?.total_cost_usd||0).toFixed(4)} → $${(u.b?.total_cost_usd||0).toFixed(4)} (Δ $${(d.delta?.total_cost_usd||0).toFixed(4)})`));
          out.appendChild(el("div", { class: "mono" }, `tok   ${fmtInt(u.a?.total_input_tokens||0)}+${fmtInt(u.a?.total_output_tokens||0)} → ${fmtInt(u.b?.total_input_tokens||0)}+${fmtInt(u.b?.total_output_tokens||0)}`));
        }).catch((e) => { out.innerHTML = ""; out.appendChild(el("div", { class: "alert alert--error" }, Icon("warning", 18), el("div", { text: e.message }))); });
        host.appendChild(el("div", { class: "dialog__actions" }, el("button", { class: "btn btn--tonal", text: "Close", onclick: () => done() })));
      },
    });
  }

  async function exportCsv() {
    try {
      const url = URL.createObjectURL(new Blob([await collectRunsCsv()], { type: "text/csv" }));
      const a = el("a", { href: url, download: "runs.csv" });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) { toast(e.message, { error: true }); }
  }

  function mini(v) { return el("span", { class: "badge badge--neutral", text: String(v || "—") }); }
  function f(label, c) { return el("div", { class: "field" }, el("label", { text: label }), c); }

  store.subscribe(render, root);
  render();

  // re-run handoff from the Run Detail view: it stashes a prefill in
  // sessionStorage and navigates here; open the New Run dialog with it.
  try {
    const raw = sessionStorage.getItem("mc.rerun");
    if (raw) { sessionStorage.removeItem("mc.rerun"); newRunDialog(JSON.parse(raw)); }
  } catch { /* ignore */ }
  // also accept a live mc:rerun custom event (used when already on the runs
  // page).  Only the newest view may hold it -- otherwise every visit arms
  // another handler and one rerun opens a dialog per visit.
  document.removeEventListener("mc:rerun", rerunHandler);
  rerunHandler = (e) => { if (e.detail) newRunDialog(e.detail); };
  document.addEventListener("mc:rerun", rerunHandler);

  return root;
}

export function toggleCompareSelection(selected, runId, checked) {
  if (!checked) selected.delete(runId);
  else { if (selected.size >= 2) selected.clear(); selected.add(runId); }
  return selected;
}

export async function collectRunsCsv(fetcher = http.fetch, maxBytes = 50 * 1024 * 1024) {
  const chunks = [], encoder = new TextEncoder();
  let header, offset = 0, bytes = 0;
  while (true) {
    const response = await fetcher(`/runs/export?format=csv&limit=10000&offset=${offset}`);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const text = await response.text(), end = text.search(/\r?\n/);
    if (end < 0) throw new Error("invalid runs CSV page");
    const nextHeader = text.slice(0, end), body = text.slice(end + (text[end] === "\r" ? 2 : 1));
    if (header !== undefined && nextHeader !== header) throw new Error("runs CSV header changed between pages");
    header ??= nextHeader;
    const chunk = chunks.length ? body : text;
    bytes += encoder.encode(chunk).byteLength;
    if (bytes > maxBytes) throw new Error(`runs CSV export exceeds ${maxBytes} bytes`);
    chunks.push(chunk);
    if (response.headers.get("X-NorthStack-Truncated") !== "true") return chunks.join("");
    const next = Number(response.headers.get("X-NorthStack-Next-Offset"));
    if (!Number.isSafeInteger(next) || next <= offset) throw new Error("runs CSV pagination made no progress");
    offset = next;
  }
}

let rerunHandler;
