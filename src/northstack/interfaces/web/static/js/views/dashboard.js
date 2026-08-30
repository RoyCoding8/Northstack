/* Dashboard — at-a-glance control room. Composes stat tiles + pipeline funnel
   + budget bullets + secret grid + recent runs. */
import { el, fmtUsd, fmtInt, toast, statusBadge } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";
import { PHASES } from "../poll.js";

const FUNNEL = ["intake", "contracted", "planned", "executing", "verifying"];

export function dashboardView() {
  const root = el("div");
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Dashboard" }),
    el("p", { text: "Runs, budgets, and provider keys at a glance." }),
  ));

  // Gradient descriptor strip (Figma "top-states" bar) over the KPI cluster.
  root.appendChild(el("div", { class: "sectionbar" },
    Icon("trendingUp", 16),
    el("span", { text: "Operations today" }),
  ));

  const statGrid = el("div", { class: "grid grid--stats" });
  statGrid.appendChild(statTile("active", "Active runs", "—", 0));
  statGrid.appendChild(statTile("today", "Runs today", "—", 1));
  statGrid.appendChild(statTile("verified", "Verified", "—", 2));
  statGrid.appendChild(statTile("tokens", "Tokens today", "—", 3));
  statGrid.appendChild(statTile("cost", "Cost today", "—", 4));
  root.appendChild(statGrid);

  // Pipeline funnel
  const fun = el("section", { class: "card", style: { marginTop: "var(--p-space-4)" } });
  fun.appendChild(sectionHead("Pipeline", "#/runs", "All runs"));
  const funnelRow = el("div", { class: "funnel-row" });
  fun.appendChild(funnelRow);
  FUNNEL.forEach((ph, i) => {
    funnelRow.appendChild(phaseNode(ph, i));
    if (i < FUNNEL.length - 1) funnelRow.appendChild(el("div", { class: "stepper__sep" }));
  });
  root.appendChild(fun);

  // Budget bullets per active run — empty state shows soft orange bar legend
  // like Figma "Top states" so the page still reads colorful with no runs.
  const budgetSection = el("section", { class: "card", style: { marginTop: "var(--p-space-4)" } });
  budgetSection.appendChild(sectionHead("Active budgets", "#/runs", "All runs"));
  const budgetBody = el("div", { class: "stack" });
  budgetSection.appendChild(budgetBody);
  root.appendChild(budgetSection);

  // Soft "Growth" placeholder card — green area chart silhouette (Figma Growth).
  const growthSection = el("section", { class: "card", style: { marginTop: "var(--p-space-4)" } });
  growthSection.appendChild(sectionHead("Token trend", "#/runs", "All runs"));
  const growthBody = el("div", { class: "growth-empty" });
  growthBody.innerHTML = `
    <svg viewBox="0 0 360 120" width="100%" height="120" aria-hidden="true" preserveAspectRatio="none">
      <defs>
        <linearGradient id="gFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="var(--chart-growth)" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="var(--chart-growth-soft)" stop-opacity="0.05"/>
        </linearGradient>
      </defs>
      <path d="M0,100 C40,95 60,70 90,55 C120,40 140,50 170,35 C200,20 230,45 260,30 C290,18 320,25 360,12 L360,120 L0,120 Z" fill="url(#gFill)"/>
      <path d="M0,100 C40,95 60,70 90,55 C120,40 140,50 170,35 C200,20 230,45 260,30 C290,18 320,25 360,12" fill="none" stroke="var(--chart-growth)" stroke-width="2" stroke-linecap="round"/>
    </svg>
    <div class="muted" style="margin-top:var(--p-space-2);font:var(--md-sys-typescale-label)">Cumulative tokens appear here once a run is active.</div>
  `;
  growthSection.appendChild(growthBody);
  root.appendChild(growthSection);

  // Secret status grid
  const secretSection = el("section", { class: "card", style: { marginTop: "var(--p-space-4)" } });
  secretSection.appendChild(sectionHead("Secret status", "#/profiles", "All profiles"));
  const secretBody = el("div", { class: "grid grid--3" });
  secretSection.appendChild(secretBody);
  root.appendChild(secretSection);

  // Recent runs
  const recent = el("section", { class: "card", style: { marginTop: "var(--p-space-4)" } });
  recent.appendChild(sectionHead("Recent runs", "#/runs", "All runs"));
  const recentBody = el("div", { class: "stack" });
  recent.appendChild(recentBody);
  root.appendChild(recent);

  // Trailing spacer so the fixed New-Run FAB never covers the last card when
  // the page is scrolled to the bottom (FAB footprint ≈ 56px + margin).
  root.appendChild(el("div", { "aria-hidden": "true", style: { height: "96px" } }));

  // New Run FAB (fixed; the spacer above is what clears it). Carries the
  // animated conic sheen ring — the one glossy signature element per page.
  const fab = el("button", { class: "fab sheen", "aria-label": "New run", onclick: () => { location.hash = "#/runs"; } },
    Icon("plus"), el("span", { text: "New run" }),
  );
  root.appendChild(fab);

  function render(state) {
    const runs = state.runs || [];
    const active = state.activeRuns || [];
    // KPIs
    setStat("active", active.length);
    const todayRuns = runs.filter(r => isToday(r.start_time));
    setStat("today", todayRuns.length);
    const finished = runs.filter(r => r.outcome);
    const verified = runs.filter(r => r.outcome === "verified");
    setStat("verified", finished.length ? `${verified.length}/${finished.length}` : "—");
    // tokens/cost today: sum over the active-run snapshots we already poll
    // (finished runs aren't replayed in the list view; usage lives in snapshots).
    let tok = 0, cost = 0;
    for (const rid of active) {
      const u = state.live[rid]?.snapshot?.usage || {};
      tok += (u.total_input_tokens || 0) + (u.total_output_tokens || 0);
      cost += u.total_cost_usd || 0;
    }
    setStat("tokens", fmtInt(tok));
    setStat("cost", fmtUsd(cost));

    // funnel: count runs in each phase (most recent run's phase glows)
    const phaseCounts = {};
    for (const ph of FUNNEL) phaseCounts[ph] = 0;
    for (const r of runs) if (FUNNEL.includes(r.status)) phaseCounts[r.status]++;
    const latestActive = active[0] && (state.live[active[0]]?.snapshot);
    const glow = latestActive?.status || (runs[0]?.status);
    funnelRow.querySelectorAll("[data-phase]").forEach(n => {
      const ph = n.dataset.phase;
      n.querySelector(".stat__value").textContent = phaseCounts[ph] || 0;
      n.style.boxShadow = ph === glow ? "var(--p-elev-2)" : "var(--p-elev-1)";
    });

    // budgets for active runs — empty: soft orange rank bars (Figma Top states)
    budgetBody.innerHTML = "";
    if (active.length === 0) {
      const demo = [
        { label: "intake", pct: 72 },
        { label: "planned", pct: 48 },
        { label: "executing", pct: 28 },
        { label: "verifying", pct: 12 },
      ];
      const list = el("div", { class: "rank-bars" });
      for (const d of demo) {
        list.appendChild(el("div", { class: "rank-bars__row" },
          el("span", { class: "rank-bars__label", text: d.label }),
          el("div", { class: "rank-bars__track" },
            el("div", { class: "rank-bars__fill", style: { width: d.pct + "%" } }),
          ),
        ));
      }
      list.appendChild(el("div", { class: "muted", style: { marginTop: "var(--p-space-2)", font: "var(--md-sys-typescale-label)" }, text: "Sample values. Live budgets replace them once a run is active." }));
      budgetBody.appendChild(list);
    } else {
      for (const rid of active) {
        const snap = state.live[rid]?.snapshot;
        budgetBody.appendChild(activeRunRow(rid, snap));
      }
    }

    // secrets
    secretBody.innerHTML = "";
    if (state.secrets && state.secrets.profiles?.length) {
      for (const p of state.secrets.profiles) secretBody.appendChild(secretTile(p));
    } else {
      secretBody.appendChild(el("div", { class: "empty" }, Icon("key", 24), el("div", { text: "No profiles configured." })));
    }

    // recent runs
    recentBody.innerHTML = "";
    if (runs.length === 0) {
      recentBody.appendChild(el("div", { class: "empty" }, Icon("runs", 24), el("div", { text: "No runs yet. Start one from the Runs page." })));
    } else {
      for (const r of runs.slice(0, 5)) recentBody.appendChild(recentRow(r));
    }
  }

  store.subscribe(render, root);
  render(store.state);
  return root;
}

/* Soft pastel accents for KPI tiles — orange first (brand), then green/teal/blue/amber */
const CHART_VAR = ["--chart-1","--chart-2","--chart-3","--chart-4","--chart-5","--chart-6","--chart-7","--chart-8"];
function statTile(id, label, value, idx) {
  return el("div", { class: "stat", dataset: { stat: id }, style: { "--accent": `var(${CHART_VAR[(idx ?? 0) % CHART_VAR.length]})` } },
    el("div", { class: "stat__accent" }),
    el("div", { class: "stat__label", text: label }),
    el("div", { class: "stat__value", text: value }),
  );
}
// Figma dashboard pattern: section title on the left, "All X →" orange link
// on the right, both in one header row.
function sectionHead(title, linkHref, linkText) {
  return el("div", { class: "section-head" },
    el("h2", { class: "section-title", text: title }),
    linkHref && el("a", { class: "section-head__link", href: linkHref },
      el("span", { text: linkText }),
      Icon("arrowForward", 16),
    ),
  );
}
function setStat(id, v) {
  document.querySelector(`[data-stat="${id}"] .stat__value`)?.replaceChildren(document.createTextNode(String(v)));
}
function phaseNode(ph, i) {
  // Glass tile; the only colour is one brand-family accent hairline (dsh:
  // emphasis is opacity and space, never a pastel fill).
  const idx = (i ?? 0) % CHART_VAR.length;
  return el("div", {
    class: "stat stat--phase",
    dataset: { phase: ph },
    style: { "--accent": `var(${CHART_VAR[idx]})` },
  },
    el("div", { class: "stat__accent" }),
    el("div", { class: "stat__label", text: ph }),
    el("div", { class: "stat__value stat__value--mono", text: "0" }),
  );
}
function activeRunRow(rid, snap) {
  const row = el("div", { class: "row" });
  row.appendChild(el("code", { text: rid.slice(0, 12) }));
  row.appendChild(statusBadge(snap?.status, snap?.outcome));
  const usage = snap?.usage || {};
  const tokens = (usage.total_input_tokens || 0) + (usage.total_output_tokens || 0);
  const cost = usage.total_cost_usd || 0;
  const bar = el("div", { class: "budget-bar", style: { flex: "1 1 200px" } },
    el("div", { class: "budget-bar__fill" + (tokens > 100000 ? "--over" : "--ok"), style: { width: "0%" } }),
  );
  row.appendChild(bar);
  row.appendChild(el("span", { class: "num mono nowrap", text: `${fmtInt(tokens)} tok · ${fmtUsd(cost)}` }));
  return row;
}
function secretTile(p) {
  const ok = p.key_status && p.key_status.includes("OK");
  const none = !p.api_key_env;
  const dotCls = none ? "none" : ok ? "ok" : "unset";
  const dotLabel = none ? "no key" : ok ? "set OK" : "unset";
  return el("div", { class: "card", style: { padding: "var(--p-space-3)" } },
    el("div", { class: "row" },
      Icon("key", 18),
      el("span", { style: { fontWeight: "600" }, text: p.name }),
    ),
    el("div", { class: "row", style: { marginTop: "var(--p-space-2)" } },
      el("span", { class: `badge badge--dot ${dotCls}`, "aria-hidden": "true" }),
      el("code", { text: p.api_key_env || "—" }),
      el("span", { class: "muted", text: dotLabel }),
    ),
    el("div", { class: "muted", style: { fontSize: "12px", marginTop: "var(--p-space-1)" }, text: "The value is set in your environment and is never shown here." }),
  );
}
function recentRow(r) {
  return el("div", { class: "card card--clickable", role: "link", tabindex: "0", onclick: () => { location.hash = `#/runs/${r.run_id}` }, onkeydown: (e) => { if (e.key === "Enter") location.hash = `#/runs/${r.run_id}`; } },
    el("div", { class: "row" },
      el("code", { text: r.run_id.slice(0, 12) }),
      statusBadge(r.status, r.outcome),
      el("span", { class: "muted nowrap", text: `${r.event_count || 0} events` }),
    ),
  );
}
function isToday(ts) {
  if (!ts) return false;
  const d = new Date(ts * 1000);
  const now = new Date();
  return d.toDateString() === now.toDateString();
}
