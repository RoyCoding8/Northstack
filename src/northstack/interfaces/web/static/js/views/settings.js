/* Settings — theme/poll preferences, company + run config, full TOML editor. */
import { el, toast, confirmDialog, dialog } from "../util.js";
import { Icon } from "../icons.js";
import { http, store, saveSettings } from "../api.js";

export function settingsView() {
  const root = el("div");
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Settings" }),
    el("p", { text: "Operator preferences and every configurable northstack.toml field. Unknown sections stay in the TOML editor." }),
  ));

  const appearance = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Appearance" }));
  appearance.appendChild(row("Theme",
    segmented(["system", "light", "dark"], store.state.settings.theme || "dark", (v) => { saveSettings({ theme: v }); applyTheme(v); })));
  appearance.appendChild(row("Landing page",
    segmented(["dashboard", "profiles", "routing", "commands", "runs", "files"], (store.state.settings.landingPage || "#/dashboard").replace(/^#\//, ""), (v) => { saveSettings({ landingPage: `#/${v}` }); })));
  root.appendChild(appearance);

  const pollCard = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Polling cadence" }));
  const evLbl = el("div", { class: "muted", text: `${store.state.settings.eventsPollMs || 700} ms` });
  const evSlider = sliderRow("Events poll (ms)", store.state.settings.eventsPollMs || 700, 200, 2000, 100, (v) => { saveSettings({ eventsPollMs: v }); evLbl.textContent = `${v} ms`; document.dispatchEvent(new CustomEvent("mc:poll-rate", { detail: { eventsPollMs: v } })); });
  pollCard.appendChild(evSlider);
  const hiLbl = el("div", { class: "muted", text: `${(store.state.settings.historyPollMs || 3000) / 1000} s` });
  const hiSlider = sliderRow("History poll (s)", (store.state.settings.historyPollMs || 3000) / 1000, 1, 10, 0.5, (v) => { const ms = Math.round(v * 1000); saveSettings({ historyPollMs: ms }); hiLbl.textContent = `${v} s`; document.dispatchEvent(new CustomEvent("mc:poll-rate", { detail: { historyPollMs: ms } })); });
  pollCard.appendChild(hiSlider);
  root.appendChild(pollCard);

  const company = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Company" }));
  const nameInput = el("input", { class: "field__input", "aria-label": "Company name" });
  company.appendChild(field("Name", nameInput));
  company.appendChild(el("div", { class: "row", style: { marginTop: "var(--p-space-3)" } },
    el("button", { class: "btn btn--filled", onclick: applyName, "aria-label": "Apply company name" }, el("span", { text: "Apply" })),
  ));
  root.appendChild(company);

  const runCard = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Run defaults" }));
  const tokensInput = el("input", { class: "field__input", type: "number", min: "0", "aria-label": "Default budget tokens" });
  const costInput = el("input", { class: "field__input", type: "number", min: "0", step: "0.01", "aria-label": "Default budget cost USD" });
  const stallInput = el("input", { class: "field__input", type: "number", min: "0", step: "0.1", "aria-label": "Stall window seconds" });
  const calInput = el("input", { class: "field__input", "aria-label": "Calibration path" });
  let plannerMode = "single";
  let falsifierMode = "off";
  const plannerSeg = segmented(["single", "model"], "single", (v) => { plannerMode = v; });
  const falsifierSeg = segmented(["off", "model"], "off", (v) => { falsifierMode = v; });
  const runGrid = el("div", { class: "grid grid--2" });
  runGrid.appendChild(field("Default budget tokens", tokensInput, "0 means unlimited."));
  runGrid.appendChild(field("Default budget cost (USD)", costInput, "0 means unlimited. Free-model pools stay at 0."));
  runGrid.appendChild(field("Stall window (seconds)", stallInput, "0 disables stall detection."));
  runGrid.appendChild(field("Calibration path", calInput, "Empty disables calibrated soft review."));
  runCard.appendChild(runGrid);
  runCard.appendChild(row("Planner mode", plannerSeg));
  runCard.appendChild(row("Falsifier mode", falsifierSeg));
  runCard.appendChild(el("div", { class: "row", style: { marginTop: "var(--p-space-3)" } },
    el("button", { class: "btn btn--filled", onclick: applyRun, "aria-label": "Apply run defaults" }, el("span", { text: "Apply" })),
  ));
  root.appendChild(runCard);

  const cfgCard = el("div", { class: "card" }, el("h3", { class: "section-title", text: "Config file" }));
  const dirtyBadge = el("span", { class: "badge badge--neutral" });
  cfgCard.appendChild(el("div", { class: "row" }, el("span", { class: "muted", text: "Unsaved changes: " }), dirtyBadge));
  const acts = el("div", { class: "row", style: { marginTop: "var(--p-space-3)" } });
  acts.appendChild(el("button", { class: "btn btn--tonal", onclick: reload, "aria-label": "Reload config from TOML" }, Icon("reload"), el("span", { text: "Reload from TOML" })));
  acts.appendChild(el("button", { class: "btn btn--tonal", onclick: validate, "aria-label": "Validate current config" }, Icon("shieldCheck"), el("span", { text: "Validate" })));
  acts.appendChild(el("div", { class: "toolbar__spacer" }));
  acts.appendChild(el("button", { class: "btn btn--filled", onclick: save, "aria-label": "Save config to TOML" }, Icon("save"), el("span", { text: "Save to TOML" })));
  cfgCard.appendChild(acts);
  const tomlArea = el("textarea", { class: "field__textarea", rows: "18", spellcheck: "false", "aria-label": "northstack.toml source" });
  cfgCard.appendChild(field("northstack.toml", tomlArea, "Full document, including unknown [northstack.*] sections. Apply loads it into memory; Save writes the file."));
  cfgCard.appendChild(el("div", { class: "row", style: { marginTop: "var(--p-space-3)" } },
    el("button", { class: "btn btn--filled", onclick: applyToml, "aria-label": "Apply TOML" }, el("span", { text: "Apply TOML" })),
  ));
  root.appendChild(cfgCard);

  const about = el("div", { class: "card" }, el("h3", { class: "section-title", text: "About" }));
  about.appendChild(el("div", { class: "row" }, el("span", { class: "muted", text: "NorthStack Control Surface" })));
  about.appendChild(el("div", { class: "muted", style: { marginTop: "var(--p-space-2)" }, text: "localhost-only. Path resolution is containment, not a security sandbox." }));
  root.appendChild(about);

  function fillFromConfig() {
    const cfg = store.state.config;
    nameInput.value = cfg?.name || "";
    const r = cfg?.run || {};
    tokensInput.value = String(r.default_budget_tokens ?? 100000);
    costInput.value = String(r.default_budget_cost_usd ?? 5);
    stallInput.value = String(r.stall_window_seconds ?? 0);
    plannerMode = r.planner_mode || "single";
    falsifierMode = r.falsifier_mode || "off";
    setSegmented(plannerSeg, plannerMode);
    setSegmented(falsifierSeg, falsifierMode);
    calInput.value = r.calibration_path || "";
  }

  function renderDirty() {
    const dirty = store.state.config?.unsaved;
    dirtyBadge.replaceChildren();
    dirtyBadge.classList.toggle("badge--neutral", !dirty);
    dirtyBadge.classList.toggle("badge--active", !!dirty);
    dirtyBadge.textContent = dirty ? "Unsaved" : "Saved";
  }

  async function loadToml() {
    try { tomlArea.value = (await http.get("/config/toml")).text; }
    catch (e) { toast(e.message, { error: true }); }
  }

  async function applyName() {
    try {
      const cfg = await http.patch("/config/name", { name: nameInput.value });
      store.set({ config: cfg });
      toast("Company name updated in memory");
      renderDirty();
    } catch (e) { toast(e.message, { error: true }); }
  }

  async function applyRun() {
    try {
      const cfg = await http.put("/config/run", {
        default_budget_tokens: Number(tokensInput.value) || 0,
        default_budget_cost_usd: Number(costInput.value) || 0,
        stall_window_seconds: Number(stallInput.value) || 0,
        planner_mode: plannerMode,
        falsifier_mode: falsifierMode,
        calibration_path: calInput.value,
      });
      store.set({ config: cfg });
      toast("Run config updated in memory");
      renderDirty();
    } catch (e) { toast(e.message, { error: true }); }
  }

  async function applyToml() {
    try {
      const cfg = await http.put("/config/toml", { text: tomlArea.value });
      store.set({ config: cfg });
      fillFromConfig();
      toast("TOML applied in memory");
      renderDirty();
    } catch (e) { toast(e.message, { error: true }); }
  }

  async function reload() {
    const dirty = store.state.config?.unsaved;
    confirmDialog({
      title: dirty ? "Discard unsaved config?" : "Reload config from TOML?",
      body: dirty ? "In-memory edits will be lost and the config re-read from northstack.toml." : "Re-reads northstack.toml into memory.",
      confirmLabel: dirty ? "Discard & reload" : "Reload",
      danger: !!dirty,
      onConfirm: async () => {
        try {
          const cfg = await http.post("/config/reload");
          store.set({ config: cfg });
          fillFromConfig();
          await loadToml();
          toast("Config reloaded");
          renderDirty();
        } catch (e) { toast(e.message, { error: true }); }
      },
    });
  }
  async function validate() {
    try { const r = await http.post("/config/validate"); dialog({ title: "Validation", builder: (host, done) => { host.appendChild(el("div", { class: "alert alert--info" }, Icon("shieldCheck", 18), el("div", {}, el("strong", { text: r.valid ? "OK" : "Invalid" }), el("div", { class: "muted", text: "No errors." })))); host.appendChild(el("div", { class: "dialog__actions" }, el("button", { class: "btn btn--tonal", text: "Close", onclick: () => done() }))); } }); }
    catch (e) { toast(e.message, { error: true }); }
  }
  async function save() {
    try {
      const r = await http.post("/config/save");
      toast(`Saved to ${r.saved || "northstack.toml"}`);
      store.set({ config: r.config });
      fillFromConfig();
      await loadToml();
      renderDirty();
    } catch (e) { toast(e.message, { error: true }); }
  }

  function applyTheme(t) {
    const rootEl = document.documentElement;
    if (t === "system") { const dark = matchMedia("(prefers-color-scheme: dark)").matches; rootEl.setAttribute("data-theme", dark ? "dark" : "light"); }
    else rootEl.setAttribute("data-theme", t);
  }

  store.subscribe(renderDirty, root);
  fillFromConfig();
  loadToml();
  renderDirty();
  return root;
}

function field(label, control, helper) {
  const f = el("div", { class: "field" });
  f.appendChild(el("label", { text: label }));
  f.appendChild(control);
  if (helper) f.appendChild(el("div", { class: "field__helper", text: helper }));
  return f;
}
function row(label, control) { return el("div", { class: "row", style: { marginTop: "var(--p-space-2)" } }, el("span", { class: "muted nowrap", style: { minWidth: "120px" }, text: label }), control); }
function segmented(options, current, onPick) {
  const wrap = el("div", { class: "row", role: "group" });
  for (const o of options) {
    const b = el("button", { class: "chip chip--filter", "aria-pressed": o === current ? "true" : "false", text: o, onclick: (e) => { wrap.querySelectorAll(".chip--filter").forEach(c => c.setAttribute("aria-pressed", "false")); e.currentTarget.setAttribute("aria-pressed", "true"); onPick(o); } });
    wrap.appendChild(b);
  }
  return wrap;
}
function setSegmented(wrap, value) {
  wrap.querySelectorAll(".chip--filter").forEach(c => c.setAttribute("aria-pressed", c.textContent === value ? "true" : "false"));
}
function sliderRow(label, value, min, max, step, onInput) {
  const wrap = el("div", { class: "stack", style: { marginTop: "var(--p-space-2)" } });
  const lbl = el("div", { class: "row" }, el("label", { text: label }), el("span", { class: "muted", style: { marginLeft: "auto" }, text: String(value) }));
  const inp = el("input", { type: "range", class: "range", min: String(min), max: String(max), step: String(step), value: String(value), style: { width: "100%" }, oninput: (e) => { const v = parseFloat(e.target.value); lbl.lastChild.textContent = String(v); onInput(v); } });
  wrap.appendChild(lbl);
  wrap.appendChild(inp);
  return wrap;
}
