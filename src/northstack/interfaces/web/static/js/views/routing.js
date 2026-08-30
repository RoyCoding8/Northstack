/* Routing — role→profile fallback chains (the tuning lever). Reorder chips,
   add by autocomplete scoped to profiles declaring the role, live validation,
   presets (list + apply with diff/confirm). */
import { el, toast, dialog, selectMenu } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";

const ROLES = ["worker", "reviewer", "planner", "specialist", "orchestrator"];

export function routingView() {
  const root = el("div");
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Routing" }),
    el("p", { text: "Assign an ordered fallback chain of profiles to each role. The company tries each in order until one can serve the request." }),
  ));

  const tb = el("div", { class: "toolbar" });
  tb.appendChild(el("button", { class: "btn btn--tonal", onclick: loadPresets, "aria-label": "Apply preset" }, Icon("sparkles"), el("span", { text: "Presets" })));
  tb.appendChild(el("div", { class: "toolbar__spacer" }));
  tb.appendChild(el("button", { class: "btn btn--filled", onclick: save, "aria-label": "Save routing" }, Icon("save"), el("span", { text: "Save routing" })));
  root.appendChild(tb);

  const cards = el("div", { class: "grid grid--2" });
  root.appendChild(cards);

  const alertHost = el("div");
  root.insertBefore(alertHost, cards);

  function render() {
    const cfg = store.state.config;
    const routing = cfg?.routing || [];
    cards.innerHTML = "";
    for (const role of ROLES) {
      const entry = routing.find(r => r.role === role) || { role, profiles: [] };
      cards.appendChild(roleCard(role, entry, cfg));
    }
  }

  function roleCard(role, entry, cfg) {
    const card = el("div", { class: "card role-card" });
    card.appendChild(el("div", { class: "card__header" },
      el("div", { class: "row" }, Icon("gitBranch", 18), el("h3", { class: "card__title", text: role })),
      el("span", { class: "badge badge--neutral", text: `${entry.profiles.length} profile${entry.profiles.length === 1 ? "" : "s"}` }),
    ));
    const chain = el("div", { class: "role-card__chain" });
    card.appendChild(chain);
    entry.profiles.forEach((pname, idx) => {
      if (idx > 0) chain.appendChild(el("span", { class: "role-card__arrow" }, Icon("chevronRight", 14)));
      const chip = el("span", { class: "role-card__chip-drag" },
        el("span", { text: pname }),
        el("button", { class: "role-card__reorder", "aria-label": `Move ${pname} up`, disabled: idx === 0, onclick: () => reorder(role, idx, -1) }, Icon("chevronUp", 14)),
        el("button", { class: "role-card__reorder", "aria-label": `Move ${pname} down`, disabled: idx === entry.profiles.length - 1, onclick: () => reorder(role, idx, +1) }, Icon("chevronDown", 14)),
        el("button", { class: "role-card__reorder", "aria-label": `Remove ${pname}`, onclick: () => removeProfile(role, idx) }, Icon("x", 14)),
      );
      chain.appendChild(chip);
    });

    // add-by-autocomplete: scoped to profiles declaring this role
    const eligible = (cfg?.profiles || []).filter(p => p.roles.includes(role)).map(p => p.name);
    const addWrap = el("div", { class: "row", style: { marginTop: "var(--p-space-2)" } });
    const sel = selectMenu({
      options: eligible.map((name) => ({ value: name, label: name })),
      placeholder: eligible.length ? "Add a profile…" : "No profiles declare this role",
      ariaLabel: `Add profile to ${role}`,
      disabled: !eligible.length,
    });
    sel.style.maxWidth = "240px";
    addWrap.appendChild(sel);
    addWrap.appendChild(el("button", { class: "btn btn--tonal btn--sm", text: "Add", onclick: () => { if (sel.value) { addProfile(role, sel.value); sel.value = ""; } } }));
    card.appendChild(addWrap);
    return card;
  }

  function workingRouting() {
    return ROLES.map(role => {
      const e = (store.state.config?.routing || []).find(r => r.role === role);
      return e ? { role, profiles: [...e.profiles] } : { role, profiles: [] };
    });
  }
  function persist(local) {
    store.state.config = { ...store.state.config, routing: local };
    alertHost.replaceChildren();
    render();
  }
  function reorder(role, idx, dir) {
    const local = workingRouting();
    const e = local.find(r => r.role === role);
    const j = idx + dir;
    if (j < 0 || j >= e.profiles.length) return;
    [e.profiles[idx], e.profiles[j]] = [e.profiles[j], e.profiles[idx]];
    persist(local);
  }
  function removeProfile(role, idx) {
    const local = workingRouting();
    const e = local.find(r => r.role === role);
    e.profiles.splice(idx, 1);
    persist(local);
  }
  function addProfile(role, name) {
    const local = workingRouting();
    const e = local.find(r => r.role === role);
    if (e.profiles.includes(name)) { toast(`“${name}” is already in the ${role} chain`); return; }
    e.profiles.push(name);
    persist(local);
  }
  async function save() {
    try {
      const routing = workingRouting()
        .filter(entry => entry.profiles.length > 0)
        .map(entry => ({ role: entry.role, profiles: entry.profiles }));
      await http.put("/config/routing", { routing });
      await refreshConfig();
      alertHost.replaceChildren();
      toast("Routing saved in memory. Use Save to TOML in Settings to persist it");
    } catch (e) {
      showError(e.message);
      await refreshConfig();
    }
  }
  async function loadPresets() {
    try {
      const { presets } = await http.get("/config/routing/presets");
      const cfg = store.state.config;
      const current = Object.fromEntries((cfg?.routing || []).map(r => [r.role, r.profiles]));
      dialog({
        title: "Apply routing preset",
        builder: (host, done) => {
          for (const p of presets) {
            const diff = p.available && ROLES.some(r => JSON.stringify(p.routing[r] || []) !== JSON.stringify(current[r] || []));
            const card = el("button", {
              class: "card routing-preset",
              disabled: !p.available,
              "aria-label": p.available ? `Apply ${p.label}` : `${p.label} unavailable: ${p.reason}`,
              onclick: async () => {
                try {
                  await http.post(`/config/routing/presets/${encodeURIComponent(p.id)}/apply`);
                  await refreshConfig();
                  toast(`Applied preset “${p.label}”`);
                  done();
                } catch (e) { toast(e.message, { error: true }); }
              },
            },
              el("h3", { class: "card__title", text: p.label }),
              el("div", { class: "muted", style: { fontSize: "12px" }, text: p.available ? p.description : p.reason }),
              el("div", { class: "muted", style: { fontSize: "12px" }, text: p.available ? (diff ? "Will change current routing" : "Already matches current routing") : "Unavailable for the current profiles" }),
            );
            for (const [role, profs] of Object.entries(p.routing)) {
              card.appendChild(el("div", { class: "row", style: { gap: "4px" } }, el("span", { class: "muted nowrap", style: { minWidth: "80px" }, text: role }), el("code", { text: profs.join(" → ") })));
            }
            host.appendChild(card);
          }
          host.appendChild(el("div", { class: "dialog__actions" }, el("button", { class: "btn btn--tonal", text: "Cancel", onclick: () => done() })));
        },
      });
    } catch (e) { toast(e.message, { error: true }); }
  }

  function showError(msg) {
    alertHost.innerHTML = "";
    alertHost.appendChild(el("div", { class: "alert alert--error", role: "alert" }, Icon("warning", 20), el("div", { text: msg })));
  }

  store.subscribe(render, root);
  render();
  return root;
}
async function refreshConfig() {
  try { store.set({ config: await http.get("/config") }); } catch (e) { toast(e.message, { error: true }); }
}
