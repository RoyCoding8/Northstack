/* Profiles — full CRUD table with derived tier badge, key-status dot, duplicate,
   modal-from-source editor, validate-on-blur, delete-with-confirm + undo. */
import { el, toast, confirmDialog, debounce, dialog, selectMenu } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";

const ROLES = ["worker", "reviewer", "planner", "specialist", "orchestrator"];
const CAPS = ["tool_use", "native_json_schema", "vision", "streaming"];
const PROTOS = ["openai_chat", "anthropic_messages", "gemini_generate_content"];
const TOKEN_LIMIT_PARAMS = ["max_tokens", "max_completion_tokens"];

export function profilesView() {
  let sortKey = "name";
  let sortDir = 1;
  let filterRole = null;
  let search = "";

  const root = el("div");
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Model profiles" }),
    el("p", { text: "Endpoints the company can route work to. API keys are referenced by environment variable name only; set the value in your environment or .env, never here." }),
  ));

  // toolbar
  const tb = el("div", { class: "toolbar" });
  tb.appendChild(filterChipRow());
  const searchBox = el("div", { class: "search" }, Icon("search"),
    (() => {
      const inp = el("input", { type: "search", placeholder: "Search profiles…", "aria-label": "Search profiles", oninput: debounce((e) => { search = e.target.value.toLowerCase(); rerender(); }, 200) });
      return inp;
    })(),
  );
  tb.appendChild(searchBox);
  tb.appendChild(el("div", { class: "toolbar__spacer" }));
  tb.appendChild(el("button", { class: "btn btn--filled", onclick: () => openEditor(null),
    "aria-label": "Add profile" }, Icon("plus"), el("span", { text: "Add profile" })));
  root.appendChild(tb);

  const tableWrap = el("div", { class: "card", style: { padding: 0, overflow: "auto" } });
  root.appendChild(tableWrap);

  function rerender() {
    const cfg = store.state.config;
    tableWrap.replaceChildren();
    if (!cfg) {
      tableWrap.appendChild(el("div", { class: "skeleton", style: { height: "160px" }, role: "status", "aria-label": "Loading profiles" }));
      return;
    }
    const profiles = cfg.profiles || [];
    let rows = profiles;
    if (filterRole) rows = rows.filter(p => p.roles.includes(filterRole));
    if (search) rows = rows.filter(p => p.name.toLowerCase().includes(search) || p.model.toLowerCase().includes(search));
    rows = [...rows].sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === "number") return (av - bv) * sortDir;
      return String(av).localeCompare(String(bv)) * sortDir;
    });

    if (rows.length === 0) {
      const message = profiles.length ? "No profiles match the current filters." : "No profiles. Add one to get started.";
      tableWrap.appendChild(el("div", { class: "empty" }, Icon("profiles", 24), el("div", { text: message })));
      return;
    }
    const table = el("table", { class: "table" });
    const head = el("thead", {}, el("tr", {},
      colHead("name", "Name"), colHead("protocol", "Protocol"), colHead("model", "Model"),
      colHead("tier", "Tier"), colHead("max_concurrency", "Conc"), colHead("roles", "Roles"),
      colHead("capabilities", "Caps"), colHead("key_status", "Key"), el("th", { text: "" }),
    ));
    table.appendChild(head);
    const body = el("tbody");
    for (const p of rows) body.appendChild(rowTr(p));
    table.appendChild(body);
    tableWrap.appendChild(table);
  }

  function colHead(key, label) {
    const th = el("th", { onclick: () => { if (sortKey === key) sortDir = -sortDir; else { sortKey = key; sortDir = 1; } rerender(); }, "aria-sort": sortKey === key ? (sortDir === 1 ? "ascending" : "descending") : "none" });
    th.textContent = label;
    return th;
  }

  function rowTr(p) {
    const tr = el("tr");
    tr.appendChild(el("td", { class: "nowrap", style: { fontWeight: 600 }, text: p.name }));
    tr.appendChild(el("td", { class: "mono nowrap", text: p.protocol }));
    tr.appendChild(el("td", { class: "mono", text: p.model }));
    tr.appendChild(el("td", {}, tierBadge(p.tier)));
    tr.appendChild(el("td", { class: "num", text: String(p.max_concurrency) }));
    tr.appendChild(el("td", {}, el("div", { class: "chip-list" }, ...p.roles.map(r => el("span", { class: "chip chip--role" }, el("span", { text: r }))))));
    tr.appendChild(el("td", {}, el("div", { class: "chip-list" }, ...p.capabilities.map(c => el("span", { class: "chip chip--cap" }, el("span", { text: c }))))));
    tr.appendChild(el("td", {}, keyDot(p.key_status, p.api_key_env)));
    const actions = el("td", { class: "row", style: { gap: "2px" } });
    actions.appendChild(iconBtn("edit", "Edit", () => openEditor(p)));
    actions.appendChild(iconBtn("duplicate", "Duplicate", () => duplicateProfile(p)));
    actions.appendChild(iconBtn("trash", "Delete", () => deleteProfile(p)));
    tr.appendChild(actions);
    return tr;
  }

  function deleteProfile(p) {
    const previousRouting = (store.state.config?.routing || []).map(entry => ({
      role: entry.role,
      profiles: [...entry.profiles],
    }));
    const routedRoles = previousRouting.filter(entry => entry.profiles.includes(p.name)).map(entry => entry.role);
    confirmDialog({
      title: `Delete profile “${p.name}”?`,
      body: routedRoles.length
        ? `This will also remove it from routing for: ${routedRoles.join(", ")}. You can undo once.`
        : "This removes the profile from the in-memory config. You can undo once.",
      confirmLabel: "Delete",
      onConfirm: async () => {
        try {
          await http.del(`/config/profiles/${encodeURIComponent(p.name)}?remove_from_routing=true`);
          await refreshConfig();
          toast(`Deleted “${p.name}”`, { icon: "trash", action: "Undo", actionHandler: async () => {
            try {
              await http.post("/config/profiles", profileToBody(p));
              if (routedRoles.length) await http.put("/config/routing", { routing: previousRouting });
              await refreshConfig();
              toast("Restored profile and routing");
            } catch (e) { toast(e.message, { error: true }); }
          } });
        } catch (e) { toast(e.message, { error: true }); }
      },
    });
  }

  async function duplicateProfile(p) {
    const close = openEditor({ ...p, name: p.name + "-copy" }, true);
  }

  function openEditor(p, isDup) {
    const data = p ? { ...p } : blankProfile();
    dialog({
      title: p && !isDup ? `Edit “${p.name}”` : isDup ? "Duplicate profile" : "Add profile",
      builder: (host, done) => {
        const form = el("div", { class: "stack" });
        form.appendChild(field("Name", el("input", { class: "field__input", value: data.name, required: true, oninput: (e) => data.name = e.target.value })));
        form.appendChild(field("Protocol", selectMenu({
          options: PROTOS.map((p) => ({ value: p, label: p })),
          value: data.protocol, ariaLabel: "Protocol",
          onChange: (v) => data.protocol = v,
        })));
        form.appendChild(field("Base URL", el("input", { class: "field__input", value: data.base_url, oninput: (e) => data.base_url = e.target.value })));
        form.appendChild(field("Model", el("input", { class: "field__input", value: data.model, oninput: (e) => data.model = e.target.value })));
        form.appendChild(field("API key env var (name only)", el("input", { class: "field__input", value: data.api_key_env || "", placeholder: "e.g. OPENAI_API_KEY", oninput: (e) => data.api_key_env = e.target.value || null }),
          "Set the value in your environment or .env, never here."));
        const insecureToggle = el("input", {
          type: "checkbox",
          checked: data.allow_insecure_http === true,
          onchange: (e) => data.allow_insecure_http = e.target.checked,
        });
        form.appendChild(field(
          "Transport security",
          el("label", { class: "switch" }, insecureToggle, el("span", { text: "Allow credentialed HTTP to a trusted non-loopback proxy" })),
          "Leave off unless you explicitly trust the plaintext network path.",
        ));
        form.appendChild(field("Roles", chipsToggle(ROLES, data.roles, (v, on) => { if (on) data.roles = [...new Set([...data.roles, v])]; else data.roles = data.roles.filter(r => r !== v); })));
        form.appendChild(field("Capabilities", chipsToggle(CAPS, data.capabilities, (v, on) => { if (on) data.capabilities = [...new Set([...data.capabilities, v])]; else data.capabilities = data.capabilities.filter(c => c !== v); })));
        const grid = el("div", { class: "grid grid--3" });
        grid.appendChild(field("Max concurrency", numInput(() => data.max_concurrency, (v) => data.max_concurrency = v, 1)));
        grid.appendChild(field("Requests/min", numInput(() => data.requests_per_minute, (v) => data.requests_per_minute = v, 60)));
        grid.appendChild(field("Context window", numInput(() => data.context_window_tokens, (v) => data.context_window_tokens = v, 128000)));
        grid.appendChild(field("Max output tokens", numInput(() => data.max_output_tokens, (v) => data.max_output_tokens = v, 4096)));
        grid.appendChild(field("Input $/M", numInput(() => data.input_price_per_million_usd, (v) => data.input_price_per_million_usd = v, 0, true)));
        grid.appendChild(field("Output $/M", numInput(() => data.output_price_per_million_usd, (v) => data.output_price_per_million_usd = v, 0, true)));
        grid.appendChild(field("Request timeout (s)", numInput(() => data.request_timeout_seconds, (v) => data.request_timeout_seconds = v, 300, true)));
        grid.appendChild(field("Transport retries", numInput(() => data.transport_retries, (v) => data.transport_retries = v, 2)));
        form.appendChild(grid);
        form.appendChild(field("Retry backoff (seconds)", el("input", { class: "field__input", value: (data.transport_retry_backoff_seconds || []).join(", "), oninput: (e) => data.transport_retry_backoff_seconds = e.target.value.split(",").map(Number).filter(Number.isFinite) })));
        form.appendChild(field("Stream completion", el("label", { class: "switch" }, el("input", { type: "checkbox", checked: data.strict_stream_completion !== false, onchange: (e) => data.strict_stream_completion = e.target.checked }), el("span", { text: "Require a protocol terminal frame" }))));
        form.appendChild(field("Auth header override", el("input", { class: "field__input", value: data.auth_header || "", placeholder: "e.g. api-key", oninput: (e) => data.auth_header = e.target.value || null }),
          "Blank uses the protocol default. Azure OpenAI wants api-key; the key is sent raw with no scheme."));
        form.appendChild(field("Extra headers", el("textarea", { class: "field__input", rows: 2, placeholder: "X-Title: NorthStack", text: kvText(data.extra_headers), oninput: (e) => data.extra_headers = kvParse(e.target.value) }),
          "One “Name: value” per line. Credentials are rejected; use the API key env var."));
        form.appendChild(field("Extra query params", el("textarea", { class: "field__input", rows: 2, placeholder: "api-version: 2024-10-21", text: kvText(data.extra_query), oninput: (e) => data.extra_query = kvParse(e.target.value) }),
          "One “name: value” per line, appended to every request URL."));
        form.appendChild(field("Output-token param", selectMenu({
          options: TOKEN_LIMIT_PARAMS.map((t) => ({ value: t, label: t })),
          value: data.token_limit_param || "max_tokens", ariaLabel: "Output-token param",
          onChange: (v) => data.token_limit_param = v,
        }),
          "openai_chat only. OpenAI’s own reasoning models reject max_tokens."));
        host.appendChild(form);
        host.appendChild(el("div", { class: "dialog__actions" },
          el("button", { class: "btn btn--tonal", text: "Cancel", onclick: () => done() }),
          el("button", { class: "btn btn--filled", text: "Save", onclick: async () => {
            try {
              const body = profileToBody(data);
              if (p && !isDup) await http.put(`/config/profiles/${encodeURIComponent(p.name)}`, body);
              else await http.post("/config/profiles", body);
              await refreshConfig();
              toast("Profile saved");
              done();
            } catch (e) { toast(e.message, { error: true }); }
          } }),
        ));
      },
    });
  }

  function filterChipRow() {
    const row = el("div", { class: "row" });
    ROLES.forEach(r => row.appendChild(el("button", {
      class: "chip chip--filter",
      "aria-pressed": filterRole === r ? "true" : "false",
      text: r,
      onclick: () => {
        filterRole = filterRole === r ? null : r;
        row.querySelectorAll(".chip--filter").forEach((c, i) => c.setAttribute("aria-pressed", ROLES[i] === filterRole ? "true" : "false"));
        rerender();
      },
    })));
    return row;
  }

  store.subscribe(rerender, root);
  if (!store.state.secrets) fetchSecrets();
  rerender();
  return root;
}

function chipsToggle(options, selected, onToggle) {
  const wrap = el("div", { class: "row" });
  for (const o of options) {
    const on = selected.includes(o);
    const c = el("button", { class: "chip chip--filter", "aria-pressed": on ? "true" : "false", text: o, onclick: (e) => { const now = e.currentTarget.getAttribute("aria-pressed") === "true"; e.currentTarget.setAttribute("aria-pressed", now ? "false" : "true"); onToggle(o, !now); } });
    wrap.appendChild(c);
  }
  return wrap;
}
function numInput(getter, setter, def, isFloat) {
  const inp = el("input", { class: "field__input", type: "number", value: String(getter() ?? def), oninput: (e) => { const v = isFloat ? parseFloat(e.target.value) : parseInt(e.target.value, 10); setter(isNaN(v) ? def : v); } });
  return inp;
}
function field(label, control, helper) {
  const f = el("div", { class: "field" });
  f.appendChild(el("label", { text: label }));
  f.appendChild(control);
  if (helper) f.appendChild(el("div", { class: "field__helper", text: helper }));
  return f;
}
function iconBtn(icon, label, onclick) {
  const b = el("button", { class: "icon-btn state-layer", "aria-label": label, title: label, onclick });
  b.appendChild(Icon(icon, 18));
  return b;
}
function tierBadge(tier) {
  const b = el("span", { class: "badge badge--neutral" });
  b.appendChild(el("span", { text: String(tier || "—") }));
  return b;
}
function keyDot(status, name) {
  const none = !name;
  const ok = status && status.includes("OK");
  const dotCls = none ? "none" : ok ? "ok" : "unset";
  const row = el("span", { class: "row", style: { gap: "6px" } });
  row.appendChild(el("span", { class: `badge badge--dot ${dotCls}`, "aria-hidden": "true" }));
  row.appendChild(el("code", { style: { fontSize: "12px" }, text: name || "—" }));
  return row;
}
function kvText(obj) {
  return Object.entries(obj || {}).map(([k, v]) => `${k}: ${v}`).join("\n");
}
function kvParse(text) {
  const out = {};
  for (const line of String(text || "").split("\n")) {
    const i = line.indexOf(":");
    if (i > 0 && line.slice(0, i).trim()) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}
function blankProfile() {
  return { name: "", protocol: "openai_chat", base_url: "", model: "", api_key_env: null, allow_insecure_http: false, roles: [], capabilities: [], max_concurrency: 1, requests_per_minute: 60, context_window_tokens: 128000, max_output_tokens: 4096, request_timeout_seconds: 300, strict_stream_completion: true, transport_retries: 2, transport_retry_backoff_seconds: [1.5, 6], input_price_per_million_usd: 0, output_price_per_million_usd: 0, auth_header: null, extra_headers: {}, extra_query: {}, token_limit_param: "max_tokens", tier: null };
}
function finiteNumber(value, fallback) { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
function profileToBody(p) {
  return {
    name: p.name, protocol: p.protocol, base_url: p.base_url, model: p.model,
    api_key_env: p.api_key_env || null, allow_insecure_http: p.allow_insecure_http === true,
    roles: p.roles || [], capabilities: p.capabilities || [],
    max_concurrency: finiteNumber(p.max_concurrency, 1), requests_per_minute: finiteNumber(p.requests_per_minute, 60),
    context_window_tokens: finiteNumber(p.context_window_tokens, 128000), max_output_tokens: finiteNumber(p.max_output_tokens, 4096),
    request_timeout_seconds: finiteNumber(p.request_timeout_seconds, 300), strict_stream_completion: p.strict_stream_completion !== false,
    transport_retries: finiteNumber(p.transport_retries, 2), transport_retry_backoff_seconds: p.transport_retry_backoff_seconds || [1.5, 6],
    input_price_per_million_usd: finiteNumber(p.input_price_per_million_usd, 0), output_price_per_million_usd: finiteNumber(p.output_price_per_million_usd, 0),
    auth_header: p.auth_header || null, extra_headers: p.extra_headers || {}, extra_query: p.extra_query || {},
    token_limit_param: p.token_limit_param || "max_tokens",
  };
}
async function refreshConfig() {
  try { const c = await http.get("/config"); store.set({ config: c }); await fetchSecrets(); } catch (e) { toast(e.message, { error: true }); }
}
async function fetchSecrets() {
  try { const s = await http.get("/secrets/status"); store.set({ secrets: s }); } catch { /* maybe no profiles yet */ }
}
