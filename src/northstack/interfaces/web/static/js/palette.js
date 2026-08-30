/*
  Command palette (Ctrl/Cmd-K).  Fuzzy search across navigation, profiles,
  runs, settings, and the "new run" action.  Keyboard-only, with a visible
  result count and selected-row highlight.
*/
import { Icon } from "./icons.js";
import { el } from "./util.js";

let open = false;
let items = [];
let selected = 0;
let query = "";

export function initPalette() {
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      toggle();
    }
  });
}

function buildItems() {
  const out = [
    { group: "Actions", icon: "plus", label: "New run", run: () => go("#/runs") }, // new run form lives on runs page
    { group: "Pages", icon: "dashboard", label: "Dashboard", run: () => go("#/dashboard") },
    { group: "Pages", icon: "profiles", label: "Profiles", run: () => go("#/profiles") },
    { group: "Pages", icon: "routing", label: "Routing", run: () => go("#/routing") },
    { group: "Pages", icon: "commands", label: "Commands", run: () => go("#/commands") },
    { group: "Pages", icon: "runs", label: "Runs", run: () => go("#/runs") },
    { group: "Pages", icon: "files", label: "Files", run: () => go("#/files") },
    { group: "Pages", icon: "settings", label: "Settings", run: () => go("#/settings") },
  ];
  // profiles
  const cfg = window.__mc?.store?.state?.config;
  if (cfg) {
    for (const p of cfg.profiles) {
      out.push({ group: "Profiles", icon: "profiles", label: `Profile: ${p.name}`, sub: p.model, run: () => go("#/profiles") });
    }
  }
  // runs (recent)
  const runs = window.__mc?.store?.state?.runs || [];
  for (const r of runs.slice(0, 12)) {
    out.push({ group: "Runs", icon: "runs", label: `Run ${r.run_id.slice(0, 12)}`, sub: r.status, run: () => go(`#/runs/${r.run_id}`) });
  }
  return out;
}

function go(hash) {
  close();
  location.hash = hash;
}

export function toggle() {
  if (open) close();
  else openPalette();
}

function openPalette() {
  open = true;
  query = "";
  items = buildItems();
  selected = 0;

  const box = el("div", { class: "palette__box" });
  const searchWrap = el("div", { class: "palette__search" },
    el("span", {}, Icon("search")),
  );
  const input = el("input", { type: "search", placeholder: "Search pages, profiles, runs…", "aria-label": "Command palette search", autocomplete: "off" });
  searchWrap.appendChild(input);
  box.appendChild(searchWrap);
  const list = el("ul", { class: "palette__list", role: "listbox", "aria-label": "Command palette results" });
  box.appendChild(list);
  const count = el("div", { class: "palette__empty sr-only" });
  box.appendChild(count);

  const overlay = el("div", { class: "palette", role: "dialog", "aria-modal": "true", "aria-label": "Command palette" });
  overlay.appendChild(box);
  document.body.appendChild(overlay);

  function render() {
    const q = query.trim().toLowerCase();
    const matches = q ? items.filter((i) => fuzzy(i.label.toLowerCase(), q) || (i.sub && i.sub.toLowerCase().includes(q))) : items;
    list.innerHTML = "";
    if (matches.length === 0) {
      list.appendChild(el("li", { class: "palette__empty", text: "No matches" }));
      count.textContent = "0 results";
    } else {
      count.textContent = `${matches.length} result${matches.length === 1 ? "" : "s"}`;
      matches.forEach((m, i) => {
        const li = el("li", { class: "palette__item", role: "option", "aria-selected": i === selected ? "true" : "false", onclick: () => m.run(), onmouseenter: () => { selected = i; renderSel(); } },
          Icon(m.icon || "search"),
          el("span", { text: m.label }),
          m.sub ? el("span", { class: "group", text: m.sub }) : el("span", { class: "group", text: m.group }),
        );
        list.appendChild(li);
      });
    }
    // keep selected in view
    const sel = list.querySelector('[aria-selected="true"]');
    sel?.scrollIntoView({ block: "nearest" });
    window.__mcPalette = { matches, selected };
  }
  function renderSel() {
    list.querySelectorAll(".palette__item").forEach((li, i) => li.setAttribute("aria-selected", i === selected ? "true" : "false"));
    const sel = list.querySelector('[aria-selected="true"]');
    sel?.scrollIntoView({ block: "nearest" });
  }

  input.addEventListener("input", () => { query = input.value; selected = 0; render(); });
  input.addEventListener("keydown", (e) => {
    const m = window.__mcPalette?.matches || [];
    if (e.key === "ArrowDown") { e.preventDefault(); selected = Math.min(selected + 1, m.length - 1); renderSel(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); selected = Math.max(selected - 1, 0); renderSel(); }
    else if (e.key === "Enter") { e.preventDefault(); m[selected]?.run(); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
  });
  overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });

  function close() {
    open = false;
    overlay.remove();
    document.removeEventListener("keydown", onDocKey);
  }
  function onDocKey(e) { if (e.key === "Escape") close(); }
  document.addEventListener("keydown", onDocKey);

  window.__mcPaletteClose = close;
  render();
  setTimeout(() => input.focus(), 20);
}

/* simple subsequence fuzzy match */
function fuzzy(text, q) {
  let i = 0;
  for (const ch of q) {
    i = text.indexOf(ch, i);
    if (i === -1) return false;
    i++;
  }
  return true;
}
