/*
  App shell + hash router for the northstack control surface.

  - Renders the top app bar (company name, unsaved pill, theme toggle, command
    palette trigger, active-runs badge) and the nav rail (desktop) / drawer
    (mobile) once; routes swap only #main.
  - Hash routing (#/dashboard, #/profiles, #/runs/:id, ...) so every screen is
    deep-linkable and back/forward is preserved.
  - Applies the persisted theme (system/light/dark) on boot; listens to
    prefers-color-scheme when system is selected.
  - Boots the polling manager and wires the mc:poll-rate custom event from
    the Settings view to live-adjust cadence.
  - Mounts the command palette (Ctrl/Cmd-K).
  No framework; ES modules. No raw hex here -- only token references.
*/
import { el, toast } from "./util.js";
import { Icon } from "./icons.js";
import { http, store, loadSettings, saveSettings } from "./api.js";
import { poll } from "./poll.js";
import { initPalette, toggle as togglePalette } from "./palette.js";
import { dashboardView } from "./views/dashboard.js";
import { profilesView } from "./views/profiles.js";
import { routingView } from "./views/routing.js";
import { commandsView } from "./views/commands.js";
import { runsView } from "./views/runs.js";
import { runDetailView } from "./views/run_detail.js";
import { filesView } from "./views/files.js";
import { settingsView } from "./views/settings.js";

const NAV = [
  { hash: "#/dashboard", icon: "dashboard", label: "Dashboard" },
  { hash: "#/profiles", icon: "profiles", label: "Profiles" },
  { hash: "#/routing", icon: "gitBranch", label: "Routing" },
  { hash: "#/commands", icon: "commands", label: "Commands" },
  { hash: "#/runs", icon: "runs", label: "Runs" },
  { hash: "#/files", icon: "folder", label: "Files" },
  { hash: "#/settings", icon: "settings", label: "Settings" },
];

function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "system") {
    const dark = matchMedia("(prefers-color-scheme: dark)").matches;
    root.setAttribute("data-theme", dark ? "dark" : "light");
  } else {
    root.setAttribute("data-theme", theme);
  }
}

let disposeView = null;

function route() {
  const hash = location.hash || "#/dashboard";
  const main = document.getElementById("main");
  if (!main) return;
  if (disposeView) disposeView();
  disposeView = null;
  // parse optional query like #/runs?rerun=1
  const [path] = hash.slice(1).split("?");
  main.setAttribute("aria-busy", "true");
  main.replaceChildren();
  let node;
  if (path === "/dashboard" || path === "/" || path === "") node = dashboardView();
  else if (path === "/profiles") node = profilesView();
  else if (path === "/routing") node = routingView();
  else if (path === "/commands") node = commandsView();
  else if (path === "/runs") node = runsView();
  else if (path.startsWith("/runs/")) {
    const runId = decodeURIComponent(path.slice("/runs/".length));
    node = runDetailView(runId);
  }
  else if (path === "/files") node = filesView();
  else if (path === "/settings") node = settingsView();
  else { node = notFound(hash); }
  main.appendChild(node);
  if (typeof node.dispose === "function") disposeView = () => node.dispose();
  main.setAttribute("aria-busy", "false");
  // focus main for SR users on route change
  main.setAttribute("tabindex", "-1");
  main.focus({ preventScroll: true });
  // update nav active state (highlight the matching top-level destination)
  const activeNav = hash.startsWith("#/runs") ? "#/runs" : hash;
  document.querySelectorAll("[data-nav]").forEach(a => {
    const isActive = a.getAttribute("data-nav") === activeNav;
    a.setAttribute("aria-current", isActive ? "page" : "false");
    a.classList.toggle("navrail__link--active", isActive);
    a.classList.toggle("drawer__link--active", isActive);
  });
  // close mobile drawer if open (rail is the drawer on mobile)
  closeDrawer();
}

function openDrawer() {
  setDrawer(true);
}
function closeDrawer() {
  setDrawer(false);
}
function setDrawer(open) {
  const app = document.getElementById("app");
  if (app) {
    if (open) app.setAttribute("data-drawer", "open");
    else app.removeAttribute("data-drawer");
  }
  const trigger = document.querySelector('[aria-controls="navrail"]');
  trigger?.setAttribute("aria-expanded", String(open));
  trigger?.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
}

function notFound(hash) {
  return el("div", { class: "empty" },
    Icon("warning", 24),
    el("div", { text: `No route for ${hash}` }),
    el("button", { class: "btn btn--tonal", style: { marginTop: "var(--p-space-3)" }, text: "Dashboard", onclick: () => location.hash = "#/dashboard" }),
  );
}

function buildShell() {
  // The shell root is #app — app.css keys the entire grid layout on this id.
  const shell = el("div", { id: "app" });

  // ---- app bar (spans full width, grid-column: 1 / -1) ----
  const appbar = el("header", { class: "appbar", role: "banner" });
  appbar.appendChild(el("button", { class: "icon-btn menu-btn", "aria-label": "Open navigation", "aria-expanded": "false", "aria-controls": "navrail", onclick: (e) => {
    const app = document.getElementById("app");
    const open = app?.getAttribute("data-drawer") === "open";
    if (open) closeDrawer(); else openDrawer();
  } }, Icon("menu")));
  const brand = el("div", { class: "appbar__brand" },
    Icon("dashboard", 22),
    el("span", { class: "appbar__name", text: "NorthStack" }),
  );
  appbar.appendChild(brand);
  appbar.appendChild(el("span", { class: "appbar__sub", text: "Control Surface" }));
  appbar.appendChild(el("span", { class: "appbar__spacer" }));

  const actions = el("div", { class: "appbar__actions" });
  const unsavedPill = el("button", { class: "unsaved-pill hidden", "aria-label": "Unsaved changes. Open Settings", onclick: () => location.hash = "#/settings" });
  actions.appendChild(unsavedPill);
  const activeBadge = el("span", { class: "badge badge--active active-badge hidden", "aria-live": "polite" });
  actions.appendChild(activeBadge);
  actions.appendChild(el("button", { class: "icon-btn", "aria-label": "Command palette (Ctrl K)", onclick: () => togglePalette() }, Icon("search", 20)));
  actions.appendChild(el("button", { class: "icon-btn", id: "themeBtn", "aria-label": "Toggle theme", onclick: () => {
    const cur = store.state.settings.theme || "system";
    const next = cur === "dark" ? "light" : "dark";
    saveSettings({ theme: next });
    applyTheme(next);
  } }, Icon("sunMoon", 20)));
  appbar.appendChild(actions);
  shell.appendChild(appbar);

  // ---- drawer scrim (mobile only; clicking it closes the drawer) ----
  const scrimEl = el("div", { class: "drawer-scrim", "aria-hidden": "true", onclick: closeDrawer });
  shell.appendChild(scrimEl);

  // ---- nav rail (desktop >= 1024px) — the same element becomes the
  //       off-canvas drawer on mobile (#app[data-drawer="open"]). No separate
  //       drawer nav: one rail, CSS transforms it per viewport. ----
  const rail = el("nav", { class: "navrail", id: "navrail", "aria-label": "Primary" });
  const railNav = el("div", { class: "navrail__nav" });
  for (const item of NAV) {
    railNav.appendChild(el("a", { class: "navrail__link", href: item.hash, "data-nav": item.hash, "aria-label": item.label, onclick: closeDrawer },
      el("span", { class: "navrail__icon" }, Icon(item.icon, 22)),
      el("span", { class: "navrail__text", text: item.label }),
    ));
  }
  rail.appendChild(railNav);

  // Operator card — the Figma sidebar's bottom user card, adapted to this
  // local control surface (no auth/logout here, so the card carries identity
  // + role tag + an inline theme toggle instead of a "Log out" action).
  const opCard = el("div", { class: "opcard", role: "group", "aria-label": "Operator" },
    el("div", { class: "opcard__avatar", "aria-hidden": "true" }, Icon("person", 22)),
    el("div", { class: "opcard__body" },
      el("div", { class: "opcard__name", text: "Operator" }),
      el("div", { class: "opcard__sub" },
        el("span", { class: "opcard__tag", text: "Admin" }),
        el("span", { class: "muted", text: "local" }),
      ),
    ),
    el("button", { class: "icon-btn opcard__theme", "aria-label": "Toggle theme", onclick: () => {
      const cur = store.state.settings.theme || "system";
      const next = cur === "dark" ? "light" : "dark";
      saveSettings({ theme: next });
      applyTheme(next);
    } }, Icon("sunMoon", 18)),
  );
  rail.appendChild(opCard);
  shell.appendChild(rail);

  const main = el("main", { id: "main", class: "page", tabindex: "-1" });
  shell.appendChild(main);

  // global toast stack is created lazily by util.toast; nothing to mount here.

  document.body.replaceChildren(shell);
  // close the drawer on large viewports / Escape
  matchMedia("(min-width: 1024px)").addEventListener?.("change", (e) => { if (e.matches) closeDrawer(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

  // live-update the unsaved pill + active badge from the store
  store.subscribe((s) => {
    const dirty = s.config?.unsaved;
    unsavedPill.classList.toggle("hidden", !dirty);
    unsavedPill.classList.toggle("unsaved-pill--dirty", !!dirty);
    unsavedPill.textContent = dirty ? "Unsaved changes" : "";
    const n = (s.activeRuns || []).length;
    activeBadge.classList.toggle("hidden", n === 0);
    activeBadge.textContent = n ? `${n} active` : "";
  });
}

async function boot() {
  // theme first to avoid flash
  applyTheme(store.state.settings.theme || "dark");
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
    if ((store.state.settings.theme || "dark") === "system") applyTheme("system");
  });
  buildShell();
  initPalette();

  // wire live poll-rate adjustments from Settings -- debounced, because the
  // sliders emit on every input event and each restart refetches immediately
  let rateTimer;
  document.addEventListener("mc:poll-rate", () => {
    clearTimeout(rateTimer);
    rateTimer = setTimeout(() => poll.start(), 250);
  });

  // initial config + start polling
  try { store.set({ config: await http.get("/config") }); } catch (e) { toast(`Could not load config: ${e.message}`, { error: true }); }
  poll.start();

  // routing
  window.addEventListener("hashchange", route);
  // open command palette on Ctrl/Cmd-K
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && (e.key === "k" || e.key === "K")) { e.preventDefault(); togglePalette(); }
  });
  // landing page if hash empty
  if (!location.hash) location.hash = (store.state.settings.landingPage || "#/dashboard").replace(/^#?\/?/, "#/");
  route();
}

boot();
