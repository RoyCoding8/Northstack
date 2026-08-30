/*
  Shared DOM helpers, toasts, dialogs, formatting.  No raw hex in JS — only
  token references via inline style where absolutely needed (kept to a minimum).
*/
import { Icon } from "./icons.js";

/* ---------- element builder ---------- */
export function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v;
    else if (k === "dataset") Object.assign(e.dataset, v);
    else if (k === "html") e.innerHTML = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else if (v === true) e.setAttribute(k, "");
    else if (v === false || v == null) { /* skip */ }
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else e.setAttribute(k, String(v));
  }
  for (const c of kids.flat()) {
    if (c == null || c === false) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

/* ---------- select menu (themed dropdown; native <select> popups are OS-drawn) ---------- */
export function selectMenu({ options: initialOptions, value = "", placeholder: initialPlaceholder = "Select…", ariaLabel, disabled = false, onChange }) {
  let options = initialOptions;
  let placeholder = initialPlaceholder;
  let open = false;
  const label = el("span", { class: "select-menu__label" });
  const btn = el("button", {
    type: "button", class: "select-menu__btn", "aria-haspopup": "listbox", "aria-expanded": "false",
    "aria-label": ariaLabel, disabled,
    onclick: (e) => { e.stopPropagation(); toggle(); },
    onkeydown: (e) => { if (!open && (e.key === "ArrowDown" || e.key === "Enter")) { e.preventDefault(); openMenu(); } },
  }, label, Icon("chevronDown", 18));
  const list = el("ul", { class: "select-menu__list", role: "listbox", "aria-label": ariaLabel });
  const root = el("div", { class: "select-menu" + (disabled ? " select-menu--disabled" : "") }, btn);

  function renderLabel() { label.textContent = options.find((o) => o.value === value)?.label ?? placeholder; }
  function renderList() {
    list.replaceChildren(...options.map((o, i) =>
      el("li", {
        class: "select-menu__opt" + (o.value === value ? " select-menu__opt--sel" : ""),
        role: "option", "aria-selected": o.value === value ? "true" : "false",
        onclick: () => pick(o.value),
        onmouseenter: () => highlight(i),
      }, el("span", { text: o.label }), o.value === value ? Icon("check", 16) : null),
    ));
  }
  function highlight(i) {
    list.querySelectorAll(".select-menu__opt").forEach((li, k) => li.classList.toggle("select-menu__opt--hover", k === i));
  }
  function setOptions(next, ph) { if (ph) placeholder = ph; options = next; renderLabel(); if (open) renderList(); }
  function toggle() {
    open ? close() : openMenu();
  }
  function openMenu() {
    if (disabled) return;
    open = true;
    btn.setAttribute("aria-expanded", "true");
    renderList();
    // fixed-position popup on <body> so dialogs' overflow:auto can't clip it
    const r = btn.getBoundingClientRect();
    list.style.left = `${Math.round(r.left)}px`;
    list.style.top = `${Math.round(r.bottom + 4)}px`;
    list.style.minWidth = `${Math.round(r.width)}px`;
    list.classList.add("select-menu__list--open");
    document.body.appendChild(list);
    document.addEventListener("click", onOutside, true);
    document.addEventListener("keydown", onKey);
  }
  function close() {
    open = false;
    btn.setAttribute("aria-expanded", "false");
    list.classList.remove("select-menu__list--open");
    list.remove();
    document.removeEventListener("click", onOutside, true);
    document.removeEventListener("keydown", onKey);
    btn.focus();
  }
  function pick(v) {
    value = v;
    renderLabel();
    onChange?.(v);
    close();
  }
  function onOutside(e) {
    if (root.contains(e.target) || list.contains(e.target)) return;
    // Capture phase, so a click on a dialog scrim closes this menu without also
    // reaching the scrim's own handler and closing the dialog behind it.
    if (e.target.classList?.contains("scrim")) e.stopPropagation();
    close();
  }
  function onKey(e) {
    const items = [...list.querySelectorAll(".select-menu__opt")];
    const i = items.findIndex((li) => li.classList.contains("select-menu__opt--hover"));
    if (e.key === "Escape") { e.preventDefault(); close(); }
    else if (e.key === "ArrowDown") { e.preventDefault(); highlight(Math.min(i + 1, items.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); highlight(Math.max(i - 1, 0)); }
    else if (e.key === "Enter" && i >= 0) { e.preventDefault(); pick(options[i].value); }
  }

  renderLabel();
  Object.defineProperty(root, "value", {
    get: () => value,
    set: (v) => { value = v; renderLabel(); },
  });
  root.setOptions = setOptions;
  return root;
}

/* ---------- toasts ---------- */
let stack;
function ensureStack() {
  if (!stack) {
    stack = el("div", { class: "snackbar-stack", role: "status", "aria-live": "polite" });
    document.body.appendChild(stack);
  }
  return stack;
}
export function toast(message, opts = {}) {
  const s = ensureStack();
  const node = el("div", { class: "snackbar" + (opts.error ? " snackbar--error" : "") });
  if (opts.icon) node.appendChild(Icon(opts.error ? "warning" : opts.icon, 18));
  node.appendChild(el("span", { text: message }));
  if (opts.action) {
    const b = el("button", { class: "snackbar__action", text: opts.action, onclick: () => { opts.actionHandler?.(); dismiss(); } });
    node.appendChild(b);
  }
  s.appendChild(node);
  let timer;
  const dismiss = () => {
    clearTimeout(timer);
    node.style.animation = "snackbar-out 150ms forwards";
    setTimeout(() => node.remove(), 160);
  };
  timer = setTimeout(dismiss, opts.duration ?? 4000);
  return dismiss;
}

/* ---------- dialogs (modal-from-source: JS scales from trigger) ---------- */
/** Open a dialog built by `builder(contentEl)`. Returns a close() handle. */
export function dialog({ title, builder, onClose, danger }) {
  const trigger = document.activeElement;
  const triggerRect = trigger?.getBoundingClientRect?.();
  const content = el("div", { class: "dialog" });
  if (title) content.appendChild(el("h2", { class: "dialog__title", id: "dlg-title", text: title }));
  const body = el("div");
  content.appendChild(body);

  const scrim = el("div", { class: "scrim", role: "dialog", "aria-modal": "true", "aria-labelledby": title ? "dlg-title" : undefined });
  scrim.appendChild(content);

  // Modal-from-source: if we know the trigger's box, start the dialog scaled
  // from it; otherwise fall back to centered scale-in (CSS default).
  if (triggerRect) {
    const cx = triggerRect.left + triggerRect.width / 2;
    const cy = triggerRect.top + triggerRect.height / 2;
    content.style.transformOrigin = `${cx}px ${cy}px`;
    // The scrim itself is full-screen; we animate content origin only.
  }
  document.body.appendChild(scrim);

  let firstFocus;
  const close = (result) => {
    document.removeEventListener("keydown", onKey);
    scrim.setAttribute("data-closing", "");
    content.setAttribute("data-closing", "");
    setTimeout(() => {
      scrim.remove();
      trigger?.focus?.();
      onClose?.(result);
    }, 160);
  };

  // close on scrim click (not on content)
  scrim.addEventListener("click", (e) => { if (e.target === scrim) close(); });
  // Esc closes
  const onKey = (e) => { if (e.key === "Escape") { e.preventDefault(); close(); } };
  document.addEventListener("keydown", onKey);

  document.body.appendChild(content); // temp append to measure
  content.remove();
  scrim.appendChild(content); // ensure inside scrim

  builder(body, close);
  // focus first field / first button
  firstFocus = content.querySelector("input, select, textarea, button") || content;
  setTimeout(() => firstFocus.focus(), 30);

  return close;
}

/* ---------- confirm helper ---------- */
export function confirmDialog({ title, body, confirmLabel = "Confirm", danger = true, onConfirm }) {
  const close = dialog({
    title,
    builder: (host, done) => {
      if (body) host.appendChild(el("p", { class: "card__sub", text: body }));
      host.appendChild(el("div", { class: "dialog__actions" },
        el("button", { class: "btn btn--tonal", text: "Cancel", onclick: () => done(false) }),
        el("button", { class: danger ? "btn btn--danger" : "btn btn--filled", text: confirmLabel, onclick: () => { onConfirm?.(); done(true); } }),
      ));
    },
  });
  return close;
}

/* ---------- formatting ---------- */
export function fmtNum(n) {
  if (n == null) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
}
export function fmtInt(n) {
  if (n == null) return "—";
  return Math.round(n).toLocaleString();
}
export function fmtUsd(n) {
  if (n == null) return "—";
  return "$" + Number(n).toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
}
export function fmtTime(ts) {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleString();
}
export function fmtTimeShort(ts) {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}
export function fmtSpan(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return [h && `${h}h`, (h || m) && `${m}m`, `${sec}s`].filter(Boolean).join(" ");
}
export function fmtElapsed(startTs) {
  if (!startTs) return "—";
  return fmtSpan(Date.now() / 1000 - startTs);
}
export function truncate(s, n = 300) {
  if (s == null) return "";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

/* ---------- copy to clipboard ---------- */
export async function copyText(text, label = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    toast(label);
  } catch {
    toast("Copy failed", { error: true });
  }
}

/* ---------- debounced ---------- */
export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* ---------- badge for a run status/outcome ---------- */
export function statusBadge(status, outcome) {
  const map = {
    verified: { cls: "badge--verified", icon: "verified", text: "Verified" },
    abstained: { cls: "badge--abstained", icon: "abstained", text: "Abstained" },
    failed: { cls: "badge--failed", icon: "failed", text: "Failed" },
    intake: { cls: "badge--active", icon: "active", text: "Intake" },
    contracted: { cls: "badge--active", icon: "active", text: "Contracted" },
    planned: { cls: "badge--active", icon: "active", text: "Planned" },
    executing: { cls: "badge--active", icon: "active", text: "Executing" },
    verifying: { cls: "badge--active", icon: "active", text: "Verifying" },
  };
  const key = outcome || status || "unknown";
  const m = map[key] || { cls: "badge--neutral", icon: "info", text: key || "Unknown" };
  const b = el("span", { class: "badge " + m.cls });
  b.appendChild(Icon(m.icon, 14));
  b.appendChild(el("span", { text: m.text }));
  return b;
}

/* ---------- event-filler <tbody> skeleton rows ---------- */
export function skeletonRows(cols, rows = 5) {
  const t = document.createDocumentFragment();
  for (let i = 0; i < rows; i++) {
    const tr = el("tr");
    for (let c = 0; c < cols; c++) {
      tr.appendChild(el("td", {}, el("div", { class: "skeleton", style: { height: "16px", maxWidth: "120px" } })));
    }
    t.appendChild(tr);
  }
  return t;
}
