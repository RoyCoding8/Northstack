/*
  Material Symbols Rounded icon set, rendered via font ligatures.

  One self-hosted variable woff2 (Material Symbols Rounded) covers the whole
  glyph set.  Icon(name) returns a <span class="icon"> whose text content is
  the glyph's ligature name (e.g. "play_arrow" -> play glyph).  Browsers with
  the font resolve the ligature into the pictograph automatically.

  No emoji anywhere in the UI (antislop: no-emoji-icons, icon-style-consistent).

  Usage: const icon = Icon("play");          // 20px (default)
         const icon = Icon("search", 18);

  The KEYS map below is the single place our internal icon names map to
  Material Symbols ligature names.  Add a row when a view needs a new glyph.
*/

// Internal name -> Material Symbols Rounded ligature name.
// (Material Symbols uses lowercase_underscore ligature names.)
const KEYS = {
  dashboard: "space_dashboard",
  profiles: "group",
  routing: "account_tree",
  commands: "terminal",
  runs: "play_circle",
  files: "folder",
  settings: "settings",
  play: "play_arrow",
  stop: "stop",
  plus: "add",
  search: "search",
  save: "save",
  reload: "refresh",
  reset: "restart_alt",
  check: "check",
  x: "close",
  copy: "content_copy",
  trash: "delete",
  edit: "edit",
  duplicate: "content_copy",
  test: "science",
  chevronRight: "chevron_right",
  chevronLeft: "chevron_left",
  chevronUp: "keyboard_arrow_up",
  chevronDown: "keyboard_arrow_down",
  arrowUp: "arrow_upward",
  arrowDown: "arrow_downward",
  arrowForward: "arrow_forward",
  trendingUp: "trending_up",
  person: "person",
  menu: "menu",
  sun: "light_mode",
  moon: "dark_mode",
  folder: "folder",
  file: "description",
  fileCode: "code",
  verified: "check_circle",
  failed: "cancel",
  abstained: "do_not_disturb_on",
  active: "play_circle",
  download: "download",
  upload: "upload",
  key: "vpn_key",
  link: "link",
  compare: "compare",
  sparkles: "auto_awesome",
  pulse: "monitoring",
  filter: "filter_alt",
  pause: "pause",
  info: "info",
  warning: "warning",
  shield: "shield",
  lock: "lock",
  clock: "schedule",
  list: "list",
  gitBranch: "merge",
  undo: "undo",
  eye: "visibility",
  shieldCheck: "verified_user",
  package: "inventory_2",
  sunMoon: "contrast",
  cpu: "memory",
  coin: "paid",
  fork: "fork_right",
};

export function Icon(name, size = 20) {
  const span = document.createElement("span");
  span.classList.add("icon");
  span.setAttribute("aria-hidden", "true");
  const ligature = KEYS[name];
  if (!ligature) {
    // unknown glyph — render the raw name so it is visible in dev (no crash)
    console.warn("unknown icon", name);
    span.textContent = name;
  } else {
    span.textContent = ligature;
  }
  // Font glyphs size by font-size, not width/height. The .icon base rule sets
  // the family + ligatures; per-call size is px font-size so the optical-size
  // axis (opsz) of the variable font tracks the chosen size crisply.
  span.style.fontSize = size + "px";
  span.style.width = size + "px";
  span.style.height = size + "px";
  // keep the opsz axis matched to the rendered size for sharp small glyphs
  span.style.fontVariationSettings = `"FILL" 0, "wght" 400, "GRAD" 0, "opsz" ${size}`;
  return span;
}

export function iconLabel(name, text, size = 20) {
  const span = document.createElement("span");
  span.style.display = "inline-flex";
  span.style.alignItems = "center";
  span.style.gap = "var(--p-space-1)";
  span.appendChild(Icon(name, size));
  if (text) {
    const t = document.createElement("span");
    t.textContent = text;
    span.appendChild(t);
  }
  return span;
}
