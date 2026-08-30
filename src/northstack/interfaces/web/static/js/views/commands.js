/* Commands — named subprocess profiles CRUD + test dry-run (not a sandbox). */
import { el, toast, confirmDialog, dialog, truncate, selectMenu } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";

export function commandsView() {
  const root = el("div");
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Command profiles" }),
    el("p", { text: "Named subprocess profiles exposed to workers as cmd_<name> tools. Test runs execute in a scratch directory, not a security sandbox." }),
  ));

  const tb = el("div", { class: "toolbar" });
  tb.appendChild(el("div", { class: "toolbar__spacer" }));
  tb.appendChild(el("button", { class: "btn btn--filled", onclick: () => openEditor(null), "aria-label": "Add command" }, Icon("plus"), el("span", { text: "Add command" })));
  root.appendChild(tb);

  const wrap = el("div", { class: "card", style: { padding: 0, overflow: "auto" } });
  root.appendChild(wrap);

  function render() {
    const cfg = store.state.config;
    const cmds = cfg?.commands || [];
    wrap.innerHTML = "";
    if (cmds.length === 0) {
      wrap.appendChild(el("div", { class: "empty" }, Icon("commands", 24), el("div", { text: "No command profiles." })));
      return;
    }
    const table = el("table", { class: "table" });
    table.appendChild(el("thead", {}, el("tr", {},
      el("th", { text: "Name" }), el("th", { text: "Argv" }), el("th", { text: "Timeout" }),
      el("th", { text: "Max bytes" }), el("th", { text: "Isolation" }),
      el("th", { text: "Env allowlist" }), el("th", { text: "" }),
    )));
    const body = el("tbody");
    for (const c of cmds) body.appendChild(rowTr(c));
    table.appendChild(body);
    wrap.appendChild(table);
  }

  function rowTr(c) {
    const tr = el("tr");
    tr.appendChild(el("td", { style: { fontWeight: 600 }, text: c.name }));
    tr.appendChild(el("td", { class: "mono", title: c.argv.join(" ") }, c.argv.join(" ")));
    tr.appendChild(el("td", { class: "num nowrap", text: `${c.timeout_seconds}s` }));
    tr.appendChild(el("td", { class: "num nowrap", text: String(c.max_output_bytes) }));
    tr.appendChild(el("td", { class: "nowrap", text: c.isolation || "host" }));
    tr.appendChild(el("td", {}, el("div", { class: "chip-list" }, ...(c.env_allowlist || []).map(a => el("span", { class: "chip" }, el("span", { text: a }))))));
    const acts = el("td", { class: "row", style: { gap: "2px" } });
    acts.appendChild(iconBtn("test", "Test run", () => testCommand(c)));
    acts.appendChild(iconBtn("edit", "Edit", () => openEditor(c)));
    acts.appendChild(iconBtn("trash", "Delete", () => del(c)));
    tr.appendChild(acts);
    return tr;
  }

  function openEditor(c) {
    const data = c
      ? { ...c, argv: [...c.argv], env_allowlist: [...(c.env_allowlist || [])], isolation: c.isolation || "host", docker_image: c.docker_image || "" }
      : { name: "", argv: [""], timeout_seconds: 10, max_output_bytes: 65536, env_allowlist: ["PATH"], isolation: "host", docker_image: "" };
    dialog({
      title: c ? `Edit “${c.name}”` : "Add command",
      builder: (host, done) => {
        const form = el("div", { class: "stack" });
        form.appendChild(field("Name", el("input", { class: "field__input", value: data.name, oninput: (e) => data.name = e.target.value })));
        const argvWrap = el("div", { class: "stack" });
        form.appendChild(el("div", { class: "field" }, el("label", { text: "Argv (one token per line)" }), argvWrap));
        function renderArgv() {
          argvWrap.innerHTML = "";
          data.argv.forEach((tok, i) => {
            argvWrap.appendChild(el("div", { class: "row" },
              el("input", { class: "field__input", value: tok, "aria-label": `argv[${i}]`, oninput: (e) => data.argv[i] = e.target.value }),
              el("button", { class: "icon-btn", "aria-label": "Remove token", onclick: () => { data.argv.splice(i, 1); renderArgv(); } }, Icon("x", 16)),
            ));
          });
          argvWrap.appendChild(el("button", { class: "btn btn--tonal btn--sm", onclick: () => { data.argv.push(""); renderArgv(); } }, Icon("plus", 16), el("span", { text: "Add token" })));
        }
        renderArgv();
        const grid = el("div", { class: "grid grid--3" });
        grid.appendChild(field("Timeout (s)", el("input", { class: "field__input", type: "number", value: data.timeout_seconds, oninput: (e) => data.timeout_seconds = parseFloat(e.target.value) || 10 })));
        grid.appendChild(field("Max output bytes", el("input", { class: "field__input", type: "number", value: data.max_output_bytes, oninput: (e) => data.max_output_bytes = parseInt(e.target.value, 10) || 65536 })));
        grid.appendChild(field("Env allowlist (comma-separated)", el("input", { class: "field__input", value: (data.env_allowlist || []).join(", "), oninput: (e) => data.env_allowlist = e.target.value.split(",").map(s => s.trim()).filter(Boolean) })));
        form.appendChild(grid);
        form.appendChild(field("Isolation", selectMenu({
          options: ["host", "docker"].map((iso) => ({ value: iso, label: iso })),
          value: data.isolation || "host", ariaLabel: "Isolation",
          onChange: (v) => data.isolation = v,
        })));
        form.appendChild(field("Docker image", el("input", { class: "field__input", value: data.docker_image, placeholder: "required when isolation is docker", oninput: (e) => data.docker_image = e.target.value })));
        host.appendChild(form);
        host.appendChild(el("div", { class: "dialog__actions" },
          el("button", { class: "btn btn--tonal", text: "Cancel", onclick: () => done() }),
          el("button", { class: "btn btn--filled", text: "Save", onclick: async () => {
            const argv = data.argv.filter(a => a !== "");
            if (!data.name || argv.length === 0) { toast("Name and at least one argv token required", { error: true }); return; }
            try {
              const body = { name: data.name, argv, timeout_seconds: Number(data.timeout_seconds), max_output_bytes: Number(data.max_output_bytes), env_allowlist: data.env_allowlist, isolation: data.isolation || "host", docker_image: data.docker_image || "" };
              if (c) await http.put(`/config/commands/${encodeURIComponent(c.name)}`, body);
              else await http.post("/config/commands", body);
              await refreshConfig();
              toast("Command saved");
              done();
            } catch (e) { toast(e.message, { error: true }); }
          } }),
        ));
      },
    });
  }

  async function testCommand(c) {
    dialog({
      title: `Test “${c.name}”`,
      builder: (host, done) => {
        host.appendChild(el("div", { class: "alert alert--info", role: "alert" }, Icon("info", 18), el("div", { text: "Runs in a scratch temp directory. Not a security sandbox. Bounded output." })));
        const out = el("div", { class: "card", style: { marginTop: "var(--p-space-3)" } }, el("div", { class: "skeleton", style: { height: "80px" } }));
        host.appendChild(out);
        http.post(`/config/commands/${encodeURIComponent(c.name)}/test`).then((r) => {
          out.innerHTML = "";
          out.appendChild(el("div", { class: "row" },
            el("span", { text: "Exit: " }), el("code", { text: String(r.exit_code) }),
            r.truncated ? el("span", { class: "badge badge--neutral", text: "truncated" }) : null,
            el("span", { class: "muted", text: "isolated: false" }),
          ));
          if (r.stdout) out.appendChild(el("pre", { class: "mono", style: { background: "var(--code-bg)", padding: "var(--p-space-3)" }, text: r.stdout }));
          if (r.stderr) out.appendChild(el("pre", { class: "mono", style: { color: "var(--md-sys-color-error)", padding: "var(--p-space-3)" }, text: r.stderr }));
        }).catch((e) => { out.innerHTML = ""; out.appendChild(el("div", { class: "alert alert--error" }, Icon("warning", 18), el("div", { text: e.message }))); });
        host.appendChild(el("div", { class: "dialog__actions" }, el("button", { class: "btn btn--tonal", text: "Close", onclick: () => done() })));
      },
    });
  }

  function del(c) {
    confirmDialog({ title: `Delete command “${c.name}”?`, confirmLabel: "Delete", onConfirm: async () => {
      try { await http.del(`/config/commands/${encodeURIComponent(c.name)}`); await refreshConfig(); toast("Deleted"); } catch (e) { toast(e.message, { error: true }); }
    } });
  }

  store.subscribe(render, root);
  render();
  return root;
}
function field(label, control) { return el("div", { class: "field" }, el("label", { text: label }), control); }
function iconBtn(icon, label, onclick) { const b = el("button", { class: "icon-btn state-layer", "aria-label": label, title: label, onclick }); b.appendChild(Icon(icon, 18)); return b; }
async function refreshConfig() { try { store.set({ config: await http.get("/config") }); } catch (e) { toast(e.message, { error: true }); } }
