/* Files — workspace browser. Workspace picker (lists dirs carrying a
   .northstack/ledger.db), breadcrumb nav, collapsible tree (driven by
   /files/tree), monocode reader with line numbers, bounded read + truncation
   indicator, artifact list for a run. Path resolution is workspace-relative
   only; traversal outside the workspace is rejected server-side. */
import { el, toast, fmtInt, debounce, selectMenu } from "../util.js";
import { Icon } from "../icons.js";
import { http, store } from "../api.js";

export function filesView() {
  const root = el("div");
  let activeWs = "", curPath = "", tree = {}, selectedFile = null, fileContent = null;
  let workspaceRequest = 0, fileRequest = 0, artifactRequest = 0;
  const requests = { workspace: null, file: null, artifacts: null };
  const treeRequests = new Map();
  root.appendChild(el("div", { class: "page-head" },
    el("h1", { text: "Files" }),
    el("p", { text: "Browse workspaces the company has operated on. Paths are workspace-relative; reads are byte-bounded." }),
  ));

  // workspace picker
  const pickerRow = el("div", { class: "toolbar" });
  const wsSelect = selectMenu({
    options: [], placeholder: "Pick a workspace…", ariaLabel: "Workspace",
    onChange: (v) => selectWorkspace(v),
  });
  wsSelect.style.maxWidth = "420px";
  const refreshWs = el("button", { class: "btn btn--tonal btn--sm", onclick: loadWorkspaces, "aria-label": "Refresh workspaces" }, Icon("reload", 16), el("span", { text: "Refresh" }));
  pickerRow.appendChild(wsSelect);
  pickerRow.appendChild(refreshWs);
  pickerRow.appendChild(el("div", { class: "toolbar__spacer" }));
  const baseInp = el("input", { class: "field__input", type: "text", placeholder: "scan base dir (optional)", style: { maxWidth: "260px" }, "aria-label": "Base directory to scan" });
  pickerRow.appendChild(baseInp);
  root.appendChild(pickerRow);

  // breadcrumb
  const crumbs = el("nav", { class: "breadcrumb", "aria-label": "Path" });
  root.appendChild(crumbs);

  const split = el("div", { class: "files-split" });
  // tree pane
  const treePane = el("div", { class: "card", style: { padding: 0, overflow: "auto", maxHeight: "70vh" } });
  const treeRoot = el("div", { class: "tree", role: "tree" });
  treePane.appendChild(treeRoot);
  split.appendChild(treePane);
  // reader pane
  const reader = el("div", { class: "card", style: { padding: 0, overflow: "auto", maxHeight: "70vh" } });
  split.appendChild(reader);

  // artifacts panel
  const artHost = el("div", { class: "card", style: { marginTop: "var(--p-space-4)" } }, el("h3", { class: "section-title", text: "Run artifacts" }),
    el("div", { class: "muted", text: "Select a run id to list its stored artifact blobs." }));
  const artRow = el("div", { class: "row", style: { marginTop: "var(--p-space-2)" } });
  const runInp = el("input", { class: "field__input", placeholder: "run-xxxxxxxxxxxx", style: { maxWidth: "260px" }, "aria-label": "Run id for artifacts" });
  artRow.appendChild(runInp);
  artRow.appendChild(el("button", { class: "btn btn--tonal btn--sm", onclick: () => loadArtifacts(runInp.value) }, Icon("package", 16), el("span", { text: "List" })));
  artHost.appendChild(artRow);
  const artList = el("div", { class: "stack", style: { marginTop: "var(--p-space-2)" } });
  artHost.appendChild(artList);
  root.appendChild(split);
  root.appendChild(artHost);

  function selectWorkspace(workspace) {
    for (const request of treeRequests.values()) request.controller.abort();
    requests.file?.abort();
    requests.artifacts?.abort();
    activeWs = workspace;
    curPath = "";
    tree = {};
    selectedFile = fileContent = null;
    treeRequests.clear();
    fileRequest += 1;
    artifactRequest += 1;
    render();
    if (activeWs) loadTree(".");
  }

  function restart(key) {
    requests[key]?.abort();
    return requests[key] = new AbortController();
  }

  function render() {
    renderCrumbs();
    renderTree();
    renderReader();
  }

  function renderCrumbs() {
    crumbs.innerHTML = "";
    if (!activeWs) { crumbs.appendChild(el("span", { class: "muted", text: "No workspace selected." })); return; }
    const segs = curPath ? curPath.split("/").filter(Boolean) : [];
    crumbs.appendChild(crumb(activeWs.split(/[\\/]/).pop(), () => { curPath = ""; render(); loadTree("."); }));
    let acc = "";
    for (const s of segs) {
      acc = acc ? acc + "/" + s : s;
      crumbs.appendChild(el("span", { class: "muted", text: "/" }));
      const segAcc = acc;
      crumbs.appendChild(crumb(s, () => { curPath = segAcc; render(); loadTree(segAcc); }));
    }
  }
  function crumb(label, onclick) { return el("button", { class: "chip", onclick }, el("span", { text: label })); }

  function renderTree() {
    treeRoot.replaceChildren();
    if (!activeWs) { treeRoot.appendChild(el("div", { class: "empty" }, el("div", { text: "Pick a workspace above." }))); return; }
    const node = tree[curPath] || { loaded: false, children: [] };
    if (!node.loaded) { treeRoot.appendChild(el("div", { class: "skeleton", style: { height: "40px" } })); return; }
    if (node.children.length === 0) { treeRoot.appendChild(el("div", { class: "muted", text: "Empty directory." })); return; }
    for (const c of node.children) treeRoot.appendChild(treeRow(c));
  }
  function treeRow(c) {
    const isDir = c.type === "dir";
    const row = el("div", { class: "tree__row", role: "treeitem" });
    const parentPath = c._p ?? curPath;
    const childPath = joinTreePath(parentPath, c.name);
    if (isDir) {
      const expanded = tree[childPath]?.expanded || false;
      const toggle = el("button", { class: "icon-btn tree__twist", "aria-label": expanded ? "Collapse" : "Expand", "aria-expanded": expanded ? "true" : "false", onclick: () => { tree[childPath] = tree[childPath] || { loaded: false, children: [] }; tree[childPath].expanded = !expanded; if (expanded) { tree[childPath].expanded = false; } else loadTree(childPath); render(); } }, Icon(expanded ? "chevronDown" : "chevronRight", 16));
      row.appendChild(toggle);
      row.appendChild(el("span", { class: "tree__icon", onclick: () => { curPath = childPath; render(); loadTree(childPath); } }, Icon("folder", 18)));
      row.appendChild(el("span", { class: "tree__name", onclick: () => { curPath = childPath; render(); loadTree(childPath); } }, el("span", { text: c.name })));
      const kidHost = el("div", { class: "tree__children", style: { marginLeft: "var(--p-space-4)" } });
      if (expanded && tree[childPath]?.loaded) for (const k of tree[childPath].children) kidHost.appendChild(treeRow({ ...k, _p: childPath }));
      row.appendChild(kidHost);
    } else {
      row.appendChild(el("span", { class: "tree__twist", style: { width: "40px", display: "inline-block" } }));
      row.appendChild(el("span", { class: "tree__icon" }, Icon("file", 18)));
      const nameBtn = el("button", { class: "tree__name", "aria-label": `Read ${c.name}`, onclick: () => { selectedFile = childPath; loadFile(childPath); } }, el("span", { text: c.name }));
      if (selectedFile === childPath) nameBtn.classList.add("tree__name--active");
      row.appendChild(nameBtn);
    }
    return row;
  }

  function renderReader() {
    reader.replaceChildren();
    if (!selectedFile) { reader.appendChild(el("div", { class: "empty" }, Icon("file", 24), el("div", { text: "Select a file to read." }))); return; }
    if (fileContent === null) { reader.appendChild(el("div", { class: "skeleton", style: { height: "120px", margin: "var(--p-space-3)" } })); return; }
    const head = el("div", { class: "row", style: { padding: "var(--p-space-2) var(--p-space-3)", borderBottom: "1px solid var(--md-sys-color-outline-variant)" } });
    head.appendChild(el("code", { text: selectedFile }));
    if (fileContent.truncated) head.appendChild(el("span", { class: "badge badge--neutral", text: `truncated @ ${fmtInt(fileContent.total_bytes)} bytes` }));
    head.appendChild(el("div", { class: "toolbar__spacer" }));
    head.appendChild(el("button", { class: "icon-btn", "aria-label": "Copy", onclick: () => navigator.clipboard?.writeText(fileContent.content).then(() => toast("Copied")) }, Icon("copy", 16)));
    reader.appendChild(head);
    const lines = fileContent.content.split("\n");
    const body = el("pre", { class: "mono", style: { margin: 0, padding: "var(--p-space-3)", overflow: "auto" } });
    const code = el("code");
    for (let i = 0; i < lines.length; i++) {
      const ln = el("div", { class: "code__line" }, el("span", { class: "code__ln", text: String(i + 1) }), el("span", { class: "code__txt", text: lines[i] || " " }));
      code.appendChild(ln);
    }
    body.appendChild(code);
    reader.appendChild(body);
  }

  async function loadWorkspaces() {
    const token = ++workspaceRequest, request = restart("workspace");
    try {
      const q = baseInp.value ? `?base=${encodeURIComponent(baseInp.value)}` : "";
      const { workspaces } = await http.get(`/files/workspaces${q}`, request.signal);
      if (token !== workspaceRequest) return;
      const prev = activeWs;
      wsSelect.setOptions(
        workspaces.map((w) => ({ value: w.path, label: w.name })),
        workspaces.length ? "Pick a workspace…" : "No workspaces found under base",
      );
      if (prev && workspaces.some(w => w.path === prev)) wsSelect.value = prev;
      else if (prev) selectWorkspace("");
      if (!activeWs && workspaces.length === 1) { wsSelect.value = workspaces[0].path; selectWorkspace(wsSelect.value); }
      render();
    } catch (e) { if (e.name !== "AbortError" && token === workspaceRequest) toast(e.message, { error: true }); }
  }
  async function loadTree(path) {
    if (!activeWs) return;
    if (tree[path]?.loaded) { render(); return; }
    treeRequests.get(path)?.controller.abort();
    const workspace = activeWs, request = { token: Symbol(path), controller: new AbortController() };
    treeRequests.set(path, request);
    try {
      const { entries } = await http.get(`/files/tree?workspace=${encodeURIComponent(workspace)}&path=${encodeURIComponent(path)}`, request.controller.signal);
      if (workspace !== activeWs || treeRequests.get(path) !== request) return;
      tree[path] = { loaded: true, expanded: path !== ".", children: entries };
      render();
    } catch (e) { if (e.name !== "AbortError" && workspace === activeWs && treeRequests.get(path) === request) toast(e.message, { error: true }); }
  }
  async function loadFile(path) {
    const workspace = activeWs, token = ++fileRequest, request = restart("file");
    fileContent = null; renderReader();
    try {
      const content = await http.get(`/files/read?workspace=${encodeURIComponent(workspace)}&path=${encodeURIComponent(path)}`, request.signal);
      if (token !== fileRequest || workspace !== activeWs || path !== selectedFile) return;
      fileContent = content;
    } catch (e) { if (e.name === "AbortError" || token !== fileRequest || workspace !== activeWs || path !== selectedFile) return; toast(e.message, { error: true }); fileContent = { content: `// ${e.message}`, truncated: false, total_bytes: 0 }; }
    renderReader();
  }
  async function loadArtifacts(runId) {
    if (!runId) { toast("Enter a run id", { error: true }); return; }
    const workspace = activeWs, token = ++artifactRequest, request = restart("artifacts");
    artList.replaceChildren(el("div", { class: "skeleton", style: { height: "40px" } }));
    try {
      const q = workspace ? `?run_id=${encodeURIComponent(runId)}&workspace=${encodeURIComponent(workspace)}` : `?run_id=${encodeURIComponent(runId)}`;
      const { artifacts } = await http.get(`/files/artifacts${q}`, request.signal);
      if (token !== artifactRequest || workspace !== activeWs) return;
      artList.replaceChildren();
      if (artifacts.length === 0) { artList.appendChild(el("div", { class: "muted", text: "No artifacts stored for this run." })); return; }
      for (const a of artifacts) artList.appendChild(el("div", { class: "row" }, Icon("package", 16), el("code", { text: a.path }), el("span", { class: "muted", style: { marginLeft: "auto" }, text: `${fmtInt(a.size_bytes)} B` })));
    } catch (e) { if (e.name !== "AbortError" && token === artifactRequest && workspace === activeWs) artList.replaceChildren(el("div", { class: "alert alert--error" }, Icon("warning", 16), el("div", { text: e.message }))); }
  }

  // discover workspaces automatically (default base = server cwd)
  loadWorkspaces();
  render();
  root.dispose = () => {
    for (const request of Object.values(requests)) request?.abort();
    for (const { controller } of treeRequests.values()) controller.abort();
  };
  return root;
}

export function joinTreePath(parent, name) { return parent === "." ? name : `${parent}/${name}`; }
