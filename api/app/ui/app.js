const $ = (id) => document.getElementById(id);

// --- API helpers ---
function apiBase() {
  return (($("apiBase").value || "").trim() || window.location.origin).replace(/\/$/, "");
}
function apiKey() { return $("apiKey").value.trim(); }

async function api(path, { method = "GET", body } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey()) headers["X-API-Key"] = apiKey();
  const res = await fetch(`${apiBase()}${path}`, {
    method, headers,
    body: body ? JSON.stringify(body) : undefined
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = text; }
  if (!res.ok) throw new Error(`${res.status}: ${data?.detail || data}`);
  return data;
}

// --- State ---
let allItems = [];
let activeTag = null;

// --- Sanitizer: all dynamic content goes through this ---
function esc(s) {
  if (!s) return "";
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// --- DOM builder helpers (avoid raw innerHTML for dynamic content) ---
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs) Object.entries(attrs).forEach(([k, v]) => {
    if (k === "className") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v);
  });
  children.forEach(c => {
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else if (c) node.appendChild(c);
  });
  return node;
}

// --- Init ---
function loadSaved() {
  $("apiBase").value = localStorage.getItem("hive_base") || window.location.origin;
  $("apiKey").value = localStorage.getItem("hive_key") || "";
}

function saveCreds() {
  localStorage.setItem("hive_base", apiBase());
  localStorage.setItem("hive_key", apiKey());
}

// --- Tab switching ---
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    $(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// --- Settings drawer ---
$("settingsToggle").addEventListener("click", () => {
  $("settingsDrawer").classList.toggle("open");
});
$("saveConn").addEventListener("click", () => {
  saveCreds();
  $("settingsDrawer").classList.remove("open");
  checkHealth();
});

// --- Health check ---
async function checkHealth() {
  const dot = $("healthDot");
  try {
    const h = await api("/health", { method: "GET" });
    dot.className = "health-dot ok";
    dot.title = `OK | ${h.embedding_provider || "no-embed"} | ${h.total_records || "?"} records`;
    $("healthOutput").textContent = JSON.stringify(h, null, 2);
    return h;
  } catch (e) {
    dot.className = "health-dot err";
    dot.title = `Error: ${e.message}`;
    $("healthOutput").textContent = e.message;
    return null;
  }
}

// --- Render a single item card ---
function renderItem(item, container) {
  const tags = (item.tags || []);
  const title = item.title || item.kind || "item";
  const date = item.created_at ? new Date(item.created_at).toLocaleDateString() : "";
  const snippet = item.snippet || item.content || "";
  const score = item.score;

  const tagsEl = el("div", { className: "item-tags" },
    ...tags.map(t => el("span", null, t))
  );

  const headerRight = el("span", { className: "item-date" });
  if (score != null) {
    const badge = el("span", { className: "score-badge" }, score.toFixed(3));
    headerRight.appendChild(badge);
    headerRight.appendChild(document.createTextNode(" "));
  }
  headerRight.appendChild(document.createTextNode(date));

  const snippetEl = el("div", { className: "item-snippet" }, snippet);

  const card = el("div", { className: "item-card", onClick: () => card.classList.toggle("expanded") },
    el("div", { className: "item-header" },
      el("span", { className: "item-title" }, title),
      headerRight
    ),
    tagsEl,
    snippetEl
  );

  container.appendChild(card);
}

// --- Browse tab ---
async function loadBrowse() {
  const ns = $("browseNamespace").value.trim() || "default";
  const q = $("browseFilter").value.trim();
  const limit = $("browseLimit").value || 30;
  const params = new URLSearchParams({ namespace: ns, limit, q });
  try {
    const data = await api(`/query?${params}`);
    allItems = data.results || data.items || [];
    updateStats(allItems);
    updateTags(allItems);
    renderBrowseItems(allItems);
  } catch (e) {
    $("itemsGrid").textContent = "";
    $("itemsGrid").appendChild(el("p", { className: "hint" }, e.message));
  }
}

function updateStats(items) {
  const projects = new Set();
  const devices = new Set();
  const metaTags = new Set(["project-context", "legacy-memory", "entity-observation", "relation",
    "core-config", "cross-llm", "agent-identity", "infrastructure", "seed",
    "project", "context", "claude"]);
  items.forEach(i => {
    (i.tags || []).forEach(t => {
      if (t.startsWith("device-")) devices.add(t.replace("device-", ""));
      else if (!metaTags.has(t)) projects.add(t);
    });
  });

  $("statItems").textContent = "";
  const itemsStrong = el("strong", null, String(items.length));
  $("statItems").appendChild(itemsStrong);
  $("statItems").appendChild(document.createTextNode("items"));

  $("statProjects").textContent = "";
  const projStrong = el("strong", null, String(projects.size));
  $("statProjects").appendChild(projStrong);
  $("statProjects").appendChild(document.createTextNode("projects"));

  $("statDevices").textContent = "";
  const devStrong = el("strong", null, String(devices.size));
  $("statDevices").appendChild(devStrong);
  $("statDevices").appendChild(document.createTextNode("devices"));
}

function updateTags(items) {
  const counts = {};
  items.forEach(i => (i.tags || []).forEach(t => { counts[t] = (counts[t] || 0) + 1; }));
  const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 25);

  const row = $("tagsRow");
  row.textContent = "";
  sorted.forEach(([tag, n]) => {
    const chip = el("span", {
      className: `tag-chip${activeTag === tag ? " active" : ""}`,
      onClick: () => {
        activeTag = activeTag === tag ? null : tag;
        renderBrowseItems(allItems);
        updateTags(allItems);
      }
    }, `${tag} (${n})`);
    row.appendChild(chip);
  });
}

function renderBrowseItems(items) {
  const grid = $("itemsGrid");
  grid.textContent = "";
  let filtered = items;
  if (activeTag) filtered = items.filter(i => (i.tags || []).includes(activeTag));
  if (!filtered.length) {
    grid.appendChild(el("p", { className: "hint" }, "No items found"));
    return;
  }
  filtered.forEach(item => renderItem(item, grid));
}

$("browseBtn").addEventListener("click", loadBrowse);
$("browseFilter").addEventListener("keydown", e => { if (e.key === "Enter") loadBrowse(); });

// --- Search tab ---
async function doSearch() {
  const text = $("searchText").value.trim();
  if (!text) return;
  const ns = $("searchNamespace").value.trim() || "default";
  const topK = Number($("searchTopK").value) || 10;
  const grid = $("searchResults");
  grid.textContent = "";
  grid.appendChild(el("p", { className: "hint" }, "Searching..."));
  try {
    const data = await api("/recall", {
      method: "POST",
      body: { namespace: ns, scope: "knowledge", query_text: text, top_k: topK }
    });
    const items = data.items || [];
    grid.textContent = "";
    if (!items.length) {
      grid.appendChild(el("p", { className: "hint" }, "No results"));
      return;
    }
    items.forEach(item => renderItem(item, grid));
  } catch (e) {
    grid.textContent = "";
    grid.appendChild(el("p", { className: "hint" }, e.message));
  }
}

$("searchBtn").addEventListener("click", doSearch);
$("searchText").addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

// --- Admin ---
$("createNsBtn").addEventListener("click", async () => {
  try {
    const res = await api("/admin/namespaces", { method: "POST", body: { name: $("newNs").value.trim() } });
    $("adminOutput").textContent = JSON.stringify(res, null, 2);
  } catch (e) { $("adminOutput").textContent = e.message; }
});

$("createKeyBtn").addEventListener("click", async () => {
  try {
    const namespaces = $("keyNs").value.split(",").map(s => s.trim()).filter(Boolean);
    const res = await api("/admin/api-keys", {
      method: "POST",
      body: { name: $("keyName").value.trim(), role: $("keyRole").value, namespaces }
    });
    $("adminOutput").textContent = JSON.stringify(res, null, 2);
  } catch (e) { $("adminOutput").textContent = e.message; }
});

$("listKeysBtn").addEventListener("click", async () => {
  try {
    const res = await api("/admin/api-keys");
    $("keysOutput").textContent = JSON.stringify(res, null, 2);
  } catch (e) { $("keysOutput").textContent = e.message; }
});

$("ingestBtn").addEventListener("click", async () => {
  try {
    const body = JSON.parse($("ingestPayload").value);
    const res = await api("/ingest", { method: "POST", body });
    $("adminOutput").textContent = JSON.stringify(res, null, 2);
  } catch (e) { $("adminOutput").textContent = e.message; }
});

// --- Boot ---
loadSaved();
checkHealth();
