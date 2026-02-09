const el = (id) => document.getElementById(id);

const output = el("output");

const defaultIngest = {
  namespace: "default",
  entities: [
    {
      slug: "project-alpha",
      type: "project",
      name: "Project Alpha",
      description: "Primary project record",
      metadata: { owner: "ops" }
    }
  ],
  requirements: [],
  knowledge_items: [
    {
      project_slug: "project-alpha",
      content: "Project Alpha is prioritized for Q1 rollout.",
      source: "dashboard",
      source_ref: "manual-seed",
      tags: ["seed"]
    }
  ],
  events: []
};

function apiBase() {
  return (el("apiBase").value || window.location.origin).replace(/\/$/, "");
}

function apiKey() {
  return el("apiKey").value.trim();
}

function print(value) {
  if (typeof value === "string") {
    output.textContent = value;
    return;
  }
  output.textContent = JSON.stringify(value, null, 2);
}

function parseJsonMaybe(text, fallback = undefined) {
  const clean = text.trim();
  if (!clean) return fallback;
  return JSON.parse(clean);
}

async function callApi(path, { method = "GET", body, useAuth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (useAuth && apiKey()) headers["X-API-Key"] = apiKey();

  const res = await fetch(`${apiBase()}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  });

  const text = await res.text();
  let parsed = text;
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    parsed = text;
  }

  if (!res.ok) {
    const detail = parsed && parsed.detail ? parsed.detail : parsed;
    throw new Error(`HTTP ${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }

  return parsed;
}

function loadSaved() {
  el("apiBase").value = localStorage.getItem("agentssot_api_base") || window.location.origin;
  el("apiKey").value = localStorage.getItem("agentssot_api_key") || "";
  el("ingestPayload").value = JSON.stringify(defaultIngest, null, 2);
}

el("saveConn").addEventListener("click", () => {
  localStorage.setItem("agentssot_api_base", apiBase());
  localStorage.setItem("agentssot_api_key", apiKey());
  print({ ok: true, message: "Connection settings saved locally." });
});

el("healthBtn").addEventListener("click", async () => {
  try {
    print(await callApi("/health", { useAuth: false }));
  } catch (err) {
    print(String(err));
  }
});

el("queryBtn").addEventListener("click", async () => {
  try {
    const params = new URLSearchParams({
      namespace: el("queryNamespace").value.trim() || "default",
      q: el("queryText").value,
      limit: String(el("queryLimit").value || 20)
    });

    const project = el("queryProject").value.trim();
    const entity = el("queryEntity").value.trim();
    if (project) params.set("project_slug", project);
    if (entity) params.set("entity_slug", entity);

    print(await callApi(`/query?${params.toString()}`));
  } catch (err) {
    print(String(err));
  }
});

el("recallBtn").addEventListener("click", async () => {
  try {
    const body = {
      namespace: el("recallNamespace").value.trim() || "default",
      scope: el("recallScope").value,
      query_text: el("recallText").value.trim() || null,
      query_embedding: parseJsonMaybe(el("recallEmbedding").value, null),
      top_k: Number(el("recallTopK").value || 5),
      project_slug: el("recallProject").value.trim() || null,
      entity_slug: el("recallEntity").value.trim() || null
    };

    print(await callApi("/recall", { method: "POST", body }));
  } catch (err) {
    print(String(err));
  }
});

el("ingestBtn").addEventListener("click", async () => {
  try {
    const body = parseJsonMaybe(el("ingestPayload").value);
    print(await callApi("/ingest", { method: "POST", body }));
  } catch (err) {
    print(String(err));
  }
});

el("sumBtn").addEventListener("click", async () => {
  try {
    const body = {
      namespace: el("sumNamespace").value.trim() || "default",
      session_id: el("sumSession").value.trim(),
      project_slug: el("sumProject").value.trim() || null,
      max_events: Number(el("sumMaxEvents").value || 500)
    };
    print(await callApi("/summarize_clear", { method: "POST", body }));
  } catch (err) {
    print(String(err));
  }
});

el("createNsBtn").addEventListener("click", async () => {
  try {
    const body = { name: el("newNamespace").value.trim() };
    print(await callApi("/admin/namespaces", { method: "POST", body }));
  } catch (err) {
    print(String(err));
  }
});

el("createKeyBtn").addEventListener("click", async () => {
  try {
    const namespaces = el("newKeyNamespaces")
      .value.split(",")
      .map((s) => s.trim())
      .filter(Boolean);

    const body = {
      name: el("newKeyName").value.trim(),
      role: el("newKeyRole").value,
      namespaces
    };
    print(await callApi("/admin/api-keys", { method: "POST", body }));
  } catch (err) {
    print(String(err));
  }
});

el("listKeysBtn").addEventListener("click", async () => {
  try {
    print(await callApi("/admin/api-keys"));
  } catch (err) {
    print(String(err));
  }
});

loadSaved();
