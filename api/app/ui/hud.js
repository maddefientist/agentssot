/* Operator gateway HUD client.
   - WebSocket /gateway/ws  : command channel (stream tokens/events back)
   - SSE       /gateway/sse/status : ambient status panels
   The surface morphs ambient -> active on first command. */
(function () {
  "use strict";

  var sessionId =
    (window.crypto && crypto.randomUUID && crypto.randomUUID()) ||
    "s-" + Date.now();

  var $ = function (id) { return document.getElementById(id); };
  var body = document.body;
  var consoleEl = $("console");
  var ring = $("ring");
  var brainLive = $("brain-live");

  // ---- clock ----
  function tickClock() {
    var d = new Date();
    var hh = String(d.getHours()).padStart(2, "0");
    var mm = String(d.getMinutes()).padStart(2, "0");
    $("clock").textContent = hh + ":" + mm;
  }
  tickClock();
  setInterval(tickClock, 15000);

  // ---- console helpers ----
  function goActive() {
    if (!body.classList.contains("active")) body.classList.add("active");
  }
  function addTurn(who, cls) {
    var turn = document.createElement("div");
    turn.className = "turn";
    var head = document.createElement("div");
    head.className = "who";
    head.textContent = who;
    var said = document.createElement("div");
    said.className = "said " + cls;
    turn.appendChild(head);
    turn.appendChild(said);
    consoleEl.appendChild(turn);
    consoleEl.scrollTop = consoleEl.scrollHeight;
    return said;
  }
  function addNote(cls, text) {
    var n = document.createElement("div");
    n.className = cls;
    n.textContent = text;
    consoleEl.appendChild(n);
    consoleEl.scrollTop = consoleEl.scrollHeight;
    return n;
  }

  // ---- websocket command channel ----
  var ws = null;
  var current = null; // current assistant response element

  function apiKey() {
    return (window.cortexAuth && window.cortexAuth.getKey()) || localStorage.getItem("cortexKey") || "";
  }
  async function issueTicket(scope) {
    var key = apiKey();
    if (!key) throw new Error("Set an admin API key from the Classic dashboard first.");
    var response = await fetch("/gateway/tickets/" + scope, {
      method: "POST",
      headers: { "X-API-Key": key }
    });
    if (!response.ok) throw new Error("Gateway authorization failed (" + response.status + ").");
    return (await response.json()).ticket;
  }
  function wsURL() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + "/gateway/ws";
  }
  function setConn(state) {
    var el = $("conn");
    el.textContent = state;
    el.className = "conn " + (state === "online" ? "online" : state === "offline" ? "offline" : "");
  }
  async function connect() {
    var ticket;
    try {
      ticket = await issueTicket("websocket");
    } catch (err) {
      setConn("auth required");
      addNote("err", "⚠ " + err.message);
      return;
    }
    // Keep the one-time ticket out of URLs and reverse-proxy access logs.
    ws = new WebSocket(wsURL(), ["agentssot-ticket", ticket]);
    ws.onopen = function () { setConn("online"); };
    ws.onclose = function () {
      setConn("offline");
      ring.classList.remove("thinking");
      setTimeout(connect, 2500);
    };
    ws.onerror = function () { setConn("offline"); };
    ws.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      handleEvent(msg);
    };
  }

  function handleEvent(msg) {
    var type = msg.type, data = msg.data;
    if (type === "event") {
      if (data && data.routing) {
        brainLive.textContent = data.executor + " · " + data.intent;
        current = addTurn("Assistant", "assistant");
      } else if (data && data.fallover) {
        addNote("fallover", "↳ fell over to " + data.to);
      } else if (data && data.hive === "recall" && data.results) {
        data.results.slice(0, 5).forEach(function (r) {
          addNote("recall", "• " + (r.title || r.snippet || ""));
        });
      }
    } else if (type === "token") {
      if (!current) current = addTurn("Assistant", "assistant");
      current.textContent += data;
      consoleEl.scrollTop = consoleEl.scrollHeight;
    } else if (type === "error") {
      ring.classList.remove("thinking");
      addNote("err", "⚠ " + (data && data.message ? data.message : "error") +
        (data && data.retryable ? " (retryable)" : ""));
      current = null;
    } else if (type === "done") {
      ring.classList.remove("thinking");
      if (data && data.model) brainLive.textContent = "served by " + data.model;
      current = null;
    }
  }

  // ---- submit ----
  $("cmd-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var input = $("cmd");
    var text = input.value.trim();
    if (!text || !ws || ws.readyState !== 1) return;
    goActive();
    addTurn("You", "user").textContent = text;
    ring.classList.add("thinking");
    current = null;
    ws.send(JSON.stringify({ text: text, session_id: sessionId }));
    input.value = "";
  });

  // ---- SSE status panels ----
  function dot(id, ok) {
    var el = $(id);
    if (!el) return;
    el.className = "dot" + (ok === true ? " on" : ok === false ? "" : "");
  }
  function renderExecutors(list) {
    var box = $("exec-list");
    box.innerHTML = "";
    if (!list) return;
    var byName = {};
    list.forEach(function (x) {
      byName[x.name] = x.available;
      var row = document.createElement("div");
      row.className = "exec";
      var d = document.createElement("i");
      d.className = "dot" + (x.available ? " on" : "");
      var label = document.createElement("span");
      label.textContent = x.name;
      row.appendChild(d);
      row.appendChild(label);
      box.appendChild(row);
    });
    dot("dot-opus", byName["opus"]);
    dot("dot-deepseek", byName["deepseek-v4-pro"]);
  }
  function renderStatus(snap) {
      if (snap.hive) {
        var ki = snap.hive.knowledge_items;
        if (ki && typeof ki === "object") {
          if (ki.total != null) $("hive-count").textContent = ki.total;
          if (ki.embedded != null) $("hive-embedded").textContent = ki.embedded;
        } else if (ki != null) {
          $("hive-count").textContent = ki;
        }
      }
      renderExecutors(snap.executors);

      // Fleet slot: populated only when the fleet-dashboard (:9105) is reachable.
      if (snap.fleet) {
        var f = snap.fleet;
        var hosts = (f.hosts && f.hosts.length != null) ? f.hosts.length
                  : (f.online != null ? f.online
                  : (f.count != null ? f.count : null));
        if (hosts != null) {
          var total = (f.total != null) ? "/" + f.total : "";
          $("fleet").textContent = hosts + total;
        }
      }

      // Synapse slot: live agent activity from the synapse plane (DB-backed).
      var syn = snap.synapse;
      if (syn && syn.active != null) {
        if (syn.active > 0) {
          var where = syn.cwd ? (" · " + syn.cwd) : "";
          $("synapse-row").textContent = syn.active + " active" + where;
          $("dot-synapse").className = "dot on";
        } else {
          $("synapse-row").textContent = "idle";
          $("dot-synapse").className = "dot";
        }
      }
  }

  async function startSSE() {
    var ticket;
    try { ticket = await issueTicket("status"); }
    catch (err) { return; }
    try {
      // EventSource cannot attach a credential header. Fetch streaming can,
      // so the status ticket never appears in a URL.
      var response = await fetch("/gateway/sse/status", {
        headers: { "X-Gateway-Ticket": ticket },
        cache: "no-store"
      });
      if (!response.ok || !response.body) throw new Error("status stream unavailable");
      var reader = response.body.getReader();
      var decoder = new TextDecoder();
      var pending = "";
      while (true) {
        var part = await reader.read();
        if (part.done) break;
        pending += decoder.decode(part.value, { stream: true });
        var frames = pending.split("\n\n");
        pending = frames.pop();
        frames.forEach(function (frame) {
          var line = frame.split("\n").find(function (x) { return x.indexOf("data: ") === 0; });
          if (!line) return;
          try { renderStatus(JSON.parse(line.slice(6))); } catch (e) { /* next frame */ }
        });
      }
    } catch (err) {
      // Bounded or revoked streams reconnect with a freshly authorized ticket.
    } finally {
      setTimeout(startSSE, 2500);
    }
  }

  connect();
  startSSE();
})();
