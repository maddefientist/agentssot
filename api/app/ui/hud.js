/* Madi HUD client.
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
  var current = null; // current madi response element

  function wsURL() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + "/gateway/ws";
  }
  function setConn(state) {
    var el = $("conn");
    el.textContent = state;
    el.className = "conn " + (state === "online" ? "online" : state === "offline" ? "offline" : "");
  }
  function connect() {
    ws = new WebSocket(wsURL());
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
        current = addTurn("Madi", "madi");
      } else if (data && data.fallover) {
        addNote("fallover", "↳ fell over to " + data.to);
      } else if (data && data.hive === "recall" && data.results) {
        data.results.slice(0, 5).forEach(function (r) {
          addNote("recall", "• " + (r.title || r.snippet || ""));
        });
      }
    } else if (type === "token") {
      if (!current) current = addTurn("Madi", "madi");
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
  function startSSE() {
    if (!window.EventSource) return;
    var es = new EventSource("/gateway/sse/status");
    es.onmessage = function (ev) {
      var snap;
      try { snap = JSON.parse(ev.data); } catch (e) { return; }
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
      $("dot-synapse").className = "dot on";
    };
    es.onerror = function () { /* EventSource auto-reconnects */ };
  }

  connect();
  startSSE();
})();
