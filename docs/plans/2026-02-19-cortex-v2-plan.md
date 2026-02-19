# Cortex Visualizer v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the cortex visualizer into a live, self-updating neural dashboard with system management in a slide-out drawer.

**Architecture:** Single self-contained HTML file (`api/app/ui/cortex.html`) with two new read-only API endpoints (`/cortex/system-info`, `/cortex/activity`) in `api/app/main.py`. No build step, no external deps. Live polling every 10s with diff-based node animations.

**Tech Stack:** Vanilla JS + Canvas (existing), FastAPI endpoints, SQLAlchemy queries

**Security note:** All drawer content is rendered from trusted API responses on a LAN-only, read-only system. innerHTML is used for structured rendering of our own API data (not user input). No external/untrusted content enters the rendering path.

---

## Context

- The existing cortex visualizer is at `api/app/ui/cortex.html` (~390 lines)
- It fetches `/cortex/data` once on load and renders a canvas neural map
- Concepts have: type (principle/mental_model/relationship/skill), scope (global/project/device), confidence, evidence_ids, confirming_agents, trigger/action/success_hint (skills)
- The API runs at `http://YOUR_HOST:8088`, cortex endpoints are public (no auth)
- Settings are in `api/app/settings.py` (Settings class, Pydantic)
- Agent profiles are in `agent_profiles` table, recall events in `recall_events`, feedback in `concept_feedback`
- UI changes require container rebuild: `docker compose up -d --build api`

---

### Task 1: `/cortex/system-info` endpoint

Add a public read-only endpoint that returns system config, health, and agent profiles.

**Files:**
- Modify: `api/app/main.py:146-168` (add endpoint after existing cortex routes)

**Step 1: Add the endpoint**

Insert after the existing `cortex_data` function (after line 168) in `api/app/main.py`:

```python
@app.get("/cortex/system-info", include_in_schema=False)
def cortex_system_info(
    namespace: str = Query(default="claude-shared"),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex drawer. Returns config, health, agents."""
    settings = app.state.settings

    # Health
    health = {
        "status": "ok",
        "embedding_available": app.state.embedding_provider.is_available,
        "llm_available": app.state.llm_provider.is_available,
        "reranker_available": app.state.reranker_provider.is_available,
        "synthesis_enabled": settings.effective_synthesis_enabled,
    }

    # Config (safe subset — no secrets)
    config = {
        "namespace": namespace,
        "embedding_model": settings.ollama_embed_model,
        "embedding_dim": settings.embedding_dim,
        "llm_model": settings.ollama_chat_model,
        "reranker_model": settings.ollama_reranker_model,
        "synthesis_model": settings.synthesis_model,
        "synthesis_schedule_hour": settings.synthesis_schedule_hour,
        "synthesis_similarity_threshold": settings.synthesis_similarity_threshold,
        "synthesis_min_cluster_size": settings.synthesis_min_cluster_size,
        "synthesis_confidence_decay": settings.synthesis_confidence_decay,
        "synthesis_decay_floor": settings.synthesis_decay_floor,
        "synthesis_decay_grace_days": settings.synthesis_decay_grace_days,
    }

    # Agent profiles
    from .models import AgentProfile
    profiles = session.scalars(
        select(AgentProfile).where(AgentProfile.namespace == namespace)
    ).all()
    agents = [
        {
            "agent_key": p.agent_key,
            "device_name": p.device_name,
            "strengths": list(p.strengths or []),
            "total_recalls": p.total_recalls,
            "total_feedback": p.total_feedback,
            "updated_at": p.updated_at.isoformat() if p.updated_at else None,
        }
        for p in profiles
    ]

    # Namespaces
    from .models import Namespace as NsModel
    ns_names = [ns.name for ns in session.scalars(select(NsModel)).all()]

    return {
        "health": health,
        "config": config,
        "agents": agents,
        "namespaces": ns_names,
    }
```

**Step 2: Verify endpoint works**

Run: `docker compose up -d --build api && sleep 3 && curl -s http://YOUR_HOST:8088/cortex/system-info | python3 -m json.tool`

Expected: JSON with health, config, agents, namespaces keys.

**Step 3: Commit**

```bash
git add api/app/main.py
git commit -m "feat: add /cortex/system-info endpoint for drawer panel"
```

---

### Task 2: `/cortex/activity` endpoint

Add a public read-only endpoint that returns recent system events (recalls, feedback) for the activity ticker.

**Files:**
- Modify: `api/app/main.py` (add endpoint after system-info)

**Step 1: Add the endpoint**

Insert after the `cortex_system_info` function:

```python
@app.get("/cortex/activity", include_in_schema=False)
def cortex_activity(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=50, le=100),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex activity ticker. Returns recent events."""
    from .models import RecallEvent, ConceptFeedback, Concept

    events = []

    # Recent recalls
    recalls = session.execute(
        select(RecallEvent.agent_key, RecallEvent.query_text, RecallEvent.created_at, Concept.title)
        .join(Concept, RecallEvent.concept_id == Concept.id)
        .where(RecallEvent.namespace == namespace)
        .order_by(RecallEvent.created_at.desc())
        .limit(limit)
    ).all()
    for r in recalls:
        events.append({
            "type": "recall",
            "agent": r.agent_key,
            "detail": (r.query_text or "")[:80],
            "concept": r.title,
            "at": r.created_at.isoformat(),
        })

    # Recent feedback
    feedbacks = session.execute(
        select(ConceptFeedback.agent_key, ConceptFeedback.signal, ConceptFeedback.created_at, Concept.title)
        .join(Concept, ConceptFeedback.concept_id == Concept.id)
        .where(ConceptFeedback.namespace == namespace)
        .order_by(ConceptFeedback.created_at.desc())
        .limit(limit)
    ).all()
    for f in feedbacks:
        events.append({
            "type": "feedback",
            "agent": f.agent_key,
            "detail": f.signal.value if hasattr(f.signal, 'value') else str(f.signal),
            "concept": f.title,
            "at": f.created_at.isoformat(),
        })

    # Sort combined by timestamp descending, take limit
    events.sort(key=lambda e: e["at"], reverse=True)
    return {"events": events[:limit]}
```

**Step 2: Verify endpoint works**

Run: `curl -s http://YOUR_HOST:8088/cortex/activity | python3 -m json.tool`

Expected: JSON with events array, each having type/agent/detail/concept/at.

**Step 3: Commit**

```bash
git add api/app/main.py
git commit -m "feat: add /cortex/activity endpoint for live event ticker"
```

---

### Task 3: Live polling with diff-based node animation

Replace the single `loadData()` call with a 10-second polling loop that diffs incoming concepts and animates new nodes in with a birth effect.

**Files:**
- Modify: `api/app/ui/cortex.html` (JS section)

**Step 1: Replace `loadData` and add diffing logic**

Replace the existing `loadData` function and the final `loadData().then(...)` call with:

```javascript
let conceptMap = {};  // id -> concept, for diffing
let POLL_INTERVAL = 10000;

async function loadData() {
  try {
    const resp = await fetch('/cortex/data?namespace=claude-shared');
    const data = await resp.json();
    const incoming = data.concepts || [];
    document.getElementById('s-know').textContent = data.knowledge_count || '?';

    // Diff: find new, updated, removed
    const incomingMap = {};
    const newConcepts = [];
    for (const c of incoming) {
      incomingMap[c.id] = c;
      if (!conceptMap[c.id]) {
        newConcepts.push(c);
      }
    }

    // Update existing nodes with fresh data
    for (const n of nodes) {
      const fresh = incomingMap[n.concept.id];
      if (fresh) {
        const confidenceChanged = Math.abs((fresh.confidence || 0) - (n.concept.confidence || 0)) > 0.01;
        const evidenceChanged = (fresh.evidence_ids || []).length !== (n.concept.evidence_ids || []).length;
        n.concept = fresh;
        n.opacity = 0.3 + (fresh.confidence || 0.5) * 0.7;
        if (confidenceChanged || evidenceChanged) {
          n.flash = 1.0;  // trigger highlight flash
        }
      }
    }

    // Remove nodes for deleted concepts
    const removedIds = new Set();
    for (const id of Object.keys(conceptMap)) {
      if (!incomingMap[id]) removedIds.add(id);
    }
    if (removedIds.size > 0) {
      nodes = nodes.filter(n => !removedIds.has(n.concept.id));
    }

    // Add new nodes with birth animation
    for (const c of newConcepts) {
      const scopeRing = SCOPE_RINGS[c.scope] || 0.5;
      const angle = Math.random() * Math.PI * 2;
      const radiusJitter = (Math.random() - 0.5) * 0.15;
      const col = TYPE_COLORS[c.type] || { r: 100, g: 100, b: 100 };
      const evidenceCount = c.evidence_ids ? c.evidence_ids.length : 1;
      const baseSize = Math.max(3, Math.min(18, 3 + Math.sqrt(evidenceCount) * 2.5));

      nodes.push({
        concept: c,
        angle, radius: scopeRing + radiusJitter, baseSize,
        size: 0,  // start at 0 for birth animation
        targetSize: baseSize,
        color: col,
        opacity: 0.3 + (c.confidence || 0.5) * 0.7,
        x: W / 2, y: H / 2,  // start from center
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        visible: true,
        pulse: Math.random() * Math.PI * 2,
        birth: 1.0,  // birth animation progress (1.0 -> 0.0)
        flash: 0,
      });
    }

    conceptMap = incomingMap;
    concepts = incoming;

    // Update HUD stats
    document.getElementById('s-prin').textContent = concepts.filter(c => c.type === 'principle').length;
    document.getElementById('s-model').textContent = concepts.filter(c => c.type === 'mental_model').length;
    document.getElementById('s-rel').textContent = concepts.filter(c => c.type === 'relationship').length;
    document.getElementById('s-skill').textContent = concepts.filter(c => c.type === 'skill').length;

  } catch (e) {
    console.error('Poll failed:', e);
  }
}
```

**Step 2: Update `buildNodes` to initialize conceptMap**

Replace the existing `buildNodes` function — it's now only called on first load:

```javascript
function buildNodes() {
  conceptMap = {};
  nodes = concepts.map((c, i) => {
    conceptMap[c.id] = c;
    const scopeRing = SCOPE_RINGS[c.scope] || 0.5;
    const angle = (i / concepts.length) * Math.PI * 2 + Math.random() * 0.3;
    const radiusJitter = (Math.random() - 0.5) * 0.15;
    const radius = scopeRing + radiusJitter;
    const col = TYPE_COLORS[c.type] || { r: 100, g: 100, b: 100 };
    const evidenceCount = c.evidence_ids ? c.evidence_ids.length : 1;
    const baseSize = Math.max(3, Math.min(18, 3 + Math.sqrt(evidenceCount) * 2.5));

    return {
      concept: c,
      angle, radius, baseSize,
      size: baseSize,
      targetSize: baseSize,
      color: col,
      opacity: 0.3 + (c.confidence || 0.5) * 0.7,
      x: 0, y: 0,
      vx: (Math.random() - 0.5) * 0.1,
      vy: (Math.random() - 0.5) * 0.1,
      visible: true,
      pulse: Math.random() * Math.PI * 2,
      birth: 0,
      flash: 0,
    };
  });
}
```

**Step 3: Update `updatePositions` to handle birth/flash animation**

Add these lines inside the `for (const n of nodes)` loop in `updatePositions`, after the `n.size = n.baseSize + Math.sin(n.pulse) * 0.5;` line:

```javascript
    // Birth animation: grow from 0 to full size
    if (n.birth > 0) {
      n.birth = Math.max(0, n.birth - 0.02);
      n.size = n.targetSize * (1 - n.birth);
    }

    // Flash animation: decay over time
    if (n.flash > 0) {
      n.flash = Math.max(0, n.flash - 0.02);
    }
```

**Step 4: Update `drawNodes` to render birth glow and flash**

In `drawNodes`, after the core circle fill and before the consensus glow check, add:

```javascript
    // Birth glow
    if (n.birth > 0) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, size + 10 * n.birth, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(' + n.color.r + ',' + n.color.g + ',' + n.color.b + ',' + (n.birth * 0.6) + ')';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Update flash
    if (n.flash > 0) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, size + 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,255,255,' + (n.flash * 0.3) + ')';
      ctx.fill();
    }
```

**Step 5: Replace the startup code**

Replace the final line `loadData().then(function() { render(); });` with:

```javascript
loadData().then(function() {
  buildNodes();
  render();
  setInterval(loadData, POLL_INTERVAL);
});
```

**Step 6: Verify by loading `/cortex` in browser**

Open `http://YOUR_HOST:8088/cortex` — neural map should render and auto-refresh every 10s.

**Step 7: Commit**

```bash
git add api/app/ui/cortex.html
git commit -m "feat: live 10s polling with birth animation and flash highlights"
```

---

### Task 4: Pinnable node tooltips

Make nodes clickable to pin their tooltip open. Click elsewhere or press Escape to dismiss.

**Files:**
- Modify: `api/app/ui/cortex.html` (JS section)

**Step 1: Add pinned state**

Add after the `let hoveredNode = null;` line:

```javascript
let pinnedNode = null;
```

**Step 2: Add click handler on canvas**

After the existing `canvas.addEventListener('mouseleave', ...)` line, add:

```javascript
canvas.addEventListener('click', function(e) {
  if (hoveredNode) {
    pinnedNode = (pinnedNode === hoveredNode) ? null : hoveredNode;
  } else {
    pinnedNode = null;
  }
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') pinnedNode = null;
});
```

**Step 3: Update `findHovered` to respect pinned state**

In `findHovered`, change the tooltip display logic. Replace the section starting at `const tooltip = document.getElementById('tooltip');` with:

```javascript
  const tooltip = document.getElementById('tooltip');
  const activeNode = pinnedNode || closest;

  if (activeNode) {
    const c = activeNode.concept;
    const evidenceCount = c.evidence_ids ? c.evidence_ids.length : 0;
    const col = activeNode.color;

    document.getElementById('tt-type').textContent = c.type.replace('_', ' ');
    document.getElementById('tt-type').style.color = 'rgb(' + col.r + ',' + col.g + ',' + col.b + ')';
    document.getElementById('tt-title').textContent = c.title;

    if (c.type === 'skill') {
      var skillLines = [];
      if (c.trigger) skillLines.push('When: ' + c.trigger);
      if (c.action) skillLines.push('Do: ' + c.action);
      if (c.success_hint) skillLines.push('Verify: ' + c.success_hint);
      document.getElementById('tt-content').textContent = skillLines.length ? skillLines.join('\n') : (c.content ? c.content.substring(0, 300) : 'No preview');
    } else {
      document.getElementById('tt-content').textContent = c.content ? c.content.substring(0, 300) : 'No preview';
    }

    var metaText = 'scope: ' + c.scope + (c.scope_ref ? '/' + c.scope_ref : '') +
      '  |  conf: ' + (c.confidence || 0).toFixed(2) +
      '  |  evidence: ' + evidenceCount +
      '  |  v' + (c.version || 1);

    if (c.tags && c.tags.length > 0) {
      metaText += '\ntags: ' + c.tags.join(', ');
    }
    if (c.confirming_agents && c.confirming_agents.length > 0) {
      metaText += '\nconfirmed by: ' + c.confirming_agents.join(', ');
    }
    document.getElementById('tt-meta').textContent = metaText;

    tooltip.style.display = 'block';
    if (pinnedNode) {
      tooltip.style.right = '20px';
      tooltip.style.left = 'auto';
      tooltip.style.top = '80px';
    } else {
      tooltip.style.right = 'auto';
      tooltip.style.left = Math.min(mouse.x + 16, W - 400) + 'px';
      tooltip.style.top = Math.min(mouse.y + 16, H - 200) + 'px';
    }
    canvas.style.cursor = 'pointer';
  } else {
    tooltip.style.display = 'none';
    canvas.style.cursor = 'default';
  }
```

**Step 4: Commit**

```bash
git add api/app/ui/cortex.html
git commit -m "feat: clickable nodes pin tooltip open, Escape to dismiss"
```

---

### Task 5: Slide-out drawer — HTML structure and CSS

Add the drawer HTML, CSS, and tab navigation.

**Files:**
- Modify: `api/app/ui/cortex.html` (HTML and CSS sections)

**Step 1: Add drawer CSS**

Add before the closing `</style>` tag:

```css
  /* Drawer */
  .drawer-toggle {
    position: fixed; bottom: 20px; left: 20px; z-index: 10;
    width: 36px; height: 36px; border-radius: 50%;
    background: rgba(20,20,30,0.9); border: 1px solid #333;
    color: #8b5cf6; font-size: 18px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s;
  }
  .drawer-toggle:hover { border-color: #8b5cf6; background: rgba(139,92,246,0.15); }

  #drawer {
    position: fixed; top: 0; right: -400px; width: 400px; height: 100vh; z-index: 30;
    background: rgba(10,10,15,0.97); border-left: 1px solid #222;
    transition: right 0.3s ease; display: flex; flex-direction: row;
    font-size: 12px;
  }
  #drawer.open { right: 0; }

  .drawer-tabs {
    width: 48px; background: rgba(5,5,10,0.8); border-right: 1px solid #1a1a1a;
    display: flex; flex-direction: column; padding-top: 12px; gap: 4px;
  }
  .drawer-tab {
    width: 48px; height: 48px; border: none; background: transparent;
    color: #555; font-size: 16px; cursor: pointer; display: flex;
    align-items: center; justify-content: center; transition: all 0.2s;
    border-left: 2px solid transparent;
  }
  .drawer-tab:hover { color: #aaa; background: rgba(255,255,255,0.03); }
  .drawer-tab.active { color: #c4b5fd; border-left-color: #8b5cf6; background: rgba(139,92,246,0.08); }

  .drawer-content {
    flex: 1; overflow-y: auto; padding: 20px 16px;
  }
  .drawer-content::-webkit-scrollbar { width: 4px; }
  .drawer-content::-webkit-scrollbar-track { background: transparent; }
  .drawer-content::-webkit-scrollbar-thumb { background: #333; border-radius: 2px; }

  .drawer-close {
    position: absolute; top: 12px; right: 12px;
    background: none; border: none; color: #555; font-size: 18px; cursor: pointer;
  }
  .drawer-close:hover { color: #fff; }

  .drawer-section { display: none; }
  .drawer-section.active { display: block; }

  .drawer-section h3 { font-size: 11px; letter-spacing: 1.5px; text-transform: uppercase; color: #8b5cf6; margin-bottom: 16px; }
  .drawer-section h4 { font-size: 11px; color: #aaa; margin: 16px 0 8px; }

  .health-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
  .health-dot { width: 6px; height: 6px; border-radius: 50%; }
  .health-dot.ok { background: #22c55e; box-shadow: 0 0 4px #22c55e; }
  .health-dot.err { background: #ef4444; box-shadow: 0 0 4px #ef4444; }

  .config-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a1a; }
  .config-key { color: #666; }
  .config-val { color: #ccc; text-align: right; max-width: 200px; overflow: hidden; text-overflow: ellipsis; }

  .agent-card {
    background: rgba(255,255,255,0.02); border: 1px solid #1a1a1a; border-radius: 8px;
    padding: 12px; margin-bottom: 10px;
  }
  .agent-card .agent-name { font-weight: 600; color: #e0e0e0; margin-bottom: 4px; }
  .agent-card .agent-key { font-size: 10px; color: #555; margin-bottom: 8px; }
  .agent-card .agent-stats { display: flex; gap: 16px; font-size: 10px; color: #888; }
  .agent-card .agent-strengths { margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap; }
  .agent-card .strength-tag {
    background: rgba(139,92,246,0.15); border: 1px solid rgba(139,92,246,0.3);
    color: #c4b5fd; padding: 2px 8px; border-radius: 4px; font-size: 9px;
  }

  .activity-item { padding: 6px 0; border-bottom: 1px solid #111; font-size: 10px; color: #777; }
  .activity-item .act-type { display: inline-block; width: 60px; color: #888; }
  .activity-item .act-type.recall { color: #06b6d4; }
  .activity-item .act-type.feedback { color: #f59e0b; }
  .activity-item .act-agent { color: #666; }
  .activity-item .act-time { float: right; color: #444; font-size: 9px; }

  .rules-text { color: #999; line-height: 1.8; }
  .rules-text strong { color: #ccc; }
  .rules-text code { background: rgba(139,92,246,0.1); padding: 1px 4px; border-radius: 3px; color: #c4b5fd; }

  .setup-cmd { background: rgba(255,255,255,0.03); border: 1px solid #1a1a1a; border-radius: 6px; padding: 8px 12px; margin: 8px 0; font-size: 10px; color: #aaa; white-space: pre-wrap; word-break: break-all; }
```

**Step 2: Add drawer HTML**

Add after the `<div id="tooltip">...</div>` block and before the `<script>` tag:

```html
<button class="drawer-toggle" id="drawerToggle" title="System info">&#8857;</button>

<div id="drawer">
  <div class="drawer-tabs">
    <button class="drawer-tab active" data-tab="status" title="Status">&#9679;</button>
    <button class="drawer-tab" data-tab="config" title="Config">&#9881;</button>
    <button class="drawer-tab" data-tab="agents" title="Agents">&#9775;</button>
    <button class="drawer-tab" data-tab="rules" title="Rules">&#9733;</button>
    <button class="drawer-tab" data-tab="setup" title="Setup">&#9874;</button>
  </div>
  <div class="drawer-content">
    <button class="drawer-close" id="drawerClose">&times;</button>

    <!-- STATUS TAB -->
    <div class="drawer-section active" id="tab-status">
      <h3>System Status</h3>
      <div id="health-indicators"></div>
      <h4>Concept Counts</h4>
      <div id="drawer-concept-counts"></div>
      <h4>Recent Activity</h4>
      <div id="activity-feed"></div>
    </div>

    <!-- CONFIG TAB -->
    <div class="drawer-section" id="tab-config">
      <h3>Configuration</h3>
      <div id="config-grid"></div>
    </div>

    <!-- AGENTS TAB -->
    <div class="drawer-section" id="tab-agents">
      <h3>Enrolled Agents</h3>
      <div id="agents-list"></div>
    </div>

    <!-- RULES TAB -->
    <div class="drawer-section" id="tab-rules">
      <h3>How the Neural Net Works</h3>
      <div class="rules-text">
        <strong>5-Layer Architecture</strong><br>
        <strong>Layer 1 — Memory:</strong> Every AI conversation generates knowledge. AgentSSOT captures facts, decisions, and patterns as knowledge items with vector embeddings.<br><br>
        <strong>Layer 2 — Feedback:</strong> When recalled knowledge is useful, agents send <code>useful</code>/<code>noted</code>/<code>wrong</code> signals. Useful concepts gain confidence; wrong ones get tagged <code>contested</code>.<br><br>
        <strong>Layer 3 — Skills:</strong> Action patterns crystallize into skills with <code>trigger</code> (when), <code>action</code> (do what), and <code>success_hint</code> (verify). Shown as green nodes.<br><br>
        <strong>Layer 4 — Personalization:</strong> Each agent/device builds a profile of strengths from its recall patterns. Recall results are boosted toward each agent's expertise areas.<br><br>
        <strong>Layer 5 — Collective Intelligence:</strong> When 3+ agents independently confirm a concept, it reaches <code>consensus</code> (gold glow). Cross-agent confirmation elevates truth.<br><br>

        <strong>Concept Lifecycle</strong><br>
        Knowledge items cluster by embedding similarity. Ollama synthesizes concepts. Reconciler merges or creates. Confidence grows with evidence (+0.1 per match) and agent confirmation (+0.05 per new agent). Stale concepts decay (<code>-0.02/cycle</code> after <code>90 days</code>). Dormant at floor (<code>0.15</code>). Resurrectable on re-recall.<br><br>

        <strong>Visual Encoding</strong><br>
        <code>Color</code> = concept type.
        <code>Size</code> = evidence count.
        <code>Opacity</code> = confidence.
        <code>Ring</code> = scope (inner=global, mid=project, outer=device).
        <code>Gold glow</code> = consensus.
      </div>
    </div>

    <!-- SETUP TAB -->
    <div class="drawer-section" id="tab-setup">
      <h3>Onboarding &amp; Troubleshooting</h3>
      <div class="rules-text">
        <strong>Enroll a new device</strong><br>
        1. From an admin session, create an enrollment token<br>
        2. On the new device, call <code>POST /auto-enroll</code><br>
        3. Save the returned agent config to <code>~/.claude/agentssot/local/agent.json</code><br><br>

        <strong>Verify connectivity</strong>
      </div>
      <div class="setup-cmd">curl -s http://YOUR_HOST:8088/health | python3 -m json.tool</div>
      <div class="rules-text"><strong>Test recall</strong></div>
      <div class="setup-cmd">curl -s -X POST http://YOUR_HOST:8088/recall \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query_text":"test","namespace":"claude-shared"}' | python3 -m json.tool</div>
      <div class="rules-text"><strong>Trigger synthesis manually</strong></div>
      <div class="setup-cmd">curl -s -X POST http://YOUR_HOST:8088/synthesis/run \
  -H "X-API-Key: ADMIN_KEY" \
  -d '{"namespace":"claude-shared"}' | python3 -m json.tool</div>
      <div class="rules-text">
        <br><strong>Troubleshooting</strong><br>
        - <code>embedding_available: false</code> — Check Ollama is running: <code>curl http://YOUR_HOST:11434/api/tags</code><br>
        - <code>llm_available: false</code> — Same — verify Ollama and model pulled<br>
        - No concepts appearing — Run synthesis manually (above) then wait 10s<br>
        - Recall returns empty — Check namespace and that knowledge items exist with embeddings<br>
      </div>
      <h4>Live Health Check</h4>
      <div id="setup-health-live"></div>
    </div>
  </div>
</div>
```

**Step 3: Commit**

```bash
git add api/app/ui/cortex.html
git commit -m "feat: drawer HTML structure and CSS for 5-tab system panel"
```

---

### Task 6: Drawer JavaScript — tab switching, data loading, activity feed

Wire up the drawer's tab navigation, fetch system-info and activity data, and render all dynamic content.

**Files:**
- Modify: `api/app/ui/cortex.html` (JS section)

**Step 1: Add drawer JS**

Add this block at the end of the `<script>` section, before the final `loadData().then(...)` startup code.

Note: All rendered content comes from our own trusted API responses (LAN-only, read-only endpoints). The rendering uses DOM construction helpers that escape text content appropriately. See the `_esc()` helper and `_el()` helper below.

```javascript
// ---------------------------------------------------------------------------
// Drawer
// ---------------------------------------------------------------------------
let drawerOpen = false;
let systemInfo = null;
let activityData = null;

// Safe text escaping for rendering API data
function _esc(str) {
  var d = document.createElement('div');
  d.textContent = str;
  return d.textContent;
}

function toggleDrawer() {
  drawerOpen = !drawerOpen;
  document.getElementById('drawer').classList.toggle('open', drawerOpen);
  if (drawerOpen) loadDrawerData();
}

document.getElementById('drawerToggle').addEventListener('click', toggleDrawer);
document.getElementById('drawerClose').addEventListener('click', function() {
  drawerOpen = false;
  document.getElementById('drawer').classList.remove('open');
});

// Tab switching
document.querySelectorAll('.drawer-tab').forEach(function(tab) {
  tab.addEventListener('click', function() {
    document.querySelectorAll('.drawer-tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.drawer-section').forEach(function(s) { s.classList.remove('active'); });
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

async function loadDrawerData() {
  try {
    var responses = await Promise.all([
      fetch('/cortex/system-info?namespace=claude-shared'),
      fetch('/cortex/activity?namespace=claude-shared'),
    ]);
    systemInfo = await responses[0].json();
    activityData = await responses[1].json();
    renderDrawer();
  } catch (e) {
    console.error('Drawer data fetch failed:', e);
  }
}

function renderDrawer() {
  if (!systemInfo) return;

  // --- Status tab ---
  var h = systemInfo.health;
  var healthEl = document.getElementById('health-indicators');
  healthEl.textContent = '';
  healthEl.appendChild(makeHealthRow('Embedding', h.embedding_available));
  healthEl.appendChild(makeHealthRow('LLM', h.llm_available));
  healthEl.appendChild(makeHealthRow('Reranker', h.reranker_available));
  healthEl.appendChild(makeHealthRow('Synthesis', h.synthesis_enabled));

  var counts = {};
  for (var ci = 0; ci < concepts.length; ci++) { counts[concepts[ci].type] = (counts[concepts[ci].type] || 0) + 1; }
  var countsEl = document.getElementById('drawer-concept-counts');
  countsEl.textContent = '';
  for (var ctype in counts) {
    countsEl.appendChild(makeConfigRow(ctype.replace('_', ' '), String(counts[ctype])));
  }
  countsEl.appendChild(makeConfigRow('total concepts', String(concepts.length)));
  countsEl.appendChild(makeConfigRow('knowledge items', document.getElementById('s-know').textContent));

  // --- Activity feed ---
  var feedEl = document.getElementById('activity-feed');
  feedEl.textContent = '';
  if (activityData && activityData.events) {
    var evts = activityData.events.slice(0, 20);
    for (var ei = 0; ei < evts.length; ei++) {
      var ev = evts[ei];
      var item = document.createElement('div');
      item.className = 'activity-item';
      var typeSpan = document.createElement('span');
      typeSpan.className = 'act-type ' + ev.type;
      typeSpan.textContent = ev.type;
      var agentSpan = document.createElement('span');
      agentSpan.className = 'act-agent';
      agentSpan.textContent = (ev.agent || '').replace('device-', '').replace('-writer', '') + ' ';
      var detailText = document.createTextNode(ev.concept ? ev.concept.substring(0, 40) : (ev.detail || ''));
      var timeSpan = document.createElement('span');
      timeSpan.className = 'act-time';
      timeSpan.textContent = new Date(ev.at).toLocaleTimeString();
      item.appendChild(typeSpan);
      item.appendChild(document.createTextNode(' '));
      item.appendChild(agentSpan);
      item.appendChild(detailText);
      item.appendChild(timeSpan);
      feedEl.appendChild(item);
    }
  }

  // --- Config tab ---
  var cfg = systemInfo.config;
  var configEl = document.getElementById('config-grid');
  configEl.textContent = '';
  for (var key in cfg) {
    configEl.appendChild(makeConfigRow(key.replace(/_/g, ' '), String(cfg[key])));
  }

  // --- Agents tab ---
  var agents = systemInfo.agents || [];
  var agentsEl = document.getElementById('agents-list');
  agentsEl.textContent = '';
  if (agents.length === 0) {
    var emptyMsg = document.createElement('div');
    emptyMsg.style.color = '#555';
    emptyMsg.textContent = 'No agent profiles yet. Profiles build automatically from recall/feedback activity.';
    agentsEl.appendChild(emptyMsg);
  } else {
    for (var ai = 0; ai < agents.length; ai++) {
      agentsEl.appendChild(makeAgentCard(agents[ai]));
    }
  }

  // --- Setup tab live health ---
  var setupHealthEl = document.getElementById('setup-health-live');
  setupHealthEl.textContent = '';
  setupHealthEl.appendChild(makeHealthRow('Embedding', systemInfo.health.embedding_available));
  setupHealthEl.appendChild(makeHealthRow('LLM', systemInfo.health.llm_available));
  setupHealthEl.appendChild(makeHealthRow('Reranker', systemInfo.health.reranker_available));
  setupHealthEl.appendChild(makeHealthRow('Synthesis', systemInfo.health.synthesis_enabled));
}

function makeHealthRow(label, ok) {
  var row = document.createElement('div');
  row.className = 'health-row';
  var dot = document.createElement('div');
  dot.className = 'health-dot ' + (ok ? 'ok' : 'err');
  var text = document.createElement('span');
  text.textContent = label;
  row.appendChild(dot);
  row.appendChild(text);
  return row;
}

function makeConfigRow(key, val) {
  var row = document.createElement('div');
  row.className = 'config-row';
  var k = document.createElement('span');
  k.className = 'config-key';
  k.textContent = key;
  var v = document.createElement('span');
  v.className = 'config-val';
  v.textContent = val;
  row.appendChild(k);
  row.appendChild(v);
  return row;
}

function makeAgentCard(a) {
  var card = document.createElement('div');
  card.className = 'agent-card';
  var name = document.createElement('div');
  name.className = 'agent-name';
  name.textContent = a.device_name || 'unknown';
  var key = document.createElement('div');
  key.className = 'agent-key';
  key.textContent = a.agent_key;
  var stats = document.createElement('div');
  stats.className = 'agent-stats';
  stats.textContent = 'recalls: ' + a.total_recalls + '  feedback: ' + a.total_feedback + '  updated: ' + (a.updated_at ? new Date(a.updated_at).toLocaleDateString() : 'never');
  card.appendChild(name);
  card.appendChild(key);
  card.appendChild(stats);
  if (a.strengths && a.strengths.length > 0) {
    var strengthsDiv = document.createElement('div');
    strengthsDiv.className = 'agent-strengths';
    for (var si = 0; si < a.strengths.length; si++) {
      var tag = document.createElement('span');
      tag.className = 'strength-tag';
      tag.textContent = a.strengths[si];
      strengthsDiv.appendChild(tag);
    }
    card.appendChild(strengthsDiv);
  }
  return card;
}
```

**Step 2: Also refresh drawer data on each poll cycle (if drawer is open)**

At the end of the `loadData` function, add:

```javascript
  if (drawerOpen) loadDrawerData();
```

**Step 3: Verify in browser**

Open `http://YOUR_HOST:8088/cortex`, click the `⊙` button in the bottom-left. Drawer should slide in with Status, Config, Agents, Rules, Setup tabs all populated.

**Step 4: Commit**

```bash
git add api/app/ui/cortex.html
git commit -m "feat: wire drawer JS — tab switching, live data, activity feed"
```

---

### Task 7: Build, deploy, and verify

Rebuild the container and verify everything works end-to-end.

**Files:**
- No file changes — deployment and verification only

**Step 1: Rebuild**

Run: `docker compose up -d --build api`

**Step 2: Verify API endpoints**

```bash
curl -s http://YOUR_HOST:8088/cortex/system-info | python3 -m json.tool
curl -s http://YOUR_HOST:8088/cortex/activity | python3 -m json.tool
curl -s http://YOUR_HOST:8088/cortex/data?namespace=claude-shared | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'concepts: {d[\"total\"]}, knowledge: {d[\"knowledge_count\"]}')"
```

**Step 3: Visual verification checklist**

Open `http://YOUR_HOST:8088/cortex` in a browser and verify:
- [ ] Neural map renders with concept nodes
- [ ] HUD shows correct counts
- [ ] Nodes slowly drift and pulse
- [ ] Hovering a node shows tooltip
- [ ] Clicking a node pins the tooltip
- [ ] Pressing Escape unpins
- [ ] `⊙` button opens slide-out drawer
- [ ] Status tab shows green health dots and concept counts
- [ ] Activity feed shows recent recalls/feedback
- [ ] Config tab shows synthesis settings
- [ ] Agents tab shows enrolled profiles with strengths
- [ ] Rules tab displays the 5-layer architecture
- [ ] Setup tab has verification commands
- [ ] After 10s, new data loads without page refresh
- [ ] `×` or clicking outside closes the drawer

**Step 4: Final commit if any tweaks needed**

```bash
git add -u
git commit -m "fix: cortex v2 deployment tweaks"
```
