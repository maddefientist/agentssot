# Cortex Visualizer v3 — 3D Neural Network Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Canvas 2D cortex visualizer with a Three.js 3D neural network showing real concept_links edges, particle flow, cluster hulls, and click-to-focus detail panel.

**Architecture:** Three.js scene with InstancedMesh nodes on spherical shells, LineSegments for edges from concept_links table, custom shader particle flow on focus, ConvexHull cluster groups, UnrealBloomPass post-processing. HTML overlays for HUD, controls, detail panel, drawer.

**Tech Stack:** Three.js r170 (CDN), OrbitControls, EffectComposer, UnrealBloomPass, FXAA — FastAPI backend, SQLAlchemy, PostgreSQL.

---

### Task 1: Add `/cortex/links` API Endpoint

**Files:**
- Modify: `api/app/crud.py` (add `list_concept_links` function after line ~1209)
- Modify: `api/app/main.py` (add endpoint after line ~168, after `cortex_data`)
- Test: `api/tests/test_cortex.py` (add `TestCortexLinks` class)

**Step 1: Write the failing test**

Add to `api/tests/test_cortex.py`:

```python
class TestCortexLinks:
    """Test the /cortex/links endpoint for neural network edges."""

    def test_cortex_links_returns_list(self):
        resp = httpx.get(f"{BASE_URL}/cortex/links", params={"namespace": "claude-shared"})
        assert resp.status_code == 200
        data = resp.json()
        assert "links" in data
        assert isinstance(data["links"], list)

    def test_cortex_links_structure(self):
        resp = httpx.get(f"{BASE_URL}/cortex/links", params={"namespace": "claude-shared"})
        assert resp.status_code == 200
        data = resp.json()
        if data["links"]:
            link = data["links"][0]
            assert "source" in link
            assert "target" in link
            assert "weight" in link
            assert "link_type" in link
            assert "co_occurrences" in link

    def test_cortex_links_respects_limit(self):
        resp = httpx.get(f"{BASE_URL}/cortex/links", params={"namespace": "claude-shared", "limit": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["links"]) <= 5
```

**Step 2: Run tests to verify they fail**

Run: `docker compose exec api python -m pytest tests/test_cortex.py::TestCortexLinks -v`
Expected: FAIL with 404 (endpoint doesn't exist)

**Step 3: Add `list_concept_links` to crud.py**

Add after the `list_concepts` function (~line 1209) in `api/app/crud.py`:

```python
def list_concept_links(
    session: Session,
    namespace: str,
    limit: int = 200,
) -> list[dict]:
    """List concept links for a namespace, ordered by weight descending."""
    from .models import ConceptLink, Concept

    # Only return links where both concepts belong to the namespace
    stmt = (
        select(
            ConceptLink.concept_a,
            ConceptLink.concept_b,
            ConceptLink.weight,
            ConceptLink.link_type,
            ConceptLink.co_occurrence_count,
        )
        .join(Concept, ConceptLink.concept_a == Concept.id)
        .where(Concept.namespace == namespace)
        .order_by(ConceptLink.weight.desc())
        .limit(min(max(limit, 1), 500))
    )

    rows = session.execute(stmt).all()
    return [
        {
            "source": str(r.concept_a),
            "target": str(r.concept_b),
            "weight": r.weight,
            "link_type": r.link_type,
            "co_occurrences": r.co_occurrence_count,
        }
        for r in rows
    ]
```

**Step 4: Add the endpoint to main.py**

Add after the `cortex_data` endpoint (~line 168) in `api/app/main.py`:

```python
@app.get("/cortex/links", include_in_schema=False)
def cortex_links(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=200, le=500),
    session: Session = Depends(get_session),
):
    """Public read-only endpoint for cortex edges. Returns concept_links."""
    return {"links": crud.list_concept_links(session, namespace, limit)}
```

**Step 5: Rebuild container and run tests**

Run: `docker compose up -d --build api && sleep 5 && docker compose exec api python -m pytest tests/test_cortex.py::TestCortexLinks -v`
Expected: All 3 PASS

**Step 6: Commit**

```
git add api/app/crud.py api/app/main.py api/tests/test_cortex.py
git commit -m "wires the synapses to the cortex window — adds /cortex/links endpoint"
```

---

### Task 2: Three.js Scene Foundation + Scope Shells

**Files:**
- Modify: `api/app/ui/cortex.html` (full rewrite — keep backup of old as `cortex-v2.html`)

**Context:** This replaces the entire cortex.html. The old canvas 2D renderer is removed. All HTML overlay structure (HUD, search, controls, legend, tooltip, drawer) is preserved and adapted.

**Step 1: Back up existing cortex.html**

```bash
cp api/app/ui/cortex.html api/app/ui/cortex-v2.html
```

**Step 2: Write the new cortex.html with Three.js scene foundation**

The new file must include:

1. **CDN imports** (in `<head>`):
   - `three` module from `https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js`
   - `OrbitControls` from `three/addons/controls/OrbitControls.js`
   - `EffectComposer` from `three/addons/postprocessing/EffectComposer.js`
   - `RenderPass` from `three/addons/postprocessing/RenderPass.js`
   - `UnrealBloomPass` from `three/addons/postprocessing/UnrealBloomPass.js`
   - `ShaderPass` + `FXAAShader` from `three/addons`
   - Use an importmap to alias `three` for the addons

2. **Scene setup** (`initScene()`):
   - `THREE.Scene` with dark background (`0x0a0a0f`)
   - `THREE.PerspectiveCamera` at position `(0, 60, 120)`, looking at origin
   - `THREE.WebGLRenderer` with `antialias: true, alpha: false` filling the viewport
   - `OrbitControls` attached to camera with damping enabled, auto-rotate at 0.3 speed
   - Post-processing chain: RenderPass → UnrealBloomPass (strength 0.6, radius 0.5, threshold 0.2) → FXAAPass

3. **Scope shells** (`createScopeShells()`):
   - Three `THREE.SphereGeometry` wireframes at radii 15, 55, 85
   - Material: `MeshBasicMaterial({ color: 0xffffff, wireframe: true, opacity: 0.03, transparent: true })`
   - Floating text labels ("global", "project", "device") via `THREE.Sprite` with `CanvasTexture`

4. **Ambient lighting**: Subtle `AmbientLight(0x404060)` + `PointLight(0x8b5cf6, 0.5)` at center

5. **Resize handler**: Updates camera aspect, renderer size, composer size, FXAA uniforms

6. **Animate loop**: `requestAnimationFrame` → controls.update() → composer.render()

7. **Preserve all HTML overlays**: HUD, search, controls, legend, tooltip, drawer — same CSS as v2, same drawer JS logic

**Step 3: Verify scene renders**

Run: `docker compose up -d --build api`
Visit: `http://YOUR_HOST:8088/cortex`
Expected: Dark scene with 3 wireframe spherical shells, auto-rotating camera, bloom glow, all HTML overlays visible

**Step 4: Commit**

```
git add api/app/ui/cortex.html api/app/ui/cortex-v2.html
git commit -m "builds the cortex a third eye — Three.js 3D scene with scope shells and bloom"
```

---

### Task 3: Node System (InstancedMesh + Golden Spiral)

**Files:**
- Modify: `api/app/ui/cortex.html` (add node rendering)

**Step 1: Implement node system**

Add to the JavaScript:

1. **Data structures**:
   - `let nodeData = []` — array of `{ concept, position: THREE.Vector3, color, baseSize, opacity, visible, birth, flash, pulse }`
   - `let conceptMap = {}` — id → concept dict
   - `let instancedNodes = null` — the InstancedMesh
   - `const TYPE_COLORS` — same color map as v2 but as `THREE.Color` objects
   - `const SCOPE_RADII = { global: 15, project: 55, device: 85 }`

2. **Golden spiral placement** (`placeOnShell(index, total, radius)`):
   - Uses golden angle `(2.399963...)` to distribute points evenly on a sphere
   - `y = 1 - (2 * index / total)` for latitude
   - `theta = goldenAngle * index` for longitude
   - Converts to cartesian `(x, y, z)` at `radius` with ±2 jitter

3. **Build nodes** (`buildNodes(concepts)`):
   - Groups concepts by scope
   - Places each group on its scope shell using golden spiral
   - Creates `THREE.InstancedMesh` with `SphereGeometry(1, 16, 16)` and `MeshStandardMaterial({ emissive, emissiveIntensity: 0.8, transparent: true })`
   - Sets instance matrix (position + scale from baseSize) and color per instance
   - Adds mesh to scene

4. **Update nodes** (`updateNodes(newConcepts)`):
   - Diff-based: detect new, removed, and changed concepts
   - New concepts: birth animation (scale from 0 to baseSize over 30 frames)
   - Changed concepts: flash animation (white pulse for 20 frames)
   - Removed: mark invisible, rebuild on next full cycle
   - Updates instance matrices and colors each frame

5. **HUD stats update**: Same counters as v2 (principles, models, relationships, skills, knowledge)

6. **Data loading** (`loadData()`):
   - `fetch('/cortex/data?namespace=claude-shared')` → concepts
   - First load: `buildNodes(concepts)`
   - Subsequent: `updateNodes(concepts)`
   - Called every 10s via `setInterval`

7. **Animation integration**: In the animate loop, update birth/flash/pulse animations per frame by iterating nodeData and updating instance matrices

**Step 2: Verify nodes render**

Run: `docker compose up -d --build api`
Expected: Colored spheres on 3 shells, auto-rotating, HUD stats populated, new nodes birth-animate

**Step 3: Commit**

```
git add api/app/ui/cortex.html
git commit -m "grows neurons in three dimensions — instanced sphere nodes on golden spiral shells"
```

---

### Task 4: Raycaster Interaction (Hover + Click + Detail Panel)

**Files:**
- Modify: `api/app/ui/cortex.html` (add interaction + detail panel)

**Step 1: Implement raycaster hover**

1. **Raycaster setup**:
   - `THREE.Raycaster` with `raycaster.params.Points.threshold = 2`
   - On `mousemove`: update raycaster from camera, intersect with `instancedNodes`
   - Map `intersection.instanceId` → `nodeData[instanceId]`
   - Show floating label (HTML div) near mouse with concept title

2. **Click-to-focus**:
   - On `click`: if intersected node, set as `focusedNode`
   - Animate camera to position 30 units from the focused node using `gsap`-style lerp in the animate loop (or manual tween: store target position, lerp camera.position toward it each frame at rate 0.05)
   - Disable auto-rotate while focused
   - Click on empty space or press Escape: unfocus, return camera to default orbit, re-enable auto-rotate

3. **Detail panel** (HTML overlay, right side):
   - Structure (new `#detail-panel` div):
     ```
     .detail-panel { position: fixed; top: 0; right: -420px; width: 420px; height: 100vh;
       background: rgba(10,10,15,0.97); border-left: 1px solid #222;
       transition: right 0.3s ease; z-index: 25; overflow-y: auto; padding: 24px; }
     .detail-panel.open { right: 0; }
     ```
   - Content populated on focus:
     - Type badge (colored pill)
     - Title (large)
     - Content text (scrollable, max 500 chars with "show more")
     - Confidence bar (gradient bar 0-100%)
     - Evidence count
     - Confirming agents (colored pills)
     - Tags (small pills)
     - Scope + scope_ref
     - Version
     - **Connected concepts** list (populated from links data — clickable items that re-focus camera)
   - Close button or click-away closes

4. **Search integration**: On search input, iterate nodeData and set visibility. Matching nodes get emissive intensity boost (glow). Non-matching get opacity 0.05.

5. **Filter buttons**: Same logic as v2 — type/scope filtering updates node visibility via instance color alpha.

**Step 2: Verify interactions**

Expected:
- Hover shows floating label
- Click focuses camera + opens detail panel
- Escape/click-away unfocuses
- Search dims non-matching
- Filters work

**Step 3: Commit**

```
git add api/app/ui/cortex.html
git commit -m "teaches the cortex to feel your touch — raycaster hover, click-focus, detail panel"
```

---

### Task 5: Edge Rendering (Global Top-N + Focus Edges)

**Files:**
- Modify: `api/app/ui/cortex.html` (add edge system)

**Step 1: Implement edge system**

1. **Load links** (`loadLinks()`):
   - `fetch('/cortex/links?namespace=claude-shared&limit=200')` → links array
   - Store in `let allLinks = []`
   - Called on startup and every 30s

2. **Global edges** (`buildGlobalEdges()`):
   - Take top 80 links by weight
   - Create `THREE.BufferGeometry` with position attribute (2 vertices per edge = source/target positions from nodeData)
   - Material: `LineBasicMaterial({ color: 0x8b5cf6, transparent: true, opacity: 0.08, blending: AdditiveBlending })`
   - Add as `THREE.LineSegments` to scene
   - Rebuild when links refresh

3. **Focus edges** (`showFocusEdges(nodeId)`):
   - Filter allLinks for any link where source or target matches nodeId
   - Create separate `THREE.LineSegments` with brighter material (opacity 0.4, thicker via `linewidth` — note: linewidth >1 doesn't work in WebGL, so use shader-based lines or `Line2` from three addons if needed, OR keep thin and rely on bloom glow to make them visible)
   - Alternative for thick lines: Use `THREE.Mesh` with `TubeGeometry` for each focus edge (max ~20 edges so performance is fine)
   - Color: source node's type color
   - Remove focus edges when unfocusing

4. **Connected concepts** in detail panel:
   - When focusing a node, filter allLinks for its connections
   - Populate the "Connected Concepts" list with target concept titles
   - Each entry is clickable → re-focuses camera to that concept

5. **Edge position updates**: In animate loop, if nodes move (birth animation), update edge positions accordingly

**Step 2: Verify edges render**

Expected: Subtle purple web of connections always visible. Clicking a node reveals brighter colored edges to its neighbors.

**Step 3: Commit**

```
git add api/app/ui/cortex.html
git commit -m "strings the neurons together with glowing synapses — edge rendering with focus highlight"
```

---

### Task 6: Particle Flow System

**Files:**
- Modify: `api/app/ui/cortex.html` (add particle system)

**Step 1: Implement particle flow on focus edges**

1. **Particle system** (`createParticleSystem()`):
   - Pool of ~500 particles as `THREE.Points` with custom `ShaderMaterial`
   - Vertex shader: takes `position`, `progress` (0-1 along edge), `edgeStart`, `edgeEnd` uniforms/attributes → lerps position
   - Fragment shader: circular point with soft falloff, type color, additive blending
   - Initially invisible (all particles at origin with size 0)

2. **Activate on focus** (`activateParticles(focusedNodeId)`):
   - Get focus edges (source/target positions)
   - Assign particles to edges (distribute ~20-30 per edge)
   - Each particle has: `progress` (random 0-1), `speed` (proportional to edge weight, range 0.005-0.02), `edgeIndex`
   - In animate loop: advance `progress` by `speed`, wrap at 1.0 → 0.0
   - Update positions buffer: `pos = lerp(edgeStart, edgeEnd, progress)`

3. **Deactivate on unfocus**: Reset all particles to origin, size 0

4. **Visual tuning**:
   - Particle size: 0.3-0.5 (small, numerous)
   - Blending: `AdditiveBlending` for glow
   - Color: matches source node type color with slight brightness boost

**Step 2: Verify particles flow**

Expected: Click a node → particles stream along its edges toward/from connected nodes. Unfocus → particles disappear.

**Step 3: Commit**

```
git add api/app/ui/cortex.html
git commit -m "sends tiny messengers racing through the neural pathways — particle flow on focus edges"
```

---

### Task 7: Cluster Hulls

**Files:**
- Modify: `api/app/ui/cortex.html` (add cluster hulls)

**Step 1: Implement cluster hulls**

1. **Group nodes by scope_ref** (`buildClusterHulls()`):
   - Group nodeData by `concept.scope_ref` (skip null/empty)
   - Only build hulls for groups with 3+ nodes (need 3 points for a triangle)

2. **Convex hull mesh**:
   - Collect Vector3 positions for group
   - Use `ConvexGeometry` from Three.js addons (`three/addons/geometries/ConvexGeometry.js`)
   - Material: `MeshBasicMaterial({ color: averageTypeColor, transparent: true, opacity: 0.04, side: DoubleSide, depthWrite: false })`
   - Wireframe copy: same geometry, `MeshBasicMaterial({ color: averageTypeColor, wireframe: true, opacity: 0.1, transparent: true })`
   - Add both to scene in a `THREE.Group` for easy removal

3. **Label sprites**:
   - Compute centroid of group positions
   - Create `THREE.Sprite` with `CanvasTexture` text label (`scope_ref` name)
   - Position at centroid, scale 5
   - Font: 11px monospace, white at opacity 0.3

4. **Rebuild hulls**: On data refresh (every 10s), remove old hull groups and rebuild. Since hull computation is cheap for <50 groups, this is fine.

5. **Filter integration**: When type/scope filters change, hide hulls whose nodes are all filtered out.

**Step 2: Verify hulls render**

Expected: Transparent colored volumes wrapping clusters of related concepts, with faint labels.

**Step 3: Commit**

```
git add api/app/ui/cortex.html
git commit -m "wraps kindred neurons in translucent membranes — cluster hull visualization"
```

---

### Task 8: Polish + Consensus Glow + Birth/Flash + Legend Update

**Files:**
- Modify: `api/app/ui/cortex.html` (visual polish)

**Step 1: Consensus glow**

- For nodes where `concept.confirming_agents.length >= 3`:
  - Add a pulsing gold `PointLight` at node position (intensity oscillates via sin wave)
  - OR: set instance emissive to gold color with oscillating intensity
  - Simpler approach: in the animate loop, for consensus nodes, oscillate the instance color between type color and gold

**Step 2: Birth animation refinement**

- New nodes scale from 0 to baseSize over 40 frames with ease-out
- Expanding ring effect: add a temporary `THREE.RingGeometry` mesh at birth position that expands and fades (remove after animation completes)

**Step 3: Flash animation**

- When a concept updates, flash the instance color to white for 20 frames, then fade back to type color

**Step 4: Update legend**

- Add to legend: "Edges = shared evidence", "Particles = active connections", "Hull = concept cluster"
- Update existing entries for 3D context

**Step 5: Verify polish**

Expected: Gold-glowing consensus nodes, smooth birth animations, flash on update, accurate legend.

**Step 6: Commit**

```
git add api/app/ui/cortex.html
git commit -m "polishes every neuron until it sparkles — consensus glow, birth rings, flash pulses"
```

---

### Task 9: Final Integration + Drawer Preservation + Testing

**Files:**
- Modify: `api/app/ui/cortex.html` (ensure drawer works)
- Test: Manual browser testing

**Step 1: Verify drawer integration**

- Drawer toggle button (`#drawerToggle`) works
- All 5 tabs load data correctly (Status, Config, Agents, Rules, Setup)
- Drawer data loading uses same `/cortex/system-info` and `/cortex/activity` endpoints
- Drawer opens on top of WebGL canvas (z-index 30)

**Step 2: Verify detail panel + drawer don't conflict**

- Detail panel (z-index 25) and drawer (z-index 30) can both be open
- Drawer overlaps detail panel when both open (acceptable)

**Step 3: Performance check**

- Verify FPS in browser devtools (target: 60fps with 400+ nodes)
- Check GPU memory usage (should be <200MB)
- Verify polling doesn't cause frame drops

**Step 4: Cross-browser smoke test**

- Chrome (primary)
- Firefox (secondary)
- Mobile/tablet viewport (orbit via touch)

**Step 5: Rebuild and final commit**

```bash
docker compose up -d --build api
```

```
git add -A
git commit -m "the cortex opens its third eye — v3 3D neural network visualization complete"
```

---

## Task Dependency Graph

```
Task 1 (API endpoint) ──┐
                         ├──▶ Task 5 (Edges — needs links API + nodes)
Task 2 (Scene foundation)──┤
                            ├──▶ Task 3 (Nodes — needs scene)
                            │       │
                            │       ├──▶ Task 4 (Interaction — needs nodes)
                            │       │       │
                            │       │       ├──▶ Task 5 (Edges — needs nodes + interaction)
                            │       │       │       │
                            │       │       │       ├──▶ Task 6 (Particles — needs edges)
                            │       │       │
                            │       ├──▶ Task 7 (Hulls — needs nodes)
                            │
                            └──▶ Task 8 (Polish — needs all above)
                                    │
                                    └──▶ Task 9 (Integration — final)
```

**Parallel opportunities:** Task 1 (backend) can run in parallel with Tasks 2-3 (frontend scene + nodes). Task 7 (hulls) can run in parallel with Tasks 5-6 (edges + particles).

## Notes for implementer

- **Three.js version:** Use r170 (`0.170.0`) — stable, well-documented
- **Import map:** Required for bare specifier `three` to work with addons. Add to `<head>`:
  ```html
  <script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"}}</script>
  ```
- **No build step:** Everything in a single HTML file with `<script type="module">`
- **Self-contained:** No local JS files, all from CDN
- **Container rebuild:** After every change: `docker compose up -d --build api`
- **Existing API:** `/cortex/data`, `/cortex/system-info`, `/cortex/activity` remain unchanged
- **ConvexGeometry addon:** `three/addons/geometries/ConvexGeometry.js`
- **Line2 for thick lines (optional):** `three/addons/lines/Line2.js`, `LineMaterial.js`, `LineGeometry.js`
