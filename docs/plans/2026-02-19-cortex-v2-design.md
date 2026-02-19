# Cortex Visualizer v2 — Live Neural Dashboard

## Goal

Upgrade the cortex visualizer from a static snapshot into a live, self-updating neural map with system management built into a slide-out drawer. The neural map remains the primary experience; system details are one click away but never intrude.

## Architecture

Single self-contained HTML file (`api/app/ui/cortex.html`), no build step, no dependencies. Two new read-only API endpoints supply system data. All endpoints are public (no auth) — read-only, LAN-only.

## Live Polling

- Fetch `/cortex/data` every 10 seconds
- Diff incoming concepts against current node set
- New nodes get a "birth" animation: expand from size 0 over 1s with a bright color pulse
- Removed nodes fade out over 0.5s
- Updated nodes (confidence/evidence changed) get a brief highlight flash
- Stats in HUD update on each poll

## Node Interaction

- **Hover**: Shows tooltip (existing behavior, unchanged)
- **Click**: Pins the tooltip open. Shows full content, all metadata, tags, confirming agents. Click elsewhere or press Escape to dismiss.

## Slide-Out Drawer

- Triggered by a small `⊙` button fixed in the bottom-left corner
- Slides in from the right edge, 380px wide, over 300ms
- Semi-transparent background (`rgba(10,10,15,0.96)`) so neural map shows through
- Close via `×` button or clicking outside the drawer
- Contains 5 tabs as vertical nav on the left edge of the drawer:

### Tab: Status
- **Source**: `/health` + `/cortex/data` stats
- Service health indicators (embedding, LLM, reranker — green/red dots)
- Concept counts by type (principle, mental_model, relationship, skill)
- Knowledge item count, namespace name
- Last synthesis run timestamp and results (if available from activity feed)

### Tab: Config
- **Source**: `/cortex/system-info` (new endpoint)
- Namespace, embedding model + dimensions, LLM model, reranker model
- Synthesis schedule hour, similarity threshold, min cluster size
- Decay rate, decay floor, grace days, feedback protection days

### Tab: Agents
- **Source**: `/cortex/system-info` (new endpoint)
- Card per enrolled agent: device name, agent key, strengths tags, total recalls, total feedback, last updated
- Visual indicator for active (recent activity) vs dormant

### Tab: Rules
- **Source**: Static HTML content
- The 5-layer architecture: Memory → Feedback → Skills → Personalization → Collective Intelligence
- How concepts form (clustering → synthesis → reconciliation)
- Confidence mechanics: reinforcement (+0.1 per evidence, +0.05 per new agent), decay (0.02/cycle after 90 days), floor (0.15 → dormant tag)
- Consensus threshold: 3+ confirming agents → gold glow
- Skill structure: trigger/action/success_hint

### Tab: Setup
- **Source**: Static content + `/health` for live verification
- Onboarding steps for new devices (enrollment token or auto-enroll)
- Troubleshooting checklist: health endpoint, embedding test, recall test
- Verification commands: curl examples for health, recall, ingest
- Link to `/onboarding` endpoint for full instructions

## New API: `/cortex/system-info`

Public, no auth. Returns:

```json
{
  "health": { "status": "ok", "embedding_available": true, "llm_available": true, "reranker_available": true },
  "config": {
    "namespace": "claude-shared",
    "embedding_model": "qwen3-embedding",
    "embedding_dim": 4096,
    "llm_model": "qwen3:latest",
    "reranker_model": "Qwen3-Reranker-8B",
    "synthesis_schedule_hour": 3,
    "synthesis_similarity_threshold": 0.6,
    "synthesis_min_cluster_size": 2,
    "synthesis_confidence_decay": 0.02,
    "synthesis_decay_floor": 0.15,
    "synthesis_decay_grace_days": 90
  },
  "agents": [
    {
      "agent_key": "device-hari-writer",
      "device_name": "hari",
      "strengths": ["docker", "python"],
      "total_recalls": 15,
      "total_feedback": 8,
      "updated_at": "2026-02-19T..."
    }
  ],
  "namespaces": ["claude-shared", "default"]
}
```

## New API: `/cortex/activity`

Public, no auth. Returns last 50 system events for a live activity ticker:

```json
{
  "events": [
    { "type": "recall", "agent": "device-hari-writer", "query": "docker networking...", "at": "2026-02-19T..." },
    { "type": "feedback", "agent": "device-hari-writer", "signal": "useful", "concept": "Container DNS resolution", "at": "..." },
    { "type": "synthesis", "new": 5, "updated": 11, "at": "..." }
  ]
}
```

Shown as a scrolling ticker at the bottom of the Status tab.

## Visual Design

- Dark theme consistent with existing (`#0a0a0f` background, monospace fonts)
- Drawer uses same color palette: purple accents (`#8b5cf6`), muted text (`#999`)
- Tab icons are single unicode characters, not images
- Health indicators: green dot = available, red dot = down
- Agent cards: subtle border, strengths as colored tags
- Activity ticker: monospace, timestamps left-aligned, fades older entries

## Constraints

- Self-contained HTML+JS, no npm/build step
- No auth on cortex endpoints (LAN-only, read-only)
- No write operations from the visualizer
- Container rebuild required to deploy UI changes
