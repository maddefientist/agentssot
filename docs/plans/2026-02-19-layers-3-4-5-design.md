# Layers 3, 4, 5: Skills, Personalization, Collective Intelligence — Design Document

## Mission

Complete the neural network architecture. Layer 3 teaches the brain how to act (not just what it knows). Layer 4 makes recall aware of who's asking. Layer 5 strengthens concepts confirmed by multiple independent agents. All learning runs on Ollama — zero Claude tokens.

## Layer 3: Skills — Actionable Knowledge

### Problem
Current concepts are descriptive: "Docker Compose is the deployment tool." The brain observes but never advises. Skills are prescriptive: "When deploying a new service, check port conflicts first."

### Data Model Changes

**New ConceptType enum value:** `skill` (alongside mental_model, relationship, principle)

**New nullable columns on Concept:**

| Column | Type | Description |
|--------|------|-------------|
| trigger | TEXT nullable | When this skill activates ("deploying a service") |
| action | TEXT nullable | What to do ("check docker-compose.yml for port conflicts") |
| success_hint | TEXT nullable | How to know it worked ("service starts without errors") |

Regular concepts keep these NULL. Skills have all three populated.

### Skill Sources

**1. Auto-discovered:** Synthesis LLM prompt extended — when it sees repeated action patterns across 3+ sessions (e.g., "user checked ports before deploy" recurring), it proposes `type: "skill"` with trigger/action/success_hint.

**2. Manually taught:** New `hive_teach` MCP tool:
```
hive_teach(trigger, action, success_hint?)
→ Creates skill concept directly, confidence 0.8 (operator-taught = high trust)
```

### Skill Surfacing

- MCP plugin formats skill-type concepts differently — trigger and action prominent
- Skills get a boost in reranking when query matches trigger text
- RecallItem includes skill fields when scope is "concepts" or "all"

### Synthesis Prompt Extension

Add to the existing synthesis system prompt:
```
Additionally, if you observe repeated action patterns across the facts (the same
action taken in similar situations 3+ times), propose a skill:
- type: "skill"
- trigger: when this skill should activate (situation description)
- action: what to do (specific steps)
- success_hint: how to verify it worked (optional)
```

---

## Layer 4: Personalization — Agent Profiles

### Problem
Every enrolled agent gets identical recall results. An agent on Hari (GPU workstation) and an agent on Air (laptop) have different contexts, strengths, and usage patterns.

### Data Model

**New table: `agent_profiles`**

| Column | Type | Description |
|--------|------|-------------|
| agent_key | TEXT PK | Matches ApiKey.name (e.g. "device-hari-writer") |
| namespace | TEXT | Primary namespace |
| device_name | TEXT | Extracted from key name (e.g. "hari") |
| model_hint | TEXT nullable | Last known model (e.g. "claude-opus", "kimi-k2.5") |
| strengths | TEXT[] | Auto-learned topic strengths (e.g. ["docker", "python"]) |
| preferences | JSONB | Auto-learned preferences (e.g. {"verbosity": "concise"}) |
| total_recalls | INT default 0 | Lifetime recall count |
| total_feedback | INT default 0 | Lifetime feedback count |
| created_at | TIMESTAMPTZ | |
| updated_at | TIMESTAMPTZ | |

### Profile Building (automatic, zero manual setup)

A new synthesis subtask runs after concept synthesis:
1. Aggregate recall_events by agent_key — extract top query topics
2. Aggregate concept_feedback by agent_key — extract preference patterns
3. Update agent_profiles with new strengths and preferences
4. All runs on Ollama — zero Claude tokens

### Recall Personalization

When `agent_key` is provided on recall:
- Boost concepts aligned with agent's strengths (+10-15% score adjustment)
- Skills with triggers matching agent's strength areas rank higher
- Light touch — no hard filters, agent still sees everything, just sorted smarter

### MCP Integration

- `hive_recall` already passes agent_key — personalization is transparent
- New `hive_profile` MCP tool to view/inspect your agent's learned profile
- Profiles auto-populate from first recall — no enrollment changes needed

---

## Layer 5: Collective Intelligence — Cross-Agent Confirmation

### Problem
If Claude on Hari and Kimi on Air independently confirm the same pattern, that's a stronger signal than either alone. Currently there's no cross-agent attribution.

### Data Model Changes

**New column on Concept:** `confirming_agents TEXT[]` — agent_keys that independently confirmed this concept.

### Confirmation Logic (in reconciler)

- When synthesis from device-hari-writer's data reinforces a concept → add agent_key to confirming_agents
- When feedback from device-air-writer marks a concept "useful" → add agent_key
- **Confidence bonus:** +0.05 per unique confirming agent (beyond the first)
- **Consensus tag:** 3+ confirming agents → tag `"consensus"` (highest trust tier)

### Cross-Agent Deduplication

When two agents produce near-identical concepts independently:
- Clustering already matches proposals against existing concepts
- If matched, merge and mark both agents as confirmers
- Merged concept gets combined evidence_ids from both

### Visualization (cortex)

- Consensus concepts glow brighter (additional visual weight)
- Node tooltip shows confirming agents list
- Future live feed: confirmation pulse when new agent validates existing concept

---

## Implementation Scope

### Layer 3 (Skills)
1. Add `skill` to ConceptType enum + DB migration
2. Add trigger/action/success_hint columns to concepts table
3. Extend synthesis prompt with skill detection
4. Add `hive_teach` MCP tool
5. Update MCP recall formatting for skill-type concepts
6. Update cortex visualization with skill node type

### Layer 4 (Personalization)
7. Create agent_profiles table
8. Add profile-building subtask to synthesis loop
9. Add recall score boosting by agent strengths
10. Add `hive_profile` MCP tool

### Layer 5 (Collective Intelligence)
11. Add confirming_agents column to concepts
12. Update reconciler: track agent attribution on reinforcement
13. Update feedback processing: add agent to confirmers on positive signal
14. Add consensus tagging logic (3+ agents)
15. Update cortex visualization for consensus nodes

### Cross-cutting
16. Update synthesis prompt for all new fields
17. Integration testing — full cycle across all layers
