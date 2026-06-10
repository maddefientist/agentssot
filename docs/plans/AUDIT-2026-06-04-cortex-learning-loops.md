# Cortex Audit — Is It Actually Learning?

**Date:** 2026-06-04
**Auditor:** Madi (Opus, MacBook Pro orchestrator)
**Trigger:** Operator observation — "things don't seem labeled right, or organized... makes me wonder if we're effectively learning, improving, optimizing, self-growing."
**Method:** Live diagnostics via hive API (no SSH). `hive_doctor`, `hive_status`, `hive_stats`, `hive_profile`, `synapse_status`, `hive_review_queue`, `hive_recall`, `hive_query`.

---

## Verdict (one line)

**The cortex is a healthy warehouse, not yet a brain.** Ingestion and retrieval — the hard 80% — work well. The loops that make it *learn, curate, and grow* (synthesis, feedback training, contradiction resolution, synapse association, and any self-model) are dormant, miscalibrated, or pointed at a dead model. Your intuition is correct and the data backs it precisely: **it accumulates but does not curate, so browsing it feels like noise.**

---

## What's healthy (the 80% you built)

| Signal | Value | Read |
|---|---|---|
| Service health | embedding / LLM / reranker / synthesis flag all **OK/ON** | plumbing alive |
| Corpus size | 7689 vectors · 6477 knowledge items · 1000 concepts | substantial |
| Reach | 122 namespaces · 140 active keys | fleet-wide adoption |
| Recall latency | vec 70ms · rerank 17ms | fast |
| Last ingest | 2026-06-04 (today) | ingest pipeline live |

Retrieval is genuinely good. This is not the problem.

---

## Findings — ranked by severity

### P0 — Self-growth loop is likely broken

**F1. Synthesis points at a RETIRED model.**
`hive_status` → `Synthesis model: qwen3.5:cloud`. Your own `CLAUDE.md` marks `qwen3.5:cloud` as **retired**, replaced by `qwen3.5:397b-cloud`. Synthesis (nightly, 3:00 UTC) is the autonomic consolidation step — the closest thing the system has to self-growth. If it's calling a dead model it is silently failing or degrading nightly.
*Evidence it once worked:* a `synthesis-promotion [auto, doctrine-promoted]` item exists in the corpus. So promotion HAS happened — question is whether it still runs.
**→ Highest-value thing to verify first.**

**F2. No self-model.**
`hive_recall("how should Madi behave and improve over time", scope=concepts)` returned **zero** items about Madi, its behavior, goals, or improvement — just unrelated ops commands (SSH into natplay, Pour App, VectorData). The system stores facts about *projects* but holds no representation of *itself*. **There is nothing for it to grow toward because it doesn't model itself.** This is the root of the "is it self-growing?" question: not "is the loop running" but "is there a self for the loop to act on."

### P1 — Curation loops dormant / unattended

**F3. Review queue unattended and half-functioning.**
50 pending, **all** `contradiction` from a single "backfill contradiction sweep," with heavy duplication (primary `c545d6c1` queued 5×, `7916007a` 3×). `kind=dup` returns **empty** — so the dedup detector isn't queueing, only the one-off contradiction sweep did, and **nothing drains it.** Self-curation mechanism exists; no human or agent works it → contradictions accumulate → recall quality erodes over time.

**F4. Feedback loop barely fed.**
Profile: **282 feedback / 1641 recalls ≈ 17%.** `CLAUDE.md` mandates `hive_feedback` after every useful recall. At 17% the relevance ranking trains on a thin, biased signal — the loop is wired but starved. (Likely cause: the mandate isn't mechanically enforced; it relies on agent discipline.)

**F5. Synapse (associative / "wonder" layer) is dormant.**
`synapse_status` on this host: `local_enabled=false`, `flag_missing`, `listener_daemon=not_installed`, `active_sessions_visible=0`. The cross-linking layer that would make the cortex feel associative rather than a flat filing cabinet is **off on the primary orchestrator machine.**

### P2 — Labeling / hygiene (the visible symptom you noticed)

**F6. Type taxonomy exists but is unenforced.**
Recall buckets by type — `[command] [rule] [skill] [entity] [decision] [concept]` — so structure is real. But application is loose: a *"weekly status update"* filed as `[decision]`, *"clarification required when typos occur"* as a `[rule]`. **The taxonomy is sound; the discipline filling it is not.** This is exactly why browsing feels disorganized.

**F7. Concepts pinned at exactly 1000.**
Suspiciously round — almost certainly a hard cap, not an organic count. If capped while the corpus grows, new concepts are either rejected or silently evict old ones. Needs confirmation.

**F8. Per-device scoping unused.**
`device-macbook-pro-private` = 0 concepts / 0 items. Everything dumps into `claude-shared`. No private/shared discipline → one undifferentiated pool.

**F9. Stale profile timestamp.**
Profile "Last Active: 2026-02-24" despite 1641 recalls and ingest today. Activity/strength tracking is frozen — another "labeled wrong" symptom; means profile-derived strengths are stale.

**F10. Config display inconsistency.**
`hive_doctor` → reranker `Qwen3-Reranker-4B:Q4_K_M`. Recall footer → `qwen3-reranker-8b`. Either two rerankers in play or a stale display. Reconcile.

---

## The shape of it

```
INGEST  ✅ alive (last ingest today)
STORE   ✅ 7689 vectors, fast
RECALL  ✅ 70ms vec / 17ms rerank, returns results
        ──────────────────────────────────────────
FEEDBACK ⚠️  wired but starved (17%)            ← loop exists, not fed
SYNTHESIS 🔴 points at retired model            ← growth likely failing
CURATE   🔴 review queue unattended, dedup silent ← rot accumulates
ASSOCIATE 🔴 synapse dormant on this host        ← no "wonder" layer
SELF-MODEL 🔴 does not exist                      ← nothing to grow toward
```

The arrows that turn a warehouse into a brain are the broken ones.

---

## Recommended pickup order (for the on-box agent at `hari:/opt/agentssot`)

Cheap verifications first — confirm before fixing:

1. **Synthesis (F1):** find synthesis config + last run logs. Is the 3:00 UTC job firing? Is it erroring on `qwen3.5:cloud`? Grep `synthesis/loop.py`, scheduler config, and recent run logs. Repoint to `qwen3.5:397b-cloud` and confirm a clean run.
2. **Concept cap (F7):** is 1000 a hard limit in code/config? What happens at the ceiling — reject or evict?
3. **Review queue (F3):** what populates it, what's meant to drain it? Is there a reconciler job? Why is `dup` empty while `contradiction` has 50? Build/restore the drain path.
4. **Feedback (F4):** can `hive_feedback` be made mechanical (auto-fire on recall-then-use) instead of relying on agent discipline?
5. **Self-model (F2):** design decision, not a quick fix — should Madi hold a first-class self-concept (identity, behavioral doctrine, improvement goals) as a dedicated namespace/scope the synthesis loop reads and updates? **This is the brainstorm-worthy thread.**
6. **Synapse (F5):** enable on this host (`~/.claude/agentssot/local/agent.json` → `synapse_enabled: true`) and confirm listener.
7. **Hygiene (F6, F8, F9, F10):** type-label enforcement at teach time; private/shared scoping policy; fix Last-Active update; reconcile reranker name.

---

## Note on scope

P0/P1 are operational fixes (verify → repair) and fit a chain/worker flow once the on-box agent confirms root causes. **F2 (self-model) is a design question, not a repair** — it deserves a proper brainstorm before any build, because it changes what the cortex *is*, not just whether a job runs. Flagging it so we don't reflexively "fix" it into existence.
