# Hive/Cortex — Next Steps (post-remediation, 3-tier execution)

**As of 2026-07-05.** The Fable-verdict remediation (P0–P4) is **landed on `main`** (commits `f857ed2..766da45`). Full context: `FABLE-VERDICT-2026-07-04.md` + `REMEDIATION-2026-07-04-roadmap.md`. This file is the procedural queue to work through **in order** — Tier 1 gates everything.

## Shipped & landed (do not redo)
Security authn+authz on 6 leaking endpoints + cortex `ssot_` bypass removed + real key revocation · reranker_provider made a runtime HOT_KEY off-switch · **hybrid RRF ported into the live `_recall_bucketed` path** + DB/rerank offloaded via `to_thread` · ingest atomic + no more silent-unrecallable items · doctrine rotation UTC/deterministic · multi-namespace sweep + honest stubs · cortex `ON CONFLICT` upsert · N+1 kills · **`sync` subsystem ripped out**. Every auth/txn/retrieval change gpt-5.5-reviewed.

---

## TIER 1 — VALIDATE (do first; cheap; gates the rest)
1. **Deploy** landed `main` (`@766da45`) to the live service on hari `:8088`. No `.land.json` → manual deploy (rebuild/restart the container from clean master).
2. **Baseline measure:** `python scripts/recall_quality_report.py --probe --api-base http://192.168.1.225:8088 --api-key <key>` → record the VERDICT line.
3. **Flip the reranker OFF (live, no restart):** `POST /admin/config {"key":"reranker_provider","value":"none"}` with an admin key. Recall should drop ~11–28s → ~300ms; footer `reranker_used=false`.
4. **Re-measure** after a real usage window; **diff** the two VERDICT lines (recalls/day, feedback%, useful%, p50 latency, reranker_used).
5. **C2 — retire the legacy stack** *(only if bucketed+RRF is at parity)*: delete `crud.recall` + `ingest_batch` (legacy `/recall`,`/ingest`), collapse the 3 dedup definitions → 1.

## TIER 2 — RETRIEVAL UPGRADES (after Tier 1 data)
6. **Embedding migration decision:** `nomic-embed-text` → BGE-M3 or Qwen3-Embedding (re-embed ~7,439 items + dimension change). Decide from Tier 1 numbers — stronger first-stage recall lets us lean less on rerank permanently.
7. **Taxonomy enforcement (audit F6):** make classification mechanical without a heavy per-ingest LLM cost.

## TIER 3 — STRATEGIC (needs brainstorm/decision, not dispatch)
8. **Self-model / warehouse→brain (P5):** brainstorm first. It's a *metrics loop* seeded by the Tier-1 harness, not doctrine prose — premature until curation is trustworthy.
9. **Async-DB-on-event-loop debt:** proper offload / async driver. Scope **only if** Tier 1 shows DB latency actually matters post-reranker-off.
10. **P4b:** `main.py` god-module split + magic-number centralization (behavior-preserving; needs real runtime verification).

---
**Rule:** don't open Tier 2 or 3 until Tier 1 has run. The measurement is the fork — it says whether the reranker/embedding work is worth doing at all. The reranker is an LLM-judge `/api/generate` hack (11–28s); turning it off is both the fix and the experiment.
