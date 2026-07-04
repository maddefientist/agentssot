# Hive/Cortex Remediation Roadmap — heeding the Fable verdict

**Date:** 2026-07-04
**Owner (custodian/architect):** Madi (Opus) — owns sequencing, the consolidation/rip-out decisions, and final verification of every dispatched unit.
**Source of truth:** `FABLE-HANDOFF-2026-07-04-cortex-architecture-review.md` (§9 risk register) + `FABLE-VERDICT-2026-07-04.md`.
**Execution model:** Opus drives; each coding unit is routed to a tier per the invariants below and dispatched with a full brief (writing-chain-briefs rubric). **No unit lands beside old code — every retrieval/ingest change is a consolidation, not an addition.**

---

## Routing invariants (which tier does what, and why)

| Tier | Use for | Never for |
|------|---------|-----------|
| **Opus (me)** | Architecture, sequencing, the RRF/legacy-kill/rip-out *decisions*, final verification of every unit | Bulk mechanical edits |
| **gpt-5.5** *(opted in)* | Correctness-critical **review** (auth, transactions, retrieval fusion) and cross-system planning a free planner can't hold. **Flags, does not fix.** | Routine edits |
| **sonnet-5** | Coding where judgment matters — auth path, retrieval RRF port, transaction atomicity, careful refactors | Boilerplate |
| **glm-5.2 (implement)** | Complete, well-specified plans with tests — bug fixes, endpoint edits, N+1 rewrites | Auth/money/txn logic without escalation review |
| **glm-5.2 (quick)** | Mechanical — deletions, narrow excepts, config, renames, single-file | Anything needing design |

**Hard rule (from CLAUDE.md escalation triggers):** anything touching **auth, transaction correctness, or the recall ranking math** gets a gpt-5.5 or sonnet-5 pass. GLM never lands auth/txn logic unreviewed. Every dispatch ends with Opus reading the diff + running the acceptance check — worker output is a claim, not proof.

**State-return guard:** after every glm/sonnet coder chain, run `state-return-check.sh` on its `progress.md` before accepting.

---

## Phase sequence (Fable's order: security → reranker+measure → consolidate → correctness → rip-out → depth)

```
P0 SECURITY (urgent, day-one)        ── Fable's real #1; Madi's §11 omitted it
P1 RERANKER OFF + MEASURE            ── the toggle IS the experiment
P2 RETRIEVAL CONSOLIDATION (RRF)     ── port RRF to live path, kill legacy
P3 CORRECTNESS BUGS                  ── the silent flywheel-killers
P4 DE-ACCRETION (rip-out)            ── Opus-driven; the un-fun custodian pass
P5 DEPTH: self-model v0 (metrics)    ── deferred; needs brainstorm, not code yet
```

---

## P0 — SECURITY (urgent)

> Rationale: unauthenticated endpoints leak concept content, agent keys, and raw recall query_text to the whole LAN; there is no working key revocation. This is live exposure. Do it first.

**S1 — Authenticate the public data endpoints.** `[route: sonnet-5 → gpt-5.5 review]`
- Files: `main.py` — `cortex_data:643`, `cortex_activity:733`, `cortex_system_info:674`, `cortex_links:663`, `dashboard_stats:780`, `doctor:472`.
- Do: add `require_api_key`/`require_role(reader)` deps. **Blocker to resolve first:** the HUD/console reads these unauth today — thread the agent's key into the HUD fetch calls (`ui/*.js`) OR mint a scoped read-only key the HUD uses. Sonnet decides the cleaner path and wires it.
- Acceptance: `curl` without key → 401 on all six; HUD still renders with key. Out of scope: changing what the endpoints return.

**S2 — Delete the cortex `ssot_` auth bypass.** `[route: sonnet-5]`
- File: `cortex.py:44-55`. Replace the "any `ssot_`-prefixed string" path with the real `require_api_key` — the 1h auth cache (`security.py:24`) already solves the bcrypt-cost concern the bypass was invented for. Keep the fast path = cache hit, not "no check."
- Acceptance: a bogus `ssot_xxx` key → 401 on `/cortex/*`; a real key still fast (cache hit, no per-call bcrypt).

**S3 — Real key revocation.** `[route: sonnet-5 → gpt-5.5 review]`
- Files: `main.py` api-key admin routes; `security.py:140` (`clear_auth_cache`). Add a deactivate/revoke endpoint (`is_active=false`) that **calls `clear_auth_cache()`**; ensure every key mutation (role/namespace change, deactivate) clears the cache. Currently only 2 call sites, neither on revocation.
- Acceptance: revoke a key → next request with it → 401 immediately (not after 3600s).

**S4 — MCP credential hygiene.** `[route: glm-5.2 implement → gpt-5.5 review]`
- File: `plugin/mcp_server.py`. Remove plaintext key from `hive_create_key` output (`:621` — return a masked confirmation + one-time reveal path); stop defaulting all tools to `admin_api_key` (`:33`); make the missing-`admin.json` case **fail loudly** instead of silently falling back to the writer key (`:72`).
- Acceptance: `hive_create_key` output contains no full key; non-admin config with no admin.json → explicit "admin creds required" error.

**S5 — secret_scanner false-positive tightening.** `[route: glm-5.2 implement]`
- File: `secret_scanner.py:152,194`. Constrain `base58_private_key`/`private_key_hex` so long hashes/IDs/data-URIs don't 422 whole ingest batches (require assignment context or known prefixes).
- Acceptance: a 64-char hex git SHA and a base64 data-URI ingest cleanly; a real `0x`-key assignment still rejects.

**S6 — Bind/exposure review.** `[route: Opus decision + glm-quick]`
- `docker-compose.yml` publishes on all host interfaces. Decide: bind to LAN/localhost + document, or accept (LAN service) once S1–S3 close the auth gap. Likely: auth is the real fix; add a firewall/bind note.

---

## P1 — RERANKER OFF + MEASUREMENT

> The reranker is an LLM-as-judge prompt hack (`reranker/ollama_provider.py:11-16`, `/api/generate` per candidate, parse-fail→0.0), 11–28s/recall, likely ~zero precision gain. Turning it off is both the fix and the experiment.

**R1 — Make the reranker toggleable and turn it off.** `[route: Opus + glm-quick]`
- Files: `runtime_config.py:HOT_KEYS` (add `reranker_provider`), then via `/admin/config` set `reranker_provider=none` (or empty the `ollama_reranker_*_model` keys). No restart.
- Acceptance: recall footer shows `reranker_used=false`; recall p50 drops to ~300ms; `rebuild_providers` picks it up live.

**R2 — Recall-quality measurement harness.** `[route: sonnet-5]`
- Files: new `scripts/recall_quality_report.py` reading `RecallEvent` + `ConceptFeedback` + profile counters. Report: recalls/day, feedback rate (already computable ≈11.6% = 108/936), useful-rate, and a before/after latency + usefulness delta around the R1 toggle.
- Acceptance: a runnable report producing the two-week baseline; documented so the reranker only re-enters on evidence.

---

## P2 — RETRIEVAL CONSOLIDATION (Opus-driven)

> The best retrieval feature (hybrid RRF) is dead code on the legacy path (`crud.py:423-461`); the live bucketed path is pure cosine (`knowledge.py`, no FTS/RRF). Port it forward, then delete the legacy stack. **This is the anti-accretion core — I drive it.**

**C1 — Port hybrid RRF into the live bucketed path.** `[route: sonnet-5 code → gpt-5.5 review → Opus verify]`
- Files: lift FTS+RRF logic from `crud.py:423-461` into `_recall_bucketed` (`knowledge.py:540`), per-tier. Keep `recall_hybrid_*` settings as the switches.
- Acceptance: bucketed recall fuses vector+FTS; exact tokens (error codes, flags, paths) that vector-only missed now surface; RRF k tunable. Out of scope: re-adding a reranker.

**C2 — Retire the legacy recall/ingest stack.** `[route: Opus decision + glm-implement mechanical]`
- After C1 proves out + a shim window: delete `crud.recall`/`ingest_batch` (legacy `/recall`,`/ingest`) or alias them to the tiered path. Collapse the three dedup definitions to one.
- Acceptance: one recall path, one ingest path, one dedup definition; tests green; MCP unaffected.

---

## P3 — CORRECTNESS BUGS

**B1 — Silent un-recallable ingest (flywheel-killer).** `[route: sonnet-5]`
- File: `knowledge.py:146-149`. Embed failure currently → `embedding=None` → item stored → filtered from recall forever ("taught it and it forgot"). Fix: on embed failure, reject (422) OR store + queue a backfill + flag; never silently unrecallable.
- Acceptance: embed-provider-down ingest either errors clearly or the item is later backfilled and recallable; a test proves it.

**B2 — Non-atomic tiered ingest.** `[route: sonnet-5 → gpt-5.5 review]`
- File: `knowledge.py:304,332,382,411` (up to 4 commits/ingest). Wrap in one transaction (SAVEPOINTs for optional sub-steps).
- Acceptance: a crash mid-ingest leaves no half-applied item; test with a forced mid-ingest failure.

**B3 — Doctrine rotation: UTC + stable index.** `[route: glm-implement → review]`
- File: `loadout.py:87`. Replace server-local `date.today()` with UTC; replace OFFSET-over-created_at with a stable per-item schedule (hash of item id + date, or a rotation cursor).
- Acceptance: rotation deterministic under insert/delete; flips at UTC midnight.

**B4 — lifecycle sweep: all namespaces + honest stubs.** `[route: sonnet-5]`
- Files: `background.py:106` (hardcoded `claude-shared`), `lifecycle_sweep.py:75` (stubs 3&4 return 0). Sweep all namespaces; either implement contradiction/supersession recheck or remove the advertised-but-fake steps.
- Acceptance: multi-namespace decay/expire runs; no method claims work it doesn't do.

**B5 — Mechanical correctness batch.** `[route: glm-implement, one brief]`
- `reconciler.py:57` `except (ValueError, Exception)` → narrow to real cases.
- `cortex.py:213-274` upsert → `ON CONFLICT`.
- `crud.py:385` return the score you ranked on (or document the split).
- `create_concept_feedback` embed call → guard for provider availability (avoid 500).
- Acceptance: each has a focused test; no behavior regressions.

**B6 — N+1 sweep.** `[route: glm-implement]`
- `entities.py:33` (per-entity count, also namespace-unscoped — fix both), `crud.py:1462` concept-history walk, `crud.py:170` per-chunk dup COUNT, `crud.py:1180/1206` delete loops, `crud.py:1542` + auth `_lookup_api_key:108` O(N) bcrypt loops (index by key prefix or lookup table).
- Acceptance: each rewritten to a set-based query; benchmarked on current corpus.

---

## P4 — DE-ACCRETION / RIP-OUT (Opus-driven)

> Fable: "mis-built in breadth." Remove what doesn't earn its keep. I make these calls directly; glm executes mechanical deletions.

- **D1 — Remove `sync.py`** (off by default, device/source conflation, dead count-trick, unused setting) unless a concrete cross-fleet need is named. `[Opus decide → glm-quick delete + route unmount]`
- **D2 — Gateway stubs:** remove or finish `DeferredBriefingExecutor`, `DispatchExecutor` v1, `chains=None`, dead `kind=="chain"` orchestrate branch, `teach_fn=None` dead-end. `[Opus decide → glm-quick]`
- **D3 — Dead code:** `clustering.py:11-12` dead defaults, `resurrect_concept` (uncalled), `sync.py:361` limit-0 trick. `[glm-quick]`
- **D4 — God-module + magic numbers:** split `main.py` (extract the two inline bash scripts + settings-metadata dicts + route groups); centralize scattered confidence/threshold/TTL constants into one module. `[route: sonnet-5, careful refactor — behavior-preserving]`
- **D5 — GPU:** unload the 8B/4B reranker models from Ollama once P1 lands. `[Opus/ops]`

---

## P5 — DEPTH: self-model v0 (deferred — needs brainstorm, not code)

> Fable's sharpening of F2: a self-model needs a *performance signal about itself*, which doesn't exist yet. **F2 v0 is a metrics loop, not doctrine prose.** P1-R2's harness is its seed.
- Do NOT build doctrine-prose self-model now. First: the scoreboard (R2). Then brainstorm (superpowers:brainstorming, with operator) whether the synthesis loop should read that scoreboard and set improvement goals.
- Gate: only after P0–P3 make recall fast, secure, and measured.

---

## Cross-cutting (fold into the phases above)
- **Swallowed-exception audit** — the swallow-and-continue pattern (embeddings→None, FTS, rerank, synapse, WAL) hides failures. Add metric/WAL breadcrumbs where correctness degrades silently. `[glm-implement, alongside B-series]`
- **Doc drift** — rewrite `README.md` to match reality (auth model, `/` = HUD, the `/api/v1/knowledge/*` default path, no-chunking-on-tiered). `[route: writer]` — last, after code settles.

---

## Dispatch discipline (every unit)
1. Opus expands the unit into a full brief (branch, files+lines, exact edits, imports, acceptance, **out-of-scope/forbidden**, verify step) via writing-chain-briefs.
2. Route to the tier per the table; escalate auth/txn/retrieval to sonnet-5 + gpt-5.5 review.
3. On return: `state-return-check.sh`, Opus reads the diff, runs the acceptance check.
4. Land via `/land` (no force push). One unit = one focused commit.
5. Parallel units touching the same files → separate worktrees.
