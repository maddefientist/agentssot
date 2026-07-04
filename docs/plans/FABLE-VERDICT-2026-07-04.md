# Fable Verdict — Hive/Cortex Architecture Review

**Date:** 2026-07-04
**Reviewer:** Fable (claude-fable-5), escalation tier
**Inputs:** FABLE-HANDOFF-2026-07-04 (read in full) + direct code verification of every load-bearing claim I cite. Anything marked [VERIFIED-F] I read in the source myself this session.

---

## 0. Executive verdict

The substrate is good and the diagnosis ("healthy warehouse, not yet a brain") is correct. But the handoff — and Madi's strategic assessment — are both built on one wrong fact that changes the sequencing: **the production recall path has no hybrid RRF, and the "reranker" is not a cross-encoder at all.** It is an 8B generative LLM prompted with "Relevance score (0-1):" and regex-parsed, called once per candidate, per tier, sequentially. The 11–28s on every session start is very plausibly buying *zero or negative* precision. That converts the reranker question from "which model do we migrate to" into "turn it off today and measure."

Ranked, the system's real order of problems is: (1) it is an open data plane on the LAN, (2) its hot path is slow for no demonstrated benefit, (3) nobody measures whether it helps, (4) it accretes parallel implementations faster than it consolidates, (5) only then, the self-model. Madi got 3, 4, 5 right and materially under-ranked 1 and misdiagnosed 2.

---

## 1. Adjudication of Madi's strategic assessment (§11)

### (a) "Impressive ≠ serving; nobody has measured the flywheel" — **AGREE, and it's cheaper to fix than Madi implies**

Correct and the most important sentence in the handoff. But "measure for one honest week" overstates the build cost: the measurement substrate already exists. `RecallEvent`, `ConceptFeedback`, per-recall `vec_ms`/`rerank_ms` diagnostics, and loadout token counts are already persisted. The live numbers in the handoff itself are a first datapoint: 936 recalls / 108 feedback ≈ **11.6% feedback rate** — i.e. ~88% of recalls produce no signal about whether they helped. The flywheel question is mostly a week of SQL over tables you already have, plus one missing instrument: log which loadout/recall items subsequently get *used* (feedback, expand, re-recall). Do not build measurement infrastructure; write queries.

Sharpening: the loadout economy question (§10.3) is measurable the same way. "ALL rules, every session" (`services/loadout.py:69-72`) is an unbounded token tax that grows linearly with rule count, while the best doctrine rotates past at 1/day on an unstable index. That is exactly backwards for token economy, and no one can currently see it because injected-vs-used is not tracked.

### (b) "The disease is accretion, not any single bug" — **AGREE, strongly, with a sharper exhibit**

The best proof was missed by everyone: **hybrid RRF — a genuine retrieval improvement — was landed on the legacy path (`crud.py:423-461`) and never reached the default production path (`knowledge.py:540-658`), which is pure cosine + rerank** [VERIFIED-F]. The improvement landed *beside* the hot path, not *in* it. This is the accretion failure mode producing a concrete quality regression on the path every session uses, and it silently corrupted the downstream reasoning: Appendix B's Path C premise ("hybrid RRF already present") is false for the path that matters. Accretion doesn't just cost maintenance — it is now generating wrong strategic inputs.

### (c) "Self-model (F2) is premature" — **AGREE, but the reason is sharper than corpus hygiene**

Madi's argument (uncurated substrate → confident noise about itself) is right but secondary. The primary blocker: **a self-model needs a performance signal about itself, and none exists.** With an 11.6% feedback rate and no injected-vs-used tracking, a self-model would introspect over a corpus that contains no ground truth about whether the system is serving well. It would synthesize identity from volume, not from outcome. The minimal viable F2 is therefore not doctrine prose — it is a metrics loop: weekly synthesis over the system's *own* RecallEvent/feedback/loadout data into a `hive-self` namespace, producing claims that are checkable against next week's numbers. Build the instrument first; the self-model is the instrument reading itself.

### (d) "Bug-hunt through the parallel pipeline deepens accretion" — **AGREE with a carve-out**

Correct for anything structural: consolidation of the two stacks must be one custodian with the whole system in context, not N chains with local briefs — chains are how you got three definitions of "duplicate." But the carve-out matters: the security batch (§3 below) is small, isolated, brief-able work with crisp acceptance criteria. Farming *that* out is fine and fast. The rule is not "no chains until consolidation"; it is "no chains on anything that touches the recall/ingest/curation core until the core is singular."

### (e) "Sequence: measure → reranker → custodian → build" — **SHARPEN: the order is wrong at both ends**

Two corrections:

1. **Security goes first, not unranked.** Madi's sequence omits it entirely. This box exposes concept content, agent keys, and raw recall query text to any LAN caller with no key (`main.py:643,733,780`, plus `/doctor` at `:472` — also unauthenticated [VERIFIED-F]); the container binds `0.0.0.0` by default (`docker-compose.yml:36` [VERIFIED-F]); the cortex write plane accepts **any string starting with `ssot_`** (`cortex.py:53` [VERIFIED-F]); and there is a cloudflared tunnel host on this LAN. This is a day of work. It goes before everything.
2. **"Measure" and "fix the reranker" are the same act, not two phases.** The reranker is behind a config flag (`reranker_provider`, HOT_KEYS-controllable). Turning it off *is* the experiment: run a week rerank-off, compare feedback/usefulness rates against the rerank-on history you already have. Madi frames the reranker as a fight requiring landscape research; it's a toggle plus a SQL query.

### Where Madi is simply wrong

- **Treating the reranker as a legitimate heavyweight cross-encoder.** It isn't (§2). The entire Appendix B framing — INT8 quantization, model-size ladders, nDCG deltas from published cross-encoder benchmarks — assumes real reranker inference. The implementation is a prompt hack whose scores are likely off-distribution for the model. Published benchmark deltas do not transfer to this implementation at all.
- **"Hybrid RRF already present"** — false on the production path (§1b). Path C's cost estimate must include porting RRF to `_recall_bucketed` (~a day).
- **Ranking the reranker above the open data plane.** On a multi-host LAN with agent keys leaking via unauthenticated endpoints and a prefix-string write plane, §9 #2/#5 outrank #1.

---

## 2. The reranker/retrieval verdict

### The finding that reframes everything [VERIFIED-F]

`api/app/reranker/ollama_provider.py:11-16,37-65`:

```python
_PROMPT_TEMPLATE = (
    "Given a query and a document, determine if the document is relevant.\n\n"
    "Query: {query}\nDocument: {document}\n\nRelevance score (0-1):"
)
...
"options": {"num_predict": 5, "temperature": 0},
...
match = _SCORE_RE.search(raw)   # r"([01](?:\.\d+)?)"
if match: return min(max(float(match.group(1)), 0.0), 1.0)
logger.warning(...); return 0.0
```

Five compounding problems:

1. **Not a cross-encoder invocation.** One `/api/generate` call per candidate, generating free text, regex-parsed. Qwen3-Reranker models are trained to emit a yes/no judgment under a *specific instruction template* and be scored on logits. This prompt is off-distribution; the "scores" are whatever an 8B model freestyles after "Relevance score (0-1):".
2. **Parse failure silently buries items** — unparseable output → `0.0` → the item drops to the bottom (warning only). A relevant memory can be hidden by a formatting quirk.
3. **Coarse, tied scores.** LLMs asked for 0–1 emit 0.8/0.9/1.0 clusters; within ties the sort preserves prior (vector) order — so much of the reranker's "reordering" is the vector order it started with, at 1000× the cost.
4. **Per-tier sequential loop** (`knowledge.py:592-624`): a default recall reranks up to 5 tiers × (top_k×multiplier) candidates = 50–75 8B generate calls, threadpooled at 8 but gated by `OLLAMA_NUM_PARALLEL`. That is the whole 11–28s.
5. **Pools are too shallow for reranking to matter anyway.** Reranker wins in the literature come from re-scoring 50–100-deep pools. Here it re-scores 10–15 candidates per tier that pgvector already ordered over a tier-filtered slice of a 7.4k corpus. Even a *perfect* reranker has almost no room to improve top-5 from a 10–15 pool.

**Conclusion: the 11–28s is very likely buying nothing, and via failure-mode (2), sometimes buying harm.**

### The pick: modified Path C — "drop from hot path, port RRF, measure, earn re-entry"

- **Path A/B (0.6B/4B + INT8) rejected as stated** — they keep the prompt-parse architecture (Ollama has no rerank endpoint; A/B via Ollama is the same hack with a smaller model) and keep a cross-encoder where the candidate geometry can't reward one. If a real reranker is ever justified by measurement, it must be served by a runtime that does logit/rerank scoring properly (TEI, Infinity, llama.cpp scoring) — not `/api/generate` — and it should run as an *optional deep mode*, never the session-start default.
- **Path C accepted with corrections:** RRF is not "already present" on the hot path — port it first. Embedding upgrade (BGE-M3 or similar) is step 3, only if measurement shows a recall-quality gap; `nomic-embed-text` at 7.4k typed-filtered items may be entirely adequate.

**Concrete first moves, in order:**

1. **Today:** turn the reranker off. The gate is clean — `build_reranker_pair` returns disabled stubs when `reranker_provider != "ollama"` (`reranker/router.py:30-33` [VERIFIED-F]) and `_recall_bucketed` skips rerank when unavailable. Caveat [VERIFIED-F]: `reranker_provider` is **not** in HOT_KEYS (`runtime_config.py:18-28`) — the env flag needs a restart. The no-restart alternative: set `ollama_reranker_model` and `ollama_reranker_fast_model` to empty via `/admin/config` (both ARE hot keys), which makes both providers `is_available=False` → `pick_reranker` returns the disabled stub. Either way, recall drops to ~30–350ms immediately. Add `reranker_provider` to HOT_KEYS while there.
2. **This week:** port the RRF/FTS fusion from `crud.py:423-461` into `_recall_bucketed` (per-tier: vector track + FTS track, fuse, take top_k). Delete the legacy weighted path once ported (§4).
3. **Two weeks of data:** compare feedback/useful rates and recall-abandonment before/after against the existing `RecallEvent` history. If (and only if) a measured precision gap appears on nuanced tiers, evaluate a properly-served 0.6B reranker as an opt-in `deep=true` flag with a widened candidate pool (top_k×10), because that's the only geometry where it can pay.

This is the lean/fast/cheap answer the North Star demands: sub-second recall at ~$0, one retrieval stack, reranking readmitted only on evidence.

---

## 3. Risk register verification + what was missed

### Register adjudication (§9)

| # | Verdict | Notes |
|---|---|---|
| 1 | **CONFIRMED, worse than stated** | Not a cross-encoder (§2); per-tier ×5 loop; parse-fail→0.0 buries items; bare `reranker.rerank` blocks the event loop at `knowledge.py:616` [VERIFIED-F] |
| 2 | **CONFIRMED, promote to #1** | [VERIFIED-F] `cortex_data`/`cortex_links`/`cortex_activity`/`dashboard_stats` take only `get_session`; **add `/doctor` (`main.py:472`) — also unauth**; `0.0.0.0` bind (`docker-compose.yml:36`) makes it whole-LAN |
| 3 | **CONFIRMED, worse: there is NO revocation endpoint at all** | [VERIFIED-F] `clear_auth_cache` called only from namespace-create (`main.py:1237`) and grant (`:1350`). No deactivate/revoke route exists anywhere — revocation is manual DB surgery that the 1h cache then ignores. Merge #11 into this as one auth work item |
| 4 | **CONFIRMED with sharper evidence** | RRF landed only on the legacy path — the accretion exhibit (§1b) |
| 5 | **CONFIRMED** | [VERIFIED-F] `cortex.py:51-55`; the "constant-time comparison" comment is false (plain `==`); the whole fast path is **unnecessary** — see missed finding N4 |
| 6 | **CONFIRMED** | [VERIFIED-F] `mcp_server.py:33` (admin-as-default), `:69-73` (silent fallback), `:620` (plaintext key in tool output) |
| 7 | **CONFIRMED** | [VERIFIED-F] `session.execute` bare in async `_recall_bucketed` (`knowledge.py:602`) |
| 8 | **CONFIRMED** | [VERIFIED-F] commits at `knowledge.py:304,332,382,411` |
| 9 | **CONFIRMED** | [VERIFIED-F] `loadout.py:87-93` — `date.today()` + OFFSET over `created_at` |
| 10,12–19 | Accepted on doc's code-read; ranks reasonable | #13 (secret-scanner FP blocks whole batch) deserves a nudge up — a blocked batch is silent memory loss from the agent's perspective |
| 20 | **CONFIRMED and slightly worse** | [VERIFIED-F] `crud.py:385,471` — the returned `score` is raw cosine **distance** (lower=better) labeled like a similarity; tiered path returns `1-distance`. Two paths, opposite score polarity, same field name |

**Re-ranked top five:** (1) open LAN data plane [old #2+#5+N5+N6], (2) reranker hot path [old #1, reframed §2], (3) no revocation path + cache ghost [old #3+#11], (4) two-stack accretion [old #4], (5) MCP key hygiene [old #6].

### New findings the passes missed

- **N1 — Production path has no hybrid retrieval** (`knowledge.py:592-627` is cosine-only) [VERIFIED-F]. High. Invalidates Appendix B's Path C premise; the best retrieval improvement in the codebase is dead code on the path that matters.
- **N2 — Reranker is a prompt-parse hack** (§2) [VERIFIED-F]. High. Changes the reranker decision from model-selection to removal.
- **N3 — Per-tier sequential rerank loop** multiplies the cost ×5 (`knowledge.py:592`) [VERIFIED-F]. Explains the latency magnitude on its own.
- **N4 — The cortex auth bypass solves a problem the auth cache already solved.** The justification comment ("bcrypt ~3s per call") ignores that `require_api_key` is a dict lookup after first verification (1h TTL cache, `security.py:128-136`). The correct fix is to delete `_require_cortex_key` and use the standard dependency — you get real auth at cache-hit cost. The entire "any `ssot_` prefix" hole exists to dodge a cost that doesn't exist.
- **N5 — `/doctor` unauthenticated** (`main.py:472`) [VERIFIED-F]. Leaks config/connectivity/model info. Low-Med, folds into the security batch.
- **N6 — `0.0.0.0` port bind by default** (`docker-compose.yml:36`) [VERIFIED-F]. Context multiplier for every unauth finding.
- **N7 — Silent un-recallable ingest.** Tiered ingest swallows embedding failure and stores the item with `embedding=None` (`knowledge.py:146-149`) [VERIFIED-F]; every recall filters `embedding.isnot(None)`. The memory is written, acknowledged, and permanently invisible — no review-queue row, no WAL marker, no metric. This is a flywheel-killer class of bug: it manufactures "I taught it that and it forgot" experiences. Should queue a `backfill` review item or fail loudly.
- **N8 — WAL persists full ingest payloads verbatim** (`knowledge.py:413-418` logs `data.model_dump()`; WAL redaction is key-name-only). Anything the ~30-pattern scanner misses lands in plaintext JSONL on disk with 30-day retention. Low-Med; document it and truncate/deny-list `content` in WAL payloads.
- **N9 — `reranker_candidate_multiplier or 3`** (`knowledge.py:589`): a deliberate 0 becomes 3, and the inline fallback disagrees with the settings default (2). Trivial, but it's the two-validators disease in miniature.

### False positives killed

None outright — the register is honest. The closest is #15 (sanitizer divergence), which the handoff already correctly downgraded from a prior overstated draft; I concur it is latent, not live, since the MCP path uses the bucketed route (verified: `mcp_server.py:158` posts to `/api/v1/knowledge/recall`).

---

## 4. Meta-judgment: over-built, under-built, or mis-built?

**Mis-built in breadth, under-built in depth.** The core substrate — Postgres+pgvector, the typed-memory taxonomy, tiered content layers, the loadout concept, WAL-as-audit, the secret-scan/sanitizer boundary pair, synthesis preflight — is genuinely good and appropriately lean. What's wrong is the ratio: eight subsystems ring a core loop (recall→use→feedback→synthesize→curate→loadout) that is itself slow, unmeasured, and duplicated. The North Star is a flywheel; flywheels fail on friction at the hub, not on missing spokes.

**The binding constraint on "partner" is trust density, not intelligence.** An 11–28s recall trains agents to avoid recalling. Silently un-recallable items (N7) train the operator that teaching doesn't stick. An 11.6% feedback rate means the learning loops are starving. Fix latency and observability and usage density rises; usage density is what synthesis, curation, and eventually a self-model feed on. No self-model can compensate for a hub agents route around.

### Invest (in order)

1. **Security batch (1 day):** auth (or 127.0.0.1/allowlist bind) on `cortex_*`, `dashboard/stats`, `/doctor`; delete `_require_cortex_key` in favor of the cached standard auth (N4); add a revoke endpoint that calls `clear_auth_cache()`; MCP: stop defaulting to admin key, stop silent admin→writer fallback.
2. **Hot path (1 week):** reranker off via HOT_KEYS; port RRF to `_recall_bucketed`; fix N7 (loud failure or auto-queued backfill); add injected-vs-used logging to loadout and recall.
3. **Measure (2 weeks, passive):** SQL over RecallEvent/feedback — usefulness rate, loadout hit rate, tokens injected vs expanded, rerank-off vs historical rerank-on. This answers §10.3 and §10.4 with data.
4. **Custodian consolidation (one agent, whole-system context, ~1 week):** single recall stack (bucketed, now with RRF), single ingest stack, one dedup definition, deprecate-and-delete the legacy `/recall`+`/ingest` after a compat shim window. Not chain-dispatched.
5. **Self-model v0 (only now):** weekly synthesis over the system's own metrics into a `hive-self` namespace — claims about its own performance that next week's numbers can falsify. That is a self-model with a truth signal; anything earlier is autobiography.

### Rip out / freeze

- **`sync.py`** — off by default, semantically broken (source/device conflation, dead code, unused settings). Delete; git remembers. It's 391 lines of future confusion for zero current value.
- **Legacy `/recall` + `/ingest` + `crud._recall_knowledge_weighted`** — after the RRF port and a deprecation window. This kills three of the top-ranked risks (score polarity, sanitizer divergence, dual dedup) as a side effect.
- **Gateway stub executors** — `teach_fn=None`, DeferredBriefing placeholder, dead `kind=="chain"` branch: either wire them or remove the router rules that point at them. A router that routes "remember X" to a dead end is worse than no route — it teaches the operator the HUD lies.
- **One decay system** — fold KnowledgeItem decay and Concept decay behind one configured policy module during the custodian pass (and un-hardcode the `claude-shared` sweep while there).
- **The 8B and 4B reranker models** from the GPU, pending §2's measurement. That VRAM is worth more to Madi/portfolio than to a reranker that may be subtracting value.

### The shortest credible path, warehouse → brain

Sub-second measured recall → rising usage density → real feedback volume → synthesis with signal → curation that keeps pace → self-model that reads its own scoreboard. Every step is ~$0, Ollama-local, and shrinks the codebase before growing it. The flywheel doesn't need a new subsystem to start turning; it needs the friction taken off the hub and a tachometer bolted on.
