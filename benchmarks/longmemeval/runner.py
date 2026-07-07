"""LongMemEval benchmark harness for agentssot.

Measures session-level recall@K: whether the recall endpoint returns at least one
KnowledgeItem whose source_ref matches a ground-truth answer_session_id.

LLM-judge QA accuracy is out of scope for this harness — session-level recall is
a fair, cheap proxy that exercises embedding + reranking + tiered retrieval.

Usage:
    python runner.py --config vector_only
    python runner.py --config with_reranker --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx
import yaml

HERE = Path(__file__).resolve().parent


def load_config(config_name: str) -> dict[str, Any]:
    with (HERE / "config.yaml").open() as f:
        raw = yaml.safe_load(f)
    merged = dict(raw["default"])
    cfg = raw["configs"].get(config_name)
    if cfg is None:
        raise SystemExit(
            f"config '{config_name}' not found. available: {list(raw['configs'])}"
        )
    merged.update(cfg)
    merged["config_name"] = config_name
    return merged


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


@dataclass
class SessionIngestResult:
    ingested: int
    failed: int
    duration_s: float
    errors: list[str] = field(default_factory=list)


# nomic-embed-text context is 2048 tokens (nomic-bert.context_length).
# At ~4 chars/token, cap at 7500 chars to stay safely within limits.
_MAX_CONTENT_CHARS = 7500


def session_to_text(session: list[dict]) -> str:
    """Flatten a list of {role, content} turns into a single document.

    Truncates to _MAX_CONTENT_CHARS to stay within embedding model limits.
    """
    parts = []
    for turn in session:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        parts.append(f"[{role}] {content}")
    text = "\n\n".join(parts)
    if len(text) > _MAX_CONTENT_CHARS:
        text = text[:_MAX_CONTENT_CHARS] + "...[truncated]"
    return text


def ingest_sessions(client: httpx.Client, cfg: dict, dataset: list[dict]) -> SessionIngestResult:
    """Ingest every haystack_session across every question.

    Each session gets a deterministic source_ref = f"{question_id}::{session_idx}"
    so we can match recall hits back to ground truth answer_session_ids.
    """
    namespace = cfg["namespace"]
    tag = cfg["tag"]

    start = time.time()
    ingested = 0
    failed = 0
    errors: list[str] = []

    seen_refs: set[str] = set()

    for q in dataset:
        qid = q["question_id"]
        sessions = q.get("haystack_sessions", [])
        session_ids = q.get("haystack_session_ids") or [f"{qid}_s{i}" for i in range(len(sessions))]

        for sid, session in zip(session_ids, sessions):
            ref = sid
            if ref in seen_refs:
                continue
            seen_refs.add(ref)

            payload = {
                "content": session_to_text(session),
                "namespace": namespace,
                "source": "longmemeval",
                "source_ref": ref,
                "tags": [tag],
                "generate_summaries": False,
            }
            try:
                r = client.post("/api/v1/knowledge/ingest", json=payload, timeout=60)
                if r.status_code >= 300:
                    failed += 1
                    if len(errors) < 5:
                        errors.append(f"{ref}: {r.status_code} {r.text[:120]}")
                else:
                    ingested += 1
            except Exception as e:
                failed += 1
                if len(errors) < 5:
                    errors.append(f"{ref}: {e!r}")

    return SessionIngestResult(
        ingested=ingested,
        failed=failed,
        duration_s=round(time.time() - start, 2),
        errors=errors,
    )


@dataclass
class QueryOutcome:
    question_id: str
    question_type: str | None
    hit_at: dict[int, bool]  # {K: hit}
    latency_ms: float
    expected: list[str]
    returned: list[str]
    reranker_used: str | None = None


def _merge_buckets(buckets: dict[str, list[dict]]) -> list[dict]:
    """Flatten all per-tier buckets into one list ranked by score desc.

    The bucketed endpoint returns {tier: [item, ...]}; each item carries its own
    score (rerank score when the reranker is on, else vector similarity). We fuse
    across tiers by score so recall@k is measured over the single ranked list the
    caller would actually see.
    """
    merged: list[dict] = []
    for items in buckets.values():
        merged.extend(items)
    merged.sort(key=lambda it: it.get("score", 0.0), reverse=True)
    return merged


def run_queries(client: httpx.Client, cfg: dict, dataset: list[dict]) -> list[QueryOutcome]:
    namespace = cfg["namespace"]
    ks = sorted(cfg["top_k"])
    max_k = max(ks)
    expand_layer = cfg.get("layer_preference", "full")
    tiers = cfg.get("recall_tiers") or ["episodic", "fact"]
    exclude_episodic = bool(cfg.get("exclude_episodic", False))
    top_per_tier = {tier: max_k for tier in tiers}

    outcomes: list[QueryOutcome] = []

    for q in dataset:
        qid = q["question_id"]
        question = q["question"]
        expected = q.get("answer_session_ids") or []
        if not expected:
            # Skip malformed questions.
            continue
        expected_set = set(expected)

        # Bucketed recall schema (BucketedRecallRequest). source_ref now rides
        # on each returned item, so we match hits back to ground truth directly.
        payload = {
            "query": question,
            "namespace": namespace,
            "tiers": tiers,
            "top_per_tier": top_per_tier,
            "exclude_episodic": exclude_episodic,
            "expand_layer": expand_layer,
        }
        t0 = time.time()
        try:
            r = client.post("/api/v1/knowledge/recall", json=payload, timeout=120)
            latency_ms = (time.time() - t0) * 1000.0
            r.raise_for_status()
            body = r.json()
        except Exception as e:
            outcomes.append(
                QueryOutcome(
                    question_id=qid,
                    question_type=q.get("question_type"),
                    hit_at={k: False for k in ks},
                    latency_ms=0.0,
                    expected=expected,
                    returned=[f"ERROR: {e!r}"],
                )
            )
            continue

        merged = _merge_buckets(body.get("buckets", {}))
        refs = [it.get("source_ref") or "" for it in merged]
        hit_at = {k: bool(set(refs[:k]) & expected_set) for k in ks}

        outcomes.append(
            QueryOutcome(
                question_id=qid,
                question_type=q.get("question_type"),
                hit_at=hit_at,
                latency_ms=round(latency_ms, 1),
                expected=expected,
                returned=refs,
                reranker_used=(body.get("diagnostics") or {}).get("reranker_used"),
            )
        )

    return outcomes


def summarize(outcomes: list[QueryOutcome], ks: list[int]) -> dict[str, Any]:
    n = len(outcomes)
    if n == 0:
        return {"questions": 0}
    recall = {f"recall@{k}": round(sum(1 for o in outcomes if o.hit_at.get(k)) / n, 4) for k in ks}
    latencies = [o.latency_ms for o in outcomes if o.latency_ms > 0]
    return {
        "questions": n,
        **recall,
        "latency_ms_p50": round(_pct(latencies, 50), 1),
        "latency_ms_p95": round(_pct(latencies, 95), 1),
    }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]


def purge_namespace(client: httpx.Client, namespace: str, tag: str) -> None:
    """No-op: the API exposes no bulk delete-by-tag endpoint.

    Purge is intentionally manual — stale items only inflate the haystack, they
    don't corrupt scoring as long as source_refs are unique per run. To wipe a
    bench namespace between runs, delete rows directly, e.g.:

        docker compose exec -T db psql -U ssot -d ssot -c \\
          "DELETE FROM review_queue WHERE primary_id IN
             (SELECT id FROM knowledge_items WHERE :tag = ANY(tags));
           DELETE FROM knowledge_items WHERE :tag = ANY(tags);"

    (review_queue FKs are primary_id/secondary_id, not knowledge_item_id.)
    """
    print(
        f"[info] --purge is a no-op (no bulk delete endpoint). To clear ns={namespace} "
        f"tag={tag}, delete rows directly in the db container (see purge_namespace docstring)."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config name from config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="max questions (overrides config)")
    parser.add_argument("--skip-ingest", action="store_true", help="reuse existing bench data")
    parser.add_argument("--purge", action="store_true", help="purge bench namespace before ingest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["limit"] = args.limit

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        print(f"error: {cfg['api_key_env']} env var not set")
        return 2

    dataset_path = HERE / cfg["dataset"]
    if not dataset_path.exists():
        print(f"error: dataset not found at {dataset_path}. Run ./download.sh first.")
        return 2

    dataset = load_dataset(dataset_path, cfg.get("limit"))
    print(f"[{cfg['config_name']}] loaded {len(dataset)} questions from {dataset_path.name}")

    client = httpx.Client(
        base_url=cfg["base_url"],
        headers={"X-API-Key": api_key},
        timeout=60,
    )

    if args.purge:
        print("[*] purging bench namespace")
        purge_namespace(client, cfg["namespace"], cfg["tag"])

    ingest_result = None
    if not args.skip_ingest:
        print("[*] ingesting sessions")
        ingest_result = ingest_sessions(client, cfg, dataset)
        print(
            f"    ingested={ingest_result.ingested} failed={ingest_result.failed} "
            f"in {ingest_result.duration_s}s"
        )
        if ingest_result.errors:
            print("    sample errors:")
            for e in ingest_result.errors:
                print(f"      {e}")

    print("[*] running queries")
    outcomes = run_queries(client, cfg, dataset)
    metrics = summarize(outcomes, sorted(cfg["top_k"]))
    print(f"[*] metrics: {metrics}")

    result_blob = {
        "config": cfg["config_name"],
        "config_detail": cfg,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ingest": asdict(ingest_result) if ingest_result else None,
        "metrics": metrics,
        "outcomes": [asdict(o) for o in outcomes],
    }
    out_path = HERE / "results" / f"{cfg['config_name']}.json"
    out_path.write_text(json.dumps(result_blob, indent=2))
    print(f"[*] wrote {out_path.relative_to(HERE.parent)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
