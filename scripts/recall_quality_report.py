#!/usr/bin/env python3
"""Recall quality + latency baseline report — the reranker-toggle scoreboard.

WHY THIS EXISTS (Fable's verdict, 2026-07-04): "the toggle is the experiment."
The reranker (qwen3-reranker-4b/8b, see api/app/reranker.py) can be switched
on/off. Whether it's worth the latency it costs can only be judged by
comparing a baseline BEFORE the toggle against a baseline AFTER it, using the
same metrics. This script produces that comparable scoreboard — nothing more.

WORKFLOW:
  1. Run this script now, with the reranker in its CURRENT state, and save
     the output (e.g. `recall_quality_report.py --json > before.json`).
  2. Flip the reranker setting (see api/app/settings.py /
     reranker_candidate_multiplier / pick_reranker in api/app/reranker.py).
  3. Wait long enough for meaningful RecallEvent volume to accumulate under
     the new setting (the --days window controls how far back DB metrics
     look; --probe latency is instantaneous and can be re-run immediately
     after the toggle).
  4. Re-run this script (`--json > after.json`) and diff `before.json` vs
     `after.json`. The VERDICT line at the bottom of the text report is
     designed to be eyeballed side by side.

DATA SOURCES:
  - api.app.models.RecallEvent   (recall volume, per-agent breakdown)
  - api.app.models.ConceptFeedback (feedback rate + signal split)
  - api.app.models.AgentProfile   (lifetime per-agent totals, sanity cross-check)
  - api.app.models.KnowledgeItem  (recall_count / positive_feedback / negative_feedback
                                   as a lifetime cross-check against the windowed
                                   RecallEvent/ConceptFeedback numbers)
  - Live HTTP probe (--probe) against POST /api/v1/knowledge/recall, reading
    the `diagnostics.vec_ms` / `diagnostics.rerank_ms` / `diagnostics.reranker_used`
    fields of BucketedRecallResponse (api/app/schemas.py, api/app/routers/knowledge.py)

NOTE ON ENVIRONMENT: this script imports api.app.db lazily (inside the
functions that need a DB session) precisely so `--help` and argument parsing
work even on a host without psycopg2 installed. The DB-querying code paths
require running on a host with real DB access (e.g. a fleet host, not a
sandbox) — see api/app/db.py for connection details (SessionLocal).

USAGE:
  python scripts/recall_quality_report.py
  python scripts/recall_quality_report.py --namespace claude-shared --days 14
  python scripts/recall_quality_report.py --json
  python scripts/recall_quality_report.py --probe --api-base https://hive.example.com \\
      --api-key sk-... --probe-count 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

# Match the repo convention used by scripts/backfill_classify.py: put api/ on
# sys.path so `app.db` / `app.models` resolve, instead of `api.app.*`. Done at
# module scope (path mutation only, no DB import) so it's safe before --help.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "api"))

# Representative queries fired in --probe mode. Kept short + varied so the
# probe exercises different memory_type tiers without needing DB access.
PROBE_QUERIES = [
    "which model should I use for a quick config change",
    "how does the reranker candidate multiplier work",
    "hive loadout push context on session start",
    "postgres statement_timeout guard for the connection pool",
    "operator preference for git commit personas",
    "what is the fleet webhook orchestrator host",
    "how to run the chain orchestrator for a coding task",
    "recall event feedback rate baseline",
    "madi-guard pretooluse gate script",
    "concept confidence versus knowledge item confidence",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _pct(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(100.0 * numerator / denominator, 2)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return round(ordered[f], 1)
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return round(d0 + d1, 1)


# ---------------------------------------------------------------------------
# DB-backed sections. Imports of api.app.* happen INSIDE these functions so
# `--help` / argparse work without a DB driver installed.
# ---------------------------------------------------------------------------


def _get_session():
    """Lazily import and open a DB session. Raises on failure; callers catch."""
    from app.db import SessionLocal  # noqa: PLC0415 (intentional lazy import)

    return SessionLocal()


def recall_volume(namespace: str, since: datetime) -> dict[str, Any]:
    """Recalls/day + per-agent breakdown over the window.

    Query (ORM):
        select(RecallEvent.agent_key, func.count())
        .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)
        .group_by(RecallEvent.agent_key)

    Total and recalls/day are derived from the same rowset (sum of per-agent
    counts / window length in days).
    """
    result: dict[str, Any] = {
        "available": False,
        "total": None,
        "per_day": None,
        "per_agent": None,
        "window_days": None,
    }
    try:
        from sqlalchemy import func, select

        from app.models import RecallEvent

        session = _get_session()
    except Exception as exc:  # missing driver, missing module, etc.
        result["error"] = f"n/a (setup: {exc})"
        return result

    try:
        stmt = (
            select(RecallEvent.agent_key, func.count().label("n"))
            .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)
            .group_by(RecallEvent.agent_key)
        )
        rows = session.execute(stmt).all()
        per_agent = {r.agent_key: r.n for r in rows}
        total = sum(per_agent.values())
        window_days = max((_now_utc() - since).total_seconds() / 86400.0, 1e-9)

        result.update(
            available=True,
            total=total,
            per_day=round(total / window_days, 2),
            per_agent=per_agent,
            window_days=round(window_days, 2),
        )
    except Exception as exc:
        result["error"] = f"n/a (query failed: {exc})"
    finally:
        try:
            session.close()
        except Exception:
            pass
    return result


def feedback_rate(namespace: str, since: datetime) -> dict[str, Any]:
    """ConceptFeedback count / RecallEvent count over the window, split by signal.

    Queries (ORM):
        recall_count = select(func.count()).select_from(RecallEvent)
            .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)

        feedback_by_signal = select(ConceptFeedback.signal, func.count())
            .where(ConceptFeedback.namespace == namespace,
                   ConceptFeedback.created_at >= since)
            .group_by(ConceptFeedback.signal)

    Operator baseline (lifetime, no window): ~11.6% = 108/936.
    """
    result: dict[str, Any] = {
        "available": False,
        "recall_count": None,
        "feedback_count": None,
        "feedback_pct": None,
        "by_signal": None,
        "by_signal_pct": None,
    }
    try:
        from sqlalchemy import func, select

        from app.models import ConceptFeedback, RecallEvent

        session = _get_session()
    except Exception as exc:
        result["error"] = f"n/a (setup: {exc})"
        return result

    try:
        recall_count = session.execute(
            select(func.count())
            .select_from(RecallEvent)
            .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)
        ).scalar_one()

        signal_rows = session.execute(
            select(ConceptFeedback.signal, func.count().label("n"))
            .where(ConceptFeedback.namespace == namespace, ConceptFeedback.created_at >= since)
            .group_by(ConceptFeedback.signal)
        ).all()
        by_signal = {str(r.signal): r.n for r in signal_rows}
        feedback_count = sum(by_signal.values())
        by_signal_pct = {
            k: _pct(v, recall_count) for k, v in by_signal.items()
        }

        result.update(
            available=True,
            recall_count=recall_count,
            feedback_count=feedback_count,
            feedback_pct=_pct(feedback_count, recall_count),
            by_signal=by_signal,
            by_signal_pct=by_signal_pct,
        )
    except Exception as exc:
        result["error"] = f"n/a (query failed: {exc})"
    finally:
        try:
            session.close()
        except Exception:
            pass
    return result


def useful_rate_proxy(namespace: str, since: datetime) -> dict[str, Any]:
    """Fraction of recall events whose concept later got a 'useful' signal,
    plus the fraction of sessions marked session_completed.

    Queries (ORM):
        recalled_concept_ids = select(distinct(RecallEvent.concept_id))
            .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)

        useful_concept_ids = select(distinct(ConceptFeedback.concept_id))
            .where(ConceptFeedback.namespace == namespace,
                   ConceptFeedback.signal == 'useful',
                   ConceptFeedback.created_at >= since)

        useful_rate = |recalled ∩ useful| / |recalled|

        session_completed_rate:
            select(RecallEvent.session_id, func.bool_or(RecallEvent.session_completed))
            .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)
            .group_by(RecallEvent.session_id)
            -> fraction of distinct sessions where bool_or() is true
    """
    result: dict[str, Any] = {
        "available": False,
        "useful_rate_pct": None,
        "recalled_concepts": None,
        "useful_concepts_among_recalled": None,
        "session_completed_pct": None,
        "sessions_total": None,
        "sessions_completed": None,
    }
    try:
        from sqlalchemy import distinct, func, select

        from app.models import ConceptFeedback, FeedbackSignal, RecallEvent

        session = _get_session()
    except Exception as exc:
        result["error"] = f"n/a (setup: {exc})"
        return result

    try:
        recalled_ids = {
            row[0]
            for row in session.execute(
                select(distinct(RecallEvent.concept_id)).where(
                    RecallEvent.namespace == namespace, RecallEvent.created_at >= since
                )
            ).all()
        }
        useful_ids = {
            row[0]
            for row in session.execute(
                select(distinct(ConceptFeedback.concept_id)).where(
                    ConceptFeedback.namespace == namespace,
                    ConceptFeedback.signal == FeedbackSignal.useful,
                    ConceptFeedback.created_at >= since,
                )
            ).all()
        }
        overlap = recalled_ids & useful_ids

        session_rows = session.execute(
            select(RecallEvent.session_id, func.bool_or(RecallEvent.session_completed))
            .where(RecallEvent.namespace == namespace, RecallEvent.created_at >= since)
            .group_by(RecallEvent.session_id)
        ).all()
        sessions_total = len(session_rows)
        sessions_completed = sum(1 for _sid, completed in session_rows if completed)

        result.update(
            available=True,
            useful_rate_pct=_pct(len(overlap), len(recalled_ids)),
            recalled_concepts=len(recalled_ids),
            useful_concepts_among_recalled=len(overlap),
            session_completed_pct=_pct(sessions_completed, sessions_total),
            sessions_total=sessions_total,
            sessions_completed=sessions_completed,
        )
    except Exception as exc:
        result["error"] = f"n/a (query failed: {exc})"
    finally:
        try:
            session.close()
        except Exception:
            pass
    return result


def lifetime_knowledge_item_crosscheck(namespace: str) -> dict[str, Any]:
    """Lifetime cross-check via KnowledgeItem aggregate columns (no window —
    these are running totals, useful to sanity check the windowed numbers
    above aren't wildly off).

    Query (ORM):
        select(func.sum(recall_count), func.sum(positive_feedback),
               func.sum(negative_feedback), func.count())
        .where(KnowledgeItem.namespace == namespace)
    """
    result: dict[str, Any] = {
        "available": False,
        "total_recall_count": None,
        "total_positive_feedback": None,
        "total_negative_feedback": None,
        "item_count": None,
        "lifetime_feedback_pct": None,
    }
    try:
        from sqlalchemy import func, select

        from app.models import KnowledgeItem

        session = _get_session()
    except Exception as exc:
        result["error"] = f"n/a (setup: {exc})"
        return result

    try:
        row = session.execute(
            select(
                func.coalesce(func.sum(KnowledgeItem.recall_count), 0),
                func.coalesce(func.sum(KnowledgeItem.positive_feedback), 0),
                func.coalesce(func.sum(KnowledgeItem.negative_feedback), 0),
                func.count(),
            ).where(KnowledgeItem.namespace == namespace)
        ).one()
        total_recall_count, total_pos, total_neg, item_count = row
        lifetime_feedback = total_pos + total_neg
        result.update(
            available=True,
            total_recall_count=total_recall_count,
            total_positive_feedback=total_pos,
            total_negative_feedback=total_neg,
            item_count=item_count,
            lifetime_feedback_pct=_pct(lifetime_feedback, total_recall_count),
        )
    except Exception as exc:
        result["error"] = f"n/a (query failed: {exc})"
    finally:
        try:
            session.close()
        except Exception:
            pass
    return result


def agent_profile_crosscheck(namespace: str) -> dict[str, Any]:
    """Lifetime per-agent totals from AgentProfile, as a cross-check against
    the windowed per-agent RecallEvent breakdown.

    Query (ORM):
        select(AgentProfile.agent_key, AgentProfile.total_recalls, AgentProfile.total_feedback)
        .where(AgentProfile.namespace == namespace)
    """
    result: dict[str, Any] = {"available": False, "agents": None}
    try:
        from sqlalchemy import select

        from app.models import AgentProfile

        session = _get_session()
    except Exception as exc:
        result["error"] = f"n/a (setup: {exc})"
        return result

    try:
        rows = session.execute(
            select(AgentProfile.agent_key, AgentProfile.total_recalls, AgentProfile.total_feedback).where(
                AgentProfile.namespace == namespace
            )
        ).all()
        result.update(
            available=True,
            agents={
                r.agent_key: {"total_recalls": r.total_recalls, "total_feedback": r.total_feedback}
                for r in rows
            },
        )
    except Exception as exc:
        result["error"] = f"n/a (query failed: {exc})"
    finally:
        try:
            session.close()
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Live probe mode: fires real /api/v1/knowledge/recall calls and reads
# diagnostics.{vec_ms,rerank_ms,reranker_used} off BucketedRecallResponse.
# ---------------------------------------------------------------------------


def run_probe(api_base: str, api_key: str, namespace: str, count: int) -> dict[str, Any]:
    """Fire `count` real recall calls (cycling through PROBE_QUERIES) and
    collect diagnostics from each response.

    Endpoint: POST {api_base}/api/v1/knowledge/recall
    Headers:  X-API-Key: {api_key}
    Body (BucketedRecallRequest): {"query": ..., "namespace": ...}
    Response (BucketedRecallResponse.diagnostics): vec_ms, rerank_ms, reranker_used
    """
    url = api_base.rstrip("/") + "/api/v1/knowledge/recall"
    vec_ms_values: list[float] = []
    rerank_ms_values: list[float] = []
    total_ms_values: list[float] = []
    rerankers_used: dict[str, int] = {}
    errors: list[str] = []

    for i in range(count):
        query = PROBE_QUERIES[i % len(PROBE_QUERIES)]
        body = json.dumps({"query": query, "namespace": namespace}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            errors.append(f"query {i}: HTTP {exc.code} {exc.reason}")
            continue
        except Exception as exc:
            errors.append(f"query {i}: {exc}")
            continue
        wall_ms = (time.perf_counter() - t0) * 1000.0

        diag = payload.get("diagnostics", {})
        vec_ms = diag.get("vec_ms")
        rerank_ms = diag.get("rerank_ms")
        reranker_used = diag.get("reranker_used", "unknown")

        if vec_ms is not None:
            vec_ms_values.append(float(vec_ms))
        if rerank_ms is not None:
            rerank_ms_values.append(float(rerank_ms))
        total_ms_values.append(wall_ms)
        rerankers_used[reranker_used] = rerankers_used.get(reranker_used, 0) + 1

    return {
        "requested": count,
        "succeeded": len(total_ms_values),
        "errors": errors,
        "vec_ms": {
            "p50": _percentile(vec_ms_values, 50),
            "p95": _percentile(vec_ms_values, 95),
        },
        "rerank_ms": {
            "p50": _percentile(rerank_ms_values, 50),
            "p95": _percentile(rerank_ms_values, 95),
        },
        "total_wall_ms": {
            "p50": _percentile(total_ms_values, 50),
            "p95": _percentile(total_ms_values, 95),
        },
        "reranker_used_counts": rerankers_used,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(namespace: str, days: int, probe: dict[str, Any] | None) -> dict[str, Any]:
    since = _now_utc() - timedelta(days=days)

    volume = recall_volume(namespace, since)
    feedback = feedback_rate(namespace, since)
    useful = useful_rate_proxy(namespace, since)
    lifetime = lifetime_knowledge_item_crosscheck(namespace)
    agents = agent_profile_crosscheck(namespace)

    verdict = {
        "recalls_per_day": volume.get("per_day"),
        "feedback_pct": feedback.get("feedback_pct"),
        "useful_rate_pct": useful.get("useful_rate_pct"),
    }
    if probe is not None:
        verdict["probe_total_p50_ms"] = probe["total_wall_ms"]["p50"]
        verdict["probe_total_p95_ms"] = probe["total_wall_ms"]["p95"]
        verdict["reranker_used"] = probe["reranker_used_counts"]

    return {
        "namespace": namespace,
        "window_days": days,
        "generated_at": _now_utc().isoformat(),
        "recall_volume": volume,
        "feedback_rate": feedback,
        "useful_rate_proxy": useful,
        "lifetime_knowledge_item_crosscheck": lifetime,
        "agent_profile_crosscheck": agents,
        "probe": probe,
        "verdict": verdict,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("RECALL QUALITY REPORT")
    lines.append(f"namespace={report['namespace']}  window_days={report['window_days']}  "
                 f"generated_at={report['generated_at']}")
    lines.append("=" * 72)

    v = report["recall_volume"]
    lines.append("\n1. RECALL VOLUME")
    if v.get("available"):
        lines.append(f"   total recalls in window : {v['total']}")
        lines.append(f"   recalls/day             : {v['per_day']}")
        lines.append("   per-agent breakdown     :")
        for agent, n in sorted((v.get("per_agent") or {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"     - {agent}: {n}")
    else:
        lines.append(f"   n/a ({v.get('error', 'unavailable')})")

    f = report["feedback_rate"]
    lines.append("\n2. FEEDBACK RATE (operator lifetime baseline ~11.6% = 108/936)")
    if f.get("available"):
        lines.append(f"   recall events in window : {f['recall_count']}")
        lines.append(f"   feedback events         : {f['feedback_count']}")
        lines.append(f"   feedback rate           : {_fmt(f['feedback_pct'])}%")
        for sig, n in (f.get("by_signal") or {}).items():
            pct = (f.get("by_signal_pct") or {}).get(sig)
            lines.append(f"     - {sig}: {n} ({_fmt(pct)}%)")
    else:
        lines.append(f"   n/a ({f.get('error', 'unavailable')})")

    u = report["useful_rate_proxy"]
    lines.append("\n3. USEFUL-RATE PROXY")
    if u.get("available"):
        lines.append(
            f"   useful rate             : {_fmt(u['useful_rate_pct'])}% "
            f"({u['useful_concepts_among_recalled']}/{u['recalled_concepts']} recalled concepts "
            "later got a 'useful' signal)"
        )
        lines.append(
            f"   session_completed rate  : {_fmt(u['session_completed_pct'])}% "
            f"({u['sessions_completed']}/{u['sessions_total']} sessions)"
        )
    else:
        lines.append(f"   n/a ({u.get('error', 'unavailable')})")

    lc = report["lifetime_knowledge_item_crosscheck"]
    lines.append("\n   [cross-check] KnowledgeItem lifetime totals")
    if lc.get("available"):
        lines.append(
            f"     recall_count={lc['total_recall_count']}, "
            f"positive_feedback={lc['total_positive_feedback']}, "
            f"negative_feedback={lc['total_negative_feedback']}, "
            f"items={lc['item_count']}, "
            f"lifetime_feedback_pct={_fmt(lc['lifetime_feedback_pct'])}%"
        )
    else:
        lines.append(f"     n/a ({lc.get('error', 'unavailable')})")

    ac = report["agent_profile_crosscheck"]
    lines.append("   [cross-check] AgentProfile lifetime totals")
    if ac.get("available"):
        for agent, stats in sorted((ac.get("agents") or {}).items()):
            lines.append(f"     - {agent}: recalls={stats['total_recalls']}, feedback={stats['total_feedback']}")
    else:
        lines.append(f"     n/a ({ac.get('error', 'unavailable')})")

    lines.append("\n4. LATENCY (live probe)")
    p = report.get("probe")
    if p is None:
        lines.append("   skipped — pass --probe --api-base <url> --api-key <key> to measure "
                      "vec_ms/rerank_ms via live /api/v1/knowledge/recall calls")
    else:
        lines.append(f"   requested={p['requested']}  succeeded={p['succeeded']}  errors={len(p['errors'])}")
        lines.append(f"   vec_ms     p50={_fmt(p['vec_ms']['p50'])}  p95={_fmt(p['vec_ms']['p95'])}")
        lines.append(f"   rerank_ms  p50={_fmt(p['rerank_ms']['p50'])}  p95={_fmt(p['rerank_ms']['p95'])}")
        lines.append(f"   total_ms   p50={_fmt(p['total_wall_ms']['p50'])}  p95={_fmt(p['total_wall_ms']['p95'])}")
        lines.append(f"   reranker_used counts: {p['reranker_used_counts']}")
        if p["errors"]:
            lines.append("   errors:")
            for e in p["errors"][:10]:
                lines.append(f"     - {e}")

    verdict = report["verdict"]
    lines.append("\n" + "=" * 72)
    verdict_bits = [
        f"recalls/day={_fmt(verdict.get('recalls_per_day'))}",
        f"feedback%={_fmt(verdict.get('feedback_pct'))}",
        f"useful%={_fmt(verdict.get('useful_rate_pct'))}",
    ]
    if "probe_total_p50_ms" in verdict:
        verdict_bits.append(f"p50_total_ms={_fmt(verdict.get('probe_total_p50_ms'))}")
        verdict_bits.append(f"p95_total_ms={_fmt(verdict.get('probe_total_p95_ms'))}")
        verdict_bits.append(f"reranker_used={verdict.get('reranker_used')}")
    lines.append("VERDICT: " + "  |  ".join(verdict_bits))
    lines.append("=" * 72)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline recall quality + latency report, for diffing before/after "
            "toggling the reranker. See module docstring for the full workflow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--namespace", default="claude-shared", help="Namespace to report on (default: claude-shared)")
    parser.add_argument("--days", type=int, default=14, help="Lookback window in days (default: 14)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    parser.add_argument("--probe", action="store_true", help="Fire live recall calls to measure latency")
    parser.add_argument("--api-base", default=None, help="Base URL of the recall API (required with --probe)")
    parser.add_argument("--api-key", default=None, help="X-API-Key value (required with --probe)")
    parser.add_argument("--probe-count", type=int, default=20, help="Number of probe calls to fire (default: 20)")

    args = parser.parse_args(argv)

    probe_result = None
    if args.probe:
        if not args.api_base or not args.api_key:
            parser.error("--probe requires both --api-base and --api-key")
        probe_result = run_probe(args.api_base, args.api_key, args.namespace, args.probe_count)

    report = build_report(args.namespace, args.days, probe_result)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_text(report))

    return 0


if __name__ == "__main__":
    sys.exit(main())
