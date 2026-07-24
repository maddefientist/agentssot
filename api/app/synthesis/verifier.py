"""Evidence-based verification of infrastructure claims in active memories.

V1 verifies only Ollama model liveness.  Probe results are deliberately kept
out-of-band from recall: contradictions become alerts/review entries by default,
and supersession is an explicit opt-in mode.
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Callable, Iterable
from uuid import UUID

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..alerting import post_alert
from ..db import SessionLocal
from ..models import (
    KnowledgeItem,
    ReviewQueueItem,
    ReviewQueueKind,
    ReviewQueueStatus,
)
from ..services.lifecycle import apply_supersession
from ..settings import get_settings

logger = logging.getLogger("agentssot.synthesis.verifier")

DEFAULT_OLLAMA_URL = "http://localhost:11434"
TAGS_TIMEOUT_SECONDS = 10.0
GENERATE_TIMEOUT_SECONDS = 60.0
CANDIDATE_LIMIT = 500

# The optional provider prefix is discarded before probing. The tag is narrow on
# purpose: v1 accepts only parameter-count, cloud, and latest Ollama tags.
_MODEL_RE = re.compile(
    r"(?<![a-z0-9._/-])(?:ollama/)?"
    r"(?P<model>[a-z0-9][a-z0-9._-]*:(?P<tag>[0-9]+b(?:-cloud)?|cloud|latest))"
    r"(?![a-z0-9._-])",
    re.IGNORECASE,
)
_LIVE_RE = re.compile(
    r"\b(?:prefer(?:red)?|routes?\s+to|routing\s+to|current|canonical|"
    r"live|active|available|primary|fallback)\b|(?<!not )(?<!n't )\buse\b",
    re.IGNORECASE,
)
_RETIRED_RE = re.compile(
    r"\b(?:retired|gone|superseded|avoid(?:ed)?|deprecated|removed|dead|"
    r"unavailable)\b|\b(?:do\s+not|don['’]t|must\s+not|should\s+not)\s+use\b|"
    r"\bnot\s+available\b",
    re.IGNORECASE,
)
_NOT_FOUND_RE = re.compile(
    r"model[^\n]{0,80}(?:not\s+found|does\s+not\s+exist|unknown)|"
    r"(?:not\s+found|unknown)[^\n]{0,80}model|pull\s+model",
    re.IGNORECASE,
)
# Phrases that immediately precede a model id and mark it as the SUCCESSOR of
# a retired/deprecated predecessor. The retirement keyword belongs to the
# predecessor, not to the id that follows it, so a successor occurrence must be
# classified LIVE without scanning its live/retired keyword window.
_SUCCESSOR_PREFIX_RE = re.compile(
    r"(?:superse[dc]ed|replaced|supplanted|deprecated|retired|migrated|moved|upgraded|switched)"
    r"\s+(?:by|to|with|in\s+favou?r\s+of)\s+$"
    r"|(?:use|prefer|now\s+use)\s+$"
    r"|(?:->|→)\s*$",
    re.IGNORECASE,
)
# A complete successor-introduction phrase: the introducer keyword(s) from
# _SUCCESSOR_PREFIX_RE immediately followed by the successor model id. Stripped
# from a predecessor occurrence's keyword window so the introducer word ("use",
# "superseded") does not leak as a live/retired keyword for the predecessor.
# The target's own "use <target>" claim is preserved (sid == model_id).
_SUCCESSOR_PHRASE_RE = re.compile(
    r"(?:"
    r"(?:superse[dc]ed|replaced|supplanted|deprecated|retired|migrated|moved|upgraded|switched)"
    r"\s+(?:by|to|with|in\s+favou?r\s+of)"
    r"|(?:use|prefer|now\s+use)"
    r"|(?:->|→)"
    r")\s*(?P<sid>[A-Za-z0-9][A-Za-z0-9._-]*:[A-Za-z0-9][A-Za-z0-9._-]*)",
    re.IGNORECASE,
)


class ProbeState(StrEnum):
    ALIVE = "ALIVE"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


class AssertionDirection(StrEnum):
    LIVE = "asserts-live"
    RETIRED = "asserts-retired"


@dataclass(frozen=True)
class ProbeResult:
    model_id: str
    state: ProbeState
    evidence: str


@dataclass(frozen=True)
class Finding:
    knowledge_id: str
    namespace: str
    model_id: str
    assertion: AssertionDirection
    observed: ProbeState
    evidence: str


@dataclass
class VerifierReport:
    enabled: bool
    dry_run: bool
    candidates_scanned: int = 0
    model_ids: list[str] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    queued: int = 0
    superseded: int = 0
    alert_attempted: bool = False


def extract_model_ids(content: str) -> list[str]:
    """Extract and normalize conservative Ollama model ids from text."""
    found: set[str] = set()
    for match in _MODEL_RE.finditer(content or ""):
        tag = match.group("tag")
        # This independently protects against port/time-like numeric suffixes if
        # the accepted tag alternatives are broadened later.
        if not any(char.isalpha() for char in tag):
            continue
        found.add(match.group("model").lower())
    return sorted(found)


def classify_assertion(content: str, model_id: str, window: int = 100) -> AssertionDirection | None:
    """Classify a claim using keyword windows around each model occurrence."""
    occurrence = re.compile(
        rf"(?:ollama/)?{re.escape(model_id)}(?![a-z0-9._-])",
        re.IGNORECASE,
    )
    directions: set[AssertionDirection] = set()
    for match in occurrence.finditer(content or ""):
        # If this occurrence is introduced as the SUCCESSOR of a retired
        # predecessor ("superseded by <id>", "use <id> instead", "-> <id>"),
        # the retirement keyword in the prefix belongs to the predecessor, not
        # to this id. Classify it LIVE and skip the live/retired window entirely.
        prefix = content[max(0, match.start() - 40):match.start()]
        if _SUCCESSOR_PREFIX_RE.search(prefix):
            directions.add(AssertionDirection.LIVE)
            continue
        nearby = content[max(0, match.start() - window):match.end() + window]
        # Strip successor-introduction phrases that refer to OTHER model ids
        # ("use <other>", "superseded by <other>", "-> <other>") so the
        # introducer keyword does not leak into this occurrence's live/retired
        # window. The target's own "use <target>" claim is preserved.
        nearby = _SUCCESSOR_PHRASE_RE.sub(
            lambda m: "" if m.group("sid").lower() != model_id.lower() else m.group(0),
            nearby,
        )
        live = bool(_LIVE_RE.search(nearby))
        retired = bool(_RETIRED_RE.search(nearby))
        if retired and live:
            # Negative phrases such as "not available" contain an otherwise
            # live keyword. Remove complete retirement phrases before deciding
            # whether the window genuinely makes both assertions.
            live = bool(_LIVE_RE.search(_RETIRED_RE.sub("", nearby)))
        if live and not retired:
            directions.add(AssertionDirection.LIVE)
        elif retired and not live:
            directions.add(AssertionDirection.RETIRED)
    if len(directions) == 1:
        return next(iter(directions))
    return None


def _response_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])[:300]
    except (ValueError, TypeError):
        pass
    return response.text[:300]


def probe_model(model_id: str, base_url: str = DEFAULT_OLLAMA_URL) -> ProbeResult:
    """Probe one normalized Ollama model id without raising transport errors."""
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(tags_url, timeout=TAGS_TIMEOUT_SECONDS)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"GET /api/tags failed: {type(exc).__name__}")
    except Exception as exc:  # defensive: a verifier must not break synthesis
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"GET /api/tags failed: {type(exc).__name__}")

    if response.status_code >= 500:
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"GET /api/tags HTTP {response.status_code}")
    if response.status_code != 200:
        return ProbeResult(
            model_id,
            ProbeState.UNKNOWN,
            f"GET /api/tags HTTP {response.status_code}: {_response_text(response)}",
        )
    try:
        payload = response.json()
        models = {
            str(value).lower()
            for item in payload.get("models", [])
            if isinstance(item, dict)
            for value in (item.get("name"), item.get("model"))
            if value
        }
    except (ValueError, TypeError, AttributeError) as exc:
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"GET /api/tags invalid response: {type(exc).__name__}")

    if model_id in models:
        return ProbeResult(model_id, ProbeState.ALIVE, "GET /api/tags membership: present")

    is_cloud = model_id.endswith(":cloud") or model_id.endswith("-cloud")
    if not is_cloud:
        return ProbeResult(model_id, ProbeState.DEAD, "GET /api/tags membership: absent (local model)")

    # CRITICAL: a tags miss for a cloud model (:cloud or parameterized -cloud)
    # is not evidence of death. Cloud models can be callable without appearing
    # in /api/tags; only generate may establish DEAD. UNKNOWN must never flow
    # into contradiction handling.
    generate_url = f"{base_url.rstrip('/')}/api/generate"
    try:
        response = httpx.post(
            generate_url,
            json={
                "model": model_id,
                "prompt": "PONG",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=GENERATE_TIMEOUT_SECONDS,
        )
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"POST /api/generate failed: {type(exc).__name__}")
    except Exception as exc:  # defensive: a verifier must not break synthesis
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"POST /api/generate failed: {type(exc).__name__}")

    if response.status_code == 200:
        return ProbeResult(model_id, ProbeState.ALIVE, "POST /api/generate HTTP 200 after tags miss")
    if response.status_code >= 500:
        return ProbeResult(model_id, ProbeState.UNKNOWN, f"POST /api/generate HTTP {response.status_code}")

    error = _response_text(response)
    if response.status_code == 404 or _NOT_FOUND_RE.search(error):
        return ProbeResult(
            model_id,
            ProbeState.DEAD,
            f"POST /api/generate HTTP {response.status_code}: {error}",
        )
    return ProbeResult(
        model_id,
        ProbeState.UNKNOWN,
        f"POST /api/generate HTTP {response.status_code}: {error}",
    )


def verify_items(
    items: Iterable[KnowledgeItem],
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    probe: Callable[[str, str], ProbeResult] | None = None,
    dry_run: bool = False,
) -> VerifierReport:
    """Probe distinct ids and return contradictions; performs no writes."""
    probe = probe or probe_model
    rows = list(items)[:CANDIDATE_LIMIT]
    item_models = [(item, extract_model_ids(item.content)) for item in rows]
    model_ids = sorted({model for _item, models in item_models for model in models})
    probes = {model: probe(model, base_url) for model in model_ids}
    findings: list[Finding] = []

    for item, models in item_models:
        for model_id in models:
            result = probes[model_id]
            if result.state == ProbeState.UNKNOWN:
                continue
            assertion = classify_assertion(item.content, model_id)
            if assertion is None:
                continue
            contradicted = (
                assertion == AssertionDirection.LIVE and result.state == ProbeState.DEAD
            ) or (
                assertion == AssertionDirection.RETIRED and result.state == ProbeState.ALIVE
            )
            if contradicted:
                findings.append(
                    Finding(
                        knowledge_id=str(item.id),
                        namespace=item.namespace,
                        model_id=model_id,
                        assertion=assertion,
                        observed=result.state,
                        evidence=result.evidence,
                    )
                )

    return VerifierReport(
        enabled=True,
        dry_run=dry_run,
        candidates_scanned=len(rows),
        model_ids=model_ids,
        probes=[probes[model] for model in model_ids],
        findings=findings,
    )


def _load_candidates(session: Session) -> list[KnowledgeItem]:
    # KnowledgeItem currently has no updated_at column, so created_at is the
    # available recency boundary for this bounded v1 scan.
    stmt = (
        select(KnowledgeItem)
        .where(KnowledgeItem.memory_type.in_(("doctrine", "rule")))
        .where(KnowledgeItem.status == "active")
        .where(KnowledgeItem.superseded_by.is_(None))
        .order_by(KnowledgeItem.created_at.desc())
        .limit(CANDIDATE_LIMIT)
    )
    return list(session.scalars(stmt).all())


def _finding_reason(finding: Finding) -> str:
    return (
        f"memory verifier: {finding.assertion.value} for {finding.model_id} "
        f"contradicted by {finding.observed.value}; {finding.evidence}"
    )


def _enqueue_findings(session: Session, findings: list[Finding]) -> int:
    queued = 0
    for finding in findings:
        reason = _finding_reason(finding)
        existing = session.scalar(
            select(ReviewQueueItem.id).where(
                ReviewQueueItem.kind == ReviewQueueKind.contradiction,
                ReviewQueueItem.status == ReviewQueueStatus.pending,
                ReviewQueueItem.primary_id == UUID(finding.knowledge_id),
                ReviewQueueItem.secondary_id.is_(None),
                ReviewQueueItem.reason == reason,
            )
        )
        if existing is not None:
            continue
        session.add(
            ReviewQueueItem(
                namespace=finding.namespace,
                kind=ReviewQueueKind.contradiction,
                priority=20,
                primary_id=UUID(finding.knowledge_id),
                reason=reason,
                status=ReviewQueueStatus.pending,
            )
        )
        queued += 1
    return queued


def _supersede_findings(
    session: Session,
    items: list[KnowledgeItem],
    findings: list[Finding],
    verified_at: datetime,
) -> int:
    by_id = {str(item.id): item for item in items}
    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.knowledge_id, []).append(finding)

    superseded = 0
    for knowledge_id, item_findings in grouped.items():
        old = by_id[knowledge_id]
        facts = [
            f"Ollama model {finding.model_id} is {finding.observed.value}. "
            f"Probe evidence: {finding.evidence}"
            for finding in item_findings
        ]
        correction = KnowledgeItem(
            namespace=old.namespace,
            content=f"Verified {verified_at.date().isoformat()}: " + "\n".join(facts),
            memory_type="correction",
            status="active",
            tags=["memory-verifier", "ollama", "verified"],
            source="evidence-memory-verifier",
            source_ref=str(old.id),
            project_id=old.project_id,
            entity_id=old.entity_id,
            entity_refs=list(old.entity_refs or []),
            embedding=list(old.embedding) if old.embedding is not None else None,
            verbatim=True,
        )
        session.add(correction)
        session.flush()
        apply_supersession(old, correction)
        superseded += 1
    return superseded


def _post_summary_alert(settings, report: VerifierReport) -> bool:
    detail = {
        "finding_count": len(report.findings),
        "findings": [asdict(finding) for finding in report.findings],
    }
    return post_alert(
        getattr(settings, "alert_webhook_url", ""),
        "memory_verifier.contradiction",
        "warning",
        f"Memory verifier found {len(report.findings)} Ollama model claim contradiction(s)",
        detail,
        host_label=getattr(settings, "alert_host_label", "hive"),
        enabled=getattr(settings, "alert_enabled", True),
    )


def _run_verifier_inner(
    settings,
    active_session: Session,
    *,
    dry_run: bool,
    base_url: str | None,
) -> VerifierReport:
    """Run one verifier pass against an open session; performs writes per mode."""
    try:
        items = _load_candidates(active_session)
        report = verify_items(
            items,
            base_url=base_url or getattr(settings, "ollama_base_url", DEFAULT_OLLAMA_URL),
            dry_run=dry_run,
        )
        if not report.findings or dry_run:
            return report

        mode = getattr(settings, "verifier_mode", "alert")
        if mode == "alert":
            report.queued = _enqueue_findings(active_session, report.findings)
            if report.queued:
                active_session.commit()
            report.alert_attempted = True
            _post_summary_alert(settings, report)
        elif mode == "supersede":
            report.superseded = _supersede_findings(
                active_session,
                items,
                report.findings,
                datetime.now(UTC),
            )
            active_session.commit()
        else:
            raise ValueError(f"unsupported verifier mode: {mode}")

        logger.warning(
            "memory verifier contradictions",
            extra={
                "mode": mode,
                "finding_count": len(report.findings),
                "queued": report.queued,
                "superseded": report.superseded,
                "findings": [asdict(finding) for finding in report.findings],
            },
        )
        return report
    except Exception:
        if not dry_run:
            active_session.rollback()
        raise


def run_verifier(
    settings=None,
    *,
    session: Session | None = None,
    dry_run: bool = False,
    base_url: str | None = None,
) -> VerifierReport:
    """Run one global verifier pass and apply the configured action mode.

    A caller-supplied session is left open (caller owns it and its lifecycle).
    When run_verifier creates its own session it is closed on every exit path,
    including exceptions, via the repo's ``with SessionLocal()`` idiom.
    """
    settings = settings or get_settings()
    if not getattr(settings, "verifier_enabled", True):
        return VerifierReport(enabled=False, dry_run=dry_run)

    if session is not None:
        return _run_verifier_inner(settings, session, dry_run=dry_run, base_url=base_url)
    with SessionLocal() as owned:
        return _run_verifier_inner(settings, owned, dry_run=dry_run, base_url=base_url)


def format_report(report: VerifierReport) -> str:
    lines = [
        "AgentSSOT evidence-based memory verifier (dry-run)",
        f"enabled: {report.enabled}",
        f"candidates scanned: {report.candidates_scanned}",
        f"distinct model ids: {len(report.model_ids)}",
    ]
    if not report.probes:
        lines.append("probes: none")
    else:
        lines.append("probes:")
        for probe in report.probes:
            lines.append(f"  - {probe.model_id}: {probe.state.value} — {probe.evidence}")
    lines.append(f"contradictions: {len(report.findings)}")
    for finding in report.findings:
        lines.append(
            f"  - KI {finding.knowledge_id} [{finding.namespace}] {finding.model_id}: "
            f"{finding.assertion.value} vs {finding.observed.value} — {finding.evidence}"
        )
    lines.append("writes performed: 0")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify memory claims against live Ollama state")
    parser.add_argument("--dry-run", action="store_true", help="probe and report without writes")
    parser.add_argument("--base-url", default=None, help="override the configured Ollama base URL")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("standalone verifier requires --dry-run")
    report = run_verifier(dry_run=True, base_url=args.base_url)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
