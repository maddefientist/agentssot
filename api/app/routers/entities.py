"""Entity admin endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, cast
from sqlalchemy.dialects.postgresql import ARRAY, TEXT
from sqlalchemy.orm import Session

from app.db import get_session
from app.models import Entity, KnowledgeItem
from app.security import require_api_key, AuthContext, ApiRole

router = APIRouter(prefix="/api/v1/entities", tags=["entities"])


@router.get("/")
async def list_entities(
    namespace: str = Query(default="claude-shared"),
    limit: int = Query(default=200, ge=1, le=1000),
    session: Session = Depends(get_session),
    auth: AuthContext = Depends(require_api_key),
):
    """List entities in the namespace with reference counts. Writer or admin required."""
    if auth.role not in (ApiRole.writer.value, ApiRole.admin.value):
        raise HTTPException(status_code=403, detail="writer or admin role required")

    entities = list(
        session.execute(
            select(Entity).where(Entity.namespace == namespace).limit(limit)
        ).scalars()
    )

    # Compute all ref-counts in ONE query: unnest each knowledge item's
    # entity_refs within the same namespace, group by referenced entity id, then
    # map counts back. The namespace filter scopes counts to this namespace only.
    counts: dict[str, int] = {}
    if entities:
        entity_id_strs = [str(e.id) for e in entities]
        ref_subq = (
            select(
                func.jsonb_array_elements_text(KnowledgeItem.entity_refs).label("eid")
            )
            .select_from(KnowledgeItem)
            .where(KnowledgeItem.namespace == namespace)
            .where(
                func.jsonb_exists_any(
                    KnowledgeItem.entity_refs, cast(entity_id_strs, ARRAY(TEXT))
                )
            )
        )
        count_rows = session.execute(
            select(ref_subq.c.eid, func.count().label("cnt"))
            .select_from(ref_subq)
            .group_by(ref_subq.c.eid)
        ).all()
        counts = {r.eid: int(r.cnt) for r in count_rows}

    out = []
    for e in entities:
        ref_count = counts.get(str(e.id), 0)

        out.append({
            "id": str(e.id),
            "slug": e.slug,
            "entity_type": e.type.value,
            "name": e.name,
            "ips": (e.meta or {}).get("ips", []) or [],
            "cwd_hints": (e.meta or {}).get("cwd_hints", []) or [],
            "device_hints": (e.meta or {}).get("device_hints", []) or [],
            "ref_count": ref_count,
        })
    return out
