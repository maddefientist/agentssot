"""Entity admin endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
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

    rows = session.execute(
        select(Entity).where(Entity.namespace == namespace).limit(limit)
    ).scalars()

    out = []
    for e in rows:
        # Count knowledge items referencing this entity
        ref_count = session.execute(
            select(func.count(KnowledgeItem.id)).where(
                KnowledgeItem.entity_refs.contains([str(e.id)])
            )
        ).scalar_one()

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
