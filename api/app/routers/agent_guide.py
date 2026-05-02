"""GET /agent-guide — per-key tailored markdown runbook."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.security import require_api_key, AuthContext
from app.settings import get_settings

router = APIRouter()

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",), default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


@router.get("/agent-guide", response_class=Response)
async def agent_guide(auth: AuthContext = Depends(require_api_key)):
    settings = get_settings()
    tmpl = _env.get_template("agent_guide.md.j2")
    api_base = getattr(settings, "public_api_base", None) or "http://192.168.1.225:8088"
    version = getattr(settings, "version", None) or "dev"
    body = tmpl.render(
        key_name=auth.key_name or "(unnamed)",
        role=auth.role,
        device_id=getattr(auth, "device_id", None) or "(unknown)",
        api_base=api_base,
        version=version,
        namespaces=auth.namespaces or ["claude-shared"],
    )
    return Response(content=body, media_type="text/plain", headers={"Cache-Control": "max-age=60"})
