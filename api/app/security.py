import secrets
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_session
from .models import ApiKey, ApiRole

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ROLE_ORDER = {
    ApiRole.reader.value: 1,
    ApiRole.writer.value: 2,
    ApiRole.admin.value: 3,
}


@dataclass
class AuthContext:
    key_id: str
    key_name: str
    role: str
    namespaces: list[str]


def hash_api_key(plaintext_key: str) -> str:
    return pwd_context.hash(plaintext_key)


def verify_api_key(plaintext_key: str, key_hash: str) -> bool:
    return pwd_context.verify(plaintext_key, key_hash)


def generate_api_key() -> str:
    return f"ssot_{secrets.token_urlsafe(32)}"


def require_role(auth: AuthContext, allowed_roles: set[str]) -> None:
    if auth.role not in ROLE_ORDER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid key role")

    current = ROLE_ORDER[auth.role]
    required = min(ROLE_ORDER[role] for role in allowed_roles)
    if current < required:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")


def ensure_namespace_access(auth: AuthContext, namespace: str, allowed_roles: set[str]) -> None:
    require_role(auth, allowed_roles)
    if namespace not in auth.namespaces and "*" not in auth.namespaces:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key is not authorized for namespace '{namespace}'",
        )


def require_admin(auth: AuthContext) -> None:
    require_role(auth, {ApiRole.admin.value})


def _lookup_api_key(session: Session, plaintext_key: str) -> AuthContext | None:
    active_keys = session.scalars(select(ApiKey).where(ApiKey.is_active.is_(True))).all()
    for key in active_keys:
        if verify_api_key(plaintext_key, key.key_hash):
            return AuthContext(
                key_id=str(key.id),
                key_name=key.name,
                role=key.role.value if isinstance(key.role, ApiRole) else str(key.role),
                namespaces=list(key.namespaces or []),
            )
    return None


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    session: Session = Depends(get_session),
) -> AuthContext:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key header")

    auth = _lookup_api_key(session, x_api_key)
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return auth
