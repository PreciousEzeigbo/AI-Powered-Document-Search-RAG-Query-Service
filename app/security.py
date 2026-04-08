"""Authentication and session management.

Provides two independent layers:
1. **Session isolation** – every request must carry an ``X-Session-Id`` header
   (UUID4). Documents and queries are scoped to the session.
2. **API-key guard** – *optional*; when ``api_access_key`` is set in config,
   all requests must also present a valid key via ``X-API-Key`` or Bearer
   token.
"""

import uuid
from dataclasses import dataclass
from secrets import compare_digest

from fastapi import Header, HTTPException, status

from app.config import settings


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Attached to every request to enforce tenant isolation."""

    session_id: str


def require_session(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
) -> SessionContext:
    """Validate and return a ``SessionContext``.

    The session ID must be a valid UUID4 string.  The frontend is expected to
    generate one on first visit and persist it (e.g. ``localStorage``).
    """

    if not x_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing X-Session-Id header",
        )

    try:
        uuid.UUID(x_session_id, version=4)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Session-Id must be a valid UUID4",
        )

    return SessionContext(session_id=x_session_id)


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Optional API-key guard.

    When ``settings.api_access_key`` is configured, this dependency rejects
    requests that do not present a matching key.  When the key is *not*
    configured, the check is skipped (a startup warning is emitted by
    ``main.py``).
    """

    expected_key = (getattr(settings, "api_access_key", None) or "").strip()
    if not expected_key:
        return

    supplied_key = (x_api_key or "").strip()
    if not supplied_key and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            supplied_key = token.strip()

    if not supplied_key or not compare_digest(supplied_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )