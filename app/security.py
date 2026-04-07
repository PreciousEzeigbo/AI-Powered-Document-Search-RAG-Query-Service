from secrets import compare_digest

from fastapi import Header, HTTPException, status

from app.config import settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
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