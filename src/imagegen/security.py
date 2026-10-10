"""Bearer/`x-api-key` authentication policy for the gateway.

The backend servers listen on loopback only and have no credentials of their own,
so the gateway is the single authentication boundary in front of them.
"""

from __future__ import annotations

import secrets

from fastapi import Request

from imagegen.errors import AuthenticationError

BEARER_PREFIX = "bearer "


def _presented_key(request: Request) -> str | None:
    """Extract the client key from ``Authorization: Bearer`` or ``x-api-key``.

    Both spellings are accepted because OpenAI SDK clients send the bearer header
    while many proxies and CLI tools send ``x-api-key``.
    """
    authorization = request.headers.get("authorization")
    if authorization and authorization.lower().startswith(BEARER_PREFIX):
        token = authorization[len(BEARER_PREFIX) :].strip()
        if token:
            return token
    header_key = request.headers.get("x-api-key")
    if header_key and header_key.strip():
        return header_key.strip()
    return None


async def require_api_key(request: Request) -> None:
    """FastAPI dependency rejecting requests whose key is not configured.

    When the service lists no keys it is deliberately open — that is the documented
    way to run behind a separate authenticating proxy or on a trusted loopback
    address — and a warning is logged once at startup rather than on every call.
    """
    allowed: list[str] = request.app.state.config.service.api_keys
    if not allowed:
        return
    presented = _presented_key(request)
    if presented is None:
        raise AuthenticationError("missing API key; send 'Authorization: Bearer <key>' or 'x-api-key: <key>'")
    # Constant-time comparison against every configured key so that a matching key
    # later in the list does not take measurably longer than one earlier on.
    if not any(secrets.compare_digest(presented, candidate) for candidate in allowed):
        raise AuthenticationError("invalid API key")
