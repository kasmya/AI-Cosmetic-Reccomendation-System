"""Access guard for operational endpoints."""
import hmac
import os

from fastapi import HTTPException, Request


def require_ops_token(request: Request) -> None:
    """If OPS_TOKEN is set, /metrics and /api/v1/ops/* need it (Authorization: Bearer <token> or X-Ops-Token).
    Unset means open, which is fine for local use and a portfolio demo but not for a public deployment."""
    token = os.environ.get("OPS_TOKEN")
    if not token:
        return
    given = request.headers.get("x-ops-token") or request.headers.get("authorization", "").removeprefix("Bearer ")
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(401, "Missing or invalid ops token")
