"""
GitHub webhook receiver.

Subscribes to `push` events on watched repos. On push to the default
branch, marks the repo for re-embedding by clearing its `embedding`
column — the next `embed` cron tick picks it up.

This is the push-based counterpart to the polling seed/embed cron. It
avoids waiting up to 3h for a re-embed of a repo whose README just
changed.

Endpoint:
  POST /webhooks/github
  Headers:
    X-Hub-Signature-256: sha256=<hmac>
    X-GitHub-Event: push
  Body: standard GitHub webhook payload

Setup:
  - Set GITHUB_WEBHOOK_SECRET env var (any random string).
  - Use `register-webhooks` CLI to subscribe to high-value repos.
"""

from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import APIRouter, Header, HTTPException, Request, status

from reporelay_mvp import data

logger = logging.getLogger(__name__)

router = APIRouter()


def _verify_signature(secret: str, body: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


@router.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> dict[str, str]:
    body = await request.body()
    secret = request.app.state.github_webhook_secret
    if not secret:
        logger.error("GITHUB_WEBHOOK_SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="webhook secret not configured",
        )

    if not _verify_signature(secret, body, x_hub_signature_256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid signature"
        )

    if x_github_event != "push":
        return {"status": "ignored", "event": x_github_event or "unknown"}

    payload = await request.json()
    repo = payload.get("repository") or {}
    full_name = repo.get("full_name")
    default_branch = repo.get("default_branch", "main")
    ref = payload.get("ref", "")

    if not full_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="missing repository.full_name"
        )

    if ref and ref != f"refs/heads/{default_branch}":
        return {"status": "ignored", "reason": f"ref {ref} is not default branch"}

    session = await data.get_session()
    try:
        cleared = await data.clear_embedding_for_reembed(
            session, full_name=full_name
        )
    finally:
        await session.close()

    logger.info(
        "webhook push: %s (cleared_embedding=%s)", full_name, cleared
    )
    return {"status": "ok", "repo": full_name, "cleared_embedding": str(bool(cleared))}
