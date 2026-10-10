"""Operational endpoints: liveness and backend state."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from imagegen import __version__
from imagegen.security import require_api_key

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    """Liveness probe; deliberately unauthenticated so orchestrators can poll it."""
    return JSONResponse({"status": "ok", "service": "imagegen", "version": __version__})


@router.get("/health/backends", dependencies=[Depends(require_api_key)])
async def backend_status(request: Request) -> JSONResponse:
    """Per-model backend state: loaded, starting, stopped, queued, busy.

    Weight loading takes tens of seconds per model, so callers need a way to see
    whether a request will run immediately or first pay for a model swap.
    """
    pool = request.app.state.pool
    return JSONResponse(
        {
            "time": int(time.time()),
            "lifecycle": request.app.state.config.backends.lifecycle,
            "max_resident_backends": request.app.state.config.backends.max_resident_backends,
            "resident_backends": sum(1 for entry in pool.status() if entry["state"] == "ready"),
            "stored_images": len(request.app.state.blobs),
            "backends": pool.status(),
        }
    )
