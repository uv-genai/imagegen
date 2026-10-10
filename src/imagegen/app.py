"""Composition root: build the FastAPI app and wire its collaborators.

Nothing in here implements behaviour — it constructs the pool, the blob store and
the authentication policy, and attaches them to ``app.state`` so the route
modules stay free of construction logic.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from imagegen import __version__
from imagegen.backend_pool import BackendPool
from imagegen.config import ServiceConfig
from imagegen.errors import ApiError, ServiceConfigurationError
from imagegen.image_blob_store import ImageBlobStore
from imagegen.routes_health import router as health_router
from imagegen.routes_images import router as images_router

logger = logging.getLogger("imagegen")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start supervised backends on boot and guarantee they are reaped on exit.

    The shutdown path must stop the child processes even when startup failed
    halfway, otherwise orphaned sd-server instances keep holding gigabytes of
    weights after the gateway exits.
    """
    pool: BackendPool = app.state.pool
    config: ServiceConfig = app.state.config
    if config.backends.lifecycle == "eager":
        for model_id in pool.model_ids:
            try:
                async with pool.acquire(model_id):
                    pass
            except ApiError as exc:
                logger.error("could not pre-start model %s: %s", model_id, exc.message)
    pool.start_background_tasks()
    logger.info(
        "imagegen %s ready with models: %s", __version__, ", ".join(pool.model_ids)
    )
    try:
        yield
    finally:
        await pool.shutdown()


def create_app(config: ServiceConfig, config_dir) -> FastAPI:
    """Build the ASGI application for a validated configuration.

    Args:
        config: Parsed service configuration.
        config_dir: Directory the config file was loaded from; relative weight
            paths are resolved against it.

    Raises:
        ServiceConfigurationError: referenced weights or binary are missing.
    """
    config.verify_weights(config_dir)
    pool = BackendPool(config.models, config.backends, config.service, config_dir)
    app = FastAPI(
        title="imagegen",
        version=__version__,
        summary="OpenAI-compatible image generation on stable-diffusion.cpp",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.config_dir = config_dir
    app.state.pool = pool
    app.state.blobs = ImageBlobStore(
        ttl_seconds=config.service.blob_ttl_seconds,
        capacity=config.service.blob_capacity,
    )

    app.include_router(images_router)
    app.include_router(health_router)

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        """Render every typed failure as an OpenAI-style error body."""
        logger.info("%s %s -> %s %s: %s", request.method, request.url.path, exc.status_code, exc.code, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_openai_body())

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Report body validation failures as OpenAI 400s instead of FastAPI 422s.

        OpenAI clients read ``error.message``; FastAPI's default envelope would
        hand them a ``detail`` array they cannot parse.
        """
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ())[1:]) or "body"
        message = f"invalid {location}: {first.get('msg', 'malformed request')}"
        logger.info("%s %s -> 400 %s", request.method, request.url.path, message)
        return JSONResponse(
            status_code=400,
            content={"error": {"message": message, "type": "invalid_request_error", "param": location}},
        )

    return app


def app_from_config_path(path: str) -> FastAPI:
    """Convenience loader used by the CLI and by ``uvicorn --factory``."""
    from imagegen.config import load_config

    config, config_dir = load_config(path)
    return create_app(config, config_dir)
