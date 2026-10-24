"""OpenAI-compatible image endpoints: generations, edits, model listing, blobs."""

from __future__ import annotations

import base64
import time
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from imagegen.backend_client import BackendImages
from imagegen.errors import InvalidRequestError
from imagegen.model_resolution import check_image_count, resolve_profile, resolve_size
from imagegen.openai_payload import (
    EDIT_STRENGTH_DEFAULT,
    build_generation_payload,
    build_native_args,
    compose_prompt,
    reject_extension_markers,
)
from imagegen.schemas import GenerationRequest, parse_sd_cpp_form_field
from imagegen.security import require_api_key

router = APIRouter()

#: Maximum accepted upload per reference image, guarding the process memory of a
#: service that is otherwise unbounded in what a client can POST at it.
MAX_UPLOAD_BYTES = 32 * 1024 * 1024


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Report every registered model in OpenAI's model-list shape.

    A profile that declares a structured-layout dialect gains a ``layout`` key.
    It is omitted entirely when unset rather than sent as ``null``: clients tell
    "this model does not do layout" apart from "this engine lacks the
    extension", and only the absent key means the latter.
    """
    config = request.app.state.config
    data = []
    for profile in config.models:
        entry = {
            "id": profile.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "imagegen",
            "description": profile.description or profile.id,
        }
        if profile.layout is not None:
            entry["layout"] = profile.layout
        data.append(entry)
    return JSONResponse({"object": "list", "data": data})


@router.post("/v1/images/generations", dependencies=[Depends(require_api_key)])
async def generate_images(payload: GenerationRequest, request: Request) -> JSONResponse:
    """Text-to-image, in the OpenAI request and response shape."""
    payload._reject_unknown_fields()
    config = request.app.state.config
    profile = resolve_profile(config, payload.model)
    size = resolve_size(profile, payload.size)
    check_image_count(profile, payload.n, size)
    _require_public_base_url_for(payload.response_format, config.service.public_base_url)

    body = build_generation_payload(
        profile,
        prompt=payload.prompt,
        size=size,
        count=payload.n,
        output_format=payload.output_format,
        output_compression=payload.output_compression,
        client_native_overrides=payload.native_overrides(),
    )
    async with request.app.state.pool.acquire(profile.id) as client:
        result = await client.generate(body)
    return _image_response(result, request, payload.response_format)


@router.post("/v1/images/edits", dependencies=[Depends(require_api_key)])
async def edit_images(request: Request) -> JSONResponse:
    """Image editing / composition, mirroring OpenAI's multipart edits endpoint."""
    form = await request.form()
    prompt = _form_text(form, "prompt")
    if not prompt:
        raise InvalidRequestError("the 'prompt' form field is required", param="prompt")
    try:
        reject_extension_markers(prompt)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="prompt") from exc

    config = request.app.state.config
    model = _form_text(form, "model") or None
    profile = resolve_profile(config, model)
    if not profile.supports_editing:
        raise InvalidRequestError(
            f"model {profile.id!r} cannot edit images; it is configured with supports_editing=false, "
            "so reference images would be ignored",
            param="model",
        )
    response_format = _form_text(form, "response_format") or "b64_json"
    if response_format not in ("url", "b64_json"):
        raise InvalidRequestError("response_format must be 'url' or 'b64_json'", param="response_format")
    _require_public_base_url_for(response_format, config.service.public_base_url)

    size = resolve_size(profile, _form_text(form, "size") or None)
    count = _form_int(form, "n", default=1)
    if not 1 <= count <= 8:
        raise InvalidRequestError(f"n must be between 1 and 8, got {count}", param="n")
    check_image_count(profile, count, size)
    output_format = _form_text(form, "output_format") or "png"
    if output_format not in ("png", "jpeg"):
        raise InvalidRequestError("output_format must be 'png' or 'jpeg' for edits", param="output_format")
    output_compression = _form_int(form, "output_compression", default=100)
    if not 0 <= output_compression <= 100:
        raise InvalidRequestError(
            f"output_compression must be between 0 and 100, got {output_compression}", param="output_compression"
        )

    uploads = await _collect_uploads(form)
    if not uploads:
        raise InvalidRequestError(
            "no image uploaded; send one or more files in the 'image[]' (or legacy 'image') field",
            param="image[]",
        )

    # Merge order: engine default first, then the sd_cpp escape hatch, then the
    # dedicated form field, so the most specific spelling wins.
    native_overrides: dict[str, Any] = {"strength": EDIT_STRENGTH_DEFAULT}
    native_overrides.update(parse_sd_cpp_form_field(_form_text(form, "sd_cpp") or None) or {})
    if _form_text(form, "strength"):
        native_overrides["strength"] = _form_float(form, "strength", default=EDIT_STRENGTH_DEFAULT)
    strength = native_overrides["strength"]
    if not isinstance(strength, (int, float)) or not 0.0 < float(strength) <= 1.0:
        raise InvalidRequestError(f"strength must be in (0, 1], got {strength!r}", param="strength")
    negative_prompt = _form_text(form, "negative_prompt")
    if negative_prompt:
        native_overrides.setdefault("negative_prompt", negative_prompt)

    fields: dict[str, str] = {
        "prompt": compose_prompt(prompt, build_native_args(profile, native_overrides)),
        "n": str(count),
        "size": size,
        "output_format": output_format,
        "output_compression": str(output_compression),
    }
    async with request.app.state.pool.acquire(profile.id) as client:
        result = await client.edit(fields=fields, files=uploads)
    return _image_response(result, request, response_format)


@router.get("/v1/images/content/{blob_id}")
async def fetch_image_blob(blob_id: str, request: Request) -> Response:
    """Serve a stored image for ``response_format=url``.

    Unauthenticated by design: the opaque, single-purpose token in the URL is the
    capability, exactly like OpenAI's pre-signed object URLs.
    """
    entry = request.app.state.blobs.get(blob_id)
    if entry is None:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "image not found or expired", "type": "invalid_request_error"}},
        )
    return Response(content=entry.content, media_type=entry.media_type, headers={"Cache-Control": "private, no-store"})


def _require_public_base_url_for(response_format: str, public_base_url: str | None) -> None:
    """Fail loudly when a client asks for URLs but the service cannot advertise them."""
    if response_format == "url" and not public_base_url:
        raise InvalidRequestError(
            "response_format='url' requires service.public_base_url to be configured",
            param="response_format",
        )


def _image_response(result: BackendImages, request: Request, response_format: str) -> JSONResponse:
    """Turn decoded backend bytes into an OpenAI image response body."""
    public_base_url = request.app.state.config.service.public_base_url
    data: list[dict[str, str]] = []
    for content in result.images:
        if response_format == "url":
            entry = request.app.state.blobs.store(content, result.output_format)
            data.append({"url": f"{public_base_url.rstrip('/')}/v1/images/content/{entry.blob_id}"})
        else:
            data.append({"b64_json": base64.b64encode(content).decode("ascii")})
    return JSONResponse({"created": int(time.time()), "output_format": result.output_format, "data": data})


def _form_text(form, key: str) -> str:  # noqa: ANN001 - starlette FormData has no exported type alias
    """Read a text form field, ignoring file parts with the same name."""
    value = form.get(key)
    if value is None or hasattr(value, "read"):
        return ""
    return str(value)


def _form_int(form, key: str, default: int) -> int:  # noqa: ANN001
    """Read an integer form field, rejecting non-numeric values."""
    raw = _form_text(form, key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise InvalidRequestError(f"{key} must be an integer, got {raw!r}", param=key) from exc


def _form_float(form, key: str, default: float) -> float:  # noqa: ANN001
    """Read a float form field, rejecting non-numeric values."""
    raw = _form_text(form, key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise InvalidRequestError(f"{key} must be a number, got {raw!r}", param=key) from exc


async def _collect_uploads(form) -> list[tuple[str, tuple[str, bytes, str]]]:  # noqa: ANN001
    """Gather reference images and the optional mask as httpx multipart tuples."""
    uploads: list[tuple[str, tuple[str, bytes, str]]] = []
    for key in ("image[]", "image", "mask"):
        for item in form.getlist(key):
            if not hasattr(item, "read"):
                continue
            content = await item.read()
            if not content:
                raise InvalidRequestError(f"uploaded file in {key!r} is empty", param=key)
            if len(content) > MAX_UPLOAD_BYTES:
                raise InvalidRequestError(
                    f"uploaded file in {key!r} is {len(content)} bytes, above the {MAX_UPLOAD_BYTES} byte limit",
                    param=key,
                )
            media_type = getattr(item, "content_type", None) or "application/octet-stream"
            name = getattr(item, "filename", None) or key
            uploads.append((key, (name, content, media_type)))
    return uploads
