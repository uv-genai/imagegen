"""Request-time resolution of the target model and generation geometry.

Kept separate from the routes so the same rules apply to generations and edits,
and so the error messages stay identical between the two endpoints.
"""

from __future__ import annotations

from imagegen.config import ModelProfile, ServiceConfig
from imagegen.errors import InvalidRequestError, ModelNotFoundError
from imagegen.openai_payload import format_size, parse_size


def resolve_profile(config: ServiceConfig, model: str | None) -> ModelProfile:
    """Pick the model profile for a request.

    A request may omit ``model`` only when the service exposes exactly one model,
    which keeps single-model deployments simple without making multi-model
    deployments guess.
    """
    if model is None or not model.strip():
        if len(config.models) == 1:
            return config.models[0]
        available = ", ".join(profile.id for profile in config.models)
        raise InvalidRequestError(
            f"the 'model' parameter is required; available models: {available}", param="model"
        )
    profile = config.profile(model)
    if profile is None:
        available = ", ".join(candidate.id for candidate in config.models)
        raise ModelNotFoundError(f"unknown model {model!r}; available models: {available}", param="model")
    return profile


def resolve_size(profile: ModelProfile, size: str | None) -> str:
    """Normalise and validate the requested size against the model's policy.

    Returns the canonical ``WIDTHxHEIGHT`` string to forward upstream.
    """
    raw = size if size else profile.default_size
    try:
        width, height = parse_size(raw)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="size") from exc
    try:
        profile.validate_size(width, height)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="size") from exc
    return format_size(width, height)


def check_image_count(profile: ModelProfile, count: int, requested_size: str) -> None:
    """Reject a batch the model profile caps out, or that cannot fit in memory."""
    if count > profile.max_images_per_request:
        raise InvalidRequestError(
            f"model {profile.id!r} produces at most {profile.max_images_per_request} image(s) per request, "
            f"got n={count} (requested {requested_size})",
            param="n",
        )
