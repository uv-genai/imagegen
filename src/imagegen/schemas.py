"""Request schemas for the OpenAI image endpoints.

Validation that depends on the resolved model lives with the routes (the model is
chosen at request time); everything that is model-independent lives here.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from imagegen.errors import InvalidRequestError
from imagegen.openai_payload import (
    GENERATION_OUTPUT_FORMATS,
    IGNORED_REQUEST_FIELDS,
    reject_extension_markers,
)


class _StrictKnownFields(BaseModel):
    """Base that accepts OpenAI's noise fields but rejects genuinely unknown ones.

    Silently dropping a misspelled parameter (`reponse_format`) makes a client
    behave differently from what its author asked for, so unknown keys are a hard
    error; the documented no-op fields are allowed through.
    """

    model_config = ConfigDict(extra="allow")

    def _reject_unknown_fields(self) -> None:
        """Raise a 400 naming the first unrecognised field."""
        extras = self.model_extra or {}
        for key in extras:
            if key not in IGNORED_REQUEST_FIELDS:
                raise InvalidRequestError(
                    f"unsupported parameter {key!r}; supported parameters are "
                    f"{', '.join(sorted(type(self).model_fields))} (plus the no-op fields "
                    f"{', '.join(IGNORED_REQUEST_FIELDS)})",
                    param=key,
                )


class GenerationRequest(_StrictKnownFields):
    """Body of ``POST /v1/images/generations``."""

    model: str | None = Field(default=None, description="Model id; required unless exactly one model is served.")
    prompt: str = Field(min_length=1, description="Text description of the image.")
    n: int = Field(default=1, ge=1, le=8, description="Number of images to produce.")
    size: str | None = Field(default=None, description="WIDTHxHEIGHT; defaults to the model's default size.")
    response_format: Literal["url", "b64_json"] = "b64_json"
    output_format: str = Field(
        default="png",
        description=f"Encoded image format, one of: {', '.join(GENERATION_OUTPUT_FORMATS)}.",
    )
    output_compression: int = Field(default=100, ge=0, le=100)
    negative_prompt: str | None = Field(
        default=None,
        description="What to avoid; overrides the model profile's default negative prompt.",
    )
    sd_cpp: dict[str, Any] | None = Field(
        default=None,
        description="Native stable-diffusion.cpp overrides, using the sdcpp request schema.",
    )

    @field_validator("output_format")
    @classmethod
    def _restrict_output_formats(cls, value: str) -> str:
        """Accept only the formats the engine advertises, from one shared list."""
        if value not in GENERATION_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {', '.join(GENERATION_OUTPUT_FORMATS)}, got {value!r}"
            )
        return value

    @field_validator("prompt")
    @classmethod
    def _reject_extension_markers(cls, value: str) -> str:
        """A prompt carrying the native block marker must be refused, not sanitised."""
        return reject_extension_markers(value)

    @field_validator("sd_cpp")
    @classmethod
    def _reject_prompt_marker_injections(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """Reject a nested prompt that would smuggle a second extension block."""
        if value is not None and "prompt" in value:
            raise ValueError("sd_cpp.prompt is not accepted; send prompt as a top-level field")
        return value

    def native_overrides(self) -> dict[str, Any]:
        """Client-supplied native parameters, with `negative_prompt` folded in."""
        overrides: dict[str, Any] = {}
        if self.negative_prompt is not None:
            overrides["negative_prompt"] = self.negative_prompt
        if self.sd_cpp:
            overrides.update(self.sd_cpp)
        return overrides


def parse_sd_cpp_form_field(raw: str | None) -> dict[str, Any] | None:
    """Decode the multipart ``sd_cpp`` field, which arrives as a JSON string."""
    if raw is None or not raw.strip():
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(f"sd_cpp must be a JSON object: {exc}", param="sd_cpp") from exc
    if not isinstance(decoded, dict):
        raise InvalidRequestError("sd_cpp must be a JSON object", param="sd_cpp")
    if "prompt" in decoded:
        raise InvalidRequestError("sd_cpp.prompt is not accepted; send prompt as a separate form field",
                                  param="sd_cpp")
    return decoded
