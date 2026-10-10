"""Translation between OpenAI image requests and stable-diffusion.cpp semantics.

The upstream server exposes an OpenAI-shaped endpoint but reads only ``prompt``,
``n``, ``size``, ``output_format`` and ``output_compression`` from it. Everything
native — steps, CFG, sigma schedules, negative prompts — travels inside a
``<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`` block appended to the prompt,
which the server strips out before sampling. This module is the only place that
knows that wire format.
"""

from __future__ import annotations

import json
from typing import Any

from imagegen.config import ModelProfile, SamplingDefaults

#: Marker strings of the upstream native-extension block.
EXTRA_ARGS_OPEN = "<sd_cpp_extra_args>"
EXTRA_ARGS_CLOSE = "</sd_cpp_extra_args>"

#: Output formats the service accepts for generations, as documented for the
#: upstream OpenAI route and confirmed live: the pinned build answers
#: GET /sdcpp/v1/capabilities with
#:   output_formats = ["png", "jpeg", "webp"].
#: A format outside this list would fail inside the backend only after a full
#: generation, so it is refused up front. When a model is loaded the pool
#: re-checks this list against what that build actually advertises.
GENERATION_OUTPUT_FORMATS = ("png", "jpeg", "webp")

#: Default `strength` for image edits.
#:
#: The upstream edits route assigns the first uploaded image to *both*
#: `init_image` and `ref_images`. With the engine's default strength of 0.75 the
#: init image dominates and the pipeline reproduces the input instead of editing
#: it — measured: "change 'BASE MODEL' to 'EDITED OK'" came back with the
#: original text intact. At 1.0 sampling starts from pure noise and the edit is
#: driven by the reference image, which is what `sd-cli -r` does and what
#: produced the correct result.
EDIT_STRENGTH_DEFAULT = 1.0

#: Request fields accepted but intentionally not acted upon. OpenAI clients send
#: them routinely and none of them has a stable-diffusion.cpp equivalent here.
IGNORED_REQUEST_FIELDS = ("user", "style", "quality", "background", "input_fidelity")


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overrides`` applied, recursing into nested dicts.

    Scalars and lists in ``overrides`` replace the value in ``base`` outright;
    only dicts merge, so a client that sends ``{"sample_params": {"eta": 0.5}}``
    keeps the profile's steps, CFG and sigma list.
    """
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def sampling_to_native(sampling: SamplingDefaults) -> dict[str, Any]:
    """Convert a profile's sampling defaults to the native request schema.

    ``cfg`` maps to ``sample_params.guidance.txt_cfg`` and the sigma schedule to
    ``sample_params.custom_sigmas``, matching the names the C++ parser expects.
    """
    guidance: dict[str, Any] = {"txt_cfg": sampling.cfg}
    sample_params: dict[str, Any] = {
        "sample_steps": sampling.steps,
        "sample_method": sampling.sample_method,
        "guidance": guidance,
    }
    if sampling.scheduler is not None:
        sample_params["scheduler"] = sampling.scheduler
    if sampling.flow_shift is not None:
        sample_params["flow_shift"] = sampling.flow_shift
    if sampling.custom_sigmas:
        sample_params["custom_sigmas"] = list(sampling.custom_sigmas)
    native: dict[str, Any] = {"sample_params": sample_params}
    if sampling.negative_prompt:
        native["negative_prompt"] = sampling.negative_prompt
    return native


def build_native_args(profile: ModelProfile, client_overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge client-supplied native overrides on top of the profile defaults.

    Args:
        profile: Resolved model profile providing the defaults.
        client_overrides: Contents of the request's ``sd_cpp`` field, if any.
    """
    defaults = sampling_to_native(profile.sampling)
    if not client_overrides:
        return defaults
    return deep_merge(defaults, client_overrides)


def reject_extension_markers(prompt: str) -> str:
    """Reject a prompt that carries the extension marker itself.

    The upstream server strips its first ``<sd_cpp_extra_args>`` block with a
    regex, so a client-supplied marker would either swallow our parameters or be
    read as one, letting the caller set values the service intended to control.
    """
    if EXTRA_ARGS_OPEN in prompt or EXTRA_ARGS_CLOSE in prompt:
        raise ValueError(f"prompt may not contain {EXTRA_ARGS_OPEN}")
    return prompt


def compose_prompt(prompt: str, native_args: dict[str, Any]) -> str:
    """Append the native extension block to a prompt.

    Raises:
        ValueError: the prompt contains an extension marker; see
            :func:`reject_extension_markers`.
    """
    reject_extension_markers(prompt)
    payload = json.dumps(native_args, separators=(",", ":"))
    return f"{prompt} {EXTRA_ARGS_OPEN}{payload}{EXTRA_ARGS_CLOSE}"


def parse_size(size: str) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` string into a pair of positive integers.

    Raises:
        ValueError: the string is not ``<positive digits>x<positive digits>``.
    """
    raw_width, separator, raw_height = size.partition("x")
    if not separator or not raw_width.isdigit() or not raw_height.isdigit():
        raise ValueError(f"size must be WIDTHxHEIGHT, got {size!r}")
    width, height = int(raw_width), int(raw_height)
    if width <= 0 or height <= 0:
        raise ValueError(f"size must be positive, got {size!r}")
    return width, height


def format_size(width: int, height: int) -> str:
    """Render a width/height pair as the ``WIDTHxHEIGHT`` string the API uses."""
    return f"{width}x{height}"


def build_generation_payload(
    profile: ModelProfile,
    *,
    prompt: str,
    size: str,
    count: int,
    output_format: str,
    output_compression: int,
    client_native_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the JSON body to POST to the backend's ``/v1/images/generations``.

    The payload is deliberately minimal: any field left out keeps whatever the
    backend was launched with, which the profile already seeded.
    """
    return {
        "prompt": compose_prompt(prompt, build_native_args(profile, client_native_overrides)),
        "n": count,
        "size": size,
        "output_format": output_format,
        "output_compression": output_compression,
    }
