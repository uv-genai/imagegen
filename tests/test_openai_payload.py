"""Tests for the OpenAI -> stable-diffusion.cpp translation layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from imagegen.config import load_config
from imagegen.errors import InvalidRequestError
from imagegen.openai_payload import (
    EXTRA_ARGS_CLOSE,
    EXTRA_ARGS_OPEN,
    build_generation_payload,
    build_native_args,
    compose_prompt,
    deep_merge,
    format_size,
    parse_size,
)
from imagegen.schemas import GenerationRequest, parse_sd_cpp_form_field

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "config" / "service.toml"

#: The schedule shipped in Qwen/Qwen-Image-2.1-Turbo/model_index.json plus 0.0.
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]


@pytest.fixture(scope="module")
def shipped_profiles():
    """The two real model profiles from config/service.toml."""
    config, _ = load_config(SHIPPED_CONFIG)
    return {profile.id: profile for profile in config.models}


def test_turbo_native_args_carry_its_exact_schedule(shipped_profiles) -> None:
    """A Turbo request must send 8 steps, CFG 1 and the checkpoint's own sigmas."""
    native = build_native_args(shipped_profiles["qwen-image-2.1-turbo"], None)
    sample = native["sample_params"]
    assert sample["sample_steps"] == 8
    assert sample["guidance"]["txt_cfg"] == 1.0
    assert sample["sample_method"] == "euler"
    assert sample["custom_sigmas"] == TURBO_SIGMAS


def test_base_profile_does_not_force_a_sigma_schedule(shipped_profiles) -> None:
    """The base checkpoint must fall back to the backend's resolution-aware schedule."""
    native = build_native_args(shipped_profiles["qwen-image-2.1"], None)
    sample = native["sample_params"]
    assert "custom_sigmas" not in sample
    assert sample["sample_steps"] == 40
    assert sample["guidance"]["txt_cfg"] == 6.0


def test_compose_prompt_embeds_valid_json_block(shipped_profiles) -> None:
    """The emitted block must parse back to exactly the native args we built."""
    profile = shipped_profiles["qwen-image-2.1"]
    composed = compose_prompt("a cat", build_native_args(profile, None))
    assert composed.startswith("a cat " + EXTRA_ARGS_OPEN)
    assert composed.endswith(EXTRA_ARGS_CLOSE)
    inner = composed.split(EXTRA_ARGS_OPEN, 1)[1].rsplit(EXTRA_ARGS_CLOSE, 1)[0]
    assert json.loads(inner) == build_native_args(profile, None)


def test_compose_prompt_rejects_nested_markers(shipped_profiles) -> None:
    """A prompt carrying the marker could rewrite the args block, so it is refused."""
    with pytest.raises(ValueError):
        compose_prompt(f"hello {EXTRA_ARGS_OPEN}evil{EXTRA_ARGS_CLOSE}", {})


def test_deep_merge_preserves_untouched_sibling_keys() -> None:
    """A client override of one sample parameter must not drop the rest."""
    base = {"sample_params": {"sample_steps": 8, "custom_sigmas": [1.0, 0.0], "guidance": {"txt_cfg": 1.0}}}
    merged = deep_merge(base, {"sample_params": {"eta": 0.5}})
    assert merged["sample_params"]["eta"] == 0.5
    assert merged["sample_params"]["sample_steps"] == 8
    assert merged["sample_params"]["custom_sigmas"] == [1.0, 0.0]
    assert merged["sample_params"]["guidance"] == {"txt_cfg": 1.0}


def test_client_negative_prompt_overrides_profile_default(shipped_profiles) -> None:
    """`negative_prompt` is a first-class field and must beat the profile default."""
    request = GenerationRequest(model="x", prompt="p", negative_prompt="blurry")
    native = build_native_args(shipped_profiles["qwen-image-2.1"], request.native_overrides())
    assert native["negative_prompt"] == "blurry"


def test_sd_cpp_wins_over_the_convenience_negative_prompt(shipped_profiles) -> None:
    """When both spellings are sent, the explicit native block takes precedence."""
    request = GenerationRequest(
        model="x", prompt="p", negative_prompt="blurry", sd_cpp={"negative_prompt": "watermark"}
    )
    native = build_native_args(shipped_profiles["qwen-image-2.1"], request.native_overrides())
    assert native["negative_prompt"] == "watermark"


def test_sd_cpp_cannot_replace_the_prompt() -> None:
    """`sd_cpp.prompt` would bypass prompt validation, so it is rejected."""
    with pytest.raises(ValueError):
        GenerationRequest(model="x", prompt="p", sd_cpp={"prompt": "other"})


def test_sd_cpp_form_field_parsing() -> None:
    """The multipart field arrives as a JSON string and is validated as an object."""
    assert parse_sd_cpp_form_field(None) is None
    assert parse_sd_cpp_form_field("  ") is None
    assert parse_sd_cpp_form_field('{"seed": 7}') == {"seed": 7}
    with pytest.raises(InvalidRequestError):
        parse_sd_cpp_form_field("[1, 2]")
    with pytest.raises(InvalidRequestError):
        parse_sd_cpp_form_field("{not json")


def test_build_generation_payload_matches_upstream_fields(shipped_profiles) -> None:
    """Only the fields the upstream OpenAI route reads may be sent at top level."""
    payload = build_generation_payload(
        shipped_profiles["qwen-image-2.1"],
        prompt="a cat",
        size="1024x768",
        count=2,
        output_format="webp",
        output_compression=90,
        client_native_overrides=None,
    )
    assert set(payload) == {"prompt", "n", "size", "output_format", "output_compression"}
    assert payload["n"] == 2
    assert payload["size"] == "1024x768"
    assert payload["output_format"] == "webp"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("1024x1024", (1024, 1024)), ("256x512", (256, 512))],
)
def test_parse_size_accepts_canonical_forms(text: str, expected: tuple[int, int]) -> None:
    """Canonical sizes round-trip through parse and format."""
    assert parse_size(text) == expected
    assert format_size(*expected) == text


@pytest.mark.parametrize("text", ["1024", "1024x", "x1024", "axb", "0x1024", "-32x32", "1024x1024x1024"])
def test_parse_size_rejects_malformed_values(text: str) -> None:
    """Anything but positive `WxH` is refused rather than silently repaired."""
    with pytest.raises(ValueError):
        parse_size(text)
