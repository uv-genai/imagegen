"""Tests for the capability cross-check between config and a loaded backend."""

from __future__ import annotations

from imagegen.backend_pool import capability_mismatches
from support import build_profile


def capabilities(**overrides) -> dict[str, object]:  # noqa: ANN003 - payload under test
    """A capabilities payload shaped like sd-server's, with fields replaced."""
    payload: dict[str, object] = {
        "supported_modes": ["img_gen"],
        "model": {"name": "m", "stem": "m", "path": "/models/m.gguf"},
        "output_formats": ["png", "jpeg", "webp"],
        "limits": {
            "max_batch_count": 4,
            "max_width": 4096,
            "max_height": 4096,
            "min_width": 64,
            "min_height": 64,
        },
    }
    payload.update(overrides)
    return payload


def test_matching_configuration_reports_no_problems() -> None:
    """The shipped shape of a profile against a healthy backend is silent."""
    profile = build_profile(max_images_per_request=2, default_size="1024x1024")
    assert capability_mismatches(profile, capabilities()) == []


def test_format_the_engine_cannot_encode_is_reported() -> None:
    """A backend that drops webp would fail requests we promised to serve."""
    problems = capability_mismatches(build_profile(), capabilities(output_formats=["png", "jpeg"]))
    assert any("webp" in problem for problem in problems)


def test_batch_size_above_the_backend_limit_is_reported() -> None:
    """Asking for more images per request than the engine allows is a config defect."""
    profile = build_profile(max_images_per_request=8)
    problems = capability_mismatches(profile, capabilities())
    assert any("max_batch_count=4" in problem for problem in problems)


def test_default_size_above_backend_dimension_limit_is_reported() -> None:
    """A default size the backend would reject must be caught at load time."""
    profile = build_profile(default_size="2048x2048", max_pixels=4194304)
    payload = capabilities(limits={"max_batch_count": 4, "max_width": 1024, "max_height": 4096})
    problems = capability_mismatches(profile, payload)
    assert any("max_width=1024" in problem for problem in problems)


def test_missing_capability_sections_do_not_raise() -> None:
    """A partial payload reports nothing rather than crashing a live request."""
    assert capability_mismatches(build_profile(), {}) == []
    assert capability_mismatches(build_profile(), {"limits": "unexpected"}) == []
