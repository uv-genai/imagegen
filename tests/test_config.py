"""Tests for configuration loading, validation and the shipped model registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagegen.config import ServiceConfig, load_config
from imagegen.errors import ServiceConfigurationError
from support import build_profile

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "config" / "service.toml"


def write_config(service_dir: Path, body: str) -> Path:
    """Write a config body into a temp service directory and return its path."""
    service_dir.mkdir(parents=True, exist_ok=True)
    path = service_dir / "service.toml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL_BACKEND = """
[backends]
sd_server_binary = "../build/bin/sd-server"
models_dir = "../models"
"""

MINIMAL_MODEL = """
[[models]]
id = "m1"
diffusion_model = "d.gguf"
vae = "v.safetensors"
text_encoder = "t.gguf"
port = 8111
default_size = "1024x1024"
max_pixels = 1048576

[models.sampling]
steps = 20
cfg = 4.0
sample_method = "euler"
"""


def test_shipped_config_loads_and_registers_every_model() -> None:
    """The config users actually run declares each model the service is expected to serve."""
    config, config_dir = load_config(SHIPPED_CONFIG)
    assert [profile.id for profile in config.models] == [
        "qwen-image-2.1",
        "qwen-image-2.1-turbo",
        "z-image-turbo",
        "moss-image",
        "flux-3-image-klein",
    ]
    assert config_dir == SHIPPED_CONFIG.parent
    assert config.backends.sd_server_binary.endswith("sd-server")


def test_shipped_weights_exist_on_this_machine() -> None:
    """verify_weights must pass for the shipped config on a provisioned machine."""
    config, config_dir = load_config(SHIPPED_CONFIG)
    missing = [
        path
        for profile in config.models
        for path in (config.backends.weight_path(config_dir, profile.diffusion_model),)
        if not path.is_file()
    ]
    if missing:
        pytest.skip("weights not downloaded; run scripts/fetch_models.sh")
    config.verify_weights(config_dir)


def test_load_reports_missing_file(tmp_path: Path) -> None:
    """A config path that does not exist is a startup error, not a silent default."""
    with pytest.raises(ServiceConfigurationError):
        load_config(tmp_path / "nope.toml")


def test_load_reports_invalid_toml(tmp_path: Path) -> None:
    """Malformed TOML is reported with the offending file, not raised as TOMLDecodeError."""
    path = write_config(tmp_path / "svc", "this is not = = toml")
    with pytest.raises(ServiceConfigurationError):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A typo in a setting name must not be silently ignored."""
    path = write_config(
        tmp_path / "svc",
        MINIMAL_BACKEND + MINIMAL_MODEL + '\n[service]\nprot = 9000\n',
    )
    with pytest.raises(ServiceConfigurationError, match="prot"):
        load_config(path)


def test_at_least_one_model_is_required(tmp_path: Path) -> None:
    """An empty registry would serve nothing but 404s; it is refused at startup."""
    path = write_config(tmp_path / "svc", MINIMAL_BACKEND)
    with pytest.raises(ServiceConfigurationError, match="models"):
        load_config(path)


def test_duplicate_model_ids_are_rejected(tmp_path: Path) -> None:
    """Two models with the same id could not be routed to unambiguously."""
    body = MINIMAL_BACKEND + MINIMAL_MODEL + MINIMAL_MODEL.replace("port = 8111", "port = 8112")
    path = write_config(tmp_path / "svc", body)
    with pytest.raises(ServiceConfigurationError, match="duplicate model id"):
        load_config(path)


def test_duplicate_ports_are_rejected(tmp_path: Path) -> None:
    """Two backends on one port cannot both bind; catch it before launching either."""
    body = MINIMAL_BACKEND + MINIMAL_MODEL + MINIMAL_MODEL.replace('id = "m1"', 'id = "m2"')
    path = write_config(tmp_path / "svc", body)
    with pytest.raises(ServiceConfigurationError, match="reuses backend port"):
        load_config(path)


def make_fake_binary(service_dir: Path) -> Path:
    """Create the sd-server executable referenced by MINIMAL_BACKEND.

    verify_weights checks the binary before the weights, so tests that target a
    weight problem have to satisfy the binary check first.
    """
    binary = service_dir.parent / "build" / "bin" / "sd-server"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    return binary


def test_verify_weights_reports_missing_file(tmp_path: Path) -> None:
    """A registry entry pointing at an absent weight file fails before serving."""
    service_dir = tmp_path / "svc"
    path = write_config(service_dir, MINIMAL_BACKEND + MINIMAL_MODEL)
    make_fake_binary(service_dir)
    config, config_dir = load_config(path)
    with pytest.raises(ServiceConfigurationError, match="weight file not found"):
        config.verify_weights(config_dir)


def test_verify_weights_reports_weight_with_wrong_extension(tmp_path: Path) -> None:
    """A bin/pth file is not loadable by the engine and is rejected by name."""
    service_dir = tmp_path / "svc"
    body = MINIMAL_BACKEND + MINIMAL_MODEL.replace('diffusion_model = "d.gguf"', 'diffusion_model = "d.bin"')
    path = write_config(service_dir, body)
    make_fake_binary(service_dir)
    config, config_dir = load_config(path)
    with pytest.raises(ServiceConfigurationError, match="not a gguf/safetensors file"):
        config.verify_weights(config_dir)


def test_setup_problems_lists_every_missing_artefact_at_once(tmp_path: Path) -> None:
    """`doctor` must report a half-provisioned machine in one pass.

    Reporting one problem per run would make a fresh install require a round trip
    for each of the five weight files.
    """
    service_dir = tmp_path / "svc"
    body = MINIMAL_BACKEND + MINIMAL_MODEL + MINIMAL_MODEL.replace('id = "m1"', 'id = "m2"').replace("port = 8111", "port = 8112")
    path = write_config(service_dir, body)
    make_fake_binary(service_dir)
    config, config_dir = load_config(path)

    problems = config.setup_problems(config_dir)
    # Three weight files per model (diffusion, vae, text encoder), two models.
    assert len(problems) == 6, problems
    assert {problem.split(":")[0].strip() for problem in problems} == {"model 'm1'", "model 'm2'"}

    # verify_weights keeps failing startup, and its message carries the whole list.
    with pytest.raises(ServiceConfigurationError) as caught:
        config.verify_weights(config_dir)
    assert len(str(caught.value).splitlines()) == 6


def test_setup_problems_is_empty_when_everything_exists(tmp_path: Path) -> None:
    """A fully provisioned machine produces no diagnostics."""
    service_dir = tmp_path / "svc"
    path = write_config(service_dir, MINIMAL_BACKEND + MINIMAL_MODEL)
    make_fake_binary(service_dir)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    for name in ("d.gguf", "v.safetensors", "t.gguf"):
        (models_dir / name).write_bytes(b"")
    config, config_dir = load_config(path)
    assert config.setup_problems(config_dir) == []


def test_size_must_be_a_multiple_of_the_model_step() -> None:
    """1000 is not representable by a model that requires multiples of 32."""
    profile = build_profile()
    with pytest.raises(ValueError, match="multiple of 32"):
        profile.validate_size(1000, 1024)


def test_z_image_uses_its_own_dimension_step() -> None:
    """Z-Image patches 2x2 over an 8x VAE, so its step is 16 rather than 32.

    Pinned to the shipped profile because 1008x1008 was generated successfully on a
    live sd-server; "simplifying" the step back to 32 would reject valid requests.
    """
    config, _ = load_config(SHIPPED_CONFIG)
    profile = config.profile("z-image-turbo")
    assert profile is not None
    assert profile.dimension_multiple == 16
    profile.validate_size(1008, 1008)  # 16 * 63 — valid here, invalid for the Qwen models
    with pytest.raises(ValueError, match="multiple of 16"):
        profile.validate_size(1000, 1024)


def test_size_must_respect_the_pixel_budget() -> None:
    """The pixel cap is the memory guard, so exceeding it is an error."""
    profile = build_profile(max_pixels=1048576)
    with pytest.raises(ValueError, match="above the 1048576 pixel limit"):
        profile.validate_size(2048, 2048)


def test_default_size_is_validated_at_startup() -> None:
    """An impossible default must stop the service rather than fail the first request."""
    with pytest.raises(ValueError, match="multiple of 32"):
        build_profile(default_size="1000x1000")


def test_editing_without_vision_encoder_is_allowed() -> None:
    """The flux2/flux3 edit preset needs no vision tower: references reach the
    DiT through the VAE. Decoupling the two was forced by FLUX.3 Image Klein."""
    profile = build_profile(supports_editing=True, vision_encoder=None)
    assert profile.supports_editing is True


def test_vision_encoder_with_editing_disabled_is_allowed() -> None:
    """A configured tower with editing off is a dormant file reference, not an
    error: the flag is simply not passed (test-turbo ships exactly this shape)."""
    profile = build_profile(supports_editing=False, vision_encoder="stub_mmproj.gguf")
    assert profile.vision_encoder == "stub_mmproj.gguf"


def test_shipped_flux3_profile_edits_without_a_vision_tower() -> None:
    """The decoupling above is load-bearing: the shipped config depends on it."""
    config, _ = load_config(SHIPPED_CONFIG)
    profile = config.profile("flux-3-image-klein")
    assert profile.supports_editing and profile.vision_encoder is None


def test_sampling_rejects_out_of_range_sigmas() -> None:
    """Sigmas outside [0, 1] describe a schedule the sampler cannot run."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        build_profile(sampling={"steps": 8, "cfg": 1.0, "sample_method": "euler", "custom_sigmas": [1.0, 1.5]})


def test_custom_config_with_relative_paths_resolves_against_config_dir(tmp_path: Path) -> None:
    """Relative weight and binary paths follow the config file, making it relocatable."""
    path = write_config(tmp_path / "deploy" / "conf", MINIMAL_BACKEND + MINIMAL_MODEL)
    config, config_dir = load_config(path)
    assert config.backends.weight_path(config_dir, "d.gguf") == (tmp_path / "deploy" / "models" / "d.gguf")
    assert config.backends.resolved_binary(config_dir) == (tmp_path / "deploy" / "build" / "bin" / "sd-server")


def test_service_config_rejects_unknown_model_id_characters() -> None:
    """Model ids appear in URLs and logs, so they are restricted to a safe alphabet."""
    with pytest.raises(ValueError):
        build_profile(id="not an id!")
