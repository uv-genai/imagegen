"""Tests for sd-server command-line construction and process supervision."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path

import httpx
import pytest

from imagegen.backend_process import BackendProcess, BackendState, build_server_args
from imagegen.config import load_config
from imagegen.errors import BackendUnavailableError

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "config" / "service.toml"
STUB_SERVER = REPO_ROOT / "tests" / "stub_sd_server.py"

#: Flags the pinned sd-server build documents and that the launcher must emit.
KNOWN_SERVER_FLAGS = {
    "--diffusion-model",
    "--vae",
    "--llm",
    "--llm_vision",
    "--tokenizer",
    "--listen-ip",
    "--listen-port",
    "--log-level",
    "--offload-to-cpu",
    "--flash-attn",
    "--vae-tiling",
    "--max-vram",
    "--cfg-scale",
    "--steps",
    "--sampling-method",
    "--scheduler",
    "--flow-shift",
    "--sigmas",
}


@pytest.fixture
def shipped():
    """The shipped config plus its directory."""
    return load_config(SHIPPED_CONFIG)


def test_generated_command_only_uses_documented_flags(shipped) -> None:
    """Every flag we pass must exist in the pinned sd-server's own help text."""
    config, config_dir = shipped
    for profile in config.models:
        args = build_server_args(profile, config.backends, config_dir)
        used = {token for token in args if token.startswith("--")}
        unknown = used - KNOWN_SERVER_FLAGS - {args[0]}
        assert not unknown, f"{profile.id} passes undocumented flags: {sorted(unknown)}"


def test_turbo_command_carries_its_sigma_schedule(shipped) -> None:
    """The launched Turbo process must be correct even without per-request args."""
    config, config_dir = shipped
    profile = config.profile("qwen-image-2.1-turbo")
    args = build_server_args(profile, config.backends, config_dir)
    assert args[args.index("--sigmas") + 1] == "1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0"
    assert args[args.index("--steps") + 1] == "8"
    assert args[args.index("--cfg-scale") + 1] == "1.0"


def test_base_command_omits_sigma_and_flow_shift(shipped) -> None:
    """The base model must let the backend pick its resolution-dependent schedule."""
    config, config_dir = shipped
    args = build_server_args(config.profile("qwen-image-2.1"), config.backends, config_dir)
    assert "--sigmas" not in args
    assert "--flow-shift" not in args


def test_weight_paths_are_absolute_and_exist(shipped) -> None:
    """The engine is handed absolute paths, so it does not depend on the CWD.

    `--llm_vision` is only expected where the profile has a vision tower: the
    flux2/flux3 edit preset passes references through the VAE and needs none,
    and a text-encoder-only model such as Z-Image-Turbo cannot edit at all.
    `--tokenizer` only appears where the vocabulary is not embedded (MOSS).
    """
    config, config_dir = shipped
    for profile in config.models:
        args = build_server_args(profile, config.backends, config_dir)
        flags = ["--diffusion-model", "--vae", "--llm"]
        if profile.vision_encoder is not None:
            flags.append("--llm_vision")
        else:
            assert "--llm_vision" not in args, f"{profile.id} has no vision tower to pass"
        if profile.tokenizer is not None:
            flags.append("--tokenizer")
        else:
            assert "--tokenizer" not in args, f"{profile.id} embeds its vocabulary"
        for flag in flags:
            value = args[args.index(flag) + 1]
            assert Path(value).is_absolute(), f"{flag} must be absolute, got {value}"
            if not Path(value).is_file():
                pytest.skip("weights not downloaded; run scripts/fetch_models.sh")


def test_moss_command_carries_its_external_tokenizer(shipped) -> None:
    """Qwen3.5's vocabulary is not embedded in sd.cpp; omitting it garbles prompts."""
    config, config_dir = shipped
    args = build_server_args(config.profile("moss-image"), config.backends, config_dir)
    assert args[args.index("--tokenizer") + 1].endswith("tokenizer.json")
    assert args[args.index("--flow-shift") + 1] == "5.0"
    assert args[args.index("--cfg-scale") + 1] == "3.0"


def test_flux3_edits_without_a_vision_tower(shipped) -> None:
    """The flux2 preset sends references to the DiT via the VAE: editing works
    with supports_editing=true and vision_encoder unset, and no --llm_vision."""
    config, config_dir = shipped
    profile = config.profile("flux-3-image-klein")
    assert profile.supports_editing is True
    assert profile.vision_encoder is None
    args = build_server_args(profile, config.backends, config_dir)
    assert "--llm_vision" not in args
    assert "--tokenizer" not in args  # Qwen3-VL's tokenizer is embedded
    assert args[args.index("--flow-shift") + 1] == "6.93"
    assert args[args.index("--steps") + 1] == "16"


def test_backend_binds_loopback_only(shipped) -> None:
    """Backends have no authentication, so they must never listen off-loopback."""
    config, config_dir = shipped
    for profile in config.models:
        args = build_server_args(profile, config.backends, config_dir)
        assert args[args.index("--listen-ip") + 1] == "127.0.0.1"


def make_stub_process(tmp_path: Path, port: int) -> BackendProcess:
    """Build a BackendProcess that launches the stub server on a free port."""
    STUB_SERVER.chmod(STUB_SERVER.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    for name in ("d.gguf", "v.safetensors", "t.gguf"):
        (models_dir / name).write_bytes(b"")
    config, config_dir = load_config(tmp_path / "config" / "service.toml")
    profile = config.profile("test-model")
    from imagegen.config import ModelProfile  # local import: only needed for the type override

    profile = ModelProfile.model_validate(
        {**profile.model_dump(), "port": port, "diffusion_model": "d.gguf", "vae": "v.safetensors",
         "text_encoder": "t.gguf", "supports_editing": False, "vision_encoder": None}
    )
    return BackendProcess(profile, config.backends, config_dir)


async def test_process_starts_ready_then_stops(tmp_path: Path, service_dir: Path) -> None:
    """A backend reaches READY only when its capabilities endpoint answers."""
    from support import free_port

    config_source = service_dir / "service.toml"
    target = tmp_path / "config" / "service.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config_source.read_text(encoding="utf-8"), encoding="utf-8")

    port = free_port()
    process = make_stub_process(tmp_path, port)
    assert process.state is BackendState.STOPPED
    await process.start()
    try:
        assert process.state is BackendState.READY
        assert process.is_alive()
        assert process.pid is not None
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{process.base_url}/sdcpp/v1/capabilities")
            assert response.status_code == 200
    finally:
        pid = process.pid
        await process.stop()

    assert process.state is BackendState.STOPPED
    assert process.pid is None
    with pytest.raises(httpx.HTTPError):
        async with httpx.AsyncClient(timeout=0.5) as client:
            await client.get(f"http://127.0.0.1:{port}/_state")
    assert pid is not None


async def test_start_reports_failure_of_a_binary_that_cannot_serve(tmp_path: Path) -> None:
    """A binary that exits immediately must fail fast, not hang for the whole timeout."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir()
    for name in ("d.gguf", "v.safetensors", "t.gguf"):
        (tmp_path / "models" / name).write_bytes(b"")
    failing_script = tmp_path / "broken_engine.py"
    failing_script.write_text("#!/usr/bin/env python3\nraise SystemExit(3)\n", encoding="utf-8")
    failing_script.chmod(0o755)

    from imagegen.config import BackendSettings, ModelProfile, ServiceConfig

    config = ServiceConfig.model_validate(
        {
            "backends": {
                "sd_server_binary": str(failing_script),
                "models_dir": str(tmp_path / "models"),
                "start_timeout_seconds": 30,
            },
            "models": [
                {
                    "id": "broken",
                    "diffusion_model": "d.gguf",
                    "vae": "v.safetensors",
                    "text_encoder": "t.gguf",
                    "port": 8765,
                    "default_size": "512x512",
                    "max_pixels": 262144,
                    "sampling": {"steps": 4, "cfg": 1.0, "sample_method": "euler"},
                }
            ],
        }
    )
    profile = config.models[0]
    process = BackendProcess(profile, config.backends, config_dir)

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(BackendUnavailableError, match="exited with code 3"):
        await process.start()
    assert loop.time() - started < 20, "a dead backend must not consume the full readiness timeout"
    assert process.state is BackendState.STOPPED
