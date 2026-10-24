"""Shared fixtures: a configured service whose backends are the stub server."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest

from imagegen.app import create_app
from imagegen.config import load_config
from support import free_port

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_SERVER = REPO_ROOT / "tests" / "stub_sd_server.py"

#: Weight file names referenced by the generated config. They exist but stay empty:
#: the stub server ignores them, while the gateway's real path checks run for real.
DUMMY_WEIGHTS = ("stub_diffusion.gguf", "stub_vae.safetensors", "stub_llm.gguf", "stub_mmproj.gguf")

TEST_API_KEY = "test-secret-key"


@pytest.fixture(scope="session")
def stub_binary() -> Path:
    """Make the stub server executable so it can be launched as the engine binary."""
    mode = STUB_SERVER.stat().st_mode
    STUB_SERVER.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return STUB_SERVER


@pytest.fixture
def service_dir(tmp_path: Path, stub_binary: Path) -> Path:
    """Write a two-model test config plus its dummy weights into a temp directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in DUMMY_WEIGHTS:
        (models_dir / name).write_bytes(b"")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "service.toml").write_text(
        f"""
[service]
host = "127.0.0.1"
port = {free_port()}
api_keys = ["{TEST_API_KEY}"]
public_base_url = "http://testserver"
request_timeout_seconds = 60
max_waiting_requests = 0
blob_ttl_seconds = 60
blob_capacity = 2

[backends]
sd_server_binary = "{stub_binary}"
models_dir = "../models"
lifecycle = "on_demand"
max_resident_backends = 1
idle_timeout_seconds = 2
start_timeout_seconds = 30
stop_timeout_seconds = 15
offload_to_cpu = false
diffusion_flash_attention = false
log_level = "error"

[[models]]
id = "test-model"
description = "first stub model"
diffusion_model = "stub_diffusion.gguf"
vae = "stub_vae.safetensors"
text_encoder = "stub_llm.gguf"
vision_encoder = "stub_mmproj.gguf"
port = {free_port()}
supports_editing = true
layout = "moss"
default_size = "1024x1024"
dimension_multiple = 32
max_pixels = 1048576
max_images_per_request = 2

[models.sampling]
steps = 20
cfg = 6.0
sample_method = "euler"
scheduler = "discrete"

[[models]]
id = "test-turbo"
description = "stub distilled model"
diffusion_model = "stub_diffusion.gguf"
vae = "stub_vae.safetensors"
text_encoder = "stub_llm.gguf"
vision_encoder = "stub_mmproj.gguf"
port = {free_port()}
supports_editing = false
default_size = "512x512"
dimension_multiple = 32
max_pixels = 1048576
max_images_per_request = 1

[models.sampling]
steps = 8
cfg = 1.0
sample_method = "euler"
custom_sigmas = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]
""",
        encoding="utf-8",
    )
    return config_dir


@pytest.fixture
def config_path(service_dir: Path) -> Path:
    """Path of the generated test config."""
    return service_dir / "service.toml"


@pytest.fixture
async def client(config_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client bound to a fully started app, running its real lifespan.

    The lifespan is entered explicitly because ``httpx.ASGITransport`` does not do
    it on its own, and the lifespan is what starts the backend pool machinery.
    """
    config, config_dir = load_config(config_path)
    app = create_app(config, config_dir)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http_client:
            yield http_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers carrying the configured API key."""
    return {"Authorization": f"Bearer {TEST_API_KEY}"}
