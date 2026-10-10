"""Shared test helpers.

Imported as ``support`` — pytest puts this directory on ``sys.path`` because the
test modules are not packaged.
"""

from __future__ import annotations

import socket

from imagegen.config import ModelProfile


def free_port() -> int:
    """Ask the OS for an unused loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_profile(**overrides) -> ModelProfile:  # noqa: ANN003 - convenience wrapper over the model
    """Construct a valid profile with the given fields replaced.

    Defaults describe a plain model: no vision tower, one image per request, a
    1024x1024 default inside a 1 MP budget.
    """
    fields: dict[str, object] = {
        "id": "p",
        "diffusion_model": "d.gguf",
        "vae": "v.safetensors",
        "text_encoder": "t.gguf",
        "port": 8111,
        "default_size": "1024x1024",
        "max_pixels": 1048576,
        "sampling": {"steps": 20, "cfg": 4.0, "sample_method": "euler"},
    }
    fields.update(overrides)
    return ModelProfile.model_validate(fields)
