"""End-to-end HTTP tests against the gateway, with stub sd-server backends.

Every test here goes through the real ASGI app, the real backend pool, real
subprocess launches and real HTTP to the backend; only inference itself is
replaced by the stub in tests/stub_sd_server.py.
"""

from __future__ import annotations

import asyncio
import base64
import time

import httpx
import pytest

from imagegen.config import load_config

PNG_MAGIC = b"\x89PNG"


def ports(config_path) -> dict[str, int]:  # noqa: ANN001 - receives the config path from the fixture
    """Map model id to its stub backend port."""
    config, _ = load_config(config_path)
    return {profile.id: profile.port for profile in config.models}


async def stub_state(port: int) -> dict[str, object]:
    """Ask a running stub backend what it last received."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"http://127.0.0.1:{port}/_state")
        response.raise_for_status()
        return response.json()


async def wait_for_backend_gone(port: int, timeout: float = 15.0) -> None:
    """Block until a backend stops accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=0.5) as client:
                await client.get(f"http://127.0.0.1:{port}/_state")
        except httpx.HTTPError:
            return
        await asyncio.sleep(0.25)
    raise AssertionError(f"backend on port {port} is still answering after {timeout}s")


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


async def test_models_endpoint_lists_configured_models(client: httpx.AsyncClient) -> None:
    """`GET /v1/models` is unauthenticated in OpenAI's shape and lists both models."""
    response = await client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [entry["id"] for entry in body["data"]] == ["test-model", "test-turbo"]
    assert all(entry["object"] == "model" for entry in body["data"])


async def test_models_endpoint_advertises_layout_only_where_configured(
    client: httpx.AsyncClient,
) -> None:
    """A profile with a dialect gains a `layout` key; one without omits it entirely.

    Omission — not `null` — is the contract: clients distinguish "this model has no
    layout support" from "this engine does not speak the extension".
    """
    response = await client.get("/v1/models")
    entries = {entry["id"]: entry for entry in response.json()["data"]}
    assert entries["test-model"]["layout"] == "moss"
    assert "layout" not in entries["test-turbo"]


async def test_health_is_unauthenticated(client: httpx.AsyncClient) -> None:
    """Liveness must not require credentials."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------


async def test_generation_without_key_is_rejected(client: httpx.AsyncClient) -> None:
    """With keys configured, an anonymous request never reaches the GPU."""
    response = await client.post("/v1/images/generations", json={"model": "test-model", "prompt": "a cat"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


async def test_generation_with_wrong_key_is_rejected(client: httpx.AsyncClient) -> None:
    """A wrong key is refused the same way a missing one is."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a cat"},
        headers={"Authorization": "Bearer not-the-key"},
    )
    assert response.status_code == 401


async def test_x_api_key_header_is_accepted(client: httpx.AsyncClient, auth_headers: dict[str, str]) -> None:
    """Both OpenAI-style auth headers work."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a cat"},
        headers={"x-api-key": auth_headers["Authorization"].removeprefix("Bearer ")},
    )
    assert response.status_code == 200


# --------------------------------------------------------------------------
# Generation happy path and wire format
# --------------------------------------------------------------------------


async def test_generation_returns_decodable_png(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """The documented default response is base64 PNG image data."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a cat holding a sign"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["output_format"] == "png"
    assert isinstance(body["created"], int)
    assert len(body["data"]) == 1
    assert base64.b64decode(body["data"][0]["b64_json"])[:4] == PNG_MAGIC


async def test_generation_launches_backend_on_first_request(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """on_demand lifecycle: the backend must be down until something needs it."""
    port = ports(config_path)["test-model"]
    with pytest.raises(httpx.HTTPError):
        await stub_state(port)

    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "wake up"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    state = await stub_state(port)
    assert state["diffusion_model"].endswith("stub_diffusion.gguf")


async def test_base_model_sends_its_own_sampling(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """The profile's 20 steps / CFG 6 must reach the backend inside the prompt block."""
    await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "sample me"},
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-model"])
    sample = state["last_extra_args"]["sample_params"]
    assert sample["sample_steps"] == 20
    assert sample["guidance"]["txt_cfg"] == 6.0
    assert "custom_sigmas" not in sample


async def test_turbo_model_sends_its_fixed_schedule(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """Turbo is only correct with 8 steps, CFG 1 and its own sigma list."""
    await client.post(
        "/v1/images/generations",
        json={"model": "test-turbo", "prompt": "turbo me"},
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-turbo"])
    sample = state["last_extra_args"]["sample_params"]
    assert sample["sample_steps"] == 8
    assert sample["guidance"]["txt_cfg"] == 1.0
    assert sample["custom_sigmas"][-1] == 0.0
    assert len(sample["custom_sigmas"]) == 9


async def test_native_overrides_reach_the_backend(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """The `sd_cpp` escape hatch must not clobber the profile's other parameters."""
    await client.post(
        "/v1/images/generations",
        json={
            "model": "test-turbo",
            "prompt": "override me",
            "sd_cpp": {"sample_params": {"eta": 0.5}, "seed": 1234},
        },
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-turbo"])
    sent = state["last_extra_args"]
    assert sent["seed"] == 1234
    assert sent["sample_params"]["eta"] == 0.5
    assert sent["sample_params"]["sample_steps"] == 8
    assert len(sent["sample_params"]["custom_sigmas"]) == 9


async def test_prompt_reaching_backend_has_no_extension_block(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """The upstream server strips the block; the prompt it samples must be clean."""
    await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a red fox"},
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-model"])
    assert "sd_cpp_extra_args" in state["last_request"]["prompt"]
    assert state["last_request"]["prompt"].startswith("a red fox <sd_cpp_extra_args>")


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "status", "fragment"),
    [
        ({"prompt": "x"}, 400, "model"),
        ({"model": "nope", "prompt": "x"}, 404, "unknown model"),
        ({"model": "test-model", "prompt": ""}, 400, "prompt"),
        ({"model": "test-model", "prompt": "x", "size": "1000x1000"}, 400, "multiple of 32"),
        ({"model": "test-model", "prompt": "x", "size": "2048x2048"}, 400, "pixel"),
        ({"model": "test-model", "prompt": "x", "size": "wide"}, 400, "size"),
        ({"model": "test-model", "prompt": "x", "n": 3}, 400, "at most 2 image"),
        ({"model": "test-turbo", "prompt": "x", "n": 2}, 400, "at most 1 image"),
        ({"model": "test-model", "prompt": "x", "reponse_format": "url"}, 400, "reponse_format"),
        ({"model": "test-model", "prompt": f"x {'<sd_cpp_extra_args>'}"}, 400, "prompt"),
    ],
)
async def test_invalid_requests_are_rejected_before_any_backend_work(
    client: httpx.AsyncClient, auth_headers: dict[str, str], payload: dict[str, object], status: int, fragment: str
) -> None:
    """Bad requests get an OpenAI-shaped 400/404 naming the offending field."""
    response = await client.post("/v1/images/generations", json=payload, headers=auth_headers)
    assert response.status_code == status, response.text
    assert fragment in response.json()["error"]["message"], response.text


async def test_openai_no_op_fields_are_accepted(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """Fields real OpenAI clients send but we cannot honour must not break requests."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a cat", "user": "u-1", "quality": "high"},
        headers=auth_headers,
    )
    assert response.status_code == 200


async def test_unencodable_output_format_is_rejected(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """A format no build can encode is refused before any sampling work."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a cat", "output_format": "tiff"},
        headers=auth_headers,
    )
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "tiff" in message and "png" in message and "jpeg" in message and "webp" in message


@pytest.mark.parametrize("output_format", ["png", "jpeg", "webp"])
async def test_formats_the_engine_advertises_are_accepted(
    client: httpx.AsyncClient, auth_headers: dict[str, str], output_format: str
) -> None:
    """All three formats sd-server reports must pass through the gateway."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "a cat", "output_format": output_format},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text


async def test_health_reports_what_the_loaded_backend_can_do(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """After a load, operators can see the backend's model and advertised limits."""
    await client.post(
        "/v1/images/generations", json={"model": "test-model", "prompt": "report me"}, headers=auth_headers
    )
    health = (await client.get("/health/backends", headers=auth_headers)).json()
    entry = next(item for item in health["backends"] if item["id"] == "test-model")
    assert entry["state"] == "ready"
    assert entry["loaded_model"] == "stub"
    assert entry["output_formats"] == ["png", "jpeg", "webp"]
    assert entry["limits"]["max_batch_count"] == 4


async def test_url_response_format_serves_downloadable_image(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """response_format=url returns a URL that yields the image bytes."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "give me a url", "response_format": "url"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    url = response.json()["data"][0]["url"]
    assert url.startswith("http://testserver/v1/images/content/")

    fetched = await client.get(url)
    assert fetched.status_code == 200
    assert fetched.content[:4] == PNG_MAGIC
    assert fetched.headers["content-type"] == "image/png"


async def test_unknown_blob_id_is_404(client: httpx.AsyncClient) -> None:
    """A made-up or expired image URL is a clean 404."""
    response = await client.get("/v1/images/content/does-not-exist")
    assert response.status_code == 404
    assert "expired" in response.json()["error"]["message"] or "not found" in response.json()["error"]["message"]


# --------------------------------------------------------------------------
# Backend failures
# --------------------------------------------------------------------------


async def test_backend_failure_is_reported_as_bad_gateway(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """The backend's own error text must survive into our response for debuggability."""
    response = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "BOOM please"},
        headers=auth_headers,
    )
    assert response.status_code == 502
    assert "stub generation failure" in response.json()["error"]["message"]


async def test_backend_crash_is_survived_and_recovered(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """A backend that dies mid-request must be relaunched for the next one."""
    port = ports(config_path)["test-model"]
    failing = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "EXIT now"},
        headers=auth_headers,
    )
    assert failing.status_code in (502, 503)

    # The pool notices a dead process either on the next request or on the next
    # reaper pass, so poll instead of assuming the very next call succeeds.
    recovered = None
    for _ in range(20):
        recovered = await client.post(
            "/v1/images/generations",
            json={"model": "test-model", "prompt": "after the crash"},
            headers=auth_headers,
        )
        if recovered.status_code == 200:
            break
        await asyncio.sleep(0.5)

    assert recovered is not None and recovered.status_code == 200, recovered.text if recovered else "no response"
    assert (await stub_state(port))["last_request"]["prompt"].startswith("after the crash")


async def test_queue_is_bounded_and_rejects(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """With max_waiting_requests=0 a second concurrent request is refused, not queued."""
    busy = asyncio.create_task(
        client.post(
            "/v1/images/generations",
            json={"model": "test-model", "prompt": "SLEEP:2.0 hold the gpu"},
            headers=auth_headers,
        )
    )
    await asyncio.sleep(0.4)
    rejected = await client.post(
        "/v1/images/generations",
        json={"model": "test-model", "prompt": "should be refused"},
        headers=auth_headers,
    )
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "queue_full"
    assert (await busy).status_code == 200


# --------------------------------------------------------------------------
# Model swapping and idle unload
# --------------------------------------------------------------------------


async def test_switching_models_unloads_the_previous_backend(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """max_resident_backends=1: loading the second model must free the first."""
    mapped = ports(config_path)
    await client.post(
        "/v1/images/generations", json={"model": "test-model", "prompt": "first"}, headers=auth_headers
    )
    assert (await stub_state(mapped["test-model"]))["diffusion_model"]

    response = await client.get("/health/backends", headers=auth_headers)
    assert response.json()["resident_backends"] == 1

    await client.post(
        "/v1/images/generations", json={"model": "test-turbo", "prompt": "second"}, headers=auth_headers
    )
    with pytest.raises(httpx.HTTPError):
        await stub_state(mapped["test-model"])
    assert (await stub_state(mapped["test-turbo"]))["diffusion_model"]

    health = (await client.get("/health/backends", headers=auth_headers)).json()
    assert health["resident_backends"] == 1


async def test_idle_backend_is_unloaded(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """idle_timeout_seconds must actually free the memory when a model goes quiet."""
    port = ports(config_path)["test-model"]
    await client.post(
        "/v1/images/generations", json={"model": "test-model", "prompt": "then idle"}, headers=auth_headers
    )
    await wait_for_backend_gone(port, timeout=20.0)
    health = (await client.get("/health/backends", headers=auth_headers)).json()
    assert all(entry["state"] != "ready" for entry in health["backends"])


# --------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------


async def test_edit_requires_prompt(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """The multipart endpoint validates before touching a backend."""
    response = await client.post(
        "/v1/images/edits",
        data={"model": "test-model"},
        files={"image[]": ("in.png", b"\x89PNG fake", "image/png")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "prompt" in response.json()["error"]["message"]


async def test_edit_requires_an_image(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """An edit with no reference image is meaningless and is refused."""
    response = await client.post(
        "/v1/images/edits", data={"model": "test-model", "prompt": "make it blue"}, headers=auth_headers
    )
    assert response.status_code == 400
    assert "image" in response.json()["error"]["message"]


async def test_edit_forwards_image_and_receives_png(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """Reference images are forwarded as multipart `image[]` and an image comes back."""
    response = await client.post(
        "/v1/images/edits",
        data={"model": "test-model", "prompt": "turn it blue", "size": "1024x1024"},
        files={"image[]": ("in.png", b"\x89PNG fake bytes", "image/png")},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["data"][0]["b64_json"])[:4] == PNG_MAGIC
    state = await stub_state(ports(config_path)["test-model"])
    assert state["last_edit"]["image_parts"] == 1


def _native_block(prompt: str) -> dict[str, object]:
    """Extract the JSON the gateway embedded in the prompt it sent upstream."""
    import json as _json

    start = prompt.index("<sd_cpp_extra_args>") + len("<sd_cpp_extra_args>")
    end = prompt.index("</sd_cpp_extra_args>")
    return _json.loads(prompt[start:end])


async def test_edit_defaults_to_reference_conditioned_strength(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """Edits must not be img2img at the engine default, which copies the input.

    The upstream route assigns the first image to init_image as well as
    ref_images, so anything below strength 1.0 reproduces the input instead of
    editing it. This is the regression that made 'change the text' come back
    with the text unchanged.
    """
    await client.post(
        "/v1/images/edits",
        data={"model": "test-model", "prompt": "change 'A' to 'B'"},
        files={"image[]": ("in.png", b"\x89PNG fake bytes", "image/png")},
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-model"])
    sent = _native_block(state["last_edit"]["fields"]["prompt"])
    assert sent["strength"] == 1.0


async def test_edit_strength_form_field_overrides_the_default(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """A caller who does want img2img can ask for it explicitly."""
    await client.post(
        "/v1/images/edits",
        data={"model": "test-model", "prompt": "soften this", "strength": "0.6"},
        files={"image[]": ("in.png", b"\x89PNG fake bytes", "image/png")},
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-model"])
    assert _native_block(state["last_edit"]["fields"]["prompt"])["strength"] == 0.6


async def test_edit_sd_cpp_strength_is_overridden_by_the_dedicated_field(
    client: httpx.AsyncClient, auth_headers: dict[str, str], config_path
) -> None:
    """The specific form field wins over the generic native escape hatch."""
    await client.post(
        "/v1/images/edits",
        data={
            "model": "test-model",
            "prompt": "precedence check",
            "strength": "0.25",
            "sd_cpp": '{"strength": 0.9}',
        },
        files={"image[]": ("in.png", b"\x89PNG fake bytes", "image/png")},
        headers=auth_headers,
    )
    state = await stub_state(ports(config_path)["test-model"])
    assert _native_block(state["last_edit"]["fields"]["prompt"])["strength"] == 0.25


@pytest.mark.parametrize("strength", ["0", "-1", "1.5", "abc"])
async def test_edit_strength_out_of_range_is_rejected(
    client: httpx.AsyncClient, auth_headers: dict[str, str], strength: str
) -> None:
    """strength is a fraction of the sampling path, so only (0, 1] is meaningful."""
    response = await client.post(
        "/v1/images/edits",
        data={"model": "test-model", "prompt": "x", "strength": strength},
        files={"image[]": ("in.png", b"\x89PNG fake bytes", "image/png")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "strength" in response.json()["error"]["message"]


async def test_edit_is_refused_for_model_without_editing_enabled(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    """supports_editing=false means reference images would be ignored."""
    response = await client.post(
        "/v1/images/edits",
        data={"model": "test-turbo", "prompt": "edit me"},
        files={"image[]": ("in.png", b"\x89PNG fake bytes", "image/png")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert "supports_editing=false" in response.json()["error"]["message"]
