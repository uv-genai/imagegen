"""HTTP client for the ``sd-server`` OpenAI-compatible endpoints.

Only the response handling and error normalisation live here; request bodies are
produced by :mod:`imagegen.openai_payload`.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

import httpx

from imagegen.errors import BackendReturnedError, BackendUnavailableError


@dataclass(frozen=True)
class BackendImages:
    """Decoded result of one backend generation call."""

    images: list[bytes]
    output_format: str


def _extract_error_message(payload: object, fallback: str) -> str:
    """Pull a human readable message out of a backend error body.

    The server emits ``{"error": "text"}`` for route level failures, while the
    HTTP layer in front of it (or a proxy) may emit the OpenAI nested form
    ``{"error": {"message": "text"}}``. Both are accepted.
    """
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, str) and error:
            return error
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return fallback


class BackendClient:
    """Talks to a single running backend over loopback HTTP."""

    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
        )

    async def aclose(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()

    async def generate(self, payload: dict[str, object]) -> BackendImages:
        """Run a text-to-image request through the backend."""
        response = await self._post("/v1/images/generations", json=payload)
        return self._decode_images(response)

    async def edit(
        self,
        *,
        fields: dict[str, str],
        files: list[tuple[str, tuple[str, bytes, str]]],
    ) -> BackendImages:
        """Run an image-editing request; ``files`` uses httpx's multipart form."""
        response = await self._post("/v1/images/edits", data=fields, files=files)
        return self._decode_images(response)

    async def capabilities(self) -> dict[str, object]:
        """Fetch the backend's capabilities report.

        Returns the limits and advertised formats the engine publishes, which the
        pool compares against the configured profile so a config that asks for
        more than this build can deliver is reported instead of failing a request.
        """
        try:
            response = await self._client.get("/sdcpp/v1/capabilities")
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"could not read backend capabilities: {exc}") from exc
        if response.status_code != 200:
            raise BackendReturnedError(
                f"capabilities request returned HTTP {response.status_code}",
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise BackendReturnedError("capabilities response is not an object")
        return payload

    async def _post(self, path: str, **kwargs: object) -> httpx.Response:
        """POST and translate transport failures into a 503 rather than a stack trace."""
        try:
            return await self._client.post(path, **kwargs)  # type: ignore[arg-type]
        except httpx.TimeoutException as exc:
            raise BackendUnavailableError(
                f"backend did not answer within the configured timeout ({exc.request.url})"
            ) from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(f"backend request failed: {exc}") from exc

    def _decode_images(self, response: httpx.Response) -> BackendImages:
        """Validate the response and decode ``data[].b64_json`` into raw bytes.

        Raises:
            BackendReturnedError: the backend reported a failure, or answered with
                a body that does not match the documented schema.
        """
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if response.status_code != 200:
            message = _extract_error_message(payload, f"backend returned HTTP {response.status_code}")
            raise BackendReturnedError(f"backend error {response.status_code}: {message}")
        if not isinstance(payload, dict):
            raise BackendReturnedError("backend returned a non-object response body")

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            message = _extract_error_message(payload, "backend returned no images")
            raise BackendReturnedError(message)

        images: list[bytes] = []
        for entry in data:
            if not isinstance(entry, dict) or not isinstance(entry.get("b64_json"), str):
                raise BackendReturnedError("backend response entry is missing b64_json")
            try:
                images.append(base64.b64decode(entry["b64_json"], validate=True))
            except (binascii.Error, ValueError) as exc:
                raise BackendReturnedError("backend returned malformed base64 image data") from exc

        output_format = payload.get("output_format")
        if not isinstance(output_format, str) or not output_format:
            raise BackendReturnedError("backend response is missing output_format")
        return BackendImages(images=images, output_format=output_format)
