#!/usr/bin/env python3
"""Stand-in for ``sd-server`` used by the test suite.

It implements the same HTTP surface the gateway depends on — the capabilities
probe, ``/v1/images/generations`` and ``/v1/images/edits`` — and accepts the same
``--listen-ip`` / ``--listen-port`` command line, so the gateway launches it
through exactly the same code path it uses for the real engine. Only the
inference is replaced by a fixed PNG.

Behaviour switches are driven by markers in the prompt so tests can exercise
latency, failures and model-swapping without inventing a new protocol:

``SLEEP:<seconds>``  sleep before answering, to hold the backend busy
``BOOM``             answer 500 with the server's ``{"error": "..."}`` shape
``EXIT``             exit the process mid-request, to test crash recovery
"""

from __future__ import annotations

import argparse
import base64
import email
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: A valid 1x1 opaque PNG, returned as the "generated" image.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

EXTRA_ARGS_RE = re.compile(r"<sd_cpp_extra_args>(.*?)</sd_cpp_extra_args>", re.DOTALL)

#: Set by the main thread so the handler can re-report the loaded model.
LOADED: dict[str, object] = {}


def capabilities_payload() -> dict[str, object]:
    """A capabilities report shaped like the real sd-server one."""
    return {
        "current_mode": "img_gen",
        "defaults": {"batch_count": 1, "height": 512, "width": 512},
        "features": {"lora": False, "hires": True},
        "limits": {
            "max_batch_count": 4,
            "max_height": 4096,
            "max_width": 4096,
            "max_queue_size": 16,
            "min_height": 64,
            "min_width": 64,
        },
        "loras": [],
        "model": {"name": "stub", "path": str(LOADED.get("diffusion_model", "stub")), "stem": "stub"},
        "output_formats": ["png", "jpeg", "webp"],
        "preview_modes": ["none"],
        "samplers": ["euler", "euler_a"],
        "schedulers": ["discrete", "normal"],
        "supported_modes": ["img_gen"],
        "upscale": False,
        "upscalers": [{"name": "None", "model": False, "image_upscale": False}],
    }


def _png_response(count: int) -> bytes:
    """Build an OpenAI-shaped generations body with ``count`` images."""
    created = int(time.time())
    data = [{"b64_json": base64.b64encode(TINY_PNG).decode("ascii")} for _ in range(count)]
    return json.dumps({"created": created, "output_format": "png", "data": data}).encode("utf-8")


class StubHandler(BaseHTTPRequestHandler):
    """Minimal re-implementation of the sd-server HTTP API."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - BaseHTTPRequestHandler signature
        """Keep test output quiet unless something goes wrong."""

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        """Write a response with an explicit Content-Length (HTTP/1.1 keep-alive)."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        """Serve the readiness probe and the upstream model id."""
        if self.path == "/sdcpp/v1/capabilities":
            # Mirrors the key set of the real sd-server payload; the gateway treats a
            # report as ready only when supported_modes and model are both present.
            self._send(200, json.dumps(capabilities_payload()).encode("utf-8"))
            return
        if self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": "sd-cpp-local", "object": "model"}]}).encode("utf-8"))
            return
        if self.path == "/_state":
            # Test-only introspection: what this instance loaded and what it last saw.
            self._send(200, json.dumps(LOADED).encode("utf-8"))
            return
        self._send(404, json.dumps({"error": "not found"}).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        """Handle the generation and edit endpoints."""
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/v1/images/generations":
            self._handle_json_generation(body)
            return
        if self.path == "/v1/images/edits":
            self._handle_edits(body)
            return
        self._send(404, json.dumps({"error": "not found"}).encode("utf-8"))

    def _handle_json_generation(self, body: bytes) -> None:
        """Emulate the upstream generations route, including the extra-args block."""
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "invalid json"}).encode("utf-8"))
            return
        prompt = payload.get("prompt", "")
        if "BOOM" in prompt:
            self._send(500, json.dumps({"error": "stub generation failure"}).encode("utf-8"))
            return
        if "EXIT" in prompt:
            # os._exit kills the whole process (sys.exit would only end this handler
            # thread), which is what a real engine crash looks like to the gateway.
            self.close_connection = True
            os._exit(9)
        sleep_match = re.search(r"SLEEP:([\d.]+)", prompt)
        if sleep_match:
            time.sleep(float(sleep_match.group(1)))

        # Record what the gateway sent before answering, so a test that asserts on
        # it after receiving the response never races the handler.
        match = EXTRA_ARGS_RE.search(prompt)
        LOADED["last_request"] = payload
        LOADED["last_extra_args"] = json.loads(match.group(1)) if match else None
        self._send(200, _png_response(int(payload.get("n", 1))))

    def _handle_edits(self, body: bytes) -> None:
        """Emulate the multipart edits route by counting file parts."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send(400, json.dumps({"error": "expected multipart"}).encode("utf-8"))
            return
        image_parts = body.count(b'name="image[]"') + body.count(b'name="image"')
        if image_parts == 0:
            self._send(400, json.dumps({"error": "no image uploaded"}).encode("utf-8"))
            return
        LOADED["last_edit"] = {
            "image_parts": image_parts,
            "has_mask": b'name="mask"' in body,
            "fields": _multipart_text_fields(self.headers.get("Content-Type", ""), body),
        }
        self._send(200, _png_response(1))


def _multipart_text_fields(content_type: str, body: bytes) -> dict[str, str]:
    """Decode the non-file parts of a multipart body as text fields."""
    message = email.message_from_bytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    fields: dict[str, str] = {}
    for part in message.walk():
        if part.get_filename() is not None:
            continue
        name = part.get_param("name", header="Content-Disposition")
        payload = part.get_payload(decode=True)
        if name and payload is not None:
            fields[name] = payload.decode("utf-8", "replace")
    return fields


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Accept the subset of sd-server flags the gateway passes; ignore the weights."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--listen-ip", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8199)
    parser.add_argument("--diffusion-model", default="stub")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--llm_vision", default=None)
    known, _unknown = parser.parse_known_args(argv)
    return known


def main() -> int:
    """Run the stub server until interrupted."""
    args = parse_args(sys.argv[1:])
    LOADED["diffusion_model"] = args.diffusion_model
    server = ThreadingHTTPServer((args.listen_ip, args.listen_port), StubHandler)
    print(f"stub sd-server on {args.listen_ip}:{args.listen_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
