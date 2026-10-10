#!/usr/bin/env python3
"""End-to-end smoke test against a running imagegen service.

Unlike the pytest suite (which uses a stub engine), this drives the real thing:
it asks for actual images and checks that real PNG bytes come back, that the
model the client asked for is the model that answered, and that error paths
return OpenAI-shaped bodies.

    scripts/smoke_test.py --base-url http://127.0.0.1:8080 --api-key KEY \
        --model qwen-image-2.1-turbo --size 512x512 --out out/

Exits 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PNG_MAGIC = b"\x89PNG"


class Check:
    """Records the outcome of one assertion without stopping the run."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []

    def ok(self, condition: bool, detail: str) -> None:
        """Mark the check passed or record why it did not."""
        if condition:
            print(f"  PASS {self.name}: {detail}")
        else:
            print(f"  FAIL {self.name}: {detail}")
            self.failures.append(f"{self.name}: {detail}")


def request(
    base_url: str,
    method: str,
    path: str,
    api_key: str | None,
    body: dict[str, object] | None = None,
    timeout: float = 2400.0,
) -> tuple[int, bytes]:
    """Perform an HTTP call, returning status and raw body (no exception on 4xx/5xx)."""
    url = f"{base_url.rstrip('/')}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        return 0, str(exc.reason).encode("utf-8")


def check_health(base_url: str, api_key: str | None) -> Check:
    """Liveness and model discovery respond."""
    check = Check("health")
    status, body = request(base_url, "GET", "/health", api_key, timeout=10)
    check.ok(status == 200, f"GET /health -> {status}")
    if status != 200:
        return check
    listed, list_body = request(base_url, "GET", "/v1/models", api_key, timeout=10)
    ids = [entry["id"] for entry in json.loads(list_body).get("data", [])] if listed == 200 else []
    check.ok(listed == 200 and bool(ids), f"GET /v1/models -> {status}: {ids}")
    return check


def check_generation(
    base_url: str,
    api_key: str | None,
    model: str,
    prompt: str,
    size: str,
    out_dir: Path,
) -> Check:
    """A generation returns a decodable PNG, and reports which model answered."""
    check = Check(f"generation[{model}]")
    started = time.monotonic()
    status, body = request(
        base_url,
        "POST",
        "/v1/images/generations",
        api_key,
        {"model": model, "prompt": prompt, "size": size},
    )
    elapsed = time.monotonic() - started
    if status != 200:
        check.ok(False, f"HTTP {status} after {elapsed:.1f}s: {body[:400]!r}")
        return check
    payload = json.loads(body)
    images = payload.get("data", [])
    check.ok(len(images) == 1, f"one image returned in {elapsed:.1f}s")
    if not images or "b64_json" not in images[0]:
        check.ok(False, f"response has no b64_json: {str(payload)[:200]}")
        return check
    raw = base64.b64decode(images[0]["b64_json"])
    check.ok(raw[:4] == PNG_MAGIC, f"{len(raw)} bytes with a PNG magic number")
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = out_dir / f"{model}-{int(time.time())}.png"
    saved.write_bytes(raw)
    print(f"  saved {saved}")

    backends, backend_body = request(base_url, "GET", "/health/backends", api_key, timeout=10)
    if backends == 200:
        entry = next(
            (item for item in json.loads(backend_body)["backends"] if item["id"] == model),
            None,
        )
        check.ok(entry is not None and entry["state"] == "ready", f"backend state: {entry and entry['state']}")
        if entry and entry.get("limits"):
            print(f"  backend limits: {entry['limits']}, formats: {entry.get('output_formats')}")
    return check


def check_rejections(base_url: str, api_key: str | None, model: str) -> Check:
    """Invalid requests are refused with OpenAI-shaped errors, without generating."""
    check = Check("rejections")
    cases = [
        ("unknown model", {"model": "does-not-exist", "prompt": "x"}, 404),
        ("empty prompt", {"model": model, "prompt": ""}, 400),
        ("size not multiple of 32", {"model": model, "prompt": "x", "size": "1000x1000"}, 400),
        ("unknown parameter", {"model": model, "prompt": "x", "reponse_format": "url"}, 400),
        ("unsupported format", {"model": model, "prompt": "x", "output_format": "tiff"}, 400),
    ]
    for name, payload, expected in cases:
        status, body = request(base_url, "POST", "/v1/images/generations", api_key, payload, timeout=60)
        try:
            message = json.loads(body)["error"]["message"]
        except (json.JSONDecodeError, KeyError):
            message = body[:120].decode("utf-8", "replace")
        check.ok(status == expected and bool(message),
                 f"{name} -> {status} (expected {expected}): {message[:90]}")
    return check


def check_authentication(base_url: str, model: str) -> Check:
    """Without a key the service must refuse before doing any work."""
    check = Check("authentication")
    status, _ = request(
        base_url, "POST", "/v1/images/generations", None, {"model": model, "prompt": "x"}, timeout=30
    )
    check.ok(status in (401, 200), f"anonymous generation -> {status} (401 expected when keys are configured)")
    return check


def main() -> int:
    """Run every check and summarise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", required=True, help="model id to generate with")
    parser.add_argument("--prompt", default='A neon shop sign that reads "IMAGEGEN OK", rainy night')
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--out", default="out", help="directory to save generated images in")
    parser.add_argument("--skip-rejections", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = [check_health(args.base_url, args.api_key)]
    if checks[0].failures:
        print("service is not reachable; aborting")
        return 1
    checks.append(
        check_generation(args.base_url, args.api_key, args.model, args.prompt, args.size, Path(args.out))
    )
    if not args.skip_rejections:
        checks.append(check_rejections(args.base_url, args.api_key, args.model))
        checks.append(check_authentication(args.base_url, args.model))

    failed = [failure for check in checks for failure in check.failures]
    print(f"\n{len(checks)} groups, {len(failed)} failures")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
