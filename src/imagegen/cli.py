"""Command line interface: ``imagegen serve|models|plan|doctor``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path

import uvicorn

from imagegen import __version__
from imagegen.app import create_app
from imagegen.backend_process import build_server_args
from imagegen.config import ServiceConfig, load_config
from imagegen.errors import ServiceConfigurationError

#: Repository-relative location of a config file built by the setup instructions.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "service.toml"


def _apply_env_overrides(config: ServiceConfig) -> ServiceConfig:
    """Apply ``IMAGEGEN_*`` environment overrides so secrets stay out of the file."""
    keys = os.environ.get("IMAGEGEN_API_KEYS")
    if keys:
        config.service.api_keys = [key.strip() for key in keys.split(",") if key.strip()]
    port = os.environ.get("IMAGEGEN_PORT")
    if port:
        config.service.port = int(port)
    host = os.environ.get("IMAGEGEN_HOST")
    if host:
        config.service.host = host
    return config


def _load(path: str) -> tuple[ServiceConfig, Path]:
    """Load config, apply environment overrides, and exit on any configuration error."""
    config, config_dir = load_config(path)
    return _apply_env_overrides(config), config_dir


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the gateway in the foreground."""
    config, config_dir = _load(args.config)
    app = create_app(config, config_dir)
    if not config.service.api_keys:
        logging.getLogger("imagegen").warning(
            "no API keys configured; the service is unauthenticated. Bound it to loopback "
            "or set IMAGEGEN_API_KEYS."
        )
    uvicorn.run(app, host=config.service.host, port=config.service.port, log_level="info")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """List the configured models and their sampling defaults."""
    config, _ = _load(args.config)
    for profile in config.models:
        sigmas = "backend default" if not profile.sampling.custom_sigmas else f"{len(profile.sampling.custom_sigmas)} sigmas"
        print(f"{profile.id}")
        print(f"  description   {profile.description or '-'}")
        print(f"  sampling      {profile.sampling.steps} steps, cfg {profile.sampling.cfg}, "
              f"{profile.sampling.sample_method}, {sigmas}")
        print(f"  weights       {profile.diffusion_model}")
        print(f"  vae           {profile.vae}")
        print(f"  text encoder  {profile.text_encoder}")
        print(f"  vision tower  {profile.vision_encoder or '- (editing disabled)'}")
        print(f"  backend port  {profile.port}")
        print(f"  size          default {profile.default_size}, max {profile.max_pixels} px, "
              f"multiple of {profile.dimension_multiple}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    """Print the exact sd-server command line for one model, for debugging launches."""
    config, config_dir = _load(args.config)
    profile = config.profile(args.model)
    if profile is None:
        known = ", ".join(candidate.id for candidate in config.models)
        print(f"unknown model {args.model!r}; available: {known}", file=sys.stderr)
        return 2
    print(" ".join(build_server_args(profile, config.backends, config_dir)))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check the binary, the weight files and the requested devices before serving."""
    config, config_dir = _load(args.config)
    problems: list[str] = config.setup_problems(config_dir)

    binary = config.backends.resolved_binary(config_dir)
    if binary.is_file() and shutil.which(binary.name) is None:
        print(f"note: {binary.name} is not on PATH; it is launched by absolute path, which is fine")
    for profile in config.models:
        if not 1 <= profile.port <= 65535:
            problems.append(f"model {profile.id}: invalid port {profile.port}")

    if binary.is_file():
        devices = asyncio.run(_list_devices(binary))
        print("compute devices reported by sd-server:")
        print(devices.rstrip() or "  <none>")

    if problems:
        print("\nproblems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nconfiguration OK: {len(config.models)} model(s), lifecycle={config.backends.lifecycle}")
    return 0


async def _list_devices(binary: Path) -> str:
    """Capture ``sd-server --list-devices`` output, or the error if it fails."""
    process = await asyncio.create_subprocess_exec(
        str(binary),
        "--list-devices",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await process.communicate()
    return stdout.decode("utf-8", "replace")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``imagegen`` entry point."""
    parser = argparse.ArgumentParser(prog="imagegen", description="OpenAI-compatible image generation service")
    parser.add_argument("--version", action="version", version=f"imagegen {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default = str(DEFAULT_CONFIG)

    serve = subparsers.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--config", default=default, help=f"path to service config (default: {default})")
    serve.set_defaults(handler=_cmd_serve)

    models = subparsers.add_parser("models", help="list configured models")
    models.add_argument("--config", default=default)
    models.set_defaults(handler=_cmd_models)

    plan = subparsers.add_parser("plan", help="print the sd-server command line for a model")
    plan.add_argument("--config", default=default)
    plan.add_argument("--model", required=True)
    plan.set_defaults(handler=_cmd_plan)

    doctor = subparsers.add_parser("doctor", help="verify binary, weights and devices")
    doctor.add_argument("--config", default=default)
    doctor.set_defaults(handler=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point returning a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    try:
        return handler(args)
    except ServiceConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
