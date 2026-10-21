"""Lifecycle of a single ``sd-server`` subprocess.

Each configured model maps to exactly one sd-server process, because the engine
loads one weight set per process. The gateway owns that process: it starts it,
waits for the health endpoint, and shuts it down again when it goes idle.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
from enum import Enum

import httpx

from imagegen.config import BackendSettings, ModelProfile
from imagegen.errors import BackendUnavailableError

logger = logging.getLogger("imagegen.backend")

#: Loopback address the backend binds to. It must never face the network: the
#: gateway is the only intended client and the backend has no authentication.
BACKEND_HOST = "127.0.0.1"

#: Keys present in a capabilities report only once the engine has finished
#: building it. Verified against sd-server master-982-0518b34, whose
#: ``GET /sdcpp/v1/capabilities`` answers with ``current_mode``, ``defaults``,
#: ``defaults_by_mode``, ``features``, ``features_by_mode``, ``generation_features``,
#: ``limits``, ``loras``, ``model``, ``output_formats``, ``output_formats_by_mode``,
#: ``preview_modes``, ``runtime``, ``samplers``, ``schedulers``, ``supported_modes``,
#: ``upscale`` and ``upscalers``. There is no ``models`` key. ``supported_modes``
#: lists generation modes (``img_gen``/``vid_gen``), NOT editing: flux3 edits are
#: accepted although its report says only ``img_gen``.
CAPABILITY_READY_KEYS = ("supported_modes", "model")


class BackendState(Enum):
    """Lifecycle states of a backend process."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"


def build_server_args(
    profile: ModelProfile,
    settings: BackendSettings,
    config_dir,
) -> list[str]:
    """Compose the sd-server command line for one model profile.

    Args:
        profile: Model whose weights the backend loads.
        settings: Shared runtime flags.
        config_dir: Directory relative paths in the config are resolved against.
    """
    resolve = settings.weight_path
    args = [
        str(settings.resolved_binary(config_dir)),
        "--diffusion-model",
        str(resolve(config_dir, profile.diffusion_model)),
        "--vae",
        str(resolve(config_dir, profile.vae)),
        "--llm",
        str(resolve(config_dir, profile.text_encoder)),
        "--listen-ip",
        BACKEND_HOST,
        "--listen-port",
        str(profile.port),
        "--log-level",
        settings.log_level,
    ]
    if profile.supports_editing and profile.vision_encoder is not None:
        args += ["--llm_vision", str(resolve(config_dir, profile.vision_encoder))]
    if profile.tokenizer is not None:
        args += ["--tokenizer", str(resolve(config_dir, profile.tokenizer))]
    if settings.offload_to_cpu:
        args.append("--offload-to-cpu")
    if settings.diffusion_flash_attention:
        args.append("--diffusion-fa")
    if settings.max_vram_gib is not None:
        args += ["--max-vram", str(settings.max_vram_gib)]
    # Seed the backend with this profile's sampling. The gateway sends these values
    # per request as well, but seeding them keeps the process correct on its own --
    # which matters for a distilled checkpoint, whose fixed sigma schedule produces
    # the wrong image if it is ever left unset.
    args += [
        "--cfg-scale",
        str(profile.sampling.cfg),
        "--steps",
        str(profile.sampling.steps),
        "--sampling-method",
        profile.sampling.sample_method,
    ]
    if profile.sampling.scheduler is not None:
        args += ["--scheduler", profile.sampling.scheduler]
    if profile.sampling.flow_shift is not None:
        args += ["--flow-shift", str(profile.sampling.flow_shift)]
    if profile.sampling.custom_sigmas:
        args += ["--sigmas", ",".join(str(sigma) for sigma in profile.sampling.custom_sigmas)]
    args += settings.extra_args
    return args


class BackendProcess:
    """One supervised sd-server instance.

    Not thread safe by design: a single asyncio loop owns the process, guarded by
    the lock in :class:`imagegen.backend_pool.BackendPool`.
    """

    def __init__(self, profile: ModelProfile, settings: BackendSettings, config_dir) -> None:
        self.profile = profile
        self.settings = settings
        self.config_dir = config_dir
        self.base_url = f"http://{BACKEND_HOST}:{profile.port}"
        self.state = BackendState.STOPPED
        self.last_used = 0.0
        #: Capabilities reported by the running backend, captured after readiness.
        self.capabilities: dict | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._log_file = None

    @property
    def pid(self) -> int | None:
        """OS pid of the running backend, or None when it is not running."""
        return self._process.pid if self._process is not None else None

    def is_alive(self) -> bool:
        """True when the child process still exists and has not exited."""
        return self._process is not None and self._process.returncode is None

    def build_args(self) -> list[str]:
        """Command line used to launch this backend."""
        return build_server_args(self.profile, self.settings, self.config_dir)

    async def start(self) -> None:
        """Launch sd-server and block until it answers the capabilities probe.

        Raises:
            BackendUnavailableError: launch failed, or readiness was not reached
                within ``start_timeout_seconds``.
        """
        if self.state is not BackendState.STOPPED or self.is_alive():
            return
        self.state = BackendState.STARTING
        args = self.build_args()
        logger.info("starting backend %s: %s", self.profile.id, " ".join(args))
        log_path = self._log_path()
        try:
            self._log_file = log_path.open("ab")
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdout=self._log_file,
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ},
            )
        except OSError as exc:
            self.state = BackendState.STOPPED
            self._close_log()
            raise BackendUnavailableError(f"could not launch sd-server for {self.profile.id}: {exc}") from exc

        try:
            await self._wait_until_ready()
        except BackendUnavailableError:
            await self.stop()
            raise
        self.state = BackendState.READY
        self.last_used = time.monotonic()
        logger.info("backend %s ready on %s (pid %s)", self.profile.id, self.base_url, self.pid)

    def _log_path(self):
        """Path of the file capturing this backend's stdout/stderr."""
        target = self.config_dir / "logs"
        target.mkdir(parents=True, exist_ok=True)
        return target / f"sd-server-{self.profile.id}.log"

    def _close_log(self) -> None:
        """Close the captured log handle, ignoring already-closed files."""
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None

    async def _wait_until_ready(self) -> None:
        """Poll the capabilities endpoint until the weights are loaded.

        A backend that exits during load is detected immediately instead of
        burning the whole timeout, because weight loading fails fast on missing
        files and out-of-memory conditions.
        """
        deadline = time.monotonic() + self.settings.start_timeout_seconds
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            while time.monotonic() < deadline:
                if self._process is not None and self._process.returncode is not None:
                    raise BackendUnavailableError(
                        f"sd-server for {self.profile.id} exited with code {self._process.returncode} "
                        f"while loading; see {self._log_path()}"
                    )
                try:
                    response = await client.get(f"{self.base_url}/sdcpp/v1/capabilities")
                    if response.status_code == 200:
                        payload = response.json()
                        if isinstance(payload, dict) and all(key in payload for key in CAPABILITY_READY_KEYS):
                            self.capabilities = payload
                            return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(1.0)
        raise BackendUnavailableError(
            f"sd-server for {self.profile.id} did not become ready within "
            f"{self.settings.start_timeout_seconds}s; see {self._log_path()}"
        )

    async def stop(self) -> None:
        """Terminate the backend, escalating from SIGINT to SIGTERM to SIGKILL.

        sd-server performs a clean shutdown on SIGINT, which is what a Ctrl-C in
        a terminal sends, so it is tried first.
        """
        process = self._process
        if process is None:
            self.state = BackendState.STOPPED
            self._close_log()
            return
        self.state = BackendState.STOPPING
        for sig, grace in ((signal.SIGINT, self.settings.stop_timeout_seconds),
                           (signal.SIGTERM, 5.0),
                           (signal.SIGKILL, 5.0)):
            if process.returncode is not None:
                break
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(sig)
                logger.info("stopping backend %s with %s", self.profile.id, sig.name)
            try:
                await asyncio.wait_for(process.wait(), timeout=grace)
                break
            except TimeoutError:
                logger.warning("backend %s did not exit after %s", self.profile.id, sig.name)
        self._process = None
        self._close_log()
        self.capabilities = None
        self.state = BackendState.STOPPED

    async def ensure_running(self) -> None:
        """Start the backend if needed, restarting it after an unexpected exit."""
        if self.state is BackendState.READY and self.is_alive():
            self.last_used = time.monotonic()
            return
        if self.is_alive() and self.state is not BackendState.READY:
            # Process exists but we never confirmed readiness: re-probe it.
            await self._wait_until_ready()
            self.state = BackendState.READY
            self.last_used = time.monotonic()
            return
        await self.stop()
        await self.start()
