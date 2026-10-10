"""Ownership, queueing and eviction of the sd-server backends.

One machine has one GPU, and the Qwen-Image-2.1 weight sets do not fit in it at
the same time: each backend needs roughly 9 GB of resident weights on top of its
activations. This module therefore treats backends as a scarce, swappable
resource — at most ``max_resident_backends`` of them are loaded, a request that
needs a cold model first waits for its turn and then for the weights to load.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from imagegen.backend_client import BackendClient
from imagegen.backend_process import BackendProcess, BackendState
from imagegen.config import BackendSettings, ModelProfile, ServiceSettings
from imagegen.openai_payload import GENERATION_OUTPUT_FORMATS
from imagegen.errors import (
    BackendUnavailableError,
    ModelNotFoundError,
    QueueFullError,
)

logger = logging.getLogger("imagegen.pool")

#: How often the reaper looks for idle backends to unload and dead ones to restart.
REAP_INTERVAL_SECONDS = 5.0

#: Poll interval while waiting for another backend to release the GPU.
EVICTION_POLL_SECONDS = 0.5


@dataclass
class _Slot:
    """Per-model runtime state: the process, its lock and its queue counter."""

    process: BackendProcess
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    waiters: int = 0
    client: BackendClient | None = None
    client_pid: int | None = None

    @property
    def in_use(self) -> bool:
        """True while a generation is running or queued on this backend."""
        return self.lock.locked() or self.waiters > 0


class BackendPool:
    """Registry of supervised backends with queueing and LRU eviction."""

    def __init__(
        self,
        profiles: list[ModelProfile],
        backend_settings: BackendSettings,
        service_settings: ServiceSettings,
        config_dir,
    ) -> None:
        self._settings = backend_settings
        self._config_dir = config_dir
        self._timeout = service_settings.request_timeout_seconds
        self._max_waiting = service_settings.max_waiting_requests
        self._slots: dict[str, _Slot] = {
            profile.id: _Slot(BackendProcess(profile, backend_settings, config_dir)) for profile in profiles
        }
        self._state_lock = asyncio.Lock()
        self._reaper: asyncio.Task[None] | None = None
        # Profiles whose advertised capabilities have already been compared
        # against the configuration, so the warning is logged once per load.
        self._checked_profiles: set[str] = set()

    @property
    def model_ids(self) -> list[str]:
        """Registered model identifiers, in configuration order."""
        return list(self._slots)

    def process(self, model_id: str) -> BackendProcess:
        """The supervised process object for a model (for status reporting)."""
        return self._slot_for(model_id).process

    def _slot_for(self, model_id: str) -> _Slot:
        """Look up a slot, raising ModelNotFoundError for unknown ids."""
        slot = self._slots.get(model_id)
        if slot is None:
            known = ", ".join(self._slots)
            raise ModelNotFoundError(f"unknown model {model_id!r}; available models: {known}")
        return slot

    def start_background_tasks(self) -> None:
        """Start eager backends and launch the idle/dead reaper loop."""
        if self._reaper is not None:
            return
        self._reaper = asyncio.create_task(self._reap_loop(), name="imagegen-backend-reaper")

    async def shutdown(self) -> None:
        """Stop the reaper and every running backend."""
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        for slot in self._slots.values():
            await slot.process.stop()
            await self._drop_client(slot)
        self._checked_profiles.clear()

    @contextlib.asynccontextmanager
    async def acquire(self, model_id: str) -> AsyncIterator[BackendClient]:
        """Reserve the backend for ``model_id`` and yield a ready client.

        Exactly one request owns a backend at a time: the GPU cannot serve two
        generations concurrently, and running them anyway would only thrash
        memory and make both slower.

        Raises:
            ModelNotFoundError: the id is not registered.
            QueueFullError: too many requests already wait for this model.
            BackendUnavailableError: weights could not be loaded.
        """
        slot = self._slot_for(model_id)
        if slot.in_use and slot.waiters >= self._max_waiting:
            raise QueueFullError(
                f"model {model_id!r} is busy and {slot.waiters} requests are already queued "
                f"(limit {self._max_waiting})"
            )
        slot.waiters += 1
        try:
            await slot.lock.acquire()
        finally:
            slot.waiters -= 1
        try:
            await self._make_ready(slot)
            client = slot.client
            if client is None:  # _make_ready guarantees a client; guarded for type clarity
                raise BackendUnavailableError(f"backend for {model_id!r} has no client after readiness")
            yield client
        finally:
            slot.lock.release()

    async def _make_ready(self, slot: _Slot) -> None:
        """Load the model's weights, evicting colder backends when they are full."""
        async with self._state_lock:
            if not slot.process.is_alive():
                await self._evict_for_capacity(slot)
            try:
                await slot.process.ensure_running()
            finally:
                await self._refresh_client(slot)
            if slot.client is not None:
                await self._verify_against_capabilities(slot)

    async def _verify_against_capabilities(self, slot: _Slot) -> None:
        """Compare a loaded backend's advertised limits with its configured profile.

        The engine publishes what it can actually do (formats, size and batch
        limits) once its weights are loaded. A profile that asks for more would
        otherwise fail later, inside a request the client is already waiting on,
        so the mismatch is reported here as a configuration defect.
        """
        profile = slot.process.profile
        if profile.id in self._checked_profiles:
            return
        self._checked_profiles.add(profile.id)
        try:
            capabilities = slot.process.capabilities or await slot.client.capabilities()  # type: ignore[union-attr]
        except Exception as exc:  # a failed cross-check must not fail the request
            logger.warning("could not read capabilities for %s: %s", profile.id, exc)
            return
        slot.process.capabilities = capabilities
        for problem in capability_mismatches(profile, capabilities):
            logger.warning("backend %s: %s", profile.id, problem)

    async def _evict_for_capacity(self, slot: _Slot) -> None:
        """Free a resident slot so this model can be loaded.

        Waits rather than failing when every resident backend is mid-generation;
        unloading one of those would discard work the client is still paying for.
        """
        limit = self._settings.max_resident_backends
        deadline = time.monotonic() + self._settings.start_timeout_seconds
        while self._resident_count() >= limit:
            victim = self._pick_eviction_victim(exclude=slot)
            if victim is None:
                if time.monotonic() >= deadline:
                    raise BackendUnavailableError(
                        f"cannot load {slot.process.profile.id!r}: {self._resident_count()} backends are "
                        f"resident and busy, and at most {limit} fit on this machine"
                    )
                await asyncio.sleep(EVICTION_POLL_SECONDS)
                continue
            logger.info(
                "unloading %s to make room for %s", victim.process.profile.id, slot.process.profile.id
            )
            await victim.process.stop()
            await self._drop_client(victim)

    def _pick_eviction_victim(self, exclude: _Slot) -> _Slot | None:
        """The least recently used resident, non-busy backend other than ``exclude``."""
        candidates = [
            other
            for other in self._slots.values()
            if other is not exclude
            and other.process.state is BackendState.READY
            and other.process.is_alive()
            and not other.in_use
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.process.last_used)

    def _resident_count(self) -> int:
        """How many backends currently hold loaded weights."""
        return sum(
            1
            for slot in self._slots.values()
            if slot.process.state is BackendState.READY and slot.process.is_alive()
        )

    async def _refresh_client(self, slot: _Slot) -> None:
        """Point the cached client at the current process, recreating it on restart.

        A restarted backend listens on the same port but has a fresh connection
        table, so the pool must not reuse sockets to the dead one.
        """
        pid = slot.process.pid
        if slot.client is not None and slot.client_pid == pid:
            return
        await self._drop_client(slot)
        if pid is not None and slot.process.state is BackendState.READY:
            slot.client = BackendClient(slot.process.base_url, self._timeout)
            slot.client_pid = pid

    async def _drop_client(self, slot: _Slot) -> None:
        """Close and forget the cached client for a slot."""
        if slot.client is not None:
            with contextlib.suppress(Exception):
                await slot.client.aclose()
        slot.client = None
        slot.client_pid = None

    async def _reap_loop(self) -> None:
        """Periodically unload idle backends and revive dead eager ones."""
        while True:
            try:
                await asyncio.sleep(REAP_INTERVAL_SECONDS)
                await self._reap_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # a failed reap must never kill the service
                logger.exception("backend reaper iteration failed")

    async def _reap_once(self) -> None:
        """One reaper pass: idle unload for on-demand models, restart for eager ones."""
        restart: list[_Slot] = []
        async with self._state_lock:
            now = time.monotonic()
            for slot in self._slots.values():
                if slot.lock.locked():
                    continue
                if slot.process.state is BackendState.READY and not slot.process.is_alive():
                    if self._settings.lifecycle == "eager":
                        restart.append(slot)
                    else:
                        await self._drop_client(slot)
                        slot.process.state = BackendState.STOPPED
                    continue
                if (
                    self._settings.lifecycle == "on_demand"
                    and slot.process.state is BackendState.READY
                    and now - slot.process.last_used > self._settings.idle_timeout_seconds
                ):
                    logger.info("unloading idle backend %s", slot.process.profile.id)
                    await slot.process.stop()
                    await self._drop_client(slot)
        for slot in restart:
            logger.warning("restarting crashed eager backend %s", slot.process.profile.id)
            await slot.process.ensure_running()
            await self._refresh_client(slot)

    def status(self) -> list[dict[str, object]]:
        """Snapshot of every backend, used by the health endpoints."""
        entries: list[dict[str, object]] = []
        for slot in self._slots.values():
            process = slot.process
            capabilities = process.capabilities or {}
            limits = capabilities.get("limits") if isinstance(capabilities.get("limits"), dict) else {}
            model_info = capabilities.get("model") if isinstance(capabilities.get("model"), dict) else {}
            entries.append(
                {
                    "id": process.profile.id,
                    "state": process.state.value,
                    "pid": process.pid,
                    "port": process.profile.port,
                    "queued": slot.waiters,
                    "busy": slot.lock.locked(),
                    "idle_seconds": round(time.monotonic() - process.last_used, 1)
                    if process.last_used
                    else None,
                    "loaded_model": model_info.get("stem") or model_info.get("name"),
                    "output_formats": capabilities.get("output_formats"),
                    "samplers": capabilities.get("samplers"),
                    "limits": limits or None,
                }
            )
        return entries


def capability_mismatches(profile: ModelProfile, capabilities: dict[str, object]) -> list[str]:
    """Configuration defects visible only once a backend reports its capabilities.

    Args:
        profile: The configured model profile.
        capabilities: The decoded ``/sdcpp/v1/capabilities`` payload.
    """
    problems: list[str] = []
    formats = capabilities.get("output_formats")
    if isinstance(formats, list) and formats:
        unsupported = [name for name in GENERATION_OUTPUT_FORMATS if name not in formats]
        if unsupported:
            problems.append(
                f"service accepts {', '.join(unsupported)} but this build advertises only "
                f"{', '.join(str(name) for name in formats)}"
            )
    limits = capabilities.get("limits")
    if isinstance(limits, dict):
        max_batch = limits.get("max_batch_count")
        if isinstance(max_batch, int) and max_batch > 0 and profile.max_images_per_request > max_batch:
            problems.append(
                f"max_images_per_request={profile.max_images_per_request} exceeds the backend's "
                f"max_batch_count={max_batch}"
            )
        width, height = profile.parse_size(profile.default_size)
        for dimension, value in (("width", width), ("height", height)):
            limit = limits.get(f"max_{dimension}")
            if isinstance(limit, int) and limit > 0 and value > limit:
                problems.append(f"default_size {profile.default_size} exceeds max_{dimension}={limit}")
    return problems
