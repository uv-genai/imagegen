"""Service configuration: load, type-check and validate a TOML config file.

The config is the single place where the model registry and the backend runtime
flags live, so that deploying a different quantization or a third model never
requires a code change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from imagegen.errors import ServiceConfigurationError

#: Directories a relative weight path may be resolved against.
_WEIGHT_SUFFIXES = (".gguf", ".safetensors")


class SamplingDefaults(BaseModel):
    """Sampling parameters applied to every request for one model.

    These are the values that make a distilled checkpoint behave correctly: the
    Turbo checkpoint is only correct at 8 steps, CFG 1 and its own sigma list.
    A request may still override them through the ``sd_cpp`` extension field.
    """

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(gt=0, description="Number of denoising steps.")
    cfg: float = Field(ge=0.0, description="Text classifier-free guidance scale.")
    sample_method: str = Field(description="Sampler name accepted by stable-diffusion.cpp.")
    scheduler: str | None = Field(default=None, description="Scheduler name, omitted when unset.")
    flow_shift: float | None = Field(default=None, description="Flow-matching schedule shift.")
    custom_sigmas: list[float] = Field(
        default_factory=list,
        description="Explicit sigma schedule; empty means let the backend derive one.",
    )
    negative_prompt: str = Field(default="", description="Default negative prompt when the client sends none.")

    @field_validator("custom_sigmas")
    @classmethod
    def _sigmas_are_in_range(cls, value: list[float]) -> list[float]:
        """Reject sigma schedules that fall outside the [0, 1] noise range."""
        for sigma in value:
            if not 0.0 <= sigma <= 1.0:
                raise ValueError(f"custom_sigmas must lie within [0, 1], got {sigma}")
        return value


class ModelProfile(BaseModel):
    """One deployable model: its weights, its backend port and its policies."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$", description="Public model identifier.")
    description: str = ""
    diffusion_model: str = Field(description="Denoiser file name inside models_dir.")
    vae: str = Field(description="VAE file name inside models_dir.")
    text_encoder: str = Field(description="Text encoder (LLM) file name inside models_dir.")
    vision_encoder: str | None = Field(
        default=None,
        description="mmproj vision tower; only VLM-based edit presets (e.g. qwen) need it.",
    )
    tokenizer: str | None = Field(
        default=None,
        description="External tokenizer.json for vocabularies not embedded in sd.cpp (MOSS/Qwen3.5).",
    )
    port: int = Field(gt=0, lt=65536, description="Loopback port the sd-server instance binds.")
    supports_editing: bool = False
    default_size: str = Field(pattern=r"^\d+x\d+$", description="Generation size when the client omits one.")
    dimension_multiple: int = Field(default=32, gt=0, description="Width and height must be a multiple of this.")
    max_pixels: int = Field(gt=0, description="Upper bound on width * height for this model.")
    max_images_per_request: int = Field(default=1, gt=0, le=8)
    sampling: SamplingDefaults

    @model_validator(mode="after")
    def _default_size_fits(self) -> Self:
        """The default size must itself satisfy the geometry policy."""
        width, height = self.parse_size(self.default_size)
        self.validate_size(width, height)
        return self

    def parse_size(self, size: str) -> tuple[int, int]:
        """Split a ``WIDTHxHEIGHT`` string, raising ValueError when malformed."""
        raw_width, _, raw_height = size.partition("x")
        if not raw_width.isdigit() or not raw_height.isdigit():
            raise ValueError(f"size must be WIDTHxHEIGHT with positive integers, got {size!r}")
        return int(raw_width), int(raw_height)

    def validate_size(self, width: int, height: int) -> None:
        """Raise ValueError when this geometry is not valid for the model."""
        if width <= 0 or height <= 0:
            raise ValueError(f"size must be positive, got {width}x{height}")
        for name, value in (("width", width), ("height", height)):
            if value % self.dimension_multiple != 0:
                raise ValueError(
                    f"{name} {value} must be a multiple of {self.dimension_multiple} for model {self.id}"
                )
        if width * height > self.max_pixels:
            raise ValueError(
                f"{width}x{height} is {width * height} pixels, above the {self.max_pixels} pixel "
                f"limit of model {self.id}"
            )


class ServiceSettings(BaseModel):
    """HTTP-facing settings of the gateway itself."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8080, gt=0, lt=65536)
    api_keys: list[str] = Field(default_factory=list, description="Accepted bearer tokens; empty disables auth.")
    public_base_url: str | None = Field(
        default=None,
        description="Externally reachable base URL; required for response_format=url.",
    )
    request_timeout_seconds: float = Field(default=1800.0, gt=0)
    max_waiting_requests: int = Field(default=8, ge=0, description="Requests allowed to queue behind a generation.")
    blob_ttl_seconds: int = Field(default=3600, gt=0, description="Lifetime of images served through response URLs.")
    blob_capacity: int = Field(default=64, gt=0, description="Maximum number of stored response images.")


class BackendSettings(BaseModel):
    """Runtime flags shared by every sd-server instance."""

    model_config = ConfigDict(extra="forbid")

    sd_server_binary: str = Field(description="Path to the sd-server executable.")
    models_dir: str = Field(description="Directory holding the weight files.")
    lifecycle: Literal["eager", "on_demand"] = "on_demand"
    max_resident_backends: int = Field(
        default=1,
        ge=1,
        description=(
            "How many weight sets may be loaded at once. Each Qwen-Image-2.1 backend needs "
            "several gigabytes of weights plus activations, so this stays at 1 on a single "
            "consumer GPU and is raised only on machines that genuinely fit more."
        ),
    )
    idle_timeout_seconds: float = Field(default=300.0, gt=0)
    start_timeout_seconds: float = Field(default=600.0, gt=0, description="Load is slow: weights are gigabytes.")
    stop_timeout_seconds: float = Field(default=30.0, gt=0)
    offload_to_cpu: bool = True
    diffusion_flash_attention: bool = True
    max_vram_gib: float | None = Field(
        default=None,
        description=(
            "Per-device budget in GiB handed to sd-server's --max-vram. Leave unset to let "
            "sd-server's auto-fit decide; set it when other processes share the GPU."
        ),
    )
    log_level: Literal["debug", "verbose", "info", "warn", "error"] = "info"
    extra_args: list[str] = Field(default_factory=list, description="Appended verbatim to every sd-server call.")

    def resolved_binary(self, config_dir: Path) -> Path:
        """Absolute path of the sd-server executable, relative to the config file."""
        return self._resolve(config_dir, self.sd_server_binary)

    def weight_path(self, config_dir: Path, name: str) -> Path:
        """Absolute path of a weight file referenced by bare name."""
        return self._resolve(config_dir, self.models_dir) / name

    @staticmethod
    def _resolve(config_dir: Path, value: str) -> Path:
        """Resolve a possibly relative path against the config file directory."""
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else (config_dir / candidate).resolve()


class ServiceConfig(BaseModel):
    """Root configuration object."""

    model_config = ConfigDict(extra="forbid")

    service: ServiceSettings = ServiceSettings()
    backends: BackendSettings
    models: list[ModelProfile]

    @model_validator(mode="after")
    def _models_are_consistent(self) -> Self:
        """Model ids and backend ports must be unique across the registry."""
        seen_ids: set[str] = set()
        seen_ports: set[int] = set()
        for profile in self.models:
            if profile.id in seen_ids:
                raise ValueError(f"duplicate model id {profile.id!r}")
            if profile.port in seen_ports:
                raise ValueError(f"model {profile.id!r} reuses backend port {profile.port}")
            seen_ids.add(profile.id)
            seen_ports.add(profile.port)
        if not self.models:
            raise ValueError("at least one [[model]] entry is required")
        return self

    def profile(self, model_id: str) -> ModelProfile | None:
        """Look up a model profile by public identifier."""
        for profile in self.models:
            if profile.id == model_id:
                return profile
        return None

    def setup_problems(self, config_dir: Path) -> list[str]:
        """Every reason this configuration could not serve, collected in one pass.

        Used by `imagegen doctor`. Reporting one problem at a time would make a
        half-provisioned machine require a round trip per missing artefact, so
        this accumulates instead of short-circuiting.
        """
        problems: list[str] = []
        binary = self.backends.resolved_binary(config_dir)
        if not binary.is_file():
            problems.append(f"sd-server binary not found: {binary}")
        for profile in self.models:
            names = [profile.diffusion_model, profile.vae, profile.text_encoder]
            if profile.supports_editing and profile.vision_encoder is not None:
                names.append(profile.vision_encoder)
            for name in names:
                if not name.endswith(_WEIGHT_SUFFIXES):
                    problems.append(f"model {profile.id!r}: {name!r} is not a gguf/safetensors file")
                    continue
                path = self.backends.weight_path(config_dir, name)
                if not path.is_file():
                    problems.append(f"model {profile.id!r}: weight file not found: {path}")
            if profile.tokenizer is not None and not self.backends.weight_path(config_dir, profile.tokenizer).is_file():
                problems.append(f"model {profile.id!r}: tokenizer file not found: {profile.tokenizer}")
        return problems

    def verify_weights(self, config_dir: Path) -> None:
        """Fail fast when a referenced weight file or the binary is missing.

        Startup raises on the first problem; `setup_problems` is the diagnostic
        variant that lists all of them.
        """
        problems = self.setup_problems(config_dir)
        if problems:
            raise ServiceConfigurationError("\n".join(problems))


def load_config(path: str | Path) -> tuple[ServiceConfig, Path]:
    """Parse and validate a TOML config, returning it with its directory.

    The directory is returned because weight and binary paths in the file are
    interpreted relative to it, which keeps a config file relocatable.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ServiceConfigurationError(f"config file not found: {config_path}")
    with config_path.open("rb") as handle:
        try:
            raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ServiceConfigurationError(f"invalid TOML in {config_path}: {exc}") from exc
    try:
        config = ServiceConfig.model_validate(raw)
    except ValueError as exc:  # pydantic raises subclasses of ValueError
        raise ServiceConfigurationError(f"invalid configuration: {exc}") from exc
    return config, config_path.parent
