# imagegen

OpenAI-compatible image generation service backed by
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) and the
**Qwen-Image-2.1** model family.

Point any OpenAI client at it and generate:

```bash
curl http://127.0.0.1:8080/v1/images/generations \
  -H "Authorization: Bearer $IMAGEGEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen-image-2.1-turbo", "prompt": "A neon shop sign that reads \"IMAGEGEN OK\", rainy night", "size": "1024x1024"}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="...")
result = client.images.generate(
    model="qwen-image-2.1",
    prompt="A vertically oriented study poster about chemical reactions",
    size="1024x1024",
)
```

## What it does

`sd-server` (from stable-diffusion.cpp) already speaks an OpenAI image API, but one
process serves one weight set and reports one hard-coded model id. `imagegen` runs
one `sd-server` per configured model and adds the parts a service needs:

- **Model routing** on the OpenAI `model` field.
- **Per-model sampling contracts.** `qwen-image-2.1-turbo` is a distilled
  checkpoint that is only correct at 8 steps, CFG 1 and its own sigma schedule;
  the gateway guarantees that schedule reaches the engine instead of leaving it to
  the caller.
- **Model swapping.** Each backend holds ~9.7 GB of weights. On a machine that
  cannot hold two, cold models are loaded on demand and the least recently used
  backend is unloaded (`max_resident_backends`).
- **Authentication, validation, queueing** and OpenAI-shaped errors.

## Install

### Prerequisites

| Needed for | Requirement | Check / install |
| --- | --- | --- |
| everything | Linux, `git`, [`uv`](https://docs.astral.sh/uv/) | `git --version`; `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| the service itself | nothing else — `uv` fetches a Python ≥ 3.12 interpreter and all dependencies | `uv --version` |
| `build_engine.sh cuda` | CUDA Toolkit (`nvcc`) + vendor driver | `nvcc --version` |
| `build_engine.sh vulkan` | Vulkan SDK (`glslc`) | `glslc --version` |
| `build_engine.sh` (any) | `cmake` ≥ 3.21, `gcc`/`g++` | `cmake --version` |

No Hugging Face account or token is required for the shipped weights, and no GPU
is needed to run the tests.

### Steps

```bash
git clone https://github.com/uv-genai/imagegen.git
cd imagegen

uv sync                        # 1. the service: deps + venv          (seconds)
scripts/build_engine.sh cuda   # 2. the inference engine              (~10 min)
scripts/fetch_models.sh        # 3. the weights, ~31.7 GB, resumable  (network-bound)
uv run imagegen doctor         # 4. verify binary, weights and devices
```

Order is not fixed, but `doctor` is only meaningful after steps 2 and 3. Step 2
clones [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) at
one pinned commit (~400 MB, blobless) into `third_party/`; step 3 writes to
`models/`. Both directories are gitignored and are never committed.

Peek before committing to the download:

```bash
scripts/fetch_models.sh --dry-run     # lists the files and ~31.7 GB total, writes nothing
```

### Compute backend

Chosen at build time, one directory per backend so engines coexist instead of
replacing each other:

| Command | Build directory | When |
| --- | --- | --- |
| `scripts/build_engine.sh cuda` | `third_party/stable-diffusion.cpp/build` | default; keeps this historic path |
| `scripts/build_engine.sh vulkan` | `third_party/stable-diffusion.cpp/build-vulkan` | no CUDA installed — AMD, Intel, other drivers |
| `scripts/build_engine.sh cpu` | `third_party/stable-diffusion.cpp/build-cpu` | reference only; slow |

Every build finishes by printing `sd-server --list-devices` and the exact
`sd_server_binary` line to paste into `config/service.toml`. Switching engines is
that one line — nothing rebuilds. The indices it prints (`CUDA0`, `Vulkan0`,
`Vulkan1`) are what `--backend` and `--max-vram` refer to, and they follow loader
enumeration order, not speed: on this laptop `Vulkan0` is the Intel iGPU and the
RTX 2070 is `Vulkan1`, so name the index you mean rather than a bare `vulkan`.

Vulkan was built and exercised at the pinned commit here: it configures, links
`libvulkan.so.1`, `imagegen doctor` lists both Vulkan devices, and a 512×512 Turbo
generation through the gateway completed in **47 s against ~16 s on CUDA**. Vulkan
is the route for machines without CUDA, not a speed improvement.

Building one backend into a directory configured for another is **refused**, not
silently reconfigured — that would replace a working engine and force a full
rebuild — and the error names the directory to remove if you really want to.

### Verify the install

```bash
uv run imagegen doctor                                  # exit 0 = ready to serve
uv run imagegen models                                  # registry + sampling defaults
uv run imagegen plan --model qwen-image-2.1-turbo       # exact sd-server command line
third_party/stable-diffusion.cpp/build/bin/sd-server --version   # must report the pinned commit
uv run pytest                                           # 100 tests; the 2 needing weights self-skip
```

`sd-server --version` should print `stable-diffusion.cpp version master-982-0518b34,
commit 0518b34`. That is the commit this service was built and measured against;
`PINNED_COMMIT` in `scripts/build_engine.sh` is the one place to change it. If the
version differs, the engine was built from something else and the results below do
not apply.

`doctor` prints the compute devices the engine sees and lists every missing
weight by path, so a half-finished setup reports exactly what is absent:

```
compute devices reported by sd-server:
CUDA0  NVIDIA GeForce RTX 2070 with Max-Q Design
CPU    Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz

configuration OK: 5 model(s), lifecycle=on_demand
```

### If something is missing

`doctor` exits non-zero and lists **every** missing artefact by absolute path, not
just the first one, so one run tells you what is left:

```
problems:
  - sd-server binary not found: /path/to/imagegen/third_party/.../build/bin/sd-server
  - model 'qwen-image-2.1': weight file not found: /path/to/imagegen/models/qwen_image_2.1-Q4_K.gguf
  ...
```

The first line means step 2 did not finish (`scripts/build_engine.sh cuda`); the rest
mean step 3 did not (`scripts/fetch_models.sh`). Both scripts are resumable.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `hf: command not found` | obsolete — weights are fetched through `uvx --from huggingface-hub`; if you see this, `git pull` |
| `nvcc: command not found` | CUDA Toolkit is not installed; install it, or build for another backend: `scripts/build_engine.sh vulkan` / `cpu` |
| `pinned commit ... is not reachable` | upstream history was rewritten. Verify a new commit builds and passes `scripts/smoke_test.py`, then update `PINNED_COMMIT` |
| backend fails to start with an allocation error | the GPU is full — stop other GPU processes, or lower `max_pixels` in `config/service.toml` |
| first request is much slower than the rest | expected: models load on demand. Turbo loads in ~2-3 s, and switching models unloads the other one |
| weights refused / 401 from Hugging Face | not expected for the shipped files, which need no account. If a future repo is gated, run `uvx --from huggingface-hub hf auth login` |

### Hardware budget

| Resource | Amount |
| --- | --- |
| Disk | ~32 GB weights + ~1.5 GB engine build tree |
| RAM | ~10 GB per loaded backend (only one is loaded by default) |
| VRAM | runs on 8 GB with CPU offload, at a ~1 MP ceiling; 12 GB+ for the 2 MP Qwen recommends |

The engine clone itself is ~400 MB of checkout plus ~1.5 GB of build tree, and
`doctor` reports the compute devices it found, so an unsupported GPU is visible
before the first generation rather than during it.

## Uninstall

Nothing installs system-wide. `rm -rf third_party models .venv` removes the
engine, the weights and the virtualenv; the repository itself is just files.

## Run

```bash
export IMAGEGEN_API_KEYS="$(openssl rand -hex 24)"
uv run imagegen serve --config config/service.toml
```

| Variable | Effect |
| --- | --- |
| `IMAGEGEN_API_KEYS` | comma-separated bearer tokens; overrides `service.api_keys` |
| `IMAGEGEN_HOST`, `IMAGEGEN_PORT` | override the bind address (default `127.0.0.1:8080`; see [Network exposure](#network-exposure) to listen publicly) |

With no keys configured the service is unauthenticated and logs a warning — only do
that on loopback or behind an authenticating proxy. See
[Authentication](#authentication) for the full picture.

## Network exposure

The bare-metal config defaults to **loopback only**; the container config binds
`0.0.0.0`. That asymmetry is deliberate — a container is reached through a
published port, so binding loopback inside it would make it unreachable.

### Bare metal

`config/service.toml` ships `host = "127.0.0.1"`. To listen on every interface:

```bash
IMAGEGEN_HOST=0.0.0.0 IMAGEGEN_API_KEYS="$(openssl rand -hex 24)" \
  uv run imagegen serve --config config/service.toml
```

or set it in the file:

```toml
[service]
host = "0.0.0.0"
```

`imagegen serve` accepts **only** `--config` — there are no `--host` or `--port`
flags. The complete set of environment overrides is `IMAGEGEN_API_KEYS`,
`IMAGEGEN_PORT` and `IMAGEGEN_HOST` (`src/imagegen/cli.py`).

Measured:

```
IMAGEGEN_HOST=0.0.0.0 IMAGEGEN_PORT=8096
  Uvicorn running on http://0.0.0.0:8096
  ss -tlnp            LISTEN 0.0.0.0:8096
  loopback /health                        200
  LAN IP (192.168.1.181) /health        200
  POST /v1/images/generations, no key   401
```

**`public_base_url` has no environment override.** `service.toml` sets it to
`http://127.0.0.1:8080`, so as soon as you listen publicly, `response_format="url"`
hands remote clients an address that points back at their own machine. Edit the key
to the address they actually use:

```toml
public_base_url = "http://192.168.1.181:8080"
```

`b64_json` responses are unaffected.

### Docker (port publishing)

The container already binds `0.0.0.0` (`config/container.toml`). What decides
external reachability is how the port is **published**:

| Form | Host bind | Reachable from |
| --- | --- | --- |
| `-p 8080:8080` | `0.0.0.0` and `[::]` | anywhere on the network |
| `-p 127.0.0.1:8080:8080` | loopback only | this machine only |
| `--network host` | the app's own bind, no NAT | anywhere (Linux only) |

`docker-compose.yml` publishes `"${IMAGEGEN_PORT:-8080}:8080"` with no address
prefix, so **all host interfaces** by default:

```bash
IMAGEGEN_API_KEYS="$(openssl rand -hex 24)" docker compose up -d   # LAN-reachable
```

Measured:

```
-p 8092:8080             host listens 0.0.0.0:8092 and [::]:8092
                         http://192.168.1.181:8092/health -> 200
-p 127.0.0.1:8093:8080   host listens 127.0.0.1:8093
                         loopback 200, LAN IP refused
--network host, port 8094  host listens 0.0.0.0:8094, LAN IP 200
```

`public_base_url` applies here too; see
[Container configuration](#container-configuration) for replacing the baked config without a rebuild.

### Before exposing on a routable interface

1. **Set keys.** Empty `IMAGEGEN_API_KEYS` means unauthenticated. A
   GPU-exhausting endpoint reachable from the LAN is a denial-of-service
   invitation; `max_waiting_requests` bounds the queue but does not remove it.
2. **Fix `public_base_url`**, or `response_format="url"` returns unreachable links.
3. **Put a TLS-terminating reverse proxy in front of it.** The bearer token travels
   in a header and is sniffable over plaintext HTTP.
4. **Keep the engine on loopback.** `sd-server` has no credentials and the gateway
   is the only authenticating layer — never publish backend ports directly.

## Docker

One image carries **both** compute backends; which one runs is an environment
variable, not a rebuild.

### Docker prerequisites

Docker plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
registered with it — `docker info --format '{{.Runtimes}}'` should list `nvidia`.
The toolkit is required for **both** backends, not just CUDA: it is what mounts the
driver's Vulkan ICD and GL libraries into the container.

### Build

```bash
docker build -t imagegen:latest .
```

The engine is compiled from source inside the image at the pinned commit, using the
same `scripts/build_engine.sh` as the bare-metal install, so the image cannot drift
from what the repository pins. `--build-arg CUDA_ARCHITECTURES` defaults to `75`
(Turing); widen it to `"75;80;86;89;90"` to distribute the image, at the cost of a
much longer CUDA compile.

### Weights

Never baked in — they are ~31.7 GB and change independently of the code. Put them
in a volume and fill it once:

```bash
docker volume create imagegen-models
docker run --rm -v imagegen-models:/models imagegen:latest fetch-models
# equivalently: docker compose run --rm fetch-models
```

**Fetch before you `up`.** The gateway validates weights at startup, so serving an
empty volume fails. Rather than crash-looping silently, the container prints the
command above and exits.

To use weights that already live on the host, skip the named volume and bind-mount
them — `IMAGEGEN_MODELS_DIR` replaces the volume for both services:

```bash
IMAGEGEN_MODELS_DIR="$HOME/models" IMAGEGEN_API_KEYS="..." docker compose up -d
```

### Compose ergonomics

`IMAGEGEN_API_KEYS` is declared required, so **every** Compose command needs it —
including `ps`, `logs` and `down`. Put it in a `.env` file next to
`docker-compose.yml` (gitignored) so you do not repeat it:

```
IMAGEGEN_API_KEYS=your-key
IMAGEGEN_BACKEND=cuda
```

### Run the container

```bash
IMAGEGEN_API_KEYS="$(openssl rand -hex 24)" docker compose up -d
```

or without Compose:

```bash
docker run -d --gpus all \
  -e IMAGEGEN_API_KEYS="$(openssl rand -hex 24)" \
  -v imagegen-models:/models -p 8080:8080 \
  --name imagegen imagegen:latest
```

### Choosing the backend

```bash
docker run --gpus all -e IMAGEGEN_BACKEND=vulkan ... imagegen:latest   # default: cuda
```

The entrypoint repoints `/opt/imagegen/engine/current` at the chosen engine and the
config always references that symlink, so switching needs neither a rebuild nor a
config edit. `IMAGEGEN_BACKEND` accepts `cuda` or `vulkan` and rejects anything
else with exit 2.

### Container configuration

The config baked into the image lives at `/etc/imagegen/config.toml` (built from
`config/container.toml`). Everything you would normally want to change at runtime
goes through the environment:

| Variable | Default | Effect |
| --- | --- | --- |
| `IMAGEGEN_API_KEYS` | — (Compose requires it) | comma-separated bearer tokens; empty means **no auth** |
| `IMAGEGEN_BACKEND` | `cuda` | `cuda` or `vulkan`; repoints `engine/current`; any other value exits 2 |
| `IMAGEGEN_CONFIG` | `/etc/imagegen/config.toml` | which config file to load |
| `IMAGEGEN_HOST`, `IMAGEGEN_PORT` | from the config file | override the bind address without editing config |
| `IMAGEGEN_MODELS_DIR` | named volume `imagegen-models` | host directory to bind-mount at `/models` |
| `NVIDIA_VISIBLE_DEVICES` | `all` | which GPUs the container sees; `all` also gets the Vulkan ICD |
| `NVIDIA_DRIVER_CAPABILITIES` | `all` (hardcoded in Compose) | must include more than `compute`: the Vulkan ICD and the GL libraries the ICD `dlopen`s ship with the driver |
| `CUDA_ARCHITECTURES` | build arg, `75` | CUDA architectures to compile for; widen only when distributing |

For how publishing works — the difference between `-p 8080:8080`,
`-p 127.0.0.1:8080:8080` and `--network host` — see
[Network exposure](#network-exposure).

For anything the table does not cover — the model registry, lifecycle policy,
timeouts, VRAM budget — mount your own config instead of rebuilding:

```bash
docker run --gpus all \
  -e IMAGEGEN_CONFIG=/etc/imagegen/custom.toml \
  -v "$PWD/my-container.toml:/etc/imagegen/custom.toml:ro" \
  -v imagegen-models:/models -p 8080:8080 imagegen:latest
```

The same in Compose:

```yaml
    volumes:
      - ./my-container.toml:/etc/imagegen/custom.toml:ro
    environment:
      IMAGEGEN_CONFIG: /etc/imagegen/custom.toml
```

Verified: a mounted config with `port = 8085` makes the gateway listen on 8085 and
leaves 8080 unused, so the baked file really is being replaced.

Editing `config/container.toml` in the repo works too but needs an image rebuild.
Note its header: the `[[models]]` blocks are duplicated from
`config/service.toml`, so a model change belongs in both. To add a model in the
container without rebuilding, follow [Adding a model](#adding-a-model) and mount
the result via `IMAGEGEN_CONFIG`.

### Measured in-container (RTX 2070 8 GB, 512×512)

Measured on this machine through the gateway after the `0518b34` rebuild:

| Backend / model | Generation time |
| --- | --- |
| `cuda` / `moss-image` (cold) | 54 s |
| `vulkan` / `flux-3-image-klein` (cold) | 31 s |
| Compose / `flux-3-image-klein` (cold, weights already in page cache) | 15 s |

Both entrypoints log `stable-diffusion.cpp version master-982-0518b34, commit
0518b34` at start, `GET /v1/models` lists all five models, and an unauthenticated
generation request returns 401. Note that only the NVIDIA device is passed into
the container, so `Vulkan0` there *is* the RTX 2070 — the iGPU-index trap
described under [Compute backend](#compute-backend) does not apply, which is why
the container Vulkan numbers are faster than the host's Vulkan1 figures: no iGPU
displaces it from enumeration.

The image rebuilds its engines from the pinned commit in `scripts/build_engine.sh`
and copies `config/container.toml` verbatim, so a fresh `docker build` after either
changes ships them. If `--version` disagrees with the line above, your image
predates a pin bump — rebuild before quoting this table.

### Entrypoint commands

| Command | Purpose |
| --- | --- |
| `serve` (default) | run the gateway |
| `fetch-models` | download weights into `/models`; resumable |
| `doctor` | validate config, list devices |
| `models` / `plan` | registry, or the exact engine argv |

## Configuration reference

Three TOML tables drive the service. Every table is strict (`extra="forbid"`), so an
unknown key is a named startup error rather than a silent no-op — a quietly ignored
sampling value produces wrong images that look like the model's fault.

Two files ship: `config/service.toml` (bare metal, loopback) and
`config/container.toml` (container paths, `0.0.0.0`). Their `[[models]]` blocks are
duplicated, so a model change belongs in both.

### `[service]`

| Key | Default | Meaning |
| --- | --- | --- |
| `host` | `"127.0.0.1"` | bind address; `IMAGEGEN_HOST` overrides. See [Network exposure](#network-exposure) |
| `port` | `8080` | bind port; `IMAGEGEN_PORT` overrides (1–65535) |
| `api_keys` | `[]` | accepted bearer tokens — **empty disables authentication**. Prefer `IMAGEGEN_API_KEYS` so keys never land in a file |
| `public_base_url` | `null` | base URL advertised by `response_format="url"`; requests for URLs fail with a clear 400 when unset. **No environment override** |
| `request_timeout_seconds` | `1800` | how long a synchronous generation may run before the gateway gives up |
| `max_waiting_requests` | `8` | requests allowed to queue behind a generation; beyond this, new requests are rejected rather than queued |
| `blob_ttl_seconds` | `3600` | how long an image is held in RAM for its response URL |
| `blob_capacity` | `64` | maximum number of stored response images; oldest evicted first |

### `[backends]`

| Key | Default | Meaning |
| --- | --- | --- |
| `sd_server_binary` | **required** | path to the `sd-server` executable |
| `models_dir` | **required** | directory holding weights; the weight names in `[[models]]` resolve against it, relative to the config file |
| `lifecycle` | `"on_demand"` | `on_demand` starts a backend the first time its model is requested; `eager` starts every configured model at boot |
| `max_resident_backends` | `1` | how many weight sets may be loaded at once. Each Qwen backend holds ~9.7 GB of weights plus activations, so this stays at 1 on an 8 GB card and is raised only where it genuinely fits |
| `idle_timeout_seconds` | `300` | unload a backend after this long without requests |
| `start_timeout_seconds` | `600` | how long to wait for a backend to report ready — loading is gigabytes |
| `stop_timeout_seconds` | `30` | grace period for a clean backend shutdown before it is killed |
| `offload_to_cpu` | `true` | pass `--offload-to-cpu`, keeping part of the weights in system RAM |
| `diffusion_flash_attention` | `true` | pass `--diffusion-fa` |
| `max_vram_gib` | `null` | per-device budget handed to `--max-vram`. Leave unset to let the engine auto-fit; set it when other processes share the GPU |
| `log_level` | `"info"` | `debug` \| `verbose` \| `info` \| `warn` \| `error` |
| `extra_args` | `[]` | appended **verbatim to every** `sd-server` command line — never use it for a per-model flag, which needs a `ModelProfile` field |

### `[[models]]` and `[models.sampling]`

Covered by [Adding a model](#adding-a-model), which lists all 15 model keys and
all 7 sampling keys with required/optional status and defaults. The same list is
repeated in the banner above `[[models]]` in `config/service.toml`.

### What can be overridden from the environment

Only three keys, deliberately the ones that must stay out of files:

| Variable | Overrides |
| --- | --- |
| `IMAGEGEN_API_KEYS` | `service.api_keys` |
| `IMAGEGEN_HOST` | `service.host` |
| `IMAGEGEN_PORT` | `service.port` |

Everything else is file-only — notably `public_base_url`. The container adds two of
its own, read by the entrypoint rather than the service: `IMAGEGEN_CONFIG` (which
file to load) and `IMAGEGEN_BACKEND` (which engine the `current` symlink points
at). See [Container configuration](#container-configuration).

## Models

Configured in `config/service.toml`:

| Model id | Weights | Sampling |
| --- | --- | --- |
| `qwen-image-2.1` | `leejet/Qwen-Image-2.1-GGUF` `qwen_image_2.1-Q4_K` | 40 steps, CFG 6.0, euler |
| `qwen-image-2.1-turbo` | `AtomicChat/Qwen-Image-2.1-Turbo-GGUF` `AD-Q4_K` | 8 steps, CFG 1.0, euler, fixed sigmas |
| `z-image-turbo` | `leejet/Z-Image-Turbo-GGUF` `Q4_K` | 8 steps, CFG 1.0, euler_a |
| `moss-image` | `realrebelai/MOSS-Image_GGUFs` `SFT-Q5_K_M` | 20 steps, CFG 3.0, euler, shift 5 |
| `flux-3-image-klein` | `unsloth/flux-3-image-klein-GGUF` `Q5_K_M` | 16 steps, CFG 1.0, euler, shift 6.93 |

Shared by the two Qwen models: VAE `Comfy-Org/Qwen-Image-2.1` →
`vae/qwen_image_2.1_vae_bf16.safetensors`, text encoder
`Qwen/Qwen3-VL-8B-Instruct-GGUF` → `Qwen3VL-8B-Instruct-Q4_K_M.gguf`, vision tower
`mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` (image editing).

Z-Image brings its own set — text encoder
`unsloth/Qwen3-4B-Instruct-2507-GGUF` `Q4_K_M` and the FLUX.1-schnell autoencoder
(`ae.safetensors`, mirrored from `Comfy-Org/z_image_turbo` because the upstream
FLUX repo is gated) — and has no vision tower, so it does not offer editing. Its
weights are Apache-2.0, unlike the Qwen ones; see [License](#license).

MOSS-Image SFT and FLUX.3 Image Klein share the **FLUX.2 VAE**
(`flux2-vae.safetensors`, mirrored ungated from `Comfy-Org/MOSS-Image`; the BFL
repo is gated). MOSS pairs a 5B denoiser with the Qwen3.5-9B text encoder and an
**external `tokenizer.json`** — Qwen3.5's vocabulary is not embedded in sd.cpp, so
the `tokenizer` config key is what makes this model servable. FLUX.3 Klein pairs
a 4B distilled denoiser with Qwen3-VL-4B and is the fastest model here at
1024×1024; unlike the Qwen models, its edits use the `flux2` preset, which sends
reference images to the DiT through the VAE and therefore needs no vision tower.

**Which model for what:** both Qwen models remain the default for text-heavy
graphics (measured letter-exact at 1024×1024); Z-Image is the fast sketcher;
FLUX.3 Klein is the fastest model here and edits images; MOSS has the strongest
prompt adherence of the set at 20 steps but the slowest prompt encoding (9B
text encoder). See [Choosing a model for posters and
infographics](#choosing-a-model-for-posters-and-infographics).

Two things worth knowing before changing quants:

- **The Turbo sigma schedule is part of the model.** It is the `sample_sigmas`
  array from `Qwen/Qwen-Image-2.1-Turbo/model_index.json` with a trailing `0.0`.
  Without it the engine applies the base schedule and the picture comes out
  different. It is asserted in the test suite for both the API payload and the
  `sd-server` command line.
- **Text-encoder quantization moves the image as much as the denoiser does.** Q8_0
  is closest; Q4_K_M is roughly equivalent to a Q4 denoiser in perceptual drift.
  Swap `text_encoder` in the config if you have the RAM for it.

### Adding a model

A model is one `[[models]]` block in `config/service.toml` — no code, no restart
ceremony beyond restarting the service. The schema is **strict**: a misspelled or
unsupported key is a startup error naming that key, never a silent no-op.

| Key | Required | Default | Notes |
| --- | --- | --- | --- |
| `id` | yes | — | Public identifier, `[a-z0-9][a-z0-9._-]*`; must be unique |
| `diffusion_model` | yes | — | Denoiser file name inside `models_dir` |
| `vae` | yes | — | VAE file name |
| `text_encoder` | yes | — | Passed to `sd-server` as `--llm` |
| `port` | yes | — | Loopback port for this instance; must be unique |
| `default_size` | yes | — | `"1024x1024"`; used when the client omits `size` |
| `max_pixels` | yes | — | Ceiling on `width * height` |
| `sampling` | yes | — | `[models.sampling]` table, below |
| `description` | no | `""` | Shown by `imagegen models` |
| `vision_encoder` | no | none | mmproj tower; only VLM-based edit presets (qwen) need it |
| `tokenizer` | no | none | External `tokenizer.json` when the vocabulary is not embedded in sd.cpp (MOSS/Qwen3.5) |
| `supports_editing` | no | `false` | Offers `/v1/images/edits`; the tower is passed as `--llm_vision` only when `vision_encoder` is also set |
| `layout` | no | none | Structured-layout dialect this checkpoint parses inside the prompt (e.g. `moss`); advertised in `GET /v1/models` |
| `dimension_multiple` | no | `32` | Width and height must be multiples of it |
| `max_images_per_request` | no | `1` | 1..8 |

`[models.sampling]` — these become the backend's own startup flags *and* the
defaults sent with every request, so a distilled checkpoint stays correct even if
something talks to `sd-server` directly:

| Key | Required | Default | `sd-server` flag |
| --- | --- | --- | --- |
| `steps` | yes | — | `--steps` |
| `cfg` | yes | — | `--cfg-scale` |
| `sample_method` | yes | — | `--sampling-method` |
| `scheduler` | no | none | `--scheduler`, when set |
| `flow_shift` | no | none | `--flow-shift`, when set |
| `custom_sigmas` | no | `[]` | `--sigmas`, comma-joined; each value in `[0, 1]` |
| `negative_prompt` | no | `""` | used when the client sends none |

`id`, `description`, `default_size`, `max_pixels` and `dimension_multiple` are
gateway-side policy: they validate requests and never reach the command line.
`supports_editing` is the exception that does change the process — the vision tower
is passed as `--llm_vision` only when it is true *and* `vision_encoder` is set.
The two are deliberately independent: the `flux2`/`flux3` edit preset routes
reference images through the VAE, so `flux-3-image-klein` edits with no tower at
all (verified: an edit request to it succeeds and the output changes).

Weight paths are **file names resolved against `[backends].models_dir`**, which is
itself relative to the config file, so a checkout stays relocatable. Files must end
in `.gguf` or `.safetensors`.

Worked example — a cheaper variant of a model already on disk:

```toml
[[models]]
id = "qwen-image-2.1-fast"
description = "20-step base variant for drafts."
diffusion_model = "qwen_image_2.1-Q4_K.gguf"
vae = "qwen_image_2.1_vae_bf16.safetensors"
text_encoder = "Qwen3VL-8B-Instruct-Q4_K_M.gguf"
vision_encoder = "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
port = 8113
supports_editing = true
default_size = "832x832"
dimension_multiple = 32
max_pixels = 1048576

[models.sampling]
steps = 20
cfg = 4.0
sample_method = "euler"
scheduler = "discrete"
```

Then confirm it in this order:

```bash
uv run imagegen doctor --config config/service.toml   # weights present, exit 0
uv run imagegen plan --model qwen-image-2.1-fast      # the exact sd-server argv
uv run imagegen models                                # it appears in the registry
curl -s localhost:8080/v1/models | python3 -m json.tool  # and to clients
```

To add a model whose weights are new to the machine, also add its files to
`scripts/fetch_models.sh` so the download stays reproducible.

**When config is not enough.** A model that needs an engine flag no field
represents cannot be added this way — `[backends].extra_args` is appended to
*every* model's command line, so it cannot express a per-model flag. Ming-Image
is the concrete case: it needs `--tokenizer <tokenizer.json>` for its Ling-mini-2.0
encoder, which has no field yet. Adding one means a `ModelProfile` field plus
`build_server_args` handling; see [AGENTS.md](AGENTS.md).

## API

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | registered models |
| `POST /v1/images/generations` | `model`, `prompt`, `n`, `size`, `response_format`, `output_format`, `output_compression`, `negative_prompt`, `sd_cpp` |
| `POST /v1/images/edits` | multipart `prompt`, `image[]`/`image`, `mask`, `size`, `n`, `strength`, `output_format`, `sd_cpp` |
| `GET /v1/images/content/{id}` | backs `response_format=url` |
| `GET /health` | liveness, unauthenticated |
| `GET /health/backends` | per-model backend state, loaded weights, advertised limits |

Extensions beyond OpenAI, both optional:

- **`sd_cpp`** — any field of the native `sdcpp` request schema, merged over the
  model profile. `{"sample_params": {"eta": 0.5}}` changes only `eta`.
- **`strength`** on edits — see the note below; defaults to `1.0`.

There is also one optional key on the **`/v1/models` responses**, and only on
responses:

- **`layout`** — a dialect identifier string, present only for a model whose
  profile sets `layout`. `"moss"` means the checkpoint parses a MOSS-Image layout
  JSON document sent as an ordinary `prompt` string: `summary.caption`, `items`
  with `label`/`caption`/`text`/`bbox`, `bbox_format: "yxyx"` and
  `bbox_coord_space: "abs_centralized"` (pixel coordinates with the canvas centre
  as origin, so a 1024×1024 canvas spans −512…+512). imagegen relays it verbatim
  and does no layout parsing or validation of its own.

  The key is **omitted entirely when unset** — never `null`. A client can
therefore tell "this model does not do layout" (key absent) apart from "this
engine predates the extension" (also absent, but the field is unknown to it), and
serialising `null` would erase that distinction.

  Availability of the dialect is in the weights: upstream documents layout input
  for both MOSS-Image Base and SFT, and the shipped `moss-image` profile sets
  `layout = "moss"`. Layout prompts are marked *experimental* upstream, and
  `bbox` values are absolute pixels for one canvas, so a client must generate at
  the size it computed the coordinates for.

Geometry rules come from the model profile: dimensions must be multiples of
`dimension_multiple` (32) and `width * height <= max_pixels`. Qwen recommends ~2 MP;
the shipped config caps at 1 MP because that is what an 8 GB card sustains.

### Editing

The upstream edits route assigns the first uploaded image to `init_image` **and**
`ref_images`. At the engine's default `strength` of 0.75 the init image dominates
and the pipeline reproduces the input — an instruction like *"change 'BASE MODEL' to
'EDITED OK'"* comes back with the text unchanged. The service therefore sends
`strength: 1.0` by default, so sampling starts from noise and is guided by the
reference image, matching `sd-cli -r`. Pass `strength` (or `sd_cpp.strength`) to get
img2img behaviour instead.

## Authentication

The key is a **static shared secret that you define**. The service does not issue,
sign, expire or hash keys — it compares the string you send against the list you
configured. That is the entire mechanism.

```
configure:  IMAGEGEN_API_KEYS="sk-demo-111,sk-demo-222"
                 |
client:     Authorization: Bearer sk-demo-111
                 |
gateway:    is this string in the configured list?  yes -> proceed
                                                   no  -> 401
```

### Configuring keys

Highest priority first:

| Source | Notes |
| --- | --- |
| `IMAGEGEN_API_KEYS` env var | comma-separated, **overrides** the config file. Preferred: keeps secrets out of git |
| `service.api_keys` in `config/service.toml` | a tracked file — only put throwaway values here |
| empty list (default) | **no authentication at all.** One warning is logged at startup |

```bash
export IMAGEGEN_API_KEYS="$(openssl rand -hex 24)"
uv run imagegen serve --config config/service.toml
```

### Sending the key

Either header is accepted. `Bearer` is matched case-insensitively and values are
whitespace-trimmed; comparison is constant-time (`secrets.compare_digest`) so a key
matched later in the list does not answer measurably slower than one matched earlier.

```bash
-H "Authorization: Bearer $IMAGEGEN_API_KEYS"   # what the OpenAI SDK sends
-H "x-api-key: $IMAGEGEN_API_KEYS"              # accepted too, for proxies and CLIs
```

With an OpenAI SDK you never set the header yourself:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-demo-111")
```

### What is and is not gated

| Endpoint | Key | Why |
| --- | --- | --- |
| `POST /v1/images/generations` | required | consumes GPU time |
| `POST /v1/images/edits` | required | consumes GPU time |
| `GET /health/backends` | required | exposes model ids, ports, PIDs, limits |
| `GET /health` | open | liveness probes must work without credentials |
| `GET /v1/models` | open | lists model ids only |
| `GET /v1/images/content/{id}` | open | the unguessable URL token *is* the capability, like OpenAI's pre-signed object URLs |

A missing key and a wrong key produce the same body, so a client cannot probe which
keys exist:

```json
{"error": {"message": "invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}}
```

### Checking auth without generating

A valid key on an invalid `size` returns `400`, not `401` — reaching validation proves
the key was accepted, and it costs milliseconds instead of a full generation:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8080/v1/images/generations \
  -H "Authorization: Bearer $IMAGEGEN_API_KEYS" -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1-turbo","prompt":"x","size":"1000x1000"}'   # 400 = key accepted
```

### Deliberate limits

- **Not per-user.** No identities, quotas or per-key usage tracking. Everyone sharing a
  key is the same caller; rotate by changing `IMAGEGEN_API_KEYS` and restarting.
- **Bearer tokens travel in cleartext.** Fine on loopback; anything crossing a network
  belongs behind an HTTPS reverse proxy (Caddy, nginx) that terminates TLS.
- **The backends themselves have no credentials.** `sd-server` listens on `127.0.0.1`
  only, which is enforced by the launcher. Anyone able to reach ports 8111/8112 can
  generate freely and bypass the queue, so those ports must stay local.

### Choosing a model for posters and infographics

Measured with a single prompt — a three-column infographic with a title, three
column headings and three captions — at 1024×1024, seed 42, on this machine:

| Model | Wall time | Lettering in the output |
| --- | --- | --- |
| `qwen-image-2.1-turbo` | 57 s | every string spelled exactly |
| `qwen-image-2.1` | 462 s | every string spelled exactly |
| `z-image-turbo` | 34 s warm, 39 s from cold | title and headings exact, but at least one caption garbled in **3 of 3 seeds** (`aromnatic`, `concnetrhaiio`) |
| `flux-3-image-klein` | 66 s | every string spelled exactly; layout left the bottom half empty and the three icons were interchangeable vessels rather than a dripper, press and cup |
| `moss-image` | 179 s | every string spelled exactly, correct icons, composition fills the frame — the strongest of the five, at the highest cost |

The practical split: **Qwen Turbo is the text-heavy default** — it costs about 20 s
more than Z-Image and it spells, at the same 8 steps. Flux3 Klein matched Qwen
Turbo on spelling in its single run with a much simpler layout; MOSS is the
quality ceiling (letter-exact *and* right icons) but takes 3× longer than Turbo
for it. Z-Image earns its place as the fast sketcher: its sparse poster
compositions are strong, it loads and runs ~40% faster, and it is Apache-2.0
alongside MOSS. Treat it as the layout comp and Qwen as the final render. This was
measured at `Q4_K` for the older denoisers (the two new models at `Q5_K_M`); the
`Q8_0` Denoiser is 2.7 GB larger and was not tested, so quantization may be part
of the lettering gap.

The two new models have **one run each** of this prompt, versus three seeds for
Z-Image — treat their rows as indicative, not settled.

## Behaviour measured on an RTX 2070 (8 GB, CPU offload, 512×512 unless noted)

| Operation | Time |
| --- | --- |
| load Turbo weights → ready | ~2-3 s |
| Turbo generation, 8 steps | ~16 s |
| base generation, 40 steps | ~88-98 s |
| edit, base model | ~100 s |
| model swap (unload + load) | ~2 s |
| Z-Image generation, 1024×1024 | ~34 s |
| flux3 generation, 16 steps, 1024×1024 (CUDA) | ~58 s cool, ~63-71 s once the card reaches 82 °C |
| flux3 generation, 16 steps, 512×512 (Vulkan1) | ~52 s cold, ~19 s warm |
| flux3 edit, 512×512 (CUDA) | ~32 s |
| MOSS generation, 20 steps, 1024×1024 (CUDA) | ~202 s cold, ~221 s including the model swap |
| MOSS generation, 20 steps, 512×512 (Vulkan1) | ~67 s cold, ~63 s |
| Qwen Turbo edit, 512×512 (CUDA) | ~25 s |

MOSS is the slow model here: 20 steps plus a 9B text encoder whose prompt
encoding alone takes tens of seconds. Flux3 Klein is the fast one — its warm
512×512 on Vulkan beats the Qwen Turbo figure that has been quoted since 0.1.0.
CUDA numbers drift with temperature: the same 1024×1024 flux3 run measured 58 s
with a cool card and ~70 s at 82 °C, so treat ±20% as clock behaviour, not noise.

Generations are synchronous, so a client that asks for 40 steps at 2 MP must be
prepared to wait; `service.request_timeout_seconds` bounds it.

## Tests

```bash
uv run pytest                    # 100 tests: config, translation, HTTP API, process supervision
                                 # no GPU needed; the 2 weight-dependent tests skip
scripts/smoke_test.py --base-url http://127.0.0.1:8080 --api-key "$KEY" \
    --model qwen-image-2.1-turbo --size 512x512
```

The pytest suite launches **real** subprocesses and talks real HTTP:
`tests/stub_sd_server.py` implements the `sd-server` surface (capabilities probe,
generations, edits) and is launched through the same code path as the real engine,
so process supervision, readiness, queueing, model swapping, crash recovery and idle
unload are all exercised for real. Only inference is stubbed. `scripts/smoke_test.py`
runs against a live service with real weights.

## Layout

```
config/service.toml        model registry and runtime settings
src/imagegen/
  app.py                   composition root, lifespan, error handlers
  cli.py                   serve | models | plan | doctor
  config.py                typed config + validation
  openai_payload.py        OpenAI <-> stable-diffusion.cpp wire format
  backend_process.py       one supervised sd-server child process
  backend_pool.py          queueing, single residency, eviction, reaper
  backend_client.py        HTTP client and response decoding
  routes_images.py         /v1/models, /v1/images/*
  routes_health.py         /health, /health/backends
  image_blob_store.py      TTL store for response_format=url
  security.py              bearer / x-api-key authentication
  errors.py                ApiError hierarchy
docs/architecture.md       design, diagrams, rationale
scripts/                   build_engine.sh, fetch_models.sh, smoke_test.py
tests/                     pytest suite + stub engine
```

See [docs/architecture.md](docs/architecture.md) for the request lifecycle, state
machines and design decisions.

## Limitations

- One backend serves one request at a time; extra callers queue and then get `429`.
- `response_format=url` keeps images in RAM for `blob_ttl_seconds` and is served
  unauthenticated, the token being the capability (like OpenAI's pre-signed URLs).
- `n > 1` is capped per model; on a small GPU keep it at 1.
- Video, ControlNet, LoRA and upscaling are reachable through `sd_cpp` and
  `backends.extra_args` but are not modelled in this service's API.

## License

MIT — see [LICENSE](LICENSE). This covers the service code in this repository
only. Several things it does not cover, and which are fetched separately by the setup
scripts rather than redistributed here:

- **stable-diffusion.cpp** (`scripts/build_engine.sh`) — MIT, © 2023 leejet.
- **The Qwen-Image-2.1 weights** (`scripts/fetch_models.sh`) — **not** open source.
  They are released by Hangzhou Tongyi Laboratory Technology Co., Ltd. under the
  *Qwen Research License Agreement*, which grants the right to use, reproduce,
  distribute and modify the weights **for non-commercial purposes only**
  (research or evaluation). Commercial use requires a separate license from the
  licensor. Note also that derivative works may not be named "Qwen" as their
  primary identifier.
- **The Z-Image-Turbo weights** (`scripts/fetch_models.sh`) — **Apache-2.0**, as are
  their Qwen3-4B text encoder and FLUX.1-schnell autoencoder.
- **The MOSS-Image weights** (`scripts/fetch_models.sh`) — **Apache-2.0** for the
  SFT denoiser, the Qwen3.5-9B text encoder and the FLUX.2 VAE mirror, per the
  license metadata of all three repositories. Alongside Z-Image this makes two of
  the five served models usable commercially.
- **The FLUX.3 Image Klein weights** (`scripts/fetch_models.sh`) — **FLUX
  Kommunity License v1.0**: non-commercial use for anyone, *plus* commercial use
  for a "Qualifying User" (affiliates under US$5M annualized revenue). Read it
  (`LICENSE.md` in `unsloth/flux-3-image-klein-GGUF`) before shipping output
  commercially above that threshold. The Qwen3-VL-4B text encoder is Apache-2.0.

MIT covers this repository's code; it does not relicense the weights, and running
the service with them commercially still requires the Qwen commercial license.
