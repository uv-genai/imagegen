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
scripts/fetch_models.sh        # 3. the weights, ~14.8 GB, resumable  (network-bound)
uv run imagegen doctor         # 4. verify binary, weights and devices
```

Order is not fixed, but `doctor` is only meaningful after steps 2 and 3. Step 2
clones [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) at
one pinned commit (~400 MB, blobless) into `third_party/`; step 3 writes to
`models/`. Both directories are gitignored and are never committed.

Peek before committing to the download:

```bash
scripts/fetch_models.sh --dry-run     # lists the files and ~14.8 GB total, writes nothing
```

### Verify the install

```bash
uv run imagegen doctor                                  # exit 0 = ready to serve
uv run imagegen models                                  # registry + sampling defaults
uv run imagegen plan --model qwen-image-2.1-turbo       # exact sd-server command line
third_party/stable-diffusion.cpp/build/bin/sd-server --version   # must report the pinned commit
uv run pytest                                           # 95 tests; the 2 needing weights self-skip
```

`sd-server --version` should print `stable-diffusion.cpp version master-951-f89d9b1,
commit f89d9b1`. That is the commit this service was built and measured against;
`PINNED_COMMIT` in `scripts/build_engine.sh` is the one place to change it. If the
version differs, the engine was built from something else and the results below do
not apply.

`doctor` prints the compute devices the engine sees and lists every missing
weight by path, so a half-finished setup reports exactly what is absent:

```
compute devices reported by sd-server:
CUDA0  NVIDIA GeForce RTX 2070 with Max-Q Design
CPU    Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz

configuration OK: 2 model(s), lifecycle=on_demand
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
| Disk | ~15 GB weights + ~1.5 GB engine build tree |
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
| `IMAGEGEN_HOST`, `IMAGEGEN_PORT` | override the bind address |

With no keys configured the service is unauthenticated and logs a warning — only do
that on loopback or behind an authenticating proxy. See
[Authentication](#authentication) for the full picture.

## Models

Configured in `config/service.toml`:

| Model id | Weights | Sampling |
| --- | --- | --- |
| `qwen-image-2.1` | `leejet/Qwen-Image-2.1-GGUF` `qwen_image_2.1-Q4_K` | 40 steps, CFG 6.0, euler |
| `qwen-image-2.1-turbo` | `AtomicChat/Qwen-Image-2.1-Turbo-GGUF` `AD-Q4_K` | 8 steps, CFG 1.0, euler, fixed sigmas |

Shared by both: VAE `Comfy-Org/Qwen-Image-2.1` → `vae/qwen_image_2.1_vae_bf16.safetensors`,
text encoder `Qwen/Qwen3-VL-8B-Instruct-GGUF` → `Qwen3VL-8B-Instruct-Q4_K_M.gguf`,
vision tower `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` (image editing).

Two things worth knowing before changing quants:

- **The Turbo sigma schedule is part of the model.** It is the `sample_sigmas`
  array from `Qwen/Qwen-Image-2.1-Turbo/model_index.json` with a trailing `0.0`.
  Without it the engine applies the base schedule and the picture comes out
  different. It is asserted in the test suite for both the API payload and the
  `sd-server` command line.
- **Text-encoder quantization moves the image as much as the denoiser does.** Q8_0
  is closest; Q4_K_M is roughly equivalent to a Q4 denoiser in perceptual drift.
  Swap `text_encoder` in the config if you have the RAM for it.

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

## Behaviour measured on an RTX 2070 (8 GB, CPU offload, 512×512)

| Operation | Time |
| --- | --- |
| load Turbo weights → ready | ~2-3 s |
| Turbo generation, 8 steps | ~16 s |
| base generation, 40 steps | ~88-98 s |
| edit, base model | ~100 s |
| model swap (unload + load) | ~2 s |

Generations are synchronous, so a client that asks for 40 steps at 2 MP must be
prepared to wait; `service.request_timeout_seconds` bounds it.

## Tests

```bash
uv run pytest                    # 95 tests: config, translation, HTTP API, process supervision
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
only. Two things it does not cover, and which are fetched separately by the setup
scripts rather than redistributed here:

- **stable-diffusion.cpp** (`scripts/build_engine.sh`) — MIT, © 2023 leejet.
- **The Qwen-Image-2.1 weights** (`scripts/fetch_models.sh`) — **not** open source.
  They are released by Hangzhou Tongyi Laboratory Technology Co., Ltd. under the
  *Qwen Research License Agreement*, which grants the right to use, reproduce,
  distribute and modify the weights **for non-commercial purposes only**
  (research or evaluation). Commercial use requires a separate license from the
  licensor. Note also that derivative works may not be named "Qwen" as their
  primary identifier.

MIT covers this repository's code; it does not relicense the weights, and running
the service with them commercially still requires the Qwen commercial license.
