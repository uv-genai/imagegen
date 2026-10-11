# AGENTS.md

Guidance for agents working on this repository.

## What this is

An OpenAI-compatible HTTP gateway in front of `sd-server` (stable-diffusion.cpp)
serving Qwen-Image-2.1 and Qwen-Image-2.1-Turbo. **The gateway performs no
inference.** All sampling happens in supervised `sd-server` child processes.

Read `docs/architecture.md` before changing request handling — it defines the
translation contract and the backend lifecycle.

## Setup on a fresh checkout

```bash
uv sync                        # the service; uv also fetches Python >= 3.12
scripts/build_engine.sh cuda   # engine; needs cmake >= 3.21, gcc/g++, nvcc
scripts/fetch_models.sh        # ~14.8 GB; no Hugging Face account or token needed
uv run imagegen doctor         # exit 0 when provisioned; lists every gap otherwise
```

`uv` and `git` are the only prerequisites for the service itself. `nvcc` is needed
only to build the CUDA engine (`vulkan` needs `glslc`, `cpu` needs neither).

## Commands

```bash
uv sync                                   # install
uv run pytest                             # 96 tests; launches real subprocesses, no GPU needed
uv run pytest tests/test_api.py -q        # one file
uv run imagegen doctor --config config/service.toml
uv run imagegen models
uv run imagegen plan --model qwen-image-2.1-turbo
scripts/build_engine.sh cuda              # build third_party engine at the pinned commit
scripts/fetch_models.sh                   # download weights (~14.8 GB, resumable)
scripts/fetch_models.sh --dry-run         # list them without writing bytes
third_party/stable-diffusion.cpp/build/bin/sd-server --version   # confirm the pin
```

`uv run pytest` passes on a checkout with no engine and no weights: the two tests
that inspect shipped weight files skip by design, so CI needs neither a GPU nor 15 GB.

There is no configured linter or formatter; keep to the style already present
(documented functions, early returns, files under ~200 lines).

## Architecture rules

- **`openai_payload.py` is the only module that knows the upstream wire format.**
  The native extension block (`<sd_cpp_extra_args>{...}</sd_cpp_extra_args>`),
  field names and merge order live there. Do not build prompt strings elsewhere.
- **Never let a client control `model`, `prompt` or sampling defaults directly.**
  `sd_cpp` merges *over* profile defaults; `sd_cpp.prompt` is rejected, and prompts
  containing the extension marker are rejected — the upstream strip is a regex.
- **Backends listen on `127.0.0.1` only and have no credentials.** Anything that
  binds them elsewhere must go through the gateway's authentication.
- **Add new tunables to `config.py`, not to code.** The model registry is data:
  adding a model should mean editing `config/service.toml` only. The key list, the
  required set and the defaults live in the banner above `[[models]]` there, and in
  `README.md#adding-a-model` — update **both** whenever a field changes.
- **Every config table is strict (`extra="forbid"`)**, so an unknown key is a named
  startup error rather than a no-op. That is deliberate: a silently ignored sampling
  value produces wrong images that look like the model's fault. New keys therefore
  go into `ModelProfile` / `SamplingDefaults` first, never into ad-hoc parsing.
- **Weight names are not paths.** They resolve through `BackendSettings.weight_path`
  against `[backends].models_dir`, relative to the config file, and must end in
  `.gguf` or `.safetensors`. `setup_problems()` checks existence — the vision tower
  only when `supports_editing` is set.
- **`[backends].extra_args` is global.** It is appended verbatim to *every*
  `sd-server` command line, so it must never be used for a per-model flag: it would
  be handed to unrelated models. A per-model flag needs a `ModelProfile` field.
- **To support a new engine flag**, add the field to `ModelProfile`, emit it in
  `build_server_args()`, cover it in `tests/test_backend_process.py`, and record it
  in the two tables above. Ming-Image is the standing example: it conditions on
  Ling-mini-2.0 and needs `--tokenizer <tokenizer.json>`, which has no field yet,
  so it cannot be served until one is added. Read `sd-server --help` at the pinned
  build before adding anything — do not infer flag names from model docs.

## Things that look wrong but are load-bearing

1. **`custom_sigmas` for Turbo.** `[1.0, 0.978453, 0.95418, 0.926626, 0.89508,
   0.845148, 0.704534, 0.414568, 0.0]` is the `sample_sigmas` array from
   `Qwen/Qwen-Image-2.1-Turbo/model_index.json` plus a trailing `0.0`. Without it
   the engine applies the base schedule and the output is wrong. Asserted in
   `tests/test_openai_payload.py` and `tests/test_backend_process.py`.
2. **Edits default to `strength: 1.0`** (`EDIT_STRENGTH_DEFAULT`). The upstream
   route assigns the first uploaded image to `init_image` *and* `ref_images`; at the
   engine default of 0.75 the pipeline reproduces the input and silently ignores the
   edit instruction. Measured both ways — do not "fix" this back to 0.75.
3. **Readiness requires `supported_modes` *and* `model`** in the capabilities
   payload (`CAPABILITY_READY_KEYS`). HTTP 200 alone is not readiness, and there is
   no `models` key.
4. **`max_resident_backends = 1` is not a conservative default**, it is required by
   8 GB of VRAM: each backend holds ~9.7 GB of weights.
5. **Startup seeds `--sigmas`, `--steps`, `--cfg-scale`, `--sampling-method` on the
   `sd-server` command line** in addition to sending them per request, so a backend
   is correct even if something bypasses the gateway.
6. **Unknown request parameters are a hard 400.** `IGNORED_REQUEST_FIELDS` is the
   explicit no-op allowlist; do not silently drop unrecognised fields.
7. **`z-image-turbo` looks mis-configured next to the Qwen entries and is not.**
   `dimension_multiple = 16` because its DiT patches 2×2 over an 8× VAE (verified
   with a 1008×1008 request, which is not a multiple of 32); `vae = "ae.safetensors"`
   because Z-Image reuses the FLUX.1-schnell autoencoder — mirrored from
   `Comfy-Org/z_image_turbo`, since `black-forest-labs/FLUX.1-schnell` is
   `gated: auto` and would need a token; and `sample_method = "euler_a"` because
   that is what `sd-server --help` says it defaults to for anything that is not
   Flux/SD3/Wan. It has no vision tower, so `supports_editing` must stay false.
   Measured on this GPU: 34 s at 1024×1024 versus 57 s for `qwen-image-2.1-turbo`,
   but it garbled at least one caption in 3 of 3 seeds while both Qwen models were
   letter-exact — so do not promote it to "the design model" on the strength of its
   marketing. Its reason for being here is Apache-2.0 and speed.
9. **The Docker runtime image installs `libegl1`, which looks unnecessary and is
   not.** The NVIDIA Vulkan ICD `dlopen`s `libEGL.so.1` at runtime, so it is not a
   `NEEDED` entry and `ldd` will never report it missing. Without it the container
   fails deep inside the loader with `Could not get 'vkCreateInstance' via
   'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`. `libx11-6`,
   `libxext6`, `libglx0` and `libgomp1` (OpenMP, needed by the binaries but absent
   from the CUDA runtime base) are required for the same reason. Do not trim them as
   unused packages.
10. **`build_engine.sh` treats its own verification as best-effort on purpose.**
    `sd-server --version` and `--list-devices` need the *driver* library
    (`libcuda.so.1`), which a build-only host — CI, a Docker build stage — does not
    have. Failing there would break builds that actually succeeded, so the script
    warns and continues. The real check belongs on the machine that will run it.
    Relatedly, the CUDA stage passes `CUDA_ARCHITECTURES` because `nvcc`'
    `-arch=native` cannot detect a card that is not present.
8. **`cuda` builds into `build/` while `vulkan` and `cpu` build into
   `build-<backend>`.** The asymmetry is intentional: shipped configs reference
   `build/bin/sd-server`, so renaming it would break every existing install for a
   cosmetic win. Engines coexist because each backend has its own directory, and
   the script refuses to reconfigure one whose CMake cache names a different
   backend — that is how a working engine gets silently replaced.

## Verifying changes

- Changes to the translation layer must be checked against a **live** `sd-server`,
  not only the stub: `scripts/smoke_test.py`, and for edits, confirm the output
  actually changed rather than assuming it did.
- Engine flags must be checked against `sd-server --help` of the pinned build. Never
  add a flag that was not read from that output or from `examples/server/api.md`.
- Weights and model metadata must be verified via the Hugging Face API
  (`/api/models/<repo>?blobs=true`) — file names and byte sizes in
  `scripts/fetch_models.sh` come from there.
- **Never truncate the output of a verification command.** A `list[:2]` in a debug
  print once made `output_formats` look like `[png, jpeg]` when it was
  `[png, jpeg, webp]`, and a working feature was removed on the strength of it.
- **Verify install instructions in a clean clone**, not on a dev box. This machine
  already had `hf`, `nvcc`, the weights and an engine at the pinned commit, which
  hid two install-breaking bugs that only appeared from scratch:
  `git clone "$URL" /tmp/x && cd /tmp/x && uv sync && uv run imagegen doctor`.
- `pkill -f` patterns match the agent's own shell. Use a bracketed pattern such as
  `pkill -f "[b]in/imagegen serve"`.
- `$?` after a pipeline is the **last** command's status, not the tool's. Pipe to
  `tail` and a failing `doctor` looks like exit 0.

## Engine pinning

`scripts/build_engine.sh` pins `f89d9b13d730eabeede7314ce49dacd18d3c90c2`, reported
by `sd-server --version` as `master-951-f89d9b1`. Sigma handling for Qwen-Image-2.1
changed as recently as `c150a6b` (2026-10-06), so upgrades are deliberate: change
the pin, rebuild, re-run the smoke test, and re-check
`GET /sdcpp/v1/capabilities` for changed `output_formats` and `limits`.

- **Do not reintroduce a shallow clone.** GitHub refuses to fetch an arbitrary SHA,
  so `git clone --depth 1` followed by `git fetch --depth 1 origin <sha>` fails the
  moment upstream moves past the pin — which is exactly how fresh installs broke.
  The script uses `--filter=blob:none` (full history, contents on demand: ~1.5 s,
  ~400 MB) so any commit is checkable, then asserts HEAD and every submodule match.
- Verify a pin change with `sd-server --version` after building, not only before.

## Environment facts about this repo's tooling

- The Hugging Face CLI runs ephemerally as `uvx --from huggingface-hub hf` inside
  `scripts/fetch_models.sh`. Do **not** add `huggingface-hub` to the service's
  runtime dependencies — nothing under `src/` imports it.
- Bumping the version means editing **`pyproject.toml` only**. `__version__` is read
  from installed package metadata; confirm with `imagegen --version` and `GET /health`.
- The weights download needs no HF account or token (verified with `HF_TOKEN` unset).
- `sd-server --list-devices` is the authority on device indices, and their order is
  loader order, not speed: here `Vulkan0` is the Intel iGPU and the RTX 2070 is
  `Vulkan1`. Never write a bare `vulkan`/`cuda` device reference from memory.
  Vulkan was built and run at the pin (512×512 generation: 47 s vs ~16 s on CUDA),
  so treat it as the no-CUDA fallback in anything you document.

## Repo conventions

- `third_party/` and `models/` are gitignored; never commit weights or the engine.
- Secrets stay in the environment (`IMAGEGEN_API_KEYS`), never in `config/service.toml`.
  `.env` is gitignored for exactly this reason; `.env.example` is the committed template
  and must never hold a real key.
- `config/container.toml` duplicates the `[[models]]` blocks from
  `config/service.toml` — only container paths and the bind address differ — so a
  model change belongs in both. In the image it is baked at
  `/etc/imagegen/config.toml`; `IMAGEGEN_CONFIG` overrides that path, so a modified
  config can be bind-mounted read-only instead of rebuilding the image.
- Tests live in `tests/`, shared helpers in `tests/support.py` (imported as
  `support`, since the tests directory is not a package).
- Documentation is part of the deliverable: user-facing facts go in `README.md`,
  agent-facing rules and traps in `AGENTS.md`, design rationale in
  `docs/architecture.md`. Any command written into them must have been run.
