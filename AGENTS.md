# AGENTS.md

Guidance for agents working on this repository.

## What this is

An OpenAI-compatible HTTP gateway in front of `sd-server` (stable-diffusion.cpp)
serving Qwen-Image-2.1 and Qwen-Image-2.1-Turbo. **The gateway performs no
inference.** All sampling happens in supervised `sd-server` child processes.

Read `docs/architecture.md` before changing request handling — it defines the
translation contract and the backend lifecycle.

## Commands

```bash
uv sync                                   # install
uv run pytest                             # full suite (launches real subprocesses)
uv run pytest tests/test_api.py -q        # one file
uv run imagegen doctor --config config/service.toml
uv run imagegen models
uv run imagegen plan --model qwen-image-2.1-turbo
scripts/build_engine.sh cuda              # build third_party engine
scripts/fetch_models.sh                   # download weights (~15 GB)
```

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
  adding a model should require editing `config/service.toml` only.

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

## Verifying changes

- Changes to the translation layer must be checked against a **live** `sd-server`,
  not only the stub: `scripts/smoke_test.py`, and for edits, confirm the output
  actually changed rather than assuming it did.
- Engine flags must be checked against `sd-server --help` of the pinned build. Never
  add a flag that was not read from that output or from `examples/server/api.md`.
- Weights and model metadata must be verified via the Hugging Face API
  (`/api/models/<repo>?blobs=true`) — file names and byte sizes in
  `scripts/fetch_models.sh` come from there.
- `pkill -f` patterns match the agent's own shell. Use a bracketed pattern such as
  `pkill -f "[b]in/imagegen serve"`.

## Engine pinning

`scripts/build_engine.sh` pins `f89d9b1` (`sd-server --version`). Sigma handling for
Qwen-Image-2.1 changed as recently as `c150a6b` (2026-10-06), so upgrades are
deliberate: change the pin, rebuild, re-run the smoke test, and re-check
`GET /sdcpp/v1/capabilities` for changed `output_formats` and `limits`.

## Repo conventions

- `third_party/` and `models/` are gitignored; never commit weights or the engine.
- Secrets stay in the environment (`IMAGEGEN_API_KEYS`), never in `config/service.toml`.
- Tests live in `tests/`, shared helpers in `tests/support.py` (imported as
  `support`, since the tests directory is not a package).
