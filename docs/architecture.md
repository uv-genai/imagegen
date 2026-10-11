# Architecture

Installation, prerequisites and troubleshooting live in
[README.md](../README.md#install); this document covers structure, contracts and
why they are shaped the way they are.

## What the service is

`imagegen` exposes an **OpenAI-compatible image API** in front of
[`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp), which runs
the Qwen-Image-2.1 family. It is a gateway plus a process supervisor, not an
inference engine: all sampling happens inside `sd-server` child processes.

```
                       client: OpenAI SDK, Open WebUI, curl
                                     │
                    POST /v1/images/generations   (JSON)
                    POST /v1/images/edits         (multipart)
                    GET  /v1/models
                    GET  /v1/images/content/{id}  (response_format=url)
                                     │
        ┌────────────────────────────▼────────────────────────────┐
        │  imagegen gateway (FastAPI, single event loop)          │
        │                                                         │
        │  security.require_api_key   Bearer / x-api-key          │
        │  schemas.GenerationRequest  validation, unknown-field   │
        │                             rejection                   │
        │  model_resolution           model + size policy         │
        │  openai_payload             OpenAI ⇄ sdcpp translation  │
        │  backend_pool               queue, eviction, reaper     │
        │  backend_client             HTTP to sd-server           │
        │  image_blob_store           TTL store for url responses │
        └───────────┬─────────────────────────┬───────────────────┘
                    │ loopback HTTP           │ loopback HTTP
        ┌───────────▼───────────┐   ┌─────────▼─────────────┐
        │ sd-server :8111       │   │ sd-server :8112       │
        │ qwen_image_2.1-Q4_K   │   │ ...Turbo-AD-Q4_K      │
        │ + VAE + Qwen3-VL-8B   │   │ + VAE + Qwen3-VL-8B   │
        │ (+ mmproj for edits)  │   │ (+ mmproj for edits)  │
        └───────────────────────┘   └───────────────────────┘
                     single GPU, at most max_resident_backends loaded
```

The two `sd-server` processes are the ones from `config/service.toml`. They listen
on loopback only and carry **no credentials**: the gateway is the sole
authentication boundary, so exposing a backend port directly would be a security
bug.

## Why a gateway at all

`sd-server` already serves `/v1/images/generations`, but one process loads exactly
one weight set and its `/v1/models` is hard-coded to `sd-cpp-local`. Choosing
between the base and the Turbo checkpoint per request therefore needs something
that:

1. routes on the OpenAI `model` field,
2. applies the per-model sampling contract (Turbo is only correct at 8 steps, CFG 1
   and its own sigma schedule),
3. keeps more than one model available on a machine that cannot hold both in
   memory,
4. authenticates, validates, and reports errors in OpenAI's shape.

Building on the existing OpenAI route instead of binding to `libsd` keeps the
C++ surface at zero and inherits upstream model support as it lands.

## Backend lifecycle

`BackendProcess` owns one child process; `BackendPool` owns the set.

```
                start()                    capabilities OK
   STOPPED ───────────────▶ STARTING ────────────────────▶ READY
      ▲                        │ failed / exited              │
      │       stop(), crash    │                              │ idle_timeout,
      └────────────────────────┴◀──────────── STOPPING ◀──────┘ eviction,
                                                               shutdown
```

Readiness is not "the port answers": a loaded engine is recognised by a
capabilities report containing both `supported_modes` and `model`
(verified against `sd-server` `master-951-f89d9b1`). A process that exits during
load is reported immediately instead of burning the whole start timeout.

### Single residency and swapping

Each backend holds ~9.7 GB of weights plus activations. `max_resident_backends`
caps how many are loaded; when a request needs a cold model, the least recently
used *idle* backend is stopped first and the new one started. If every resident
backend is mid-generation the request waits — unloading one would throw away work
the client is still paying for.

Measured on an RTX 2070 (8 GB) with `--offload-to-cpu`:

| Operation | Time |
| --- | --- |
| load Turbo weights → ready | ~2-3 s |
| generation 512×512, 8 steps (Turbo) | ~16 s |
| generation 512×512, 40 steps (base) | ~88 s |
| edit 512×512 (Turbo) | ~19 s |
| model swap (unload + load other) | ~2 s |

### Concurrency

One backend serves one request at a time (`asyncio.Lock`). Excess callers are
counted as waiters; beyond `max_waiting_requests` a request is refused with
`429 queue_full` rather than piling up in front of a GPU that cannot absorb it.

## Request translation

The upstream OpenAI route reads only `prompt`, `n`, `size`, `output_format` and
`output_compression` from the JSON body. Everything native travels inside a block
appended to the prompt, which the server strips before sampling:

```
a red fox <sd_cpp_extra_args>{"sample_params":{"sample_steps":8,"sample_method":"euler",
"guidance":{"txt_cfg":1.0},"custom_sigmas":[1.0,...,0.0]},"negative_prompt":"..."}</sd_cpp_extra_args>
```

| OpenAI / gateway field | Native destination |
| --- | --- |
| `prompt` | prompt text, block appended |
| `n` | `n` (→ `batch_count`) |
| `size` | `size` (→ `width`/`height`) |
| `negative_prompt` | `negative_prompt` |
| profile `sampling.steps` | `sample_params.sample_steps` |
| profile `sampling.cfg` | `sample_params.guidance.txt_cfg` |
| profile `sampling.sample_method` | `sample_params.sample_method` |
| profile `sampling.custom_sigmas` | `sample_params.custom_sigmas` |
| profile `sampling.flow_shift` | `sample_params.flow_shift` |
| `sd_cpp` (escape hatch) | merged over all of the above |

`from_json_str` on the C++ side only assigns keys that are present
(`load_if_exists`), which is what makes partial overrides safe: a client sending
`{"sample_params":{"eta":0.5}}` keeps the profile's steps, CFG and sigmas.

A prompt containing the marker `<sd_cpp_extra_args>` is rejected outright — the
upstream strip is a regex, so a client-supplied marker could otherwise override
parameters the service intended to control.

The same values are also passed on the `sd-server` command line at launch, so a
backend is correct even for a request that bypasses the gateway.

## Error mapping

Every client-visible failure is an `ApiError` subclass, rendered by one handler
into OpenAI's `{"error": {message, type, param, code}}` envelope.

| Cause | Status | `code` |
| --- | --- | --- |
| missing/invalid key | 401 | `invalid_api_key` |
| schema or policy violation | 400 | — |
| unknown model id | 404 | `model_not_found` |
| queue full | 429 | `queue_full` |
| backend answered non-200 | 502 | `backend_error` |
| backend unreachable / timed out | 503 | `backend_unavailable` |

FastAPI's own `422` bodies are rewritten to the same 400 shape, because OpenAI
clients read `error.message` and not a `detail` array.

## Capability cross-check

Once a backend is ready, its advertised capabilities are compared against the
configured profile (`capability_mismatches`): accepted output formats versus
`output_formats`, `max_images_per_request` versus `limits.max_batch_count`, and
the default size versus `limits.max_width`/`max_height`. Mismatches are logged as
configuration defects instead of surfacing mid-request. The report is also exposed
through `GET /health/backends`.

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | TOML loading, typed models, geometry and registry validation |
| `errors.py` | `ApiError` hierarchy with OpenAI rendering |
| `schemas.py` | request schema, unknown-field policy, native overrides |
| `openai_payload.py` | the wire format: prompt block, native arg merging |
| `model_resolution.py` | model selection and size policy per request |
| `backend_process.py` | command line construction, child process, readiness |
| `backend_client.py` | HTTP to the backend, response decoding, error mapping |
| `backend_pool.py` | queueing, single residency, eviction, reaper, status |
| `image_blob_store.py` | bounded TTL store behind `response_format=url` |
| `routes_images.py` | `/v1/models`, `/v1/images/*` |
| `routes_health.py` | `/health`, `/health/backends` |
| `app.py` | composition root, lifespan, exception handlers |
| `cli.py` | `serve`, `models`, `plan`, `doctor` |

## Deliberate limits

- **Synchronous generations.** The OpenAI endpoint is synchronous; a request holds
  its connection for the whole run. The native async job API
  (`/sdcpp/v1/img_gen` + job polling) is intentionally not re-exposed: doing so
  would define a second, non-OpenAI contract for clients to learn.
- **`response_format=url` keeps bytes in RAM** for `blob_ttl_seconds`. Images are
  never written to disk by the service, which is the right default for picture
  data a client is expected to fetch once.
- **No user accounts or per-key quotas.** One shared bearer-token list; put it
  behind an authenticating proxy if tenancy is needed.
- **One GPU per deployment.** Multi-GPU means running several gateway instances,
  one per GPU, behind a load balancer; `--backend`/`--split-mode` device pinning is
  reachable through `backends.extra_args` but is not modelled in the config.
