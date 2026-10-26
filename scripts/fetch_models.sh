#!/usr/bin/env bash
# Download the weight sets required by the models in config/service.toml:
# Qwen-Image-2.1 (base and Turbo), Z-Image-Turbo, MOSS-Image SFT and
# FLUX.3 Image Klein.
#
# Every artifact below is pinned to an explicit repository revision so that a
# rebuilt machine reproduces byte-identical weights. Files already present with
# the expected size are skipped, which makes the script safe to re-run.
#
# Usage: scripts/fetch_models.sh [target_dir] [--dry-run]
#        (default target: ./models)
#
# --dry-run reports what would be fetched without transferring any bytes.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
DRY_RUN=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=(--dry-run) ;;
    *) MODEL_DIR="$arg" ;;
  esac
done
mkdir -p "$MODEL_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is required (https://docs.astral.sh/uv/); it supplies the Hugging Face CLI used below." >&2
  exit 1
fi

# The Hugging Face CLI is fetched ephemerally rather than added to the service's
# runtime dependencies: nothing under src/ imports it, and a deployed gateway
# should not carry a download tool in its dependency graph.
hf() { uvx --from huggingface-hub hf "$@"; }

# Downloads that share a repository are issued as one `hf download` call per repo
# so that the Hub connection is reused. Each entry is whitespace separated:
#   repo_id revision path [path...]
DOWNLOADS=(
  # Qwen-Image-2.1 denoiser (base checkpoint), GGUF quantizations.
  "leejet/Qwen-Image-2.1-GGUF main qwen_image_2.1-Q4_K.gguf"
  # Qwen-Image-2.1-Turbo denoiser, imatrix quantization by AtomicChat.
  # Turbo is a distinct set of weights (distilled for 8 steps at CFG 1).
  "AtomicChat/Qwen-Image-2.1-Turbo-GGUF main Qwen-Image-2.1-Turbo-AD-Q4_K.gguf"
  # Qwen-Image-2.1 VAE. Not interchangeable with the older Qwen Image / Wan 2.2 VAEs.
  "Comfy-Org/Qwen-Image-2.1 main vae/qwen_image_2.1_vae_bf16.safetensors"
  # Qwen3-VL-8B-Instruct text encoder. The mmproj file is the vision tower,
  # required only for image editing (/v1/images/edits).
  "Qwen/Qwen3-VL-8B-Instruct-GGUF main Qwen3VL-8B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"
  # Z-Image-Turbo denoiser: 6B single-stream DiT distilled for 8 steps at CFG 1,
  # with English and Chinese text rendering as a design goal. Apache-2.0.
  "leejet/Z-Image-Turbo-GGUF main z_image_turbo-Q4_K.gguf"
  # Z-Image text encoder: Qwen3-4B-Instruct-2507, not the Qwen3-VL used by
  # Qwen-Image-2.1. Apache-2.0.
  "unsloth/Qwen3-4B-Instruct-2507-GGUF main Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
  # Z-Image VAE. It is the FLUX.1-schnell autoencoder, mirrored here by Comfy-Org
  # because black-forest-labs/FLUX.1-schnell is gated (needs a logged-in token).
  "Comfy-Org/z_image_turbo main split_files/vae/ae.safetensors"
  # MOSS-Image SFT denoiser (5B, instruction-tuned; Base is the raw counterpart),
  # GGUF quantization by unsloth. Apache-2.0.
  #
  # Q8_0 rather than a 4-bit build: measured on an 8GB card at 1024x1024, Q8 is
  # both faster (~160 s vs ~220 s) and better than Q5_K_M, because 8-bit weights
  # dequantize more cheaply. With --offload-to-cpu the weights live in RAM
  # anyway, so this only needs ~6.1 GB of VRAM at 1 MP. Lighter alternatives,
  # if host RAM or VRAM headroom is ever a problem:
  #   unsloth MOSS-Image-SFT-UD-Q6_K_XL 4.53 GB · Q4_K_M 3.20 GB
  "unsloth/MOSS-Image-SFT-GGUF main MOSS-Image-SFT-Q8_0.gguf"
  # MOSS-Image text encoder: Qwen3.5-9B. Q4_K_M matches the upstream
  # docs/moss_image.md example; Q5_K_M exists but adds 0.9 GB for no measured gain.
  "unsloth/Qwen3.5-9B-GGUF main Qwen3.5-9B-Q4_K_M.gguf"
  # MOSS-Image tokenizer.json. Qwen3.5 uses a different vocabulary from the
  # embedded Qwen tokenizer, so sd-server needs --tokenizer with this file.
  "OpenMOSS-Team/MOSS-Image-Base main tokenizer/tokenizer.json"
  # FLUX.3 Image Klein denoiser (4B distilled). GGUF by unsloth; Q5_K_M keeps
  # detail at 4.2 GB, comfortably inside the 8 GB card with the 4B encoder.
  "unsloth/flux-3-image-klein-GGUF main flux-3-image-klein-Q5_K_M.gguf"
  # FLUX.3 Image Klein BASE: same 4B architecture before distillation, so the file
  # is the same size and it reuses the Klein text encoder and VAE. Offered as a
  # separate model because it trades speed for literal prompt adherence.
  "unsloth/flux-3-image-klein-base-GGUF main flux-3-image-klein-base-Q5_K_M.gguf"
  # FLUX.3 Image Klein text encoder: Qwen3-VL-4B. No mmproj needed: edits use
  # the 'flux2' preset (VLM=No), passing reference images to the DiT via the VAE.
  "Qwen/Qwen3-VL-4B-Instruct-GGUF main Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  # FLUX.2 VAE, shared by MOSS-Image and FLUX.3 Klein. Mirrored by Comfy-Org
  # because black-forest-labs/FLUX.2-dev is gated; byte size matches BFL's
  # ae.safetensors (336211292) exactly.
  "Comfy-Org/MOSS-Image main vae/flux2-vae.safetensors"
)

# Expected sizes in bytes, used as the idempotency check. Verified against the
# Hugging Face Hub API (GET /api/models/<repo>?blobs=true).
declare -A EXPECTED_BYTES=(
  [qwen_image_2.1-Q4_K.gguf]=4197494816
  [Qwen-Image-2.1-Turbo-AD-Q4_K.gguf]=4201694944
  [qwen_image_2.1_vae_bf16.safetensors]=675509688
  [Qwen3VL-8B-Instruct-Q4_K_M.gguf]=5027784800
  [mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf]=752289728
  [z_image_turbo-Q4_K.gguf]=3864250304
  [Qwen3-4B-Instruct-2507-Q4_K_M.gguf]=2497281120
  [ae.safetensors]=335304388
  [MOSS-Image-SFT-Q8_0.gguf]=5482213536
  [Qwen3.5-9B-Q4_K_M.gguf]=5680522464
  [tokenizer.json]=19989325
  [flux-3-image-klein-Q5_K_M.gguf]=4216230816
  [flux-3-image-klein-base-Q5_K_M.gguf]=4195783424
  [Qwen3VL-4B-Instruct-Q4_K_M.gguf]=2497281664
  [flux2-vae.safetensors]=336211292
)

# Returns 0 when the local file exists and matches its expected byte size.
file_is_complete() {
  local name="$1" path="$2"
  local expected="${EXPECTED_BYTES[$name]:-}"
  [[ -f "$path" ]] || return 1
  [[ -z "$expected" ]] && return 0
  local actual
  actual=$(stat -c %s "$path")
  [[ "$actual" == "$expected" ]]
}

for entry in "${DOWNLOADS[@]}"; do
  read -r repo revision paths <<<"$entry"

  pending=()
  for path in $paths; do
    name=$(basename "$path")
    if file_is_complete "$name" "$MODEL_DIR/$name"; then
      echo "present   $name"
    else
      pending+=("$path")
    fi
  done

  if ((${#pending[@]} == 0)); then
    continue
  fi

  echo "fetching  $repo -> ${pending[*]}"
  # --local-dir with flat layout: the service config addresses files by name.
  hf download "$repo" "${pending[@]}" --revision "$revision" --local-dir "$MODEL_DIR" "${DRY_RUN[@]}"

  # `hf download` mirrors the repository subdirectory (the VAE lives under vae/),
  # while the service config addresses weights by bare filename.
  for path in "${pending[@]}"; do
    if [[ "$path" == */* && -f "$MODEL_DIR/$path" ]]; then
      mv -f "$MODEL_DIR/$path" "$MODEL_DIR/$(basename "$path")"
      rmdir -p --ignore-fail-on-non-empty "$MODEL_DIR/$(dirname "$path")" 2>/dev/null || true
    fi
  done
done

if ((${#DRY_RUN[@]})); then
  echo
  echo "dry run complete: nothing was written to $MODEL_DIR"
  exit 0
fi

echo
echo "All weights available in $MODEL_DIR"
du -sh "$MODEL_DIR"
