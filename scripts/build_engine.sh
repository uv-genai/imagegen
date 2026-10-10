#!/usr/bin/env bash
# Build the vendored stable-diffusion.cpp engine (sd-server + sd-cli).
#
# The engine is pinned to a specific commit rather than a moving branch, because
# the Qwen-Image-2.1 sigma handling changed as recently as 2026-10-06 (commit
# c150a6b fixed custom sigma lists) and a silent upgrade must not change the
# pictures the service produces.
#
# Usage: scripts/build_engine.sh [cuda|vulkan|cpu]     (default: cuda)
set -euo pipefail

BACKEND="${1:-cuda}"
PINNED_COMMIT="f89d9b1"
REPO_URL="https://github.com/leejet/stable-diffusion.cpp.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/third_party/stable-diffusion.cpp"

case "$BACKEND" in
  cuda)   CMAKE_FLAGS=(-DSD_CUDA=ON -DGGML_CUDA=ON) ;;
  vulkan) CMAKE_FLAGS=(-DSD_VULKAN=ON) ;;
  cpu)    CMAKE_FLAGS=() ;;
  *) echo "unknown backend: $BACKEND (choose cuda, vulkan or cpu)" >&2; exit 2 ;;
esac

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "cloning stable-diffusion.cpp @ $PINNED_COMMIT"
  git clone --depth 1 --recurse-submodules --shallow-submodules -j4 "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"
# Detach at the pinned commit. The shallow clone above may already be at it; if
# the commit has been garbage collected the fetch fails loudly instead of
# silently building something newer.
if [[ "$(git rev-parse HEAD)" != "$PINNED_COMMIT"* ]]; then
  git fetch --depth 1 origin "$PINNED_COMMIT"
  git checkout --detach FETCH_HEAD
  git submodule update --init --depth 1 --recursive
fi

echo "building engine at $(git describe --tags --always) with backend=$BACKEND"
cmake -B build -DCMAKE_BUILD_TYPE=Release "${CMAKE_FLAGS[@]}"
cmake --build build -j"$(nproc)"

echo
echo "built:"
ls -1 "$SRC_DIR/build/bin"/sd-server "$SRC_DIR/build/bin"/sd-cli
"$SRC_DIR/build/bin"/sd-server --version
