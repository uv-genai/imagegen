#!/usr/bin/env bash
# Build the vendored stable-diffusion.cpp engine (sd-server + sd-cli).
#
# The engine is pinned to one exact commit rather than a moving branch, because
# Qwen-Image-2.1 sigma handling changed as recently as 2026-10-06 (commit
# c150a6b fixed custom sigma lists) and a silent upgrade must not change the
# pictures this service produces.
#
# The clone is *blobless* (--filter=blob:none): full commit history, file
# contents fetched on checkout. That costs ~1.5 s and ~400 MB and makes the pin
# checkable for any commit. A shallow clone cannot do this — GitHub refuses to
# fetch an arbitrary SHA, so `git fetch --depth 1 origin <sha>` fails once
# upstream has moved past the pin, which is exactly how this script used to
# break for fresh installs.
#
# Each backend gets its own build directory, so `cuda` and `vulkan` can coexist
# and switching between them does not throw away a working engine. `cuda` keeps
# the historic `build/` path because that is what released configs point at;
# every other backend builds into `build-<backend>`.
#
# Usage: scripts/build_engine.sh [cuda|vulkan|cpu]     (default: cuda)
set -euo pipefail

BACKEND="${1:-cuda}"
PINNED_COMMIT="f89d9b13d730eabeede7314ce49dacd18d3c90c2"
REPO_URL="https://github.com/leejet/stable-diffusion.cpp.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/third_party/stable-diffusion.cpp"

die() { echo "error: $*" >&2; exit 1; }

# -DSD_CUDA already sets GGML_CUDA in the engine's CMakeLists, so the GGML option
# is not passed here. One option per backend keeps the cache readable.
case "$BACKEND" in
  cuda)   CMAKE_FLAGS=(-DSD_CUDA=ON);   BUILD_DIR="build" ;;
  vulkan) CMAKE_FLAGS=(-DSD_VULKAN=ON); BUILD_DIR="build-vulkan" ;;
  cpu)    CMAKE_FLAGS=();               BUILD_DIR="build-cpu" ;;
  *) die "unknown backend: $BACKEND (choose cuda, vulkan or cpu)" ;;
esac
BUILD_PATH="$SRC_DIR/$BUILD_DIR"

# Which accelerator a build directory was configured for, read from its CMake
# cache. Returns an empty string when nothing is configured there yet. This is
# checked before configuring because reconfiguring a directory for a different
# backend silently replaces a working engine and forces a full rebuild.
cached_backend() {
  local cache="$1/CMakeCache.txt" option
  [[ -f "$cache" ]] || return 0
  for option in SD_CUDA SD_VULKAN SD_METAL SD_OPENCL SD_HIPBLAS SD_SYCL SD_MUSA; do
    if grep -q "^${option}:BOOL=ON$" "$cache"; then
      # Lower-cased so it compares equal to the $BACKEND argument.
      echo "${option#SD_}" | tr '[:upper:]' '[:lower:]'
      return 0
    fi
  done
  echo "cpu"
}

if ! command -v git >/dev/null; then
  die "git is required to fetch the engine"
fi

if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo "cloning stable-diffusion.cpp (blobless) into $SRC_DIR"
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --filter=blob:none --no-checkout "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"

# Fetch enough history to contain the pin. Fetching all branches and tags works;
# fetching the SHA directly does not (see the header comment).
if ! git cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null; then
  echo "fetching history until ${PINNED_COMMIT:0:12} is available"
  git fetch --filter=blob:none origin '+refs/heads/*:refs/remotes/origin/*' --tags
fi
git cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null || die \
  "pinned commit ${PINNED_COMMIT} is not reachable from any branch on $REPO_URL.
       Upstream history may have been rewritten. Verify a new commit builds and
       passes scripts/smoke_test.py, then update PINNED_COMMIT in this script."

git checkout --quiet --detach "$PINNED_COMMIT"
current="$(git rev-parse HEAD)"
[[ "$current" == "$PINNED_COMMIT" ]] || die "checked out $current, expected $PINNED_COMMIT"

git submodule sync --recursive >/dev/null
git submodule update --init --recursive --filter=blob:none

# An uninitialized submodule surfaces later as a confusing compile error, so
# check now: `git submodule status` prefixes missing entries with '-'.
if git submodule status --recursive | grep -q '^[+-U]'; then
  die "one or more submodules are not at their recorded commit:
$(git submodule status --recursive)"
fi
echo "engine at $(git rev-parse --short HEAD) with submodules in place"

existing="$(cached_backend "$BUILD_PATH")"
if [[ -n "$existing" && "$existing" != "$BACKEND" ]]; then
  die "$BUILD_DIR is configured for $existing, not $BACKEND. Reconfiguring it here
       would replace that engine and rebuild everything. Either keep them apart
       (cuda builds into 'build', the others into 'build-<backend>'), or remove
       the directory you no longer want:
         rm -rf \"$BUILD_PATH\""
fi

echo "building backend=$BACKEND into $BUILD_DIR"
cmake -B "$BUILD_PATH" -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON "${CMAKE_FLAGS[@]}"
cmake --build "$BUILD_PATH" -j"$(nproc)"

echo
echo "built:"
ls -1 "$BUILD_PATH/bin/sd-server" "$BUILD_PATH/bin/sd-cli"
"$BUILD_PATH/bin/sd-server" --version

# Device indices are what --backend and --max-vram refer to, and their order comes
# from loader enumeration, not from speed: a laptop that exposes an iGPU as Vulkan0
# would send a bare `--backend vulkan` to the slow chip. Print them after every
# build so the index used in config is a deliberate choice.
echo
echo "devices this engine can address:"
"$BUILD_PATH/bin/sd-server" --list-devices

echo
echo "to serve it, set in config/service.toml:"
echo "  sd_server_binary = \"../third_party/stable-diffusion.cpp/$BUILD_DIR/bin/sd-server\""
