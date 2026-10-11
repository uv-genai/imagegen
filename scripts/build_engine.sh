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
# Usage: scripts/build_engine.sh [cuda|vulkan|cpu]     (default: cuda)
set -euo pipefail

BACKEND="${1:-cuda}"
PINNED_COMMIT="f89d9b13d730eabeede7314ce49dacd18d3c90c2"
REPO_URL="https://github.com/leejet/stable-diffusion.cpp.git"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT/third_party/stable-diffusion.cpp"

die() { echo "error: $*" >&2; exit 1; }

case "$BACKEND" in
  cuda)   CMAKE_FLAGS=(-DSD_CUDA=ON -DGGML_CUDA=ON) ;;
  vulkan) CMAKE_FLAGS=(-DSD_VULKAN=ON) ;;
  cpu)    CMAKE_FLAGS=() ;;
  *) die "unknown backend: $BACKEND (choose cuda, vulkan or cpu)" ;;
esac

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

echo "building with backend=$BACKEND"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_EXAMPLES=ON "${CMAKE_FLAGS[@]}"
cmake --build build -j"$(nproc)"

echo
echo "built:"
ls -1 "$SRC_DIR/build/bin/sd-server" "$SRC_DIR/build/bin/sd-cli"
"$SRC_DIR/build/bin/sd-server" --version
