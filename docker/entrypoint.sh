#!/bin/sh
# imagegen container entrypoint.
#
# Selects the compute backend at start rather than at build time, then runs the
# requested command. The config always points at
#   /opt/imagegen/engine/current/bin/sd-server
# and this script repoints that symlink, so switching CUDA <-> Vulkan needs neither
# a rebuild nor a config edit.
set -eu

ENGINE_ROOT=/opt/imagegen/engine
BACKEND="${IMAGEGEN_BACKEND:-cuda}"
CONFIG="${IMAGEGEN_CONFIG:-/etc/imagegen/config.toml}"

case "$BACKEND" in
    cuda|vulkan) ;;
    *)
        echo "imagegen: IMAGEGEN_BACKEND must be 'cuda' or 'vulkan', got '$BACKEND'" >&2
        exit 2
        ;;
esac

if [ ! -x "$ENGINE_ROOT/$BACKEND/bin/sd-server" ]; then
    echo "imagegen: no engine installed for backend '$BACKEND'" >&2
    exit 1
fi

ln -sfn "$ENGINE_ROOT/$BACKEND" "$ENGINE_ROOT/current"

cmd="${1:-serve}"
if [ "$#" -gt 0 ]; then
    shift
fi

# A fresh volume has no weights, and the gateway validates them at startup -- so
# without this check the container just crash-loops and the cause scrolls away.
# Say what to run instead, before anything else.
preflight_weights() {
    if [ -n "$(find /models -maxdepth 1 -name '*.gguf' -print -quit 2>/dev/null)" ]; then
        return 0
    fi
    cat >&2 <<'MSG'
imagegen: no weights in /models.

Populate the volume first (about 14.8 GB, resumable):

    docker compose run --rm fetch-models

or, with docker directly:

    docker run --rm -v imagegen-models:/models imagegen:latest fetch-models

If your weights live on the host instead, point the mount at them:

    IMAGEGEN_MODELS_DIR=/path/to/models docker compose up -d
MSG
    exit 1
}

case "$cmd" in
    serve)
        preflight_weights
        echo "imagegen: backend=$BACKEND, engine=$("$ENGINE_ROOT/current/bin/sd-server" --version 2>/dev/null | head -1)"
        exec imagegen serve --config "$CONFIG"
        ;;
    doctor)
        exec imagegen doctor --config "$CONFIG" "$@"
        ;;
    models)
        exec imagegen models --config "$CONFIG" "$@"
        ;;
    plan)
        exec imagegen plan --config "$CONFIG" "$@"
        ;;
    fetch-models)
        # Populates the /models volume. Safe to re-run: existing files of the
        # expected size are skipped.
        exec /opt/imagegen/scripts/fetch_models.sh /models "$@"
        ;;
    *)
        exec "$cmd" "$@"
        ;;
esac
