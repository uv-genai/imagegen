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

case "$cmd" in
    serve)
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
