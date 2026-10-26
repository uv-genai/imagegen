#!/usr/bin/env bash
# Launch the imagegen HTTP service, bound to every interface by default.
#
# `imagegen serve` takes only --config: there are no --host/--port flags, so
# this script passes the settings through the environment overrides that
# src/imagegen/cli.py applies on top of the config file (IMAGEGEN_HOST,
# IMAGEGEN_PORT, IMAGEGEN_API_KEYS). Nothing is written to
# config/service.toml, so the repository stays free of secrets.
#
# Binding 0.0.0.0 without a key would hand every host on the network an
# unauthenticated GPU endpoint, so a key is required unless --host is loopback
# or --allow-unauthenticated is passed. --api-key is accepted as asked, but a
# command-line value is visible in `ps` output and shell history; prefer
# --api-key-env, which reads the same value from a variable set elsewhere.
#
# Usage: scripts/serve.sh [--api-key KEY]... [--api-key-env NAME]...
#                         [--port N] [--host H] [--config PATH]
#                         [--allow-unauthenticated] [--dry-run]
#
#   --api-key KEY        Bearer token accepted on /v1/*. Repeatable.
#   --api-key-env NAME   Take key(s) from environment variable NAME, which may
#                        hold a comma-separated list. Repeatable.
#   --port N             Listen port (default: whatever the config file says).
#   --host H             Bind address (default: 0.0.0.0).
#   --config PATH        Service config (default: config/service.toml).
#   --allow-unauthenticated
#                        Serve with no key on a non-loopback bind. Refused by
#                        default; only for a network you control.
#   --dry-run            Print the effective settings and exit without serving.
#                        Keys are reported as [redacted].
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/config/service.toml"
HOST="0.0.0.0"
PORT=""
DRY_RUN=0
ALLOW_UNAUTH=0
KEY_FROM_ARGV=0
KEYS=()

die() {
  echo "error: $*" >&2
  exit 2
}

add_key() {
  local cleaned="${1//[[:space:]]/}"
  [[ -n "$cleaned" ]] || die "an API key cannot be empty"
  KEYS+=("$cleaned")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key)
      [[ $# -ge 2 ]] || die "--api-key needs a value"
      add_key "$2"
      KEY_FROM_ARGV=1
      shift 2
      ;;
    --api-key=*)
      add_key "${1#*=}"
      KEY_FROM_ARGV=1
      shift
      ;;
    --api-key-env)
      [[ $# -ge 2 ]] || die "--api-key-env needs a variable name"
      env_name="$2"
      [[ -n "${!env_name:-}" ]] || die "--api-key-env: \$$env_name is unset or empty"
      IFS=',' read -r -a from_env <<< "${!env_name}"
      for k in "${from_env[@]}"; do
        add_key "$k"
      done
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port needs a value"
      PORT="$2"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --host)
      [[ $# -ge 2 ]] || die "--host needs a value"
      HOST="$2"
      shift 2
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --config)
      [[ $# -ge 2 ]] || die "--config needs a value"
      CONFIG="$2"
      shift 2
      ;;
    --config=*)
      CONFIG="${1#*=}"
      shift
      ;;
    --allow-unauthenticated)
      ALLOW_UNAUTH=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      die "unknown argument: $1 (see --help)"
      ;;
  esac
done

command -v uv >/dev/null 2>&1 ||
  die "uv is required (https://docs.astral.sh/uv/); it provides the service venv."

[[ -f "$CONFIG" ]] || die "no config file at $CONFIG"

if [[ -n "$PORT" ]]; then
  [[ "$PORT" =~ ^[0-9]+$ ]] && ((PORT >= 1 && PORT <= 65535)) ||
    die "--port must be an integer between 1 and 65535 (got: $PORT)"
fi

case "$HOST" in
  127.0.0.1|localhost|::1) LOOPBACK=1 ;;
  *) LOOPBACK=0 ;;
esac

if [[ ${#KEYS[@]} -eq 0 && $LOOPBACK -eq 0 && $ALLOW_UNAUTH -eq 0 ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "warning: would refuse to bind $HOST with no API key; pass --api-key/--api-key-env or --allow-unauthenticated"
  else
    die "refusing to bind $HOST with no API key: that exposes an unauthenticated
       image endpoint to every host that can reach this machine. Pass
       --api-key/--api-key-env, bind with --host 127.0.0.1, or use
       --allow-unauthenticated if the network is yours to control."
  fi
fi

# Fail here rather than with a bind traceback, and name the squatter.
if command -v ss >/dev/null 2>&1; then
  effective_port="${PORT:-$(awk -F'=' '/^port[[:space:]]*=/ {gsub(/[ "]/, "", $2); print $2; exit}' "$CONFIG")}"
  squatter="$(ss -ltnp 2>/dev/null | awk -v p=":${effective_port}\$" '$4 ~ p {print $NF; exit}')"
  if [[ -n "$squatter" ]]; then
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "warning: port $effective_port is already in use (${squatter:-unknown process})"
    else
      die "port $effective_port is already in use (${squatter:-unknown process})"
    fi
  fi
fi

API_KEYS_JOINED=""
if [[ ${#KEYS[@]} -gt 0 ]]; then
  API_KEYS_JOINED="$(IFS=','; echo "${KEYS[*]}")"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo "dry run: nothing started, nothing written"
  echo "  config     $CONFIG"
  echo "  bind       ${HOST}:${PORT:-from config}"
  echo "  api keys   ${#KEYS[@]} [redacted]"
  echo "  command    IMAGEGEN_HOST=${HOST}${PORT:+ IMAGEGEN_PORT=$PORT}${API_KEYS_JOINED:+ IMAGEGEN_API_KEYS=[redacted]} \\"
  echo "               uv run imagegen serve --config $CONFIG"
  exit 0
fi

export IMAGEGEN_HOST="$HOST"
if [[ -n "$PORT" ]]; then
  export IMAGEGEN_PORT="$PORT"
fi
if [[ -n "$API_KEYS_JOINED" ]]; then
  export IMAGEGEN_API_KEYS="$API_KEYS_JOINED"
fi

echo "imagegen on ${HOST}:${PORT:-from config}, ${#KEYS[@]} api key(s) [redacted]"

if [[ $KEY_FROM_ARGV -eq 1 ]]; then
  echo "note: a key given with --api-key is visible to other users via \`ps\` and" \
       "stays in your shell history. --api-key-env reads it from a variable instead."
fi

if [[ $LOOPBACK -eq 0 ]] &&
   grep -Eq '^[[:space:]]*public_base_url[[:space:]]*=[[:space:]]*"https?://(127\.0\.0\.1|localhost)' "$CONFIG"; then
  echo "note: public_base_url in $CONFIG still points at loopback. It has no" \
       "environment override, so response_format=\"url\" replies hand remote clients" \
       "an address that resolves on their own machine. Edit it to the address they" \
       "actually use, or serve b64_json instead."
fi

cd "$REPO_ROOT"
exec uv run imagegen serve --config "$CONFIG"
