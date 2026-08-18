#!/bin/bash
# Start both gateway (background) and dashboard (foreground) for Railway
set -e

# As of Nous Research's July 2026 security update, HERMES_DASHBOARD_INSECURE / --insecure
# is a deprecated no-op, and the dashboard refuses to start on a non-loopback bind (0.0.0.0,
# required so Railway can route to it) without an auth provider configured. Bootstrap basic
# auth credentials on first boot if the deployer hasn't set their own via Railway variables,
# and persist them to the data volume so they survive restarts/redeploys instead of rotating
# every time the container starts.
export HERMES_DASHBOARD_BASIC_AUTH_USERNAME="${HERMES_DASHBOARD_BASIC_AUTH_USERNAME:-admin}"

DATA_DIR="${HERMES_HOME:-/opt/data}"
mkdir -p "$DATA_DIR"

# ── Mnemosyne memory provider ─────────────────────────────────────────
# Install the plugin into the Hermes home (plugins are auto-discovered from
# <hermes-home>/plugins/memory/<name>/) and expose the engine import path.
# Activation is the MNEMOSYNE_ENABLED Railway variable — the mem0-off switch.
mkdir -p "$DATA_DIR/plugins/memory"
if [ -d /opt/hermes/mnemosyne-plugin ]; then
  cp -r /opt/hermes/mnemosyne-plugin "$DATA_DIR/plugins/memory/mnemosyne"
fi
if [ -d /opt/hermes/mnemosyne/src ]; then
  export PYTHONPATH="/opt/hermes/mnemosyne/src:/opt/hermes/mnemosyne-plugin${PYTHONPATH:+:$PYTHONPATH}"
fi
# psycopg for the engine's Postgres adapter. The gateway service runs as
# the `hermes` user and cannot write into the root-owned venv, so pip/
# ensurepip can't bootstrap there. Instead install into a site dir on the
# hermes-writable /opt/data volume with uv (present in the image) and add
# it to PYTHONPATH. NEVER fatal: the provider logs a clear error at use-time.
V="/opt/hermes/.venv/bin/python"
SITE="/opt/data/mnemosyne-site"
export PYTHONPATH="$SITE${PYTHONPATH:+:$PYTHONPATH}"
if ! "$V" -c "import psycopg" 2>/dev/null; then
  mkdir -p "$SITE" 2>/dev/null || true
  uv pip install --python "$V" --target "$SITE" --no-cache "psycopg[binary]>=3.1" 2>&1 | tail -3 || true
fi
if "$V" -c "import psycopg" 2>/dev/null; then
  echo "[hermes-agent-railway] psycopg ready"
else
  echo "[hermes-agent-railway] WARNING psycopg missing — Mnemosyne Postgres path unavailable"
fi
if [ "${MNEMOSYNE_ENABLED:-false}" = "true" ]; then
  hermes config set memory.provider mnemosyne
  echo "[hermes-agent-railway] Mnemosyne memory provider ACTIVE (memory.provider=mnemosyne)"
fi

# persist_or_generate <env-var-name> <file-name> <openssl-args...>
# If the deployer already set the var (Railway variable), use it as-is and don't touch
# the file. Otherwise reuse a previously-generated value from the volume if one exists,
# or generate a fresh one and persist it so it's stable across restarts/redeploys.
persist_or_generate() {
  local var_name="$1" file_name="$2"; shift 2
  local file_path="$DATA_DIR/$file_name"
  if [ -n "${!var_name}" ]; then
    return
  fi
  if [ -f "$file_path" ]; then
    export "$var_name=$(cat "$file_path")"
  else
    local generated
    generated="$(openssl rand "$@")"
    export "$var_name=$generated"
    echo "$generated" > "$file_path"
    chmod 600 "$file_path"
  fi
}

persist_or_generate HERMES_DASHBOARD_BASIC_AUTH_PASSWORD .dashboard_auth_password -hex 12
persist_or_generate HERMES_DASHBOARD_BASIC_AUTH_SECRET .dashboard_auth_secret -hex 16

echo "[hermes-agent-railway] Dashboard login — username: $HERMES_DASHBOARD_BASIC_AUTH_USERNAME  password: $HERMES_DASHBOARD_BASIC_AUTH_PASSWORD"
echo "[hermes-agent-railway] To set your own, add HERMES_DASHBOARD_BASIC_AUTH_PASSWORD as a Railway variable."

# Gateway handles messaging channels (Telegram, Discord, etc.)
#hermes gateway run &

# Dashboard WebUI — bind to 0.0.0.0 so Railway can route to it
hermes config set platforms.api_server.host 0.0.0.0
hermes config set platforms.api_server.port $PORT
exec hermes gateway run --accept-hooks
