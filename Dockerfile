FROM nousresearch/hermes-agent@sha256:23d7fdefc42ef4f874938835dcc9543468b45c3fe082415095ab48056c56c32a

USER root

ENV HERMES_HOME=/opt/data
ENV PORT=9119
# Nous Research's July 2026 security update refuses to start `hermes gateway run` as
# root unless explicitly allowed (this image already ran as USER root before that
# change). Opting in to their sanctioned override rather than switching users.
ENV HERMES_ALLOW_ROOT_GATEWAY=1

# ── Mnemosyne memory provider (Odaro Memory Engine) ────────────────────
# The plugin is baked into the image and installed into HERMES_HOME/plugins
# by start.sh on boot (HERMES_HOME is a volume, so this survives redeploys).
# Activate with MNEMOSYNE_ENABLED=true (Railway variable) — the mem0-off
# switch. The engine import path is provided via PYTHONPATH in start.sh.
COPY mnemosyne/ /opt/hermes/mnemosyne/
# Bundle the plugin like the first-party memory providers (mem0, honcho…):
# Hermes auto-detects memory providers from /opt/hermes/plugins/memory/.
COPY mnemosyne-plugin/ /opt/hermes/plugins/memory/mnemosyne/
# psycopg is installed at RUNTIME in start.sh: the base image creates its
# venv at container start (it does not exist during docker build), so the
# build-time pip install cannot target the environment the gateway uses.

# Copy start script
COPY start.sh /opt/hermes/start.sh
RUN chmod +x /opt/hermes/start.sh

EXPOSE 9119

# Use the original entrypoint (activates venv, sets up config)
# Pass our start script as the command — entrypoint will exec it since it's on PATH
CMD ["bash", "/opt/hermes/start.sh"]
