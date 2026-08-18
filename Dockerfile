FROM nousresearch/hermes-agent@sha256:7ecc5d71b25658dc9fb8b78773c49302fd2708ee973f641d6c42bae05e01119d

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
COPY mnemosyne-plugin/ /opt/hermes/mnemosyne-plugin/
# The base image's Python lives in the Hermes venv; `pip` is not on the
# root PATH at build time, so call the venv's pip explicitly.
RUN /opt/hermes/.venv/bin/pip install --no-cache-dir "psycopg[binary]>=3.1"

# Copy start script
COPY start.sh /opt/hermes/start.sh
RUN chmod +x /opt/hermes/start.sh

EXPOSE 9119

# Use the original entrypoint (activates venv, sets up config)
# Pass our start script as the command — entrypoint will exec it since it's on PATH
CMD ["bash", "/opt/hermes/start.sh"]
