# Hermes Agent — Self-Improving AI Agent on Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/TEMPLATE_CODE)

Deploy [Hermes Agent](https://hermes-agent.nousresearch.com/) by Nous Research on Railway with one click. An open-source, self-improving AI agent with persistent memory, 200+ LLM model support, and multi-channel messaging — self-hosted with full data ownership.

## What's Included

| Service | Image | Purpose |
|---|---|---|
| **Hermes Agent** | `nousresearch/hermes-agent` | AI agent gateway + WebUI dashboard |

## Features

- **Self-Improving** — Creates skills from experience and improves them over time
- **Persistent Memory** — Remembers conversations and builds a model of who you are across sessions via Honcho
- **200+ LLM Models** — OpenRouter, Nous Portal, NVIDIA NIM, OpenAI, Anthropic, custom endpoints
- **Multi-Channel** — Telegram, Discord, Slack, WhatsApp, WeChat messaging
- **40+ Built-in Tools** — Web browsing, code execution, file management, and more
- **Cron Scheduling** — Automate recurring tasks without supervision
- **WebUI Dashboard** — Configure and manage your agent from the browser

## How to Deploy

1. Click the "Deploy on Railway" button above
2. Set your LLM API key (OpenRouter, OpenAI, etc.) in Variables
3. Open the generated URL — you'll hit a login screen (see **Dashboard Login** below)
4. Connect messaging channels (Telegram, Discord, etc.)

## Dashboard Login

As of Nous Research's July 2026 update, the dashboard requires a login — it can no longer run fully open. On first boot, this template auto-generates a username/password and prints them to the deploy logs (Railway dashboard → Deployments → View Logs), e.g.:

```
[hermes-agent-railway] Dashboard login — username: admin  password: <generated>
```

This password is saved to the persistent volume, so it stays the same across restarts/redeploys — it won't rotate on you. To set your own instead of using the generated one, add `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` as a Railway variable before first deploy.

## Environment Variables

| Variable | Description |
|---|---|
| `PORT` | Dashboard port (default: 9119) |
| `HERMES_HOME` | Data directory (default: /opt/data) |
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | Dashboard login username. Defaults to `admin`. |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | Dashboard login password. Auto-generated and persisted on first boot if not set. |

Configure LLM providers and messaging channels via the WebUI dashboard after deployment.

## Estimated Cost

~$5-10/month on Railway for the agent container. LLM API costs are separate and depend on your chosen provider and usage.

## Notes

- **Persistent storage**: Volume at `/opt/data` stores configuration, memories, skills, and databases
- **Security**: The dashboard is login-gated by default (see Dashboard Login above) — Nous Research made this mandatory in July 2026 after unauthenticated public dashboards were used as an attack entry point
- **Upgrades**: This image is pinned by digest for reproducibility. Update the `FROM` line's digest in the Dockerfile to upgrade to a newer Hermes Agent release

## License

Hermes Agent is licensed under [MIT](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE).
