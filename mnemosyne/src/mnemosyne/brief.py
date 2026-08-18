"""Recall path (§4.3) — the situation brief.

The ONLY LLM inference in the engine's read path: retrieved experiences +
somatic markers are synthesized into a budget-bounded brief. Without an API
key the brief falls back to a deterministic extractive summary, so the
engine — and the standalone proof — runs anywhere. The LLM is a pluggable
READER, never the memory.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .affect import SomaticMarker
from .storage import Experience


def _resolve_deepseek_key() -> str | None:
    """Find a DeepSeek API key: env var, then ~/.hermes/.env (Hermes home)."""
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MNEMOSYNE_LLM_KEY")
    if key:
        return key
    for env_path in (os.path.expanduser("~/.hermes/.env"), os.path.expanduser("~/.hermes-netsuite-agency/.env")):
        try:
            with open(env_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("DEEPSEEK_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return None


def llm_complete(prompt: str, model: str = "deepseek-chat", max_tokens: int = 800) -> str | None:
    """OpenAI-compatible completion against DeepSeek. Returns None on failure."""
    key = _resolve_deepseek_key()
    if not key:
        return None
    base = os.environ.get("MNEMOSYNE_LLM_URL", "https://api.deepseek.com/v1")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]
    except Exception:  # noqa: BLE001 — network/API failure → caller falls back
        return None


def _extractive_brief(experiences: list[Experience], markers: list[SomaticMarker]) -> str:
    """Deterministic fallback: no LLM. Uses structure only."""
    lines = ["[extractive brief — no LLM used]"]
    for m in markers:
        lines.append(f"- marker {m.summary()['entity']}: risk={m.risk}, trust={m.trust}, "
                     f"evidence={m.evidence_count}, feelings={m.feelings[-5:]}")
    if experiences:
        lines.append(f"- experiences retrieved: {len(experiences)}")
        for exp in experiences[:5]:
            lines.append(f"  - salience={exp.salience:.2f} strength={exp.strength:.2f} "
                         f"state={exp.state} refs={len(exp.refs)}")
    return "\n".join(lines)


def synthesize_brief(
    query: str,
    experiences: list[Experience],
    markers: list[SomaticMarker],
    budget: int = 600,
    use_llm: bool = True,
) -> tuple[str, str]:
    """Return (brief_text, mode) where mode is 'llm' or 'extractive'."""
    exp_lines = []
    for exp in experiences[:8]:
        exp_lines.append(
            f"- experience salience={exp.salience:.2f} strength={exp.strength:.2f} "
            f"state={exp.state} refs={len(exp.refs)} feeling={exp.feeling.get('label', 'n/a')}"
        )
    marker_lines = []
    for m in markers[:8]:
        marker_lines.append(
            f"- marker {m.entity_type}:{m.entity_id} counts={m.counts} "
            f"calibrated_risk={m.calibrated_risk} state={m.recovery_state} "
            f"trust={m.trust} evidence={m.evidence_count}"
        )

    if use_llm:
        prompt = (
            "You are the memory readout of an AI agency agent. Given the retrieved "
            "experiences and somatic markers below, write a concise situation brief "
            f"(max {budget} tokens) for the question: {query}\n\n"
            "EXPERIENCES:\n" + ("\n".join(exp_lines) or "(none)") +
            "\n\nSOMATIC MARKERS:\n" + ("\n".join(marker_lines) or "(none)") +
            "\n\nBrief:"
        )
        text = llm_complete(prompt)
        if text:
            return text.strip(), "llm"

    return _extractive_brief(experiences, markers), "extractive"
