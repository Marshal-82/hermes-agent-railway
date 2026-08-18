"""Event model, tokenizer, and vocabulary for Mnemosyne.

Events are the append-only source of truth (L0). Experiences (episodic
tier) are projections over events. This module defines the event record and
the deterministic tokenization that feeds the entropy gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Fixed vocabulary seed — event types + common field/entity tokens.
# Anything unseen at tokenize time maps to UNKNOWN (index 0 reserved).
SEED_VOCAB = [
    "UNKNOWN",
    "chat", "task", "codegen", "docgen", "review_gate", "go_live",
    "deploy", "test", "probe", "incident", "session", "project",
    "tenant", "user", "vendor", "integration", "suitelet", "restlet",
    "sdf", "suite_script", "map_reduce", "workflow", "saved_search",
    "pass", "fail", "blocked", "approved", "rejected", "error",
    "retry", "timeout", "governance", "cost", "evidence", "outcome",
]

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Event:
    """An immutable observation. Never edited in place (L0)."""

    tenant_id: str
    type: str
    payload: dict
    project_id: str | None = None
    session_id: str | None = None
    source: str = "agent"
    evidence: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Vocabulary:
    """Bidirectional token <-> symbol mapping. Fixed size for the coder.

    Symbols 0..n-1 map to SEED_VOCAB entries; index 0 is UNKNOWN. Unseen
    tokens tokenize to UNKNOWN. Phase B replaces this with a dynamic vocab
    rebuilt during consolidation.
    """

    def __init__(self, seed: list[str] | None = None):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []
        for token in (seed or SEED_VOCAB):
            self.add(token)

    @property
    def size(self) -> int:
        return len(self._id_to_token)

    def add(self, token: str) -> int:
        tok = token.lower()
        if tok not in self._token_to_id:
            self._token_to_id[tok] = len(self._id_to_token)
            self._id_to_token.append(tok)
        return self._token_to_id[tok]

    def id(self, token: str) -> int:
        return self._token_to_id.get(token.lower(), 0)  # 0 = UNKNOWN

    def token(self, sym: int) -> str:
        if 0 <= sym < len(self._id_to_token):
            return self._id_to_token[sym]
        return "UNKNOWN"


def tokenize(event: Event, vocab: Vocabulary, max_tokens: int = 64) -> list[int]:
    """Deterministic token stream for an event, as symbol ids.

    Structure: [type] + payload field/value tokens + outcome/evidence flags.
    Content beyond max_tokens is truncated — the gate only needs the signal,
    not the full fidelity (eidetic tier holds the raw payload).
    """
    raw: list[str] = [event.type]
    for key in sorted(event.payload.keys()):
        value = event.payload[key]
        raw.append(f"{key}")
        if isinstance(value, str):
            raw.extend(_TOKEN_RE.findall(value.lower()))
        elif isinstance(value, (int, float, bool)):
            raw.append(f"{key}_{value}")

    if event.evidence:
        raw.append("evidence")
        for key in sorted(event.evidence.keys()):
            raw.append(f"ev_{key}")

    symbols: list[int] = []
    for tok in raw[:max_tokens]:
        symbols.append(vocab.id(tok))
    if not symbols:
        symbols.append(0)  # UNKNOWN — an empty event still carries a signal
    return symbols
