"""Mnemosyne — Hermes memory provider plugin.

Drop-in for Hermes: copy this directory into
    <hermes-home>/plugins/memory/mnemosyne/
(`hermes plugins` / plugin discovery auto-detects memory providers there),
set `memory.provider: mnemosyne` in config.yaml, and enable:

    MNEMOSYNE_ENABLED=true
    MNEMOSYNE_DATABASE_URL=postgresql://...   # optional: PG persistence
    MNEMOSYNE_TENANT=<tenant-id>              # optional default tenant

What the agent gets:

  prefetch()  — every turn, the engine RETRIEVES and returns a deterministic
                situation brief (0 LLM tokens). No extraction, no embedding
                API, no hallucinated summaries: the model reads what was
                actually stored.
  sync_turn() — user/assistant turns are appended to the event log and
                gated by surprise (novelty in bits). Redundancy reinforces
                the nearest memory instead of duplicating.

Tenant key: Hermes memory providers receive session_id. The Odaro wiring
uses composite keys (`tenant:user`); the provider derives the tenant from
session_id (first ':' segment) or from MNEMOSYNE_TENANT. The mem0
composite-key surface is preserved so existing callers keep working.

The plugin imports cleanly WITHOUT a Hermes install (it falls back to a
minimal MemoryProvider base), so it is unit-testable standalone.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # inside a real Hermes install
    from agent.memory_provider import MemoryProvider as _HermesMemoryProvider
except ImportError:  # standalone testing
    class _HermesMemoryProvider:  # minimal stand-in, same surface
        @property
        def name(self) -> str:  # pragma: no cover
            raise NotImplementedError

        def is_available(self) -> bool:  # pragma: no cover
            raise NotImplementedError

        def initialize(self, session_id: str, **kwargs) -> None:  # pragma: no cover
            raise NotImplementedError

        def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None) -> None:  # pragma: no cover
            raise NotImplementedError

        def prefetch(self, query, *, session_id="") -> str:  # pragma: no cover
            raise NotImplementedError

        def get_tool_schemas(self) -> list[dict]:  # pragma: no cover
            raise NotImplementedError


# ── engine wiring ─────────────────────────────────────────────────────
@dataclass
class _TenantEngine:
    """One tenant's full memory: stores, gate, cortex, recall."""

    tenant_id: str
    vocab: Any
    gate: Any
    events: Any
    exps: Any
    semantic: Any
    markers: Any
    sdm: Any
    recall: Any
    embed: Any
    governance: Any


def _make_engine(tenant_id: str, database_url: str | None = None) -> _TenantEngine:
    """Build a tenant's engine. PG-backed when a database URL is given;
    otherwise in-memory (dev / single-process)."""
    from mnemosyne.affect import SomaticMarkerStore
    from mnemosyne.consolidation import SemanticStore
    from mnemosyne.embed import event_vector
    from mnemosyne.events import SEED_VOCAB, Event, Vocabulary
    from mnemosyne.gate import EntropyGate
    from mnemosyne.governance import Governance, MemoryPolicy
    from mnemosyne.recall import RecallEngine
    from mnemosyne.sdm import SparseDistributedMemory
    from mnemosyne.storage import InMemoryEventStore, InMemoryExperienceStore

    DIM = 1024  # matches supabase vector(1024) + bge-m3 embeddings
    vocab = Vocabulary(SEED_VOCAB)
    gate = EntropyGate(tenant_id, vocab, theta_factor=1.2, min_theta_bits=16)
    if database_url:
        from mnemosyne.pgstore import PgEventStore, PgExperienceStore, PgSemanticStore
        events = PgEventStore(database_url)
        exps = PgExperienceStore(database_url)
        semantic = PgSemanticStore(database_url)
    else:
        events = InMemoryEventStore()
        exps = InMemoryExperienceStore()
        semantic = SemanticStore()
    markers = SomaticMarkerStore()
    sdm = SparseDistributedMemory(DIM, num_locations=2048, neighbors=16,
                                  rng_seed=sum(ord(c) for c in tenant_id) % (2**32))
    gov = Governance(MemoryPolicy())

    def embed(ev: Event):
        return event_vector(ev, vocab, DIM)

    recall = RecallEngine(exps, markers, events, semantic, sdm, vocab, k=3, brief_budget=700)
    return _TenantEngine(tenant_id, vocab, gate, events, exps, semantic, markers, sdm, recall, embed, gov)


def _tenant_of(session_id: str) -> str:
    """Composite key `tenant:user` → tenant; fallback env → 'default'."""
    if session_id and ":" in session_id:
        return session_id.split(":", 1)[0]
    return os.environ.get("MNEMOSYNE_TENANT", "default")


class MnemosyneMemoryProvider(_HermesMemoryProvider):
    """Hermes memory backend powered by the Odaro Memory Engine."""

    def __init__(self) -> None:
        self._engines: dict[str, _TenantEngine] = {}
        self._database_url = os.environ.get("MNEMOSYNE_DATABASE_URL")

    # ── MemoryProvider surface ────────────────────────────────────────
    @property
    def name(self) -> str:
        return "mnemosyne"

    def is_available(self) -> bool:
        return os.environ.get("MNEMOSYNE_ENABLED", "").lower() in ("1", "true", "yes")

    def initialize(self, session_id: str, **kwargs) -> None:
        tenant = kwargs.get("tenant") or _tenant_of(session_id)
        if tenant not in self._engines:
            self._engines[tenant] = _make_engine(tenant, self._database_url)

    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None) -> None:
        """Write the turn into memory: surprise-gated, constitution-filtered,
        0 LLM tokens. Redundant turns reinforce the nearest memory."""
        tenant = _tenant_of(session_id)
        engine = self._engines.setdefault(tenant, _make_engine(tenant, self._database_url))
        for text in (user_content, assistant_content):
            if not text or not str(text).strip():
                continue
            # dynamic tenant lexicon: new tokens join the vocabulary; the
            # entropy gate's adaptive model grows with them (no rebuild)
            for tok in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()):
                if tok not in ("the", "a", "an", "is", "are", "of", "to", "in", "and"):
                    engine.vocab.add(tok)
            from mnemosyne.events import Event
            from mnemosyne.write import process_event
            ev = Event(tenant_id=tenant, type="chat", payload={"text": str(text)[:2000]}, source="chat")
            r = process_event(ev, engine.gate, engine.events, engine.exps, engine.embed, governance=engine.governance)
            if r.stored and r.experience_id:
                # mirror the cortex write (production: write path owns SDM)
                vec = engine.embed(ev)
                from mnemosyne.sdm import binarize
                engine.sdm.write(binarize(vec), vec)

    def prefetch(self, query, *, session_id="") -> str:
        """Deterministic situation brief for the query — the model's memory
        context for this turn, built from stored structure. 0 LLM tokens."""
        tenant = _tenant_of(session_id)
        engine = self._engines.setdefault(tenant, _make_engine(tenant, self._database_url))
        if not query or not str(query).strip():
            return ""
        try:
            text, mode = engine.recall.brief(tenant, str(query)[:500])
            if mode != "extractive":
                return ""
            return text if text.strip() and "experiences:" in text or "somatic markers:" in text else ""
        except Exception:  # noqa: BLE001 — prefetch must never break the turn
            return ""

    def get_tool_schemas(self) -> list[dict]:
        return []

    # ── engine access (tests / admin) ─────────────────────────────────
    def engine(self, tenant: str) -> _TenantEngine | None:
        return self._engines.get(tenant)


def register(ctx):
    """Hermes plugin entrypoint."""
    ctx.register_memory_provider(MnemosyneMemoryProvider())
