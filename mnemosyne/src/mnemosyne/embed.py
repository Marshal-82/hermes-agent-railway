"""Deterministic embeddings (Phase A stub; production: local bge-m3-class).

Two vectorizers in ONE space, so queries land next to the events they
mention:

  event_vector() — a stored event's memory signature: tokenized type +
                   payload + the PROJECT/entity context (project_id is part
                   of the signature even when the payload text omits it —
                   a manufacturing incident about beta IS a beta memory).
  query_vector()  — a free-text cue, stopword-filtered. The brain does not
                   attend to function words when recalling episodes; neither
                   does the engine. Filtering keeps long natural-language
                   questions from diluting the match.

Both hash symbol ids through md5 into `dim` bins (L2-normalized), so
event and query vectors are directly comparable with cosine.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable

from .events import Event, Vocabulary, tokenize

STOPWORDS = frozenset({
    "what", "do", "we", "know", "about", "the", "of", "is", "are", "for",
    "and", "to", "in", "on", "with", "a", "an", "this", "that", "how",
    "why", "when", "where", "who", "please", "give", "me", "tell",
    "summarize", "brief", "can", "you", "it", "from", "at", "by", "as",
    "project", "status", "update", "query",
})


def _hash(sym: int, dim: int) -> int:
    return int(hashlib.md5(f"f{sym}".encode()).hexdigest(), 16) % dim


def _normalize(v: list[float]) -> list[float]:
    norm = max(1.0, sum(x * x for x in v) ** 0.5)
    return [x / norm for x in v]


def event_vector(event: Event, vocab: Vocabulary, dim: int) -> list[float]:
    """Memory signature for an event (write path)."""
    v = [0.0] * dim
    for sym in tokenize(event, vocab):
        v[_hash(sym, dim)] += 1.0
    if event.project_id:
        v[_hash(vocab.id(event.project_id), dim)] += 1.0
    return _normalize(v)


def query_vector(query: str, vocab: Vocabulary, dim: int) -> list[float]:
    """Cue signature for a free-text query (recall path), stopwords out."""
    v = [0.0] * dim
    for sym in query_token_ids(query, vocab):
        v[_hash(sym, dim)] += 1.0
    return _normalize(v)


def query_token_ids(query: str, vocab: Vocabulary) -> list[int]:
    """Stopword-filtered symbol ids of a query (for entity activation)."""
    probe = Event(tenant_id="__query__", type="query", payload={"text": query})
    return [sym for sym in tokenize(probe, vocab) if vocab.token(sym) not in STOPWORDS]


def query_tokens(query: str, vocab: Vocabulary) -> set[str]:
    """Stopword-filtered TOKEN strings of a query (for entity activation)."""
    return {vocab.token(sym) for sym in query_token_ids(query, vocab)}


# ── pluggable embedder (local bge-m3-class swap) ─────────────────────
EmbedFn = Callable[[Event], list[float]]


def is_local_embedder_available() -> bool:
    """True when sentence-transformers is installed (make_embedder still
    falls back at load time if the specific model cannot be loaded)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def make_embedder(vocab: Vocabulary, dim: int, prefer_local: bool | None = None) -> EmbedFn:
    """The embedding provider seam.

    prefer_local=True (or env MNEMOSYNE_EMBED=bge-m3): a local bge-m3-class
    model (sentence-transformers) embeds events semantically — 1024 dims,
    matching the production pgvector schema. If the model cannot load, the
    engine FALLS BACK to the deterministic hash embedder: memory never
    breaks because a model is missing (and the write path stays 0-cost).
    """
    if prefer_local is None:
        prefer_local = os.environ.get("MNEMOSYNE_EMBED", "").lower() in ("bge-m3", "local", "1", "true")
    if prefer_local:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get("MNEMOSYNE_EMBED_MODEL", "BAAI/bge-m3")
            model = SentenceTransformer(model_name)

            def local_embed(event: Event) -> list[float]:
                text = " ".join([
                    event.type,
                    event.project_id or "",
                    *[str(v) for v in event.payload.values()],
                ])
                vec = model.encode([text], normalize_embeddings=True)[0]
                return [float(x) for x in vec]

            local_embed.__mnemosyne_provider__ = f"local:{model_name}"  # type: ignore[attr-defined]
            return local_embed
        except Exception:  # noqa: BLE001 — model missing → graceful fallback
            pass

    def hash_embed(event: Event) -> list[float]:
        return event_vector(event, vocab, dim)

    hash_embed.__mnemosyne_provider__ = "hash"  # type: ignore[attr-defined]
    return hash_embed
