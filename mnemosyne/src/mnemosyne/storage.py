"""Storage interfaces + in-memory implementation (Phase A prototype).

Production stores are Supabase Postgres + pgvector (see
supabase/migrations/0001_mnemosyne_schema.sql). The in-memory implementation
exists so the gate + write path are testable and runnable without a DB, and
so Phase A can dual-write against mem0 with a drop-in adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .events import Event


@dataclass
class Experience:
    """Episodic-tier memory: an index + signature + stats over events (L1).

    The payload lives in the events table (hippocampal index analog); the
    experience row is the pointer, the sparse signature (glyph), salience,
    and decay/reinforcement statistics.
    """

    tenant_id: str
    vector: list[float]
    signature: str = ""
    salience: float = 1.0
    strength: float = 1.0
    half_life_hours: float = 24.0 * 30
    appraisal: dict = field(default_factory=dict)
    feeling: dict = field(default_factory=dict)
    refs: list[str] = field(default_factory=list)
    state: str = "active"  # active | cold | superseded | archived
    access_count: int = 0
    last_access_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # reconsolidation lineage: this experience superseded the ids in
    # `supersedes` (older), and was itself replaced by `superseded_by`.
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None


class EventStore:
    """Append-only event log (L0)."""

    def append(self, event: Event) -> str:
        raise NotImplementedError

    def get(self, event_id: str) -> Event | None:
        raise NotImplementedError

    def delete(self, event_id: str) -> None:
        """Right-to-be-forgotten: remove an event row."""
        raise NotImplementedError

    def list_all(self, tenant_id: str) -> list[tuple[str, Event]]:
        raise NotImplementedError


class ExperienceStore:
    """Episodic buffer (L1)."""

    def insert(self, exp: Experience) -> str:
        raise NotImplementedError

    def reinforce(self, experience_id: str, strength_delta: float, ref: str) -> None:
        raise NotImplementedError

    def nearest(self, tenant_id: str, vector: list[float], k: int = 5) -> list[tuple[str, "Experience"]]:
        raise NotImplementedError

    def get(self, experience_id: str) -> Experience | None:
        raise NotImplementedError

    def delete(self, experience_id: str) -> None:
        """Right-to-be-forgotten: remove an experience row."""
        raise NotImplementedError

    def list_active(self, tenant_id: str) -> list[tuple[str, "Experience"]]:
        raise NotImplementedError

    def list_all(self, tenant_id: str) -> list[tuple[str, "Experience"]]:
        """Every row for a tenant regardless of state (forget cascade)."""
        raise NotImplementedError

    def update_state(self, experience_id: str, state: str) -> None:
        raise NotImplementedError


class InMemoryEventStore(EventStore):
    def __init__(self):
        self._rows: dict[str, Event] = {}
        self._seq = 0

    def append(self, event: Event) -> str:
        self._seq += 1
        eid = f"ev_{self._seq}"
        self._rows[eid] = event
        return eid

    def get(self, event_id: str) -> Event | None:
        return self._rows.get(event_id)

    def delete(self, event_id: str) -> None:
        self._rows.pop(event_id, None)

    def list_all(self, tenant_id: str) -> list[tuple[str, Event]]:
        return [(eid, ev) for eid, ev in self._rows.items() if ev.tenant_id == tenant_id]


class InMemoryExperienceStore(ExperienceStore):
    def __init__(self):
        self._rows: dict[str, Experience] = {}
        self._seq = 0

    def insert(self, exp: Experience) -> str:
        self._seq += 1
        eid = f"exp_{self._seq}"
        self._rows[eid] = exp
        return eid

    def reinforce(self, experience_id: str, strength_delta: float, ref: str) -> None:
        exp = self._rows.get(experience_id)
        if exp is None:
            return
        exp.strength += strength_delta
        exp.refs.append(ref)
        exp.last_access_at = datetime.now(timezone.utc)

    def nearest(self, tenant_id: str, vector: list[float], k: int = 5) -> list[tuple[str, Experience]]:
        scored: list[tuple[float, str, Experience]] = []
        for eid, exp in self._rows.items():
            if exp.tenant_id != tenant_id or exp.state != "active":
                continue
            scored.append((_cosine(vector, exp.vector), eid, exp))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(eid, exp) for _, eid, exp in scored[:k]]

    def get(self, experience_id: str) -> Experience | None:
        return self._rows.get(experience_id)

    def delete(self, experience_id: str) -> None:
        self._rows.pop(experience_id, None)

    def list_active(self, tenant_id: str) -> list[tuple[str, Experience]]:
        return [(eid, exp) for eid, exp in self._rows.items()
                if exp.tenant_id == tenant_id and exp.state == "active"]

    def list_all(self, tenant_id: str) -> list[tuple[str, Experience]]:
        return [(eid, exp) for eid, exp in self._rows.items() if exp.tenant_id == tenant_id]

    def update_state(self, experience_id: str, state: str) -> None:
        exp = self._rows.get(experience_id)
        if exp is not None:
            exp.state = state


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
