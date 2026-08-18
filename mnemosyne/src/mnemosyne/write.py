"""Write path (L0 → L1): append event, gate it, store or reinforce.

Per docs/memory-engine-design.md §4.2:
  1. append event (0 LLM tokens)
  2. measure surprise in bits against the tenant model (entropy gate)
  3. if novel → create experience (sparse signature + salience + refs)
  4. if redundant → reinforce nearest experience (strength, half-life, refs)
  5. teach the tenant model the event (learn) regardless of branch

Phase A keeps this pure and DB-agnostic via the EventStore/ExperienceStore
interfaces; a Postgres+pgvector implementation and the mem0 dual-write
adapter land next.
"""

from __future__ import annotations

from .events import Event, Vocabulary, tokenize
from .gate import EntropyGate
from .governance import Governance
from .storage import EventStore, Experience, ExperienceStore

# Evidence-weighted source salience (design doc §4.1).
SOURCE_WEIGHT = {
    "probe": 1.0,
    "tool": 0.7,
    "agent": 0.4,
    "chat": 0.2,
    "external": 0.8,
}


class WriteResult:
    def __init__(self, event_id: str, experience_id: str | None, stored: bool, surprise_bits: int,
                 rejected: bool = False, reason: str | None = None):
        self.event_id = event_id
        self.experience_id = experience_id  # new experience id if stored
        self.stored = stored
        self.surprise_bits = surprise_bits
        self.rejected = rejected  # governance: event refused at the door
        self.reason = reason      # governance reason

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "experience_id": self.experience_id,
            "stored": self.stored,
            "surprise_bits": self.surprise_bits,
            "rejected": self.rejected,
            "reason": self.reason,
        }


def salience_of(event: Event) -> float:
    """Deterministic evidence-weighted salience (§4.1, arousal placeholder 1.0)."""
    w = SOURCE_WEIGHT.get(event.source, 0.4)
    importance = 1.0
    if event.project_id:
        importance = 1.1
    if event.evidence:
        importance *= 1.2
    outcome_sign = 1.0
    pl = str(event.payload.get("outcome", "")).lower()
    if pl in ("pass", "approved", "success"):
        outcome_sign = 1.3
    elif pl in ("fail", "rejected", "blocked"):
        outcome_sign = 1.2
    return w * importance * outcome_sign


def process_event(
    event: Event,
    gate: EntropyGate,
    events: EventStore,
    experiences: ExperienceStore,
    vector_embed: callable,
    governance: Governance | None = None,
) -> WriteResult:
    """Run one event through the write path. Returns the decision record.

    When a Governance is supplied, the event is constitutionally reviewed
    BEFORE ingestion: rejected events never enter the memory (a memory that
    is never written cannot leak).
    """
    if governance is not None:
        g = governance.filter_event(event)
        if not g.allowed:
            return WriteResult("", None, False, 0, rejected=True, reason=g.reason)

    event_id = events.append(event)
    symbols = tokenize(event, gate.vocab)
    should_store, bits = gate.decide(symbols)

    if should_store:
        vec = list(vector_embed(event))
        exp = Experience(
            tenant_id=event.tenant_id,
            vector=vec,
            salience=salience_of(event),
            refs=[event_id],
        )
        exp_id = experiences.insert(exp)
        gate.learn(symbols)
        gate.record_outcome(bits, stored=True)
        return WriteResult(event_id, exp_id, True, bits)

    # redundant → reinforce the nearest active experience
    vec = list(vector_embed(event))
    near = experiences.nearest(event.tenant_id, vec, k=1)
    if near:
        exp_id, _ = near[0]
        experiences.reinforce(exp_id, 0.15, event_id)
    gate.learn(symbols)
    gate.record_outcome(bits, stored=False)
    return WriteResult(event_id, None, False, bits)
