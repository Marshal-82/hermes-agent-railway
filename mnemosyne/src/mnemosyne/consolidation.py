"""Phase B: sleep consolidation — merge, supersede, synthesize, archive.

The hippocampus replays episodes during sleep; the neocortex gradually
extracts regularities (systems consolidation). Mnemosyne's Consolidator is
that replay, and it is the engine's marquee differentiator: memories are
not just written and forgotten — they are CONSOLIDATED.

  sleep()  — the nightly full pass: decay, cluster, merge duplicates,
             supersede contradictions (lineage preserved), synthesize
             semantic facts (deterministic extractive by default; an LLM
             hook is available but the write path stays 0-LLM), aggregate
             somatic markers, archive decayed rows (recoverable).
  nap()    — a lighter pass triggered by review-gate pass / go-live / phase
             end (the sharp-wave-ripple analog): merge + supersede only.
             Naps NEVER archive and NEVER synthesize — high-value sessions
             consolidate immediately without disturbing the long-term curve.

Everything is per-tenant: a consolidation run on tenant A never touches
tenant B's rows or markers (isolation is structural, not incidental).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

from .affect import MarkerStoreProtocol, SomaticMarkerStore
from .events import Event
from .sdm import SparseDistributedMemory
from .storage import Experience, ExperienceStore, EventStore, _cosine


class SemanticStoreProtocol(Protocol):
    """The semantic-tier surface both in-memory and Postgres implement."""

    def insert(self, fact: "SemanticFact") -> int: ...

    def list(self, tenant_id: str) -> list["SemanticFact"]: ...

    def delete(self, tenant_id: str, entity: str) -> int: ...


def decay_strength(exp: Experience, now: datetime | None = None) -> float:
    """Ebbinghaus decay: S(t) = S0 · exp(−t/λ), λ = half_life / ln2.

    Low-salience experiences carry shorter half-lives, so noise fades fast
    while salient, reinforced memories persist (Schacter's transience as
    adaptive filtration — forgetting is a feature, not a bug).
    """
    now = now or datetime.now(timezone.utc)
    age_h = max(0.0, (now - exp.created_at).total_seconds() / 3600.0)
    lam = exp.half_life_hours / math.log(2.0)
    return exp.strength * math.exp(-age_h / lam)


@dataclass
class SemanticFact:
    """A regularity extracted from ≥N episodes (neocortical gist)."""

    tenant_id: str
    entity: str
    pattern: str
    counts: dict
    source_memory_ids: list[str]


class SemanticStore:
    """Semantic tier (L2): facts with source-memory lineage. In-memory."""

    def __init__(self) -> None:
        self._facts: list[SemanticFact] = []

    def insert(self, fact: SemanticFact) -> int:
        self._facts.append(fact)
        return len(self._facts) - 1

    def list(self, tenant_id: str) -> list[SemanticFact]:
        return [f for f in self._facts if f.tenant_id == tenant_id]

    def delete(self, tenant_id: str, entity: str) -> int:
        """Right-to-be-forgotten: remove facts about an entity."""
        kept = [f for f in self._facts if not (f.tenant_id == tenant_id and f.entity == entity)]
        removed = len(self._facts) - len(kept)
        self._facts = kept
        return removed


@dataclass
class TenantModel:
    """Per-tenant consolidation state — the model of a tenant's memory.

    Updated at the end of every sleep pass (§4.4): the engine's view of how
    big, how strong, how organized, and how expensive this tenant's memory
    is. The falling cost curve shows here: a mature tenant stores less and
    recalls better.
    """

    tenant_id: str
    events_total: int = 0
    experiences_active: int = 0
    semantic_facts: int = 0
    markers: int = 0
    avg_strength: float = 0.0
    avg_salience: float = 0.0
    cortex_capacity: float = 0.0
    cortex_written: int = 0

    def summary(self) -> dict:
        return {
            "tenant": self.tenant_id,
            "events": self.events_total,
            "experiences": self.experiences_active,
            "facts": self.semantic_facts,
            "markers": self.markers,
            "avg_strength": round(self.avg_strength, 3),
            "avg_salience": round(self.avg_salience, 3),
            "cortex_capacity": round(self.cortex_capacity, 3),
            "cortex_written": self.cortex_written,
        }


@dataclass
class SleepReport:
    tenant_id: str
    scanned: int = 0
    merged: int = 0          # duplicate rows absorbed (refs unioned)
    superseded: int = 0      # contradictions superseded (lineage kept)
    synthesized: int = 0     # semantic facts extracted
    archived: int = 0        # decayed below floor, moved to recoverable
    markers_aggregated: int = 0
    facts: list[SemanticFact] = field(default_factory=list)


def _cluster(rows: list[tuple[str, Experience]], eps: float) -> list[list[tuple[str, Experience]]]:
    """Greedy agglomerative clustering on embeddings (deterministic).

    Strongest experiences seed clusters; each row joins the closest centroid
    when cosine ≥ eps, else starts a new cluster. eps is per-tenant policy.
    """
    clusters: list[list[tuple[str, Experience]]] = []
    centroids: list[list[float]] = []
    for row in sorted(rows, key=lambda r: r[1].strength, reverse=True):
        eid, exp = row
        best, best_cos = -1, eps
        for ci, cen in enumerate(centroids):
            c = _cosine(exp.vector, cen)
            if c >= best_cos:
                best, best_cos = ci, c
        if best >= 0:
            clusters[best].append(row)
            n = len(clusters[best])
            cen = centroids[best]
            centroids[best] = [(v * (n - 1) + exp.vector[i]) / n for i, v in enumerate(cen)]
        else:
            clusters.append([row])
            centroids.append(list(exp.vector))
    return clusters


class Consolidator:
    """Runs the consolidation ("sleep") cycle for one tenant at a time."""

    def __init__(
        self,
        store: ExperienceStore,
        markers: SomaticMarkerStore,
        events_store: EventStore,
        semantic: SemanticStore | None = None,
        eps: float = 0.86,
        merge_eps: float = 0.93,
        min_episodes: int = 3,
        archive_floor: float = 0.12,
        synthesizer: Callable[[str, str, dict, list[str]], str] | None = None,
        sdm: SparseDistributedMemory | None = None,
    ) -> None:
        self.store = store
        self.markers: MarkerStoreProtocol = markers
        self.events_store = events_store
        self.semantic: SemanticStoreProtocol = semantic if semantic is not None else SemanticStore()
        self.eps = eps
        self.merge_eps = merge_eps
        self.min_episodes = min_episodes
        self.archive_floor = archive_floor
        # pluggable LLM synthesizer; default = deterministic extractive
        self.synthesizer = synthesizer
        self.sdm = sdm
        self.tenant_models: dict[str, TenantModel] = {}

    # ── public API ────────────────────────────────────────────────────
    def sleep(self, tenant_id: str, now: datetime | None = None) -> SleepReport:
        """Nightly full pass: decay → cluster → merge/supersede/synthesize
        → aggregate markers → archive decayed rows."""
        now = now or datetime.now(timezone.utc)
        rep = SleepReport(tenant_id=tenant_id)
        active = self.store.list_active(tenant_id)
        rep.scanned = len(active)

        # 1. decay (Ebbinghaus)
        for _, exp in active:
            exp.strength = decay_strength(exp, now)

        # 2+3. cluster, then merge/supersede/synthesize per cluster
        for cluster in _cluster(active, self.eps):
            self._process_cluster(cluster, rep, full=True)

        # 4. aggregate somatic markers per entity
        rep.markers_aggregated = len(self.markers.aggregate(tenant_id))

        # 5. archive decayed rows (recoverable — never deleted)
        for eid, exp in self.store.list_active(tenant_id):
            if exp.strength < self.archive_floor:
                self.store.update_state(eid, "archived")
                rep.archived += 1
        # 6. update the tenant model (new centroid, entropy, cortex state)
        self.tenant_models[tenant_id] = self._build_tenant_model(tenant_id)
        return rep

    def nap(self, tenant_id: str) -> SleepReport:
        """Sharp-wave-ripple analog: gate pass / go-live / phase end.

        Merges duplicates and supersedes contradictions immediately, but
        NEVER archives and NEVER synthesizes — the long-term curve is only
        touched by the nightly sleep pass.
        """
        rep = SleepReport(tenant_id=tenant_id)
        active = self.store.list_active(tenant_id)
        rep.scanned = len(active)
        for cluster in _cluster(active, self.eps):
            self._process_cluster(cluster, rep, full=False)
        return rep

    # ── internals ─────────────────────────────────────────────────────
    def _process_cluster(self, cluster: list[tuple[str, Experience]], rep: SleepReport, full: bool) -> None:
        if not cluster:
            return
        groups: dict[str, list[tuple[str, Experience]]] = {}
        for eid, exp in cluster:
            groups.setdefault(self._entity_of(exp), []).append((eid, exp))
        for entity, rows in groups.items():
            if len(rows) >= 2:
                keep = self._merge_duplicates(rows, rep)
                self._supersede_contradictions(keep, rep)
            else:
                keep = rows
            if full:
                self._synthesize_fact(entity, keep, rep)

    def _merge_duplicates(self, rows: list[tuple[str, Experience]], rep: SleepReport) -> list[tuple[str, Experience]]:
        """Near-duplicate episodes (cosine ≥ merge_eps) collapse into the
        strongest: refs union, strength max, weaker row superseded with
        lineage. Recall always resolves to the head."""
        rows = sorted(rows, key=lambda r: r[1].strength, reverse=True)
        keep: list[tuple[str, Experience]] = [rows[0]]
        head_id, head = rows[0]
        for eid, exp in rows[1:]:
            if _cosine(exp.vector, head.vector) >= self.merge_eps:
                head.refs = sorted(set(head.refs) | set(exp.refs))
                head.strength = max(head.strength, exp.strength) + 0.01
                head.supersedes.append(eid)
                exp.superseded_by = head_id
                self.store.update_state(eid, "superseded")
                rep.merged += 1
            else:
                keep.append((eid, exp))
        return keep

    def _supersede_contradictions(self, rows: list[tuple[str, Experience]], rep: SleepReport) -> None:
        """Same entity with conflicting outcomes: the OLD rows are marked
        superseded; live recall returns the chain head; audit keeps history
        (reconsolidation with lineage)."""
        outcomes = {self._outcome_of(exp) for _, exp in rows}
        real = {o for o in outcomes if o in ("pass", "fail")}
        if len(real) < 2:
            return
        ordered = sorted(rows, key=lambda r: r[1].created_at)
        newest_id = ordered[-1][0]
        for eid, exp in ordered[:-1]:
            exp.superseded_by = newest_id
            self.store.update_state(eid, "superseded")
            rep.superseded += 1

    def _synthesize_fact(self, entity: str, rows: list[tuple[str, Experience]], rep: SleepReport) -> None:
        """≥N episodes about one entity → a semantic fact (gist).

        Default synthesizer is DETERMINISTIC extractive (0 LLM tokens):
        outcome statistics over the episode set. An LLM synthesizer can be
        injected (amortized cost, cold tier), but the write path never
        requires one.
        """
        episodes = sorted({ref for _, exp in rows for ref in exp.refs})
        if len(episodes) < self.min_episodes:
            return
        counts = {"pass": 0, "fail": 0, "neutral": 0}
        for ref in episodes:
            ev = self.events_store.get(ref)
            if ev is None:
                continue
            o = str(ev.payload.get("outcome", "neutral"))
            counts[o if o in counts else "neutral"] += 1
        # regularity requires outcome-bearing episodes; pure-neutral chatter
        # (e.g., chat logs) carries no extractable fact — skip it
        decided = counts["pass"] + counts["fail"]
        if decided < 2:
            return
        if self.synthesizer is not None:
            pattern = self.synthesizer(entity, "event", counts, episodes)
        else:
            neutral_note = f" ({counts['neutral']} neutral)" if counts["neutral"] else ""
            pattern = (f"{entity}: {counts['pass']}/{decided} outcomes passed "
                       f"across {len(episodes)} episodes{neutral_note}")
        fact = SemanticFact(
            tenant_id=rows[0][1].tenant_id,
            entity=entity,
            pattern=pattern,
            counts=dict(counts),
            source_memory_ids=episodes,
        )
        self.semantic.insert(fact)
        rep.synthesized += 1
        rep.facts.append(fact)

    # ── helpers ───────────────────────────────────────────────────────
    def _build_tenant_model(self, tenant_id: str) -> TenantModel:
        active = self.store.list_active(tenant_id)
        strengths = [exp.strength for _, exp in active]
        saliences = [exp.salience for _, exp in active]
        written = 0
        capacity = 0.0
        if self.sdm is not None:
            written = int(self.sdm.capacity_used * self.sdm.num_locations)
            capacity = self.sdm.capacity_used
        return TenantModel(
            tenant_id=tenant_id,
            events_total=len(self.events_store.list_all(tenant_id)),
            experiences_active=len(active),
            semantic_facts=len(self.semantic.list(tenant_id)),
            markers=len(self.markers.visible(tenant_id)),
            avg_strength=sum(strengths) / len(strengths) if strengths else 0.0,
            avg_salience=sum(saliences) / len(saliences) if saliences else 0.0,
            cortex_capacity=capacity,
            cortex_written=written,
        )

    def _events_of(self, exp: Experience) -> list[Event]:
        events = []
        for ref in exp.refs:
            ev = self.events_store.get(ref)
            if ev is not None:
                events.append(ev)
        return events

    def _entity_of(self, exp: Experience) -> str:
        entities = [ev.project_id or ev.type for ev in self._events_of(exp)]
        entities = [e for e in entities if e]
        if not entities:
            return "unknown"
        return max(set(entities), key=entities.count)

    def _outcome_of(self, exp: Experience) -> str:
        events = self._events_of(exp)
        if not events:
            return "neutral"
        return str(events[-1].payload.get("outcome", "neutral"))
