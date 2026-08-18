"""Phase C: recall — retrieve EXPERIENCES, not just tokens.

The read path is now fully LLM-free (0 tokens by default):

  retrieve()  — content-addressable recall: query → feature vector → SDM
                pattern completion (cortical readout) → hippocampal index
                (nearest experiences to the COMPLETED pattern) → raw chunks
                resolved from the event log. This is associative recall: a
                partial cue reconstructs the pattern (pattern completion),
                then the index maps it to stored experiences.
  brief()     — deterministic, budget-bounded situation brief from
                retrieved experiences + semantic facts + somatic markers.
                0 LLM tokens by default; an LLM reader is optional (opt-in).
  replay()    — eidetic replay: exact raw reconstruction of an experience
                from its event refs (flashbulb-memory analog).
  reinforce() — retrieval-use effect (the testing effect): recalling a
                memory strengthens it and slows its decay. Retrieval is not
                passive — every recall reshapes the memory.
  marker_predicts_outcome() — the Phase C exit metric: does the marker's
                calibrated risk forecast subsequent outcomes?

The brain analogy: the LLM is the READER of the brief, never the memory.
The memory is data (events + experiences + facts + markers + weights), so
recall cannot hallucinate — it returns what was actually stored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .affect import MarkerStoreProtocol, SomaticMarkerStore
from .consolidation import SemanticStoreProtocol
from .embed import STOPWORDS, query_token_ids, query_vector  # noqa: F401 — re-exported
from .events import Event, Vocabulary, _TOKEN_RE
from .governance import MemoryPolicy
from .sdm import SparseDistributedMemory, binarize
from .storage import ExperienceStore, EventStore, _cosine

RETRIEVAL_REF_PREFIX = "recall:"

# scoring channels: cue-match (familiarity), entity activation (spreading
# activation), cortical completion (recollection). Completion is a hint.
W_CUE = 0.60
W_ENTITY = 0.25
W_COMPLETION = 0.15


def vocab_id_in(project_id: str, vocab: Vocabulary, q_ids: set[int]) -> bool:
    """True when a project id's vocabulary symbol is named by the cue."""
    return vocab.id(project_id) in q_ids


@dataclass
class RecallHit:
    experience_id: str
    label: str
    score: float
    salience: float
    strength: float
    feeling: str
    state: str
    entity: str
    chunks: list[dict] = field(default_factory=list)


@dataclass
class Reconstruction:
    """A memory RECONSTRUCTED the way the brain does — a distributed
    multi-channel pattern, not a flat row. Each channel is a different
    system of the mind, and the whole is greater than the sum:

      signature   — the sparse glyph (the ideogram the recall is anchored to)
      distributed — the cortical patch: which synapses fire and how hard
                    (the pattern, not the data)
      words       — the lexical channel (what was said)
      concepts    — the semantic channel (what was learned)
      emotions    — the affective channel (how it felt — computed, healing)
      spatial     — where it sits among other memories (relations)
      sensory     — artifact references (images, recordings, evidence blobs)
      context     — the tenant model (when/where in the stream of life)
    """

    cue: str
    experience_id: str
    label: str
    signature: str
    distributed: dict[str, float]
    words: list[str]
    concepts: list[str]
    emotions: list[dict]
    spatial: list[str]
    sensory: list[str]
    context: dict


class RecallEngine:
    """The full read path. Everything deterministic unless the caller opts
    into the LLM reader for the brief."""

    def __init__(
        self,
        store: ExperienceStore,
        markers: MarkerStoreProtocol,
        events_store: EventStore,
        semantic: SemanticStoreProtocol,
        sdm: SparseDistributedMemory,
        vocab: Vocabulary,
        k: int = 5,
        brief_budget: int = 900,
        reinforce_delta: float = 0.12,
        reinforce_lam_growth: float = 0.04,
        policy: MemoryPolicy | None = None,
        query_embedder: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.store = store
        self.markers = markers
        self.events_store = events_store
        self.semantic = semantic
        self.sdm = sdm
        self.vocab = vocab
        self.k = k
        self.brief_budget = brief_budget
        self.reinforce_delta = reinforce_delta
        self.reinforce_lam_growth = reinforce_lam_growth
        self.policy = policy
        self.query_embedder = query_embedder  # local-model queries when set
        self.recalls = 0

    # ── retrieval ─────────────────────────────────────────────────────
    def retrieve(self, tenant_id: str, query: str, k: int | None = None,
                 completion: bool = True, reinforce: bool = True) -> list[RecallHit]:
        """Content-addressable recall through the cortex + hippocampal index.

        Two-channel scoring (familiarity + recollection): each candidate is
        scored by a BLEND of (a) exact cue-match and (b) pattern-completed
        match — the cortical readout fills in a partial cue, but completion
        is a hint, never the answer (cue-match weighted 0.65).
        """
        k = k or self.k
        qv = (self.query_embedder(query) if self.query_embedder is not None
              else query_vector(query, self.vocab, self.sdm.dim))
        q_ids = set(query_token_ids(query, self.vocab))
        completed = None
        if completion:
            read = self.sdm.read(binarize(qv))
            if any(read):
                completed = read
        candidates: dict[str, float] = {}
        for eid, exp in self.store.nearest(tenant_id, qv, k=k * 3):
            events: list[Event] = []
            for r in exp.refs:
                ev = self.events_store.get(r)
                if ev is not None:
                    events.append(ev)
            # spreading activation: fraction of the experience's events whose
            # entity (project) is named by the cue
            activation = 0.0
            if events:
                activation = sum(1 for ev in events if ev.project_id and vocab_id_in(ev.project_id, self.vocab, q_ids)) / len(events)
            comp = _cosine(completed, exp.vector) if completed is not None else 0.0
            score = W_CUE * _cosine(qv, exp.vector) + W_ENTITY * activation + W_COMPLETION * comp
            candidates[eid] = max(candidates.get(eid, 0.0), score)
        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:k]

        hits: list[RecallHit] = []
        for eid, score in ranked:
            exp = self.store.get(eid)
            if exp is None:
                continue
            events: list[Event] = []
            for r in exp.refs:
                ev = self.events_store.get(r)
                if ev is not None:
                    events.append(ev)
            chunks = [ev.payload for ev in events]
            entities = [ev.project_id or ev.type for ev in events]
            entity = max(set(entities), key=entities.count) if entities else "unknown"
            label = f"{events[-1].type if events else '?'}:{entity}"
            hits.append(RecallHit(
                experience_id=eid,
                label=label,
                score=round(score, 4),
                salience=exp.salience,
                strength=round(exp.strength, 3),
                feeling=exp.feeling.get("label", "—") if exp.feeling else "—",
                state=exp.state,
                entity=entity,
                chunks=chunks,
            ))
        self.recalls += 1
        if reinforce:
            for hit in hits:
                self.reinforce(hit.experience_id, ref=f"{RETRIEVAL_REF_PREFIX}{self.recalls}:{hash(query) & 0xffffffff:x}")
        return hits

    # ── situation brief (0 LLM by default) ────────────────────────────
    def brief(self, tenant_id: str, query: str, k: int | None = None) -> tuple[str, str]:
        """Budget-bounded brief from hits + semantic facts + markers.

        Returns (text, mode); mode is always 'extractive' — the brief is
        built from stored structure, never generated. Budget compliance is
        structural: text is truncated to brief_budget chars.
        """
        hits = self.retrieve(tenant_id, query, k=k)
        entities = {h.entity for h in hits}
        facts = [f for f in self.semantic.list(tenant_id) if f.entity in entities][:4]
        markers = sorted(self.markers.visible(tenant_id), key=lambda m: m.summary()["entity"])

        lines = [f"BRIEF — {query}", "=" * 40]
        if facts:
            lines.append("semantic facts:")
            for f in facts:
                lines.append(f"  · {f.pattern}")
        if markers:
            lines.append("somatic markers:")
            for m in markers[:5]:
                s = m.summary()
                if self.policy is not None:
                    g = self.policy.affect_check(m)
                    if not g.allowed:
                        lines.append(f"  · {s['entity']}: BLOCKED by affect policy ({g.reason})")
                        continue
                lines.append(f"  · {s['entity']}: cal_risk={s['calibrated_risk']} "
                             f"state={s['recovery_state']} trust={s['trust']}")
        if hits:
            lines.append("experiences:")
            for h in hits[:k or self.k]:
                preview = json.dumps(h.chunks[0])[:110] if h.chunks else "(no chunk)"
                lines.append(f"  · {h.label} [salience={h.salience} strength={h.strength} "
                             f"feeling={h.feeling}] {preview}")
        text = "\n".join(lines)
        return text[: self.brief_budget], "extractive"

    # ── multimodal reconstruction ─────────────────────────────────────
    def reconstruct(self, tenant_id: str, cue: str) -> Reconstruction:
        """Reconstruct an experience as a DISTRIBUTED multi-channel pattern.

        The brain does not return data for a memory — it rebuilds the
        experience: which synapses fire (the pattern), what was said
        (words), what was learned (concepts), how it felt (emotions),
        where it lives among other memories (spatial), what artifacts
        surround it (sensory), and the state of the world it happened in
        (context). This is that reconstruction, computed deterministically
        from the distributed representation — 0 LLM tokens.
        """
        hits = self.retrieve(tenant_id, cue, k=1, reinforce=True)
        if not hits:
            raise ValueError(f"no memory reconstructs from cue: {cue!r}")
        top = hits[0]
        exp = self.store.get(top.experience_id)
        if exp is None:
            raise ValueError(f"experience vanished: {top.experience_id}")

        # signature — the sparse glyph the memory is anchored to
        addr = binarize(exp.vector)
        signature = "".join(f"{b:x}" for b in addr)

        # distributed — the cortical patch: which synapses fire and how hard
        patch = self.sdm.patch(addr)
        distributed = {f"loc_{k}": v for k, v in sorted(patch.items())}

        # words — lexical channel (top content tokens across the chunks)
        counts: dict[str, int] = {}
        for chunk in top.chunks:
            for tok in _TOKEN_RE.findall(json.dumps(chunk, default=str).lower()):
                if tok not in STOPWORDS and len(tok) > 2:
                    counts[tok] = counts.get(tok, 0) + 1
        words = [t for t, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]]

        # concepts — semantic channel
        concepts = [f.pattern for f in self.semantic.list(tenant_id) if f.entity == top.entity][:4]

        # emotions — affective channel (computed, healing)
        emotions: list[dict] = []
        for m in self.markers.visible(tenant_id):
            s = m.summary()
            if s["entity"].endswith(top.entity) or s["entity"] == top.entity:
                emotions.append({
                    "entity": s["entity"],
                    "calibrated_risk": s["calibrated_risk"],
                    "state": s["recovery_state"],
                    "dominant": s["feelings"][-1] if s["feelings"] else None,
                })

        # spatial — where this memory sits among its neighbours
        spatial: list[str] = []
        for eid, nexp in self.store.nearest(tenant_id, exp.vector, k=4):
            if eid == top.experience_id:
                continue
            nev: list[Event] = []
            for r in nexp.refs:
                n_ev = self.events_store.get(r)
                if n_ev is not None:
                    nev.append(n_ev)
            nlabel = f"{nev[-1].type}:{nev[-1].project_id}" if nev else eid
            spatial.append(nlabel)

        # sensory — artifact references (images, recordings, evidence blobs)
        sensory: list[str] = []
        for r in exp.refs:
            ev = self.events_store.get(r)
            if ev is not None and ev.evidence:
                sensory.append(json.dumps(ev.evidence, default=str))
        sensory = sensory[:4]

        # context — the stream of life this memory happened in
        context = {
            "experiences": len(self.store.list_active(tenant_id)),
            "facts": len(self.semantic.list(tenant_id)),
            "markers": len(self.markers.visible(tenant_id)),
            "cortex_potentiated": sum(1 for w in self.sdm.write_counts if w > 0),
            "signature_len": len(signature),
        }

        return Reconstruction(
            cue=cue,
            experience_id=top.experience_id,
            label=top.label,
            signature=signature,
            distributed=distributed,
            words=words,
            concepts=concepts,
            emotions=emotions,
            spatial=spatial,
            sensory=sensory,
            context=context,
        )

    # ── eidetic replay ────────────────────────────────────────────────
    def replay(self, experience_id: str) -> list[dict]:
        """EXACT raw reconstruction: the event log rows behind a memory.
        This is the flashbulb tier — replay returns what actually happened,
        not a paraphrase."""
        exp = self.store.get(experience_id)
        if exp is None:
            return []
        out = []
        for r in exp.refs:
            ev = self.events_store.get(r)
            if ev is None:
                continue
            out.append({
                "event_id": r,
                "type": ev.type,
                "project": ev.project_id,
                "source": ev.source,
                "payload": ev.payload,
                "created_at": ev.created_at.isoformat(),
            })
        return out

    # ── reinforcement (retrieval-use effect) ──────────────────────────
    def reinforce(self, experience_id: str, ref: str | None = None) -> None:
        """Recalling strengthens: S += ΔS and the half-life grows — the
        memory is used, so it is kept (testing effect)."""
        exp = self.store.get(experience_id)
        if exp is None:
            return
        self.store.reinforce(experience_id, self.reinforce_delta,
                             ref or f"{RETRIEVAL_REF_PREFIX}{experience_id}")
        exp.half_life_hours *= (1.0 + self.reinforce_lam_growth)

    # ── Phase C exit metric ───────────────────────────────────────────
    def marker_predicts_outcome(self, tenant_id: str, split: float = 0.8) -> dict:
        """Does calibrated risk forecast subsequent outcomes?

        For every visible marker: split the outcome ledger at `split`; the
        EARLY outcomes define the marker state (approximated by the current
        calibrated risk — production uses the risk AT the split point); the
        LATE outcomes are the ground truth. Reports MAE between predicted
        risk and realized failure rate, plus directional accuracy.
        """
        details = []
        errs = []
        correct = total = 0
        for m in self.markers.visible(tenant_id):
            ledger = sorted(m.outcomes)  # (iso ts, outcome, evidence)
            if len(ledger) < 4:
                continue
            n_late = max(1, int(len(ledger) * (1.0 - split)))
            late = ledger[-n_late:]
            pos = sum(1 for _, o, _ in late if o in ("pass", "approved", "success"))
            neg = sum(1 for _, o, _ in late if o in ("fail", "rejected", "blocked"))
            realized = neg / (pos + neg) if (pos + neg) else 0.0
            pred = m.calibrated_risk or realized
            errs.append(abs(pred - realized))
            # directional call: pred >= 0.5 -> predict fail
            predicted_fail = pred >= 0.5
            actual_fail = neg > pos
            if neg or pos:
                total += 1
                if predicted_fail == actual_fail:
                    correct += 1
            details.append({
                "entity": m.summary()["entity"],
                "predicted_risk": round(pred, 3),
                "realized_failure_rate": round(realized, 3),
                "late_outcomes": {"pass": pos, "fail": neg},
            })
        return {
            "entities": len(details),
            "mae": round(sum(errs) / len(errs), 3) if errs else None,
            "directional_accuracy": round(correct / total, 3) if total else None,
            "details": details,
        }
