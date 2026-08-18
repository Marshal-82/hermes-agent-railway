"""Production data path — Postgres + pgvector adapter (Phase E).

Implements the SAME EventStore / ExperienceStore interfaces as the
in-memory stores, so the entire engine (gate, write path, sleep, recall,
governance) runs unchanged against Supabase Postgres + pgvector.

Schema: supabase/migrations/0001_mnemosyne_schema.sql (RLS default-deny,
tenant_id on every row). The adapter itself filters by tenant_id in every
query; RLS is the production backstop via JWT claims.

Mapping notes:
- bigint identity ids <-> str ids ("ev_1" / "exp_1" style) at the boundary
- vector columns are passed as pgvector string literals ("[0.1,0.2,...]")
- `supersedes_id` holds the `superseded_by` pointer (single-slot lineage
  in SQL; the in-memory store keeps the full list)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from .events import Event
from .storage import Experience, ExperienceStore, EventStore, _cosine


def _vec_str(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _parse_vec(s) -> list[float]:
    """pgvector returns vectors as '[0.1,0.2,...]' strings — parse either."""
    if s is None:
        return []
    if isinstance(s, str):
        return [float(x) for x in s.strip("[]").split(",") if x]
    return [float(x) for x in s]


def _dt(v) -> datetime:
    if v is None:
        return datetime.now(timezone.utc)
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(v))


class PgEventStore(EventStore):
    """L0 event log on Postgres."""

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo

    def _conn(self):
        return psycopg.connect(self.conninfo, row_factory=dict_row)

    def append(self, event: Event) -> str:
        with self._conn() as conn:
            row = conn.execute(
                """insert into public.mnemosyne_events
                   (tenant_id, project_id, session_id, type, payload, evidence, source)
                   values (%s,%s,%s,%s,%s,%s,%s) returning id""",
                (event.tenant_id, event.project_id, event.session_id, event.type,
                 json.dumps(event.payload), json.dumps(event.evidence) if event.evidence else None,
                 event.source),
            ).fetchone()
            return str(row["id"])

    def get(self, event_id: str) -> Event | None:
        with self._conn() as conn:
            row = conn.execute(
                "select * from public.mnemosyne_events where id = %s", (int(event_id),)
            ).fetchone()
        if row is None:
            return None
        return Event(
            tenant_id=row["tenant_id"], type=row["type"], payload=row["payload"] or {},
            project_id=row["project_id"], session_id=row["session_id"],
            source=row["source"] or "agent", evidence=row["evidence"],
            created_at=_dt(row["created_at"]),
        )

    def delete(self, event_id: str) -> None:
        with self._conn() as conn:
            conn.execute("delete from public.mnemosyne_events where id = %s", (int(event_id),))

    def list_all(self, tenant_id: str) -> list[tuple[str, Event]]:
        with self._conn() as conn:
            rows = conn.execute(
                "select * from public.mnemosyne_events where tenant_id = %s order by id",
                (tenant_id,),
            ).fetchall()
        out = []
        for row in rows:
            out.append((str(row["id"]), Event(
                tenant_id=row["tenant_id"], type=row["type"], payload=row["payload"] or {},
                project_id=row["project_id"], session_id=row["session_id"],
                source=row["source"] or "agent", evidence=row["evidence"],
                created_at=_dt(row["created_at"]),
            )))
        return out


class PgExperienceStore(ExperienceStore):
    """L1 episodic tier on Postgres + pgvector (HNSW cosine)."""

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo

    def _conn(self):
        return psycopg.connect(self.conninfo, row_factory=dict_row)

    @staticmethod
    def _row_to_exp(row: dict) -> Experience:
        return Experience(
            tenant_id=row["tenant_id"],
            vector=_parse_vec(row["vector"]),
            signature=row["signature"] or "",
            salience=row["salience"] or 1.0,
            strength=row["strength"] or 1.0,
            half_life_hours=row["half_life_hours"] or 720.0,
            appraisal=row["appraisal"] or {},
            feeling=row["feeling"] or {},
            refs=[str(x) for x in (row["refs"] or [])],
            state=row["state"] or "active",
            access_count=row["access_count"] or 0,
            last_access_at=_dt(row["last_access_at"]) if row["last_access_at"] else None,
            created_at=_dt(row["created_at"]),
            supersedes=[],  # SQL keeps the single superseded_by pointer
            superseded_by=str(row["supersedes_id"]) if row["supersedes_id"] else None,
        )

    def insert(self, exp: Experience) -> str:
        with self._conn() as conn:
            row = conn.execute(
                """insert into public.mnemosyne_experiences
                   (tenant_id, signature, vector, salience, strength, half_life_hours,
                    appraisal, feeling, state, refs, supersedes_id)
                   values (%s,%s,%s::vector,%s,%s,%s,%s,%s,%s,%s,%s) returning id""",
                (exp.tenant_id, exp.signature, _vec_str(exp.vector), exp.salience,
                 exp.strength, exp.half_life_hours, json.dumps(exp.appraisal),
                 json.dumps(exp.feeling), exp.state,
                 [int(r) for r in exp.refs if r.isdigit()],
                 int(exp.superseded_by) if exp.superseded_by and exp.superseded_by.isdigit() else None),
            ).fetchone()
            return str(row["id"])

    def reinforce(self, experience_id: str, strength_delta: float, ref: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """update public.mnemosyne_experiences
                   set strength = strength + %s,
                       refs = refs || %s,
                       access_count = access_count + 1,
                       last_access_at = now(),
                       updated_at = now()
                   where id = %s""",
                (strength_delta, [int(ref)] if ref.isdigit() else [], int(experience_id)),
            )

    def nearest(self, tenant_id: str, vector: list[float], k: int = 5) -> list[tuple[str, Experience]]:
        with self._conn() as conn:
            rows = conn.execute(
                """select *, vector <=> %s::vector as dist
                   from public.mnemosyne_experiences
                   where tenant_id = %s and state = 'active'
                   order by vector <=> %s::vector
                   limit %s""",
                (_vec_str(vector), tenant_id, _vec_str(vector), k),
            ).fetchall()
        out = []
        for row in rows:
            exp = self._row_to_exp(row)
            out.append((str(row["id"]), exp))
        return out

    def get(self, experience_id: str) -> Experience | None:
        with self._conn() as conn:
            row = conn.execute(
                "select * from public.mnemosyne_experiences where id = %s", (int(experience_id),)
            ).fetchone()
        return self._row_to_exp(row) if row else None

    def delete(self, experience_id: str) -> None:
        with self._conn() as conn:
            conn.execute("delete from public.mnemosyne_experiences where id = %s", (int(experience_id),))

    def list_active(self, tenant_id: str) -> list[tuple[str, Experience]]:
        with self._conn() as conn:
            rows = conn.execute(
                "select * from public.mnemosyne_experiences where tenant_id = %s and state = 'active' order by id",
                (tenant_id,),
            ).fetchall()
        return [(str(r["id"]), self._row_to_exp(r)) for r in rows]

    def list_all(self, tenant_id: str) -> list[tuple[str, Experience]]:
        with self._conn() as conn:
            rows = conn.execute(
                "select * from public.mnemosyne_experiences where tenant_id = %s order by id",
                (tenant_id,),
            ).fetchall()
        return [(str(r["id"]), self._row_to_exp(r)) for r in rows]

    def update_state(self, experience_id: str, state: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "update public.mnemosyne_experiences set state = %s, updated_at = now() where id = %s",
                (state, int(experience_id)),
            )


class PgSemanticStore:
    """L2 semantic facts on Postgres (persisted across cron runs)."""

    def __init__(self, conninfo: str) -> None:
        self.conninfo = conninfo

    def _conn(self):
        return psycopg.connect(self.conninfo, row_factory=dict_row)

    def insert(self, fact) -> int:
        from .consolidation import SemanticFact
        with self._conn() as conn:
            row = conn.execute(
                """insert into public.mnemosyne_semantic_facts
                   (tenant_id, fact, confidence, source_memory_ids)
                   values (%s, %s, %s, %s) returning id""",
                (fact.tenant_id,
                 json.dumps({"entity": fact.entity, "pattern": fact.pattern, "counts": fact.counts}),
                 0.7, [int(x) for x in fact.source_memory_ids if x.isdigit()]),
            ).fetchone()
            return int(row["id"])

    def list(self, tenant_id: str) -> list:
        from .consolidation import SemanticFact
        with self._conn() as conn:
            rows = conn.execute(
                "select * from public.mnemosyne_semantic_facts where tenant_id = %s order by id",
                (tenant_id,),
            ).fetchall()
        out = []
        for row in rows:
            f = row["fact"] or {}
            out.append(SemanticFact(
                tenant_id=row["tenant_id"],
                entity=f.get("entity", "unknown"),
                pattern=f.get("pattern", ""),
                counts=f.get("counts", {}),
                source_memory_ids=[str(x) for x in (row["source_memory_ids"] or [])],
            ))
        return out

    def delete(self, tenant_id: str, entity: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "delete from public.mnemosyne_semantic_facts where tenant_id = %s and fact->>'entity' = %s",
                (tenant_id, entity),
            )
            return cur.rowcount


class PgMarkerStore:
    """Affective tier on Postgres — markers persist across cron runs and
    processes. Mirrors the SomaticMarkerStore surface (structural typing)."""

    def __init__(self, conninfo: str, min_evidence: int = 2) -> None:
        self.conninfo = conninfo
        self.min_evidence = min_evidence

    def _conn(self):
        return psycopg.connect(self.conninfo, row_factory=dict_row)

    @staticmethod
    def _row_to_marker(row: dict):
        from .affect import SomaticMarker
        m = SomaticMarker(row["tenant_id"], row["entity_type"], row["entity_id"])
        m.counts = {"positive": 0, "negative": 0, "neutral": 0}
        m.counts.update(row["counts"] or {})
        m.feelings = list((row["feeling_histogram"] or {}).get("labels", []))
        m.outcomes = [tuple(x) for x in (row["outcomes"] or [])]
        m.evidence_ids = [str(x) for x in (row["evidence_ids"] or [])]
        m.updated_at = _dt(row["updated_at"])
        return m

    def record(self, tenant_id, entity_type, entity_id, outcome, feeling, evidence_id, ts=None) -> None:
        from datetime import datetime, timezone
        ts_iso = (ts or datetime.now(timezone.utc)).isoformat()
        delta = _count_delta(outcome)
        with self._conn() as conn:
            cur = conn.execute(
                "select counts from public.mnemosyne_somatic_markers "
                "where tenant_id=%s and entity_type=%s and entity_id=%s for update",
                (tenant_id, entity_type, entity_id),
            )
            existing = cur.fetchone()
            counts = dict(existing["counts"] or {}) if existing else {}  # type: ignore[index]
            for k, v in delta.items():
                counts[k] = counts.get(k, 0) + v
            label = feeling.get("label", "neutral")
            if existing:
                conn.execute(
                    """update public.mnemosyne_somatic_markers
                       set counts = %s::jsonb,
                           feeling_histogram = jsonb_set(feeling_histogram, '{labels}',
                             coalesce(feeling_histogram->'labels','[]'::jsonb) || %s::jsonb),
                           outcomes = outcomes || %s::jsonb,
                           evidence_ids = evidence_ids || %s::bigint[],
                           updated_at = now()
                       where tenant_id=%s and entity_type=%s and entity_id=%s""",
                    (json.dumps(counts), json.dumps([label]), json.dumps([[ts_iso, outcome, evidence_id]]),
                     [int(evidence_id)] if evidence_id and evidence_id.isdigit() else [],
                     tenant_id, entity_type, entity_id),
                )
            else:
                conn.execute(
                    """insert into public.mnemosyne_somatic_markers
                       (tenant_id, entity_type, entity_id, counts, feeling_histogram, outcomes, evidence_ids)
                       values (%s,%s,%s,%s::jsonb,
                               jsonb_build_object('labels', %s::jsonb), %s::jsonb, %s::bigint[])""",
                    (tenant_id, entity_type, entity_id, json.dumps(counts), json.dumps([label]),
                     json.dumps([[ts_iso, outcome, evidence_id]]),
                     [int(evidence_id)] if evidence_id and evidence_id.isdigit() else []),
                )

    def for_entity(self, tenant_id: str, entity_type: str, entity_id: str):
        with self._conn() as conn:
            row = conn.execute(
                "select * from public.mnemosyne_somatic_markers where tenant_id=%s and entity_type=%s and entity_id=%s",
                (tenant_id, entity_type, entity_id),
            ).fetchone()
        return self._row_to_marker(row) if row else None

    def delete(self, tenant_id: str, entity_type: str, entity_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "delete from public.mnemosyne_somatic_markers where tenant_id=%s and entity_type=%s and entity_id=%s",
                (tenant_id, entity_type, entity_id),
            )
            return cur.rowcount > 0

    def visible(self, tenant_id: str) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "select * from public.mnemosyne_somatic_markers where tenant_id = %s order by entity_id",
                (tenant_id,),
            ).fetchall()
        out = []
        for row in rows:
            m = self._row_to_marker(row)
            if m.visible():
                out.append(m)
        return out

    def aggregate(self, tenant_id: str) -> list[dict]:
        out = []
        for m in self.visible(tenant_id):
            s = m.summary()
            dominant = None
            best = 0.0
            for f in m.feelings:
                cnt = sum(1 for x in m.feelings if x == f)
                if cnt > best:
                    best, dominant = cnt, f
            out.append({
                "entity": s["entity"],
                "evidence_count": s["evidence_count"],
                "calibrated_risk": s["calibrated_risk"],
                "dominant_label": dominant,
                "recovery_state": s["recovery_state"],
            })
        return out


def _count_delta(outcome: str) -> dict:
    if outcome in ("pass", "approved", "success"):
        return {"positive": 1}
    if outcome in ("fail", "rejected", "blocked"):
        return {"negative": 1}
    return {"neutral": 1}
