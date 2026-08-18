"""Affective layer — appraisal, feelings, somatic markers (§4.8).

Feelings are COMPUTED from appraisal — goal congruence, expectedness,
certainty, control, stakes — never labeled by an LLM. Somatic markers
aggregate feelings per entity and only surface when backed by evidence
(anti-superstition: intuition with receipts, never a vibe).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

# Evidence-backed sources only build markers (anti-superstition rule).
EVIDENCE_SOURCES = {"probe", "tool", "external"}

MIN_MARKER_EVIDENCE = 2  # design doc: markers exist only if evidence_count >= N

# Emotional hygiene (design doc v1.2 — "do not traumatize the model"):
# risk is recency-weighted with this half-life, so old failures heal.
AFFECT_HALF_LIFE_DAYS = 90.0
# number of trailing positive outcomes that downgrade anxiety -> caution
RECOVERY_CREDIT = 3


@dataclass
class Appraisal:
    goal_congruence: float  # +1 outcome met goal, -1 missed
    expectedness: float     # 1.0 fully predicted ... 0.0 total surprise
    certainty: float        # evidence weight of the source
    control: float          # 1.0 agent-caused outcome, 0.0 external
    stakes: float           # 0..1 project criticality
    repeats: int = 0        # consecutive same-outcome count for this entity


def appraise(
    outcome: str,
    expectedness: float,
    evidence_weight: float,
    control: float,
    stakes: float,
    repeats: int = 0,
) -> Appraisal:
    congruence = 1.0 if outcome in ("pass", "approved", "success") else -1.0
    if outcome in ("neutral", "info", "note"):
        congruence = 0.0
    return Appraisal(
        goal_congruence=congruence,
        expectedness=max(0.0, min(1.0, expectedness)),
        certainty=evidence_weight,
        control=control,
        stakes=stakes,
        repeats=repeats,
    )


def feeling(a: Appraisal) -> dict:
    """Deterministic appraisal → (label, valence, arousal). No LLM."""
    success = a.goal_congruence > 0
    failure = a.goal_congruence < 0
    surprise = a.expectedness < 0.4
    if success and surprise:
        return {"label": "surprise_euphoria", "valence": 1.0, "arousal": 0.8}
    if success:
        return {"label": "satisfaction", "valence": 1.0, "arousal": 0.3}
    if failure and a.stakes >= 0.7:
        return {"label": "anxiety", "valence": -1.0, "arousal": 0.8}
    if failure and a.repeats >= 2:
        return {"label": "frustration", "valence": -1.0, "arousal": min(1.0, 0.4 + 0.1 * a.repeats)}
    if failure:
        return {"label": "disappointment", "valence": -1.0, "arousal": 0.4}
    if surprise:
        return {"label": "curiosity", "valence": 0.0, "arousal": 0.5}
    return {"label": "neutral", "valence": 0.0, "arousal": 0.1}


@dataclass
class SomaticMarker:
    tenant_id: str
    entity_type: str
    entity_id: str
    counts: dict = field(default_factory=lambda: {"positive": 0, "negative": 0, "neutral": 0})
    evidence_ids: list = field(default_factory=list)
    feelings: list = field(default_factory=list)
    # outcome ledger: (iso timestamp, outcome, evidence_id) — the basis for
    # recency-weighted risk and recovery credit. Facts persist; affect fades.
    outcomes: list = field(default_factory=list)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def evidence_count(self) -> int:
        return len(self.evidence_ids)

    @property
    def risk(self) -> float | None:
        """RAW risk: negative / (positive + negative). Used for audit."""
        total = self.counts["positive"] + self.counts["negative"]
        if total == 0:
            return None
        return self.counts["negative"] / total

    @property
    def calibrated_risk(self) -> float | None:
        """RECENCY-WEIGHTED risk — the emotional-hygiene measure.

        Old failures lose weight exponentially (half-life AFFECT_HALF_LIFE_DAYS),
        so a project that failed months ago but has recovered does not stay
        traumatized. This is the structural answer to 'do not traumatize the
        model': risk reflects current reality, not accumulated scar tissue.
        """
        if not self.outcomes:
            return None
        now = datetime.now(timezone.utc)
        pos = neg = 0.0
        for ts_s, outcome, _ in self.outcomes:
            try:
                ts = datetime.fromisoformat(ts_s)
            except (ValueError, TypeError):
                ts = now
            age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
            w = 0.5 ** (age_days / AFFECT_HALF_LIFE_DAYS)
            if outcome in ("pass", "approved", "success"):
                pos += w
            elif outcome in ("fail", "rejected", "blocked"):
                neg += w
        total = pos + neg
        if total == 0:
            return None
        return neg / total

    @property
    def consecutive_recoveries(self) -> int:
        """Trailing positive outcomes after the last negative (recovery credit)."""
        n = 0
        for _, outcome, _ in reversed(self.outcomes):
            if outcome in ("pass", "approved", "success"):
                n += 1
            else:
                break
        return n

    @property
    def recovery_state(self) -> str:
        """healthy | recovering | caution | risk | unknown"""
        cal = self.calibrated_risk
        if cal is None:
            return "unknown"
        if cal < 0.15:
            return "healthy"
        if self.consecutive_recoveries >= RECOVERY_CREDIT:
            return "recovering"  # anxiety -> caution via reappraisal
        if cal >= 0.4:
            return "risk"
        return "caution"

    @property
    def trust(self) -> float | None:
        """Evidence-weighted reliability: 1 - calibrated risk, when evidenced."""
        if self.evidence_count < MIN_MARKER_EVIDENCE:
            return None
        r = self.calibrated_risk
        return None if r is None else 1.0 - r

    def visible(self) -> bool:
        return self.evidence_count >= MIN_MARKER_EVIDENCE

    def summary(self) -> dict:
        return {
            "entity": f"{self.entity_type}:{self.entity_id}",
            "counts": self.counts,
            "risk": self.risk,
            "calibrated_risk": self.calibrated_risk,
            "recovery_state": self.recovery_state,
            "trust": self.trust,
            "evidence_count": self.evidence_count,
            "feelings": self.feelings[-5:],
        }


class MarkerStoreProtocol(Protocol):
    """The affective-tier surface both in-memory and Postgres implement."""

    def record(self, tenant_id: str, entity_type: str, entity_id: str, outcome: str,
               feeling: dict, evidence_id: str | None, ts: datetime | None = None) -> None: ...

    def for_entity(self, tenant_id: str, entity_type: str, entity_id: str) -> SomaticMarker | None: ...

    def delete(self, tenant_id: str, entity_type: str, entity_id: str) -> bool: ...

    def visible(self, tenant_id: str) -> list[SomaticMarker]: ...

    def aggregate(self, tenant_id: str) -> list[dict]: ...


class SomaticMarkerStore:
    """Aggregates markers per (tenant, entity_type, entity_id). In-memory."""

    def __init__(self, min_evidence: int = MIN_MARKER_EVIDENCE):
        self._markers: dict[tuple[str, str, str], SomaticMarker] = {}
        self.min_evidence = min_evidence

    def record(
        self,
        tenant_id: str,
        entity_type: str,
        entity_id: str,
        outcome: str,
        feeling: dict,
        evidence_id: str | None,
        ts: datetime | None = None,
    ) -> None:
        """Record an outcome. `ts` enables recency-weighted healing in demos;
        production timestamps default to now."""
        key = (tenant_id, entity_type, entity_id)
        m = self._markers.setdefault(key, SomaticMarker(tenant_id, entity_type, entity_id))
        if outcome in ("pass", "approved", "success"):
            m.counts["positive"] += 1
        elif outcome in ("fail", "rejected", "blocked"):
            m.counts["negative"] += 1
        else:
            m.counts["neutral"] += 1
        m.feelings.append(feeling["label"])
        m.outcomes.append(((ts or datetime.now(timezone.utc)).isoformat(), outcome, evidence_id))
        if evidence_id:
            m.evidence_ids.append(evidence_id)
        m.updated_at = datetime.now(timezone.utc)

    def for_entity(self, tenant_id: str, entity_type: str, entity_id: str) -> SomaticMarker | None:
        return self._markers.get((tenant_id, entity_type, entity_id))

    def delete(self, tenant_id: str, entity_type: str, entity_id: str) -> bool:
        """Right-to-be-forgotten: remove a marker (affect fades, fact gone)."""
        return self._markers.pop((tenant_id, entity_type, entity_id), None) is not None

    def visible(self, tenant_id: str) -> list[SomaticMarker]:
        return [
            m for key, m in self._markers.items()
            if key[0] == tenant_id and m.visible()
        ]

    def aggregate(self, tenant_id: str) -> list[dict]:
        """Evidence-weighted per-entity profiles (consolidation output).

        Each entry: entity, evidence_count, calibrated_risk, dominant
        feeling label, recovery_state. Deterministic — computed, never
        LLM-labeled.
        """
        out: list[dict] = []
        for key, m in self._markers.items():
            if key[0] != tenant_id:
                continue
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
        out.sort(key=lambda d: d["entity"])
        return out


def reappraise(feeling: dict, marker: SomaticMarker | None) -> dict:
    """Reappraisal: downgrade anxiety to caution when the entity is recovering.

    Emotional regulation analog: the agent does not stay traumatized after
    recovery — the affective load is re-appraised, the facts are not erased.
    """
    if not marker or feeling.get("label") != "anxiety":
        return feeling
    if marker.recovery_state in ("recovering", "healthy"):
        return {"label": "caution", "valence": -0.2, "arousal": 0.3, "reappraised": True}
    return feeling


def balance_audit(markers: list[SomaticMarker]) -> list[dict]:
    """Consolidation-time 'therapist pass'.

    Flags markers where stale negative evidence dominates recent positive
    evidence — the entity has recovered, but the marker still carries the
    old emotional load. Suggests reappraisal + supersession of the affective
    generalization while KEEPING the factual outcome ledger (facts persist,
    affect fades).
    """
    suggestions: list[dict] = []
    for m in markers:
        raw = m.risk
        cal = m.calibrated_risk
        if raw is None or cal is None:
            continue
        if raw - cal > 0.1 and m.consecutive_recoveries >= 2:
            suggestions.append({
                "entity": f"{m.entity_type}:{m.entity_id}",
                "raw_risk": round(raw, 3),
                "calibrated_risk": round(cal, 3),
                "recoveries": m.consecutive_recoveries,
                "action": "reappraise + supersede affective generalization; keep factual outcomes",
            })
    return suggestions
