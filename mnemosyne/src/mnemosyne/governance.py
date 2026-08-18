"""Phase D: governance — constitutional memory, right-to-be-forgotten,
affect policy.

The engine has a constitution (docs/memory-engine-design.md §4.7):

  filter_event()  — the write-time gate: events carrying forbidden content
                    classes (secrets, credentials, cross-tenant identifiers)
                    are REJECTED before they ever enter the memory. A memory
                    that is never written cannot leak.
  forget()        — right-to-be-forgotten: CASCADE erasure of an entity —
                    events, experiences, markers, semantic facts, and the
                    distributed cortical weights (SDM anti-write). The erasure
                    is audited and provably scoped to the tenant.
  affect_check()  — markers respect policy: a per-tenant risk appetite caps
                    how much emotional load a marker may exert on recall;
                    affect can be disabled per tenant entirely.

Everything is per-tenant; a forget() on tenant A never touches tenant B.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .affect import SomaticMarker, SomaticMarkerStore
from .consolidation import SemanticStore
from .events import Event
from .sdm import SparseDistributedMemory, binarize
from .storage import ExperienceStore, EventStore

# ── forbidden content classes (constitution) ─────────────────────────
DEFAULT_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("secret", re.compile(r"(?i)(api[_-]?key|secret|password|token|bearer)\s*[=:]\s*\S+")),
    ("credential", re.compile(r"(?i)(sk-[a-zA-Z0-9]{16,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})")),
    ("email", re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("cross_tenant", re.compile(r"(?i)tenant[_-]?id\s*[=:]\s*[0-9a-f-]{16,}")),
    ("aws_key", re.compile(r"(?i)AKIA[0-9A-Z]{16}")),
]

DEFAULT_FORBIDDEN_TOKENS = frozenset({
    "api_key", "apikey", "secret_key", "password", "passwd", "private_key",
    "bearer", "jwt", "csrf_token", "session_token",
})


@dataclass
class GovernanceResult:
    allowed: bool
    action: str  # "write" | "reject"
    reason: str | None = None


@dataclass
class ForgetReport:
    tenant_id: str
    entity: str
    events_deleted: int = 0
    experiences_deleted: int = 0
    markers_deleted: int = 0
    facts_deleted: int = 0
    cortex_locations_updated: int = 0
    audit_id: str = ""


class MemoryPolicy:
    """The constitution, per tenant (or per platform default)."""

    def __init__(
        self,
        forbidden_patterns: list[tuple[str, re.Pattern]] | None = None,
        forbidden_tokens: frozenset[str] | None = None,
        affect_enabled: bool = True,
        risk_appetite: float = 1.0,
    ) -> None:
        self.forbidden_patterns = forbidden_patterns or DEFAULT_FORBIDDEN_PATTERNS
        self.forbidden_tokens = forbidden_tokens or DEFAULT_FORBIDDEN_TOKENS
        self.affect_enabled = affect_enabled
        self.risk_appetite = risk_appetite

    def check_event(self, event: Event) -> GovernanceResult:
        """Constitutional review of an event BEFORE it is written."""
        text_parts: list[str] = [str(event.type)]
        for key, val in event.payload.items():
            text_parts.append(str(key))
            if isinstance(val, (str, int, float, bool)):
                text_parts.append(str(val))
            elif isinstance(val, dict):
                text_parts.append(json.dumps(val, default=str))
        text = " ".join(text_parts).lower()

        for name, pat in self.forbidden_patterns:
            if pat.search(text):
                return GovernanceResult(False, "reject", f"forbidden class: {name}")
        for tok in self.forbidden_tokens:
            if tok in text:
                return GovernanceResult(False, "reject", f"forbidden token: {tok}")
        return GovernanceResult(True, "write", None)

    def affect_check(self, marker: SomaticMarker) -> GovernanceResult:
        """Markers respect policy: risk appetite caps emotional influence."""
        if not self.affect_enabled:
            return GovernanceResult(False, "block", "affect disabled by policy")
        cal = marker.calibrated_risk
        if cal is None:
            return GovernanceResult(True, "write", None)
        if cal > self.risk_appetite:
            return GovernanceResult(False, "block",
                                    f"calibrated_risk {cal:.2f} > appetite {self.risk_appetite:.2f}")
        return GovernanceResult(True, "write", None)


class Governance:
    """Enforces the constitution + executes right-to-be-forgotten."""

    def __init__(self, policy: MemoryPolicy | None = None) -> None:
        self.policy = policy or MemoryPolicy()
        self._audit: list[dict] = []
        self._seq = 0

    # ── write-time gate ───────────────────────────────────────────────
    def filter_event(self, event: Event) -> GovernanceResult:
        result = self.policy.check_event(event)
        if not result.allowed:
            self._log("filter_reject", event.tenant_id, event.project_id or event.type,
                      result.reason or "rejected", event)
        return result

    # ── right-to-be-forgotten (cascade) ───────────────────────────────
    def forget(
        self,
        tenant_id: str,
        entity: str,
        events: EventStore,
        experiences: ExperienceStore,
        markers: SomaticMarkerStore,
        semantic: SemanticStore,
        sdm: SparseDistributedMemory | None = None,
    ) -> ForgetReport:
        """Cascade erasure of an entity: rows, markers, facts, weights."""
        report = ForgetReport(tenant_id=tenant_id, entity=entity)
        doomed_event_ids: list[str] = []
        for eid, ev in events.list_all(tenant_id):
            if ev.project_id == entity:
                doomed_event_ids.append(eid)
                events.delete(eid)
        report.events_deleted = len(doomed_event_ids)
        doomed = set(doomed_event_ids)

        # experiences whose events were deleted OR whose refs are gone —
        # EVERY state (active, superseded, archived) is erased: a forgotten
        # entity must leave no trace in any tier
        gone: list[str] = []
        for eid, exp in experiences.list_all(tenant_id):
            refs = list(exp.refs)
            if any(r in doomed for r in refs) or not refs:
                gone.append(eid)
                experiences.delete(eid)
                if sdm is not None:
                    report.cortex_locations_updated += sdm.forget(binarize(exp.vector), exp.vector)
        report.experiences_deleted = len(gone)

        # markers + facts for the entity
        for et, eid_ in (("project", entity), ("vendor", entity), ("entity", entity)):
            if markers.delete(tenant_id, et, eid_):
                report.markers_deleted += 1
        report.facts_deleted = semantic.delete(tenant_id, entity)

        self._seq += 1
        report.audit_id = f"forget_{self._seq}"
        self._log("forget", tenant_id, entity,
                  (f"events={report.events_deleted} exps={report.experiences_deleted} "
                   f"markers={report.markers_deleted} facts={report.facts_deleted} "
                   f"cortex={report.cortex_locations_updated}"), None)
        return report

    def audit_log(self) -> list[dict]:
        return list(self._audit)

    # ── internals ─────────────────────────────────────────────────────
    def _log(self, action: str, tenant_id: str, entity: str, detail: str, event: Event | None) -> None:
        self._audit.append({
            "action": action,
            "tenant_id": tenant_id,
            "entity": entity,
            "detail": detail,
            "at": datetime.now(timezone.utc).isoformat(),
            "event_type": event.type if event else None,
        })
