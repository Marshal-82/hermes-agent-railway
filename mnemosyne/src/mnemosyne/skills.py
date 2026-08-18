"""Skill Economy — which skills pay rent.

Agents accumulate skills; the memory engine now tells them which skills
actually pay. Every `skill_use` event (type="skill_use", payload
{"skill": "<id>", "outcome": "pass"|"fail"|...}) is ordinary memory — gated,
stored, reinforced — and simultaneously a trace in the skill ledger.

skill_economy() aggregates the ledger per tenant:
  uses, success_rate, RECENT rate (last 5 uses — a skill that just started
  failing shows it), and `rent` = success_rate × sqrt(uses) — the skill
  economy score. Verdicts are deterministic:

    rent ≥ 0.9 & uses ≥ 2  → keep      (pays rent, keep investing)
    success_rate < 0.5 & uses ≥ 3 → review (costs more than it earns)
    rent < 0.4             → retire    (prune — the Skill Economy's
                                         counterpart to memory decay)
    uses == 1              → seeding   (too early to judge)
    else                   → reinforce (promising, invest more)

No LLM: verdicts are arithmetic, auditable, per-tenant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .events import Event
from .storage import EventStore


@dataclass
class SkillStat:
    skill_id: str
    uses: int
    passes: int
    fails: int
    neutrals: int
    success_rate: float
    recent_rate: float | None
    rent: float
    verdict: str

    def as_dict(self) -> dict:
        return {
            "skill": self.skill_id,
            "uses": self.uses,
            "passes": self.passes,
            "fails": self.fails,
            "success_rate": round(self.success_rate, 3),
            "recent_rate": round(self.recent_rate, 3) if self.recent_rate is not None else None,
            "rent": round(self.rent, 3),
            "verdict": self.verdict,
        }


def make_skill_event(tenant_id: str, skill_id: str, outcome: str) -> Event:
    """A skill-use trace as an ordinary memory event."""
    return Event(
        tenant_id=tenant_id,
        type="skill_use",
        project_id=None,
        payload={"skill": skill_id, "outcome": outcome},
        source="tool",
    )


def skill_economy(events: EventStore, tenant_id: str, min_uses: int = 2) -> list[SkillStat]:
    """Aggregate the skill ledger from the event log (deterministic, 0 LLM)."""
    per_skill: dict[str, dict] = {}
    for _, ev in events.list_all(tenant_id):
        if ev.type != "skill_use":
            continue
        skill = str(ev.payload.get("skill", "?"))
        outcome = str(ev.payload.get("outcome", "neutral"))
        bucket = per_skill.setdefault(skill, {"outcomes": []})
        bucket["outcomes"].append(outcome)

    stats: list[SkillStat] = []
    for skill, bucket in per_skill.items():
        outcomes = bucket["outcomes"]
        passes = sum(1 for o in outcomes if o in ("pass", "approved", "success"))
        fails = sum(1 for o in outcomes if o in ("fail", "rejected", "blocked"))
        neutrals = len(outcomes) - passes - fails
        decided = passes + fails
        rate = (passes / decided) if decided else 0.0
        recent = outcomes[-5:]
        recent_decided = [o for o in recent if o in ("pass", "fail")]
        recent_rate = (sum(1 for o in recent_decided if o == "pass") / len(recent_decided)) if recent_decided else None
        rent = rate * math.sqrt(len(outcomes))

        if rent >= 0.9 and len(outcomes) >= min_uses and (recent_rate is None or recent_rate >= 0.5):
            verdict = "keep"
        elif (rate < 0.5 and len(outcomes) >= 3) or (recent_rate is not None and recent_rate < 0.4 and len(outcomes) >= 4):
            verdict = "review"  # struggling, OR declining (was good, now failing)
        elif rent < 0.4:
            verdict = "retire"
        elif len(outcomes) == 1:
            verdict = "seeding"
        else:
            verdict = "reinforce"

        stats.append(SkillStat(
            skill_id=skill, uses=len(outcomes), passes=passes, fails=fails,
            neutrals=neutrals, success_rate=rate, recent_rate=recent_rate,
            rent=rent, verdict=verdict,
        ))
    stats.sort(key=lambda s: s.rent, reverse=True)
    return stats
