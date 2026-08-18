"""The entropy gate — prediction-error encoding as a write-time filter.

Per docs/memory-engine-design.md §4.2: an event becomes a new memory only if
its surprise — measured in bits against the tenant model — exceeds a
per-tenant threshold. Redundancy reinforces the existing trace instead of
duplicating it. This is the single largest cost and storage lever: writes
cost ~0 LLM tokens, and the gate self-calibrates so mature tenants (lower
model entropy) store even less — the falling cost curve.
"""

from __future__ import annotations

import statistics
from collections import deque

from .entropy import AdaptiveModel, information_content
from .events import Vocabulary


class EntropyGate:
    """Per-tenant entropy gate.

    The tenant model (AdaptiveModel) learns the tenant's event distribution.
    Surprise of an event is measured against a CLONE of the model (what the
    model knew BEFORE the event) as theoretical information content in bits;
    the live model learns only after the decision, so repeated events stay
    cheap forever.

    The threshold is self-calibrating: an EMA of recent stored-event bit
    costs, scaled by `theta_factor`, floored at `min_theta_bits`.
    """

    def __init__(
        self,
        tenant_id: str,
        vocab: Vocabulary,
        model: AdaptiveModel | None = None,
        theta_factor: float = 1.5,
        min_theta_bits: int = 32,
        warmup_events: int = 10,
        history: int = 64,
    ):
        self.tenant_id = tenant_id
        self.vocab = vocab
        self.model = model if model is not None else AdaptiveModel(vocab.size)
        self.theta_factor = theta_factor
        self.min_theta_bits = min_theta_bits
        self.warmup_events = warmup_events
        self._ema_bits: float | None = None
        self._recent: deque[float] = deque(maxlen=history)

    # ── measurement ─────────────────────────────────────────────
    def _sync_model(self) -> None:
        """Dynamic tenant lexicon: grow the model when the vocabulary grew."""
        if self.vocab.size > self.model.vocab:
            self.model.grow(self.vocab.size)

    def measure(self, symbols: list[int]) -> float:
        """Surprise of `symbols` in bits against a clone of the tenant model.

        The live model is NOT updated here — measurement is non-destructive.
        Call `learn()` after the store/reinforce decision.
        """
        self._sync_model()
        return information_content(symbols, self.vocab.size, self.model)

    def learn(self, symbols: list[int]) -> None:
        """Teach the tenant model the event's tokens (post-decision)."""
        self._sync_model()
        for s in symbols:
            self.model.update(s)

    # ── decision ────────────────────────────────────────────────
    @property
    def theta(self) -> int:
        """Current per-tenant threshold in bits.

        Warmup: until `warmup_events` samples exist, the floor applies so the
        first genuinely novel events can store before self-calibration kicks
        in (avoids the startup transient where theta = first_event_bits × f).
        """
        if self._ema_bits is None or len(self._recent) < self.warmup_events:
            return self.min_theta_bits
        return max(self.min_theta_bits, int(self._ema_bits * self.theta_factor))

    def decide(self, symbols: list[int]) -> tuple[bool, float]:
        """Return (should_store, surprise_bits).

        True = novelty above threshold → create a new experience.
        False = redundant → reinforce the nearest existing experience.
        """
        bits = self.measure(symbols)
        return bits >= self.theta, bits

    def record_outcome(self, bits: int, stored: bool) -> None:
        """Feed measured bits into threshold calibration."""
        self._recent.append(bits)
        if self._ema_bits is None:
            self._ema_bits = float(bits)
        else:
            self._ema_bits = 0.9 * self._ema_bits + 0.1 * bits

    # ── introspection ───────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "theta_bits": self.theta,
            "ema_bits": self._ema_bits,
            "model_total": self.model.total,
            "recent_median_bits": (
                statistics.median(self._recent) if self._recent else None
            ),
        }
