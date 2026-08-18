"""Sparse Distributed Memory (Kanerva) — the synaptic layer of Mnemosyne.

The brain's cortex stores knowledge in synaptic weights, not rows. Kanerva's
Sparse Distributed Memory (SDM; Kanerva 1988, NASA NTRS 19920021480) is the
canonical engineering model of that substrate: a fixed random set of "hard
locations" (the cortex) whose connection weights are strengthened by
experience. Memory IS the weight matrix; recall is a similarity-weighted
readout from the activated neighbourhood; capacity is bounded by the matrix.

Prototype semantics:
- binary addresses (sign-thresholded feature vectors) address the locations
- write: activate the M nearest locations, add the value vector to weights
- read: activate, return the mean of stored weights — content-addressable,
  tolerant of partial cues (pattern completion)
- novelty: fraction of the activated neighbourhood never written — the
  "this region is unlearned" signal (synaptic freshness)
- separation: similar-but-distinct patterns activate partially overlapping
  neighbourhoods, so they reinforce shared structure without blurring
"""

from __future__ import annotations

import math
import random


def binarize(vector: list[float]) -> list[int]:
    """Threshold a float vector into a binary address ({0,1})."""
    return [1 if v > 0 else 0 for v in vector]


def hamming(a: list[int], b: list[int]) -> int:
    if len(a) != len(b):
        raise ValueError("address length mismatch")
    return sum(1 for x, y in zip(a, b) if x != y)


class SparseDistributedMemory:
    def __init__(self, dim: int, num_locations: int = 512, neighbors: int = 16, rng_seed: int = 7):
        if dim < 8:
            raise ValueError("dim too small for a meaningful address space")
        if neighbors > num_locations:
            raise ValueError("neighbors must be <= num_locations")
        self.dim = dim
        self.num_locations = num_locations
        self.neighbors = neighbors
        rng = random.Random(rng_seed)
        # the "cortex": fixed random binary addresses
        self.locations = [[rng.randint(0, 1) for _ in range(dim)] for _ in range(num_locations)]
        # synaptic weights: each location accumulates the value vectors written through it
        self.weights = [[0.0] * dim for _ in range(num_locations)]
        self.write_counts = [0] * num_locations
        self.total_writes = 0

    # ── activation ─────────────────────────────────────────────
    def _activate(self, address: list[int]) -> list[int]:
        """Indices of the M nearest hard locations (Hamming distance)."""
        scored = sorted(range(self.num_locations), key=lambda i: hamming(address, self.locations[i]))
        return scored[: self.neighbors]

    # ── synaptic ops ───────────────────────────────────────────
    def write(self, address: list[int], value: list[float], alpha: float = 1.0) -> None:
        """Strengthen the weights of the activated neighbourhood by `value`."""
        if len(address) != self.dim or len(value) != self.dim:
            raise ValueError("address/value length mismatch")
        active = self._activate(address)
        w = alpha / len(active)
        for i in active:
            row = self.weights[i]
            for j in range(self.dim):
                row[j] += value[j] * w
            self.write_counts[i] += 1
        self.total_writes += 1

    def read(self, address: list[int]) -> list[float]:
        """Content-addressable readout: mean of the activated, POTENTIATED
        weights. Unwritten locations contribute nothing (a silent synapse
        does not fire), so a sparse cortex still completes patterns."""
        active = self._activate(address)
        rows = [self.weights[i] for i in active if any(self.weights[i])]
        if not rows:
            return [0.0] * self.dim
        out = [0.0] * self.dim
        for row in rows:
            for j in range(self.dim):
                out[j] += row[j]
        return [v / len(rows) for v in out]

    def patch(self, address: list[int]) -> dict[int, float]:
        """The DISTRIBUTED pattern behind a cue: location -> total weight
        for the activated, potentiated neighbourhood. This is the memory's
        cortical footprint — the pattern, not the row."""
        out: dict[int, float] = {}
        for i in self._activate(address):
            row = self.weights[i]
            if any(row):
                out[i] = round(sum(row), 4)
        return out

    def forget(self, address: list[int], value: list[float], alpha: float = 1.0) -> int:
        """Approximate cortical erasure (anti-write): subtract the memory's
        contribution from its activated locations. Overlapping memories may
        lose shared weight — unlearning is approximate, exactly like the
        brain. Returns the number of locations updated."""
        active = self._activate(address)
        w = alpha / len(active)
        updated = 0
        for i in active:
            row = self.weights[i]
            changed = False
            for j in range(self.dim):
                row[j] -= value[j] * w
                if row[j] < 0.0:
                    row[j] = 0.0
                changed = changed or value[j] * w != 0.0
            if changed:
                self.write_counts[i] = max(0, self.write_counts[i] - 1)
                updated += 1
        self.total_writes = max(0, self.total_writes - 1)
        return updated

    def novelty(self, address: list[int], learned_threshold: int = 1) -> float:
        """0..1 — how unlearned this region is (synaptic freshness)."""
        active = self._activate(address)
        unlearned = sum(1 for i in active if self.write_counts[i] < learned_threshold)
        return unlearned / len(active)

    # ── introspection ──────────────────────────────────────────
    @property
    def capacity_used(self) -> float:
        """Fraction of locations that have ever been written."""
        used = sum(1 for c in self.write_counts if c > 0)
        return used / self.num_locations

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)
