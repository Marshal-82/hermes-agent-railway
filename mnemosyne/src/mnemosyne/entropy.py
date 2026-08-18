"""Adaptive arithmetic coding — the entropy gate engine.

Mnemosyne stores surprises, not replays. The surprise of a symbol stream under
an adaptive model is its information content in bits: -log2 p. Arithmetic
coding (Rissanen 1979; classic implementation per Witten, Neal & Cleary,
CACM 1987) is the optimal coder for that quantity: the encoded length IS the
measured surprise.

Design notes (see docs/memory-engine-design.md §2.3, §4.2):
- One AdaptiveModel per tenant = the "tenant model". Surprise of an event is
  measured against a CLONE of the model (pre-event knowledge); the live model
  is updated only after the decision (learn), so redundant events cost few
  bits and novel events cost many.
- Fixed vocabulary + UNKNOWN symbol for the prototype. Phase B adds dynamic
  vocabulary with periodic model rebuilds during consolidation ("sleep").
"""

from __future__ import annotations

import math

# Fixed-point precision — classic 16-bit arithmetic coder constants.
TOP_VALUE = 0xFFFF
FIRST_QTR = 16384
HALF = 32768
THIRD_QTR = 49152

# Adaptive model rescale threshold (halve counts to keep totals bounded).
MAX_TOTAL = 8192

EOF_SYMBOL = -1  # marker returned by the decoder when the EOF symbol is read


class AdaptiveModel:
    """Order-0 adaptive frequency model over symbols 0..vocab-1 plus EOF.

    All counts start at 1 (additive smoothing) so no symbol ever has zero
    probability — the coder never divides by zero and any stream is encodable.
    """

    def __init__(self, vocab_size: int):
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        self.vocab = vocab_size
        # symbols 0..vocab-1, EOF at index vocab
        self.counts = [1] * (vocab_size + 1)
        self.total = vocab_size + 1

    @property
    def eof_index(self) -> int:
        return self.vocab

    def cumulative(self, sym: int) -> int:
        """Cumulative frequency below `sym`."""
        return sum(self.counts[:sym])

    def update(self, sym: int) -> None:
        """Learn a symbol (adaptive model)."""
        self.counts[sym] += 1
        self.total += 1
        if self.total > MAX_TOTAL:
            for i in range(len(self.counts)):
                self.counts[i] = (self.counts[i] + 1) // 2
            self.total = sum(self.counts)

    def grow(self, new_vocab_size: int) -> None:
        """Extend the model when the tenant lexicon grows at runtime.

        The old EOF slot becomes a regular symbol (its prior of 1 is
        preserved); the new EOF moves to `new_vocab_size`. Safe to call
        repeatedly; a no-op when the size did not grow. This is what makes
        the entropy gate dynamic-lexicon-safe: new tokens get ids and
        probabilities without ever out-of-ranging the model.
        """
        if new_vocab_size <= self.vocab:
            return
        added = new_vocab_size - self.vocab
        self.counts.extend([1] * added)
        self.total += added
        self.vocab = new_vocab_size

    def clone(self) -> "AdaptiveModel":
        m = AdaptiveModel(self.vocab)
        m.counts = list(self.counts)
        m.total = self.total
        return m


class _Encoder:
    def __init__(self, model: AdaptiveModel):
        self.model = model
        self.low = 0
        self.high = TOP_VALUE
        self.pending = 0
        self.bits: list[int] = []

    def _bit_plus_follow(self, bit: int) -> None:
        self.bits.append(bit)
        while self.pending > 0:
            self.bits.append(1 - bit)
            self.pending -= 1

    def encode(self, sym: int) -> None:
        m = self.model
        rng = self.high - self.low + 1
        lo_cum = m.cumulative(sym)
        hi_cum = lo_cum + m.counts[sym]
        self.high = self.low + (rng * hi_cum) // m.total - 1
        self.low = self.low + (rng * lo_cum) // m.total
        while True:
            if self.high < HALF:
                self._bit_plus_follow(0)
            elif self.low >= HALF:
                self._bit_plus_follow(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.pending += 1
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break
            self.low = (self.low << 1) & TOP_VALUE
            self.high = ((self.high << 1) | 1) & TOP_VALUE
        m.update(sym)

    def finish(self) -> list[int]:
        self.pending += 1
        if self.low < FIRST_QTR:
            self._bit_plus_follow(0)
        else:
            self._bit_plus_follow(1)
        return self.bits


class _Decoder:
    def __init__(self, model: AdaptiveModel, bits: list[int]):
        self.model = model
        self.bits = bits
        self.pos = 0
        self.low = 0
        self.high = TOP_VALUE
        self.value = 0
        for _ in range(16):
            self.value = ((self.value << 1) | self._read_bit()) & TOP_VALUE

    def _read_bit(self) -> int:
        if self.pos < len(self.bits):
            b = self.bits[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self) -> int:
        m = self.model
        rng = self.high - self.low + 1
        cum = ((self.value - self.low + 1) * m.total - 1) // rng
        sym = 0
        acc = 0
        while acc + m.counts[sym] <= cum:
            acc += m.counts[sym]
            sym += 1
        lo_cum = acc
        hi_cum = acc + m.counts[sym]
        self.high = self.low + (rng * hi_cum) // m.total - 1
        self.low = self.low + (rng * lo_cum) // m.total
        while True:
            if self.high < HALF:
                pass  # 0 bit already reflected in `value`
            elif self.low >= HALF:
                self.value -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= FIRST_QTR and self.high < THIRD_QTR:
                self.value -= FIRST_QTR
                self.low -= FIRST_QTR
                self.high -= FIRST_QTR
            else:
                break
            self.low = (self.low << 1) & TOP_VALUE
            self.high = ((self.high << 1) | 1) & TOP_VALUE
            self.value = ((self.value << 1) | self._read_bit()) & TOP_VALUE
        m.update(sym)
        return sym


def encode_symbols(symbols: list[int], vocab_size: int, model: AdaptiveModel | None = None) -> list[int]:
    """Encode a symbol stream (0..vocab-1) with an adaptive model.

    If no model is given, a fresh AdaptiveModel is used. The stream is
    terminated with the EOF symbol so decoding is self-delimiting.
    """
    m = model.clone() if model is not None else AdaptiveModel(vocab_size)
    enc = _Encoder(m)
    for s in symbols:
        if not 0 <= s < vocab_size:
            raise ValueError(f"symbol {s} out of range [0,{vocab_size})")
        enc.encode(s)
    enc.encode(m.eof_index)
    return enc.finish()


def decode_symbols(bits: list[int], vocab_size: int, model: AdaptiveModel | None = None) -> list[int]:
    """Decode a bit stream back into symbols (EOF-terminated)."""
    m = model.clone() if model is not None else AdaptiveModel(vocab_size)
    dec = _Decoder(m, bits)
    out: list[int] = []
    while True:
        s = dec.decode()
        if s == m.eof_index:
            break
        out.append(s)
    return out


def surprise_bits(symbols: list[int], vocab_size: int, model: AdaptiveModel | None = None) -> int:
    """Information content of `symbols` under `model`, in bits.

    This is the entropy-gate signal: the number of bits arithmetic coding
    needs to express the stream given what the model already knows. Redundant
    streams (well-predicted symbols) cost few bits; novel streams cost many.
    """
    return len(encode_symbols(symbols, vocab_size, model))


def information_content(symbols: list[int], vocab_size: int, model: AdaptiveModel | None = None) -> float:
    """Theoretical surprise of `symbols` under `model`: sum of -log2 p(symbol).

    This is what arithmetic coding APPROXIMATES on long streams. The encoded
    length additionally carries EOF + flush overhead (constant per message),
    which damps the signal for short events — so the gate measures the
    theoretical quantity, and the coder remains the lossless codec for the
    eidetic tier where the overhead amortizes over large payloads.
    """
    m = model.clone() if model is not None else AdaptiveModel(vocab_size)
    total = float(m.total)
    bits = 0.0
    for s in symbols:
        p = m.counts[s] / total
        bits += -math.log2(max(p, 1e-12))
    return bits


def roundtrip(symbols: list[int], vocab_size: int, model: AdaptiveModel | None = None) -> list[int]:
    """Encode then decode; used heavily by tests. Raises on mismatch."""
    bits = encode_symbols(symbols, vocab_size, model)
    return decode_symbols(bits, vocab_size, model)
