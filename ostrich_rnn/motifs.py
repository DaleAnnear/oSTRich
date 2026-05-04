"""Motif vocabulary and canonicalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


DNA_ALPHABET = "ACGT"


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def rotations(seq: str) -> list[str]:
    return [seq[i:] + seq[:i] for i in range(len(seq))]


def canonical_motif(motif: str, collapse_reverse_complement: bool = True) -> str:
    """Return a canonical representation for a tandem-repeat motif.

    Tandem repeats are invariant to rotation: CAG, AGC, and GCA describe the
    same cyclic motif. Optionally, reverse-complement-equivalent motifs are also
    collapsed into one class.
    """

    motif = motif.upper()
    candidates = rotations(motif)
    if collapse_reverse_complement:
        rc = reverse_complement(motif)
        candidates.extend(rotations(rc))
    return min(candidates)


def generate_motifs(
    min_len: int = 1,
    max_len: int = 6,
    collapse_reverse_complement: bool = False,
) -> list[str]:
    """Generate canonical motif classes up to `max_len`.

    The returned vocabulary keeps only motifs whose shortest periodic unit is the
    same length as the motif. For example, ACAC is represented as AC.
    """

    motifs: set[str] = set()
    for length in range(min_len, max_len + 1):
        for bases in product(DNA_ALPHABET, repeat=length):
            motif = "".join(bases)
            if shortest_period(motif) != motif:
                continue
            motifs.add(canonical_motif(motif, collapse_reverse_complement))
    return sorted(motifs, key=lambda m: (len(m), m))


def shortest_period(seq: str) -> str:
    """Return the shortest repeat unit that exactly reconstructs `seq`."""

    seq = seq.upper()
    for k in range(1, len(seq) + 1):
        if len(seq) % k == 0:
            unit = seq[:k]
            if unit * (len(seq) // k) == seq:
                return unit
    return seq


@dataclass(frozen=True)
class MotifVocab:
    """Mapping between motif strings and model class IDs."""

    motifs: tuple[str, ...]
    collapse_reverse_complement: bool = False

    @classmethod
    def build(
        cls,
        max_len: int = 6,
        extra_motifs: tuple[str, ...] = ("A", "C", "G", "T", "AC", "AG", "AT", "CG", "CAG", "CGG", "GATA"),
        collapse_reverse_complement: bool = False,
    ) -> "MotifVocab":
        motifs = set(generate_motifs(1, max_len, collapse_reverse_complement))
        for motif in extra_motifs:
            motifs.add(canonical_motif(motif, collapse_reverse_complement))
        ordered = tuple(sorted(motifs, key=lambda m: (len(m), m)))
        return cls(ordered, collapse_reverse_complement)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_index", {motif: i for i, motif in enumerate(self.motifs)})

    def __len__(self) -> int:
        return len(self.motifs)

    def encode(self, motif: str) -> int:
        canonical = canonical_motif(shortest_period(motif.upper()), self.collapse_reverse_complement)
        return self._index[canonical]

    def decode(self, idx: int) -> str:
        return self.motifs[int(idx)]

    def get(self, motif: str, default: int | None = None) -> int | None:
        canonical = canonical_motif(shortest_period(motif.upper()), self.collapse_reverse_complement)
        return self._index.get(canonical, default)
