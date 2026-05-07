"""Synthetic STR training-data generator."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable

import numpy as np

from .labels import PAD_MOTIF_LABEL, PAD_MOTIF_LENGTH_LABEL, RepeatState
from .motifs import DNA_ALPHABET, MotifVocab, canonical_motif


@dataclass
class RepeatAnnotation:
    start: int
    end: int
    length_bp: int
    motifs: list[str]
    motif_ids: list[int]
    copy_number: float
    is_compound: bool
    interruption_count: int = 0
    interruptions: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "length_bp": self.length_bp,
            "motifs": self.motifs,
            "motif_ids": self.motif_ids,
            "copy_number": self.copy_number,
            "is_compound": self.is_compound,
            "interruption_count": self.interruption_count,
            "interruptions": self.interruptions,
        }


@dataclass
class SyntheticSTRGenerator:
    """Create synthetic DNA sequences with known STR annotations.

    Long copy numbers are supported, but a repeat can only be inserted if the
    resulting tract fits within the configured sequence length.
    """

    motif_vocab: MotifVocab
    sequence_length: int = 1024
    min_repeats_per_sequence: int = 0
    max_repeats_per_sequence: int = 3
    min_motif_len: int = 1
    max_motif_len: int = 6
    min_copy_number: int = 3
    max_copy_number: int = 2000
    substitution_rate: float = 0.0
    insertion_rate: float = 0.0
    deletion_rate: float = 0.0
    motif_interruption_rate: float = 0.0
    interruption_min_len: int | None = None
    interruption_max_len: int | None = None
    compound_probability: float = 0.2
    max_compound_motifs: int = 3
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

    def random_background(self, length: int | None = None) -> str:
        length = self.sequence_length if length is None else length
        return "".join(self.rng.choices(DNA_ALPHABET, k=length))

    def sample_motif(self) -> str:
        candidates_by_length: dict[int, list[str]] = {}
        for motif in self.motif_vocab.motifs:
            if self.min_motif_len <= len(motif) <= self.max_motif_len:
                candidates_by_length.setdefault(len(motif), []).append(motif)
        if not candidates_by_length:
            raise ValueError(
                f"No motifs available between lengths {self.min_motif_len} and {self.max_motif_len}."
            )
        motif_length = self.rng.choice(sorted(candidates_by_length))
        return self.rng.choice(candidates_by_length[motif_length])

    def mutate_repeat(self, repeat: str) -> str:
        mutated: list[str] = []
        for base in repeat:
            if self.rng.random() < self.deletion_rate:
                continue
            if self.rng.random() < self.substitution_rate:
                choices = [b for b in DNA_ALPHABET if b != base]
                mutated.append(self.rng.choice(choices))
            else:
                mutated.append(base)
            if self.rng.random() < self.insertion_rate:
                mutated.append(self.rng.choice(DNA_ALPHABET))
        return "".join(mutated)

    def random_interruption(self, motif: str) -> str:
        """Create a non-motif sequence used to interrupt a tandem tract."""

        min_len = self.interruption_min_len or len(motif)
        max_len = self.interruption_max_len or len(motif)
        if min_len > max_len:
            raise ValueError("interruption_min_len cannot be greater than interruption_max_len.")
        length = self.rng.randint(min_len, max_len)
        for _ in range(20):
            interruption = "".join(self.rng.choices(DNA_ALPHABET, k=length))
            if interruption != motif:
                return interruption
        return "".join(self.rng.choice([base for base in DNA_ALPHABET if base != motif[0]]) for _ in range(length))

    def build_repeat(self) -> tuple[str, list[str], list[tuple[int, int, str]], list[dict]]:
        """Build one perfect/imperfect simple or compound repeat tract.

        Returns the tract sequence, motif list, per-segment motif spans, and
        interruption annotations relative to the tract start.
        """

        is_compound = self.rng.random() < self.compound_probability
        n_motifs = self.rng.randint(2, self.max_compound_motifs) if is_compound else 1
        motifs = [self.sample_motif() for _ in range(n_motifs)]

        pieces: list[str] = []
        spans: list[tuple[int, int, str]] = []
        interruptions: list[dict] = []
        cursor = 0
        for motif in motifs:
            max_copies_by_length = max(1, self.sequence_length // max(1, len(motif)))
            high = min(self.max_copy_number, max_copies_by_length)
            copies = self.rng.randint(self.min_copy_number, max(self.min_copy_number, high))
            segment_start = cursor
            for _ in range(copies):
                is_interruption = self.rng.random() < self.motif_interruption_rate
                unit = self.random_interruption(motif) if is_interruption else motif
                unit_start = cursor
                unit = self.mutate_repeat(unit)
                if not unit:
                    unit = motif[:1]
                pieces.append(unit)
                cursor += len(unit)
                if is_interruption:
                    interruptions.append(
                        {
                            "start": unit_start,
                            "end": cursor,
                            "sequence": unit,
                            "expected_motif": motif,
                            "type": "motif_interruption",
                        }
                    )
            spans.append((segment_start, cursor, motif))
        return "".join(pieces), motifs, spans, interruptions

    def generate(self, sequence_id: str = "seq_000") -> dict:
        sequence = list(self.random_background())
        state_labels = [RepeatState.OUTSIDE] * self.sequence_length
        motif_labels = [PAD_MOTIF_LABEL] * self.sequence_length
        motif_length_labels = [PAD_MOTIF_LENGTH_LABEL] * self.sequence_length
        repeats: list[RepeatAnnotation] = []
        occupied = np.zeros(self.sequence_length, dtype=bool)

        requested = self.rng.randint(self.min_repeats_per_sequence, self.max_repeats_per_sequence)
        attempts = 0
        while len(repeats) < requested and attempts < requested * 50 + 20:
            attempts += 1
            tract, motifs, spans, interruptions = self.build_repeat()
            if len(tract) >= self.sequence_length:
                continue
            start = self.rng.randint(0, self.sequence_length - len(tract))
            end = start + len(tract)
            if occupied[max(0, start - 1) : min(self.sequence_length, end + 1)].any():
                continue

            sequence[start:end] = list(tract)
            occupied[start:end] = True
            if len(tract) == 1:
                state_labels[start] = RepeatState.START
            else:
                state_labels[start] = RepeatState.START
                for pos in range(start + 1, end - 1):
                    state_labels[pos] = RepeatState.INSIDE
                state_labels[end - 1] = RepeatState.END

            motif_ids: list[int] = []
            for rel_start, rel_end, motif in spans:
                motif_id = self.motif_vocab.encode(motif)
                motif_length_id = len(motif) - 1
                motif_ids.append(motif_id)
                for pos in range(start + rel_start, min(start + rel_end, end)):
                    motif_labels[pos] = motif_id
                    motif_length_labels[pos] = motif_length_id

            dominant_len = len(motifs[0])
            repeats.append(
                RepeatAnnotation(
                    start=start,
                    end=end,
                    length_bp=end - start,
                    motifs=[canonical_motif(m, self.motif_vocab.collapse_reverse_complement) for m in motifs],
                    motif_ids=motif_ids,
                    copy_number=(end - start) / dominant_len,
                    is_compound=len(motifs) > 1,
                    interruption_count=len(interruptions),
                    interruptions=[
                        {
                            **interruption,
                            "start": start + int(interruption["start"]),
                            "end": start + int(interruption["end"]),
                        }
                        for interruption in interruptions
                    ],
                )
            )

        return {
            "sequence_id": sequence_id,
            "sequence": "".join(sequence),
            "state_labels": [int(label) for label in state_labels],
            "motif_labels": motif_labels,
            "motif_length_labels": motif_length_labels,
            "repeats": [repeat.as_dict() for repeat in sorted(repeats, key=lambda r: r.start)],
        }

    def generate_many(self, n: int, prefix: str = "seq") -> list[dict]:
        width = max(3, len(str(n)))
        return [self.generate(f"{prefix}_{i:0{width}d}") for i in range(n)]


def examples_to_records(examples: Iterable[dict]) -> list[dict]:
    """Keep a stable public hook for adapting synthetic and real records."""

    return list(examples)
