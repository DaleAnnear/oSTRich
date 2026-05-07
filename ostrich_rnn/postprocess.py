"""Convert base-level predictions into STR tract tables."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .labels import RepeatState
from .motifs import DNA_ALPHABET, MotifVocab, canonical_motif, rotations, shortest_period


@dataclass
class RepeatCall:
    sequence_id: str
    start: int
    end: int
    length_bp: int
    motif: str
    motif_length: int
    copy_number: float
    confidence: float


def repeat_mask_from_states(states: list[int]) -> list[bool]:
    return [state in (RepeatState.START, RepeatState.INSIDE, RepeatState.END) for state in states]


def contiguous_regions(mask: list[bool], min_length: int = 1, merge_gap: int = 0) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            regions.append((start, idx))
            start = None
    if start is not None:
        regions.append((start, len(mask)))

    if merge_gap > 0 and regions:
        merged = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        regions = merged

    return [(start, end) for start, end in regions if end - start >= min_length]


def estimate_motif_from_sequence(
    tract: str,
    motif_vocab: MotifVocab,
    max_motif_len: int | None = None,
    length_priors: dict[int, float] | None = None,
) -> tuple[str, float]:
    """Estimate dominant motif by periodicity scoring over sequence-derived candidates.

    Candidate motifs are collected from k-mers found anywhere in the tract, then
    scored against the whole tract while allowing cyclic phase shifts. This makes
    motif calling less sensitive to small boundary errors than using only the
    first k bases of the predicted tract.
    """

    clean = "".join(base for base in tract.upper() if base in DNA_ALPHABET)
    if not clean:
        return motif_vocab.decode(0), 0.0

    vocab_max_len = max(len(motif) for motif in motif_vocab.motifs)
    max_motif_len = min(max_motif_len or vocab_max_len, vocab_max_len, len(clean))
    candidates: set[str] = set()
    for k in range(1, max_motif_len + 1):
        for start in range(0, len(clean) - k + 1):
            motif = shortest_period(clean[start : start + k])
            canonical = canonical_motif(motif, motif_vocab.collapse_reverse_complement)
            if motif_vocab.get(canonical) is not None:
                candidates.add(canonical)

    if not candidates:
        return motif_vocab.decode(0), 0.0

    best_motif = motif_vocab.decode(0)
    best_score = -1.0
    best_rank = (-1.0, -1.0, 0)
    for motif in candidates:
        score = periodicity_score(clean, motif)
        length_prior = (length_priors or {}).get(len(motif), 0.0)
        rank = (score, length_prior, -len(motif))
        if rank > best_rank:
            best_rank = rank
            best_score = score
            best_motif = motif
    return best_motif, best_score


def periodicity_score(tract: str, motif: str) -> float:
    """Return the best match fraction between a tract and any motif phase."""

    if not tract or not motif:
        return 0.0
    best = 0.0
    for phased_motif in rotations(motif):
        tiled = (phased_motif * ((len(tract) // len(phased_motif)) + 1))[: len(tract)]
        matches = sum(a == b for a, b in zip(tract, tiled))
        best = max(best, matches / len(tract))
    return best


def calls_from_logits(
    sequence_id: str,
    sequence: str,
    state_logits: torch.Tensor,
    motif_logits: torch.Tensor,
    motif_vocab: MotifVocab,
    motif_length_logits: torch.Tensor | None = None,
    min_repeat_length: int = 3,
    merge_gap: int = 1,
    min_sequence_motif_score: float = 0.30,
    sequence_motif_confidence_weight: float = 0.8,
) -> list[RepeatCall]:
    """Create repeat calls from raw model logits for one sequence."""

    length = len(sequence)
    state_probs = torch.softmax(state_logits[:length], dim=-1)
    motif_probs = torch.softmax(motif_logits[:length], dim=-1)
    motif_length_probs = torch.softmax(motif_length_logits[:length], dim=-1) if motif_length_logits is not None else None
    states = state_probs.argmax(dim=-1).cpu().tolist()
    regions = contiguous_regions(repeat_mask_from_states(states), min_repeat_length, merge_gap)

    calls: list[RepeatCall] = []
    repeat_prob = state_probs[:, [RepeatState.START, RepeatState.INSIDE, RepeatState.END]].sum(dim=-1)
    motif_pred = motif_probs.argmax(dim=-1)
    for start, end in regions:
        tract = sequence[start:end]
        model_motifs = motif_pred[start:end].cpu().tolist()
        motif_id, count = Counter(model_motifs).most_common(1)[0]
        model_motif = motif_vocab.decode(motif_id)
        length_priors = None
        if motif_length_probs is not None:
            mean_length_probs = motif_length_probs[start:end].mean(dim=0)
            length_priors = {idx + 1: float(value.item()) for idx, value in enumerate(mean_length_probs)}
        seq_motif, seq_score = estimate_motif_from_sequence(tract, motif_vocab, length_priors=length_priors)
        motif = seq_motif if seq_score >= min_sequence_motif_score else model_motif
        motif_length = len(motif)
        selected_motif_id = motif_vocab.get(motif, motif_id)
        model_support = motif_probs[start:end, int(selected_motif_id)].mean()
        motif_confidence = sequence_motif_confidence_weight * seq_score + (
            1.0 - sequence_motif_confidence_weight
        ) * float(model_support.item())
        confidence = float((repeat_prob[start:end].mean() * motif_confidence).item())
        calls.append(
            RepeatCall(
                sequence_id=sequence_id,
                start=start,
                end=end,
                length_bp=end - start,
                motif=motif,
                motif_length=motif_length,
                copy_number=(end - start) / motif_length,
                confidence=confidence,
            )
        )
    return calls


def calls_to_rows(calls: list[RepeatCall]) -> list[dict]:
    rows = []
    for call in calls:
        row = asdict(call)
        row["copy_number"] = round(row["copy_number"], 2)
        row["confidence"] = round(row["confidence"], 4)
        rows.append(row)
    return rows


def write_repeat_csv(calls: list[RepeatCall], output_csv: str | Path) -> None:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "start",
        "end",
        "length_bp",
        "motif",
        "motif_length",
        "copy_number",
        "confidence",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calls_to_rows(calls))
