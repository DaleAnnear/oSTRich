"""Convert base-level predictions into STR tract tables."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .labels import RepeatState
from .motifs import MotifVocab, canonical_motif, shortest_period


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
    max_motif_len: int = 8,
) -> tuple[str, float]:
    """Estimate dominant motif by minimizing mismatch rate over periodic units."""

    tract = tract.upper().replace("N", "")
    if not tract:
        return motif_vocab.decode(0), 0.0

    best_motif = tract[:1]
    best_score = -1.0
    for k in range(1, min(max_motif_len, len(tract)) + 1):
        motif = tract[:k]
        motif = shortest_period(motif)
        tiled = (motif * ((len(tract) // len(motif)) + 1))[: len(tract)]
        matches = sum(a == b for a, b in zip(tract, tiled))
        score = matches / len(tract)
        canonical = canonical_motif(motif, motif_vocab.collapse_reverse_complement)
        if motif_vocab.get(canonical) is not None and score > best_score:
            best_score = score
            best_motif = canonical
    return best_motif, best_score


def calls_from_logits(
    sequence_id: str,
    sequence: str,
    state_logits: torch.Tensor,
    motif_logits: torch.Tensor,
    motif_vocab: MotifVocab,
    min_repeat_length: int = 3,
    merge_gap: int = 1,
) -> list[RepeatCall]:
    """Create repeat calls from raw model logits for one sequence."""

    length = len(sequence)
    state_probs = torch.softmax(state_logits[:length], dim=-1)
    motif_probs = torch.softmax(motif_logits[:length], dim=-1)
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
        seq_motif, seq_score = estimate_motif_from_sequence(tract, motif_vocab)
        motif = seq_motif if seq_score >= 0.70 else model_motif
        motif_length = len(motif)
        confidence = float((repeat_prob[start:end].mean() * motif_probs[start:end, motif_id].mean()).item())
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
