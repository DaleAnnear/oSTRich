"""Persistence and summaries for generated synthetic STR datasets."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

from .labels import PAD_MOTIF_LABEL, PAD_MOTIF_LENGTH_LABEL


def make_unique_run_dir(root: str | Path = "runs", prefix: str = "synthetic", name: str | None = None) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if name:
        candidate = root / name
        if candidate.exists():
            candidate = root / f"{name}_{uuid4().hex[:8]}"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = root / f"{prefix}_{stamp}_{uuid4().hex[:8]}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def gc_content(sequence: str) -> float:
    sequence = sequence.upper()
    acgt = sum(base in "ACGT" for base in sequence)
    if acgt == 0:
        return 0.0
    gc = sequence.count("G") + sequence.count("C")
    return gc / acgt


def repeat_label(repeat: dict) -> str:
    motifs = "+".join(str(motif) for motif in repeat.get("motifs", []))
    return f"{repeat['start']}-{repeat['end']}:{motifs}:{repeat['length_bp']}bp"


def sequence_summary_row(record: dict) -> dict:
    sequence = record["sequence"]
    repeats = record.get("repeats", [])
    total_repeat_bp = sum(int(repeat["length_bp"]) for repeat in repeats)
    total_interruptions = sum(int(repeat.get("interruption_count", 0)) for repeat in repeats)
    return {
        "sequence_id": record["sequence_id"],
        "length_bp": len(sequence),
        "gc_content": round(gc_content(sequence), 6),
        "n_count": sequence.upper().count("N"),
        "repeat_count": len(repeats),
        "interruption_count": total_interruptions,
        "total_repeat_bp": total_repeat_bp,
        "repeat_fraction": round(total_repeat_bp / len(sequence), 6) if sequence else 0.0,
        "motifs": ";".join("+".join(repeat.get("motifs", [])) for repeat in repeats),
        "repeats": ";".join(repeat_label(repeat) for repeat in repeats),
    }


def repeat_summary_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for record in records:
        for idx, repeat in enumerate(record.get("repeats", [])):
            rows.append(
                {
                    "sequence_id": record["sequence_id"],
                    "repeat_index": idx,
                    "start": repeat["start"],
                    "end": repeat["end"],
                    "length_bp": repeat["length_bp"],
                    "motifs": "+".join(repeat.get("motifs", [])),
                    "motif_lengths": "+".join(str(len(motif)) for motif in repeat.get("motifs", [])),
                    "motif_ids": "+".join(str(item) for item in repeat.get("motif_ids", [])),
                    "copy_number": round(float(repeat.get("copy_number", 0.0)), 6),
                    "is_compound": bool(repeat.get("is_compound", False)),
                    "interruption_count": int(repeat.get("interruption_count", 0)),
                    "interruptions": ";".join(
                        f"{item.get('start')}-{item.get('end')}:{item.get('sequence')}"
                        for item in repeat.get("interruptions", [])
                    ),
                }
            )
    return rows


def dataset_summary(records: list[dict]) -> dict:
    lengths = [len(record["sequence"]) for record in records]
    gc_values = [gc_content(record["sequence"]) for record in records]
    repeat_counts = [len(record.get("repeats", [])) for record in records]
    repeat_lengths = [int(repeat["length_bp"]) for record in records for repeat in record.get("repeats", [])]
    interruption_count = sum(
        int(repeat.get("interruption_count", 0)) for record in records for repeat in record.get("repeats", [])
    )
    motif_counts: dict[str, int] = {}
    motif_length_counts: dict[str, int] = {}
    compound_count = 0
    for record in records:
        for repeat in record.get("repeats", []):
            if repeat.get("is_compound", False):
                compound_count += 1
            for motif in repeat.get("motifs", []):
                motif_counts[motif] = motif_counts.get(motif, 0) + 1
                motif_length = str(len(motif))
                motif_length_counts[motif_length] = motif_length_counts.get(motif_length, 0) + 1
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_sequences": len(records),
        "sequence_length_min": min(lengths) if lengths else 0,
        "sequence_length_mean": mean(lengths) if lengths else 0.0,
        "sequence_length_median": median(lengths) if lengths else 0.0,
        "sequence_length_max": max(lengths) if lengths else 0,
        "gc_content_mean": mean(gc_values) if gc_values else 0.0,
        "repeat_count_total": sum(repeat_counts),
        "repeat_count_mean_per_sequence": mean(repeat_counts) if repeat_counts else 0.0,
        "repeat_count_max_per_sequence": max(repeat_counts) if repeat_counts else 0,
        "repeat_length_min": min(repeat_lengths) if repeat_lengths else 0,
        "repeat_length_mean": mean(repeat_lengths) if repeat_lengths else 0.0,
        "repeat_length_median": median(repeat_lengths) if repeat_lengths else 0.0,
        "repeat_length_max": max(repeat_lengths) if repeat_lengths else 0,
        "compound_repeat_count": compound_count,
        "interruption_count_total": interruption_count,
        "motif_counts": dict(sorted(motif_counts.items())),
        "motif_length_counts": dict(sorted(motif_length_counts.items(), key=lambda item: int(item[0]))),
    }


def save_synthetic_dataset(records: list[dict], output_dir: str | Path, config: dict | None = None) -> dict:
    """Save synthetic records plus sequence/repeat/dataset summaries."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    fasta_path = output_dir / "sequences.fa"
    sequence_summary_path = output_dir / "sequence_summary.csv"
    repeat_summary_path = output_dir / "repeat_summary.csv"
    dataset_summary_path = output_dir / "dataset_summary.json"
    config_path = output_dir / "generation_config.json"

    with records_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    with fasta_path.open("w") as handle:
        for record in records:
            handle.write(f">{record['sequence_id']}\n")
            sequence = record["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

    sequence_rows = [sequence_summary_row(record) for record in records]
    with sequence_summary_path.open("w", newline="") as handle:
        fieldnames = [
            "sequence_id",
            "length_bp",
            "gc_content",
            "n_count",
            "repeat_count",
            "interruption_count",
            "total_repeat_bp",
            "repeat_fraction",
            "motifs",
            "repeats",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sequence_rows)

    repeat_rows = repeat_summary_rows(records)
    with repeat_summary_path.open("w", newline="") as handle:
        fieldnames = [
            "sequence_id",
            "repeat_index",
            "start",
            "end",
            "length_bp",
            "motifs",
            "motif_lengths",
            "motif_ids",
            "copy_number",
            "is_compound",
            "interruption_count",
            "interruptions",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repeat_rows)

    summary = dataset_summary(records)
    with dataset_summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with config_path.open("w") as handle:
        json.dump(config or {}, handle, indent=2)
        handle.write("\n")

    return {
        "records": str(records_path),
        "fasta": str(fasta_path),
        "sequence_summary": str(sequence_summary_path),
        "repeat_summary": str(repeat_summary_path),
        "dataset_summary": str(dataset_summary_path),
        "generation_config": str(config_path),
    }


def load_records_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def add_motif_length_labels(records: list[dict], motif_vocab) -> list[dict]:
    """Backfill per-base motif-length labels for generated or loaded records.

    Labels are 0-based classes: motif length 1 -> class 0, motif length 6 ->
    class 5. Outside-repeat bases use PAD_MOTIF_LENGTH_LABEL.
    """

    for record in records:
        sequence_length = len(record["sequence"])
        existing = record.get("motif_length_labels")
        if existing is not None and len(existing) == sequence_length:
            continue
        labels = [PAD_MOTIF_LENGTH_LABEL] * sequence_length
        motif_labels = record.get("motif_labels")
        if motif_labels is not None:
            for idx, motif_id in enumerate(motif_labels[:sequence_length]):
                if int(motif_id) != PAD_MOTIF_LABEL:
                    labels[idx] = len(motif_vocab.decode(int(motif_id))) - 1
        else:
            for repeat in record.get("repeats", []):
                motifs = repeat.get("motifs", [])
                if not motifs:
                    continue
                motif_length_id = len(motifs[0]) - 1
                for idx in range(int(repeat["start"]), min(sequence_length, int(repeat["end"]))):
                    labels[idx] = motif_length_id
        record["motif_length_labels"] = labels
    return records


def resolve_records_path(path: str | Path) -> Path:
    """Resolve a saved synthetic dataset path to its records.jsonl file.

    Accepts any of these:
    - /path/to/records.jsonl
    - /path/to/data/
    - /path/to/run_dir/ where run_dir/data/records.jsonl exists
    """

    path = Path(path)
    if path.is_file():
        return path
    candidates = [path / "records.jsonl", path / "data" / "records.jsonl"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find records.jsonl from '{path}'. "
        "Provide records.jsonl, its data directory, or a run directory containing data/records.jsonl."
    )
