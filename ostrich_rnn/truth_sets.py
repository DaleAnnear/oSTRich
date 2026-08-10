"""Utilities for external genomic truth sets."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import re
from typing import Iterable, TextIO


TRGT_ATTRIBUTE_PATTERN = re.compile(r"<(?P<kind>TR|VC):(?P<value>[^>]+)>")


def open_text(path: str | Path, mode: str = "rt") -> TextIO:
    """Open plain text or gzip-compressed text from a path."""

    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return path.open(mode)


def parse_bed_attributes(attribute_text: str) -> dict[str, str]:
    """Parse semicolon-delimited BED attributes such as ID=...;MOTIFS=..."""

    attributes: dict[str, str] = {}
    for item in attribute_text.split(";"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        attributes[key] = value
    return attributes


def parse_trgt_structure(structure: str) -> tuple[str, str | None]:
    """Return a normalized TRGT structure kind and the raw structure payload."""

    match = TRGT_ATTRIBUTE_PATTERN.fullmatch(structure)
    if not match:
        return "unknown", None
    kind = "variation_cluster" if match.group("kind") == "VC" else "isolated_repeat"
    return kind, match.group("value")


def iter_trexplorer_bed(path: str | Path) -> Iterable[dict]:
    """Yield normalized rows from a TRExplorer TRGT BED file.

    Coordinates remain BED-style: 0-based, inclusive start, exclusive end.
    """

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                raise ValueError(f"Expected at least 4 BED columns at line {line_number}: {line!r}")
            attributes = parse_bed_attributes(fields[3])
            kind, structure = parse_trgt_structure(attributes.get("STRUC", ""))
            motifs = [motif for motif in attributes.get("MOTIFS", "").split(",") if motif]
            yield {
                "chrom": fields[0],
                "start": int(fields[1]),
                "end": int(fields[2]),
                "id": attributes.get("ID", ""),
                "motifs": motifs,
                "structure": structure,
                "kind": kind,
                "line": line,
            }


def summarize_trexplorer_bed(path: str | Path) -> dict:
    """Summarize a TRExplorer BED file without loading it into memory."""

    kind_counts: Counter[str] = Counter()
    chrom_counts: Counter[str] = Counter()
    motif_length_counts: Counter[str] = Counter()
    total_bp = 0
    min_length: int | None = None
    max_length = 0
    examples: dict[str, dict] = {}

    for row in iter_trexplorer_bed(path):
        length = row["end"] - row["start"]
        kind = row["kind"]
        kind_counts[kind] += 1
        chrom_counts[row["chrom"]] += 1
        total_bp += length
        min_length = length if min_length is None else min(min_length, length)
        max_length = max(max_length, length)
        examples.setdefault(
            kind,
            {
                "chrom": row["chrom"],
                "start": row["start"],
                "end": row["end"],
                "id": row["id"],
                "motifs": row["motifs"],
                "structure": row["structure"],
            },
        )
        for motif in row["motifs"]:
            motif_length_counts[str(len(motif))] += 1

    total_records = sum(kind_counts.values())
    return {
        "path": str(Path(path)),
        "total_records": total_records,
        "total_bp": total_bp,
        "length_min": min_length or 0,
        "length_max": max_length,
        "kind_counts": dict(sorted(kind_counts.items())),
        "chrom_counts": dict(sorted(chrom_counts.items())),
        "motif_length_counts": dict(sorted(motif_length_counts.items(), key=lambda item: int(item[0]))),
        "examples": examples,
    }


def extract_variation_clusters(input_path: str | Path, output_path: str | Path) -> int:
    """Write only variation-cluster rows to a BED/BED.GZ file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wt"
    opener = gzip.open if output_path.suffix == ".gz" else Path.open
    count = 0
    with opener(output_path, mode) as output:
        for row in iter_trexplorer_bed(input_path):
            if row["kind"] != "variation_cluster":
                continue
            output.write(row["line"] + "\n")
            count += 1
    return count


def write_summary(path: str | Path, summary: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
