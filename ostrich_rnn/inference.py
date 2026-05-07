"""FASTA inference entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import STRRecordDataset
from .encoding import collate_examples
from .model import RNNSTRDetector
from .motifs import MotifVocab
from .postprocess import RepeatCall, calls_from_logits, write_repeat_csv
from .runtime import describe_device, resolve_device


def read_fasta(path: str | Path) -> list[dict]:
    records: list[dict] = []
    name: str | None = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append({"sequence_id": name, "sequence": "".join(chunks).upper()})
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            records.append({"sequence_id": name, "sequence": "".join(chunks).upper()})
    return records


def window_records(records: list[dict], window_size: int, stride: int) -> list[dict]:
    """Split long FASTA records into overlapping model windows."""

    windows: list[dict] = []
    for record in records:
        sequence = record["sequence"]
        if len(sequence) <= window_size:
            windows.append({**record, "offset": 0})
            continue
        for start in range(0, len(sequence), stride):
            end = min(start + window_size, len(sequence))
            windows.append(
                {
                    "sequence_id": record["sequence_id"],
                    "sequence": sequence[start:end],
                    "offset": start,
                }
            )
            if end == len(sequence):
                break
    return windows


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[RNNSTRDetector, MotifVocab]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    motif_vocab = MotifVocab(
        motifs=tuple(checkpoint["motifs"]),
        collapse_reverse_complement=checkpoint.get("collapse_reverse_complement", False),
    )
    model_config = checkpoint.get("model_config") or {"motif_classes": len(motif_vocab)}
    model_config["motif_classes"] = len(motif_vocab)
    model_config.setdefault("motif_length_classes", max(len(motif) for motif in motif_vocab.motifs))
    model = RNNSTRDetector(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, motif_vocab


@torch.no_grad()
def predict_fasta(
    fasta_path: str | Path,
    checkpoint_path: str | Path,
    output_csv: str | Path,
    batch_size: int = 8,
    window_size: int = 2048,
    stride: int = 1800,
    device: str | torch.device | None = "auto",
) -> list[RepeatCall]:
    device = resolve_device(device)
    print(f"inference_device={describe_device(device)}")
    model, motif_vocab = load_model_from_checkpoint(checkpoint_path, device)
    records = window_records(read_fasta(fasta_path), window_size=window_size, stride=stride)
    calls = predict_records(records, model, motif_vocab, batch_size=batch_size, device=device)
    write_repeat_csv(calls, output_csv)
    return calls


@torch.no_grad()
def predict_records(
    records: list[dict],
    model: RNNSTRDetector,
    motif_vocab: MotifVocab,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> list[RepeatCall]:
    device = torch.device(next(model.parameters()).device) if device is None else resolve_device(device)
    loader = DataLoader(STRRecordDataset(records), batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    calls: list[RepeatCall] = []
    seen = 0
    for batch in loader:
        outputs = model(batch.input_ids.to(device), batch.lengths.to(device))
        for i, sequence in enumerate(batch.sequences):
            record = records[seen + i]
            offset = int(record.get("offset", 0))
            window_calls = calls_from_logits(
                sequence_id=batch.sequence_ids[i],
                sequence=sequence,
                state_logits=outputs.state_logits[i].cpu(),
                motif_logits=outputs.motif_logits[i].cpu(),
                motif_length_logits=outputs.motif_length_logits[i].cpu(),
                motif_vocab=motif_vocab,
            )
            for call in window_calls:
                call.start += offset
                call.end += offset
                calls.append(call)
        seen += len(batch.sequences)
    return deduplicate_overlapping_calls(calls)


def deduplicate_overlapping_calls(calls: list[RepeatCall]) -> list[RepeatCall]:
    calls = sorted(calls, key=lambda c: (c.sequence_id, c.start, -c.confidence))
    kept: list[RepeatCall] = []
    for call in calls:
        overlaps = [
            idx
            for idx, existing in enumerate(kept)
            if existing.sequence_id == call.sequence_id and max(existing.start, call.start) < min(existing.end, call.end)
        ]
        if not overlaps:
            kept.append(call)
            continue
        best_idx = max(overlaps, key=lambda idx: kept[idx].confidence)
        if call.confidence > kept[best_idx].confidence:
            kept[best_idx] = call
    return sorted(kept, key=lambda c: (c.sequence_id, c.start, c.end))


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect STRs in FASTA with a trained RNN model.")
    parser.add_argument("fasta")
    parser.add_argument("output_csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1800)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    predict_fasta(
        fasta_path=args.fasta,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
    )


if __name__ == "__main__":
    main()
