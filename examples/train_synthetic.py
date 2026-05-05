"""Brief synthetic training run and example prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ostrich_rnn.dataset import STRRecordDataset
from ostrich_rnn.encoding import collate_examples
from ostrich_rnn.model import RNNSTRDetector
from ostrich_rnn.motifs import MotifVocab
from ostrich_rnn.postprocess import calls_from_logits, calls_to_rows
from ostrich_rnn.runtime import describe_device, resolve_device
from ostrich_rnn.synthetic import SyntheticSTRGenerator
from ostrich_rnn.synthetic_io import (
    load_records_jsonl,
    make_unique_run_dir,
    resolve_records_path,
    save_synthetic_dataset,
)
from ostrich_rnn.training import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without validation loss improvement.")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--synthetic-data", default=None, help="Existing records.jsonl, data directory, or run directory.")
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-repeats-per-sequence", type=int, default=1)
    parser.add_argument("--max-repeats-per-sequence", type=int, default=3)
    parser.add_argument("--min-motif-len", type=int, default=1)
    parser.add_argument("--max-motif-len", type=int, default=6)
    parser.add_argument("--max-copy-number", type=int, default=2000)
    parser.add_argument("--substitution-rate", type=float, default=0.0)
    parser.add_argument("--insertion-rate", type=float, default=0.0)
    parser.add_argument("--deletion-rate", type=float, default=0.0)
    parser.add_argument("--motif-interruption-rate", type=float, default=0.0)
    parser.add_argument("--interruption-min-len", type=int, default=None)
    parser.add_argument("--interruption-max-len", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"selected_device={describe_device(device)}")
    motif_vocab = MotifVocab.build(max_len=args.max_motif_len)
    run_dir = make_unique_run_dir(args.run_root, name=args.run_name)
    print(f"run_dir={run_dir}")

    generator = None
    if args.synthetic_data:
        records_path = resolve_records_path(args.synthetic_data)
        config_path = records_path.parent / "generation_config.json"
        if config_path.exists():
            with config_path.open() as handle:
                source_config = json.load(handle)
            if "motifs" in source_config:
                motif_vocab = MotifVocab(
                    motifs=tuple(source_config["motifs"]),
                    collapse_reverse_complement=source_config.get("collapse_reverse_complement", False),
                )
        records = load_records_jsonl(records_path)
        source_metadata = {
            "source_records": str(records_path),
            "num_records": len(records),
            "validation_fraction": args.validation_fraction if args.validation_fraction is not None else 0.2,
            "motif_count": len(motif_vocab),
        }
        with (run_dir / "source_dataset.json").open("w") as handle:
            json.dump(source_metadata, handle, indent=2)
            handle.write("\n")
        print(f"loaded_synthetic_records={records_path}")
        print(f"source_dataset_metadata={run_dir / 'source_dataset.json'}")
    else:
        generator = SyntheticSTRGenerator(
            motif_vocab=motif_vocab,
            sequence_length=args.sequence_length,
            min_repeats_per_sequence=args.min_repeats_per_sequence,
            max_repeats_per_sequence=args.max_repeats_per_sequence,
            min_motif_len=args.min_motif_len,
            max_motif_len=args.max_motif_len,
            max_copy_number=args.max_copy_number,
            substitution_rate=args.substitution_rate,
            insertion_rate=args.insertion_rate,
            deletion_rate=args.deletion_rate,
            motif_interruption_rate=args.motif_interruption_rate,
            interruption_min_len=args.interruption_min_len,
            interruption_max_len=args.interruption_max_len,
            seed=args.seed,
        )
        records = generator.generate_many(args.train_size + args.val_size, prefix="synthetic")
        paths = save_synthetic_dataset(
            records,
            run_dir / "data",
            config={
                "train_size": args.train_size,
            "val_size": args.val_size,
            "patience": args.patience,
            "sequence_length": args.sequence_length,
                "min_repeats_per_sequence": args.min_repeats_per_sequence,
                "max_repeats_per_sequence": args.max_repeats_per_sequence,
                "min_motif_len": args.min_motif_len,
                "max_motif_len": args.max_motif_len,
                "max_copy_number": args.max_copy_number,
                "substitution_rate": args.substitution_rate,
                "insertion_rate": args.insertion_rate,
                "deletion_rate": args.deletion_rate,
                "motif_interruption_rate": args.motif_interruption_rate,
                "interruption_min_len": args.interruption_min_len,
                "interruption_max_len": args.interruption_max_len,
                "seed": args.seed,
                "motif_count": len(motif_vocab),
                "motifs": motif_vocab.motifs,
                "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
            },
        )
        print(f"synthetic_records={paths['records']}")
        print(f"sequence_summary={paths['sequence_summary']}")
        print(f"repeat_summary={paths['repeat_summary']}")
        print(f"dataset_summary={paths['dataset_summary']}")

    if len(records) < 2:
        raise ValueError("Need at least 2 records for a train/validation split.")
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else (0.2 if args.synthetic_data else args.val_size / (args.train_size + args.val_size))
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be greater than 0 and less than 1.")

    dataset = STRRecordDataset(records)
    model = RNNSTRDetector(motif_classes=len(motif_vocab), hidden_dim=64, num_layers=1)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=validation_fraction,
        patience=args.patience,
        checkpoint_dir=str(run_dir / "checkpoints"),
    )
    result = train_model(model, dataset, motif_vocab, config, device=device)
    print(f"best_checkpoint={result['best_checkpoint']}")

    example = generator.generate("seq_demo") if generator is not None else records[0]
    batch = collate_examples([example])
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(batch.input_ids.to(device), batch.lengths.to(device))
    calls = calls_from_logits(
        sequence_id=example.get("sequence_id", "seq_demo"),
        sequence=example["sequence"],
        state_logits=outputs.state_logits[0].cpu(),
        motif_logits=outputs.motif_logits[0].cpu(),
        motif_vocab=motif_vocab,
    )
    print("truth:", example["repeats"])
    print("predicted:", calls_to_rows(calls))


if __name__ == "__main__":
    main()
