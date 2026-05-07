"""Brief synthetic training run and example prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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
    add_motif_length_labels,
    load_records_jsonl,
    make_unique_run_dir,
    resolve_records_path,
    save_synthetic_dataset,
)
from ostrich_rnn.training import TrainConfig, train_model


CURRICULUM_PHASES = (
    {
        "name": "phase1_perfect",
        "description": "Perfect simple repeats only.",
        "substitution_rate": 0.0,
        "insertion_rate": 0.0,
        "deletion_rate": 0.0,
        "motif_interruption_rate": 0.0,
        "compound_probability": 0.0,
    },
    {
        "name": "phase2_low_imperfect",
        "description": "Low substitution/insertion/deletion/interruption rates.",
        "substitution_rate": 0.002,
        "insertion_rate": 0.0005,
        "deletion_rate": 0.0005,
        "motif_interruption_rate": 0.01,
        "compound_probability": 0.05,
    },
    {
        "name": "phase3_harder_imperfect_compound",
        "description": "Harder imperfect and compound repeats.",
        "substitution_rate": 0.01,
        "insertion_rate": 0.002,
        "deletion_rate": 0.002,
        "motif_interruption_rate": 0.05,
        "compound_probability": 0.30,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many epochs without validation loss improvement.")
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--motif-loss-weight", type=float, default=0.2)
    parser.add_argument("--motif-length-loss-weight", type=float, default=0.1)
    parser.add_argument("--no-motif-class-weighting", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--synthetic-data", default=None, help="Existing records.jsonl, data directory, or run directory.")
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--curriculum", action="store_true", help="Train through perfect, low-imperfect, then harder imperfect/compound phases.")
    parser.add_argument(
        "--curriculum-phase-epochs",
        default="20,35,45",
        help="Comma-separated phase percentages of --epochs for the 3 curriculum phases, e.g. 20,35,45.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-repeats-per-sequence", type=int, default=1)
    parser.add_argument("--max-repeats-per-sequence", type=int, default=3)
    parser.add_argument("--min-motif-len", type=int, default=1)
    parser.add_argument("--max-motif-len", type=int, default=6)
    parser.add_argument("--max-copy-number", type=int, default=2000)
    parser.add_argument("--compound-probability", type=float, default=0.2)
    parser.add_argument("--max-compound-motifs", type=int, default=3)
    parser.add_argument("--substitution-rate", type=float, default=0.0)
    parser.add_argument("--insertion-rate", type=float, default=0.0)
    parser.add_argument("--deletion-rate", type=float, default=0.0)
    parser.add_argument("--motif-interruption-rate", type=float, default=0.0)
    parser.add_argument("--interruption-min-len", type=int, default=None)
    parser.add_argument("--interruption-max-len", type=int, default=None)
    return parser.parse_args()


def parse_curriculum_phase_epochs(value: str | None, total_epochs: int) -> list[int]:
    if total_epochs < 3:
        raise ValueError("--epochs must be at least 3 when --curriculum is used.")

    percentages = [float(item.strip()) for item in (value or "20,35,45").split(",")]
    if len(percentages) != 3 or any(percentage <= 0 for percentage in percentages):
        raise ValueError("--curriculum-phase-epochs must contain 3 positive percentages, e.g. 20,35,45.")
    if abs(sum(percentages) - 100.0) > 1e-6:
        raise ValueError("--curriculum-phase-epochs percentages must sum to 100.")

    raw_epochs = [total_epochs * percentage / 100.0 for percentage in percentages]
    phase_epochs = [max(1, int(value + 0.5)) for value in raw_epochs]

    while sum(phase_epochs) > total_epochs:
        candidates = [
            (phase_epochs[idx] - raw_epochs[idx], idx)
            for idx in range(3)
            if phase_epochs[idx] > 1
        ]
        if not candidates:
            break
        _, idx = max(candidates)
        phase_epochs[idx] -= 1

    while sum(phase_epochs) < total_epochs:
        candidates = [(raw_epochs[idx] - phase_epochs[idx], idx) for idx in range(3)]
        _, idx = max(candidates)
        phase_epochs[idx] += 1

    return phase_epochs


def parse_curriculum_phase_percentages(value: str | None) -> list[float]:
    percentages = [float(item.strip()) for item in (value or "20,35,45").split(",")]
    if len(percentages) != 3:
        raise ValueError("--curriculum-phase-epochs must contain 3 percentages.")
    return percentages


def make_generator(args, motif_vocab: MotifVocab, seed: int, overrides: dict) -> SyntheticSTRGenerator:
    return SyntheticSTRGenerator(
        motif_vocab=motif_vocab,
        sequence_length=args.sequence_length,
        min_repeats_per_sequence=args.min_repeats_per_sequence,
        max_repeats_per_sequence=args.max_repeats_per_sequence,
        min_motif_len=args.min_motif_len,
        max_motif_len=args.max_motif_len,
        max_copy_number=args.max_copy_number,
        substitution_rate=overrides.get("substitution_rate", args.substitution_rate),
        insertion_rate=overrides.get("insertion_rate", args.insertion_rate),
        deletion_rate=overrides.get("deletion_rate", args.deletion_rate),
        motif_interruption_rate=overrides.get("motif_interruption_rate", args.motif_interruption_rate),
        interruption_min_len=args.interruption_min_len,
        interruption_max_len=args.interruption_max_len,
        compound_probability=overrides.get("compound_probability", args.compound_probability),
        max_compound_motifs=args.max_compound_motifs,
        seed=seed,
    )


def generation_config(args, motif_vocab: MotifVocab, extra: dict | None = None) -> dict:
    config = {
        "train_size": args.train_size,
        "val_size": args.val_size,
        "patience": args.patience,
        "sequence_length": args.sequence_length,
        "min_repeats_per_sequence": args.min_repeats_per_sequence,
        "max_repeats_per_sequence": args.max_repeats_per_sequence,
        "min_motif_len": args.min_motif_len,
        "max_motif_len": args.max_motif_len,
        "max_copy_number": args.max_copy_number,
        "compound_probability": args.compound_probability,
        "max_compound_motifs": args.max_compound_motifs,
        "substitution_rate": args.substitution_rate,
        "insertion_rate": args.insertion_rate,
        "deletion_rate": args.deletion_rate,
        "motif_interruption_rate": args.motif_interruption_rate,
        "interruption_min_len": args.interruption_min_len,
        "interruption_max_len": args.interruption_max_len,
        "seed": args.seed,
        "motif_count": len(motif_vocab),
        "motif_length_classes": max(len(motif) for motif in motif_vocab.motifs),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "motif_loss_weight": args.motif_loss_weight,
        "motif_length_loss_weight": args.motif_length_loss_weight,
        "motif_class_weighting": not args.no_motif_class_weighting,
        "motifs": motif_vocab.motifs,
        "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
    }
    if extra:
        config.update(extra)
    return config


def train_records(records: list[dict], model, motif_vocab, args, validation_fraction: float, checkpoint_dir: Path, device):
    dataset = STRRecordDataset(records)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        motif_loss_weight=args.motif_loss_weight,
        motif_length_loss_weight=args.motif_length_loss_weight,
        motif_class_weighting=not args.no_motif_class_weighting,
        validation_fraction=validation_fraction,
        patience=args.patience,
        checkpoint_dir=str(checkpoint_dir),
    )
    return train_model(model, dataset, motif_vocab, config, device=device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"selected_device={describe_device(device)}")
    motif_vocab = MotifVocab.build(max_len=args.max_motif_len)
    run_dir = make_unique_run_dir(args.run_root, name=args.run_name)
    print(f"run_dir={run_dir}")

    generator = None
    if args.synthetic_data:
        if args.curriculum:
            raise ValueError("--curriculum cannot be combined with --synthetic-data.")
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
            "motif_length_classes": max(len(motif) for motif in motif_vocab.motifs),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "motif_class_weighting": not args.no_motif_class_weighting,
        }
        with (run_dir / "source_dataset.json").open("w") as handle:
            json.dump(source_metadata, handle, indent=2)
            handle.write("\n")
        print(f"loaded_synthetic_records={records_path}")
        print(f"source_dataset_metadata={run_dir / 'source_dataset.json'}")
    else:
        if args.curriculum:
            records = []
        else:
            generator = make_generator(args, motif_vocab, args.seed, {})
            records = generator.generate_many(args.train_size + args.val_size, prefix="synthetic")
            add_motif_length_labels(records, motif_vocab)
            paths = save_synthetic_dataset(
                records,
                run_dir / "data",
                config=generation_config(args, motif_vocab),
            )
            print(f"synthetic_records={paths['records']}")
            print(f"sequence_summary={paths['sequence_summary']}")
            print(f"repeat_summary={paths['repeat_summary']}")
            print(f"dataset_summary={paths['dataset_summary']}")

    if not args.curriculum:
        add_motif_length_labels(records, motif_vocab)

    if not args.curriculum and len(records) < 2:
        raise ValueError("Need at least 2 records for a train/validation split.")
    validation_fraction = (
        args.validation_fraction
        if args.validation_fraction is not None
        else (0.2 if args.synthetic_data else args.val_size / (args.train_size + args.val_size))
    )
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be greater than 0 and less than 1.")

    motif_length_classes = max(len(motif) for motif in motif_vocab.motifs)
    model = RNNSTRDetector(
        motif_classes=len(motif_vocab),
        motif_length_classes=motif_length_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    if args.curriculum:
        phase_epochs = parse_curriculum_phase_epochs(args.curriculum_phase_epochs, args.epochs)
        phase_percentages = parse_curriculum_phase_percentages(args.curriculum_phase_epochs)
        curriculum_summary = []
        final_checkpoint = None
        for idx, (phase, epochs, percentage) in enumerate(
            zip(CURRICULUM_PHASES, phase_epochs, phase_percentages),
            start=1,
        ):
            phase_name = phase["name"]
            print(
                f"curriculum_phase={idx} name={phase_name} percentage={percentage:g} "
                f"epochs={epochs} description={phase['description']}"
            )
            phase_generator = make_generator(args, motif_vocab, args.seed + idx * 1000, phase)
            phase_records = phase_generator.generate_many(args.train_size + args.val_size, prefix=phase_name)
            add_motif_length_labels(phase_records, motif_vocab)
            phase_paths = save_synthetic_dataset(
                phase_records,
                run_dir / "data" / phase_name,
                config=generation_config(
                    args,
                    motif_vocab,
                    {
                        "curriculum": True,
                        "phase": phase_name,
                        "phase_index": idx,
                        "phase_epoch_percentage": percentage,
                        "phase_epochs": epochs,
                        **{key: value for key, value in phase.items() if key != "description"},
                    },
                ),
            )
            print(f"phase_records={phase_paths['records']}")
            old_epochs = args.epochs
            args.epochs = epochs
            result = train_records(
                phase_records,
                model,
                motif_vocab,
                args,
                validation_fraction,
                run_dir / "checkpoints" / phase_name,
                device,
            )
            args.epochs = old_epochs
            checkpoint = Path(result["best_checkpoint"])
            checkpoint_state = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_state["model_state_dict"], strict=False)
            final_checkpoint = checkpoint
            generator = phase_generator
            records = phase_records
            curriculum_summary.append(
                {
                    "phase": phase_name,
                    "percentage": percentage,
                    "epochs": epochs,
                    "records": phase_paths["records"],
                    "best_checkpoint": str(checkpoint),
                }
            )
        final_best = run_dir / "checkpoints" / "best.pt"
        final_best.parent.mkdir(parents=True, exist_ok=True)
        if final_checkpoint is not None:
            shutil.copyfile(final_checkpoint, final_best)
        with (run_dir / "curriculum_summary.json").open("w") as handle:
            json.dump(curriculum_summary, handle, indent=2)
            handle.write("\n")
        result = {"best_checkpoint": str(final_best), "curriculum": curriculum_summary}
    else:
        result = train_records(records, model, motif_vocab, args, validation_fraction, run_dir / "checkpoints", device)
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
