"""Automated benchmark runs for comparing oSTRich training configurations."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import sys
import traceback
from typing import Any

import torch
from torch.utils.data import DataLoader

from .dataset import STRRecordDataset
from .encoding import collate_examples
from .evaluation import EvaluationResult, evaluate_model
from .model import RNNSTRDetector
from .motifs import MotifVocab
from .runtime import describe_device, resolve_device
from .synthetic import SyntheticSTRGenerator
from .synthetic_io import (
    add_motif_length_labels,
    load_records_jsonl,
    make_unique_run_dir,
    resolve_records_path,
    save_synthetic_dataset,
)
from .training import TrainConfig, train_model


def default_benchmark_config() -> dict:
    """Return a small editable benchmark configuration."""

    return {
        "benchmark_name": "synthetic_metric_sweep",
        "output_root": "benchmarks",
        "device": "auto",
        "continue_on_error": False,
        "dataset": {
            "train_size": 256,
            "val_size": 64,
            "sequence_length": 512,
            "min_repeats_per_sequence": 1,
            "max_repeats_per_sequence": 3,
            "min_motif_len": 1,
            "max_motif_len": 6,
            "min_copy_number": 3,
            "max_copy_number": 100,
            "compound_probability": 0.2,
            "max_compound_motifs": 3,
            "substitution_rate": 0.0,
            "insertion_rate": 0.0,
            "deletion_rate": 0.0,
            "motif_interruption_rate": 0.0,
            "interruption_min_len": None,
            "interruption_max_len": None,
            "seed": 7,
            "collapse_reverse_complement": False,
        },
        "model": {
            "hidden_dim": 64,
            "num_layers": 1,
            "embedding_dim": 16,
            "dropout": 0.2,
            "rnn_type": "lstm",
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "motif_loss_weight": 0.2,
            "motif_length_loss_weight": 0.1,
            "motif_class_weighting": True,
            "patience": 3,
            "num_workers": 0,
        },
        "holdout": {
            "size": 64,
            "sequence_length": 512,
            "max_copy_number": 100,
            "compound_probability": 0.3,
            "substitution_rate": 0.005,
            "insertion_rate": 0.001,
            "deletion_rate": 0.001,
            "motif_interruption_rate": 0.02,
            "seed": 1007,
        },
        "curriculum": {
            "phases": [
                {
                    "name": "phase1_perfect",
                    "dataset": {
                        "compound_probability": 0.0,
                        "substitution_rate": 0.0,
                        "insertion_rate": 0.0,
                        "deletion_rate": 0.0,
                        "motif_interruption_rate": 0.0,
                    },
                },
                {
                    "name": "phase2_low_imperfect",
                    "dataset": {
                        "compound_probability": 0.05,
                        "substitution_rate": 0.002,
                        "insertion_rate": 0.0005,
                        "deletion_rate": 0.0005,
                        "motif_interruption_rate": 0.01,
                    },
                },
                {
                    "name": "phase3_harder",
                    "dataset": {
                        "compound_probability": 0.3,
                        "substitution_rate": 0.01,
                        "insertion_rate": 0.002,
                        "deletion_rate": 0.002,
                        "motif_interruption_rate": 0.05,
                    },
                },
            ],
        },
        "experiments": [
            {
                "name": "baseline",
                "description": "Current balanced auxiliary-loss defaults.",
            },
            {
                "name": "higher_motif_loss",
                "description": "Increase motif identity loss to test motif accuracy gains.",
                "training": {
                    "motif_loss_weight": 0.5,
                },
            },
            {
                "name": "no_motif_class_weights",
                "description": "Disable motif class weighting to quantify imbalance effects.",
                "training": {
                    "motif_class_weighting": False,
                },
            },
        ],
    }


def load_benchmark_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def deep_update(base: dict, override: dict | None) -> dict:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "experiment"


def short_value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value).replace(".", "p")


def split_epochs(total_epochs: int, phase_count: int) -> list[int]:
    if phase_count <= 0:
        return []
    total_epochs = max(phase_count, int(total_epochs))
    base = total_epochs // phase_count
    remainder = total_epochs % phase_count
    return [base + (1 if idx < remainder else 0) for idx in range(phase_count)]


DATASET_FIELDS = {
    "train_size",
    "val_size",
    "sequence_length",
    "min_repeats_per_sequence",
    "max_repeats_per_sequence",
    "min_motif_len",
    "max_motif_len",
    "min_copy_number",
    "max_copy_number",
    "compound_probability",
    "max_compound_motifs",
    "substitution_rate",
    "insertion_rate",
    "deletion_rate",
    "motif_interruption_rate",
    "interruption_min_len",
    "interruption_max_len",
    "seed",
    "synthetic_data",
}


def expand_matrix_experiments(config: dict) -> list[dict]:
    matrix = config.get("matrix")
    if not matrix:
        return config.get("experiments") or [{"name": "baseline"}]

    training_fields = {field.name for field in fields(TrainConfig)}
    axes = [(key, values) for key, values in matrix.items()]
    experiments: list[dict] = []

    def build(index: int, selected: dict) -> None:
        if index == len(axes):
            model: dict[str, Any] = {}
            training: dict[str, Any] = {}
            dataset: dict[str, Any] = {}
            experiment: dict[str, Any] = {}
            name_parts = []
            for key, value in selected.items():
                if key in training_fields:
                    training[key] = value
                elif key in DATASET_FIELDS:
                    dataset[key] = value
                elif key == "curriculum":
                    experiment["curriculum"] = value
                else:
                    model[key] = value
                name_parts.append(f"{key}{short_value(value)}")
            experiment["name"] = safe_name("_".join(name_parts))
            if model:
                experiment["model"] = model
            if training:
                experiment["training"] = training
            if dataset:
                experiment["dataset"] = dataset
            experiments.append(experiment)
            return

        key, values = axes[index]
        if not isinstance(values, list):
            values = [values]
        for value in values:
            build(index + 1, {**selected, key: value})

    build(0, {})
    return experiments


def environment_snapshot(device: torch.device) -> dict:
    cuda_available = torch.cuda.is_available()
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "selected_device": str(device),
        "selected_device_description": describe_device(device),
    }


def build_motif_vocab(dataset_config: dict, source_config: dict | None = None) -> MotifVocab:
    if source_config and "motifs" in source_config:
        return MotifVocab(
            motifs=tuple(source_config["motifs"]),
            collapse_reverse_complement=source_config.get("collapse_reverse_complement", False),
        )
    return MotifVocab.build(
        max_len=int(dataset_config.get("max_motif_len", 6)),
        collapse_reverse_complement=bool(dataset_config.get("collapse_reverse_complement", False)),
    )


def build_synthetic_generator(dataset_config: dict, motif_vocab: MotifVocab) -> SyntheticSTRGenerator:
    return SyntheticSTRGenerator(
        motif_vocab=motif_vocab,
        sequence_length=int(dataset_config.get("sequence_length", 1024)),
        min_repeats_per_sequence=int(dataset_config.get("min_repeats_per_sequence", 1)),
        max_repeats_per_sequence=int(dataset_config.get("max_repeats_per_sequence", 3)),
        min_motif_len=int(dataset_config.get("min_motif_len", 1)),
        max_motif_len=int(dataset_config.get("max_motif_len", 6)),
        min_copy_number=int(dataset_config.get("min_copy_number", 3)),
        max_copy_number=int(dataset_config.get("max_copy_number", 2000)),
        substitution_rate=float(dataset_config.get("substitution_rate", 0.0)),
        insertion_rate=float(dataset_config.get("insertion_rate", 0.0)),
        deletion_rate=float(dataset_config.get("deletion_rate", 0.0)),
        motif_interruption_rate=float(dataset_config.get("motif_interruption_rate", 0.0)),
        interruption_min_len=dataset_config.get("interruption_min_len"),
        interruption_max_len=dataset_config.get("interruption_max_len"),
        compound_probability=float(dataset_config.get("compound_probability", 0.2)),
        max_compound_motifs=int(dataset_config.get("max_compound_motifs", 3)),
        seed=dataset_config.get("seed", 7),
    )


def generate_synthetic_records(
    dataset_config: dict,
    motif_vocab: MotifVocab,
    count: int,
    prefix: str,
) -> list[dict]:
    generator = build_synthetic_generator(dataset_config, motif_vocab)
    records = generator.generate_many(count, prefix=prefix)
    add_motif_length_labels(records, motif_vocab)
    return records


def prepare_records(config: dict, benchmark_dir: Path) -> tuple[list[dict], MotifVocab, dict]:
    dataset_config = deepcopy(config.get("dataset", {}))
    synthetic_data = dataset_config.get("synthetic_data")
    source_metadata: dict[str, Any] = {"kind": "generated_synthetic"}

    if synthetic_data:
        records_path = resolve_records_path(synthetic_data)
        source_config = None
        config_path = records_path.parent / "generation_config.json"
        if config_path.exists():
            with config_path.open() as handle:
                source_config = json.load(handle)
        motif_vocab = build_motif_vocab(dataset_config, source_config)
        records = load_records_jsonl(records_path)
        add_motif_length_labels(records, motif_vocab)
        source_metadata = {
            "kind": "saved_synthetic",
            "source_records": str(records_path),
            "source_generation_config": str(config_path) if config_path.exists() else None,
            "num_records": len(records),
        }
        write_json(benchmark_dir / "source_dataset.json", source_metadata)
        return records, motif_vocab, source_metadata

    motif_vocab = build_motif_vocab(dataset_config)
    train_size = int(dataset_config.get("train_size", 256))
    val_size = int(dataset_config.get("val_size", 64))
    records = generate_synthetic_records(dataset_config, motif_vocab, train_size + val_size, "benchmark")

    saved_config = {
        **dataset_config,
        "train_size": train_size,
        "val_size": val_size,
        "motif_count": len(motif_vocab),
        "motif_length_classes": max(len(motif) for motif in motif_vocab.motifs),
        "motifs": motif_vocab.motifs,
        "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
    }
    paths = save_synthetic_dataset(records, benchmark_dir / "data", config=saved_config)
    source_metadata.update({"num_records": len(records), "paths": paths})
    write_json(benchmark_dir / "source_dataset.json", source_metadata)
    return records, motif_vocab, source_metadata


def prepare_holdout_records(config: dict, benchmark_dir: Path, motif_vocab: MotifVocab) -> tuple[list[dict], dict]:
    dataset_config = deepcopy(config.get("dataset", {}))
    holdout_config = deep_update(dataset_config, config.get("holdout", {}))
    synthetic_data = holdout_config.get("synthetic_data")

    if synthetic_data:
        records_path = resolve_records_path(synthetic_data)
        records = load_records_jsonl(records_path)
        add_motif_length_labels(records, motif_vocab)
        metadata = {
            "kind": "saved_synthetic_holdout",
            "source_records": str(records_path),
            "num_records": len(records),
        }
        write_json(benchmark_dir / "holdout_dataset.json", metadata)
        return records, metadata

    size = int(holdout_config.get("size", dataset_config.get("val_size", 64)))
    holdout_config["seed"] = holdout_config.get("seed", int(dataset_config.get("seed", 7)) + 1000)
    records = generate_synthetic_records(holdout_config, motif_vocab, size, "holdout")
    saved_config = {
        **holdout_config,
        "size": size,
        "motif_count": len(motif_vocab),
        "motif_length_classes": max(len(motif) for motif in motif_vocab.motifs),
        "motifs": motif_vocab.motifs,
        "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
    }
    paths = save_synthetic_dataset(records, benchmark_dir / "holdout", config=saved_config)
    metadata = {"kind": "generated_synthetic_holdout", "num_records": len(records), "paths": paths}
    write_json(benchmark_dir / "holdout_dataset.json", metadata)
    return records, metadata


def prepare_experiment_records(
    config: dict,
    experiment: dict,
    experiment_dir: Path,
    motif_vocab: MotifVocab,
    base_records: list[dict],
    base_metadata: dict,
    seed: int,
    generate_records: bool = True,
) -> tuple[list[dict], dict, dict]:
    """Return training records for one experiment, generating data if needed."""

    if not experiment.get("dataset"):
        return base_records, base_metadata, deepcopy(config.get("dataset", {}))

    dataset_config = deep_update(config.get("dataset", {}), experiment.get("dataset"))
    dataset_config["seed"] = int(dataset_config.get("seed", seed))
    if not generate_records:
        metadata = {
            "kind": "experiment_generated_by_curriculum_phases",
            "num_records": None,
            "dataset_config": dataset_config,
        }
        write_json(experiment_dir / "source_dataset.json", metadata)
        return base_records, metadata, dataset_config

    synthetic_data = dataset_config.get("synthetic_data")
    if synthetic_data:
        records_path = resolve_records_path(synthetic_data)
        records = load_records_jsonl(records_path)
        add_motif_length_labels(records, motif_vocab)
        metadata = {
            "kind": "experiment_saved_synthetic",
            "source_records": str(records_path),
            "num_records": len(records),
        }
        write_json(experiment_dir / "source_dataset.json", metadata)
        return records, metadata, dataset_config

    train_size = int(dataset_config.get("train_size", 256))
    val_size = int(dataset_config.get("val_size", 64))
    records = generate_synthetic_records(
        dataset_config,
        motif_vocab,
        train_size + val_size,
        "experiment",
    )
    saved_config = {
        **dataset_config,
        "train_size": train_size,
        "val_size": val_size,
        "motif_count": len(motif_vocab),
        "motif_length_classes": max(len(motif) for motif in motif_vocab.motifs),
        "motifs": motif_vocab.motifs,
        "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
    }
    paths = save_synthetic_dataset(records, experiment_dir / "data", config=saved_config)
    metadata = {"kind": "experiment_generated_synthetic", "num_records": len(records), "paths": paths}
    write_json(experiment_dir / "source_dataset.json", metadata)
    return records, metadata, dataset_config


def make_model(model_config: dict, motif_vocab: MotifVocab) -> RNNSTRDetector:
    config = deepcopy(model_config)
    config.pop("motif_classes", None)
    config.pop("motif_length_classes", None)
    return RNNSTRDetector(
        motif_classes=len(motif_vocab),
        motif_length_classes=max(len(motif) for motif in motif_vocab.motifs),
        **config,
    )


def make_train_config(training_config: dict, checkpoint_dir: Path) -> TrainConfig:
    config = deepcopy(training_config)
    config["checkpoint_dir"] = str(checkpoint_dir)
    allowed = {field.name for field in fields(TrainConfig)}
    return TrainConfig(**{key: value for key, value in config.items() if key in allowed})


def write_history(path_json: Path, path_csv: Path, history: list[dict]) -> None:
    write_json(path_json, history)
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "phase",
        "phase_epoch",
        "epoch",
        "train_loss",
        "val_loss",
        "base_f1",
        "tract_f1",
        "motif_accuracy",
        "motif_length_accuracy",
    ]
    extras = sorted({key for row in history for key in row} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in history)] + extras
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in history)


def summarize_history(history: list[dict]) -> dict:
    if not history:
        return {}
    best = min(history, key=lambda row: row.get("val_loss", float("inf")))
    latest = history[-1]
    return {
        "epochs_run": len(history),
        "best_epoch": best.get("epoch"),
        "best_val_loss": best.get("val_loss"),
        "best_train_loss": best.get("train_loss"),
        "best_base_f1": best.get("base_f1"),
        "best_tract_f1": best.get("tract_f1"),
        "best_motif_accuracy": best.get("motif_accuracy"),
        "best_motif_length_accuracy": best.get("motif_length_accuracy"),
        "final_epoch": latest.get("epoch"),
        "final_val_loss": latest.get("val_loss"),
        "final_base_f1": latest.get("base_f1"),
        "final_tract_f1": latest.get("tract_f1"),
        "final_motif_accuracy": latest.get("motif_accuracy"),
        "final_motif_length_accuracy": latest.get("motif_length_accuracy"),
    }


def flatten_evaluation(prefix: str, metrics: EvaluationResult) -> dict:
    return {
        f"{prefix}_base_precision": metrics.base["precision"],
        f"{prefix}_base_recall": metrics.base["recall"],
        f"{prefix}_base_f1": metrics.base["f1"],
        f"{prefix}_motif_accuracy": metrics.motif_accuracy,
        f"{prefix}_motif_length_accuracy": metrics.motif_length_accuracy,
        f"{prefix}_tract_precision": metrics.tract["precision"],
        f"{prefix}_tract_recall": metrics.tract["recall"],
        f"{prefix}_tract_f1": metrics.tract["f1"],
        f"{prefix}_tract_motif_accuracy": metrics.tract["tract_motif_accuracy"],
        f"{prefix}_start_mae": metrics.tract["start_mae"],
        f"{prefix}_end_mae": metrics.tract["end_mae"],
        f"{prefix}_length_mae": metrics.tract["length_mae"],
        f"{prefix}_true_positives": metrics.tract["true_positives"],
        f"{prefix}_false_positives": metrics.tract["false_positives"],
        f"{prefix}_false_negatives": metrics.tract["false_negatives"],
    }


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_config: dict,
    motif_vocab: MotifVocab,
    device: torch.device,
) -> RNNSTRDetector:
    model = make_model(model_config, motif_vocab)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def evaluate_records(
    model: RNNSTRDetector,
    records: list[dict],
    motif_vocab: MotifVocab,
    batch_size: int,
    device: torch.device,
) -> dict:
    loader = DataLoader(
        STRRecordDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_examples,
    )
    metrics = evaluate_model(model, loader, motif_vocab, device)
    return flatten_evaluation("holdout", metrics)


def curriculum_config_for_experiment(config: dict, experiment: dict) -> dict | None:
    selected = experiment.get("curriculum", False)
    if not selected:
        return None
    base = deepcopy(config.get("curriculum", {}))
    if isinstance(selected, dict):
        return deep_update(base, selected)
    return base


def curriculum_phase_epoch_counts(curriculum_config: dict, total_epochs: int, phase_count: int) -> list[int]:
    phase_epochs = curriculum_config.get("phase_epochs")
    if phase_epochs is not None:
        if len(phase_epochs) != phase_count:
            raise ValueError("curriculum.phase_epochs must match the number of curriculum phases.")
        return [int(value) for value in phase_epochs]
    return split_epochs(total_epochs, phase_count)


def run_standard_training(
    records: list[dict],
    model,
    motif_vocab: MotifVocab,
    training_config: dict,
    experiment_dir: Path,
    device: torch.device,
) -> dict:
    train_config = make_train_config(training_config, experiment_dir / "checkpoints")
    result = train_model(
        model,
        STRRecordDataset(records),
        motif_vocab,
        train_config,
        device=device,
    )
    history = result["history"]
    write_history(experiment_dir / "history.json", experiment_dir / "history.csv", history)
    return {
        "history": history,
        "best_checkpoint": result.get("best_checkpoint"),
        "phase_summaries": [],
    }


def run_curriculum_training(
    config: dict,
    experiment: dict,
    model,
    motif_vocab: MotifVocab,
    training_config: dict,
    dataset_config: dict,
    experiment_dir: Path,
    seed: int,
    device: torch.device,
) -> dict:
    curriculum_config = curriculum_config_for_experiment(config, experiment)
    if curriculum_config is None:
        raise ValueError("Curriculum training requested without a curriculum configuration.")
    phases = curriculum_config.get("phases") or []
    if not phases:
        raise ValueError("Curriculum configuration must contain at least one phase.")

    phase_epochs = curriculum_phase_epoch_counts(curriculum_config, int(training_config.get("epochs", 1)), len(phases))
    combined_history: list[dict] = []
    phase_summaries: list[dict] = []
    global_epoch = 0
    best_checkpoint = None

    for phase_index, (phase, epochs) in enumerate(zip(phases, phase_epochs), start=1):
        phase_name = safe_name(phase.get("name", f"phase_{phase_index}"))
        phase_dir = experiment_dir / "curriculum" / f"{phase_index:02d}_{phase_name}"
        phase_dataset_config = deep_update(dataset_config, curriculum_config.get("dataset"))
        phase_dataset_config = deep_update(phase_dataset_config, phase.get("dataset"))
        phase_dataset_config["seed"] = int(phase_dataset_config.get("seed", seed)) + phase_index * 1000
        train_size = int(phase_dataset_config.get("train_size", dataset_config.get("train_size", 256)))
        val_size = int(phase_dataset_config.get("val_size", dataset_config.get("val_size", 64)))
        phase_records = generate_synthetic_records(
            phase_dataset_config,
            motif_vocab,
            train_size + val_size,
            phase_name,
        )
        phase_paths = save_synthetic_dataset(
            phase_records,
            phase_dir / "data",
            config={
                **phase_dataset_config,
                "phase": phase_name,
                "phase_epochs": epochs,
                "motif_count": len(motif_vocab),
                "motif_length_classes": max(len(motif) for motif in motif_vocab.motifs),
                "motifs": motif_vocab.motifs,
                "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
            },
        )

        phase_training_config = deep_update(training_config, phase.get("training"))
        phase_training_config["epochs"] = epochs
        phase_training_config["validation_fraction"] = val_size / max(1, train_size + val_size)
        phase_result = train_model(
            model,
            STRRecordDataset(phase_records),
            motif_vocab,
            make_train_config(phase_training_config, phase_dir / "checkpoints"),
            device=device,
        )
        phase_history = phase_result["history"]
        write_history(phase_dir / "history.json", phase_dir / "history.csv", phase_history)
        for row in phase_history:
            global_epoch += 1
            combined_history.append(
                {
                    **row,
                    "phase": phase_name,
                    "phase_epoch": row.get("epoch"),
                    "epoch": global_epoch,
                }
            )
        phase_summary = {
            "phase": phase_name,
            "epochs": epochs,
            "records": len(phase_records),
            "paths": phase_paths,
            "best_checkpoint": phase_result.get("best_checkpoint"),
            **summarize_history(phase_history),
        }
        phase_summaries.append(phase_summary)
        best_checkpoint = phase_result.get("best_checkpoint")

    write_history(experiment_dir / "history.json", experiment_dir / "history.csv", combined_history)
    write_json(experiment_dir / "curriculum_summary.json", phase_summaries)
    return {
        "history": combined_history,
        "best_checkpoint": best_checkpoint,
        "phase_summaries": phase_summaries,
    }


def write_comparison(path_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    metric_fields = [
        "experiment",
        "status",
        "epochs_run",
        "best_epoch",
        "best_val_loss",
        "best_base_f1",
        "best_tract_f1",
        "best_motif_accuracy",
        "best_motif_length_accuracy",
        "final_val_loss",
        "final_base_f1",
        "final_tract_f1",
        "final_motif_accuracy",
        "final_motif_length_accuracy",
        "holdout_base_f1",
        "holdout_tract_f1",
        "holdout_motif_accuracy",
        "holdout_motif_length_accuracy",
        "holdout_tract_motif_accuracy",
        "holdout_start_mae",
        "holdout_end_mae",
        "holdout_length_mae",
        "hidden_dim",
        "num_layers",
        "curriculum",
        "train_size",
        "val_size",
        "sequence_length",
        "learning_rate",
        "batch_size",
        "epochs",
        "motif_loss_weight",
        "motif_length_loss_weight",
        "motif_class_weighting",
        "best_checkpoint",
        "error",
    ]
    path_csv.parent.mkdir(parents=True, exist_ok=True)
    with path_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    config: dict | None = None,
    output_root: str | Path | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Run a complete benchmark and return paths plus comparison rows."""

    config = deep_update(default_benchmark_config(), config or {})
    resolved_device = resolve_device(device or config.get("device", "auto"))
    benchmark_dir = make_unique_run_dir(
        output_root or config.get("output_root", "benchmarks"),
        prefix="benchmark",
        name=config.get("benchmark_name"),
    )
    write_json(benchmark_dir / "benchmark_config.json", config)
    write_json(benchmark_dir / "environment.json", environment_snapshot(resolved_device))

    records, motif_vocab, source_metadata = prepare_records(config, benchmark_dir)
    holdout_records, holdout_metadata = prepare_holdout_records(config, benchmark_dir, motif_vocab)
    base_training = deepcopy(config.get("training", {}))
    dataset_config = config.get("dataset", {})
    if "validation_fraction" not in base_training:
        if dataset_config.get("synthetic_data"):
            base_training["validation_fraction"] = 0.2
        else:
            train_size = int(dataset_config.get("train_size", 256))
            val_size = int(dataset_config.get("val_size", 64))
            base_training["validation_fraction"] = val_size / max(1, train_size + val_size)

    comparison_rows: list[dict] = []
    experiments = expand_matrix_experiments(config)
    continue_on_error = bool(config.get("continue_on_error", False))

    for index, experiment in enumerate(experiments, start=1):
        name = safe_name(experiment.get("name", f"experiment_{index:02d}"))
        experiment_dir = benchmark_dir / "experiments" / f"{index:02d}_{name}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        seed = int(experiment.get("seed", dataset_config.get("seed", 7))) + index
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model_config = deep_update(config.get("model", {}), experiment.get("model"))
        training_config = deep_update(base_training, experiment.get("training"))
        use_curriculum = bool(curriculum_config_for_experiment(config, experiment))
        experiment_records, experiment_dataset_metadata, experiment_dataset_config = prepare_experiment_records(
            config,
            experiment,
            experiment_dir,
            motif_vocab,
            records,
            source_metadata,
            seed,
            generate_records=not use_curriculum,
        )
        if (
            "validation_fraction" not in config.get("training", {})
            and "validation_fraction" not in experiment.get("training", {})
            and not experiment_dataset_config.get("synthetic_data")
        ):
            train_size = int(experiment_dataset_config.get("train_size", dataset_config.get("train_size", 256)))
            val_size = int(experiment_dataset_config.get("val_size", dataset_config.get("val_size", 64)))
            training_config["validation_fraction"] = val_size / max(1, train_size + val_size)

        experiment_config = {
            "name": name,
            "description": experiment.get("description"),
            "seed": seed,
            "curriculum": use_curriculum,
            "model": model_config,
            "training": training_config,
            "dataset": experiment_dataset_metadata,
            "holdout": holdout_metadata,
        }
        write_json(experiment_dir / "experiment_config.json", experiment_config)

        row = {
            "experiment": name,
            "status": "completed",
            "error": None,
            "curriculum": use_curriculum,
            **model_config,
            "train_size": experiment_dataset_config.get("train_size"),
            "val_size": experiment_dataset_config.get("val_size"),
            "sequence_length": experiment_dataset_config.get("sequence_length"),
            **training_config,
        }
        try:
            model = make_model(model_config, motif_vocab)
            if use_curriculum:
                result = run_curriculum_training(
                    config,
                    experiment,
                    model,
                    motif_vocab,
                    training_config,
                    experiment_dataset_config,
                    experiment_dir,
                    seed,
                    resolved_device,
                )
            else:
                result = run_standard_training(
                    experiment_records,
                    model,
                    motif_vocab,
                    training_config,
                    experiment_dir,
                    resolved_device,
                )
            history = result["history"]
            metrics = {
                **summarize_history(history),
                "best_checkpoint": result.get("best_checkpoint"),
            }
            if result.get("best_checkpoint") and holdout_records:
                eval_model = load_model_from_checkpoint(
                    result["best_checkpoint"],
                    model_config,
                    motif_vocab,
                    resolved_device,
                )
                metrics.update(
                    evaluate_records(
                        eval_model,
                        holdout_records,
                        motif_vocab,
                        int(training_config.get("batch_size", 16)),
                        resolved_device,
                    )
                )
            write_json(experiment_dir / "metrics.json", metrics)
            row.update(metrics)
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            write_json(
                experiment_dir / "error.json",
                {
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if not continue_on_error:
                comparison_rows.append(row)
                write_json(benchmark_dir / "comparison.json", comparison_rows)
                write_comparison(benchmark_dir / "comparison.csv", comparison_rows)
                raise
        comparison_rows.append(row)
        write_json(benchmark_dir / "comparison.json", comparison_rows)
        write_comparison(benchmark_dir / "comparison.csv", comparison_rows)

    return {
        "benchmark_dir": str(benchmark_dir),
        "comparison_csv": str(benchmark_dir / "comparison.csv"),
        "comparison_json": str(benchmark_dir / "comparison.json"),
        "rows": comparison_rows,
    }
