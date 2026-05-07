from pathlib import Path

import pytest
import torch

from scripts.train_synthetic import parse_curriculum_phase_epochs
from ostrich_rnn.benchmark import run_benchmark
from ostrich_rnn.benchmark_report import create_benchmark_report
from ostrich_rnn.dataset import STRRecordDataset
from ostrich_rnn.encoding import collate_examples
from ostrich_rnn.model import RNNSTRDetector
from ostrich_rnn.motifs import MotifVocab, canonical_motif
from ostrich_rnn.postprocess import calls_from_logits, estimate_motif_from_sequence
from ostrich_rnn.synthetic import SyntheticSTRGenerator
from ostrich_rnn.synthetic_io import (
    add_motif_length_labels,
    load_records_jsonl,
    resolve_records_path,
    save_synthetic_dataset,
)
from ostrich_rnn.training import class_weights_from_dataset


def test_canonical_motif_rotations_and_reverse_complement():
    assert canonical_motif("CAG") == canonical_motif("AGC")
    assert canonical_motif("CAG") == canonical_motif("CTG")


def test_curriculum_phase_percentages_round_to_total_epochs():
    assert parse_curriculum_phase_epochs("20,35,45", 60) == [12, 21, 27]
    assert parse_curriculum_phase_epochs("20,35,45", 10) == [2, 4, 4]
    assert sum(parse_curriculum_phase_epochs("20,35,45", 11)) == 11


def test_generator_shapes():
    vocab = MotifVocab.build(max_len=4)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        sequence_length=128,
        min_repeats_per_sequence=1,
        max_repeats_per_sequence=1,
        seed=1,
    )
    example = generator.generate("seq_test")
    assert len(example["sequence"]) == 128
    assert len(example["state_labels"]) == 128
    assert len(example["motif_labels"]) == 128
    assert len(example["motif_length_labels"]) == 128
    assert len(example["repeats"]) == 1


def test_generator_defaults_to_1024_and_perfect_repeats():
    vocab = MotifVocab.build(max_len=3)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        min_repeats_per_sequence=1,
        max_repeats_per_sequence=1,
        min_copy_number=3,
        max_copy_number=3,
        compound_probability=0.0,
        seed=4,
    )
    example = generator.generate("seq_default")
    repeat = example["repeats"][0]
    motif = repeat["motifs"][0]
    tract = example["sequence"][repeat["start"] : repeat["end"]]
    assert len(example["sequence"]) == 1024
    assert repeat["interruption_count"] == 0
    assert tract == motif * 3


def test_generator_can_add_motif_interruptions():
    vocab = MotifVocab.build(max_len=3)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        sequence_length=128,
        min_repeats_per_sequence=1,
        max_repeats_per_sequence=1,
        min_copy_number=4,
        max_copy_number=4,
        compound_probability=0.0,
        motif_interruption_rate=1.0,
        seed=5,
    )
    example = generator.generate("seq_interrupted")
    repeat = example["repeats"][0]
    assert repeat["interruption_count"] == 4
    assert len(repeat["interruptions"]) == 4


def test_motif_sampling_balances_lengths():
    vocab = MotifVocab.build(max_len=6)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        min_motif_len=1,
        max_motif_len=6,
        seed=6,
    )
    counts = {length: 0 for length in range(1, 7)}
    for _ in range(600):
        counts[len(generator.sample_motif())] += 1
    assert set(counts) == set(range(1, 7))
    assert min(counts.values()) > 70
    assert max(counts.values()) < 130


def test_model_forward_and_postprocess_smoke():
    vocab = MotifVocab.build(max_len=3)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        sequence_length=96,
        min_repeats_per_sequence=1,
        max_repeats_per_sequence=1,
        seed=2,
    )
    example = generator.generate("seq_test")
    batch = collate_examples([example])
    model = RNNSTRDetector(motif_classes=len(vocab), hidden_dim=16, num_layers=1)
    outputs = model(batch.input_ids, batch.lengths)
    assert outputs.state_logits.shape[:2] == batch.input_ids.shape
    assert outputs.motif_logits.shape[:2] == batch.input_ids.shape
    assert outputs.motif_length_logits.shape[:2] == batch.input_ids.shape
    calls = calls_from_logits(
        sequence_id="seq_test",
        sequence=example["sequence"],
        state_logits=outputs.state_logits[0],
        motif_logits=outputs.motif_logits[0],
        motif_vocab=vocab,
    )
    assert isinstance(calls, list)


def test_sequence_periodicity_motif_calling_handles_boundary_noise():
    vocab = MotifVocab.build(max_len=6)
    motif, score = estimate_motif_from_sequence("TCAGCAGCAGCAA", vocab)
    assert motif == canonical_motif("CAG", collapse_reverse_complement=False)
    assert score > 0.75


def test_calls_prefer_sequence_motif_over_wrong_motif_logits():
    vocab = MotifVocab.build(max_len=6)
    sequence = "CAGCAGCAGCAG"
    state_logits = torch.zeros(len(sequence), 4)
    state_logits[:, 2] = 5.0
    motif_logits = torch.zeros(len(sequence), len(vocab))
    motif_logits[:, vocab.encode("A")] = 8.0
    calls = calls_from_logits(
        sequence_id="seq_test",
        sequence=sequence,
        state_logits=state_logits,
        motif_logits=motif_logits,
        motif_vocab=vocab,
    )
    assert calls[0].motif == canonical_motif("CAG", collapse_reverse_complement=False)


def test_save_synthetic_dataset_summaries(tmp_path):
    vocab = MotifVocab.build(max_len=3)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        sequence_length=96,
        min_repeats_per_sequence=1,
        max_repeats_per_sequence=1,
        seed=3,
    )
    records = generator.generate_many(3)
    paths = save_synthetic_dataset(records, tmp_path / "data", config={"seed": 3})
    for path in paths.values():
        assert Path(path).exists()
    loaded = load_records_jsonl(paths["records"])
    assert len(loaded) == 3
    assert resolve_records_path(paths["records"]) == Path(paths["records"])
    assert resolve_records_path(tmp_path / "data") == Path(paths["records"])
    assert "sequence_summary.csv" in paths["sequence_summary"]
    assert "repeat_summary.csv" in paths["repeat_summary"]
    assert "dataset_summary.json" in paths["dataset_summary"]


def test_backfill_motif_length_labels_and_weights():
    vocab = MotifVocab.build(max_len=3)
    generator = SyntheticSTRGenerator(
        motif_vocab=vocab,
        sequence_length=96,
        min_repeats_per_sequence=1,
        max_repeats_per_sequence=1,
        seed=8,
    )
    records = generator.generate_many(4)
    for record in records:
        record.pop("motif_length_labels")
    add_motif_length_labels(records, vocab)
    assert all("motif_length_labels" in record for record in records)
    dataset = STRRecordDataset(records)
    weights = class_weights_from_dataset(dataset, "motif_labels", len(vocab), -100, torch.device("cpu"))
    assert weights is not None
    assert weights.shape[0] == len(vocab)


def test_benchmark_pipeline_writes_artifacts(tmp_path):
    config = {
        "benchmark_name": "pytest_benchmark",
        "output_root": str(tmp_path),
        "device": "cpu",
        "dataset": {
            "train_size": 4,
            "val_size": 2,
            "sequence_length": 64,
            "min_repeats_per_sequence": 1,
            "max_repeats_per_sequence": 1,
            "max_motif_len": 2,
            "max_copy_number": 8,
            "compound_probability": 0.0,
            "seed": 11,
        },
        "model": {
            "embedding_dim": 8,
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "patience": 1,
            "motif_loss_weight": 0.2,
            "motif_length_loss_weight": 0.1,
        },
        "experiments": [
            {"name": "baseline"},
            {"name": "no_weights", "training": {"motif_class_weighting": False}},
        ],
    }
    result = run_benchmark(config, device="cpu")
    benchmark_dir = Path(result["benchmark_dir"])
    assert Path(result["comparison_csv"]).exists()
    assert Path(result["comparison_json"]).exists()
    assert (benchmark_dir / "benchmark_config.json").exists()
    assert (benchmark_dir / "environment.json").exists()
    assert len(result["rows"]) == 2
    assert (benchmark_dir / "experiments" / "01_baseline" / "history.csv").exists()
    assert (benchmark_dir / "experiments" / "01_baseline" / "metrics.json").exists()


def test_benchmark_report_writes_summary_and_plots(tmp_path):
    pytest.importorskip("matplotlib")
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    (benchmark_dir / "comparison.csv").write_text(
        "experiment,status,holdout_tract_f1,holdout_base_f1,holdout_motif_accuracy,"
        "holdout_motif_length_accuracy,holdout_tract_motif_accuracy,holdout_false_positives,"
        "holdout_false_negatives,holdout_length_mae,batch_size,hidden_dim,num_layers,epochs,curriculum,"
        "best_val_loss,best_epoch,epochs_run,best_checkpoint\n"
        "small,completed,0.40,0.50,0.30,0.60,0.25,4,6,3.1,8,64,1,3,False,1.2,2,3,small.pt\n"
        "large_curriculum,completed,0.62,0.70,0.45,0.75,0.40,2,3,1.5,16,128,2,6,True,0.9,5,6,large.pt\n"
    )
    report = create_benchmark_report(benchmark_dir)
    assert Path(report.summary_path).exists()
    assert Path(report.html_path).exists()
    assert Path(report.ranked_csv).exists()
    assert report.best_experiment == "large_curriculum"
    assert any(Path(path).exists() for path in report.plot_paths)
    assert "All Ranked Results" in Path(report.html_path).read_text()
