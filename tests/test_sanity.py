from pathlib import Path

from ostrich_rnn.encoding import collate_examples
from ostrich_rnn.model import RNNSTRDetector
from ostrich_rnn.motifs import MotifVocab, canonical_motif
from ostrich_rnn.postprocess import calls_from_logits
from ostrich_rnn.synthetic import SyntheticSTRGenerator
from ostrich_rnn.synthetic_io import load_records_jsonl, resolve_records_path, save_synthetic_dataset


def test_canonical_motif_rotations_and_reverse_complement():
    assert canonical_motif("CAG") == canonical_motif("AGC")
    assert canonical_motif("CAG") == canonical_motif("CTG")


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
    calls = calls_from_logits(
        sequence_id="seq_test",
        sequence=example["sequence"],
        state_logits=outputs.state_logits[0],
        motif_logits=outputs.motif_logits[0],
        motif_vocab=vocab,
    )
    assert isinstance(calls, list)


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
