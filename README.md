# oSTRich RNN STR detector

<p align="center">
  <img src="ostrich_rnn/181bd6fe-5b3b-42bc-97e6-81e66178f051.png" alt="oSTRich logo" width="240">
</p>

This is a modular PyTorch starter implementation for detecting short tandem
repeat (STR) tracts in DNA sequences with a bidirectional recurrent neural
network.

The model frames STR detection as sequence labeling:

- each base is labeled as outside, repeat start, repeat inside, or repeat end
- a second per-base head predicts the repeat motif class for bases inside a
  tract
- post-processing converts base-level predictions into a repeat table

The synthetic generator is deliberately separated from the dataset/model code so
real labeled data can later replace it.

## Files

- `ostrich_rnn/motifs.py` - motif vocabulary, canonicalization, reverse-complement handling
- `ostrich_rnn/encoding.py` - DNA integer encoding and padded batching
- `ostrich_rnn/synthetic.py` - synthetic STR sequence generator
- `ostrich_rnn/dataset.py` - PyTorch dataset wrappers
- `ostrich_rnn/model.py` - BiLSTM/BiGRU sequence labeler
- `ostrich_rnn/postprocess.py` - repeat tract extraction and motif estimation
- `ostrich_rnn/evaluation.py` - base-level and tract-level metrics
- `ostrich_rnn/training.py` - training loop, validation, checkpoints, early stopping
- `ostrich_rnn/runtime.py` - CPU/GPU device selection helpers
- `ostrich_rnn/synthetic_io.py` - saving generated synthetic datasets and summaries
- `ostrich_rnn/inference.py` - FASTA-to-CSV inference
- `examples/train_synthetic.py` - brief synthetic training run
- `tests/test_sanity.py` - simple sanity checks

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run sanity tests:

```bash
pytest -q
```

Run a tiny synthetic training example:

```bash
python examples/train_synthetic.py --epochs 2 --train-size 128 --val-size 32
```

Run inference after training:

```bash
python -m ostrich_rnn.inference input.fa repeats.csv --checkpoint runs/<run_name>/checkpoints/best.pt
```

The output CSV has this shape:

```csv
sequence_id,start,end,length_bp,motif,motif_length,copy_number,confidence
seq_001,145,183,39,CAG,3,13.00,0.94
```

Coordinates are 0-based, inclusive `start`, exclusive `end`, matching Python
slicing.

## Usage

These commands assume you are running from WSL Ubuntu:

```bash
cd "/mnt/c/Users/Dale/OneDrive - Universiteit Antwerpen/Post Doc/Bioinfo/AI/oSTRich"
```

Install or refresh dependencies:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

Run the sanity checks:

```bash
python3 -m pytest -q
```

Check whether PyTorch can see your GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Train a synthetic model:

```bash
python3 examples/train_synthetic.py \
  --epochs 10 \
  --patience 3 \
  --train-size 1000 \
  --val-size 200 \
  --sequence-length 1024 \
  --max-motif-len 6 \
  --batch-size 16 \
  --device auto
```

`--device auto` uses CUDA when PyTorch can see a GPU and falls back to CPU
otherwise. Use `--device cuda` if you want the run to fail unless the GPU is
available.

`--patience` controls early stopping. For example, `--patience 8` lets training
continue until validation loss has failed to improve for 8 consecutive epochs.

Synthetic sequences are 1024 bp by default. Repeats are perfect by default:
substitution, insertion, deletion, and motif-interruption rates are all `0.0`
unless you explicitly enable them. To generate imperfect repeats:

```bash
python3 examples/train_synthetic.py \
  --epochs 10 \
  --train-size 1000 \
  --val-size 200 \
  --substitution-rate 0.01 \
  --insertion-rate 0.002 \
  --deletion-rate 0.002 \
  --motif-interruption-rate 0.05 \
  --device auto
```

`--motif-interruption-rate` replaces occasional motif copies with non-motif DNA
inside the tract. By default, interruption units are the same length as the
motif; use `--interruption-min-len` and `--interruption-max-len` to vary their
length.

Motifs are generated up to 6 bp by default. Synthetic motif sampling is
length-balanced: the generator first samples a motif length, for example
monomer, dimer, trimer, and so on, then samples a motif from that length class.
This keeps short motifs from being swamped by the much larger number of possible
longer motifs. The balance is approximate across repeat tracts, not necessarily
across total repeat bases.

Each synthetic training run creates a unique directory under `runs/`, for
example:

```text
runs/synthetic_20260504_123456_a1b2c3d4/
```

That run directory contains the generated data and the best checkpoint:

```text
runs/<run_name>/data/records.jsonl
runs/<run_name>/data/sequences.fa
runs/<run_name>/data/sequence_summary.csv
runs/<run_name>/data/repeat_summary.csv
runs/<run_name>/data/dataset_summary.json
runs/<run_name>/data/generation_config.json
runs/<run_name>/checkpoints/best.pt
```

`sequence_summary.csv` reports each generated sequence length, GC content,
number of `N` bases, repeat count, interruption count, total repeat bases, repeat
fraction, motifs, and compact repeat locations. `repeat_summary.csv` has one row
per repeat with start, end, length, motif or compound motifs, motif IDs, copy
number, whether the repeat is compound, and interruption coordinates/sequences.
`dataset_summary.json` gives dataset-level totals, motif counts, motif length
counts, and total interruption count.

Train from an existing saved synthetic dataset:

```bash
python3 examples/train_synthetic.py \
  --synthetic-data runs/<previous_run>/data/records.jsonl \
  --epochs 10 \
  --batch-size 16 \
  --validation-fraction 0.2 \
  --device auto
```

`--synthetic-data` can point to any of these:

```text
runs/<previous_run>/data/records.jsonl
runs/<previous_run>/data/
runs/<previous_run>/
```

When `--synthetic-data` is provided, no new synthetic dataset is generated. A
new run directory is still created for the checkpoint and a small
`source_dataset.json` file that records which dataset was used.

Run FASTA inference with a trained checkpoint:

```bash
python3 -m ostrich_rnn.inference input.fa repeats.csv \
  --checkpoint runs/<run_name>/checkpoints/best.pt \
  --window-size 2048 \
  --stride 1800 \
  --batch-size 8 \
  --device auto
```

The output CSV contains one row per predicted repeat tract:

```csv
sequence_id,start,end,length_bp,motif,motif_length,copy_number,confidence
seq_001,145,184,39,CAG,3,13.00,0.94
```

Programmatic training looks like this:

```python
from ostrich_rnn.dataset import STRRecordDataset
from ostrich_rnn.model import RNNSTRDetector
from ostrich_rnn.motifs import MotifVocab
from ostrich_rnn.synthetic import SyntheticSTRGenerator
from ostrich_rnn.synthetic_io import make_unique_run_dir, save_synthetic_dataset
from ostrich_rnn.training import TrainConfig, train_model

motif_vocab = MotifVocab.build(max_len=6)
generator = SyntheticSTRGenerator(
    motif_vocab=motif_vocab,
    min_repeats_per_sequence=1,
    max_repeats_per_sequence=3,
    max_copy_number=2000,
    substitution_rate=0.0,
    insertion_rate=0.0,
    deletion_rate=0.0,
    motif_interruption_rate=0.0,
    seed=7,
)
records = generator.generate_many(1200)
run_dir = make_unique_run_dir("runs")
save_synthetic_dataset(records, run_dir / "data", config={"seed": 7})

dataset = STRRecordDataset(records)
model = RNNSTRDetector(motif_classes=len(motif_vocab), hidden_dim=128)
config = TrainConfig(epochs=10, batch_size=16, checkpoint_dir=str(run_dir / "checkpoints"))

train_model(model, dataset, motif_vocab, config)
```

To train on a specific device from Python:

```python
train_model(model, dataset, motif_vocab, config, device="cuda")
```

`device="auto"` is the default and chooses CUDA when available.

To replace synthetic data with real labels, create records with the same shape
used by `STRRecordDataset`:

```python
from ostrich_rnn.motifs import MotifVocab

motif_vocab = MotifVocab.build(max_len=6)
cag_id = motif_vocab.encode("CAG")

record = {
    "sequence_id": "sample_001",
    "sequence": "ACGTCAGCAGCAGTT",
    "state_labels": [0] * 4 + [1] + [2] * 7 + [3] + [0] * 2,
    "motif_labels": [-100] * 4 + [cag_id] * 9 + [-100] * 2,
    "repeats": [
        {
            "start": 4,
            "end": 13,
            "length_bp": 9,
            "motifs": ["CAG"],
            "motif_ids": [cag_id],
            "copy_number": 3.0,
            "is_compound": False,
        }
    ],
}
```

Repeat-state labels are `0=outside`, `1=start`, `2=inside`, and `3=end`.
Motif labels use IDs from `MotifVocab`; outside-repeat bases should use `-100`
so the motif loss ignores them.

## Design notes

STR detection is naturally a sequence-labeling problem because repeat tracts are
contiguous spans within a larger DNA sequence. A label at each base gives the
model enough information to recover tract boundaries while still allowing
standard minibatch training.

A bidirectional LSTM or GRU is useful because the label for a base often depends
on flanking context. For example, deciding whether a base is at the boundary of a
repeat requires seeing both the preceding repeat pattern and the sequence that
follows.

Motif prediction is handled by a second per-base classification head. During
training, bases inside a repeat are assigned the canonical motif class for that
tract; outside bases use an ignore label. During post-processing, motif evidence
from the model is combined with a direct sequence-based motif estimate.
By default motifs are strand-specific, so `A` and `T` are separate classes. Use
`MotifVocab.build(collapse_reverse_complement=True)` if you want
reverse-complement-equivalent motifs collapsed.

Important limitations:

- Synthetic repeats are simpler than real genomic repeats.
- Very long repeat copy numbers should be handled with windows, otherwise
  recurrent training becomes memory-heavy.
- Imperfect and compound repeats can blur exact motif labels.
- The current post-processor is intentionally transparent rather than exhaustive.

Future improvements:

- Add a CRF layer over the repeat-state head to enforce legal label transitions.
- Add a small CNN front-end to expose local periodicity before the RNN.
- Use repeat-aware dynamic programming during post-processing.
- Train with real labeled loci and hard negatives from low-complexity sequence.
- Add multi-label motif targets for compound repeats instead of one dominant
  motif per base.
