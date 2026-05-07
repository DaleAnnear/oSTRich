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
- a third per-base head predicts motif length, which helps constrain motif
  learning
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
- `ostrich_rnn/benchmark.py` - automated benchmark sweeps and comparison tables
- `ostrich_rnn/benchmark_report.py` - benchmark interpretation reports and plots
- `ostrich_rnn/runtime.py` - CPU/GPU device selection helpers
- `ostrich_rnn/synthetic_io.py` - saving generated synthetic datasets and summaries
- `ostrich_rnn/inference.py` - FASTA-to-CSV inference
- `scripts/train_synthetic.py` - brief synthetic training run
- `scripts/benchmark_synthetic.py` - JSON-driven synthetic benchmark runner
- `scripts/benchmark_interpret.py` - benchmark plotting and interpretation runner
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
python scripts/train_synthetic.py --epochs 2 --train-size 128 --val-size 32
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
python3 scripts/train_synthetic.py \
  --epochs 10 \
  --patience 3 \
  --train-size 1000 \
  --val-size 200 \
  --sequence-length 1024 \
  --max-motif-len 6 \
  --batch-size 16 \
  --hidden-dim 128 \
  --num-layers 1 \
  --motif-loss-weight 0.2 \
  --motif-length-loss-weight 0.1 \
  --device auto
```

`--device auto` uses CUDA when PyTorch can see a GPU and falls back to CPU
otherwise. Use `--device cuda` if you want the run to fail unless the GPU is
available.

`--patience` controls early stopping. For example, `--patience 8` lets training
continue until validation loss has failed to improve for 8 consecutive epochs.

Motif class weighting is enabled by default. This upweights rare motif classes
in the motif cross-entropy loss so common motifs do not dominate training. Use
`--no-motif-class-weighting` to disable it. The motif identity and motif-length
auxiliary losses are controlled with `--motif-loss-weight` and
`--motif-length-loss-weight`. The defaults are intentionally modest, `0.2` and
`0.1`, so boundary detection remains the main training objective.

The default RNN capacity for the synthetic training script is
`--hidden-dim 128 --num-layers 1`. Increase `--num-layers` to `2` only if
validation metrics suggest the one-layer model is underfitting.

Synthetic sequences are 1024 bp by default. Repeats are perfect by default:
substitution, insertion, deletion, and motif-interruption rates are all `0.0`
unless you explicitly enable them. To generate imperfect repeats:

```bash
python3 scripts/train_synthetic.py \
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
python3 scripts/train_synthetic.py \
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

Train with a synthetic curriculum:

```bash
python3 scripts/train_synthetic.py \
  --curriculum \
  --epochs 60 \
  --curriculum-phase-epochs 20,35,45 \
  --patience 6 \
  --train-size 2000 \
  --val-size 400 \
  --batch-size 16 \
  --device cuda \
  --max-copy-number 80
```

Curriculum mode trains the same model through three generated datasets:

- `phase1_perfect`: perfect simple repeats, no mutations, no interruptions, no compounds
- `phase2_low_imperfect`: low mutation/interruption rates and rare compounds
- `phase3_harder_imperfect_compound`: higher mutation/interruption rates and more compounds

`--curriculum-phase-epochs` is interpreted as percentages of `--epochs`, not
literal epoch counts. The default is `20,35,45`, so `--epochs 60` trains the
three phases for `12`, `21`, and `27` epochs. If percentages do not produce whole
epochs, they are rounded and adjusted so the phase epochs still add up exactly
to `--epochs`.

Each phase writes its own data directory:

```text
runs/<run_name>/data/phase1_perfect/
runs/<run_name>/data/phase2_low_imperfect/
runs/<run_name>/data/phase3_harder_imperfect_compound/
```

Each phase also writes its own checkpoint directory. After phase 3, the final
phase best checkpoint is copied to:

```text
runs/<run_name>/checkpoints/best.pt
```

The run directory also contains `curriculum_summary.json`, which records the
phase datasets and checkpoints used.

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

Run an automated benchmark sweep:

```bash
python3 scripts/benchmark_synthetic.py --write-template benchmark_config.json
python3 scripts/benchmark_synthetic.py --config benchmark_config.json --device auto
```

A ready-to-run pilot grid is included:

```bash
python3 scripts/benchmark_synthetic.py \
  --config benchmark_configs/ostrich_grid_pilot.json \
  --device auto
```

The benchmark runner creates one shared dataset, trains each experiment against
that same dataset/split, and writes a comparison table. Each benchmark directory
contains:

```text
benchmarks/<benchmark_name>/
benchmarks/<benchmark_name>/benchmark_config.json
benchmarks/<benchmark_name>/environment.json
benchmarks/<benchmark_name>/source_dataset.json
benchmarks/<benchmark_name>/comparison.csv
benchmarks/<benchmark_name>/comparison.json
benchmarks/<benchmark_name>/data/records.jsonl
benchmarks/<benchmark_name>/data/dataset_summary.json
benchmarks/<benchmark_name>/experiments/01_baseline/experiment_config.json
benchmarks/<benchmark_name>/experiments/01_baseline/history.csv
benchmarks/<benchmark_name>/experiments/01_baseline/history.json
benchmarks/<benchmark_name>/experiments/01_baseline/metrics.json
benchmarks/<benchmark_name>/experiments/01_baseline/checkpoints/best.pt
```

Use benchmark experiments to compare training settings, model sizes, auxiliary
loss weights, class weighting, imperfect-repeat data, or any other variable that
should be tested against a fixed dataset:

```json
{
  "benchmark_name": "motif_loss_sweep",
  "output_root": "benchmarks",
  "device": "auto",
  "dataset": {
    "train_size": 1000,
    "val_size": 200,
    "sequence_length": 1024,
    "max_motif_len": 6,
    "max_copy_number": 100,
    "substitution_rate": 0.0,
    "insertion_rate": 0.0,
    "deletion_rate": 0.0,
    "motif_interruption_rate": 0.0,
    "seed": 7
  },
  "model": {
    "hidden_dim": 128,
    "num_layers": 1
  },
  "training": {
    "epochs": 10,
    "batch_size": 16,
    "patience": 3,
    "motif_loss_weight": 0.2,
    "motif_length_loss_weight": 0.1,
    "motif_class_weighting": true
  },
  "experiments": [
    {
      "name": "baseline"
    },
    {
      "name": "higher_motif_loss",
      "training": {
        "motif_loss_weight": 0.5
      }
    },
    {
      "name": "larger_model",
      "model": {
        "hidden_dim": 256,
        "num_layers": 2
      }
    }
  ]
}
```

For factorial sweeps, use the compact `matrix` form instead of hand-writing
every experiment. The pilot grid tests batch size, hidden dimension, one versus
two layers, epoch count, and curriculum on/off:

```json
{
  "matrix": {
    "batch_size": [8, 16],
    "hidden_dim": [64, 128],
    "num_layers": [1, 2],
    "epochs": [3, 6],
    "curriculum": [false, true]
  }
}
```

Curriculum experiments train through the configured phase datasets, then every
experiment is evaluated against the same saved holdout dataset. This makes the
final `holdout_*` columns in `comparison.csv` the fairest way to compare normal
training against curriculum training.

To benchmark against a previously saved synthetic dataset instead of generating
a new one, set `dataset.synthetic_data` to a `records.jsonl`, data directory, or
run directory. The runner records the source path in `source_dataset.json`.

Create plots and an interpretation report after a benchmark finishes:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/benchmark_interpret.py benchmarks/batch_hidden_layers_epochs_curriculum_pilot
```

This writes:

```text
benchmarks/batch_hidden_layers_epochs_curriculum_pilot/report/report.html
benchmarks/batch_hidden_layers_epochs_curriculum_pilot/report/summary.md
benchmarks/batch_hidden_layers_epochs_curriculum_pilot/report/ranked_results.csv
benchmarks/batch_hidden_layers_epochs_curriculum_pilot/report/top_runs.png
benchmarks/batch_hidden_layers_epochs_curriculum_pilot/report/factor_effects.png
benchmarks/batch_hidden_layers_epochs_curriculum_pilot/report/hidden_dim_by_batch_size.png
```

By default, the report ranks settings by `holdout_tract_f1`. To rank by another
metric:

```bash
python3 scripts/benchmark_interpret.py \
  benchmarks/batch_hidden_layers_epochs_curriculum_pilot \
  --metric holdout_motif_accuracy
```

Programmatic training looks like this:

```python
from ostrich_rnn.dataset import STRRecordDataset
from ostrich_rnn.model import RNNSTRDetector
from ostrich_rnn.motifs import MotifVocab
from ostrich_rnn.synthetic import SyntheticSTRGenerator
from ostrich_rnn.synthetic_io import add_motif_length_labels, make_unique_run_dir, save_synthetic_dataset
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
add_motif_length_labels(records, motif_vocab)
run_dir = make_unique_run_dir("runs")
save_synthetic_dataset(records, run_dir / "data", config={"seed": 7})

dataset = STRRecordDataset(records)
model = RNNSTRDetector(
    motif_classes=len(motif_vocab),
    motif_length_classes=6,
    hidden_dim=128,
    num_layers=1,
)
config = TrainConfig(
    epochs=10,
    batch_size=16,
    motif_loss_weight=0.2,
    motif_length_loss_weight=0.1,
    motif_class_weighting=True,
    checkpoint_dir=str(run_dir / "checkpoints"),
)

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
    "motif_length_labels": [-100] * 4 + [2] * 9 + [-100] * 2,
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
so the motif loss ignores them. Motif-length labels are 0-based, so motif length
1 uses class `0`, length 2 uses class `1`, and length 3 uses class `2`.
Outside-repeat bases should also use `-100` for this head.

During training, progress lines include both motif identity and motif-length
accuracy:

```text
epoch=1 train_loss=5.1748 val_loss=5.0719 base_f1=0.403 tract_f1=0.200 motif_acc=0.000 motif_len_acc=0.381
```

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
tract; outside bases use an ignore label. During post-processing, final motif
calls are driven primarily by sequence periodicity scoring over the predicted
tract. The neural motif head is still used as supporting evidence, but the
reported motif is usually recovered directly from the DNA sequence.
Motif length is handled by a separate per-base head with classes for motif
lengths 1 through the configured maximum. This gives the model an easier
auxiliary task and can improve motif identity learning.
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
