<p align="center">
  <img src="ostrich_rnn/181bd6fe-5b3b-42bc-97e6-81e66178f051.png" alt="oSTRich logo" width="240">
</p>

<h3 align="center">oSTRich RNN STR detector</h3>

---

oSTRich is a modular PyTorch starter implementation for detecting short tandem repeat (STR) tracts in DNA sequences with a bidirectional recurrent neural network.

## Table of Contents

- [About](#about)
- [Repository layout](#repository-layout)
- [Getting Started](#getting-started)
  - [Prerequisites and installation](#prerequisites-and-installation)
  - [Quick start](#quick-start)
- [Running the tests](#running-the-tests)
- [Usage](#usage)
  - [Synthetic training](#synthetic-training)
  - [Training on an existing synthetic dataset](#training-on-an-existing-synthetic-dataset)
  - [Curriculum training](#curriculum-training)
  - [FASTA inference](#fasta-inference)
  - [Benchmarking](#benchmarking)
  - [Benchmark reports](#benchmark-reports)
  - [Programmatic training](#programmatic-training)
  - [Real labelled data](#real-labelled-data)
- [TRExplorer truth set](#trexplorer-truth-set)
- [Design notes](#design-notes)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

## About

oSTRich frames STR detection as sequence labelling. For each base, the model predicts whether it is outside a repeat, at a repeat start, inside a repeat, or at a repeat end. Two auxiliary per-base heads predict the repeat motif class and motif length for bases within a tract. Post-processing turns those base-level predictions into a repeat table.

The synthetic generator is separate from the dataset and model code, so a real labelled dataset can replace it later without changing the rest of the pipeline.

## Repository layout

- `ostrich_rnn/motifs.py` - motif vocabulary, canonicalisation, and reverse-complement handling
- `ostrich_rnn/encoding.py` - DNA integer encoding and padded batching
- `ostrich_rnn/synthetic.py` - synthetic STR sequence generator
- `ostrich_rnn/dataset.py` - PyTorch dataset wrappers
- `ostrich_rnn/model.py` - BiLSTM/BiGRU sequence labeler
- `ostrich_rnn/postprocess.py` - repeat-tract extraction and motif estimation
- `ostrich_rnn/evaluation.py` - base-level and tract-level metrics
- `ostrich_rnn/training.py` - training loop, validation, checkpoints, and early stopping
- `ostrich_rnn/benchmark.py` - automated benchmark sweeps and comparison tables
- `ostrich_rnn/benchmark_report.py` - benchmark interpretation reports and plots
- `ostrich_rnn/runtime.py` - CPU/GPU device-selection helpers
- `ostrich_rnn/synthetic_io.py` - saving generated synthetic datasets and summaries
- `ostrich_rnn/inference.py` - FASTA-to-CSV inference
- `scripts/train_synthetic.py` - a brief synthetic-training run
- `scripts/benchmark_synthetic.py` - a JSON-driven synthetic benchmark runner
- `scripts/interpret_benchmark.py` - a benchmark plotting and interpretation runner
- `tests/test_sanity.py` - simple sanity checks

## Getting Started

### Prerequisites and installation

The usage examples below assume that you are working in WSL Ubuntu. Navigate to the repository:

```bash
cd "/mnt/c/Users/Dale/OneDrive - Universiteit Antwerpen/Post Doc/Bioinfo/AI/oSTRich"
```

Install or refresh the dependencies:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

For a standard Python environment, the equivalent installation command is:

```bash
pip install -r requirements.txt
```

### Quick start

Run the sanity tests:

```bash
pytest -q
```

Run a small synthetic training example:

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

Coordinates are 0-based, with an inclusive `start` and exclusive `end`, matching Python slicing.

## Running the tests

From WSL Ubuntu, run:

```bash
python3 -m pytest -q
```

## Usage

Check whether PyTorch can see your GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

`--device auto` uses CUDA when PyTorch can see a GPU and falls back to CPU otherwise. Use `--device cuda` when a run should fail unless a GPU is available.

### Synthetic training

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

`--patience` controls early stopping. For example, `--patience 8` allows training to continue until validation loss has failed to improve for eight consecutive epochs.

Motif class weighting is enabled by default. It upweights rare motif classes in the motif cross-entropy loss so common motifs do not dominate training. Use `--no-motif-class-weighting` to disable it. The motif-identity and motif-length auxiliary losses are controlled with `--motif-loss-weight` and `--motif-length-loss-weight`. Their defaults are deliberately modest: `0.2` and `0.1`, so boundary detection remains the main training objective.

The default RNN capacity for the synthetic-training script is `--hidden-dim 128 --num-layers 1`. Increase `--num-layers` to `2` only when validation metrics suggest that the one-layer model is underfitting.

Synthetic sequences are 1024 bp by default. Repeats are perfect by default: substitution, insertion, deletion, and motif-interruption rates are all `0.0` unless explicitly enabled. To generate imperfect repeats:

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

`--motif-interruption-rate` replaces occasional motif copies with non-motif DNA inside a tract. By default, interruption units are the same length as the motif; use `--interruption-min-len` and `--interruption-max-len` to vary their length.

Motifs are generated up to 6 bp by default. Synthetic motif sampling is length-balanced: the generator first samples a motif length (for example, monomer, dimer, or trimer), then samples a motif from that length class. This keeps short motifs from being swamped by the much larger number of possible longer motifs. The balance is approximate across repeat tracts, not necessarily across total repeat bases.

Each synthetic-training run creates a unique directory under `runs/`, for example:

```text
runs/synthetic_20260504_123456_a1b2c3d4/
```

The directory contains the generated data and the best checkpoint:

```text
runs/<run_name>/data/records.jsonl
runs/<run_name>/data/sequences.fa
runs/<run_name>/data/sequence_summary.csv
runs/<run_name>/data/repeat_summary.csv
runs/<run_name>/data/dataset_summary.json
runs/<run_name>/data/generation_config.json
runs/<run_name>/checkpoints/best.pt
```

`sequence_summary.csv` reports the length, GC content, number of `N` bases, repeat count, interruption count, total repeat bases, repeat fraction, motifs, and compact repeat locations for each generated sequence. `repeat_summary.csv` has one row per repeat with its start, end, length, motif or compound motifs, motif IDs, copy number, compound status, and interruption coordinates and sequences. `dataset_summary.json` gives dataset-level totals, motif counts, motif-length counts, and the total interruption count.

### Training on an existing synthetic dataset

Train from an existing saved synthetic dataset:

```bash
python3 scripts/train_synthetic.py \
  --synthetic-data runs/<previous_run>/data/records.jsonl \
  --epochs 10 \
  --batch-size 16 \
  --validation-fraction 0.2 \
  --device auto
```

`--synthetic-data` can point to any of the following:

```text
runs/<previous_run>/data/records.jsonl
runs/<previous_run>/data/
runs/<previous_run>/
```

When `--synthetic-data` is provided, no new synthetic dataset is generated. A new run directory is still created for the checkpoint and a small `source_dataset.json` file that records which dataset was used.

### Curriculum training

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

- `phase1_perfect`: perfect simple repeats, with no mutations, interruptions, or compounds
- `phase2_low_imperfect`: low mutation and interruption rates, with rare compounds
- `phase3_harder_imperfect_compound`: higher mutation and interruption rates, with more compounds

`--curriculum-phase-epochs` is interpreted as percentages of `--epochs`, not literal epoch counts. The default, `20,35,45`, trains the three phases for `12`, `21`, and `27` epochs when `--epochs 60` is supplied. If the percentages do not produce whole epochs, the values are rounded and adjusted so that the phase epochs still add up exactly to `--epochs`.

Each phase writes its own data directory:

```text
runs/<run_name>/data/phase1_perfect/
runs/<run_name>/data/phase2_low_imperfect/
runs/<run_name>/data/phase3_harder_imperfect_compound/
```

Each phase also writes its own checkpoint directory. After phase 3, the final phase's best checkpoint is copied to:

```text
runs/<run_name>/checkpoints/best.pt
```

The run directory also contains `curriculum_summary.json`, which records the phase datasets and checkpoints used.

### FASTA inference

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

### Benchmarking

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

The benchmark runner creates one shared dataset, trains each experiment against the same dataset and split, and writes a comparison table. Each benchmark directory contains:

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

Use benchmark experiments to compare training settings, model sizes, auxiliary-loss weights, class weighting, imperfect-repeat data, or another variable that should be tested against a fixed dataset:

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

For factorial sweeps, use the compact `matrix` form instead of writing every experiment by hand. The pilot grid tests batch size, hidden dimension, one versus two layers, epoch count, and curriculum on or off:

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

Curriculum experiments train through the configured phase datasets, then every experiment is evaluated against the same saved holdout dataset. This makes the final `holdout_*` columns in `comparison.csv` the fairest way to compare ordinary training against curriculum training.

To benchmark against a previously saved synthetic dataset instead of generating a new one, set `dataset.synthetic_data` to a `records.jsonl` file, data directory, or run directory. The runner records the source path in `source_dataset.json`.

### Benchmark reports

Create plots and an interpretation report after a benchmark finishes:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/interpret_benchmark.py benchmarks/batch_hidden_layers_epochs_curriculum_pilot
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

By default, the report ranks settings by `holdout_tract_f1`. To rank by a different metric:

```bash
python3 scripts/interpret_benchmark.py \
  benchmarks/batch_hidden_layers_epochs_curriculum_pilot \
  --metric holdout_motif_accuracy
```

### Programmatic training

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

### Real labelled data

To replace synthetic data with real labels, create records in the shape expected by `STRRecordDataset`:

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

Repeat-state labels are `0=outside`, `1=start`, `2=inside`, and `3=end`. Motif labels use IDs from `MotifVocab`; outside-repeat bases should use `-100` so the motif loss ignores them. Motif-length labels are 0-based: motif length 1 uses class `0`, length 2 uses class `1`, and length 3 uses class `2`. Outside-repeat bases should also use `-100` for this head.

During training, progress lines include motif-identity and motif-length accuracy:

```text
epoch=1 train_loss=5.1748 val_loss=5.0719 base_f1=0.403 tract_f1=0.200 motif_acc=0.000 motif_len_acc=0.381
```

## TRExplorer truth set

Download the TRExplorer v2 variation-cluster catalogue and prepare a clusters-only BED truth set:

```bash
python3 scripts/download_trexplorer_truth_set.py
```

This writes ignored local data under:

```text
truth_sets/trexplorer_v2/TRExplorer.variation_clusters_and_isolated_TRs_v2.hg38.TRGT.bed.gz
truth_sets/trexplorer_v2/TRExplorer.variation_clusters_v2.hg38.TRGT.bed.gz
truth_sets/trexplorer_v2/truth_set_summary.json
```

The upstream BED is the Broad TRExplorer v2 release asset for TRGT. It contains variation clusters and isolated TRs. The extracted `TRExplorer.variation_clusters_v2.hg38.TRGT.bed.gz` file keeps only rows whose TRGT `STRUC` attribute is a variation cluster (`<VC:...>`), creating a clean truth-set BED for complex variation-cluster regions. Coordinates follow the BED convention: 0-based, inclusive start and exclusive end.

## Design notes

STR detection is naturally a sequence-labelling problem because repeat tracts are contiguous spans within a larger DNA sequence. A label at each base gives the model enough information to recover tract boundaries while allowing standard minibatch training.

A bidirectional LSTM or GRU is useful because a base's label often depends on flanking context. For example, deciding whether a base is at a repeat boundary requires seeing both the preceding repeat pattern and the sequence that follows.

Motif prediction uses a second per-base classification head. During training, bases inside a repeat are assigned the canonical motif class for that tract; outside bases use an ignore label. During post-processing, final motif calls are driven primarily by sequence-periodicity scoring over the predicted tract. The neural motif head is supporting evidence, but the reported motif is usually recovered directly from the DNA sequence.

Motif length uses a separate per-base head with classes for lengths 1 through the configured maximum. This gives the model an easier auxiliary task and can improve motif-identity learning. By default motifs are strand-specific, so `A` and `T` are separate classes. Use `MotifVocab.build(collapse_reverse_complement=True)` to collapse reverse-complement-equivalent motifs.

## Limitations

- Synthetic repeats are simpler than real genomic repeats.
- Very long repeat copy numbers should be handled with windows; otherwise, recurrent training becomes memory-heavy.
- Imperfect and compound repeats can blur exact motif labels.
- The current post-processor is intentionally transparent rather than exhaustive.

## Future improvements

- Add a CRF layer over the repeat-state head to enforce legal label transitions.
- Add a small CNN front end to expose local periodicity before the RNN.
- Use repeat-aware dynamic programming during post-processing.
- Train with real labelled loci and hard negatives from low-complexity sequence.
- Add multi-label motif targets for compound repeats instead of one dominant motif per base.
