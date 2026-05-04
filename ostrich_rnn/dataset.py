"""Dataset wrappers for synthetic or real labeled sequence records."""

from __future__ import annotations

from collections.abc import Callable

from torch.utils.data import Dataset

from .motifs import MotifVocab
from .synthetic import SyntheticSTRGenerator


class STRRecordDataset(Dataset):
    """Dataset over already materialized labeled records.

    A real-data replacement can return records with the same keys:
    sequence_id, sequence, state_labels, motif_labels, repeats.
    """

    def __init__(self, records: list[dict], transform: Callable[[dict], dict] | None = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        return self.transform(record) if self.transform else record


class SyntheticSTRDataset(Dataset):
    """On-the-fly synthetic STR dataset."""

    def __init__(
        self,
        size: int,
        motif_vocab: MotifVocab,
        sequence_length: int = 1024,
        seed: int | None = None,
        **generator_kwargs,
    ):
        self.size = size
        self.generator = SyntheticSTRGenerator(
            motif_vocab=motif_vocab,
            sequence_length=sequence_length,
            seed=seed,
            **generator_kwargs,
        )

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        return self.generator.generate(f"synthetic_{idx:06d}")
