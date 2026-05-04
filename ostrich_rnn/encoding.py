"""DNA sequence encoding and padded batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .labels import PAD_MOTIF_LABEL, PAD_STATE_LABEL


BASE_TO_ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
ID_TO_BASE = {v: k for k, v in BASE_TO_ID.items()}
PAD_BASE_ID = 5


def encode_sequence(seq: str) -> torch.Tensor:
    """Encode DNA bases as integer IDs, mapping unknown bases to N."""

    return torch.tensor([BASE_TO_ID.get(base.upper(), BASE_TO_ID["N"]) for base in seq], dtype=torch.long)


def decode_sequence(ids: Sequence[int]) -> str:
    return "".join(ID_TO_BASE.get(int(i), "N") for i in ids if int(i) != PAD_BASE_ID)


@dataclass
class EncodedBatch:
    sequence_ids: list[str]
    input_ids: torch.Tensor
    lengths: torch.Tensor
    state_labels: torch.Tensor | None = None
    motif_labels: torch.Tensor | None = None
    sequences: list[str] | None = None
    repeats: list[list[dict]] | None = None


def collate_examples(examples: list[dict]) -> EncodedBatch:
    """Pad variable-length examples for RNN minibatching."""

    lengths = torch.tensor([len(example["sequence"]) for example in examples], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_ids = torch.full((len(examples), max_len), PAD_BASE_ID, dtype=torch.long)
    state_labels = torch.full((len(examples), max_len), PAD_STATE_LABEL, dtype=torch.long)
    motif_labels = torch.full((len(examples), max_len), PAD_MOTIF_LABEL, dtype=torch.long)

    for row, example in enumerate(examples):
        encoded = encode_sequence(example["sequence"])
        length = encoded.numel()
        input_ids[row, :length] = encoded
        if "state_labels" in example and example["state_labels"] is not None:
            state_labels[row, :length] = torch.as_tensor(example["state_labels"], dtype=torch.long)
        if "motif_labels" in example and example["motif_labels"] is not None:
            motif_labels[row, :length] = torch.as_tensor(example["motif_labels"], dtype=torch.long)

    return EncodedBatch(
        sequence_ids=[example.get("sequence_id", f"seq_{i}") for i, example in enumerate(examples)],
        input_ids=input_ids,
        lengths=lengths,
        state_labels=state_labels,
        motif_labels=motif_labels,
        sequences=[example["sequence"] for example in examples],
        repeats=[example.get("repeats", []) for example in examples],
    )
