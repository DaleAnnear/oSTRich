"""RNN sequence labeler for STR detection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .encoding import PAD_BASE_ID


@dataclass
class ModelOutput:
    state_logits: torch.Tensor
    motif_logits: torch.Tensor
    motif_length_logits: torch.Tensor


class RNNSTRDetector(nn.Module):
    """Bidirectional LSTM/GRU model with repeat-state and motif heads."""

    def __init__(
        self,
        motif_classes: int,
        base_vocab_size: int = 6,
        embedding_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        rnn_type: str = "lstm",
        motif_length_classes: int = 6,
    ):
        super().__init__()
        self.config = {
            "motif_classes": motif_classes,
            "base_vocab_size": base_vocab_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "rnn_type": rnn_type,
            "motif_length_classes": motif_length_classes,
        }
        self.embedding = nn.Embedding(base_vocab_size, embedding_dim, padding_idx=PAD_BASE_ID)
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        feature_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.state_head = nn.Linear(feature_dim, 4)
        self.motif_head = nn.Linear(feature_dim, motif_classes)
        self.motif_length_head = nn.Linear(feature_dim, motif_length_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> ModelOutput:
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        unpacked, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.shape[1])
        features = self.dropout(unpacked)
        return ModelOutput(
            state_logits=self.state_head(features),
            motif_logits=self.motif_head(features),
            motif_length_logits=self.motif_length_head(features),
        )
