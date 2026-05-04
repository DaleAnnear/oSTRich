"""Training utilities for the RNN STR detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .encoding import EncodedBatch, collate_examples
from .evaluation import evaluate_model
from .labels import PAD_MOTIF_LABEL, PAD_STATE_LABEL, RepeatState
from .runtime import describe_device, resolve_device


def move_batch(batch: EncodedBatch, device: torch.device) -> EncodedBatch:
    batch.input_ids = batch.input_ids.to(device)
    batch.lengths = batch.lengths.to(device)
    if batch.state_labels is not None:
        batch.state_labels = batch.state_labels.to(device)
    if batch.motif_labels is not None:
        batch.motif_labels = batch.motif_labels.to(device)
    return batch


def default_state_weights(device: torch.device) -> torch.Tensor:
    """Upweight repeat labels to counter background class imbalance."""

    weights = torch.ones(4, dtype=torch.float32, device=device)
    weights[RepeatState.OUTSIDE] = 0.2
    weights[RepeatState.START] = 3.0
    weights[RepeatState.INSIDE] = 1.5
    weights[RepeatState.END] = 3.0
    return weights


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    motif_loss_weight: float = 0.5
    validation_fraction: float = 0.2
    patience: int = 5
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0


def compute_loss(outputs, batch: EncodedBatch, state_loss_fn, motif_loss_fn, motif_loss_weight: float) -> torch.Tensor:
    state_loss = state_loss_fn(outputs.state_logits.transpose(1, 2), batch.state_labels)
    motif_loss = motif_loss_fn(outputs.motif_logits.transpose(1, 2), batch.motif_labels)
    return state_loss + motif_loss_weight * motif_loss


def make_loaders(dataset, config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    val_size = max(1, int(len(dataset) * config.validation_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(13))
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_examples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_examples,
    )
    return train_loader, val_loader


def save_checkpoint(path: Path, model, optimizer, epoch: int, val_loss: float, motif_vocab) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "motifs": motif_vocab.motifs,
            "collapse_reverse_complement": motif_vocab.collapse_reverse_complement,
            "model_config": getattr(model, "config", None),
        },
        path,
    )


def train_model(model, dataset, motif_vocab, config: TrainConfig, device: str | torch.device | None = "auto") -> dict:
    device = resolve_device(device)
    print(f"training_device={describe_device(device)}")
    model.to(device)
    train_loader, val_loader = make_loaders(dataset, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    state_loss_fn = nn.CrossEntropyLoss(weight=default_state_weights(device), ignore_index=PAD_STATE_LABEL)
    motif_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_MOTIF_LABEL)

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    best_path = Path(config.checkpoint_dir) / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch.input_ids, batch.lengths)
            loss = compute_loss(outputs, batch, state_loss_fn, motif_loss_fn, config.motif_loss_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                outputs = model(batch.input_ids, batch.lengths)
                loss = compute_loss(outputs, batch, state_loss_fn, motif_loss_fn, config.motif_loss_weight)
                val_losses.append(float(loss.item()))

        train_loss = sum(train_losses) / max(1, len(train_losses))
        val_loss = sum(val_losses) / max(1, len(val_losses))
        metrics = evaluate_model(model, val_loader, motif_vocab, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "base_f1": metrics.base["f1"],
            "tract_f1": metrics.tract["f1"],
            "motif_accuracy": metrics.motif_accuracy,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"base_f1={row['base_f1']:.3f} tract_f1={row['tract_f1']:.3f} motif_acc={row['motif_accuracy']:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_loss, motif_vocab)
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    return {"history": history, "best_checkpoint": str(best_path)}
