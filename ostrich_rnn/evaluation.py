"""Evaluation metrics for STR sequence labeling and tract calls."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import torch

from .labels import PAD_MOTIF_LABEL, PAD_MOTIF_LENGTH_LABEL, PAD_STATE_LABEL, RepeatState
from .motifs import MotifVocab
from .postprocess import RepeatCall, calls_from_logits


def binary_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def base_repeat_metrics(state_logits: torch.Tensor, labels: torch.Tensor) -> dict:
    pred_states = state_logits.argmax(dim=-1)
    valid = labels != PAD_STATE_LABEL
    pred_repeat = valid & (pred_states != RepeatState.OUTSIDE)
    true_repeat = valid & (labels != RepeatState.OUTSIDE)
    tp = int((pred_repeat & true_repeat).sum().item())
    fp = int((pred_repeat & ~true_repeat).sum().item())
    fn = int((~pred_repeat & true_repeat).sum().item())
    return binary_prf(tp, fp, fn)


def motif_accuracy(motif_logits: torch.Tensor, motif_labels: torch.Tensor) -> float:
    valid = motif_labels != PAD_MOTIF_LABEL
    if not bool(valid.any()):
        return 0.0
    pred = motif_logits.argmax(dim=-1)
    return float((pred[valid] == motif_labels[valid]).float().mean().item())


def motif_length_accuracy(motif_length_logits: torch.Tensor, motif_length_labels: torch.Tensor) -> float:
    valid = motif_length_labels != PAD_MOTIF_LENGTH_LABEL
    if not bool(valid.any()):
        return 0.0
    pred = motif_length_logits.argmax(dim=-1)
    return float((pred[valid] == motif_length_labels[valid]).float().mean().item())


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    intersection = max(0, right - left)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union else 0.0


def match_calls(
    predicted: list[RepeatCall],
    truth: list[dict],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[RepeatCall, dict]], list[RepeatCall], list[dict]]:
    unmatched_truth = truth.copy()
    matched: list[tuple[RepeatCall, dict]] = []
    unmatched_pred: list[RepeatCall] = []

    for call in predicted:
        best_idx = None
        best_iou = 0.0
        for idx, item in enumerate(unmatched_truth):
            iou = interval_iou((call.start, call.end), (int(item["start"]), int(item["end"])))
            if iou > best_iou:
                best_idx = idx
                best_iou = iou
        if best_idx is not None and best_iou >= iou_threshold:
            matched.append((call, unmatched_truth.pop(best_idx)))
        else:
            unmatched_pred.append(call)
    return matched, unmatched_pred, unmatched_truth


def tract_metrics(
    predictions: list[list[RepeatCall]],
    truths: list[list[dict]],
    iou_threshold: float = 0.5,
) -> dict:
    tp = fp = fn = 0
    start_errors: list[int] = []
    end_errors: list[int] = []
    length_errors: list[int] = []
    motif_correct = 0

    for predicted, truth in zip(predictions, truths):
        matched, unmatched_pred, unmatched_truth = match_calls(predicted, truth, iou_threshold)
        tp += len(matched)
        fp += len(unmatched_pred)
        fn += len(unmatched_truth)
        for call, item in matched:
            start_errors.append(abs(call.start - int(item["start"])))
            end_errors.append(abs(call.end - int(item["end"])))
            length_errors.append(abs(call.length_bp - int(item["length_bp"])))
            true_motifs = set(item.get("motifs", []))
            motif_correct += int(call.motif in true_motifs)

    prf = binary_prf(tp, fp, fn)
    prf.update(
        {
            "start_mae": mean(start_errors) if start_errors else 0.0,
            "end_mae": mean(end_errors) if end_errors else 0.0,
            "length_mae": mean(length_errors) if length_errors else 0.0,
            "tract_motif_accuracy": motif_correct / tp if tp else 0.0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    )
    return prf


@dataclass
class EvaluationResult:
    base: dict
    motif_accuracy: float
    motif_length_accuracy: float
    tract: dict


@torch.no_grad()
def evaluate_model(model, dataloader, motif_vocab: MotifVocab, device: torch.device) -> EvaluationResult:
    model.eval()
    base_scores = []
    motif_scores = []
    motif_length_scores = []
    all_predictions: list[list[RepeatCall]] = []
    all_truths: list[list[dict]] = []
    for batch in dataloader:
        input_ids = batch.input_ids.to(device)
        lengths = batch.lengths.to(device)
        state_labels = batch.state_labels.to(device)
        motif_labels = batch.motif_labels.to(device)
        motif_length_labels = batch.motif_length_labels.to(device)
        outputs = model(input_ids, lengths)
        base_scores.append(base_repeat_metrics(outputs.state_logits.cpu(), state_labels.cpu()))
        motif_scores.append(motif_accuracy(outputs.motif_logits.cpu(), motif_labels.cpu()))
        motif_length_scores.append(motif_length_accuracy(outputs.motif_length_logits.cpu(), motif_length_labels.cpu()))
        for i, sequence in enumerate(batch.sequences):
            length = len(sequence)
            all_predictions.append(
                calls_from_logits(
                    sequence_id=batch.sequence_ids[i],
                    sequence=sequence,
                    state_logits=outputs.state_logits[i, :length].cpu(),
                    motif_logits=outputs.motif_logits[i, :length].cpu(),
                    motif_length_logits=outputs.motif_length_logits[i, :length].cpu(),
                    motif_vocab=motif_vocab,
                )
            )
        all_truths.extend(batch.repeats)

    base = {key: mean(score[key] for score in base_scores) for key in base_scores[0]}
    return EvaluationResult(
        base=base,
        motif_accuracy=mean(motif_scores) if motif_scores else 0.0,
        motif_length_accuracy=mean(motif_length_scores) if motif_length_scores else 0.0,
        tract=tract_metrics(all_predictions, all_truths),
    )
