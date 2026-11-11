"""
Evaluation Metrics for Concept Prediction and Species Recognition
Calibrated Concept-Based Feedback Model (CSE 4385)

Computes:
  - Concept metrics: BCEWithLogitsLoss, macro F1 score across 112 concepts, binary accuracy, precision, recall.
  - Species metrics: CrossEntropyLoss, top-1 accuracy, top-5 accuracy across 200 bird species.
  - Joint recognition metrics: Weighted total loss and combined diagnostic indicators.
"""

from typing import Dict, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F


def compute_concept_metrics(
    logits: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes comprehensive concept-level prediction metrics.

    Args:
        logits: Unnormalized concept logits of shape [N, 112].
        targets: Binary ground truth concept targets of shape [N, 112] with values in {0, 1}.
        threshold: Decision threshold on sigmoid probabilities (default: 0.5).

    Returns:
        Dict with:
          - 'loss': BCEWithLogitsLoss float
          - 'macro_f1': Unweighted mean F1 score across all 112 concepts
          - 'mean_accuracy': Mean element-wise binary accuracy
          - 'macro_precision': Unweighted mean precision across all 112 concepts
          - 'macro_recall': Unweighted mean recall across all 112 concepts
          - 'exact_match_ratio': Proportion of images where all 112 concepts are exact matches
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().cpu().float()

    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).float()
    else:
        targets_t = targets.detach().cpu().float()

    if logits_t.shape != targets_t.shape:
        raise ValueError(
            f"Shape mismatch: logits {logits_t.shape} vs targets {targets_t.shape}"
        )

    # Compute loss using stable binary cross entropy with logits
    loss_val = F.binary_cross_entropy_with_logits(logits_t, targets_t).item()

    # Apply sigmoid to obtain concept probabilities in [0.0, 1.0]
    probs = torch.sigmoid(logits_t)
    preds = (probs >= threshold).float()

    probs_np = probs.numpy()
    preds_np = preds.numpy()
    targets_np = targets_t.numpy()

    # Binary element-wise accuracy across all N * 112 predictions
    correct_elements = (preds_np == targets_np).astype(np.float32)
    mean_accuracy = float(np.mean(correct_elements))

    # Exact match ratio: all 112 concepts must be correct for an image
    exact_matches = np.all(preds_np == targets_np, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))

    # Multi-label classification metrics across 112 concept attributes
    # Macro average gives equal weight to each of the 112 concept classes
    macro_f1 = float(f1_score(targets_np, preds_np, average="macro", zero_division=0))
    macro_precision = float(precision_score(targets_np, preds_np, average="macro", zero_division=0))
    macro_recall = float(recall_score(targets_np, preds_np, average="macro", zero_division=0))

    return {
        "loss": float(loss_val),
        "macro_f1": macro_f1,
        "mean_accuracy": mean_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "exact_match_ratio": exact_match_ratio,
    }


def compute_species_metrics(
    logits: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Computes species classification metrics across 200 bird species classes.

    Args:
        logits: Unnormalized species logits of shape [N, 200].
        targets: 0-indexed species ground truth labels of shape [N] with values in [0..199].

    Returns:
        Dict with:
          - 'loss': CrossEntropyLoss float
          - 'top1_accuracy': Top-1 species classification accuracy in [0.0, 1.0]
          - 'top5_accuracy': Top-5 species classification accuracy in [0.0, 1.0]
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().cpu().float()

    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).long()
    else:
        targets_t = targets.detach().cpu().long()

    if logits_t.dim() != 2 or targets_t.dim() != 1:
        raise ValueError(
            f"Expected logits [N, C] and targets [N], got {logits_t.shape} and {targets_t.shape}"
        )

    loss_val = F.cross_entropy(logits_t, targets_t).item()

    # Top-1 accuracy
    preds_top1 = torch.argmax(logits_t, dim=-1)
    top1_correct = (preds_top1 == targets_t).float()
    top1_acc = float(torch.mean(top1_correct).item())

    # Top-5 accuracy
    k = min(5, logits_t.shape[1])
    topk_preds = torch.topk(logits_t, k=k, dim=-1).indices
    top5_correct = (topk_preds == targets_t.unsqueeze(1)).any(dim=1).float()
    top5_acc = float(torch.mean(top5_correct).item())

    return {
        "loss": float(loss_val),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
    }


def compute_recognition_metrics(
    concept_logits: Union[torch.Tensor, np.ndarray],
    species_logits: Union[torch.Tensor, np.ndarray],
    concept_targets: Union[torch.Tensor, np.ndarray],
    species_targets: Union[torch.Tensor, np.ndarray],
    lambda_concept: float = 1.0,
    lambda_species: float = 1.0,
) -> Dict[str, float]:
    """
    Computes joint recognition metrics combining concept prediction and species recognition.

    Args:
        concept_logits: [N, 112] concept logits.
        species_logits: [N, 200] species logits.
        concept_targets: [N, 112] binary concept targets.
        species_targets: [N] 0-indexed species targets.
        lambda_concept: Concept loss weight multiplier.
        lambda_species: Species loss weight multiplier.

    Returns:
        Dict with total_loss, concept metrics, and species metrics.
    """
    c_metrics = compute_concept_metrics(concept_logits, concept_targets)
    s_metrics = compute_species_metrics(species_logits, species_targets)

    total_loss = lambda_concept * c_metrics["loss"] + lambda_species * s_metrics["loss"]

    return {
        "total_loss": float(total_loss),
        "concept_loss": c_metrics["loss"],
        "concept_macro_f1": c_metrics["macro_f1"],
        "concept_accuracy": c_metrics["mean_accuracy"],
        "species_loss": s_metrics["loss"],
        "species_top1_accuracy": s_metrics["top1_accuracy"],
        "species_top5_accuracy": s_metrics["top5_accuracy"],
    }
