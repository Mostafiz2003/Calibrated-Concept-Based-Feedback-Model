"""
Concept Evaluation and Per-Concept Breakdown Analysis
Calibrated Concept-Based Feedback Model (CSE 4385)

Provides detailed per-concept and aggregate evaluation for the 112 CUB-200-2011 concepts.
Computes:
  - Per-concept metrics: Precision, Recall, F1-score, Support, Positive Prevalence, Accuracy.
  - Aggregate metrics: Macro / Micro Precision, Recall, F1, Mean Accuracy, BCE Loss.
  - Diagnostic distributions: Top/bottom concepts, zero-support concepts, zero-predicted concepts,
    and class imbalance analysis.
"""

from csv import DictWriter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def load_concept_metadata(mapping_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads concept metadata from data/selected_concepts_112.json.

    Returns:
        List of dicts with 'concept_index', 'cub_attribute_id', 'attribute_name', etc.
    """
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(f"Concept mapping file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    concepts = data.get("concepts", [])
    if len(concepts) != 112:
        raise ValueError(f"Expected 112 concepts in mapping, found {len(concepts)}.")

    # Sort strictly by concept_index
    concepts_sorted = sorted(concepts, key=lambda c: int(c["concept_index"]))
    return concepts_sorted


@torch.no_grad()
def collect_validation_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Evaluates model across the given DataLoader, collecting all raw logits and ground-truth targets.

    Args:
        model: ConceptPredictor model instance.
        data_loader: DataLoader for the validation split (strictly non-training).
        device: Device to run evaluation on ('cpu' or 'cuda').
        criterion: Optional loss function (defaults to BCEWithLogitsLoss).

    Returns:
        Dict with:
          - 'logits': np.ndarray of shape [N, 112]
          - 'targets': np.ndarray of shape [N, 112]
          - 'probabilities': np.ndarray of shape [N, 112]
          - 'loss': float, overall dataset BCEWithLogitsLoss
          - 'num_samples': int (N)
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(data_loader):
        images = batch["image"].to(device)
        targets = batch["concept_labels"].to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss ({loss.item()}) at evaluation batch {batch_idx}")

        total_loss += loss.item()
        batch_count += 1

        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    if batch_count == 0:
        raise ValueError("Data loader contained 0 batches.")

    logits_concat = np.vstack(all_logits)
    targets_concat = np.vstack(all_targets)
    probs_concat = 1.0 / (1.0 + np.exp(-logits_concat))
    avg_loss = total_loss / batch_count

    return {
        "logits": logits_concat,
        "targets": targets_concat,
        "probabilities": probs_concat,
        "loss": float(avg_loss),
        "num_samples": int(len(targets_concat)),
    }


def compute_per_concept_breakdown(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    concept_metadata: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Computes per-concept metrics for all 112 concepts.

    Args:
        targets: Binary ground truth array of shape [N, 112].
        probabilities: Predicted concept probabilities of shape [N, 112] in [0, 1].
        threshold: Decision threshold on probabilities (default: 0.5).
        concept_metadata: Optional metadata list containing concept names and attribute IDs.

    Returns:
        List of 112 dicts containing metrics for each concept.
    """
    num_samples, num_concepts = targets.shape
    if probabilities.shape != (num_samples, num_concepts):
        raise ValueError(
            f"Shape mismatch: targets {targets.shape} vs probabilities {probabilities.shape}"
        )

    predictions = (probabilities >= threshold).astype(np.float32)

    results: List[Dict[str, Any]] = []

    for j in range(num_concepts):
        y_true = targets[:, j]
        y_pred = predictions[:, j]

        support = int(np.sum(y_true == 1))
        prevalence = float(support / num_samples) if num_samples > 0 else 0.0

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        pred_pos = tp + fp

        accuracy = float((tp + tn) / num_samples) if num_samples > 0 else 0.0
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        # Retrieve metadata if available
        if concept_metadata and j < len(concept_metadata):
            c_meta = concept_metadata[j]
            attr_id = int(c_meta.get("cub_attribute_id", j + 1))
            attr_name = str(c_meta.get("attribute_name", f"concept_{j}"))
        else:
            attr_id = j + 1
            attr_name = f"concept_{j}"

        results.append({
            "concept_index": j,
            "cub_attribute_id": attr_id,
            "attribute_name": attr_name,
            "support": support,
            "positive_prevalence": round(prevalence, 6),
            "predicted_positives": pred_pos,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "accuracy": round(accuracy, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1_score": round(f1, 6),
        })

    return results


def compute_aggregate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    loss_val: float,
    threshold: float = 0.5,
    per_concept_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes macro and micro aggregate metrics across all 112 concepts.

    Args:
        targets: Binary ground truth array of shape [N, 112].
        probabilities: Predicted concept probabilities of shape [N, 112].
        loss_val: Precomputed BCEWithLogitsLoss.
        threshold: Decision threshold on probabilities (default: 0.5).
        per_concept_metrics: Precomputed per-concept breakdown list (optional).

    Returns:
        Dict of aggregate metrics and diagnostic counts.
    """
    num_samples, num_concepts = targets.shape
    predictions = (probabilities >= threshold).astype(np.float32)

    # Element-wise mean binary accuracy across all N * 112 elements
    mean_accuracy = float(np.mean(predictions == targets))

    # Exact match ratio (all 112 concepts predicted correctly for a sample)
    exact_matches = np.all(predictions == targets, axis=1)
    exact_match_ratio = float(np.mean(exact_matches))

    # Multi-label Macro metrics (unweighted mean across concepts)
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    macro_precision = float(precision_score(targets, predictions, average="macro", zero_division=0))
    macro_recall = float(recall_score(targets, predictions, average="macro", zero_division=0))

    # Multi-label Micro metrics (aggregate TP, FP, FN across all cells)
    micro_f1 = float(f1_score(targets, predictions, average="micro", zero_division=0))
    micro_precision = float(precision_score(targets, predictions, average="micro", zero_division=0))
    micro_recall = float(recall_score(targets, predictions, average="micro", zero_division=0))

    # Diagnostics from per_concept_metrics
    if per_concept_metrics is None:
        per_concept_metrics = compute_per_concept_breakdown(targets, probabilities, threshold=threshold)

    zero_support_concepts = [
        c["attribute_name"] for c in per_concept_metrics if c["support"] == 0
    ]
    zero_predicted_concepts = [
        c["attribute_name"] for c in per_concept_metrics if c["predicted_positives"] == 0
    ]

    # Prevalences for imbalance analysis
    prevalences = [c["positive_prevalence"] for c in per_concept_metrics]
    min_prev = float(np.min(prevalences))
    max_prev = float(np.max(prevalences))
    mean_prev = float(np.mean(prevalences))
    median_prev = float(np.median(prevalences))

    return {
        "evaluation_split": "validation",
        "num_samples": num_samples,
        "num_concepts": num_concepts,
        "threshold": float(threshold),
        "loss": round(float(loss_val), 6),
        "macro_f1": round(macro_f1, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "micro_f1": round(micro_f1, 6),
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "mean_concept_accuracy": round(mean_accuracy, 6),
        "exact_match_ratio": round(exact_match_ratio, 6),
        "zero_support_count": len(zero_support_concepts),
        "zero_support_concepts": zero_support_concepts,
        "zero_predicted_positives_count": len(zero_predicted_concepts),
        "zero_predicted_positives_concepts": zero_predicted_concepts,
        "prevalence_summary": {
            "min_prevalence": round(min_prev, 6),
            "max_prevalence": round(max_prev, 6),
            "mean_prevalence": round(mean_prev, 6),
            "median_prevalence": round(median_prev, 6),
        },
    }


def save_week5_analysis_artifacts(
    per_concept_metrics: List[Dict[str, Any]],
    aggregate_metrics: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Saves:
      1. outputs/week5_analysis/per_concept_metrics.csv
      2. outputs/week5_analysis/aggregate_metrics.json
      3. outputs/week5_analysis/analysis_summary.md

    Returns:
        Dict mapping artifact name to Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "per_concept_metrics.csv"
    json_path = out_dir / "aggregate_metrics.json"
    summary_path = out_dir / "analysis_summary.md"

    # 1. Save CSV
    if per_concept_metrics:
        fieldnames = list(per_concept_metrics[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_concept_metrics:
                writer.writerow(row)
        logger.info(f"Saved per-concept metrics to {csv_path}")

    # 2. Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_metrics, f, indent=2)
    logger.info(f"Saved aggregate metrics to {json_path}")

    # Sort concepts by F1 descending for top / bottom extraction
    sorted_by_f1 = sorted(per_concept_metrics, key=lambda c: c["f1_score"], reverse=True)
    top_10 = sorted_by_f1[:10]
    bottom_10 = sorted_by_f1[-10:]

    # 3. Save Markdown Analysis Summary
    summary_md = f"""# Week 5 Baseline Analysis Summary: ResNet-18 Concept Predictor

**Dataset:** CUB-200-2011 Validation Split (600 images)  
**Model Checkpoint:** `outputs/concept_resnet18_3epochs/best_concept_model.pth`  
**Backbone:** ResNet-18 (ImageNet Pretrained, 3 epochs fine-tuning)  
**Concepts:** 112 Canonical Filtered Benchmark Concepts  
**Decision Threshold:** {aggregate_metrics['threshold']} (Fixed documented default; no validation tuning)

---

## 1. Aggregate Validation Metrics

| Metric | Value |
| :--- | :--- |
| **Validation BCE Loss** | **{aggregate_metrics['loss']:.4f}** |
| **Macro F1** | **{aggregate_metrics['macro_f1']:.4f}** ({aggregate_metrics['macro_f1']*100:.2f}%) |
| **Macro Precision** | **{aggregate_metrics['macro_precision']:.4f}** ({aggregate_metrics['macro_precision']*100:.2f}%) |
| **Macro Recall** | **{aggregate_metrics['macro_recall']:.4f}** ({aggregate_metrics['macro_recall']*100:.2f}%) |
| **Micro F1** | **{aggregate_metrics['micro_f1']:.4f}** ({aggregate_metrics['micro_f1']*100:.2f}%) |
| **Micro Precision** | **{aggregate_metrics['micro_precision']:.4f}** ({aggregate_metrics['micro_precision']*100:.2f}%) |
| **Micro Recall** | **{aggregate_metrics['micro_recall']:.4f}** ({aggregate_metrics['micro_recall']*100:.2f}%) |
| **Mean Concept Accuracy** | **{aggregate_metrics['mean_concept_accuracy']:.4f}** ({aggregate_metrics['mean_concept_accuracy']*100:.2f}%) |
| **Exact Match Ratio** | **{aggregate_metrics['exact_match_ratio']:.4f}** ({aggregate_metrics['exact_match_ratio']*100:.2f}%) |

---

## 2. Top 10 Concepts by F1 Score

| Rank | Concept Index | CUB Attr ID | Attribute Name | Support (N=600) | Prevalence | Precision | Recall | F1 Score | Accuracy |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, c in enumerate(top_10, 1):
        summary_md += f"| {rank} | {c['concept_index']} | {c['cub_attribute_id']} | `{c['attribute_name']}` | {c['support']} | {c['positive_prevalence']:.3f} | {c['precision']:.4f} | {c['recall']:.4f} | **{c['f1_score']:.4f}** | {c['accuracy']:.4f} |\n"

    summary_md += """
---

## 3. Bottom 10 Concepts by F1 Score

| Rank | Concept Index | CUB Attr ID | Attribute Name | Support (N=600) | Prevalence | Precision | Recall | F1 Score | Accuracy |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, c in enumerate(bottom_10, 1):
        summary_md += f"| {rank} | {c['concept_index']} | {c['cub_attribute_id']} | `{c['attribute_name']}` | {c['support']} | {c['positive_prevalence']:.3f} | {c['precision']:.4f} | {c['recall']:.4f} | **{c['f1_score']:.4f}** | {c['accuracy']:.4f} |\n"

    summary_md += f"""
---

## 4. Diagnostic Indicators & Imbalance Findings

* **Concepts with Zero Ground-Truth Support:** {aggregate_metrics['zero_support_count']}
"""
    if aggregate_metrics['zero_support_concepts']:
        for name in aggregate_metrics['zero_support_concepts']:
            summary_md += f"  - `{name}`\n"
    else:
        summary_md += "  *(None: all 112 concepts appear at least once in the validation set)*\n"

    summary_md += f"""
* **Concepts with Zero Predicted Positives:** {aggregate_metrics['zero_predicted_positives_count']}
"""
    if aggregate_metrics['zero_predicted_positives_concepts']:
        for name in aggregate_metrics['zero_predicted_positives_concepts']:
            summary_md += f"  - `{name}`\n"
    else:
        summary_md += "  *(None)*\n"

    summary_md += f"""
* **Prevalence Distribution Across 112 Concepts:**
  - Minimum Prevalence: `{aggregate_metrics['prevalence_summary']['min_prevalence']:.4f}`
  - Maximum Prevalence: `{aggregate_metrics['prevalence_summary']['max_prevalence']:.4f}`
  - Mean Prevalence: `{aggregate_metrics['prevalence_summary']['mean_prevalence']:.4f}`
  - Median Prevalence: `{aggregate_metrics['prevalence_summary']['median_prevalence']:.4f}`

* **Analysis of Class Imbalance & Threshold Behavior:**
  - The mean positive prevalence across all 112 concepts is only ~{aggregate_metrics['prevalence_summary']['mean_prevalence']*100:.1f}%, indicating significant label sparsity (over 80% negative labels).
  - High mean binary accuracy ({aggregate_metrics['mean_concept_accuracy']*100:.2f}%) contrasts with modest Macro F1 ({aggregate_metrics['macro_f1']*100:.2f}%), reflecting standard class imbalance where predicting 0 achieves high accuracy.
  - Using a fixed 0.5 threshold causes conservative positive predictions for rare concepts with low training frequency, yielding zero predicted positives for several rare attributes. Post-hoc calibration (Week 7) will directly address this.
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info(f"Saved analysis summary markdown to {summary_path}")

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": summary_path,
    }

