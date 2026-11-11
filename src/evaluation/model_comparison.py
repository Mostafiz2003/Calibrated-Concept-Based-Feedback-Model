"""
Standardized Model Evaluation and Comparison System
Calibrated Concept-Based Feedback Model (CSE 4385)

Evaluates multiple trained recognition architectures on the exact same 600 validation images:
- Species Top-1 Accuracy, Top-5 Accuracy, Species Macro-F1, Precision, Recall
- Concept Macro-F1, Concept Micro-F1, Mean Accuracy
- Total and Trainable Parameter Counts
- Generates:
  - outputs/model_comparison/model_comparison.csv
  - outputs/model_comparison/model_comparison.json
  - outputs/model_comparison/model_comparison_summary.md
"""

from csv import DictWriter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.recognition_model import DualHeadOutput
from src.training.metrics import compute_recognition_metrics

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_recognition_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: Union[str, torch.device] = "cpu",
    lambda_concept: float = 0.2,
) -> Dict[str, float]:
    """
    Evaluates a dual-head model across the validation split.

    Args:
        model: Dual-head model returning DualHeadOutput.
        val_loader: DataLoader over validation split (strictly 600 images).
        device: 'cpu' or 'cuda'.
        lambda_concept: Loss weight for concept BCE.

    Returns:
        Dictionary of validation metrics.
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    all_species_logits = []
    all_species_targets = []
    all_concept_logits = []
    all_concept_targets = []

    for batch in val_loader:
        images = batch["image"].to(device)
        sp_targets = batch["species_label"].to(device)
        c_targets = batch["concept_labels"].to(device)

        outputs = model(images)
        if isinstance(outputs, DualHeadOutput):
            sp_logits = outputs.species_logits
            c_logits = outputs.concept_logits
        else:
            c_logits, sp_logits = outputs[0], outputs[1]

        all_species_logits.append(sp_logits.detach().cpu().numpy())
        all_species_targets.append(sp_targets.detach().cpu().numpy())
        all_concept_logits.append(c_logits.detach().cpu().numpy())
        all_concept_targets.append(c_targets.detach().cpu().numpy())

    metrics = compute_recognition_metrics(
        concept_logits=np.vstack(all_concept_logits),
        concept_targets=np.vstack(all_concept_targets),
        species_logits=np.vstack(all_species_logits),
        species_targets=np.concatenate(all_species_targets),
        lambda_concept=lambda_concept,
    )
    return metrics


def save_model_comparison_reports(
    comparison_records: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "outputs/model_comparison",
) -> Dict[str, Path]:
    """
    Saves comparison records to CSV, JSON, and formatted Markdown summary.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "model_comparison.csv"
    json_path = out_dir / "model_comparison.json"
    summary_path = out_dir / "model_comparison_summary.md"

    # 1. CSV
    if comparison_records:
        fieldnames = list(comparison_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in comparison_records:
                writer.writerow(r)

    # 2. JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_records, f, indent=2)

    # 3. Markdown
    md_content = """# Model Architecture Comparison Report: Bird Species Recognition

**Evaluation Split:** CUB-200-2011 Validation Set (600 images, 3 images/class across all 200 species)  
**Primary Objective:** Bird Species Recognition (Top-1 / Top-5 Accuracy)  
**Auxiliary Objective:** 112-Concept Interpretability (Macro / Micro F1)  

---

## Comparative Performance Overview

| Architecture | Experiment ID | Total Parameters | Species Top-1 (%) | Species Top-5 (%) | Species Macro-F1 | Concept Macro-F1 | Concept Micro-F1 | Total Val Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in comparison_records:
        md_content += (
            f"| `{r['model_name']}` | {r.get('experiment_id', 'N/A')} | {r.get('total_params', 0):,} | "
            f"**{r.get('species_top1_acc', 0.0)*100:.2f}%** | {r.get('species_top5_acc', 0.0)*100:.2f}% | "
            f"{r.get('species_macro_f1', 0.0):.4f} | {r.get('concept_macro_f1', 0.0):.4f} | "
            f"{r.get('concept_micro_f1', 0.0):.4f} | {r.get('total_loss', 0.0):.4f} |\n"
        )

    md_content += """
---

## Model Selection Guidelines:
- Primary selection criterion is **Bird Species Top-1 Accuracy** on the locked validation set.
- Concept prediction quality is evaluated as an auxiliary interpretability asset.
- Calibration images (600) and official test images (5,794) are strictly held out from model selection.
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "csv": csv_path,
        "json": json_path,
        "markdown": summary_path,
    }

