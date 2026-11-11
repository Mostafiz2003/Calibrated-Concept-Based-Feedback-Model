"""
Multi-Task Loss Formulations for Bird Species Recognition and Concept Prediction
Calibrated Concept-Based Feedback Model (CSE 4385)

Joint Objective:
  L_total = L_species + lambda_concept * L_concept

Where:
  - L_species: CrossEntropyLoss over 200 bird species classes
  - L_concept: BCEWithLogitsLoss over 112 visual concept attributes
  - lambda_concept: Multi-task trade-off weight (default: 0.2)
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.recognition_model import DualHeadOutput


class MultiTaskRecognitionLoss(nn.Module):
    """
    Joint loss function combining primary species classification and auxiliary concept prediction.

    Args:
        lambda_concept: Weighting factor for auxiliary concept loss (default: 0.2).
        label_smoothing: Optional label smoothing for cross-entropy (default: 0.0).
        pos_weight: Optional positive weight tensor for BCEWithLogitsLoss [112].
    """

    def __init__(
        self,
        lambda_concept: float = 0.2,
        label_smoothing: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_concept = float(lambda_concept)
        self.species_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.concept_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        predictions: Union[DualHeadOutput, Tuple[torch.Tensor, torch.Tensor]],
        species_targets: torch.Tensor,
        concept_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates joint multi-task loss.

        Args:
            predictions: DualHeadOutput or tuple of (concept_logits, species_logits).
            species_targets: 0-indexed species ground truth tensor [B] (long).
            concept_targets: Binary concept ground truth tensor [B, 112] (float32).

        Returns:
            Dict containing:
              - 'total_loss': scalar weighted total loss
              - 'species_loss': scalar cross-entropy loss
              - 'concept_loss': scalar binary cross-entropy loss
        """
        if isinstance(predictions, DualHeadOutput):
            species_logits = predictions.species_logits
            concept_logits = predictions.concept_logits
        else:
            concept_logits, species_logits = predictions[0], predictions[1]

        species_loss = self.species_criterion(species_logits, species_targets)
        concept_loss = self.concept_criterion(concept_logits, concept_targets)

        total_loss = species_loss + (self.lambda_concept * concept_loss)

        return {
            "total_loss": total_loss,
            "species_loss": species_loss,
            "concept_loss": concept_loss,
        }
