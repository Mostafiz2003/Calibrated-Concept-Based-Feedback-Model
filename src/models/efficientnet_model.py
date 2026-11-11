"""
Dual-Head EfficientNet-B2 Architecture for Bird Recognition and Concept Prediction
Calibrated Concept-Based Feedback Model (CSE 4385)

Experiment ID: E5
Architecture:
- Backbone: ImageNet-pretrained EfficientNet-B2
- Feature extractor: AdaptiveAvgPool2d -> 1408-dimensional feature embedding
- Primary Head: Linear(1408, 200) -> 200 bird species logits
- Auxiliary Head: Linear(1408, 112) -> 112 concept logits
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.recognition_model import DualHeadOutput

logger = logging.getLogger(__name__)


class DualHeadEfficientNetB2(nn.Module):
    """
    EfficientNet-B2 with dual prediction heads for species recognition and concept prediction.
    """

    def __init__(
        self,
        num_classes: int = 200,
        num_concepts: int = 112,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.feature_dim = 1408  # EfficientNet-B2 penultimate channel count
        self.model_type = "efficientnet_b2"
        self.experiment_id = "E5"

        # Load backbone
        if weights_path is not None:
            base_model = models.efficientnet_b2(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            base_model.load_state_dict(state_dict)
        elif pretrained:
            try:
                base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            except Exception as e:
                logger.warning(f"Could not load official weights for EfficientNet-B2: {e}. Using random init.")
                base_model = models.efficientnet_b2(weights=None)
        else:
            base_model = models.efficientnet_b2(weights=None)

        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Prediction Heads
        self.dropout = nn.Dropout(p=dropout_rate)
        self.species_head = nn.Linear(self.feature_dim, self.num_classes)
        self.concept_head = nn.Linear(self.feature_dim, self.num_concepts)

        # Initialize heads
        nn.init.xavier_normal_(self.species_head.weight)
        nn.init.constant_(self.species_head.bias, 0.0)
        nn.init.xavier_normal_(self.concept_head.weight)
        nn.init.constant_(self.concept_head.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 1408-dimensional pooled features."""
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> DualHeadOutput:
        """
        Forward pass returning species and concept logits.

        Args:
            x: Input image tensor [B, 3, 224, 224]

        Returns:
            DualHeadOutput(concept_logits=[B, 112], species_logits=[B, 200], features=[B, 1408])
        """
        features = self.extract_features(x)
        features_drop = self.dropout(features)

        species_logits = self.species_head(features_drop)
        concept_logits = self.concept_head(features_drop)

        return DualHeadOutput(
            concept_logits=concept_logits,
            species_logits=species_logits,
            features=features,
        )

    def freeze_backbone(self) -> None:
        """Freezes feature extractor layers for linear probing."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

