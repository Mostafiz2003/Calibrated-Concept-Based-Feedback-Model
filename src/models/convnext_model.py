"""
Dual-Head ConvNeXt-Tiny Architecture for Bird Recognition and Concept Prediction
Calibrated Concept-Based Feedback Model (CSE 4385)

Experiment ID: E6
Architecture:
- Backbone: ImageNet-pretrained ConvNeXt-Tiny
- Feature extractor: ConvNeXt stages + LayerNorm + AdaptiveAvgPool2d -> 768-dimensional embedding
- Primary Head: Linear(768, 200) -> 200 bird species logits
- Auxiliary Head: Linear(768, 112) -> 112 concept logits
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.recognition_model import DualHeadOutput

logger = logging.getLogger(__name__)


class DualHeadConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny with dual prediction heads for species recognition and concept prediction.
    """

    def __init__(
        self,
        num_classes: int = 200,
        num_concepts: int = 112,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_concepts = int(num_concepts)
        self.feature_dim = 768  # ConvNeXt-Tiny channel dimension
        self.model_type = "convnext_tiny"
        self.experiment_id = "E6"

        # Load backbone
        if weights_path is not None:
            base_model = models.convnext_tiny(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            base_model.load_state_dict(state_dict)
        elif pretrained:
            try:
                base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            except Exception as e:
                logger.warning(f"Could not load official weights for ConvNeXt-Tiny: {e}. Using random init.")
                base_model = models.convnext_tiny(weights=None)
        else:
            base_model = models.convnext_tiny(weights=None)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # ConvNeXt classifier stage 0 is a LayerNorm(768)
        if hasattr(base_model.classifier, "__getitem__") and len(base_model.classifier) > 0:
            self.norm = base_model.classifier[0]
        else:
            self.norm = nn.LayerNorm(self.feature_dim, eps=1e-6)

        # Prediction Heads
        self.species_head = nn.Linear(self.feature_dim, self.num_classes)
        self.concept_head = nn.Linear(self.feature_dim, self.num_concepts)

        # Initialize heads
        nn.init.xavier_normal_(self.species_head.weight)
        nn.init.constant_(self.species_head.bias, 0.0)
        nn.init.xavier_normal_(self.concept_head.weight)
        nn.init.constant_(self.concept_head.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 768-dimensional normalized pooled features."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> DualHeadOutput:
        """
        Forward pass returning species and concept logits.

        Args:
            x: Input image tensor [B, 3, 224, 224]

        Returns:
            DualHeadOutput(concept_logits=[B, 112], species_logits=[B, 200], features=[B, 768])
        """
        features = self.extract_features(x)

        species_logits = self.species_head(features)
        concept_logits = self.concept_head(features)

        return DualHeadOutput(
            concept_logits=concept_logits,
            species_logits=species_logits,
            features=features,
        )

    def freeze_backbone(self) -> None:
        """Freezes feature extractor layers."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.norm.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreezes all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_counts(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

