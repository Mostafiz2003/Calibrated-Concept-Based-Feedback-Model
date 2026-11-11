"""
Concept Predictor Model for Calibrated Concept-Based Feedback Model (CSE 4385)
Architecture: Pretrained ResNet Backbone + Linear Concept Prediction Head

Maps bird images x in R^(3 x 224 x 224) to 112 concept logits z in R^112.
Supports modular backbone selection (ResNet-18, ResNet-34, ResNet-50) and
explicit handling of pretrained ImageNet weights.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import load_config


class PretrainedWeightsLoadError(RuntimeError):
    """Raised when pretrained weights are requested but cannot be downloaded or loaded."""
    pass


class ConceptPredictor(nn.Module):
    """
    Concept Predictor Network:
      Backbone: Configurable ResNet feature extractor h(x) in R^D
      Concept Head: Linear projection layer W * h(x) + b in R^112

    Args:
        backbone_name: Model backbone identifier ('resnet18', 'resnet34', 'resnet50').
        num_concepts: Number of concept prediction targets (default: 112).
        pretrained: If True, attempts to load official pretrained ImageNet weights.
                    If False, initializes backbone randomly.
                    Never silently falls back to random weights if pretrained=True fails.
        weights_path: Optional explicit local path to custom weights checkpoint.
    """

    SUPPORTED_BACKBONES = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, 512),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    }

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_concepts: int = 112,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_concepts = int(num_concepts)
        self.is_pretrained = bool(pretrained)
        self.weights_path = Path(weights_path) if weights_path is not None else None

        if self.backbone_name not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Supported backbones are: {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        _, default_weights, feature_dim = self.SUPPORTED_BACKBONES[self.backbone_name]
        arch_fn = getattr(models, self.backbone_name)
        self.feature_dim = feature_dim

        # Load Backbone Architecture & Weights
        if self.weights_path is not None:
            if not self.weights_path.exists():
                raise FileNotFoundError(f"Custom weights file not found: {self.weights_path}")
            try:
                base_model = arch_fn(weights=None)
                state_dict = torch.load(self.weights_path, map_location="cpu", weights_only=True)
                base_model.load_state_dict(state_dict)
                self.pretrained_source = str(self.weights_path)
            except Exception as e:
                raise PretrainedWeightsLoadError(
                    f"Failed to load custom weights from {self.weights_path}: {e}"
                )
        elif self.is_pretrained:
            try:
                base_model = arch_fn(weights=default_weights)
                self.pretrained_source = f"torchvision:{default_weights.name}"
            except Exception as e:
                # Do NOT silently fall back to random weights when pretrained=True was requested!
                raise PretrainedWeightsLoadError(
                    f"Failed to download/load official pretrained weights for '{self.backbone_name}': {e}.\n"
                    "If working in an isolated or offline environment, configure 'pretrained: false' "
                    "in configs/default.yaml or pass pretrained=False explicitly to use random initialization, "
                    "or provide a local weights file via weights_path."
                )
        else:
            base_model = arch_fn(weights=None)
            self.pretrained_source = "random_initialization"

        # Replace classification fc layer with Identity so backbone outputs feature vector R^D
        base_model.fc = nn.Identity()
        self.backbone = base_model

        # Linear Concept Prediction Head: R^D -> R^112 (produces unnormalized logits)
        self.concept_head = nn.Linear(self.feature_dim, self.num_concepts)

        # Initialize concept head weights using standard Xavier normal initialization
        nn.init.xavier_normal_(self.concept_head.weight)
        nn.init.constant_(self.concept_head.bias, 0.0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts pooled feature representation h(x) before the concept head.

        Args:
            x: Input image tensor of shape [batch_size, 3, 224, 224].

        Returns:
            features: Feature representation tensor of shape [batch_size, feature_dim].
        """
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized concept logits.

        Args:
            x: Input image tensor of shape [batch_size, 3, 224, 224].

        Returns:
            logits: Concept prediction logits of shape [batch_size, num_concepts].
                    Note: Unnormalized logits are returned for stable training with
                    torch.nn.BCEWithLogitsLoss.
        """
        features = self.extract_features(x)
        logits = self.concept_head(features)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning calibrated/normalized concept probabilities in [0.0, 1.0].
        Applies sigmoid activation to unnormalized concept logits.

        Args:
            x: Input image tensor of shape [batch_size, 3, 224, 224].

        Returns:
            probabilities: Concept probabilities of shape [batch_size, num_concepts].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @staticmethod
    def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
        """
        Utility converting precomputed logits into probabilities via sigmoid.

        Args:
            logits: Logits tensor of shape [..., num_concepts].

        Returns:
            probabilities in range [0.0, 1.0].
        """
        return torch.sigmoid(logits)


def build_concept_predictor(
    config: Optional[dict] = None,
    backbone_name: Optional[str] = None,
    num_concepts: Optional[int] = None,
    pretrained: Optional[bool] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> ConceptPredictor:
    """
    Factory function to construct a ConceptPredictor from configuration or keyword arguments.

    Args:
        config: Loaded project configuration dict (optional).
        backbone_name: Override backbone model name.
        num_concepts: Override number of concepts.
        pretrained: Override pretrained flag.
        weights_path: Override weights file path.

    Returns:
        Configured ConceptPredictor instance.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    resolved_backbone = backbone_name or model_cfg.get("fallback_backbone") or model_cfg.get("backbone", "resnet18")
    resolved_num_concepts = num_concepts or data_cfg.get("num_concepts", 112)
    resolved_pretrained = pretrained if pretrained is not None else model_cfg.get("pretrained", True)

    return ConceptPredictor(
        backbone_name=resolved_backbone,
        num_concepts=resolved_num_concepts,
        pretrained=resolved_pretrained,
        weights_path=weights_path,
    )
