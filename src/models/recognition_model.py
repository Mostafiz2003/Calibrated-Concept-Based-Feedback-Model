"""
Recognition-Focused Dual-Head Architecture for CUB-200-2011 (CSE 4385)
Architecture: Pretrained ResNet-50 + Optional CBAM Attention + Optional FPN Multi-Scale Aggregation + Dual Prediction Heads

Produces:
  1. 112 concept logits (binary attribute prediction via BCEWithLogitsLoss)
  2. 200 bird species logits (species classification via CrossEntropyLoss)

Supports Experiments:
  E1: ResNet-18 concept baseline (existing ConceptPredictor)
  E2: ResNet-50 with Dual Heads (use_cbam=False, use_fpn=False)
  E3: ResNet-50 + CBAM with Dual Heads (use_cbam=True, use_fpn=False)
  E4: ResNet-50 + CBAM + FPN with Dual Heads (use_cbam=True, use_fpn=True) [Proposed Model]
"""

from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import load_config
from src.models.attention import CBAM
from src.models.concept_model import PretrainedWeightsLoadError
from src.models.fpn import FeaturePyramidNetwork, MultiScaleFeatureAggregation


class DualHeadOutput:
    """
    Output container for DualHeadRecognitionModel.

    Supports:
      1. Named attribute access: out.concept_logits, out.species_logits
      2. Tuple unpacking: c_logits, s_logits = out
      3. Index access: out[0], out[1]
      4. Dict-style access: out["concept_logits"], out["species_logits"]
    """

    def __init__(
        self,
        concept_logits: torch.Tensor,
        species_logits: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ):
        self.concept_logits = concept_logits
        self.species_logits = species_logits
        self.features = features

    def __iter__(self):
        yield self.concept_logits
        yield self.species_logits

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: Union[int, str]) -> torch.Tensor:
        if isinstance(item, int):
            if item == 0:
                return self.concept_logits
            elif item == 1:
                return self.species_logits
            else:
                raise IndexError(f"Index {item} out of range for DualHeadOutput (length 2)")
        elif isinstance(item, str):
            if item == "concept_logits":
                return self.concept_logits
            elif item == "species_logits":
                return self.species_logits
            else:
                raise KeyError(
                    f"Key '{item}' not found in DualHeadOutput. Expected 'concept_logits' or 'species_logits'."
                )
        else:
            raise TypeError(f"DualHeadOutput indices must be integers or strings, not {type(item).__name__}")

    def __repr__(self) -> str:
        return (
            f"DualHeadOutput(concept_logits={list(self.concept_logits.shape)}, "
            f"species_logits={list(self.species_logits.shape)})"
        )


class DualHeadRecognitionModel(nn.Module):
    """
    Recognition-Focused Dual-Head Model.

    Combines a ResNet-50 backbone with modular CBAM attention and FPN multi-scale
    aggregation, feeding dual prediction heads for explainable concepts and accurate species recognition.

    Args:
        num_concepts: Number of concept prediction targets (default: 112).
        num_classes: Number of bird species classes (default: 200).
        use_cbam: Whether to apply CBAM attention to backbone stage features (default: True).
        use_fpn: Whether to construct multi-scale feature pyramid and aggregation (default: True).
        pretrained: Whether to load official pretrained ImageNet weights for ResNet-50.
        weights_path: Optional explicit local path to custom weights checkpoint.
        fpn_channels: Uniform channel depth for FPN pyramid levels (default: 256).
        feature_dim: Dimension of the aggregated feature vector before dual heads (default: 2048).
        experiment_mode: Optional shortcut string ("E2", "E3", "E4") overriding use_cbam / use_fpn.
    """

    def __init__(
        self,
        num_concepts: int = 112,
        num_classes: int = 200,
        use_cbam: bool = True,
        use_fpn: bool = True,
        pretrained: bool = True,
        weights_path: Optional[Union[str, Path]] = None,
        fpn_channels: int = 256,
        feature_dim: int = 2048,
        experiment_mode: Optional[str] = None,
    ):
        super().__init__()
        self.num_concepts = int(num_concepts)
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.fpn_channels = fpn_channels
        self.feature_dim = feature_dim

        # Configure experiment mode if provided
        if experiment_mode is not None:
            mode_upper = experiment_mode.upper()
            if mode_upper == "E2":
                self.use_cbam = False
                self.use_fpn = False
            elif mode_upper == "E3":
                self.use_cbam = True
                self.use_fpn = False
            elif mode_upper == "E4":
                self.use_cbam = True
                self.use_fpn = True
            else:
                raise ValueError(
                    f"Unknown experiment_mode '{experiment_mode}'. Expected 'E2', 'E3', or 'E4'."
                )
        else:
            self.use_cbam = bool(use_cbam)
            self.use_fpn = bool(use_fpn)

        # 1. Initialize ResNet-50 Backbone
        self._init_backbone()

        # 2. Initialize CBAM Modules (if enabled)
        # ResNet-50 stage channel depths: layer1=256, layer2=512, layer3=1024, layer4=2048
        self.stage_channels = [256, 512, 1024, 2048]
        if self.use_cbam:
            self.cbam_modules = nn.ModuleList([
                CBAM(in_channels=ch, reduction_ratio=16, kernel_size=7, use_residual=True)
                for ch in self.stage_channels
            ])
        else:
            self.cbam_modules = None

        # 3. Initialize FPN and Multi-Scale Feature Aggregation (if enabled)
        if self.use_fpn:
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=self.stage_channels,
                out_channels=self.fpn_channels,
            )
            self.aggregation = MultiScaleFeatureAggregation(
                num_levels=len(self.stage_channels),
                in_channels=self.fpn_channels,
                out_dim=self.feature_dim,
            )
        else:
            self.fpn = None
            self.aggregation = None
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Initialize Dual Prediction Heads
        # Concept prediction head: R^feature_dim -> R^112 (produces unnormalized logits)
        self.concept_head = nn.Linear(self.feature_dim, self.num_concepts)

        # Species classification head: R^feature_dim -> R^200 (produces unnormalized logits)
        self.species_head = nn.Linear(self.feature_dim, self.num_classes)

        # Initialize heads
        nn.init.xavier_normal_(self.concept_head.weight)
        nn.init.constant_(self.concept_head.bias, 0.0)
        nn.init.xavier_normal_(self.species_head.weight)
        nn.init.constant_(self.species_head.bias, 0.0)

    def _init_backbone(self):
        """Builds and loads pretrained ResNet-50 backbone components."""
        arch_fn = models.resnet50
        default_weights = models.ResNet50_Weights.DEFAULT

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
        elif self.pretrained:
            try:
                base_model = arch_fn(weights=default_weights)
                self.pretrained_source = f"torchvision:{default_weights.name}"
            except Exception as e:
                raise PretrainedWeightsLoadError(
                    f"Failed to load pretrained weights for 'resnet50': {e}.\n"
                    "If working in an offline environment, pass pretrained=False explicitly "
                    "or provide a local weights path via weights_path."
                )
        else:
            base_model = arch_fn(weights=None)
            self.pretrained_source = "random_initialization"

        # Separate ResNet-50 into initial stem and 4 sequential stages
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1  # C2: [B, 256, 56, 56]
        self.layer2 = base_model.layer2  # C3: [B, 512, 28, 28]
        self.layer3 = base_model.layer3  # C4: [B, 1024, 14, 14]
        self.layer4 = base_model.layer4  # C5: [B, 2048, 7, 7]

    @property
    def experiment_name(self) -> str:
        """Returns the canonical experiment code (E2, E3, or E4)."""
        if self.use_cbam and self.use_fpn:
            return "E4 (ResNet-50 + CBAM + FPN)"
        elif self.use_cbam and not self.use_fpn:
            return "E3 (ResNet-50 + CBAM)"
        else:
            return "E2 (ResNet-50 Dual-Head)"

    def extract_stage_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extracts bottom-up multi-stage features [C2, C3, C4, C5] from ResNet-50.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            List of 4 stage feature maps [C2, C3, C4, C5].
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Bottom-up stages
        c2 = self.layer1(x)   # [B, 256, 56, 56]
        c3 = self.layer2(c2)  # [B, 512, 28, 28]
        c4 = self.layer3(c3)  # [B, 1024, 14, 14]
        c5 = self.layer4(c4)  # [B, 2048, 7, 7]

        return [c2, c3, c4, c5]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the unified global feature representation f in R^feature_dim.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            Global feature representation of shape [B, feature_dim].
        """
        # 1. Extract stage feature maps [C2, C3, C4, C5]
        stage_features = self.extract_stage_features(x)

        # 2. Apply CBAM attention if enabled (E3 / E4)
        if self.use_cbam:
            refined_features = [
                cbam(f) for cbam, f in zip(self.cbam_modules, stage_features)
            ]
        else:
            refined_features = stage_features

        # 3. Apply FPN and Multi-Scale Aggregation if enabled (E4)
        if self.use_fpn:
            pyramid_features = self.fpn(refined_features)
            global_feature = self.aggregation(pyramid_features)
        else:
            # E2 / E3: Global average pool the top stage feature C5
            c5 = refined_features[-1]
            global_feature = self.gap(c5).flatten(1)

        return global_feature

    def forward(self, x: torch.Tensor) -> DualHeadOutput:
        """
        Forward pass producing concept and species prediction logits.

        Args:
            x: Input image tensor of shape [B, 3, 224, 224].

        Returns:
            DualHeadOutput namedtuple with:
              - concept_logits: FloatTensor of shape [B, 112]
              - species_logits: FloatTensor of shape [B, 200]
        """
        features = self.extract_features(x)
        concept_logits = self.concept_head(features)
        species_logits = self.species_head(features)
        return DualHeadOutput(
            concept_logits=concept_logits,
            species_logits=species_logits,
            features=features,
        )

    def predict_concept_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes calibrated/normalized concept probabilities in [0.0, 1.0] via sigmoid."""
        out = self.forward(x)
        return torch.sigmoid(out.concept_logits)

    def predict_species_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes species probability distribution in [0.0, 1.0] via softmax."""
        out = self.forward(x)
        return F.softmax(out.species_logits, dim=-1)

    def predict_species(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts 0-indexed species class index [0..199] via argmax."""
        out = self.forward(x)
        return torch.argmax(out.species_logits, dim=-1)


class JointRecognitionLoss(nn.Module):
    """
    Weighted Joint Loss Function for Concept Prediction and Species Recognition.

    Computes:
        total_loss = lambda_concept * concept_loss + lambda_species * species_loss
    where:
        concept_loss = BCEWithLogitsLoss(concept_logits, concept_targets)
        species_loss = CrossEntropyLoss(species_logits, species_targets)

    Args:
        lambda_concept: Weight multiplier for concept binary loss (default: 1.0).
        lambda_species: Weight multiplier for species classification loss (default: 1.0).
    """

    def __init__(self, lambda_concept: float = 1.0, lambda_species: float = 1.0):
        super().__init__()
        self.lambda_concept = float(lambda_concept)
        self.lambda_species = float(lambda_species)

        self.concept_criterion = nn.BCEWithLogitsLoss()
        self.species_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        concept_logits: torch.Tensor,
        species_logits: torch.Tensor,
        concept_targets: torch.Tensor,
        species_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            concept_logits: Unnormalized concept logits of shape [B, 112].
            species_logits: Unnormalized species logits of shape [B, 200].
            concept_targets: Binary concept labels of shape [B, 112] in {0.0, 1.0}.
            species_targets: 0-indexed species labels of shape [B] in [0..199].

        Returns:
            Dictionary with:
              - 'total_loss': Weighted scalar loss for backpropagation
              - 'concept_loss': Raw BCE concept loss scalar
              - 'species_loss': Raw CE species loss scalar
        """
        c_loss = self.concept_criterion(concept_logits, concept_targets)
        s_loss = self.species_criterion(species_logits, species_targets)
        total_loss = self.lambda_concept * c_loss + self.lambda_species * s_loss

        return {
            "total_loss": total_loss,
            "concept_loss": c_loss,
            "species_loss": s_loss,
        }


def build_recognition_model(
    config: Optional[dict] = None,
    experiment_mode: Optional[str] = "E4",
    use_cbam: Optional[bool] = None,
    use_fpn: Optional[bool] = None,
    num_concepts: Optional[int] = None,
    num_classes: Optional[int] = None,
    pretrained: Optional[bool] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> DualHeadRecognitionModel:
    """
    Factory function to instantiate DualHeadRecognitionModel from config or parameters.

    Args:
        config: Loaded project configuration dict (optional).
        experiment_mode: One of 'E2', 'E3', 'E4' (default: 'E4').
        use_cbam: Explicit CBAM override.
        use_fpn: Explicit FPN override.
        num_concepts: Concept target dimension (default: 112).
        num_classes: Species classes dimension (default: 200).
        pretrained: Whether to load ImageNet pretrained weights (default: True).
        weights_path: Optional custom checkpoint path.

    Returns:
        Instantiated and configured DualHeadRecognitionModel.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    resolved_num_concepts = num_concepts or data_cfg.get("num_concepts", 112)
    resolved_num_classes = num_classes or data_cfg.get("num_classes", 200)
    resolved_pretrained = pretrained if pretrained is not None else model_cfg.get("pretrained", True)

    return DualHeadRecognitionModel(
        num_concepts=resolved_num_concepts,
        num_classes=resolved_num_classes,
        use_cbam=use_cbam if use_cbam is not None else (experiment_mode in ("E3", "E4")),
        use_fpn=use_fpn if use_fpn is not None else (experiment_mode == "E4"),
        pretrained=resolved_pretrained,
        weights_path=weights_path,
        experiment_mode=experiment_mode if (use_cbam is None and use_fpn is None) else None,
    )
