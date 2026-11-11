"""
Unified Model Factory for Bird Species Recognition & Concept Bottlenecks
Calibrated Concept-Based Feedback Model (CSE 4385)

Supports 5 Primary Multi-Task Dual-Head Architectures:
  1. E2: ResNet-50 Dual-Head Baseline
  2. E3: ResNet-50 + CBAM Dual-Head
  3. E4: ResNet-50 + CBAM + FPN Multiscale Dual-Head
  4. E5: EfficientNet-B2 Dual-Head
  5. E6: ConvNeXt-Tiny Dual-Head
And preserves the single-head baseline:
  - E1: ResNet-18 Concept Predictor
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.models.concept_model import ConceptPredictor, build_concept_predictor
from src.models.convnext_model import DualHeadConvNeXtTiny
from src.models.efficientnet_model import DualHeadEfficientNetB2
from src.models.recognition_model import DualHeadOutput, DualHeadRecognitionModel

logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = {
    "resnet50": "E2",
    "resnet50_cbam": "E3",
    "resnet50_cbam_fpn": "E4",
    "efficientnet_b2": "E5",
    "convnext_tiny": "E6",
    # Legacy / Baseline
    "resnet18_concept_baseline": "E1",
}


def build_recognition_model(
    model_name: str,
    num_classes: int = 200,
    num_concepts: int = 112,
    pretrained: bool = True,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Factory creating a dual-head bird recognition model.

    Args:
        model_name: Identifier ('resnet50', 'resnet50_cbam', 'resnet50_cbam_fpn',
                    'efficientnet_b2', 'convnext_tiny', or experiment ID 'E2'..'E6').
        num_classes: Bird species target count (default: 200).
        num_concepts: Auxiliary concept target count (default: 112).
        pretrained: If True, loads ImageNet pretrained weights.
        weights_path: Optional explicit local path to custom checkpoint.
        config: Optional configuration dict.

    Returns:
        Configured nn.Module instance with dual prediction heads.
    """
    name = model_name.lower().strip()

    # Map Experiment ID to canonical model name
    id_map = {
        "e1": "resnet18_concept_baseline",
        "e2": "resnet50",
        "e3": "resnet50_cbam",
        "e4": "resnet50_cbam_fpn",
        "e5": "efficientnet_b2",
        "e6": "convnext_tiny",
    }
    if name in id_map:
        name = id_map[name]

    if name == "resnet50":
        return DualHeadRecognitionModel(
            num_classes=num_classes,
            num_concepts=num_concepts,
            pretrained=pretrained,
            use_cbam=False,
            use_fpn=False,
            weights_path=weights_path,
        )

    elif name == "resnet50_cbam":
        return DualHeadRecognitionModel(
            num_classes=num_classes,
            num_concepts=num_concepts,
            pretrained=pretrained,
            use_cbam=True,
            use_fpn=False,
            weights_path=weights_path,
        )

    elif name == "resnet50_cbam_fpn":
        fpn_channels = 256
        if config and "fpn_channels" in config:
            fpn_channels = int(config["fpn_channels"])
        return DualHeadRecognitionModel(
            num_classes=num_classes,
            num_concepts=num_concepts,
            pretrained=pretrained,
            use_cbam=True,
            use_fpn=True,
            fpn_channels=fpn_channels,
            weights_path=weights_path,
        )

    elif name == "efficientnet_b2":
        return DualHeadEfficientNetB2(
            num_classes=num_classes,
            num_concepts=num_concepts,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    elif name == "convnext_tiny":
        return DualHeadConvNeXtTiny(
            num_classes=num_classes,
            num_concepts=num_concepts,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    elif name == "resnet18_concept_baseline":
        return build_concept_predictor(
            backbone_name="resnet18",
            num_concepts=num_concepts,
            pretrained=pretrained,
            weights_path=weights_path,
        )

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported options are: {list(SUPPORTED_ARCHITECTURES.keys())} "
            f"or experiment IDs {list(id_map.keys())}"
        )

