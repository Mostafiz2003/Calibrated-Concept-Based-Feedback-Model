"""
Neural Network Architectures for Calibrated Bird Recognition (CSE 4385)
"""

from src.models.attention import CBAM, ChannelAttention, SpatialAttention
from src.models.concept_model import (
    ConceptPredictor,
    PretrainedWeightsLoadError,
    build_concept_predictor,
)
from src.models.convnext_model import DualHeadConvNeXtTiny
from src.models.efficientnet_model import DualHeadEfficientNetB2
from src.models.factory import SUPPORTED_ARCHITECTURES, build_recognition_model
from src.models.fpn import FeaturePyramidNetwork, MultiScaleFeatureAggregation
from src.models.recognition_model import (
    DualHeadOutput,
    DualHeadRecognitionModel,
    JointRecognitionLoss,
)

__all__ = [
    # Attention & Multiscale modules
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "FeaturePyramidNetwork",
    "MultiScaleFeatureAggregation",
    # Outputs & Losses
    "DualHeadOutput",
    "JointRecognitionLoss",
    # Architectures
    "ConceptPredictor",
    "DualHeadRecognitionModel",
    "DualHeadEfficientNetB2",
    "DualHeadConvNeXtTiny",
    # Factories
    "build_concept_predictor",
    "build_recognition_model",
    "SUPPORTED_ARCHITECTURES",
    "PretrainedWeightsLoadError",
]
