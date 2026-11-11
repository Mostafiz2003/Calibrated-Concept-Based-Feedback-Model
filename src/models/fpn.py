"""
Feature Pyramid Network (FPN) and Multi-Scale Feature Aggregation
Calibrated Concept-Based Feedback Model (CSE 4385)

Reference:
  Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017).
  Feature Pyramid Networks for Object Detection.
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2117-2125.

Provides:
  - FeaturePyramidNetwork: Top-down architecture with lateral connections building multi-scale
    feature representations from bottom-up ResNet feature maps (C2, C3, C4, C5).
  - MultiScaleFeatureAggregation: Aggregates multi-scale pyramid levels (P2..P5) into a unified
    high-dimensional feature representation for image-level concept and species recognition.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN).

    Constructs multi-scale feature maps {P2, P3, P4, P5} from backbone stage features
    {C2, C3, C4, C5} using 1x1 lateral convolutions, top-down upsampling, and 3x3 smoothing convolutions.

    Args:
        in_channels_list: Channel counts of the incoming feature maps (default for ResNet-50: [256, 512, 1024, 2048]).
        out_channels: Uniform channel depth of the resulting pyramid levels (default: 256).
    """

    def __init__(
        self,
        in_channels_list: List[int] = [256, 512, 1024, 2048],
        out_channels: int = 256,
    ):
        super().__init__()
        self.in_channels_list = list(in_channels_list)
        self.out_channels = out_channels

        # 1x1 Lateral convolutions to map variable backbone channels to uniform out_channels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False)
            for in_c in self.in_channels_list
        ])

        # 3x3 Smoothing convolutions to eliminate aliasing artifacts from upsampling
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in self.in_channels_list
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass constructing top-down feature pyramid.

        Args:
            features: List of bottom-up feature maps [C2, C3, C4, C5].

        Returns:
            List of multi-scale pyramid feature maps [P2, P3, P4, P5].
        """
        if len(features) != len(self.in_channels_list):
            raise ValueError(
                f"Expected {len(self.in_channels_list)} input feature maps, got {len(features)}"
            )

        # 1. Apply lateral 1x1 convolutions
        laterals = [
            conv(f) for conv, f in zip(self.lateral_convs, features)
        ]

        # 2. Build top-down pathway
        # Start from highest-level feature map (lowest spatial resolution)
        pyramid_levels = [None] * len(laterals)
        pyramid_levels[-1] = laterals[-1]

        for i in range(len(laterals) - 2, -1, -1):
            target_size = laterals[i].shape[-2:]
            upsampled = F.interpolate(
                pyramid_levels[i + 1],
                size=target_size,
                mode="nearest",
            )
            pyramid_levels[i] = laterals[i] + upsampled

        # 3. Apply 3x3 smoothing convolutions
        outputs = [
            smooth(p) for smooth, p in zip(self.smooth_convs, pyramid_levels)
        ]

        return outputs


class MultiScaleFeatureAggregation(nn.Module):
    """
    Multi-Scale Feature Aggregation Module.

    Aggregates multi-scale pyramid levels (P2, P3, P4, P5) into a unified 1D global feature
    representation. Performs global average pooling on each pyramid level, concatenates the
    pooled vectors, and projects through a fusion layer with LayerNorm and Dropout.

    Args:
        num_levels: Number of multi-scale pyramid levels (default: 4 for [P2, P3, P4, P5]).
        in_channels: Channel depth of each pyramid level (default: 256).
        out_dim: Output fused feature dimension (default: 2048, matching ResNet-50 feature dimension).
        dropout: Dropout probability in fusion block (default: 0.1).
    """

    def __init__(
        self,
        num_levels: int = 4,
        in_channels: int = 256,
        out_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.in_channels = in_channels
        self.out_dim = out_dim
        concat_dim = num_levels * in_channels

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        nn.init.xavier_normal_(self.fusion[0].weight)
        nn.init.constant_(self.fusion[0].bias, 0.0)

    def forward(self, pyramid_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            pyramid_features: List of pyramid level tensors [P2, P3, P4, P5].

        Returns:
            Fused feature vector of shape [B, out_dim].
        """
        pooled_levels = [
            self.gap(p).flatten(1) for p in pyramid_features
        ]
        concat_features = torch.cat(pooled_levels, dim=1)
        fused = self.fusion(concat_features)
        return fused

