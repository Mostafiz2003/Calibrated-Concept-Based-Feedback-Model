"""
Convolutional Block Attention Module (CBAM)
Calibrated Concept-Based Feedback Model (CSE 4385)

Reference:
  Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).
  CBAM: Convolutional Block Attention Module.
  Proceedings of the European Conference on Computer Vision (ECCV), pp. 3-19.

Provides:
  - ChannelAttention: Exploits inter-channel relationships via max & avg pooling + shared MLP.
  - SpatialAttention: Exploits inter-spatial relationships via max & avg pooling + 7x7 conv.
  - CBAM: Sequentially applies ChannelAttention then SpatialAttention with optional residual.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module of CBAM.

    Applies adaptive average and max pooling along spatial dimensions, passes both
    through a shared bottleneck MLP, sums the outputs, and applies a sigmoid activation
    to produce channel attention weights M_c in [0.0, 1.0].

    Args:
        in_channels: Number of input channels C.
        reduction_ratio: Reduction ratio r for the shared MLP bottleneck (default: 16).
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        bottleneck_channels = max(1, in_channels // reduction_ratio)

        # Shared MLP implemented with 1x1 convolutions to preserve [B, C, 1, 1] tensor shape
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape [B, C, H, W].

        Returns:
            Channel attention map M_c of shape [B, C, 1, 1].
        """
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module of CBAM.

    Applies average and max pooling along the channel dimension, concatenates them
    into a 2-channel descriptor, and passes through a 7x7 convolution + sigmoid to
    produce spatial attention weights M_s in [0.0, 1.0].

    Args:
        kernel_size: Convolution kernel size (default: 7, with padding=3).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"kernel_size must be 3 or 7, got {kernel_size}")

        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape [B, C, H, W].

        Returns:
            Spatial attention map M_s of shape [B, 1, H, W].
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)

        spatial_att = self.sigmoid(self.bn(self.conv(concat)))
        return spatial_att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies Channel Attention and Spatial Attention:
        F'  = M_c(F) * F
        F'' = M_s(F') * F'
    Optionally adds a residual skip connection: F_out = F + F''

    Args:
        in_channels: Number of input channels C.
        reduction_ratio: Bottleneck reduction ratio for channel attention (default: 16).
        kernel_size: Spatial attention kernel size (default: 7).
        use_residual: If True, adds residual connection x + F'' (default: True).
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_residual = use_residual

        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map of shape [B, C, H, W].

        Returns:
            Attention-refined feature map of shape [B, C, H, W].
        """
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        if self.use_residual:
            out = x + out
        return out

