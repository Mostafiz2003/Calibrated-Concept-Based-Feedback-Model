"""
Standardized Image Preprocessing and Caching Engine for CUB-200-2011
Calibrated Concept-Based Feedback Model (CSE 4385)

Implements:
- Standardized aspect-ratio-preserving resize + crop policy
  (Resizes shorter side to 256, then crops to 224x224 to prevent anatomical distortion)
- Deterministic evaluation pipeline (Validation, Calibration, Test)
- Augmentation pipeline strictly for training (Random crop, Horizontal flip, mild Color Jitter)
- ImageNet channel normalization ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- Cached preprocessed dataset manager with metadata and checksum verification
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PreprocessingConfig:
    """Encapsulates all preprocessing parameters with strict versioning."""
    VERSION: str = "1.0.0"

    def __init__(
        self,
        input_size: int = 224,
        resize_size: int = 256,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        color_jitter_brightness: float = 0.1,
        color_jitter_contrast: float = 0.1,
        horizontal_flip_prob: float = 0.5,
    ):
        self.input_size = input_size
        self.resize_size = resize_size
        self.mean = mean or list(IMAGENET_MEAN)
        self.std = std or list(IMAGENET_STD)
        self.color_jitter_brightness = color_jitter_brightness
        self.color_jitter_contrast = color_jitter_contrast
        self.horizontal_flip_prob = horizontal_flip_prob

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "input_size": self.input_size,
            "resize_size": self.resize_size,
            "mean": self.mean,
            "std": self.std,
            "color_jitter_brightness": self.color_jitter_brightness,
            "color_jitter_contrast": self.color_jitter_contrast,
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "aspect_ratio_policy": "resize_shorter_side_then_crop",
        }


def get_standardized_transforms(
    split: str,
    config: Optional[PreprocessingConfig] = None,
) -> transforms.Compose:
    """
    Constructs torchvision transforms pipeline for the specified split.

    Aspect-ratio handling:
      Resizes the shorter dimension to resize_size (default: 256) maintaining aspect ratio,
      then extracts a 224x224 crop. This avoids unnatural anatomical stretching of bird bodies.

    Args:
        split: 'train', 'val', 'calib', or 'test'.
        config: Optional PreprocessingConfig instance.

    Returns:
        torchvision transforms.Compose pipeline.
    """
    if config is None:
        config = PreprocessingConfig()

    split_lower = split.lower()
    normalize = transforms.Normalize(mean=config.mean, std=config.std)

    if split_lower == "train":
        # Data augmentation applied ONLY to the training split
        return transforms.Compose([
            transforms.Resize(config.resize_size),
            transforms.RandomCrop(config.input_size),
            transforms.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            transforms.ColorJitter(
                brightness=config.color_jitter_brightness,
                contrast=config.color_jitter_contrast,
            ),
            transforms.ToTensor(),
            normalize,
        ])
    elif split_lower in ("val", "calib", "test"):
        # Strictly deterministic evaluation transforms (no random perturbations)
        return transforms.Compose([
            transforms.Resize(config.resize_size),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected one of: 'train', 'val', 'calib', 'test'."
        )


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 checksum of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class ProcessedDatasetManager:
    """
    Manages structured processed cache under data/processed/CUB_200_2011/.
    Supports on-the-fly loading as well as generating cached metadata manifests.
    """

    def __init__(
        self,
        processed_dir: Union[str, Path] = "data/processed/CUB_200_2011",
        config: Optional[PreprocessingConfig] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.config = config or PreprocessingConfig()
        self.metadata_file = self.processed_dir / "processing_metadata.json"

    def initialize_storage(self) -> None:
        """Creates processed storage directory structure."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_processing_metadata(
        self,
        total_images: int,
        verified_images: int,
        split_counts: Dict[str, int],
        dataset_checksums: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Saves preprocessing configuration, dataset statistics, and checksums to metadata JSON.
        """
        self.initialize_storage()
        payload = {
            "preprocessing_config": self.config.to_dict(),
            "total_images": total_images,
            "verified_images": verified_images,
            "split_counts": split_counts,
            "checksums": dataset_checksums or {},
        }
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved preprocessing metadata to {self.metadata_file}")
        return self.metadata_file

    def is_initialized(self) -> bool:
        """Checks if processed storage has been initialized with metadata."""
        return self.metadata_file.exists()

