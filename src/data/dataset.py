"""
PyTorch Dataset and DataLoader Pipeline for CUB-200-2011 with 112 Filtered Concepts
Calibrated Concept-Based Feedback Model (CSE 4385)

Loads CUB-200-2011 images, species labels (0..199), and the frozen 112 binary concept
labels according to data/selected_concepts_112.json and outputs/dataset_manifest.json.

Guarantees:
  1. Preserves canonical concept index ordering (0..111).
  2. Applies data augmentations strictly to the training split.
  3. Uses deterministic evaluation transforms for validation and calibration.
  4. Strictly isolates the official test set (raises PermissionError if accessed
     without explicit authorization).
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import load_config
from src.data.concept_audit import load_concept_mapping
from src.data.metadata_parser import CUBMetadataParser, CUBMetadata

logger = logging.getLogger(__name__)

# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_cub_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """
    Returns image transformations for a given dataset split.

    Args:
        split: One of 'train', 'val', 'calib', or 'test'.
        image_size: Target square image dimension (default: 224).

    Returns:
        torchvision transforms.Compose pipeline.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize_dim = int(round(image_size * 256.0 / 224.0))  # 256 for 224

    if split == "train":
        # Data augmentation applied ONLY to the training split
        return transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif split in ("val", "calib", "test"):
        # Deterministic evaluation transforms (no random crops or flips)
        return transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(
            f"Invalid split '{split}'. Expected one of: 'train', 'val', 'calib', 'test'."
        )


def load_canonical_concept_ids(mapping_file: Path) -> List[int]:
    """
    Loads the canonical 112 CUB attribute IDs from the frozen mapping JSON.
    Guarantees strict preserved ordering of concept_index (0..111).
    """
    mapping = load_concept_mapping(mapping_file)
    concepts = mapping.get("concepts", [])
    if len(concepts) != 112:
        raise ValueError(f"Expected 112 concepts in {mapping_file}, found {len(concepts)}")

    # Sort strictly by concept_index (0..111) to guarantee ordering
    sorted_concepts = sorted(concepts, key=lambda c: c["concept_index"])
    return [c["cub_attribute_id"] for c in sorted_concepts]


def extract_or_load_concept_matrix(
    dataset_root: Path,
    mapping_file: Path,
    cache_path: Optional[Path] = None,
    total_images: int = 11788,
) -> np.ndarray:
    """
    Extracts or loads cached [total_images + 1, 112] binary concept matrix.
    Index 0 is unused so that image_id (1..total_images) directly indexes row image_id.
    """
    if cache_path is not None and cache_path.exists():
        try:
            matrix = np.load(cache_path)
            if matrix.shape == (total_images + 1, 112):
                return matrix
            logger.warning(f"Cached matrix shape {matrix.shape} mismatched; regenerating.")
        except Exception as e:
            logger.warning(f"Failed to load cached concept matrix: {e}; regenerating.")

    canonical_cub_ids = load_canonical_concept_ids(mapping_file)
    attr_to_concept_idx = {attr_id: idx for idx, attr_id in enumerate(canonical_cub_ids)}

    attr_labels_file = dataset_root / "attributes" / "image_attribute_labels.txt"
    if not attr_labels_file.exists():
        attr_labels_file = dataset_root / "image_attribute_labels.txt"
    if not attr_labels_file.exists():
        raise FileNotFoundError(f"Attribute labels file not found under {dataset_root}")

    matrix = np.zeros((total_images + 1, 112), dtype=np.float32)
    with open(attr_labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            attr_id = int(parts[1])
            if attr_id in attr_to_concept_idx:
                img_id = int(parts[0])
                is_present = float(parts[2])
                matrix[img_id, attr_to_concept_idx[attr_id]] = is_present

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, matrix)
        except Exception as e:
            logger.warning(f"Could not save concept matrix cache: {e}")

    return matrix


class CUBConceptDataset(Dataset):
    """
    PyTorch Dataset loading CUB-200-2011 images, 0-indexed species labels (0..199),
    and 112 binary concept labels (values in {0.0, 1.0}).

    Args:
        dataset_root: Path to CUB_200_2011 directory.
        split: One of 'train', 'val', 'calib', or 'test'.
        manifest_path: Path to dataset partition manifest JSON.
        concept_mapping_path: Path to selected_concepts_112.json.
        transform: Optional torchvision transform. If None, default split transform is used.
        image_size: Input image dimension (default: 224).
        allow_test_split: Safety flag. If False and split=='test', raises PermissionError.
        custom_image_paths: Optional dict mapping image_id -> relative image path (for synthetic test fixtures).
        custom_image_classes: Optional dict mapping image_id -> 1-based class ID (for synthetic test fixtures).
        custom_concept_matrix: Optional np.ndarray of shape [N, 112] (for synthetic test fixtures).
        custom_split_ids: Optional list of image IDs for this partition (for synthetic test fixtures).
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        split: str,
        manifest_path: Optional[Union[str, Path]] = None,
        concept_mapping_path: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        image_size: int = 224,
        allow_test_split: bool = False,
        custom_image_paths: Optional[Dict[int, str]] = None,
        custom_image_classes: Optional[Dict[int, int]] = None,
        custom_concept_matrix: Optional[np.ndarray] = None,
        custom_split_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split.lower()
        self.image_size = image_size
        self.allow_test_split = allow_test_split

        # Split Isolation Boundary Check
        if self.split == "test" and not self.allow_test_split:
            raise PermissionError(
                "CRITICAL: The official CUB-200-2011 test set is strictly locked and isolated "
                "during model development. It cannot be loaded into training, validation, "
                "calibration, augmentation, or model selection pipelines."
            )

        if self.split not in ("train", "val", "calib", "test"):
            raise ValueError(
                f"Unknown split '{split}'. Expected 'train', 'val', 'calib', or 'test'."
            )

        # Set image transformations
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_cub_transforms(self.split, image_size=self.image_size)

        project_root = Path(__file__).resolve().parent.parent.parent

        # 1. Resolve Split Image IDs
        if custom_split_ids is not None:
            self.image_ids = list(custom_split_ids)
        else:
            if manifest_path is None:
                manifest_path = project_root / "outputs" / "dataset_manifest.json"
            manifest_file = Path(manifest_path)
            if not manifest_file.exists():
                raise FileNotFoundError(
                    f"Partition manifest not found at: {manifest_file}. "
                    "Run scripts/inspect_dataset.py or generate outputs/dataset_manifest.json first."
                )
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            key = f"{self.split}_ids"
            if key not in manifest_data:
                raise KeyError(f"Key '{key}' not found in partition manifest {manifest_file}")
            self.image_ids = manifest_data[key]

        # 2. Resolve Image Metadata & File Paths
        if custom_image_paths is not None and custom_image_classes is not None:
            self.image_paths = dict(custom_image_paths)
            self.image_classes = dict(custom_image_classes)
        else:
            parser = CUBMetadataParser(self.dataset_root)
            meta: CUBMetadata = parser.parse_all(validate_cross_references=False)
            self.image_paths = meta.images
            self.image_classes = meta.image_classes

        # 3. Resolve 112 Concept Labels Matrix
        if custom_concept_matrix is not None:
            self.concept_matrix = custom_concept_matrix
        else:
            if concept_mapping_path is None:
                concept_mapping_path = project_root / "data" / "selected_concepts_112.json"
            cache_path = project_root / "outputs" / "cub_112_concept_labels.npy"
            self.concept_matrix = extract_or_load_concept_matrix(
                dataset_root=self.dataset_root,
                mapping_file=Path(concept_mapping_path),
                cache_path=cache_path,
                total_images=len(self.image_paths),
            )

        # Validate that all partition IDs exist in metadata
        missing_ids = [iid for iid in self.image_ids if iid not in self.image_paths]
        if missing_ids:
            raise KeyError(
                f"Found {len(missing_ids)} partition IDs missing from dataset metadata (e.g., ID {missing_ids[0]})."
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, str]]:
        image_id = self.image_ids[idx]
        rel_path = self.image_paths[image_id]
        full_image_path = self.dataset_root / "images" / rel_path

        # Check image existence
        if not full_image_path.exists():
            # Fallback for synthetic fixtures or direct paths
            if (self.dataset_root / rel_path).exists():
                full_image_path = self.dataset_root / rel_path
            else:
                raise FileNotFoundError(
                    f"Image file for ID {image_id} not found at expected path: {full_image_path}"
                )

        try:
            with Image.open(full_image_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
        except Exception as e:
            raise IOError(f"Corrupt or unreadable image file for ID {image_id} at {full_image_path}: {e}")

        # Apply transformation
        img_tensor = self.transform(rgb_img)

        # Species label: CUB class IDs are 1..200. Convert to 0-indexed for CrossEntropyLoss [0..199]
        raw_class_id = self.image_classes[image_id]
        species_label = raw_class_id - 1
        if not (0 <= species_label < 200):
            raise ValueError(
                f"Invalid 0-indexed species label {species_label} (raw {raw_class_id}) for image ID {image_id}"
            )

        # Concept labels: 112 binary floats in {0.0, 1.0}
        concept_row = self.concept_matrix[image_id]
        concept_tensor = torch.from_numpy(concept_row.astype(np.float32))

        return {
            "image": img_tensor,                                    # [3, 224, 224] FloatTensor
            "concept_labels": concept_tensor,                       # [112] FloatTensor in {0.0, 1.0}
            "species_label": torch.tensor(species_label, dtype=torch.long),  # Scalar LongTensor [0..199]
            "image_id": image_id,                                   # int
            "image_path": str(rel_path),                            # str
        }


def create_cub_dataloaders(
    config: Optional[dict] = None,
    dataset_root: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    concept_mapping_path: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    image_size: int = 224,
    pin_memory: bool = False,
    splits: Tuple[str, ...] = ("train", "val", "calib"),
    allow_test_split: bool = False,
) -> Dict[str, DataLoader]:
    """
    Creates PyTorch DataLoaders for CUB-200-2011 partitions.

    Args:
        config: Loaded project configuration dict (optional).
        dataset_root: Path to CUB dataset directory.
        manifest_path: Path to dataset_manifest.json.
        concept_mapping_path: Path to selected_concepts_112.json.
        batch_size: Batch size (defaults to config['training']['batch_size'] or 32).
        num_workers: Number of DataLoader worker threads (defaults to config or 0).
        image_size: Input image dimension (default: 224).
        pin_memory: Pin memory for GPU transfer (default: False).
        splits: Tuple of splits to instantiate (default: ('train', 'val', 'calib')).
        allow_test_split: Whether to permit creating a test DataLoader (default: False).

    Returns:
        Dict mapping split name -> torch.utils.data.DataLoader
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
            config = {}

    project_root = Path(__file__).resolve().parent.parent.parent
    if dataset_root is None:
        dataset_root = project_root / config.get("data", {}).get("dataset_root", "data/CUB_200_2011")
    if manifest_path is None:
        manifest_path = project_root / "outputs" / "dataset_manifest.json"
    if concept_mapping_path is None:
        concept_mapping_path = project_root / config.get("paths", {}).get(
            "attribute_mapping_file", "data/selected_concepts_112.json"
        )
    if batch_size is None:
        batch_size = int(config.get("training", {}).get("batch_size", 32))
    if num_workers is None:
        num_workers = int(config.get("training", {}).get("num_workers", 0))

    loaders: Dict[str, DataLoader] = {}

    for split in splits:
        split_lower = split.lower()
        dataset = CUBConceptDataset(
            dataset_root=dataset_root,
            split=split_lower,
            manifest_path=manifest_path,
            concept_mapping_path=concept_mapping_path,
            image_size=image_size,
            allow_test_split=allow_test_split,
        )

        is_train = split_lower == "train"
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,       # Shuffle only training data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,        # Preserve final partial batches
        )
        loaders[split_lower] = loader

    return loaders

