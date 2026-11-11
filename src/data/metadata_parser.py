"""
Dataset Metadata Parser for CUB-200-2011
Calibrated Concept-Based Feedback Model (CSE 4385)

This module provides strict, robust parsing and validation for the official
CUB-200-2011 dataset annotation files:
  - images.txt: Mapping of image IDs to relative file paths.
  - image_class_labels.txt: Mapping of image IDs to species class labels (1-200).
  - train_test_split.txt: Official train (1) vs. test (0) partition indicators.
  - attributes/attributes.txt: Attribute IDs to attribute descriptive names.
  - attributes/image_attribute_labels.txt: Dense image-attribute presence matrix.
  - classes.txt (optional): Class IDs to species taxonomy names.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the dataset directory or an essential annotation file is missing."""
    pass


class CorruptAnnotationError(ValueError):
    """Raised when an annotation file contains malformed, incomplete, or invalid data."""
    pass


@dataclass
class CUBMetadata:
    """Holds parsed and cross-validated metadata for the CUB-200-2011 dataset."""
    dataset_dir: Path
    # image_id (1-indexed int) -> relative file path (str)
    images: Dict[int, str] = field(default_factory=dict)
    # image_id -> class_id (1-indexed int)
    image_classes: Dict[int, int] = field(default_factory=dict)
    # image_id -> is_train (1 for train, 0 for test)
    is_train: Dict[int, int] = field(default_factory=dict)
    # class_id -> class_name
    classes: Dict[int, str] = field(default_factory=dict)
    # attribute_id (1-indexed int) -> attribute_name (str)
    attributes: Dict[int, str] = field(default_factory=dict)

    @property
    def total_images(self) -> int:
        return len(self.images)

    @property
    def train_image_ids(self) -> List[int]:
        return sorted([img_id for img_id, train_flag in self.is_train.items() if train_flag == 1])

    @property
    def test_image_ids(self) -> List[int]:
        return sorted([img_id for img_id, train_flag in self.is_train.items() if train_flag == 0])

    @property
    def total_classes(self) -> int:
        return len(set(self.image_classes.values()))

    @property
    def total_attributes(self) -> int:
        return len(self.attributes)


class CUBMetadataParser:
    """
    Parses and cross-validates official CUB-200-2011 dataset annotations.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def parse_images_txt(self) -> Dict[int, str]:
        """Parses images.txt -> {image_id: rel_path}."""
        file_path = self.dataset_dir / "images.txt"
        if not file_path.exists():
            raise DatasetNotFoundError(f"Missing required file: {file_path}")

        images: Dict[int, str] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise CorruptAnnotationError(
                        f"Malformed line in {file_path} at line {line_no}: "
                        f"Expected '<image_id> <path>', got: '{raw_line.strip()}'"
                    )
                try:
                    img_id = int(parts[0])
                except ValueError:
                    raise CorruptAnnotationError(
                        f"Invalid image_id at line {line_no} in {file_path}: '{parts[0]}' is not an integer."
                    )
                if img_id in images:
                    raise CorruptAnnotationError(
                        f"Duplicate image_id {img_id} found at line {line_no} in {file_path}."
                    )
                images[img_id] = parts[1]

        if not images:
            raise CorruptAnnotationError(f"Annotation file {file_path} contains 0 valid image records.")
        return images

    def parse_class_labels_txt(self) -> Dict[int, int]:
        """Parses image_class_labels.txt -> {image_id: class_id}."""
        file_path = self.dataset_dir / "image_class_labels.txt"
        if not file_path.exists():
            raise DatasetNotFoundError(f"Missing required file: {file_path}")

        class_labels: Dict[int, int] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise CorruptAnnotationError(
                        f"Malformed line in {file_path} at line {line_no}: "
                        f"Expected '<image_id> <class_id>', got: '{raw_line.strip()}'"
                    )
                try:
                    img_id = int(parts[0])
                    class_id = int(parts[1])
                except ValueError:
                    raise CorruptAnnotationError(
                        f"Invalid numeric values at line {line_no} in {file_path}: '{raw_line.strip()}'"
                    )
                if class_id < 1 or class_id > 200:
                    raise CorruptAnnotationError(
                        f"Class ID {class_id} at line {line_no} in {file_path} is out of expected range [1, 200]."
                    )
                if img_id in class_labels:
                    raise CorruptAnnotationError(
                        f"Duplicate image_id {img_id} at line {line_no} in {file_path}."
                    )
                class_labels[img_id] = class_id

        return class_labels

    def parse_train_test_split_txt(self) -> Dict[int, int]:
        """Parses train_test_split.txt -> {image_id: is_train}."""
        file_path = self.dataset_dir / "train_test_split.txt"
        if not file_path.exists():
            raise DatasetNotFoundError(f"Missing required file: {file_path}")

        split_dict: Dict[int, int] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise CorruptAnnotationError(
                        f"Malformed line in {file_path} at line {line_no}: "
                        f"Expected '<image_id> <is_train>', got: '{raw_line.strip()}'"
                    )
                try:
                    img_id = int(parts[0])
                    is_train = int(parts[1])
                except ValueError:
                    raise CorruptAnnotationError(
                        f"Invalid numeric values at line {line_no} in {file_path}: '{raw_line.strip()}'"
                    )
                if is_train not in (0, 1):
                    raise CorruptAnnotationError(
                        f"Invalid is_training flag {is_train} at line {line_no} in {file_path}. Must be 0 or 1."
                    )
                if img_id in split_dict:
                    raise CorruptAnnotationError(
                        f"Duplicate image_id {img_id} at line {line_no} in {file_path}."
                    )
                split_dict[img_id] = is_train

        return split_dict

    def parse_classes_txt(self) -> Dict[int, str]:
        """Parses optional classes.txt -> {class_id: class_name}."""
        file_path = self.dataset_dir / "classes.txt"
        if not file_path.exists():
            return {}

        classes: Dict[int, str] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise CorruptAnnotationError(
                        f"Malformed line in {file_path} at line {line_no}: "
                        f"Expected '<class_id> <class_name>', got: '{raw_line.strip()}'"
                    )
                try:
                    class_id = int(parts[0])
                except ValueError:
                    raise CorruptAnnotationError(
                        f"Invalid class_id at line {line_no} in {file_path}: '{parts[0]}' is not an integer."
                    )
                classes[class_id] = parts[1]

        return classes

    def parse_attributes_txt(self) -> Dict[int, str]:
        """Parses attributes/attributes.txt -> {attribute_id: attribute_name}."""
        file_path = self.dataset_dir / "attributes" / "attributes.txt"
        if not file_path.exists():
            # Check fallback in parent if attributes folder isn't nested
            file_path = self.dataset_dir / "attributes.txt"
            if not file_path.exists():
                raise DatasetNotFoundError(f"Missing required file: attributes/attributes.txt in {self.dataset_dir}")

        attributes: Dict[int, str] = {}
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise CorruptAnnotationError(
                        f"Malformed line in {file_path} at line {line_no}: "
                        f"Expected '<attr_id> <name>', got: '{raw_line.strip()}'"
                    )
                try:
                    attr_id = int(parts[0])
                except ValueError:
                    raise CorruptAnnotationError(
                        f"Invalid attribute_id at line {line_no} in {file_path}: '{parts[0]}' is not an integer."
                    )
                attributes[attr_id] = parts[1]

        if not attributes:
            raise CorruptAnnotationError(f"No attributes parsed from {file_path}.")
        return attributes

    def parse_all(self, validate_cross_references: bool = True) -> CUBMetadata:
        """
        Parses all primary metadata files and validates consistency across files.
        """
        if not self.dataset_dir.exists():
            raise DatasetNotFoundError(
                f"Dataset directory not found at: {self.dataset_dir}\n"
                f"Please ensure CUB-200-2011 is extracted into this directory."
            )

        images = self.parse_images_txt()
        class_labels = self.parse_class_labels_txt()
        splits = self.parse_train_test_split_txt()
        classes = self.parse_classes_txt()
        attributes = self.parse_attributes_txt()

        if validate_cross_references:
            # 1. Verify image_ids in class_labels match images.txt
            if set(images.keys()) != set(class_labels.keys()):
                missing_classes = set(images.keys()) - set(class_labels.keys())
                extra_classes = set(class_labels.keys()) - set(images.keys())
                raise CorruptAnnotationError(
                    f"Mismatch between images.txt and image_class_labels.txt: "
                    f"{len(missing_classes)} missing, {len(extra_classes)} extra records."
                )

            # 2. Verify image_ids in train_test_split match images.txt
            if set(images.keys()) != set(splits.keys()):
                missing_splits = set(images.keys()) - set(splits.keys())
                extra_splits = set(splits.keys()) - set(images.keys())
                raise CorruptAnnotationError(
                    f"Mismatch between images.txt and train_test_split.txt: "
                    f"{len(missing_splits)} missing, {len(extra_splits)} extra records."
                )

            # 3. If classes.txt is present, verify class_labels reference known classes
            if classes:
                referenced_classes = set(class_labels.values())
                unknown_classes = referenced_classes - set(classes.keys())
                if unknown_classes:
                    raise CorruptAnnotationError(
                        f"image_class_labels.txt references unknown classes: {unknown_classes}"
                    )

        return CUBMetadata(
            dataset_dir=self.dataset_dir,
            images=images,
            image_classes=class_labels,
            is_train=splits,
            classes=classes,
            attributes=attributes,
        )

