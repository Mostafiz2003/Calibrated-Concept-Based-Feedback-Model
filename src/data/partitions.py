"""
Deterministic Data Partitioning for CUB-200-2011
Calibrated Concept-Based Feedback Model (CSE 4385)

Creates reproducible Train (~80%), Validation (~10%), and Calibration (~10%)
partitions from the official training portion (5,994 images), strictly keeping
the official test set (5,794 images) locked and untouched.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple

from src.data.metadata_parser import CUBMetadata


@dataclass
class PartitionManifest:
    """Stores image IDs for each partition and split metadata."""
    seed: int
    train_ids: List[int]
    val_ids: List[int]
    calib_ids: List[int]
    test_ids: List[int]
    stratified: bool = True

    @property
    def train_count(self) -> int:
        return len(self.train_ids)

    @property
    def val_count(self) -> int:
        return len(self.val_ids)

    @property
    def calib_count(self) -> int:
        return len(self.calib_ids)

    @property
    def test_count(self) -> int:
        return len(self.test_ids)

    @property
    def official_train_count(self) -> int:
        return self.train_count + self.val_count + self.calib_count

    def validate_isolation(self, official_train_ids: Set[int], official_test_ids: Set[int]) -> bool:
        """
        Validates mathematical properties of data partitions:
          1. Mutual exclusivity between all train, val, and calib partitions.
          2. Completeness: Union of (train, val, calib) equals official train portion.
          3. Complete isolation of the official test set.
        """
        train_set = set(self.train_ids)
        val_set = set(self.val_ids)
        calib_set = set(self.calib_ids)
        test_set = set(self.test_ids)

        # 1. Mutual exclusivity among training splits
        if train_set & val_set:
            raise ValueError(f"Data leakage detected! Train and Validation intersect: {train_set & val_set}")
        if train_set & calib_set:
            raise ValueError(f"Data leakage detected! Train and Calibration intersect: {train_set & calib_set}")
        if val_set & calib_set:
            raise ValueError(f"Data leakage detected! Validation and Calibration intersect: {val_set & calib_set}")

        # 2. Completeness with official training set
        union_train = train_set | val_set | calib_set
        if union_train != official_train_ids:
            missing = official_train_ids - union_train
            extra = union_train - official_train_ids
            raise ValueError(
                f"Split completeness check failed! Missing: {len(missing)}, Extra: {len(extra)}"
            )

        # 3. Test isolation check
        if union_train & test_set:
            raise ValueError("CRITICAL: Leakage detected between training partitions and test set!")

        if test_set != official_test_ids:
            raise ValueError("Test set modified! Must exactly equal official test partition.")

        return True


def create_deterministic_splits(
    metadata: CUBMetadata,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    calib_ratio: float = 0.10,
    seed: int = 42,
    stratify: bool = True,
) -> PartitionManifest:
    """
    Creates deterministic Train, Validation, and Calibration partitions from
    the official training set, keeping the test set locked and untouched.

    Args:
        metadata: Parsed CUB metadata.
        train_ratio: Fraction for backbone and classifier training (~0.80).
        val_ratio: Fraction for model selection & temperature scaling (~0.10).
        calib_ratio: Fraction for conformal quantile estimation (~0.10).
        seed: Random seed for reproducibility.
        stratify: If True, balances species classes across partitions.

    Returns:
        PartitionManifest with partition image IDs.
    """
    ratio_sum = train_ratio + val_ratio + calib_ratio
    if abs(ratio_sum - 1.0) > 1e-5:
        raise ValueError(
            f"Partition ratios must sum to 1.0! Got: {train_ratio} + {val_ratio} + {calib_ratio} = {ratio_sum}"
        )

    official_train_ids = set(metadata.train_image_ids)
    official_test_ids = set(metadata.test_image_ids)

    train_ids: List[int] = []
    val_ids: List[int] = []
    calib_ids: List[int] = []

    if stratify:
        # Group training image IDs by class
        class_to_train_images: Dict[int, List[int]] = {}
        for img_id in metadata.train_image_ids:
            cls_id = metadata.image_classes[img_id]
            class_to_train_images.setdefault(cls_id, []).append(img_id)

        rng = random.Random(seed)
        for cls_id in sorted(class_to_train_images.keys()):
            imgs = list(class_to_train_images[cls_id])
            # Deterministic per-class shuffle
            rng.shuffle(imgs)

            n = len(imgs)
            n_val = max(1, int(round(n * val_ratio))) if n >= 10 else (1 if n >= 5 else 0)
            n_calib = max(1, int(round(n * calib_ratio))) if n >= 10 else (1 if n >= 6 else 0)
            # Ensure at least 1 image remains in train if possible
            if n - (n_val + n_calib) < 1 and n > 0:
                if n_calib > 0:
                    n_calib -= 1
                elif n_val > 0:
                    n_val -= 1

            cls_val = imgs[:n_val]
            cls_calib = imgs[n_val : n_val + n_calib]
            cls_train = imgs[n_val + n_calib :]

            val_ids.extend(cls_val)
            calib_ids.extend(cls_calib)
            train_ids.extend(cls_train)
    else:
        rng = random.Random(seed)
        shuffled_train = list(metadata.train_image_ids)
        rng.shuffle(shuffled_train)

        n = len(shuffled_train)
        n_val = int(round(n * val_ratio))
        n_calib = int(round(n * calib_ratio))

        val_ids = shuffled_train[:n_val]
        calib_ids = shuffled_train[n_val : n_val + n_calib]
        train_ids = shuffled_train[n_val + n_calib :]

    manifest = PartitionManifest(
        seed=seed,
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        calib_ids=sorted(calib_ids),
        test_ids=sorted(metadata.test_image_ids),
        stratified=stratify,
    )

    # Perform strict self-validation
    manifest.validate_isolation(official_train_ids, official_test_ids)
    return manifest


def save_partition_manifest(manifest: PartitionManifest, output_dir: Path) -> Dict[str, Path]:
    """
    Saves partition image IDs as text manifests and a summary JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train": output_dir / "train_ids.txt",
        "val": output_dir / "val_ids.txt",
        "calib": output_dir / "calib_ids.txt",
        "test": output_dir / "test_ids.txt",
        "summary": output_dir / "partition_summary.json",
    }

    for split_name, path in [
        ("train", files["train"]),
        ("val", files["val"]),
        ("calib", files["calib"]),
        ("test", files["test"]),
    ]:
        ids = getattr(manifest, f"{split_name}_ids")
        with open(path, "w", encoding="utf-8") as f:
            for item in ids:
                f.write(f"{item}\n")

    summary_data = {
        "seed": manifest.seed,
        "stratified": manifest.stratified,
        "train_count": manifest.train_count,
        "val_count": manifest.val_count,
        "calib_count": manifest.calib_count,
        "test_count": manifest.test_count,
        "total_official_train": manifest.official_train_count,
    }

    with open(files["summary"], "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return files


def load_partition_manifest(manifest_dir: Path) -> PartitionManifest:
    """Loads partition IDs and summary from directory."""
    manifest_dir = Path(manifest_dir)
    summary_file = manifest_dir / "partition_summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Partition summary not found: {summary_file}")

    with open(summary_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    def read_ids(filename: str) -> List[int]:
        filepath = manifest_dir / filename
        with open(filepath, "r", encoding="utf-8") as f:
            return [int(line.strip()) for line in f if line.strip()]

    return PartitionManifest(
        seed=meta["seed"],
        train_ids=read_ids("train_ids.txt"),
        val_ids=read_ids("val_ids.txt"),
        calib_ids=read_ids("calib_ids.txt"),
        test_ids=read_ids("test_ids.txt"),
        stratified=meta.get("stratified", True),
    )

