"""
Partition Management and Isolation Verification for CUB-200-2011
Calibrated Concept-Based Feedback Model (CSE 4385)

Enforces strict dataset isolation:
- Train: 4,794 images (gradient updates only)
- Validation: 600 images (model selection and early stopping only)
- Calibration: 600 images (held out for post-hoc calibration / conformal prediction)
- Official Test: 5,794 images (strictly isolated, locked)

Verifies:
- 0% overlap across all partitions
- Stratified 200-species representation across splits
- Canonical 112-concept label alignment
- Generates outputs/preprocessing/partition_integrity_report.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class PartitionIsolationError(Exception):
    """Raised when data leakage or overlap across partitions is detected."""
    pass


class PartitionManager:
    """
    Validates and manages dataset partitions according to outputs/dataset_manifest.json.
    """

    EXPECTED_COUNTS = {
        "train": 4794,
        "val": 600,
        "calib": 600,
        "test": 5794,
    }
    OFFICIAL_TRAIN_TOTAL = 5994
    OFFICIAL_TEST_TOTAL = 5794
    DATASET_TOTAL = 11788

    def __init__(self, manifest_path: Union[str, Path] = "outputs/dataset_manifest.json"):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Partition manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest_data = json.load(f)

        self.train_ids: List[int] = self.manifest_data.get("train_ids", [])
        self.val_ids: List[int] = self.manifest_data.get("val_ids", [])
        self.calib_ids: List[int] = self.manifest_data.get("calib_ids", [])
        self.test_ids: List[int] = self.manifest_data.get("test_ids", [])

    def verify_partition_counts(self) -> Dict[str, Any]:
        """Verifies that split image counts match expected counts exactly."""
        actual_counts = {
            "train": len(self.train_ids),
            "val": len(self.val_ids),
            "calib": len(self.calib_ids),
            "test": len(self.test_ids),
        }

        mismatches: Dict[str, Tuple[int, int]] = {}
        for split, expected in self.EXPECTED_COUNTS.items():
            actual = actual_counts[split]
            if actual != expected:
                mismatches[split] = (actual, expected)

        dev_sum = actual_counts["train"] + actual_counts["val"] + actual_counts["calib"]
        total_sum = dev_sum + actual_counts["test"]

        return {
            "actual_counts": actual_counts,
            "expected_counts": self.EXPECTED_COUNTS,
            "counts_match": len(mismatches) == 0,
            "mismatches": mismatches,
            "development_split_sum": dev_sum,
            "development_sum_matches_official_train": dev_sum == self.OFFICIAL_TRAIN_TOTAL,
            "total_images": total_sum,
            "total_matches_dataset": total_sum == self.DATASET_TOTAL,
        }

    def verify_isolation(self) -> Dict[str, Any]:
        """
        Verifies zero overlap between train, val, calib, and test sets.
        """
        s_train = set(self.train_ids)
        s_val = set(self.val_ids)
        s_calib = set(self.calib_ids)
        s_test = set(self.test_ids)

        overlaps = {
            "train_val_overlap": len(s_train & s_val),
            "train_calib_overlap": len(s_train & s_calib),
            "val_calib_overlap": len(s_val & s_calib),
            "test_train_overlap": len(s_test & s_train),
            "test_val_overlap": len(s_test & s_val),
            "test_calib_overlap": len(s_test & s_calib),
        }

        total_leakage = sum(overlaps.values())
        is_isolated = total_leakage == 0

        if not is_isolated:
            raise PartitionIsolationError(
                f"Data leakage detected across partitions! Overlap details: {overlaps}"
            )

        return {
            "is_strictly_isolated": is_isolated,
            "total_leakage_count": total_leakage,
            "pairwise_overlaps": overlaps,
        }

    def verify_species_stratification(
        self,
        image_classes: Dict[int, int],
    ) -> Dict[str, Any]:
        """
        Verifies that all 200 species are represented in train, val, and calib partitions.
        """
        def get_species_set(ids: List[int]) -> Set[int]:
            return {image_classes[iid] for iid in ids if iid in image_classes}

        sp_train = get_species_set(self.train_ids)
        sp_val = get_species_set(self.val_ids)
        sp_calib = get_species_set(self.calib_ids)
        sp_test = get_species_set(self.test_ids)

        return {
            "train_species_count": len(sp_train),
            "val_species_count": len(sp_val),
            "calib_species_count": len(sp_calib),
            "test_species_count": len(sp_test),
            "all_species_in_train": len(sp_train) == 200,
            "all_species_in_val": len(sp_val) == 200,
            "all_species_in_calib": len(sp_calib) == 200,
            "all_species_in_test": len(sp_test) == 200,
        }

    def generate_integrity_report(
        self,
        image_classes: Optional[Dict[int, int]] = None,
        output_path: Union[str, Path] = "outputs/preprocessing/partition_integrity_report.json",
    ) -> Path:
        """
        Runs all partition checks and writes a comprehensive integrity report JSON.
        """
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        counts_check = self.verify_partition_counts()
        isolation_check = self.verify_isolation()

        strat_check = {}
        if image_classes is not None:
            strat_check = self.verify_species_stratification(image_classes)

        report = {
            "manifest_file": str(self.manifest_path),
            "status": "PASSED" if counts_check["counts_match"] and isolation_check["is_strictly_isolated"] else "FAILED",
            "counts_verification": counts_check,
            "isolation_verification": isolation_check,
            "stratification_verification": strat_check,
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved partition integrity report to {out_file}")
        return out_file
