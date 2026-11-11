"""
Unit Tests for Dataset Preprocessing, Image Validation, and Partition Integrity
Calibrated Concept-Based Feedback Model (CSE 4385)
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_statistics import compute_dataset_statistics
from src.data.image_validation import validate_single_image
from src.data.partition_manager import PartitionIsolationError, PartitionManager
from src.data.preprocessing import PreprocessingConfig, get_standardized_transforms


class TestPreprocessingPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_image_validation_valid_and_corrupt(self):
        # 1. Create a valid test image
        valid_img_path = self.test_path / "valid.jpg"
        img = Image.new("RGB", (300, 200), color=(100, 150, 200))
        img.save(valid_img_path, format="JPEG")

        rec_valid = validate_single_image(valid_img_path, image_id=1)
        self.assertTrue(rec_valid["valid"])
        self.assertEqual(rec_valid["width"], 300)
        self.assertEqual(rec_valid["height"], 200)
        self.assertEqual(rec_valid["channels"], 3)
        self.assertTrue(rec_valid["rgb_convertible"])

        # 2. Create a corrupted file
        corrupt_path = self.test_path / "corrupt.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"not_an_image_header_bytes_123456")

        rec_corrupt = validate_single_image(corrupt_path, image_id=2)
        self.assertFalse(rec_corrupt["valid"])
        self.assertIsNotNone(rec_corrupt["error_message"])

    def test_standardized_transforms_aspect_ratio_preservation(self):
        cfg = PreprocessingConfig(input_size=224, resize_size=256)
        train_tx = get_standardized_transforms("train", cfg)
        val_tx = get_standardized_transforms("val", cfg)

        img = Image.new("RGB", (400, 300))
        train_out = train_tx(img)
        val_out = val_tx(img)

        self.assertEqual(train_out.shape, (3, 224, 224))
        self.assertEqual(val_out.shape, (3, 224, 224))
        self.assertEqual(train_out.dtype, torch.float32)
        self.assertEqual(val_out.dtype, torch.float32)

    def test_partition_isolation_detection(self):
        # Create a manifest with artificial leakage
        leak_manifest = self.test_path / "leak_manifest.json"
        import json
        with open(leak_manifest, "w", encoding="utf-8") as f:
            json.dump({
                "train_ids": [1, 2, 3],
                "val_ids": [3, 4],  # ID 3 is leaked into both train and val!
                "calib_ids": [5, 6],
                "test_ids": [7, 8],
            }, f)

        pm = PartitionManager(leak_manifest)
        with self.assertRaises(PartitionIsolationError):
            pm.verify_isolation()


if __name__ == "__main__":
    unittest.main()

