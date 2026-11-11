"""
Unit Tests for PyTorch CUB Concept Dataset and DataLoader Pipeline
Calibrated Concept-Based Feedback Model (CSE 4385)

Tests:
  - Dataset sample structure, tensor types, shapes, and value ranges
  - Canonical 112 concept mapping alignment
  - Deterministic evaluation transforms vs. training augmentations
  - DataLoader batching and final partial batch preservation
  - Test set isolation and protection (PermissionError on unauthorized access)
  - Missing/corrupted image error handling
"""

from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    CUBConceptDataset,
    create_cub_dataloaders,
    get_cub_transforms,
    load_canonical_concept_ids,
)


class TestCUBConceptDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_root = PROJECT_ROOT / "data" / "CUB_200_2011"
        cls.manifest_path = PROJECT_ROOT / "outputs" / "dataset_manifest.json"
        cls.mapping_path = PROJECT_ROOT / "data" / "selected_concepts_112.json"
        cls.has_cub = cls.dataset_root.exists() and cls.manifest_path.exists()

    def test_canonical_concept_ids_load_and_order(self):
        """Verifies that 112 canonical IDs are loaded in strict concept_index order."""
        ids = load_canonical_concept_ids(self.mapping_path)
        self.assertEqual(len(ids), 112)
        self.assertEqual(len(set(ids)), 112)
        # Check first and last canonical IDs against verified benchmark
        self.assertEqual(ids[0], 2)    # 0-index 1 -> ID 2 (has_bill_shape::dagger)
        self.assertEqual(ids[-1], 312) # 0-index 311 -> ID 312 (has_wing_pattern::multi-colored)

    def test_deterministic_eval_transforms(self):
        """Verifies that validation and calibration transforms produce bitwise identical outputs."""
        eval_transform = get_cub_transforms("val", image_size=224)
        img = Image.new("RGB", (300, 400), color=(120, 150, 180))

        t1 = eval_transform(img)
        t2 = eval_transform(img)
        self.assertTrue(torch.equal(t1, t2), "Evaluation transform produced non-deterministic outputs!")

        calib_transform = get_cub_transforms("calib", image_size=224)
        t3 = calib_transform(img)
        self.assertTrue(torch.equal(t1, t3), "Val and Calib transforms should be identical evaluation pipelines.")

    def test_training_augmentation_configuration(self):
        """Verifies that training transform includes stochastic data augmentations."""
        train_transform = get_cub_transforms("train", image_size=224)
        transform_types = [type(t).__name__ for t in train_transform.transforms]
        self.assertIn("RandomCrop", transform_types)
        self.assertIn("RandomHorizontalFlip", transform_types)

        # Confirm eval transform does NOT have random augmentations
        eval_transform = get_cub_transforms("val", image_size=224)
        eval_types = [type(t).__name__ for t in eval_transform.transforms]
        self.assertNotIn("RandomCrop", eval_types)
        self.assertNotIn("RandomHorizontalFlip", eval_types)
        self.assertIn("CenterCrop", eval_types)

    def test_split_isolation_blocks_test_split(self):
        """Verifies that attempting to load the official test set without authorization raises PermissionError."""
        with self.assertRaises(PermissionError):
            CUBConceptDataset(
                dataset_root=self.dataset_root,
                split="test",
                allow_test_split=False,
            )

    def test_dataset_sample_structure_and_ranges(self):
        """Verifies sample keys, tensor shapes, data types, and valid ranges on the validation partition."""
        if not self.has_cub:
            self.skipTest("CUB dataset or manifest not present; skipping physical sample verification.")

        dataset = CUBConceptDataset(
            dataset_root=self.dataset_root,
            split="val",
            manifest_path=self.manifest_path,
            concept_mapping_path=self.mapping_path,
        )
        self.assertEqual(len(dataset), 600)

        sample = dataset[0]
        # Required sample dictionary keys
        expected_keys = {"image", "concept_labels", "species_label", "image_id", "image_path"}
        self.assertEqual(set(sample.keys()), expected_keys)

        # Image tensor shape and dtype
        img = sample["image"]
        self.assertEqual(img.shape, torch.Size([3, 224, 224]))
        self.assertEqual(img.dtype, torch.float32)

        # Concept labels shape, dtype, and values in {0.0, 1.0}
        concepts = sample["concept_labels"]
        self.assertEqual(concepts.shape, torch.Size([112]))
        self.assertEqual(concepts.dtype, torch.float32)
        unique_concept_vals = set(concepts.numpy().tolist())
        self.assertTrue(unique_concept_vals.issubset({0.0, 1.0}))

        # Species label: 0-indexed integer in [0, 199]
        species = sample["species_label"]
        self.assertEqual(species.dtype, torch.int64)
        self.assertTrue(0 <= species.item() < 200)

    def test_dataloader_batch_shapes_and_partial_batches(self):
        """Verifies DataLoader batching, preservation of final partial batch, and tensor collation."""
        if not self.has_cub:
            self.skipTest("CUB dataset or manifest not present; skipping DataLoader verification.")

        loaders = create_cub_dataloaders(
            dataset_root=self.dataset_root,
            manifest_path=self.manifest_path,
            concept_mapping_path=self.mapping_path,
            batch_size=32,
            splits=("val",),
        )
        val_loader = loaders["val"]

        # 600 total images with batch_size 32:
        # 600 // 32 = 18 full batches (576 images) + 1 final partial batch (24 images) = 19 batches
        self.assertEqual(len(val_loader), 19)

        # Verify first batch shapes
        first_batch = next(iter(val_loader))
        self.assertEqual(first_batch["image"].shape, torch.Size([32, 3, 224, 224]))
        self.assertEqual(first_batch["concept_labels"].shape, torch.Size([32, 112]))
        self.assertEqual(first_batch["species_label"].shape, torch.Size([32]))

        # Verify partial batch preservation using a mini-subset (5 samples with batch_size=3 -> [3], [2])
        mini_subset = torch.utils.data.Subset(val_loader.dataset, indices=[0, 1, 2, 3, 4])
        mini_loader = torch.utils.data.DataLoader(mini_subset, batch_size=3, shuffle=False, drop_last=False)
        self.assertEqual(len(mini_loader), 2)
        mini_batches = list(mini_loader)
        self.assertEqual(mini_batches[0]["image"].shape, torch.Size([3, 3, 224, 224]))
        self.assertEqual(mini_batches[1]["image"].shape, torch.Size([2, 3, 224, 224]))
        self.assertEqual(mini_batches[1]["concept_labels"].shape, torch.Size([2, 112]))

    def test_synthetic_fixture_missing_image_handling(self):
        """Verifies that missing or corrupted image files raise FileNotFoundError with diagnostic message."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            custom_paths = {1: "nonexistent_bird.jpg"}
            custom_classes = {1: 10}
            custom_matrix = np.zeros((2, 112), dtype=np.float32)

            ds = CUBConceptDataset(
                dataset_root=tmp_root,
                split="val",
                custom_split_ids=[1],
                custom_image_paths=custom_paths,
                custom_image_classes=custom_classes,
                custom_concept_matrix=custom_matrix,
            )
            with self.assertRaises(FileNotFoundError) as ctx:
                _ = ds[0]
            self.assertIn("Image file for ID 1 not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
