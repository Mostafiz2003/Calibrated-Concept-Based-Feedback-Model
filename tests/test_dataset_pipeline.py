"""
Unit Tests for CUB-200-2011 Dataset Pipeline
Calibrated Concept-Based Feedback Model (CSE 4385)

Uses synthetic fixtures to test metadata parsing, malformed annotation handling,
deterministic partitions, test-set isolation, and concept audit functionality
without requiring the 1.15 GB external dataset download.
"""

from pathlib import Path
import shutil
import tempfile
import unittest

from src.data.metadata_parser import (
    CUBMetadataParser,
    CUBMetadata,
    DatasetNotFoundError,
    CorruptAnnotationError,
)
from src.data.partitions import (
    create_deterministic_splits,
    save_partition_manifest,
    load_partition_manifest,
    PartitionManifest,
)
from src.data.concept_audit import (
    ConceptAuditor,
    save_concept_mapping,
    load_concept_mapping,
    AttributeAuditReport,
)


class TestDatasetPipeline(unittest.TestCase):

    def setUp(self):
        """Creates a synthetic mini-CUB dataset fixture in a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.temp_dir) / "CUB_synthetic"
        self.dataset_dir.mkdir(parents=True)
        (self.dataset_dir / "attributes").mkdir(parents=True)

        # 1. images.txt: 20 images across 4 classes (5 images per class)
        # Class 1: img 1..5, Class 2: img 6..10, Class 3: img 11..15, Class 4: img 16..20
        with open(self.dataset_dir / "images.txt", "w", encoding="utf-8") as f:
            for i in range(1, 21):
                cls_idx = (i - 1) // 5 + 1
                f.write(f"{i} class_{cls_idx}/bird_{i:04d}.jpg\n")

        # 2. image_class_labels.txt
        with open(self.dataset_dir / "image_class_labels.txt", "w", encoding="utf-8") as f:
            for i in range(1, 21):
                cls_idx = (i - 1) // 5 + 1
                f.write(f"{i} {cls_idx}\n")

        # 3. train_test_split.txt: First 3 of each class are train (12 total), last 2 are test (8 total)
        with open(self.dataset_dir / "train_test_split.txt", "w", encoding="utf-8") as f:
            for i in range(1, 21):
                in_class_idx = (i - 1) % 5
                is_train = 1 if in_class_idx < 3 else 0
                f.write(f"{i} {is_train}\n")

        # 4. classes.txt
        with open(self.dataset_dir / "classes.txt", "w", encoding="utf-8") as f:
            for c in range(1, 5):
                f.write(f"{c} 00{c}.Species_{c}\n")

        # 5. attributes/attributes.txt: 5 attributes
        self.attr_names = {
            1: "has_bill_shape::all-purpose",
            2: "has_wing_color::black",
            3: "has_throat_color::yellow",
            4: "has_crown_color::red",
            5: "has_belly_color::white",
        }
        with open(self.dataset_dir / "attributes" / "attributes.txt", "w", encoding="utf-8") as f:
            for aid, name in self.attr_names.items():
                f.write(f"{aid} {name}\n")

        # 6. attributes/image_attribute_labels.txt
        # Attribute 1: present in all 12 training images (qualifies for all classes)
        # Attribute 2: present in Class 1 & Class 2 training images (qualifies for 2 classes)
        # Attribute 3: present in only 1 training image in Class 1 (qualifies for 0 classes under 50% rule)
        # Attribute 4: absent in all images
        # Attribute 5: present in Class 1, 2, 3 training images (qualifies for 3 classes)
        with open(self.dataset_dir / "attributes" / "image_attribute_labels.txt", "w", encoding="utf-8") as f:
            for img_id in range(1, 21):
                cls_idx = (img_id - 1) // 5 + 1
                # Attr 1: present everywhere
                f.write(f"{img_id} 1 1 3 1.0\n")
                # Attr 2: present in Class 1 and 2
                a2_present = 1 if cls_idx in (1, 2) else 0
                f.write(f"{img_id} 2 {a2_present} 3 1.0\n")
                # Attr 3: present only in image 1
                a3_present = 1 if img_id == 1 else 0
                f.write(f"{img_id} 3 {a3_present} 3 1.0\n")
                # Attr 4: absent
                f.write(f"{img_id} 4 0 3 1.0\n")
                # Attr 5: present in Class 1, 2, 3
                a5_present = 1 if cls_idx in (1, 2, 3) else 0
                f.write(f"{img_id} 5 {a5_present} 3 1.0\n")

    def tearDown(self):
        """Cleans up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --- Parser Tests ---

    def test_metadata_parser_success(self):
        """Verifies that all annotations parse cleanly from valid files."""
        parser = CUBMetadataParser(self.dataset_dir)
        meta = parser.parse_all()

        self.assertEqual(meta.total_images, 20)
        self.assertEqual(meta.total_classes, 4)
        self.assertEqual(len(meta.train_image_ids), 12)
        self.assertEqual(len(meta.test_image_ids), 8)
        self.assertEqual(meta.total_attributes, 5)

    def test_metadata_parser_malformed_line_raises_error(self):
        """Verifies that malformed tokens in images.txt trigger CorruptAnnotationError."""
        corrupt_file = self.dataset_dir / "images.txt"
        with open(corrupt_file, "a", encoding="utf-8") as f:
            f.write("not_an_int path/to/image.jpg\n")

        parser = CUBMetadataParser(self.dataset_dir)
        with self.assertRaises(CorruptAnnotationError):
            parser.parse_images_txt()

    def test_metadata_parser_out_of_range_class_raises_error(self):
        """Verifies that class ID > 200 or < 1 raises CorruptAnnotationError."""
        bad_file = self.dataset_dir / "image_class_labels.txt"
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("1 999\n")  # Out of range!

        parser = CUBMetadataParser(self.dataset_dir)
        with self.assertRaises(CorruptAnnotationError):
            parser.parse_class_labels_txt()

    def test_metadata_parser_missing_file_raises_error(self):
        """Verifies that missing essential files raises DatasetNotFoundError."""
        (self.dataset_dir / "train_test_split.txt").unlink()
        parser = CUBMetadataParser(self.dataset_dir)
        with self.assertRaises(DatasetNotFoundError):
            parser.parse_train_test_split_txt()

    # --- Partition Tests ---

    def test_deterministic_partitions_isolation_and_completeness(self):
        """Verifies 80/10/10 split reproducibility, completeness, and test isolation."""
        parser = CUBMetadataParser(self.dataset_dir)
        meta = parser.parse_all()

        manifest1 = create_deterministic_splits(meta, train_ratio=0.80, val_ratio=0.10, calib_ratio=0.10, seed=42)
        manifest2 = create_deterministic_splits(meta, train_ratio=0.80, val_ratio=0.10, calib_ratio=0.10, seed=42)

        # 1. Determinism
        self.assertEqual(manifest1.train_ids, manifest2.train_ids)
        self.assertEqual(manifest1.val_ids, manifest2.val_ids)
        self.assertEqual(manifest1.calib_ids, manifest2.calib_ids)
        self.assertEqual(manifest1.test_ids, manifest2.test_ids)

        # 2. Mutual exclusivity
        train_s = set(manifest1.train_ids)
        val_s = set(manifest1.val_ids)
        calib_s = set(manifest1.calib_ids)
        test_s = set(manifest1.test_ids)

        self.assertEqual(train_s & val_s, set(), "Train and Val must not overlap")
        self.assertEqual(train_s & calib_s, set(), "Train and Calib must not overlap")
        self.assertEqual(val_s & calib_s, set(), "Val and Calib must not overlap")

        # 3. Completeness with official train
        self.assertEqual(train_s | val_s | calib_s, set(meta.train_image_ids))

        # 4. Strict test set isolation
        self.assertEqual((train_s | val_s | calib_s) & test_s, set(), "Training splits must never touch test set")
        self.assertEqual(test_s, set(meta.test_image_ids), "Test set must remain untouched")

    def test_partition_manifest_save_and_load(self):
        """Verifies that partition manifests serialize and deserialize without loss."""
        parser = CUBMetadataParser(self.dataset_dir)
        meta = parser.parse_all()
        manifest = create_deterministic_splits(meta, seed=42)

        split_dir = self.dataset_dir / "splits"
        save_partition_manifest(manifest, split_dir)

        loaded = load_partition_manifest(split_dir)
        self.assertEqual(manifest.train_ids, loaded.train_ids)
        self.assertEqual(manifest.val_ids, loaded.val_ids)
        self.assertEqual(manifest.calib_ids, loaded.calib_ids)
        self.assertEqual(manifest.test_ids, loaded.test_ids)

    # --- Concept Audit & Selection Tests ---

    def test_concept_auditor_filtering_behavior(self):
        """Verifies that concept filtering strictly counts qualifying classes without fabricating."""
        parser = CUBMetadataParser(self.dataset_dir)
        meta = parser.parse_all()

        auditor = ConceptAuditor(meta)

        # Rule: class_majority_50 with threshold min_classes = 2
        # - Attr 1 qualifies for all 4 classes -> qualifies (>= 2)
        # - Attr 2 qualifies for 2 classes (1, 2) -> qualifies (>= 2)
        # - Attr 3 has 1 image out of 3 in Class 1 (33% < 50%) -> 0 classes -> does NOT qualify
        # - Attr 4 has 0 images -> does NOT qualify
        # - Attr 5 qualifies for 3 classes (1, 2, 3) -> qualifies (>= 2)
        report = auditor.audit_attributes(counting_rule="class_majority_50", min_classes=2, train_only=True)

        self.assertEqual(report.total_attributes_audited, 5)
        self.assertEqual(set(report.qualifying_attribute_ids), {1, 2, 5})
        self.assertEqual(report.qualifying_count, 3)

    def test_concept_mapping_format(self):
        """Verifies that concept mapping separates concept_idx (0..K-1) from cub_attribute_id."""
        mapping_file = Path(self.temp_dir) / "test_mapping.json"
        selected_ids = [2, 5]
        save_concept_mapping(
            selected_attribute_ids=selected_ids,
            attributes_dict=self.attr_names,
            output_path=mapping_file,
            metadata_info={"rule": "test_rule"},
        )

        loaded = load_concept_mapping(mapping_file)
        concepts = loaded["concepts"]
        self.assertEqual(len(concepts), 2)

        # Check concept 0
        self.assertEqual(concepts[0]["concept_idx"], 0)
        self.assertEqual(concepts[0]["cub_attribute_id"], 2)
        self.assertEqual(concepts[0]["attribute_name"], "has_wing_color::black")

        # Check concept 1
        self.assertEqual(concepts[1]["concept_idx"], 1)
        self.assertEqual(concepts[1]["cub_attribute_id"], 5)
        self.assertEqual(concepts[1]["attribute_name"], "has_belly_color::white")


if __name__ == "__main__":
    unittest.main()

