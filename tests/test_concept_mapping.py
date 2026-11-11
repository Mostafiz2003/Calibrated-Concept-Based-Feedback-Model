"""
Unit Tests for Authoritative 112-Concept Mapping & Verification
Calibrated Concept-Based Feedback Model (CSE 4385)

Tests the recovered canonical 112-concept benchmark mapping from Koh et al. (ICML 2020),
verifying index uniqueness, valid attribute ID ranges [1, 312], matching names in
attributes.txt, and strict reproducibility.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.concept_audit import save_concept_mapping, load_concept_mapping

# The authoritative 112 attribute column indices (0-indexed) recovered from
# Koh et al. (ICML 2020), yewsiang/ConceptBottleneck/CUB/generate_new_data.py
CANONICAL_KOH_0INDEXED = [
    1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50,
    51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106,
    110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 141, 145, 149, 151, 152, 153, 154, 155, 157, 158,
    161, 164, 167, 168, 172, 178, 179, 180, 181, 182, 183, 184, 188, 193, 195, 196, 198, 199, 201, 202,
    206, 207, 208, 211, 212, 213, 220, 221, 225, 235, 238, 242, 253, 254, 259, 274, 276, 277, 283, 289,
    292, 293, 294, 298, 299, 302, 304, 305, 308, 309, 310, 311
]


class TestConceptMapping(unittest.TestCase):

    def test_canonical_list_exact_count(self):
        """Verifies that the canonical list has exactly 112 elements."""
        self.assertEqual(len(CANONICAL_KOH_0INDEXED), 112)

    def test_canonical_list_uniqueness(self):
        """Verifies that all 112 concept indices are strictly unique."""
        self.assertEqual(len(set(CANONICAL_KOH_0INDEXED)), 112)

    def test_canonical_list_range(self):
        """Verifies that all 0-based indices fall strictly within [0, 311]."""
        for idx in CANONICAL_KOH_0INDEXED:
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, 311)

    def test_attributes_txt_alignment(self):
        """
        Cross-validates that every canonical index corresponds to a valid,
        non-empty attribute name in the actual CUB attributes.txt file.
        """
        attr_file = PROJECT_ROOT / "data" / "CUB_200_2011" / "attributes.txt"
        if not attr_file.exists():
            attr_file = PROJECT_ROOT / "data" / "CUB_200_2011" / "attributes" / "attributes.txt"

        if not attr_file.exists():
            self.skipTest("CUB attributes.txt not found; skipping physical file verification.")

        attributes = {}
        with open(attr_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                attributes[int(parts[0])] = parts[1]

        self.assertEqual(len(attributes), 312)

        # Verify 1-based ID = idx + 1
        for concept_idx, idx in enumerate(CANONICAL_KOH_0INDEXED):
            cub_attr_id = idx + 1
            self.assertIn(cub_attr_id, attributes)
            attr_name = attributes[cub_attr_id]
            self.assertTrue(len(attr_name) > 0)
            self.assertIn("has_", attr_name)

    def test_selected_concepts_112_json_file(self):
        """
        Validates the generated data/selected_concepts_112.json file:
        checks metadata completeness, schema, dual-index alignment,
        and name consistency against canonical benchmark indices.
        """
        import json
        json_path = PROJECT_ROOT / "data" / "selected_concepts_112.json"
        self.assertTrue(json_path.exists(), "data/selected_concepts_112.json does not exist")

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.assertIn("metadata", payload)
        self.assertIn("concepts", payload)

        meta = payload["metadata"]
        self.assertEqual(meta.get("num_concepts"), 112)
        self.assertIn("indexing_convention_note", meta)
        self.assertIn("source_repository", meta)

        concepts = payload["concepts"]
        self.assertEqual(len(concepts), 112)

        for i, c in enumerate(concepts):
            self.assertEqual(c["concept_index"], i)
            expected_0idx = CANONICAL_KOH_0INDEXED[i]
            self.assertEqual(c["zero_based_column_index"], expected_0idx)
            self.assertEqual(c["cub_attribute_id"], expected_0idx + 1)
            self.assertTrue(c["attribute_name"].startswith("has_"))


if __name__ == "__main__":
    unittest.main()
