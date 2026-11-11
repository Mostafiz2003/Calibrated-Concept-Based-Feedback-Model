"""
Automated Unit Tests for Configuration Management
Calibrated Concept-Based Feedback Model (CSE 4385)

These tests run quickly without requiring dataset download or model training.
Compatible with standard unittest and pytest.
"""

import unittest
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, validate_config, ConfigError, get_project_root


class TestConfig(unittest.TestCase):

    def test_default_config_loads_successfully(self):
        """Verifies that configs/default.yaml is present and parses into a valid dict."""
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn("project", config)
        self.assertIn("data", config)
        self.assertIn("model", config)

    def test_proposal_specified_constants(self):
        """Ensures all constants from the proposal document are accurately captured."""
        config = load_config()
        data_cfg = config["data"]

        # Number of species, concepts, and attributes
        self.assertEqual(data_cfg["num_classes"], 200, "Species count must be 200 as per proposal")
        self.assertEqual(data_cfg["num_concepts"], 112, "Retained concepts must be 112 as per proposal")
        self.assertEqual(data_cfg["num_total_attributes"], 312, "Total raw attributes must be 312")

        # Official dataset image counts
        self.assertEqual(data_cfg["total_images"], 11788)
        self.assertEqual(data_cfg["official_train_count"], 5994)
        self.assertEqual(data_cfg["official_test_count"], 5794)

        # Backbone specifications
        model_cfg = config["model"]
        self.assertEqual(model_cfg["backbone"], "resnet50")
        self.assertEqual(model_cfg["fallback_backbone"], "resnet18")

        # Conformal alpha values (Proposal Section 5.4: 0.05, 0.10, 0.20)
        conformal_cfg = config["conformal"]
        self.assertEqual(conformal_cfg["alpha_levels"], [0.05, 0.10, 0.20])

        # Three seeds protocol (Proposal Section 5.1)
        reprod_cfg = config["reproducibility"]
        self.assertEqual(len(reprod_cfg["experiment_seeds"]), 3)

    def test_missing_config_raises_informative_error(self):
        """Verifies that attempting to load a non-existent config file raises ConfigError."""
        non_existent_path = PROJECT_ROOT / "configs" / "non_existent_file.yaml"
        with self.assertRaises(ConfigError) as ctx:
            load_config(non_existent_path)
        self.assertIn("not found", str(ctx.exception).lower())

    def test_invalid_concept_count_fails_validation(self):
        """Verifies that tampering with concept count triggers validation failure."""
        config = load_config()
        tampered_config = dict(config)
        tampered_config["data"] = dict(config["data"])
        tampered_config["data"]["num_concepts"] = 50  # Invalid! Must be 112

        with self.assertRaises(ConfigError) as ctx:
            validate_config(tampered_config)
        self.assertIn("num_concepts", str(ctx.exception))

    def test_missing_required_section_fails_validation(self):
        """Verifies that omitting a required section triggers validation failure."""
        incomplete_config = {
            "project": {"name": "Test"},
            "reproducibility": {"seed": 42}
            # Missing data, model, calibration, conformal
        }
        with self.assertRaises(ConfigError) as ctx:
            validate_config(incomplete_config)
        self.assertIn("Missing required configuration section", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

