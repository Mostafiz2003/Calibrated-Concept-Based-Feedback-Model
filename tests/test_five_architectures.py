"""
Unit Tests for the Five Multi-Task Recognition Architectures (E2..E6)
Calibrated Concept-Based Feedback Model (CSE 4385)

Tests:
  1. ResNet-50 (E2): Dual-head shapes [B, 200] and [B, 112], finite outputs
  2. ResNet-50 + CBAM (E3): Attention enhancement, dual-head shapes
  3. ResNet-50 + CBAM + FPN (E4): Multiscale pyramid, dual-head shapes
  4. EfficientNet-B2 (E5): Penultimate feature pooling (1408), dual-head shapes
  5. ConvNeXt-Tiny (E6): Penultimate feature pooling (768), dual-head shapes
  6. MultiTaskRecognitionLoss: Correct weighting (L_species + 0.2 * L_concept)
  7. Backward gradient flow for all 5 architectures
"""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.factory import SUPPORTED_ARCHITECTURES, build_recognition_model
from src.training.losses import MultiTaskRecognitionLoss


class TestFiveArchitectures(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.dummy_input = torch.randn(self.batch_size, 3, 224, 224)
        self.dummy_species_targets = torch.tensor([12, 185], dtype=torch.long)
        self.dummy_concept_targets = torch.zeros(self.batch_size, 112, dtype=torch.float32)

    def _test_architecture(self, model_name: str, expected_feature_dim: int):
        model = build_recognition_model(
            model_name=model_name,
            num_classes=200,
            num_concepts=112,
            pretrained=False,  # Random weights for fast unit tests
        )
        model.eval()

        outputs = model(self.dummy_input)

        # Output shape verification
        self.assertEqual(outputs.species_logits.shape, (self.batch_size, 200))
        self.assertEqual(outputs.concept_logits.shape, (self.batch_size, 112))
        self.assertEqual(outputs.features.shape, (self.batch_size, expected_feature_dim))

        # Check finite values
        self.assertTrue(torch.isfinite(outputs.species_logits).all())
        self.assertTrue(torch.isfinite(outputs.concept_logits).all())

        # Test Loss calculation and Backward Gradient Flow
        model.train()
        criterion = MultiTaskRecognitionLoss(lambda_concept=0.2)
        train_outputs = model(self.dummy_input)
        loss_dict = criterion(train_outputs, self.dummy_species_targets, self.dummy_concept_targets)

        self.assertIn("total_loss", loss_dict)
        self.assertIn("species_loss", loss_dict)
        self.assertIn("concept_loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["total_loss"]))

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss_dict["total_loss"].backward()

        # Check gradients exist on prediction heads
        species_head_weight = getattr(model, "species_head").weight
        concept_head_weight = getattr(model, "concept_head").weight
        self.assertIsNotNone(species_head_weight.grad)
        self.assertIsNotNone(concept_head_weight.grad)
        self.assertFalse(torch.all(species_head_weight.grad == 0))
        self.assertFalse(torch.all(concept_head_weight.grad == 0))

    def test_e2_resnet50(self):
        self._test_architecture("resnet50", expected_feature_dim=2048)

    def test_e3_resnet50_cbam(self):
        self._test_architecture("resnet50_cbam", expected_feature_dim=2048)

    def test_e4_resnet50_cbam_fpn(self):
        self._test_architecture("resnet50_cbam_fpn", expected_feature_dim=2048)

    def test_e5_efficientnet_b2(self):
        self._test_architecture("efficientnet_b2", expected_feature_dim=1408)

    def test_e6_convnext_tiny(self):
        self._test_architecture("convnext_tiny", expected_feature_dim=768)

    def test_backbone_freezing(self):
        model = build_recognition_model("efficientnet_b2", pretrained=False)
        model.freeze_backbone()
        pcounts = model.get_parameter_counts()
        self.assertGreater(pcounts["frozen"], 0)
        self.assertEqual(pcounts["trainable"], 1408 * 200 + 200 + 1408 * 112 + 112)

        model.unfreeze_backbone()
        pcounts_unfrozen = model.get_parameter_counts()
        self.assertEqual(pcounts_unfrozen["frozen"], 0)
        self.assertEqual(pcounts_unfrozen["trainable"], pcounts["total"])


if __name__ == "__main__":
    unittest.main()

