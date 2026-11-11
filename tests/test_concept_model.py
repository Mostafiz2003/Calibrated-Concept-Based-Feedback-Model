"""
Unit Tests for Initial Backbone and Concept Predictor Model
Calibrated Concept-Based Feedback Model (CSE 4385)

Tests:
  - Model forward pass output shape [batch_size, 112] and finite outputs
  - Distinction between logits and probabilities (sigmoid activation)
  - Feature extraction shape [batch_size, feature_dim]
  - Modular backbone support (resnet18, resnet34, resnet50)
  - Rejection of unsupported backbones (ValueError)
  - Pretrained weight failure handling (no silent substitution with random weights)
  - Factory function integration with project configuration
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.concept_model import (
    ConceptPredictor,
    build_concept_predictor,
    PretrainedWeightsLoadError,
)


class TestConceptModel(unittest.TestCase):

    def test_model_output_shape_and_finite_values(self):
        """Verifies forward pass returns finite logits of shape [batch_size, 112]."""
        model = ConceptPredictor(backbone_name="resnet18", num_concepts=112, pretrained=False)
        model.eval()

        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            logits = model(dummy_input)

        self.assertEqual(logits.shape, torch.Size([batch_size, 112]))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(logits).all(), "Model output contains NaN or Inf!")

    def test_logits_vs_probabilities(self):
        """Verifies that predict_probabilities returns values strictly in [0.0, 1.0] matching sigmoid(logits)."""
        model = ConceptPredictor(backbone_name="resnet18", num_concepts=112, pretrained=False)
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = model.predict_probabilities(dummy_input)
            static_probs = ConceptPredictor.logits_to_probabilities(logits)

        self.assertEqual(probs.shape, torch.Size([2, 112]))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        self.assertTrue(torch.allclose(probs, torch.sigmoid(logits), atol=1e-6))
        self.assertTrue(torch.allclose(probs, static_probs, atol=1e-6))

    def test_extract_features_shape(self):
        """Verifies feature extractor outputs R^512 for ResNet-18/34 and R^2048 for ResNet-50."""
        # ResNet-18
        m18 = ConceptPredictor(backbone_name="resnet18", pretrained=False)
        m18.eval()
        feat18 = m18.extract_features(torch.randn(2, 3, 224, 224))
        self.assertEqual(feat18.shape, torch.Size([2, 512]))

        # ResNet-34
        m34 = ConceptPredictor(backbone_name="resnet34", pretrained=False)
        m34.eval()
        feat34 = m34.extract_features(torch.randn(2, 3, 224, 224))
        self.assertEqual(feat34.shape, torch.Size([2, 512]))

        # ResNet-50
        m50 = ConceptPredictor(backbone_name="resnet50", pretrained=False)
        m50.eval()
        feat50 = m50.extract_features(torch.randn(2, 3, 224, 224))
        self.assertEqual(feat50.shape, torch.Size([2, 2048]))

    def test_unsupported_backbone_raises_value_error(self):
        """Verifies that attempting to configure an unsupported backbone raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            ConceptPredictor(backbone_name="vgg16", pretrained=False)
        self.assertIn("Unsupported backbone 'vgg16'", str(ctx.exception))

    def test_missing_custom_weights_file_raises_error(self):
        """Verifies that specifying a nonexistent weights_path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ConceptPredictor(
                backbone_name="resnet18",
                pretrained=True,
                weights_path="nonexistent_checkpoint.pth",
            )

    def test_pretrained_failure_raises_informative_error_without_silent_substitution(self):
        """Verifies that if downloading pretrained weights fails, PretrainedWeightsLoadError is raised."""
        with patch("src.models.concept_model.models.resnet18", side_effect=Exception("Network connection timed out")):
            with self.assertRaises(PretrainedWeightsLoadError) as ctx:
                ConceptPredictor(backbone_name="resnet18", pretrained=True)
            self.assertIn("Failed to download/load official pretrained weights", str(ctx.exception))
            self.assertIn("pretrained: false", str(ctx.exception))

    def test_factory_build_concept_predictor(self):
        """Verifies factory constructor with config dictionary."""
        config = {
            "model": {"backbone": "resnet34", "pretrained": False},
            "data": {"num_concepts": 112},
        }
        model = build_concept_predictor(config=config)
        self.assertEqual(model.backbone_name, "resnet34")
        self.assertEqual(model.num_concepts, 112)
        self.assertFalse(model.is_pretrained)


if __name__ == "__main__":
    unittest.main()
