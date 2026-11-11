"""
Unit Tests for Recognition-Focused Dual-Head Architecture (ResNet-50 + CBAM + FPN + Dual Heads)
Calibrated Concept-Based Feedback Model (CSE 4385)

Tests:
  - ResNet-50 multi-stage feature extraction [C2, C3, C4, C5]
  - CBAM channel and spatial attention output shapes, finite values, and gradient flow
  - FPN multi-scale feature pyramid construction and multi-scale feature aggregation
  - Dual head output shapes: Concept logits [B, 112] and Species logits [B, 200]
  - Concept probabilities in [0.0, 1.0] and species predictions in [0..199]
  - Joint recognition loss calculation and non-zero finite backward gradients
  - Configurable architecture modes (E2, E3, E4)
  - Pretrained weight loading and explicit failure handling (PretrainedWeightsLoadError)
  - Checkpoint save and reload compatibility
"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.attention import CBAM, ChannelAttention, SpatialAttention
from src.models.concept_model import PretrainedWeightsLoadError
from src.models.fpn import FeaturePyramidNetwork, MultiScaleFeatureAggregation
from src.models.recognition_model import (
    DualHeadOutput,
    DualHeadRecognitionModel,
    JointRecognitionLoss,
    build_recognition_model,
)


class TestRecognitionArchitecture(unittest.TestCase):

    def test_cbam_attention_shapes_and_finite_values(self):
        """Verifies CBAM channel and spatial attention preserves tensor shape and produces finite outputs."""
        cbam = CBAM(in_channels=256, reduction_ratio=16, kernel_size=7, use_residual=True)
        cbam.eval()

        dummy_feat = torch.randn(2, 256, 28, 28)
        with torch.no_grad():
            out = cbam(dummy_feat)

        self.assertEqual(out.shape, dummy_feat.shape)
        self.assertTrue(torch.isfinite(out).all())

        # Test Channel Attention submodule
        ca = ChannelAttention(in_channels=256, reduction_ratio=16)
        ca_weights = ca(dummy_feat)
        self.assertEqual(ca_weights.shape, torch.Size([2, 256, 1, 1]))
        self.assertTrue(((ca_weights >= 0.0) & (ca_weights <= 1.0)).all())

        # Test Spatial Attention submodule
        sa = SpatialAttention(kernel_size=7)
        sa_weights = sa(dummy_feat)
        self.assertEqual(sa_weights.shape, torch.Size([2, 1, 28, 28]))
        self.assertTrue(((sa_weights >= 0.0) & (sa_weights <= 1.0)).all())

    def test_fpn_pyramid_and_aggregation(self):
        """Verifies FPN builds multi-scale feature maps and aggregation yields a unified vector."""
        in_channels = [256, 512, 1024, 2048]
        fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=256)
        agg = MultiScaleFeatureAggregation(num_levels=4, in_channels=256, out_dim=2048)
        fpn.eval()
        agg.eval()

        # Synthetic multi-stage feature maps [C2, C3, C4, C5]
        c2 = torch.randn(2, 256, 56, 56)
        c3 = torch.randn(2, 512, 28, 28)
        c4 = torch.randn(2, 1024, 14, 14)
        c5 = torch.randn(2, 2048, 7, 7)

        with torch.no_grad():
            pyramid_levels = fpn([c2, c3, c4, c5])

        self.assertEqual(len(pyramid_levels), 4)
        self.assertEqual(pyramid_levels[0].shape, torch.Size([2, 256, 56, 56]))  # P2
        self.assertEqual(pyramid_levels[1].shape, torch.Size([2, 256, 28, 28]))  # P3
        self.assertEqual(pyramid_levels[2].shape, torch.Size([2, 256, 14, 14]))  # P4
        self.assertEqual(pyramid_levels[3].shape, torch.Size([2, 256, 7, 7]))    # P5

        with torch.no_grad():
            fused = agg(pyramid_levels)

        self.assertEqual(fused.shape, torch.Size([2, 2048]))
        self.assertTrue(torch.isfinite(fused).all())

    def test_resnet50_stage_feature_extraction(self):
        """Verifies that ResNet-50 extracts 4 multi-stage feature maps with correct dimensions."""
        model = DualHeadRecognitionModel(
            num_concepts=112,
            num_classes=200,
            experiment_mode="E2",
            pretrained=False,
        )
        model.eval()

        dummy_x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            stages = model.extract_stage_features(dummy_x)

        self.assertEqual(len(stages), 4)
        self.assertEqual(stages[0].shape, torch.Size([2, 256, 56, 56]))  # C2
        self.assertEqual(stages[1].shape, torch.Size([2, 512, 28, 28]))  # C3
        self.assertEqual(stages[2].shape, torch.Size([2, 1024, 14, 14])) # C4
        self.assertEqual(stages[3].shape, torch.Size([2, 2048, 7, 7]))   # C5

    def test_dual_head_output_shapes_and_values(self):
        """Verifies that the full E4 model produces 112 concept logits and 200 species logits."""
        model = DualHeadRecognitionModel(
            num_concepts=112,
            num_classes=200,
            experiment_mode="E4",
            pretrained=False,
        )
        model.eval()

        batch_size = 3
        dummy_x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            out = model(dummy_x)

        # Verify container access methods
        self.assertIsInstance(out, DualHeadOutput)
        self.assertEqual(out.concept_logits.shape, torch.Size([batch_size, 112]))
        self.assertEqual(out.species_logits.shape, torch.Size([batch_size, 200]))
        self.assertEqual(out["concept_logits"].shape, torch.Size([batch_size, 112]))
        self.assertEqual(out["species_logits"].shape, torch.Size([batch_size, 200]))

        # Tuple unpacking test
        c_logits, s_logits = out
        self.assertEqual(c_logits.shape, torch.Size([batch_size, 112]))
        self.assertEqual(s_logits.shape, torch.Size([batch_size, 200]))
        self.assertTrue(torch.isfinite(c_logits).all())
        self.assertTrue(torch.isfinite(s_logits).all())

    def test_concept_and_species_probabilities_helpers(self):
        """Verifies prediction probability calculations and species class index extraction."""
        model = DualHeadRecognitionModel(
            num_concepts=112,
            num_classes=200,
            experiment_mode="E4",
            pretrained=False,
        )
        model.eval()

        dummy_x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            c_probs = model.predict_concept_probabilities(dummy_x)
            s_probs = model.predict_species_probabilities(dummy_x)
            s_preds = model.predict_species(dummy_x)

        self.assertEqual(c_probs.shape, torch.Size([2, 112]))
        self.assertTrue(((c_probs >= 0.0) & (c_probs <= 1.0)).all())

        self.assertEqual(s_probs.shape, torch.Size([2, 200]))
        self.assertTrue(((s_probs >= 0.0) & (s_probs <= 1.0)).all())
        # Softmax probabilities must sum to 1 across the 200 species
        sums = torch.sum(s_probs, dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

        self.assertEqual(s_preds.shape, torch.Size([2]))
        self.assertEqual(s_preds.dtype, torch.int64)
        self.assertTrue(((s_preds >= 0) & (s_preds < 200)).all())

    def test_joint_loss_calculation_and_finite_gradients(self):
        """Verifies joint weighted loss computation and gradient backpropagation through dual heads."""
        model = DualHeadRecognitionModel(
            num_concepts=112,
            num_classes=200,
            experiment_mode="E4",
            pretrained=False,
        )
        model.train()

        loss_fn = JointRecognitionLoss(lambda_concept=1.5, lambda_species=2.0)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        batch_size = 2
        dummy_x = torch.randn(batch_size, 3, 224, 224)
        concept_targets = torch.randint(0, 2, (batch_size, 112)).float()
        species_targets = torch.tensor([10, 45], dtype=torch.long)

        optimizer.zero_grad()
        out = model(dummy_x)
        loss_dict = loss_fn(
            concept_logits=out.concept_logits,
            species_logits=out.species_logits,
            concept_targets=concept_targets,
            species_targets=species_targets,
        )

        self.assertIn("total_loss", loss_dict)
        self.assertIn("concept_loss", loss_dict)
        self.assertIn("species_loss", loss_dict)

        total_loss = loss_dict["total_loss"]
        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(total_loss.item() > 0.0)

        # Check gradient flow
        total_loss.backward()

        self.assertIsNotNone(model.concept_head.weight.grad)
        self.assertTrue(torch.count_nonzero(model.concept_head.weight.grad) > 0)

        self.assertIsNotNone(model.species_head.weight.grad)
        self.assertTrue(torch.count_nonzero(model.species_head.weight.grad) > 0)

        # Check gradients in FPN and CBAM
        self.assertIsNotNone(model.fpn.smooth_convs[0][0].weight.grad)
        self.assertIsNotNone(model.cbam_modules[0].channel_att.mlp[0].weight.grad)

        optimizer.step()

    def test_configurable_experiments_e2_e3_e4(self):
        """Verifies that E2, E3, and E4 instantiate correct component configurations."""
        # E2: ResNet-50 Dual-Head (no CBAM, no FPN)
        m_e2 = DualHeadRecognitionModel(experiment_mode="E2", pretrained=False)
        self.assertFalse(m_e2.use_cbam)
        self.assertFalse(m_e2.use_fpn)
        self.assertIsNone(m_e2.cbam_modules)
        self.assertIsNone(m_e2.fpn)
        self.assertIn("E2", m_e2.experiment_name)

        # E3: ResNet-50 + CBAM (with CBAM, no FPN)
        m_e3 = DualHeadRecognitionModel(experiment_mode="E3", pretrained=False)
        self.assertTrue(m_e3.use_cbam)
        self.assertFalse(m_e3.use_fpn)
        self.assertIsNotNone(m_e3.cbam_modules)
        self.assertIsNone(m_e3.fpn)
        self.assertIn("E3", m_e3.experiment_name)

        # E4: ResNet-50 + CBAM + FPN (with CBAM, with FPN)
        m_e4 = DualHeadRecognitionModel(experiment_mode="E4", pretrained=False)
        self.assertTrue(m_e4.use_cbam)
        self.assertTrue(m_e4.use_fpn)
        self.assertIsNotNone(m_e4.cbam_modules)
        self.assertIsNotNone(m_e4.fpn)
        self.assertIn("E4", m_e4.experiment_name)

    def test_factory_build_recognition_model(self):
        """Verifies factory function with default and custom configs."""
        config = {
            "model": {
                "pretrained": False,
                "recognition_model": {
                    "experiment_mode": "E3",
                    "use_cbam": True,
                    "use_fpn": False,
                }
            },
            "data": {"num_concepts": 112, "num_classes": 200},
        }
        m = build_recognition_model(config=config)
        self.assertEqual(m.num_concepts, 112)
        self.assertEqual(m.num_classes, 200)
        self.assertFalse(m.pretrained)

    def test_pretrained_failure_raises_informative_error(self):
        """Verifies that failure to load pretrained weights raises PretrainedWeightsLoadError."""
        with patch("src.models.recognition_model.models.resnet50", side_effect=Exception("Network failure")):
            with self.assertRaises(PretrainedWeightsLoadError) as ctx:
                DualHeadRecognitionModel(experiment_mode="E4", pretrained=True)
            self.assertIn("Failed to load pretrained weights for 'resnet50'", str(ctx.exception))
            self.assertIn("pretrained=False", str(ctx.exception))

    def test_checkpoint_save_and_load_compatibility(self):
        """Verifies that DualHeadRecognitionModel saves and restores all weights and architecture state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_recognition_checkpoint.pth"

            # Create original model and record output
            orig_model = DualHeadRecognitionModel(experiment_mode="E4", pretrained=False)
            orig_model.eval()

            dummy_x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                orig_out = orig_model(dummy_x)

            # Save checkpoint
            checkpoint_payload = {
                "model_state_dict": orig_model.state_dict(),
                "experiment_mode": "E4",
                "use_cbam": orig_model.use_cbam,
                "use_fpn": orig_model.use_fpn,
                "num_concepts": 112,
                "num_classes": 200,
            }
            torch.save(checkpoint_payload, tmp_path)
            self.assertTrue(tmp_path.exists())

            # Load into fresh model instance
            restored_model = DualHeadRecognitionModel(experiment_mode="E4", pretrained=False)
            loaded_checkpoint = torch.load(tmp_path, map_location="cpu", weights_only=False)
            restored_model.load_state_dict(loaded_checkpoint["model_state_dict"])
            restored_model.eval()

            with torch.no_grad():
                restored_out = restored_model(dummy_x)

            self.assertTrue(torch.allclose(orig_out.concept_logits, restored_out.concept_logits, atol=1e-6))
            self.assertTrue(torch.allclose(orig_out.species_logits, restored_out.species_logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

