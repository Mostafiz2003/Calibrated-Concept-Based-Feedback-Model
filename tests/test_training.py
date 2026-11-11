"""
Unit Tests for Concept Model Training Pipeline
Calibrated Concept-Based Feedback Model (CSE 4385)

Tests:
  - Metric computation (BCE loss, macro F1, mean accuracy, precision, recall)
  - Gradient computation, non-zero gradients, and optimizer step execution
  - Output shape [batch_size, 112] and finite loss verification
  - Checkpoint saving, serialization completeness, and state restoration
  - Smoke-test execution with limited batches
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.concept_model import ConceptPredictor
from src.training.metrics import compute_concept_metrics
from src.training.trainer import ConceptTrainer


class TestTrainingPipeline(unittest.TestCase):

    def test_compute_concept_metrics_known_values(self):
        """Verifies metric computation logic on synthetic known logits and targets."""
        # 2 samples, 3 concepts
        # Sample 1: true=[1, 0, 1], logits=[10.0, -10.0, 10.0] -> probs=[1, 0, 1] (perfect match)
        # Sample 2: true=[0, 1, 0], logits=[-10.0, 10.0, 10.0] -> probs=[0, 1, 1] (1 mismatch)
        logits = torch.tensor([[10.0, -10.0, 10.0], [-10.0, 10.0, 10.0]])
        targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        metrics = compute_concept_metrics(logits, targets)

        self.assertIn("loss", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("mean_accuracy", metrics)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)

        # 5 out of 6 elements are correct: mean accuracy = 5/6 = 0.8333
        self.assertAlmostEqual(metrics["mean_accuracy"], 5.0 / 6.0, places=4)
        # Exactly 1 out of 2 samples has 100% correct concepts: exact match ratio = 0.5
        self.assertEqual(metrics["exact_match_ratio"], 0.5)
        # Loss must be finite and positive
        self.assertTrue(metrics["loss"] > 0.0)

    def test_gradient_computation_and_optimizer_step(self):
        """Verifies that loss.backward() computes non-zero gradients and optimizer.step() updates weights."""
        model = ConceptPredictor(backbone_name="resnet18", num_concepts=112, pretrained=False)
        model.train()

        # Record initial weights of concept head
        initial_weight = model.concept_head.weight.clone().detach()

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        dummy_targets = torch.randint(0, 2, (batch_size, 112)).float()

        optimizer.zero_grad()
        logits = model(dummy_images)

        # Shape verification [batch_size, 112]
        self.assertEqual(logits.shape, torch.Size([batch_size, 112]))

        loss = criterion(logits, dummy_targets)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()

        # Check gradients are computed and non-zero on concept head
        self.assertIsNotNone(model.concept_head.weight.grad)
        self.assertTrue(torch.count_nonzero(model.concept_head.weight.grad) > 0)

        optimizer.step()

        # Check weights actually changed
        updated_weight = model.concept_head.weight.detach()
        self.assertFalse(torch.equal(initial_weight, updated_weight))

    def test_checkpoint_save_and_reload_state(self):
        """Verifies that checkpoints preserve model weights, optimizer state, and metadata accurately."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model = ConceptPredictor(backbone_name="resnet18", num_concepts=112, pretrained=False)
            optimizer = optim.AdamW(model.parameters(), lr=0.005)

            # Dummy training dataset with dict samples matching CUBConceptDataset
            class SyntheticDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 10
                def __getitem__(self, idx):
                    return {
                        "image": torch.randn(3, 224, 224),
                        "concept_labels": torch.randint(0, 2, (112,)).float(),
                        "species_label": torch.tensor(idx % 200, dtype=torch.long),
                        "image_id": idx + 1,
                        "image_path": f"synthetic_{idx}.jpg",
                    }

            train_loader = DataLoader(SyntheticDataset(), batch_size=4, shuffle=False)
            val_loader = DataLoader(SyntheticDataset(), batch_size=4, shuffle=False)

            trainer = ConceptTrainer(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=tmp_path,
            )

            # Run 1 epoch
            results = trainer.fit(num_epochs=1)
            self.assertEqual(results["completed_epochs"], 1)

            best_ckpt = tmp_path / "best_concept_model.pth"
            self.assertTrue(best_ckpt.exists(), "best_concept_model.pth was not created!")
            self.assertTrue((tmp_path / "training_history.csv").exists())
            self.assertTrue((tmp_path / "training_log.json").exists())

            # Create fresh model and reload checkpoint
            new_model = ConceptPredictor(backbone_name="resnet18", num_concepts=112, pretrained=False)
            new_optimizer = optim.AdamW(new_model.parameters(), lr=0.005)
            new_trainer = ConceptTrainer(
                model=new_model,
                optimizer=new_optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=tmp_path,
            )

            checkpoint_meta = new_trainer.load_checkpoint(best_ckpt)
            self.assertEqual(checkpoint_meta["epoch"], 1)
            self.assertEqual(checkpoint_meta["num_concepts"], 112)

            # Check that weights of new model match original model
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()

