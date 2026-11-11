"""
Unit Tests for Google Drive Management and Automatic Checkpoint Recovery
Calibrated Concept-Based Feedback Model (CSE 4385)
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud.cloud_checkpoint_manager import CloudCheckpointManager
from src.cloud.colab_environment import assert_gpu_available, inspect_hardware_environment
from src.cloud.dataset_sync import DatasetStagingManager
from src.cloud.google_drive_manager import GoogleDriveManager
from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class TestCloudAndCheckpointRecovery(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_google_drive_manager_initialization(self):
        drive_dir = self.test_path / "mock_drive"
        mgr = GoogleDriveManager(base_drive_dir=drive_dir, local_fallback_dir=drive_dir)
        paths = mgr.initialize_project_structure()

        self.assertIn("dataset/CUB_200_2011", paths)
        self.assertIn("checkpoints/resnet50", paths)
        self.assertTrue((drive_dir / "logs").exists())

        perm = mgr.verify_permissions()
        self.assertTrue(perm["can_read"])
        self.assertTrue(perm["can_write"])

    def test_checkpoint_atomic_saving_and_backup(self):
        ckpt_dir = self.test_path / "checkpoints"
        mgr = CheckpointManager(ckpt_dir, "test_exp")

        model = SimpleTestModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # 1. Save epoch 1
        res1 = mgr.save_checkpoint(model, optimizer, epoch=1, val_metric=0.5, is_best=True)
        self.assertTrue(res1["latest"].exists())
        self.assertTrue(res1["best"].exists())

        # 2. Save epoch 2 (creates backup of epoch 1)
        res2 = mgr.save_checkpoint(model, optimizer, epoch=2, val_metric=0.6, is_best=True)
        self.assertTrue(mgr.backup_path.exists())

        # 3. Resume
        new_model = SimpleTestModel()
        new_opt = optim.SGD(new_model.parameters(), lr=0.01)
        loaded = mgr.load_latest_checkpoint(new_model, new_opt)

        self.assertEqual(loaded["epoch"], 2)
        self.assertEqual(loaded["val_metric"], 0.6)

    def test_cloud_checkpoint_drive_mirroring(self):
        local_dir = self.test_path / "local_ckpt"
        drive_dir = self.test_path / "drive_ckpt"

        cloud_mgr = CloudCheckpointManager(local_dir, drive_dir, "cloud_exp")
        model = SimpleTestModel()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Save checkpoint
        cloud_mgr.save_checkpoint(model, optimizer, epoch=1, val_metric=0.75, is_best=True)

        # Verify mirrored file exists on Drive
        drive_latest = drive_dir / "cloud_exp_latest.pth"
        drive_best = drive_dir / "cloud_exp_best.pth"
        self.assertTrue(drive_latest.exists())
        self.assertTrue(drive_best.exists())

        # Simulate fresh Colab session by deleting local directory
        shutil.rmtree(local_dir)
        self.assertFalse(local_dir.exists())

        # Instantiate fresh manager pointing to the same Drive
        fresh_mgr = CloudCheckpointManager(local_dir, drive_dir, "cloud_exp")
        self.assertTrue(fresh_mgr.has_checkpoint())

        new_model = SimpleTestModel()
        loaded = fresh_mgr.load_latest_checkpoint(new_model)
        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["val_metric"], 0.75)

    def test_early_stopping(self):
        es = EarlyStopping(patience=3, mode="max")

        # Epoch 1: improve
        self.assertTrue(es.update(0.50, 1))
        self.assertFalse(es.should_stop)

        # Epoch 2: improve
        self.assertTrue(es.update(0.55, 2))
        self.assertFalse(es.should_stop)

        # Epoch 3: no improve (count=1)
        self.assertFalse(es.update(0.54, 3))
        self.assertFalse(es.should_stop)

        # Epoch 4: no improve (count=2)
        self.assertFalse(es.update(0.53, 4))
        self.assertFalse(es.should_stop)

        # Epoch 5: no improve (count=3 -> triggers stop)
        self.assertFalse(es.update(0.52, 5))
        self.assertTrue(es.should_stop)
        self.assertEqual(es.best_epoch, 2)
        self.assertEqual(es.best_score, 0.55)

    def test_gpu_assert_guard(self):
        # On CPU environment, assert_gpu_available(allow_cpu=False) must raise RuntimeError
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                assert_gpu_available(allow_cpu=False)

            # Bypassed when allow_cpu=True
            assert_gpu_available(allow_cpu=True)


if __name__ == "__main__":
    unittest.main()

