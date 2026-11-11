"""
Cloud-Synced Checkpoint Manager for Google Colab and Google Drive
Calibrated Concept-Based Feedback Model (CSE 4385)

Guarantees Resilience Against Colab Disconnects:
1. Checkpoints are saved locally on fast SSD (/content/checkpoints/)
2. Automatically synced atomically to persistent Google Drive (MyDrive/CalibratedBirdRecognition/checkpoints/)
3. When resuming in a fresh Colab session, checks Drive first, stages checkpoint locally,
   and resumes training seamlessly without starting from epoch 1.
"""

import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.training.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class CloudCheckpointManager(CheckpointManager):
    """
    Extends CheckpointManager with automated Google Drive mirroring and recovery.

    Args:
        local_dir: Local fast directory on SSD (/content/checkpoints/<experiment_id>).
        drive_dir: Persistent Google Drive directory (MyDrive/CalibratedBirdRecognition/checkpoints/<experiment_id>).
        experiment_id: Unique experiment name.
    """

    def __init__(
        self,
        local_dir: Union[str, Path],
        drive_dir: Optional[Union[str, Path]],
        experiment_id: str,
    ):
        super().__init__(checkpoint_dir=local_dir, experiment_id=experiment_id)
        self.drive_dir = Path(drive_dir) if drive_dir is not None else None
        if self.drive_dir is not None:
            self.drive_dir.mkdir(parents=True, exist_ok=True)

    def _sync_to_drive(self, local_path: Path, drive_filename: str) -> None:
        """Mirrors local checkpoint to Google Drive atomically."""
        if self.drive_dir is None or not self.drive_dir.exists():
            return

        drive_target = self.drive_dir / drive_filename
        drive_tmp = self.drive_dir / f".tmp_{drive_filename}"

        try:
            shutil.copy2(local_path, drive_tmp)
            drive_tmp.replace(drive_target)
            logger.info(f"Mirrored checkpoint to Google Drive: {drive_target}")
        except Exception as e:
            logger.warning(f"Could not mirror checkpoint to Google Drive: {e}")
            if drive_tmp.exists():
                try:
                    drive_tmp.unlink()
                except Exception:
                    pass

    def _restore_from_drive(self, filename: str) -> bool:
        """Restores missing local checkpoint from Google Drive."""
        if self.drive_dir is None:
            return False

        drive_file = self.drive_dir / filename
        local_file = self.checkpoint_dir / filename

        if drive_file.exists() and not local_file.exists():
            logger.info(f"Restoring checkpoint from Google Drive ({drive_file}) to local ({local_file})...")
            shutil.copy2(drive_file, local_file)
            return True
        return False

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_metric: float,
        is_best: bool,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Saves locally and syncs to Google Drive."""
        saved_paths = super().save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_metric=val_metric,
            is_best=is_best,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            config=config,
            extra_metadata=extra_metadata,
        )

        # Mirror to Drive
        if "latest" in saved_paths:
            self._sync_to_drive(saved_paths["latest"], f"{self.experiment_id}_latest.pth")

        if "best" in saved_paths:
            self._sync_to_drive(saved_paths["best"], f"{self.experiment_id}_best.pth")

        return saved_paths

    def has_checkpoint(self) -> bool:
        """Checks both local and Drive storage for existing checkpoints."""
        if super().has_checkpoint():
            return True

        if self.drive_dir is not None:
            drive_latest = self.drive_dir / f"{self.experiment_id}_latest.pth"
            if drive_latest.exists():
                return True

        return False

    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Loads latest checkpoint, retrieving from Drive if local directory is empty."""
        filename = f"{self.experiment_id}_latest.pth"
        if not (self.checkpoint_dir / filename).exists():
            self._restore_from_drive(filename)

        best_filename = f"{self.experiment_id}_best.pth"
        if not (self.checkpoint_dir / best_filename).exists():
            self._restore_from_drive(best_filename)

        return super().load_latest_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            device=device,
        )

