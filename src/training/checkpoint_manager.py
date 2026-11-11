"""
Atomic and Resilient Checkpoint Management
Calibrated Concept-Based Feedback Model (CSE 4385)

Guarantees:
- Atomic file writes (writes to .tmp before atomic rename) to eliminate corruption on sudden session aborts
- Maintains backup (.bak) of the previous valid checkpoint
- Verifies checkpoint readability upon save
- Stores complete recovery state (model, optimizer, scheduler, AMP scaler, RNG states, early stopping, configs)
- Seamless resumption without restarting from epoch 1
"""

import logging
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages saving and loading of model training checkpoints.

    Args:
        checkpoint_dir: Directory where checkpoints are saved.
        experiment_id: Unique experiment name or ID.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        experiment_id: str,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_id = experiment_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.latest_path = self.checkpoint_dir / f"{experiment_id}_latest.pth"
        self.best_path = self.checkpoint_dir / f"{experiment_id}_best.pth"
        self.backup_path = self.checkpoint_dir / f"{experiment_id}_latest.pth.bak"

    def _get_rng_states(self) -> Dict[str, Any]:
        """Captures complete random number generator states across Python, NumPy, and PyTorch."""
        states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state_all()
        return states

    def _set_rng_states(self, states: Dict[str, Any]) -> None:
        """Restores random number generator states."""
        if "python" in states:
            random.setstate(states["python"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch" in states:
            torch.set_rng_state(states["torch"])
        if "cuda" in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["cuda"])

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
        """
        Saves checkpoint atomically.

        Returns:
            Dict with 'latest' and optionally 'best' Paths.
        """
        checkpoint_payload = {
            "experiment_id": self.experiment_id,
            "epoch": int(epoch),
            "val_metric": float(val_metric),
            "is_best": bool(is_best),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "early_stopping_state": early_stopping.state_dict() if early_stopping is not None else None,
            "rng_states": self._get_rng_states(),
            "config": config or {},
            "metadata": extra_metadata or {},
        }

        # Atomic Save using temporary file in the same directory
        tmp_file = self.checkpoint_dir / f".tmp_{self.experiment_id}_{epoch}.pth"
        try:
            torch.save(checkpoint_payload, tmp_file)

            # Test readability of the newly written checkpoint
            test_load = torch.load(tmp_file, map_location="cpu", weights_only=False)
            if "model_state_dict" not in test_load:
                raise ValueError("Checkpoint validation failed: missing model_state_dict")

            # Backup current latest checkpoint if it exists
            if self.latest_path.exists():
                shutil.copy2(self.latest_path, self.backup_path)

            # Atomic rename / replace
            if os.name == "nt" and self.latest_path.exists():
                # On Windows, os.replace is atomic across same filesystem
                os.replace(str(tmp_file), str(self.latest_path))
            else:
                os.replace(str(tmp_file), str(self.latest_path))

            saved_paths = {"latest": self.latest_path}

            # If this is the best checkpoint, copy to best_path
            if is_best:
                shutil.copy2(self.latest_path, self.best_path)
                saved_paths["best"] = self.best_path
                logger.info(f"Saved new BEST checkpoint (metric: {val_metric:.4f}) to {self.best_path}")

            return saved_paths

        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def has_checkpoint(self) -> bool:
        """Returns True if a valid latest checkpoint exists."""
        return self.latest_path.exists() or self.backup_path.exists()

    def load_latest_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        early_stopping: Optional[Any] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """
        Loads the latest checkpoint and restores complete training state.

        Returns:
            Checkpoint payload dict.
        """
        target_path = self.latest_path
        if not target_path.exists() and self.backup_path.exists():
            logger.warning(f"Latest checkpoint missing, falling back to backup: {self.backup_path}")
            target_path = self.backup_path

        if not target_path.exists():
            raise FileNotFoundError(f"No checkpoint found to resume from in {self.checkpoint_dir}")

        device = torch.device(device)
        logger.info(f"Loading checkpoint from {target_path} onto {device}...")
        checkpoint = torch.load(target_path, map_location=device, weights_only=False)

        # Validate experiment compatibility
        ckpt_exp_id = checkpoint.get("experiment_id")
        if ckpt_exp_id and ckpt_exp_id != self.experiment_id:
            raise ValueError(
                f"Incompatible checkpoint: expected experiment_id '{self.experiment_id}', "
                f"but found '{ckpt_exp_id}' in {target_path}"
            )

        # 1. Restore Model
        model.load_state_dict(checkpoint["model_state_dict"])

        # 2. Restore Optimizer
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 3. Restore Scheduler
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # 4. Restore AMP Scaler
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # 5. Restore Early Stopping
        if early_stopping is not None and checkpoint.get("early_stopping_state") is not None:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])

        # 6. Restore RNG
        if "rng_states" in checkpoint:
            try:
                self._set_rng_states(checkpoint["rng_states"])
            except Exception as e:
                logger.warning(f"Could not restore full RNG states: {e}")

        logger.info(
            f"Successfully resumed experiment '{self.experiment_id}' at epoch {checkpoint['epoch']} "
            f"(Best metric: {checkpoint.get('val_metric', 'N/A')})"
        )
        return checkpoint

