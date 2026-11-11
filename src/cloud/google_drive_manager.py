"""
Google Drive Directory and Storage Manager for Google Colab
Calibrated Concept-Based Feedback Model (CSE 4385)

Automates Google Drive integration:
- Standard mount point: /content/drive/MyDrive/CalibratedBirdRecognition
- Automatic folder structure initialization (dataset, checkpoints, logs, reports)
- Safe read/write permission verification
- Graceful mock/fallback for local development outside Google Colab
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class GoogleDriveManager:
    """
    Manages persistent project storage on Google Drive during Colab execution.

    Args:
        base_drive_dir: Base directory on Drive (defaults to '/content/drive/MyDrive/CalibratedBirdRecognition').
        local_fallback_dir: Fallback directory when running locally outside Colab.
    """

    SUBDIRECTORIES = [
        "dataset/CUB_200_2011",
        "dataset/processed",
        "dataset/partitions",
        "dataset/concept_mapping",
        "checkpoints/resnet50",
        "checkpoints/resnet50_cbam",
        "checkpoints/resnet50_cbam_fpn",
        "checkpoints/efficientnet_b2",
        "checkpoints/convnext_tiny",
        "logs",
        "metrics",
        "predictions",
        "configs",
        "reports",
        "experiment_registry",
    ]

    def __init__(
        self,
        base_drive_dir: Union[str, Path] = "/content/drive/MyDrive/CalibratedBirdRecognition",
        local_fallback_dir: Union[str, Path] = "outputs/colab_drive_mirror",
    ):
        self.is_colab = self._detect_colab_environment()

        if self.is_colab:
            self.root = Path(base_drive_dir)
        else:
            self.root = Path(local_fallback_dir)

    @staticmethod
    def _detect_colab_environment() -> bool:
        """Detects whether running inside Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return "COLAB_GPU" in os.environ or Path("/content").exists()

    def mount_drive(self) -> bool:
        """
        Attempts to mount Google Drive in Colab.
        Must be called from a notebook or Colab runtime where interactive auth is possible.
        """
        if not self.is_colab:
            logger.info("Not running in Google Colab. Using local mirror directory.")
            self.root.mkdir(parents=True, exist_ok=True)
            return True

        try:
            from google.colab import drive
            drive.mount("/content/drive")
            logger.info("Successfully mounted Google Drive at /content/drive")
            return True
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            return False

    def initialize_project_structure(self) -> Dict[str, Path]:
        """
        Creates all required project subdirectories on Drive.

        Returns:
            Dict mapping subdirectory key -> Path.
        """
        created_paths: Dict[str, Path] = {}
        for subdir in self.SUBDIRECTORIES:
            folder = self.root / subdir
            folder.mkdir(parents=True, exist_ok=True)
            created_paths[subdir] = folder

        logger.info(f"Initialized {len(self.SUBDIRECTORIES)} Drive directories under {self.root}")
        return created_paths

    def verify_permissions(self) -> Dict[str, Any]:
        """Verifies read and write permissions in the base directory."""
        test_file = self.root / ".perm_test.tmp"
        can_read = False
        can_write = False
        error = None

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test_permission")
            can_write = True

            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            can_read = (content == "test_permission")

            test_file.unlink()
        except Exception as e:
            error = str(e)
            if test_file.exists():
                try:
                    test_file.unlink()
                except Exception:
                    pass

        return {
            "root_path": str(self.root),
            "is_colab": self.is_colab,
            "can_read": can_read,
            "can_write": can_write,
            "error": error,
        }

