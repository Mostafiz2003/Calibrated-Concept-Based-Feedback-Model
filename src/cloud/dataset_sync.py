"""
High-Performance Dataset Synchronization and Staging for Google Colab
Calibrated Concept-Based Feedback Model (CSE 4385)

Solves the Google Drive FUSE I/O Bottleneck:
- Accessing thousands of individual image files directly through /content/drive/ is extremely
  slow and causes heavy I/O throttling or runtime timeouts.
- This module stages the dataset into Colab's high-speed local NVMe/SSD (/content/data/CUB_200_2011/).
- Supports extracting from a single archive (CUB_200_2011.tgz or .zip) on Drive in seconds.
- Avoids redundant re-copying or re-extracting if already staged and intact.
"""

import logging
from pathlib import Path
import shutil
import tarfile
from typing import Any, Dict, Optional, Union
import zipfile

logger = logging.getLogger(__name__)


class DatasetStagingManager:
    """
    Manages staging CUB-200-2011 dataset from persistent Drive storage to local Colab SSD.

    Args:
        drive_source_dir: Directory or archive on Google Drive.
        local_staging_dir: Local fast-access directory on Colab SSD (/content/data/CUB_200_2011).
    """

    ESSENTIAL_FILES = [
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "attributes/attributes.txt",
        "attributes/image_attribute_labels.txt",
    ]

    def __init__(
        self,
        drive_source_dir: Union[str, Path] = "/content/drive/MyDrive/CalibratedBirdRecognition/dataset/CUB_200_2011",
        local_staging_dir: Union[str, Path] = "/content/data/CUB_200_2011",
    ):
        self.drive_source = Path(drive_source_dir)
        self.local_staging = Path(local_staging_dir)

    def is_staged_and_intact(self) -> bool:
        """Checks if local staging directory already contains all essential annotation files and images."""
        if not self.local_staging.exists():
            return False

        for rel in self.ESSENTIAL_FILES:
            if not (self.local_staging / rel).exists():
                return False

        images_dir = self.local_staging / "images"
        if not images_dir.exists() or not any(images_dir.iterdir()):
            return False

        return True

    def stage_dataset(self, archive_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Stages dataset into local fast SSD storage.

        Args:
            archive_path: Optional path to .tgz or .zip archive on Drive.

        Returns:
            Dict with staging status and statistics.
        """
        if self.is_staged_and_intact():
            logger.info(f"Dataset already staged and verified at {self.local_staging}. Skipping re-extraction.")
            return {"status": "ALREADY_STAGED", "path": str(self.local_staging)}

        self.local_staging.parent.mkdir(parents=True, exist_ok=True)

        # 1. Check if archive exists on Drive
        resolved_archive = Path(archive_path) if archive_path else None
        if resolved_archive is None:
            # Check standard archive names in drive parent directory
            for candidate in [
                self.drive_source.parent / "CUB_200_2011.tgz",
                self.drive_source.parent / "CUB_200_2011.tar.gz",
                self.drive_source.parent / "CUB_200_2011.zip",
                self.drive_source / "CUB_200_2011.tgz",
            ]:
                if candidate.exists():
                    resolved_archive = candidate
                    break

        # 2. Extract from archive (Fastest method: 15-30s on Colab SSD)
        if resolved_archive and resolved_archive.exists():
            logger.info(f"Extracting dataset archive from {resolved_archive} to {self.local_staging.parent}...")
            if resolved_archive.suffix in (".tgz", ".gz"):
                with tarfile.open(resolved_archive, "r:*") as tar:
                    tar.extractall(path=self.local_staging.parent)
            elif resolved_archive.suffix == ".zip":
                with zipfile.ZipFile(resolved_archive, "r") as zip_ref:
                    zip_ref.extractall(path=self.local_staging.parent)

        # 3. Fallback: Copy directly from Drive folder (only if valid dataset directory with images.txt)
        elif self.drive_source.exists() and self.drive_source.is_dir() and (self.drive_source / "images.txt").exists():
            logger.info(f"Copying dataset from {self.drive_source} to local staging {self.local_staging}...")
            shutil.copytree(self.drive_source, self.local_staging, dirs_exist_ok=True)

        else:
            raise FileNotFoundError(
                f"Cannot stage dataset: Neither valid Drive dataset directory ({self.drive_source}) "
                f"nor archive ({resolved_archive}) was found on Google Drive."
            )

        # Verify staged integrity
        if not self.is_staged_and_intact():
            raise RuntimeError(f"Staged dataset verification failed at {self.local_staging}.")

        logger.info(f"Successfully staged CUB-200-2011 to {self.local_staging}")
        return {"status": "STAGED_SUCCESSFULLY", "path": str(self.local_staging)}

