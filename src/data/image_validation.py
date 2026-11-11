"""
Image Validation and Integrity Engine for CUB-200-2011
Calibrated Concept-Based Feedback Model (CSE 4385)

Provides comprehensive inspection of dataset image files:
- Checks file existence and decodability using Pillow
- Detects corrupted, truncated, or unreadable images
- Records original dimensions (width, height), format, and color mode
- Validates RGB conversion integrity
- Detects missing image references or duplicate file paths
- Generates structured CSV and JSON validation reports
"""

from csv import DictWriter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFile

# Ensure truncated images are caught rather than silently ignored
ImageFile.LOAD_TRUNCATED_IMAGES = False

logger = logging.getLogger(__name__)


class ImageValidationError(Exception):
    """Raised when an image fails critical validation checks."""
    pass


def validate_single_image(
    image_path: Path,
    image_id: int,
    expected_mode: str = "RGB",
) -> Dict[str, Any]:
    """
    Validates a single image file on disk.

    Args:
        image_path: Absolute or relative Path to image file.
        image_id: Dataset image ID.
        expected_mode: Target color mode (default: 'RGB').

    Returns:
        Dict with validation status, dimensions, mode, and error details if any.
    """
    record: Dict[str, Any] = {
        "image_id": image_id,
        "path": str(image_path),
        "exists": False,
        "valid": False,
        "width": 0,
        "height": 0,
        "channels": 0,
        "format": None,
        "mode": None,
        "rgb_convertible": False,
        "file_size_bytes": 0,
        "error_message": None,
    }

    if not image_path.exists():
        record["error_message"] = f"File not found: {image_path}"
        return record

    record["exists"] = True
    try:
        record["file_size_bytes"] = image_path.stat().st_size
    except Exception as e:
        record["error_message"] = f"Failed to get file stats: {e}"
        return record

    try:
        with Image.open(image_path) as img:
            # Verify basic structure
            record["format"] = img.format
            record["mode"] = img.mode
            record["width"] = img.width
            record["height"] = img.height

            # Run verify() to detect truncated/corrupted stream
            img.verify()

        # Re-open to test full pixel decoding and RGB conversion (verify leaves file in unreadable state)
        with Image.open(image_path) as img:
            rgb_img = img.convert(expected_mode)
            record["rgb_convertible"] = True
            record["channels"] = len(rgb_img.getbands())

            # Force pixel load into memory to catch decode-time CRC/checksum errors
            rgb_img.load()

        record["valid"] = True

    except Exception as e:
        record["error_message"] = f"Decode/Verification error: {type(e).__name__} - {e}"
        record["valid"] = False

    return record


def validate_dataset_images(
    images_dict: Dict[int, str],
    dataset_root: Union[str, Path],
    max_images: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validates all images declared in the images metadata dictionary.

    Args:
        images_dict: Dict mapping image_id -> relative file path.
        dataset_root: Root directory of CUB dataset (containing 'images/' subfolder).
        max_images: Optional limit on images inspected (useful for tests).

    Returns:
        Tuple of (all_records, failure_records).
    """
    root = Path(dataset_root)
    all_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    seen_paths: Dict[str, int] = {}
    items = list(images_dict.items())
    if max_images is not None:
        items = items[:max_images]

    for image_id, rel_path in items:
        # Check for duplicate path references
        norm_path = str(Path(rel_path)).lower()
        if norm_path in seen_paths:
            logger.warning(
                f"Duplicate image path detected: '{rel_path}' mapped to ID {image_id} "
                f"and ID {seen_paths[norm_path]}"
            )

        seen_paths[norm_path] = image_id

        # Resolve path (support both root/images/rel_path and root/rel_path)
        img_path = root / "images" / rel_path
        if not img_path.exists() and (root / rel_path).exists():
            img_path = root / rel_path

        record = validate_single_image(img_path, image_id)
        all_records.append(record)

        if not record["valid"]:
            failures.append(record)
            logger.error(f"Image validation failed for ID {image_id} ({rel_path}): {record['error_message']}")

    return all_records, failures


def save_validation_reports(
    records: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Saves validation results to CSV and JSON reports.

    Returns:
        Dict mapping artifact key to Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "image_validation_failures.csv"
    summary_json_path = out_dir / "image_validation_summary.json"

    # Save failures CSV
    fieldnames = [
        "image_id",
        "path",
        "exists",
        "valid",
        "width",
        "height",
        "channels",
        "format",
        "mode",
        "rgb_convertible",
        "file_size_bytes",
        "error_message",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fail in failures:
            writer.writerow(fail)

    # Save summary JSON
    total = len(records)
    valid_count = sum(1 for r in records if r["valid"])
    summary = {
        "total_images_checked": total,
        "valid_images": valid_count,
        "failed_images": len(failures),
        "integrity_rate_pct": round((valid_count / total * 100.0) if total > 0 else 0.0, 3),
        "failures": failures,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "failures_csv": csv_path,
        "summary_json": summary_json_path,
    }

