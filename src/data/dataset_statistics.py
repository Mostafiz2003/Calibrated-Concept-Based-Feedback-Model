"""
Comprehensive Dataset Statistics and Quality Metrics for CUB-200-2011
Calibrated Concept-Based Feedback Model (CSE 4385)

Generates:
- Dataset dimension distributions (widths, heights, aspect ratios)
- Species distribution and class balance metrics across 200 species
- Concept positive/negative support and prevalence across all 112 concepts
- Partition summary (Train: 4,794, Val: 600, Calib: 600, Test: 5,794)
- Outputs outputs/preprocessing/dataset_statistics.json, .csv, and preprocessing_report.md
"""

from csv import DictWriter
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def compute_dataset_statistics(
    image_records: List[Dict[str, Any]],
    image_classes: Dict[int, int],
    concept_matrix: np.ndarray,
    concept_metadata: List[Dict[str, Any]],
    partition_counts: Dict[str, int],
) -> Dict[str, Any]:
    """
    Computes comprehensive structural and statistical metrics for CUB-200-2011.

    Args:
        image_records: List of validation records from validate_dataset_images.
        image_classes: Mapping of image_id -> 1-based class_id.
        concept_matrix: Concept label matrix of shape [total_images + 1, 112].
        concept_metadata: 112-concept metadata list.
        partition_counts: Dict of split counts.

    Returns:
        Structured statistics dictionary.
    """
    total_images = len(image_records)
    valid_images = [r for r in image_records if r.get("valid", False)]
    failed_images = [r for r in image_records if not r.get("valid", False)]

    # 1. Dimension distributions
    widths = [r["width"] for r in valid_images if r["width"] > 0]
    heights = [r["height"] for r in valid_images if r["height"] > 0]
    aspect_ratios = [w / h for w, h in zip(widths, heights) if h > 0]

    dim_stats = {
        "width_min": int(np.min(widths)) if widths else 0,
        "width_max": int(np.max(widths)) if widths else 0,
        "width_mean": float(np.mean(widths)) if widths else 0.0,
        "width_median": float(np.median(widths)) if widths else 0.0,
        "height_min": int(np.min(heights)) if heights else 0,
        "height_max": int(np.max(heights)) if heights else 0,
        "height_mean": float(np.mean(heights)) if heights else 0.0,
        "height_median": float(np.median(heights)) if heights else 0.0,
        "aspect_ratio_min": float(np.min(aspect_ratios)) if aspect_ratios else 0.0,
        "aspect_ratio_max": float(np.max(aspect_ratios)) if aspect_ratios else 0.0,
        "aspect_ratio_mean": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
        "aspect_ratio_median": float(np.median(aspect_ratios)) if aspect_ratios else 0.0,
    }

    # 2. Species distribution
    species_counts: Dict[int, int] = {}
    for iid, cid in image_classes.items():
        species_counts[cid] = species_counts.get(cid, 0) + 1

    counts_list = list(species_counts.values())
    species_stats = {
        "num_species": len(species_counts),
        "min_images_per_species": int(np.min(counts_list)) if counts_list else 0,
        "max_images_per_species": int(np.max(counts_list)) if counts_list else 0,
        "mean_images_per_species": float(np.mean(counts_list)) if counts_list else 0.0,
        "median_images_per_species": float(np.median(counts_list)) if counts_list else 0.0,
    }

    # 3. Concept statistics across the full dataset
    concept_stats_list: List[Dict[str, Any]] = []
    # Note: image IDs are 1-based, matrix has total_images + 1 rows
    active_concept_matrix = concept_matrix[1:total_images + 1, :]

    num_concepts = active_concept_matrix.shape[1]
    prevalences = []

    for j in range(num_concepts):
        col = active_concept_matrix[:, j]
        pos_count = int(np.sum(col == 1.0))
        neg_count = total_images - pos_count
        prev = float(pos_count / total_images) if total_images > 0 else 0.0
        prevalences.append(prev)

        attr_name = concept_metadata[j]["attribute_name"] if j < len(concept_metadata) else f"concept_{j}"
        attr_id = concept_metadata[j]["cub_attribute_id"] if j < len(concept_metadata) else (j + 1)

        concept_stats_list.append({
            "concept_index": j,
            "cub_attribute_id": attr_id,
            "attribute_name": attr_name,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "prevalence": round(prev, 6),
            "imbalance_ratio": round((neg_count / pos_count) if pos_count > 0 else float("inf"), 2),
        })

    concept_summary = {
        "num_concepts": num_concepts,
        "min_prevalence": float(np.min(prevalences)) if prevalences else 0.0,
        "max_prevalence": float(np.max(prevalences)) if prevalences else 0.0,
        "mean_prevalence": float(np.mean(prevalences)) if prevalences else 0.0,
        "median_prevalence": float(np.median(prevalences)) if prevalences else 0.0,
    }

    return {
        "dataset_name": "CUB-200-2011",
        "total_images": total_images,
        "valid_images": len(valid_images),
        "failed_images": len(failed_images),
        "partition_counts": partition_counts,
        "dimension_statistics": dim_stats,
        "species_statistics": species_stats,
        "concept_summary": concept_summary,
        "per_concept_statistics": concept_stats_list,
    }


def save_statistics_artifacts(
    stats: Dict[str, Any],
    output_dir: Union[str, Path] = "outputs/preprocessing",
) -> Dict[str, Path]:
    """
    Saves:
      1. outputs/preprocessing/dataset_statistics.json
      2. outputs/preprocessing/dataset_statistics.csv (per-concept breakdown)
      3. outputs/preprocessing/preprocessing_report.md
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataset_statistics.json"
    csv_path = out_dir / "dataset_statistics.csv"
    report_path = out_dir / "preprocessing_report.md"

    # 1. JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # 2. CSV
    per_concept = stats.get("per_concept_statistics", [])
    if per_concept:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = DictWriter(f, fieldnames=list(per_concept[0].keys()))
            writer.writeheader()
            for row in per_concept:
                writer.writerow(row)

    # 3. Markdown Report
    dim = stats["dimension_statistics"]
    sp = stats["species_statistics"]
    cs = stats["concept_summary"]
    pc = stats["partition_counts"]

    md_content = f"""# CUB-200-2011 Dataset Preprocessing & Statistics Report

**Project:** Calibrated Concept-Based Feedback Model (CSE 4385)  
**Dataset:** Caltech-UCSD Birds-200-2011 (CUB-200-2011)  
**Total Images Verified:** {stats['total_images']} ({stats['valid_images']} valid, {stats['failed_images']} corrupted/missing)

---

## 1. Dataset Partitioning

| Partition | Image Count | Proportion of Official Train | Purpose |
| :--- | :---: | :---: | :--- |
| **Training** | **{pc.get('train', 4794)}** | 80.0% | Gradient updates (Data augmentation active) |
| **Validation** | **{pc.get('val', 600)}** | 10.0% | Model selection & checkpointing (Deterministic) |
| **Calibration** | **{pc.get('calib', 600)}** | 10.0% | Conformal prediction & post-hoc calibration (Locked) |
| **Official Test** | **{pc.get('test', 5794)}** | N/A (100% of official test) | Final evaluation only (Strictly isolated) |
| **Total** | **{stats['total_images']}** | - | Complete CUB-200-2011 dataset |

---

## 2. Image Dimension & Aspect-Ratio Analysis

| Metric | Width (px) | Height (px) | Aspect Ratio (W/H) |
| :--- | :---: | :---: | :---: |
| **Minimum** | {dim['width_min']} | {dim['height_min']} | {dim['aspect_ratio_min']:.3f} |
| **Maximum** | {dim['width_max']} | {dim['height_max']} | {dim['aspect_ratio_max']:.3f} |
| **Mean** | {dim['width_mean']:.1f} | {dim['height_mean']:.1f} | {dim['aspect_ratio_mean']:.3f} |
| **Median** | {dim['width_median']:.1f} | {dim['height_median']:.1f} | {dim['aspect_ratio_median']:.3f} |

### Aspect Ratio Handling Policy:
To prevent unnatural geometrical distortion of avian body shapes, raw images are **not** naively stretched to 224x224. Instead:
1. The shorter image dimension is scaled to 256 pixels while strictly preserving the natural aspect ratio.
2. A centered 224x224 crop (deterministic evaluation) or random 224x224 crop (training) is extracted.

---

## 3. Species Distribution (200 Bird Species)

- **Total Classes:** {sp['num_species']} species
- **Images per Species:** Min: {sp['min_images_per_species']}, Max: {sp['max_images_per_species']}, Mean: {sp['mean_images_per_species']:.1f}, Median: {sp['median_images_per_species']:.1f}
- **Stratification:** Exactly 3 images per species allocated to Validation (600 total) and Calibration (600 total).

---

## 4. Concept Prevalence & Imbalance Analysis (112 Concepts)

- **Number of Auxiliary Concepts:** {cs['num_concepts']}
- **Prevalence Range:** [{cs['min_prevalence']:.4f}, {cs['max_prevalence']:.4f}]
- **Mean Prevalence:** {cs['mean_prevalence']:.4f} (~{cs['mean_prevalence']*100:.1f}% positive labels)
- **Median Prevalence:** {cs['median_prevalence']:.4f}
- **Imbalance Signature:** The vast majority of concepts exhibit over 80% negative class labels, necessitating balanced multi-task loss weighting (L_total = L_species + 0.2 * L_concept) and macro-F1 tracking.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {
        "json": json_path,
        "csv": csv_path,
        "report_md": report_path,
    }
