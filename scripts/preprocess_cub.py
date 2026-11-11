#!/usr/bin/env python3
"""
CUB-200-2011 Preprocessing and Image Verification Pipeline
Calibrated Concept-Based Feedback Model (CSE 4385)

Usage:
  python scripts/preprocess_cub.py
  python scripts/preprocess_cub.py --limit 100 # Quick smoke test on 100 images
"""

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import extract_or_load_concept_matrix
from src.data.dataset_statistics import compute_dataset_statistics, save_statistics_artifacts
from src.data.image_validation import validate_dataset_images, save_validation_reports
from src.data.metadata_parser import CUBMetadataParser
from src.data.partition_manager import PartitionManager
from src.data.preprocessing import PreprocessingConfig, ProcessedDatasetManager
from src.evaluation.concept_analysis import load_concept_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess and verify CUB-200-2011 dataset")
    parser.add_argument("--dataset-root", type=str, default="data/CUB_200_2011", help="Path to raw CUB dataset")
    parser.add_argument("--manifest-path", type=str, default="outputs/dataset_manifest.json", help="Path to split manifest")
    parser.add_argument("--concept-mapping", type=str, default="data/selected_concepts_112.json", help="112 concepts JSON")
    parser.add_argument("--output-dir", type=str, default="outputs/preprocessing", help="Directory for output statistics")
    parser.add_argument("--processed-dir", type=str, default="data/processed/CUB_200_2011", help="Processed cache directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for fast smoke check")

    args = parser.parse_args()

    dataset_root = PROJECT_ROOT / args.dataset_root
    manifest_path = PROJECT_ROOT / args.manifest_path
    concept_mapping_path = PROJECT_ROOT / args.concept_mapping
    output_dir = PROJECT_ROOT / args.output_dir
    processed_dir = PROJECT_ROOT / args.processed_dir

    print("=" * 75)
    print("      CUB-200-2011 DATASET PREPROCESSING & VERIFICATION PIPELINE       ")
    print("=" * 75)
    print(f"Dataset Root:      {dataset_root}")
    print(f"Manifest Path:     {manifest_path}")
    print(f"Concept Mapping:   {concept_mapping_path}")
    print(f"Output Directory:  {output_dir}")
    print(f"Processed Cache:   {processed_dir}")
    print("=" * 75)

    # 1. Parse Metadata
    print("\n[1/5] Parsing CUB-200-2011 metadata...")
    parser = CUBMetadataParser(dataset_root)
    meta = parser.parse_all(validate_cross_references=False)
    print(f"      Parsed {len(meta.images)} images across {len(meta.classes)} species.")

    # 2. Image Verification
    print(f"\n[2/5] Validating image integrity (decoding, dimensions, RGB conversion)...")
    records, failures = validate_dataset_images(
        images_dict=meta.images,
        dataset_root=dataset_root,
        max_images=args.limit,
    )
    val_files = save_validation_reports(records, failures, output_dir)
    print(f"      Verified {len(records)} images: {len(records) - len(failures)} VALID, {len(failures)} CORRUPTED/MISSING.")
    if failures:
        print(f"      [WARNING] Detected {len(failures)} image failures! See: {val_files['failures_csv']}")

    # 3. Partition Verification
    print("\n[3/5] Verifying partition isolation and manifest counts...")
    pm = PartitionManager(manifest_path)
    partition_counts = pm.verify_partition_counts()["actual_counts"]
    pm.verify_isolation()
    pm.generate_integrity_report(image_classes=meta.image_classes, output_path=output_dir / "partition_integrity_report.json")
    print(f"      Partitions verified: Train={partition_counts['train']}, Val={partition_counts['val']}, "
          f"Calib={partition_counts['calib']}, Test={partition_counts['test']}. 0% leakage.")

    # 4. Load Concepts & Compute Statistics
    print("\n[4/5] Computing dimension, species, and concept prevalence statistics...")
    concept_meta = load_concept_metadata(concept_mapping_path)
    concept_matrix = extract_or_load_concept_matrix(
        dataset_root=dataset_root,
        mapping_file=concept_mapping_path,
        cache_path=PROJECT_ROOT / "outputs" / "cub_112_concept_labels.npy",
        total_images=len(meta.images),
    )

    stats = compute_dataset_statistics(
        image_records=records,
        image_classes=meta.image_classes,
        concept_matrix=concept_matrix,
        concept_metadata=concept_meta,
        partition_counts=partition_counts,
    )
    stat_files = save_statistics_artifacts(stats, output_dir)
    print(f"      Saved statistics JSON: {stat_files['json']}")
    print(f"      Saved statistics CSV:  {stat_files['csv']}")
    print(f"      Saved Markdown report: {stat_files['report_md']}")

    # 5. Initialize Processed Storage Metadata
    print("\n[5/5] Initializing processed dataset cache metadata...")
    pdm = ProcessedDatasetManager(processed_dir)
    meta_file = pdm.save_processing_metadata(
        total_images=len(records),
        verified_images=len(records) - len(failures),
        split_counts=partition_counts,
    )
    print(f"      Saved processed cache metadata: {meta_file}")

    print("\n" + "=" * 75)
    print("                     PREPROCESSING COMPLETED SUCCESSFULLY              ")
    print("=" * 75)


if __name__ == "__main__":
    main()

