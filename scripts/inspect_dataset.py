#!/usr/bin/env python3
"""
Comprehensive CUB-200-2011 Dataset & Concept Audit Tool
Calibrated Concept-Based Feedback Model (CSE 4385)

Inspects the dataset directory, parses metadata annotations, checks partition
integrity, evaluates concept selection rules, and reports status cleanly.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, ConfigError
from src.data.metadata_parser import (
    CUBMetadataParser,
    CUBMetadata,
    DatasetNotFoundError,
    CorruptAnnotationError,
)
from src.data.partitions import (
    create_deterministic_splits,
    load_partition_manifest,
    PartitionManifest,
)
from src.data.concept_audit import (
    ConceptAuditor,
    load_concept_mapping,
)

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_section(title: str):
    print(f"\n{BLUE}{'=' * 68}{RESET}")
    print(f"{BLUE}{title:^68}{RESET}")
    print(f"{BLUE}{'=' * 68}{RESET}")


def run_dataset_audit():
    print_section("CUB-200-2011 Dataset Audit (Week 2)")

    # 1. Load Configuration
    try:
        config = load_config()
        dataset_rel = config.get("data", {}).get("dataset_root", "data/CUB_200_2011")
        concept_file_rel = config.get("paths", {}).get("attribute_mapping_file", "data/selected_concepts_112.json")
    except ConfigError as e:
        print(f"{RED}[CONFIG ERROR]{RESET} Could not load configuration: {e}")
        return

    dataset_path = PROJECT_ROOT / dataset_rel
    concept_mapping_path = PROJECT_ROOT / concept_file_rel

    print(f"Dataset Root Directory:   {dataset_path}")
    print(f"Concept Mapping File:     {concept_mapping_path}")

    # 2. Check Directory Existence
    if not dataset_path.exists():
        print_section("Dataset Status: NOT EXTRACTED")
        print(f"{YELLOW}[NOTICE]{RESET} The CUB-200-2011 dataset folder does not exist at this location.")
        print("This is completely expected before manual dataset download.")
        print("\nTo set up the dataset:")
        print("  1. Review 'data/README.md' for download instructions.")
        print("  2. Download 'CUB_200_2011.tgz' from Caltech's official vision repository.")
        print(f"  3. Extract archive to '{dataset_path}'")
        print("  4. Re-run this inspection script.\n")
        return

    # 3. Check and Parse Annotation Files
    print_section("Checking Dataset Files & Parsing Annotations")
    parser = CUBMetadataParser(dataset_path)

    try:
        metadata: CUBMetadata = parser.parse_all(validate_cross_references=True)
        print(f"  {GREEN}[OK]{RESET} All required annotation files present and cross-validated!")
        print(f"       - Total Images:             {metadata.total_images:,}")
        print(f"       - Total Species Classes:     {metadata.total_classes}")
        print(f"       - Official Train Images:     {len(metadata.train_image_ids):,}")
        print(f"       - Official Test Images:      {len(metadata.test_image_ids):,} (Frozen & Locked)")
        print(f"       - Total Raw Attributes:      {metadata.total_attributes}")
    except DatasetNotFoundError as e:
        print(f"  {RED}[ERROR]{RESET} Missing annotation file: {e}")
        return
    except CorruptAnnotationError as e:
        print(f"  {RED}[ERROR]{RESET} Corrupt or malformed annotation record: {e}")
        return

    # 4. Check Partition Manifests
    print_section("Data Partitions Status (Train / Val / Calib / Test)")
    partition_dir = dataset_path / "splits"
    if (partition_dir / "partition_summary.json").exists():
        try:
            manifest = load_partition_manifest(partition_dir)
            manifest.validate_isolation(set(metadata.train_image_ids), set(metadata.test_image_ids))
            print(f"  {GREEN}[OK]{RESET} Saved partition manifests verified at: {partition_dir}")
            print(f"       - Training Split (~80%):    {manifest.train_count:,} images")
            print(f"       - Validation Split (~10%):  {manifest.val_count:,} images (Model selection & temperature)")
            print(f"       - Calibration Split (~10%): {manifest.calib_count:,} images (Conformal quantiles)")
            print(f"       - Official Test Split:      {manifest.test_count:,} images (Locked)")
        except Exception as e:
            print(f"  {RED}[WARNING]{RESET} Partition validation error: {e}")
    else:
        print(f"  {YELLOW}[INFO]{RESET} Partition manifest not yet saved on disk.")
        print("       Generating in-memory preview using seed 42...")
        seed = config.get("reproducibility", {}).get("seed", 42)
        preview_manifest = create_deterministic_splits(
            metadata,
            train_ratio=config["data"]["train_split_ratio"],
            val_ratio=config["data"]["val_split_ratio"],
            calib_ratio=config["data"]["calib_split_ratio"],
            seed=seed,
            stratify=True,
        )
        print(f"       - Deterministic Train Split: {preview_manifest.train_count:,} images")
        print(f"       - Deterministic Val Split:   {preview_manifest.val_count:,} images")
        print(f"       - Deterministic Calib Split: {preview_manifest.calib_count:,} images")
        print(f"       - Official Test Split:       {preview_manifest.test_count:,} images (Untouched)")

    # 5. Concept Selection & Mapping Status
    print_section("Concept Selection Status (112 Concepts Target)")
    if concept_mapping_path.exists():
        try:
            mapping = load_concept_mapping(concept_mapping_path)
            num_c = len(mapping.get("concepts", []))
            print(f"  {GREEN}[OK]{RESET} Found concept mapping at: {concept_mapping_path}")
            print(f"       - Number of Concepts:       {num_c}")
            if num_c == 112:
                print(f"       - Target Alignment:         EXACT MATCH (112 concepts)")
            else:
                print(f"       - Target Alignment:         MISMATCH ({num_c} != 112)")
        except Exception as e:
            print(f"  {RED}[ERROR]{RESET} Error reading concept mapping file: {e}")
    else:
        print(f"  {YELLOW}[PENDING APPROVAL]{RESET} '{concept_mapping_path}' has not been generated yet.")
        print("       Proposal requires 112 concepts but does not specify an explicit numerical rule.")
        print("       Running audit with candidate filtering rules...")

        auditor = ConceptAuditor(metadata)
        rules_to_test = [
            ("class_majority_50", 10, "Present in >= 10 species where >= 50% of images exhibit attribute"),
            ("class_any_positive", 10, "Present in >= 10 species with at least 1 positive image"),
            ("class_min_images_2", 10, "Present in >= 10 species with >= 2 positive images"),
        ]

        print("\n  Candidate Rule Evaluation:")
        for rule_name, min_cls, desc in rules_to_test:
            try:
                rep = auditor.audit_attributes(counting_rule=rule_name, min_classes=min_cls)
                match_str = f"{GREEN}EXACT MATCH (112){RESET}" if rep.qualifying_count == 112 else f"{YELLOW}{rep.qualifying_count} concepts (Discrepancy: {rep.discrepancy_from_target:+d}){RESET}"
                print(f"    • {rule_name:<20} -> {match_str}")
                print(f"      Rule: {desc}")
            except Exception as e:
                print(f"    • {rule_name:<20} -> Audit failed: {e}")

        print("\n  [DECISION NEEDED] Review candidate rules and approve the exact rule before saving mapping.")

    print(f"\n{BLUE}{'=' * 68}{RESET}\n")


if __name__ == "__main__":
    run_dataset_audit()
