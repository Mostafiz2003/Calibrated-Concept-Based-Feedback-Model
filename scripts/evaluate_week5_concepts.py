#!/usr/bin/env python3
"""
Week 5 Baseline Evaluation Script: ResNet-18 Per-Concept Analysis
Calibrated Concept-Based Feedback Model (CSE 4385)

Evaluates the trained ResNet-18 concept predictor on the validation split ONLY (600 images).
Computes:
  - Per-concept metrics across all 112 concepts (Precision, Recall, F1, Support, Prevalence, Accuracy)
  - Aggregate metrics (Macro/Micro Precision, Recall, F1, Mean Accuracy, BCE Loss)
  - Top 10 and Bottom 10 concepts by F1
  - Diagnostic zero-support and zero-predicted counts
  - Class imbalance analysis

Safety Guarantees:
  - Read-only evaluation (torch.no_grad(), eval mode).
  - Validation split only (strictly isolates calibration and official test splits).
  - Preserves existing checkpoints and manifests.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import create_cub_dataloaders
from src.models.concept_model import build_concept_predictor
from src.evaluation.concept_analysis import (
    load_concept_metadata,
    collect_validation_predictions,
    compute_per_concept_breakdown,
    compute_aggregate_metrics,
    save_week5_analysis_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ResNet-18 Per-Concept Baseline on Validation Split")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/concept_resnet18_3epochs/best_concept_model.pth",
        help="Path to trained concept model checkpoint",
    )
    parser.add_argument(
        "--concept-mapping",
        type=str,
        default="data/selected_concepts_112.json",
        help="Path to 112-concept mapping JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/week5_analysis",
        help="Directory to save analysis artifacts",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for validation DataLoader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu, cuda)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Primary binary decision threshold (default: 0.5)",
    )

    args = parser.parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    mapping_path = PROJECT_ROOT / args.concept_mapping
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 75)
    print("      WEEK 5 EVALUATION: RESNET-18 PER-CONCEPT BASELINE ANALYSIS      ")
    print("=" * 75)
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Concept Mapping:  {mapping_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Threshold:        {args.threshold} (Documented primary threshold; no val tuning)")
    print(f"Device:           {args.device}")
    print("=" * 75)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load Concept Metadata
    print("\n[1/5] Loading 112 concept metadata...")
    concept_metadata = load_concept_metadata(mapping_path)
    print(f"      Loaded {len(concept_metadata)} concept definitions.")

    # 2. Build Model and Load Trained Weights
    print("\n[2/5] Initializing ResNet-18 and loading weights from checkpoint...")
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone_name = checkpoint_data.get("backbone_name", "resnet18")
    num_concepts = checkpoint_data.get("num_concepts", 112)

    model = build_concept_predictor(
        backbone_name=backbone_name,
        num_concepts=num_concepts,
        pretrained=False,  # Weights loaded directly from checkpoint
    )
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    print(f"      Successfully restored weights from epoch {checkpoint_data.get('epoch', 'unknown')} "
          f"(Best Val Loss: {checkpoint_data.get('best_val_loss', 'N/A'):.4f})")

    # 3. Create Validation DataLoader ONLY
    print("\n[3/5] Instantiating validation DataLoader (600 images)...")
    config = load_config()
    loaders = create_cub_dataloaders(
        config=config,
        batch_size=args.batch_size,
        splits=("val",),  # Strictly validation partition ONLY
        allow_test_split=False,
    )
    val_loader = loaders["val"]
    val_count = len(val_loader.dataset)
    print(f"      Validation DataLoader: {len(val_loader)} batches ({val_count} images)")
    if val_count != 600:
        raise ValueError(f"Expected exactly 600 validation images, but got {val_count}!")

    # 4. Collect Model Predictions
    print("\n[4/5] Running forward inference on validation split...")
    eval_results = collect_validation_predictions(
        model=model,
        data_loader=val_loader,
        device=args.device,
    )
    print(f"      Completed inference on {eval_results['num_samples']} images.")
    print(f"      Validation BCE Loss: {eval_results['loss']:.4f}")

    # 5. Compute Per-Concept and Aggregate Metrics
    print("\n[5/5] Computing per-concept breakdown and aggregate statistics...")
    per_concept_metrics = compute_per_concept_breakdown(
        targets=eval_results["targets"],
        probabilities=eval_results["probabilities"],
        threshold=args.threshold,
        concept_metadata=concept_metadata,
    )

    aggregate_metrics = compute_aggregate_metrics(
        targets=eval_results["targets"],
        probabilities=eval_results["probabilities"],
        loss_val=eval_results["loss"],
        threshold=args.threshold,
        per_concept_metrics=per_concept_metrics,
    )

    # 6. Save Artifacts
    saved_files = save_week5_analysis_artifacts(
        per_concept_metrics=per_concept_metrics,
        aggregate_metrics=aggregate_metrics,
        output_dir=output_dir,
    )

    # 7. Print Console Summary
    print("\n" + "=" * 75)
    print("                     EVALUATION RESULTS SUMMARY                        ")
    print("=" * 75)
    print(f"Validation BCE Loss:    {aggregate_metrics['loss']:.4f}")
    print(f"Macro F1:               {aggregate_metrics['macro_f1']:.4f} ({aggregate_metrics['macro_f1']*100:.2f}%)")
    print(f"Macro Precision:        {aggregate_metrics['macro_precision']:.4f} ({aggregate_metrics['macro_precision']*100:.2f}%)")
    print(f"Macro Recall:           {aggregate_metrics['macro_recall']:.4f} ({aggregate_metrics['macro_recall']*100:.2f}%)")
    print(f"Micro F1:               {aggregate_metrics['micro_f1']:.4f} ({aggregate_metrics['micro_f1']*100:.2f}%)")
    print(f"Micro Precision:        {aggregate_metrics['micro_precision']:.4f} ({aggregate_metrics['micro_precision']*100:.2f}%)")
    print(f"Micro Recall:           {aggregate_metrics['micro_recall']:.4f} ({aggregate_metrics['micro_recall']*100:.2f}%)")
    print(f"Mean Concept Accuracy:  {aggregate_metrics['mean_concept_accuracy']:.4f} ({aggregate_metrics['mean_concept_accuracy']*100:.2f}%)")
    print(f"Exact Match Ratio:      {aggregate_metrics['exact_match_ratio']:.4f} ({aggregate_metrics['exact_match_ratio']*100:.2f}%)")
    print("-" * 75)
    print(f"Zero Support Concepts:       {aggregate_metrics['zero_support_count']}")
    print(f"Zero Predicted Positives:    {aggregate_metrics['zero_predicted_positives_count']}")
    print(f"Prevalence Range:            [{aggregate_metrics['prevalence_summary']['min_prevalence']:.4f}, {aggregate_metrics['prevalence_summary']['max_prevalence']:.4f}] (Mean: {aggregate_metrics['prevalence_summary']['mean_prevalence']:.4f})")
    print("=" * 75)

    sorted_by_f1 = sorted(per_concept_metrics, key=lambda c: c["f1_score"], reverse=True)
    print("\nTop 5 Concepts by F1:")
    for rank, c in enumerate(sorted_by_f1[:5], 1):
        print(f"  {rank}. [{c['concept_index']}] {c['attribute_name']} - F1: {c['f1_score']:.4f} (Prec: {c['precision']:.4f}, Rec: {c['recall']:.4f}, Support: {c['support']})")

    print("\nBottom 5 Concepts by F1:")
    for rank, c in enumerate(sorted_by_f1[-5:], 1):
        print(f"  {rank}. [{c['concept_index']}] {c['attribute_name']} - F1: {c['f1_score']:.4f} (Prec: {c['precision']:.4f}, Rec: {c['recall']:.4f}, Support: {c['support']})")

    print("\nArtifacts Saved Successfully:")
    for name, p in saved_files.items():
        print(f"  - {name.upper()}: {p}")


if __name__ == "__main__":
    main()

