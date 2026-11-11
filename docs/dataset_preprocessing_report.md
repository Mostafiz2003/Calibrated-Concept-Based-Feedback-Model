# Dataset Preprocessing & Partition Integrity Report

**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition (CSE 4385)  
**Dataset:** Caltech-UCSD Birds-200-2011 (CUB-200-2011)  
**Date:** November 10, 2026  
**Status:** 100% Image Integrity Verified (11,788/11,788 valid) | Zero Partition Leakage

---

## 1. Executive Summary

A comprehensive, non-destructive audit and preprocessing pipeline was executed across the physical CUB-200-2011 dataset. Every image was decoded with Pillow, tested for truncation/corruption, verified for RGB conversion, and checked against the official annotations.

### Key Milestones:
- **Zero Missing / Zero Corrupted Images:** All **11,788 images** exist, decode cleanly, and convert to 3-channel RGB without failure.
- **Aspect-Ratio Preservation Policy:** Rather than naively distorting rectangular avian anatomy to $224 \times 224$, the shorter dimension is scaled to 256 pixels, followed by a $224 \times 224$ crop (center crop for evaluation; random crop for training).
- **Strict Partition Isolation (0% Leakage):** The 4 partitions defined in `outputs/dataset_manifest.json` have 0 mutual overlap. The official test partition (5,794 images) remains strictly locked.

---

## 2. Dataset Partitioning & Stratification

| Partition | Images | Proportion of Official Train | Purpose | Data Augmentation |
| :--- | :---: | :---: | :--- | :---: |
| **Train** | **4,794** | 80.0% | Gradient updates | Active (Flip, Crop, Jitter) |
| **Validation** | **600** | 10.0% | Model selection & checkpointing | None (Deterministic) |
| **Calibration** | **600** | 10.0% | Conformal prediction & post-hoc calibration | None (Deterministic) |
| **Official Test** | **5,794** | 100% of official test | Final benchmark evaluation (Locked) | None (Deterministic) |
| **Total** | **11,788** | - | Complete Dataset | - |

- **Species Stratification:** Exactly 3 images per species are allocated to Validation ($3 \times 200 = 600$) and Calibration ($3 \times 200 = 600$). All 200 species are present in every partition.

---

## 3. Physical Image Dimensions & Aspect Ratios

| Property | Width (px) | Height (px) | Aspect Ratio (Width / Height) |
| :--- | :---: | :---: | :---: |
| **Minimum** | 120 | 120 | 0.400 |
| **Maximum** | 500 | 500 | 3.030 |
| **Mean** | 465.1 | 385.7 | 1.304 |
| **Median** | 500.0 | 375.0 | 1.333 |

The aspect ratio ranges from 0.400 (tall vertical birds, e.g. wading herons) to 3.030 (wide horizontal flying postures). Scaling the shorter side to 256 and cropping to 224 preserves natural plumage patterns and bill proportions.

---

## 4. Concept Prevalence & Imbalance Analysis (112 Concepts)

- **Number of Auxiliary Concepts:** 112
- **Prevalence Range:** 0.0050 to 0.8383
- **Mean Concept Prevalence:** 0.1869 (~18.7% positive annotations)
- **Median Concept Prevalence:** 0.1617
- **Multi-Task Objective:** The high class sparsity ($>80\%$ negative class labels across attributes) justifies balancing the primary species cross-entropy loss with an auxiliary concept binary cross-entropy loss weighted at $\lambda = 0.2$:
  $$\mathcal{L}_{total} = \mathcal{L}_{species} + 0.2 \cdot \mathcal{L}_{concept}$$

---

## 5. Artifacts and Output Files

- `outputs/preprocessing/partition_integrity_report.json`: JSON record verifying 0% pairwise overlap.
- `outputs/preprocessing/dataset_statistics.json`: Machine-readable dataset dimension, species, and concept distributions.
- `outputs/preprocessing/dataset_statistics.csv`: Table of positive/negative counts and prevalence for all 112 concepts.
- `outputs/preprocessing/preprocessing_report.md`: Markdown summary.
- `data/processed/CUB_200_2011/processing_metadata.json`: Cache metadata with configuration parameters.

