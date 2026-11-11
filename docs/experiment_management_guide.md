# Experiment Management & Tracking Guide

**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition (CSE 4385)

---

## 1. Centralized Experiment Registry

All experiments are registered and tracked in `outputs/experiment_registry/`:
- `outputs/experiment_registry/registry.json`: Detailed machine-readable JSON history.
- `outputs/experiment_registry/experiments_summary.csv`: Tabular overview for rapid comparison.

### Recorded Fields:
- `experiment_id`: Unique identifier (e.g., `resnet50_cub200`)
- `model_name`: Architecture identifier (`resnet50`, `resnet50_cbam`, `resnet50_cbam_fpn`, `efficientnet_b2`, `convnext_tiny`)
- `status`: `RUNNING` or `COMPLETED`
- `completed_epochs`: Number of completed epochs
- `best_epoch`: Epoch index that achieved highest species Top-1 accuracy
- `best_val_species_top1`: Top-1 accuracy achieved on validation set
- `best_val_concept_f1`: Unweighted macro-F1 achieved on 112 concepts
- `best_val_total_loss`: Total validation loss
- `total_params`: Parameter footprint

---

## 2. Standardized Experiment CLI Commands

| Operational Goal | Exact Command Line |
| :--- | :--- |
| **Verify Dataset & Partitions** | `python scripts/verify_preprocessed_dataset.py` |
| **Inspect Hardware Environment** | `python scripts/inspect_colab_environment.py` |
| **Run ResNet-50 Smoke Test** | `python scripts/run_experiment.py --model resnet50 --smoke-test` |
| **Run ResNet-50 + CBAM Smoke Test** | `python scripts/run_experiment.py --model resnet50_cbam --smoke-test` |
| **Run ResNet-50 + CBAM + FPN Smoke Test** | `python scripts/run_experiment.py --model resnet50_cbam_fpn --smoke-test` |
| **Run EfficientNet-B2 Smoke Test** | `python scripts/run_experiment.py --model efficientnet_b2 --smoke-test` |
| **Run ConvNeXt-Tiny Smoke Test** | `python scripts/run_experiment.py --model convnext_tiny --smoke-test` |
| **Resume Interrupted Run** | `python scripts/run_experiment.py --model resnet50 --resume` |

