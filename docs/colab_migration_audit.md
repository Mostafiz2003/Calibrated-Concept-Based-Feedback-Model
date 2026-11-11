# Google Colab & Google Drive Migration Audit

**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition (CSE 4385)  
**Date:** November 10, 2026  
**Auditor:** Mostafizur Rahman  
**Status:** Architecture & Cloud Infrastructure Migration Verified

---

## 1. Migration Overview

To train fine-grained recognition models (ResNet-50, CBAM, FPN, EfficientNet-B2, ConvNeXt-Tiny) efficiently without local GPU hardware on Windows 11, the project has been upgraded for seamless execution on **Google Colab Free (T4 GPU)** with persistent artifact mirroring to **Google Drive**.

---

## 2. Solved Bottlenecks & Architectural Upgrades

| Challenge / Risk | Prior Local Status | Google Colab & Drive Solution | Verification Status |
| :--- | :--- | :--- | :---: |
| **GPU Compute Access** | Windows local environment is pure CPU (`2.15.0+cpu`). Multi-epoch training took ~35 min/epoch. | Integrated Google Colab Free T4 GPU (16 GB VRAM) with Automatic Mixed Precision (`torch.cuda.amp`). | **Verified** |
| **Google Drive I/O Bottleneck** | Reading 11,788 individual image files over Drive FUSE filesystem causes severe I/O throttling and timeouts. | Implemented `DatasetStagingManager` (`src/cloud/dataset_sync.py`), which stages the dataset archive onto Colab's local fast NVMe SSD (`/content/data/CUB_200_2011`). | **Verified** |
| **Colab Session Disconnects** | Unplanned session timeouts risk losing hours of training progress. | Implemented `CloudCheckpointManager` (`src/cloud/cloud_checkpoint_manager.py`), saving checkpoints locally and atomically mirroring to Google Drive with automated resumption. | **Verified** |
| **Accidental CPU Training** | Risk of mistakenly running full training on Colab if GPU is not assigned. | Implemented `assert_gpu_available()` (`src/cloud/colab_environment.py`), which halts full training launches unless a GPU is detected or explicitly bypassed. | **Verified** |
| **Partition Leakage** | Danger of training on validation, calibration, or test splits. | Strict partition guards enforced via `PartitionManager`. Only 4,794 training images are loaded into gradient update pipelines. | **Verified** |

---

## 3. Storage Hierarchy on Google Drive

When mounted in Colab, the project automatically establishes the following directory structure under `MyDrive/CalibratedBirdRecognition/`:

```
MyDrive/CalibratedBirdRecognition/
├── dataset/
│   ├── CUB_200_2011/
│   ├── processed/
│   ├── partitions/
│   └── concept_mapping/
├── checkpoints/
│   ├── resnet50/
│   ├── resnet50_cbam/
│   ├── resnet50_cbam_fpn/
│   ├── efficientnet_b2/
│   └── convnext_tiny/
├── logs/
├── metrics/
├── predictions/
├── configs/
├── reports/
└── experiment_registry/
```

---

## 4. Hardware Diagnostic Checks

The diagnostics module (`scripts/inspect_colab_environment.py`) verifies:
- GPU device name and compute capability
- Total, reserved, and allocated VRAM
- PyTorch, Torchvision, and CUDA runtime compatibility
- Dynamic batch-size recommendations (32 for $\ge 15$ GB VRAM, 16 for $\ge 8$ GB VRAM)
- Multi-worker DataLoader optimization (2 workers on Linux/Colab, 0 on Windows)

