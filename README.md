# Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition

> **Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
> **Student:** Mostafizur Rahman (ID: 41230301942)  
> **Institution:** Northern University Bangladesh  
> **Instructor:** Ayesha Siddiqua, Senior Lecturer  
> **System Status:** Fully Implemented & Tested (64/64 Unit Tests Passed) | Training Ready on Google Colab

---

## 📖 1. Project Overview & Research Objectives

This research system addresses fine-grained bird species recognition on **CUB-200-2011** while providing human-interpretable visual explanations through concept bottlenecks:

1. **Primary Research Objective (Bird Species Recognition):**
   - Accurately recognize one of **200 fine-grained bird species** from an input image ($x \to y$).
   - Output dimension: `[batch_size, 200]` logits.
2. **Auxiliary Interpretability Objective (Concept Prediction):**
   - Simultaneously predict **112 canonical visual attributes** ($x \to c$, e.g., bill shape, wing pattern, plumage color).
   - Output dimension: `[batch_size, 112]` logits.
3. **Multi-Task Joint Objective:**
   - $\mathcal{L}_{total} = \mathcal{L}_{species} + 0.2 \cdot \mathcal{L}_{concept}$
   - Primary species classification continues smoothly even when concept predictions are imperfect.

---

## 🗂️ 2. Dataset Structure & Partition Integrity

The dataset consists of **11,788 bird images across 200 species**, with 112 filtered concepts.

| Partition | Image Count | Proportion of Official Train | Purpose | Data Augmentation |
| :--- | :---: | :---: | :--- | :---: |
| **Training** | **4,794** | 80.0% | Gradient updates | Active (Random Crop, Flip, Jitter) |
| **Validation** | **600** | 10.0% | Model selection & early stopping | None (Deterministic Center Crop) |
| **Calibration** | **600** | 10.0% | Conformal prediction & post-hoc calibration | None (Deterministic Center Crop) |
| **Official Test** | **5,794** | 100% of official test | Final benchmark evaluation (Locked) | None (Deterministic Center Crop) |
| **Total** | **11,788** | - | Complete Dataset | - |

- **Strict Isolation (0% Leakage):** Verified zero mutual overlap across all partitions via `src/data/partition_manager.py`.
- **Species Stratification:** Exactly 3 images per species in Validation ($3 \times 200 = 600$) and Calibration ($3 \times 200 = 600$).

---

## 📐 3. Preprocessing & Aspect-Ratio Preservation Policy

Avian species exhibit extreme aspect ratios ranging from 0.400 (slender, vertical wading birds) to 3.030 (wide horizontal flying postures). To avoid unnatural anatomical distortion:
1. Images are **not** naively stretched to $224 \times 224$.
2. The shorter dimension is scaled to 256 pixels while strictly preserving the natural aspect ratio.
3. A $224 \times 224$ crop (center crop for evaluation; random crop for training) is extracted.
4. Channel values are normalized using standard ImageNet parameters:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`

---

## 🧠 4. Five Recognition Architectures (Unified Dual-Head Interface)

Every architecture supports dual heads (`[B, 200]` species logits and `[B, 112]` concept logits):

| Model Name | Exp ID | Feature Dim | Key Architectural Components | Total Parameters |
| :--- | :---: | :---: | :--- | :---: |
| **ResNet-50** | **E2** | 2048 | Pretrained ResNet-50 + Dual Linear Heads | 24,147,320 |
| **ResNet-50 + CBAM** | **E3** | 2048 | ResNet-50 + Channel & Spatial Attention (CBAM) | 24,844,040 |
| **ResNet-50 + CBAM + FPN** | **E4** | 2048 | ResNet-50 + CBAM + 4-Stage Feature Pyramid Network | 30,291,720 |
| **EfficientNet-B2** | **E5** | 1408 | Compound Scaling + Squeeze-and-Excitation | 8,149,606 |
| **ConvNeXt-Tiny** | **E6** | 768 | 7x7 Depthwise Convolutions + LayerNorm | 28,145,552 |
| *ResNet-18 Baseline* | *E1* | 512 | Preserved single-head concept predictor | 11,233,968 |

---

## ⚡ 5. Google Colab GPU & Google Drive Automation

### 5.1 Google Drive Directory Structure
When mounted in Google Colab, the system automatically creates:
```
MyDrive/CalibratedBirdRecognition/
├── dataset/            # CUB_200_2011 archive and raw annotations
├── checkpoints/        # Mirrored model checkpoints by architecture
├── logs/               # Epoch-level training logs and CSV files
├── metrics/            # Validation evaluation metrics
├── predictions/        # Generated prediction sets
├── configs/            # Experiment YAML configurations
├── reports/            # Preprocessing & model comparison reports
└── experiment_registry/# Centralized JSON registry
```

### 5.2 Fast Local Staging (Eliminating Drive I/O Bottlenecks)
Reading 11,788 individual image files over Google Drive FUSE causes extreme I/O latency. The system stages the dataset archive into Colab's fast NVMe SSD (`/content/data/CUB_200_2011/`) in under 30 seconds.

### 5.3 Automated Checkpoint Recovery
- Checkpoints are saved atomically locally and mirrored to Google Drive.
- Each checkpoint contains complete recovery state: model weights, AdamW momentum buffers, cosine scheduler state, AMP scaler, epoch counter, RNG states, and early stopping state.
- If a Colab session disconnects, adding `--resume` seamlessly continues from the next unfinished epoch.

---

## 🚀 6. Step-by-Step Execution Commands

### Local Verification & Dataset Audit:
```powershell
# 1. Run all unit tests (64 passing tests)
.\venv\Scripts\pytest.exe tests/ -v

# 2. Verify dataset partition integrity and 0% leakage
.\venv\Scripts\python.exe scripts/verify_preprocessed_dataset.py

# 3. Inspect compute hardware & storage environment
.\venv\Scripts\python.exe scripts/inspect_colab_environment.py

# 4. Run ResNet-50 local smoke test (2 batches, safe verification)
.\venv\Scripts\python.exe scripts/run_experiment.py --model resnet50 --smoke-test
```

### Google Colab GPU Execution:
1. Open `notebooks/colab_training_pipeline.ipynb` in Google Colab.
2. Select **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
3. Execute the automated notebook cells to stage the dataset and train models.

---

## 🔒 7. Strict Evaluation & Calibration Boundaries

- **Calibration Images (600 images):** Strictly held out from training, validation, and model selection. Reserved exclusively for post-hoc calibration and split conformal prediction.
- **Official Test Set (5,794 images):** Locked. Accessing this split requires explicit separate authorization and is reserved solely for final testing of the single selected model.
