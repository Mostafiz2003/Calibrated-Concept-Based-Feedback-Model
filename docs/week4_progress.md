# Week 4 Progress Report: Training Pipeline and First Concept Predictor Training

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Project:** Calibrated Concept-Based Feedback Model  
**Student:** Mostafizur Rahman (41230301942)  
**Date:** October 2026  
**Status:** Training Pipeline Implemented, Smoke-Tested, and Verified; 35 Unit Tests Passing; Hardware Runtime Analysis Complete

---

## 1. Executive Summary

During Week 4, the complete training pipeline for the 112-concept predictor was implemented, smoke-tested on the real CUB-200-2011 dataset, and validated across a comprehensive suite of 35 unit tests.

### Key Milestones Achieved:
1. **Hardware & Environment Inspection:** Audited the execution environment, confirming PyTorch CPU mode (`2.15.0+cpu`), 8 logical CPU cores, 7.74 GB RAM (0.81 GB available), and genuine local availability of pretrained ResNet-18 (44.66 MB) and ResNet-34 (83.27 MB) weights.
2. **Partition & Isolation Verification:** Confirmed 100% mutual exclusivity and complete test set isolation ($5,794$ images locked).
3. **Modular Training Pipeline (`src/training/`):**
   - Implemented `src/training/metrics.py` computing `BCEWithLogitsLoss`, macro-F1 across 112 concepts, binary mean accuracy, macro precision, macro recall, and exact match ratio.
   - Implemented `src/training/trainer.py` (`ConceptTrainer`) managing forward/backward passes, gradient tracking, optimizer steps, evaluation mode switches, epoch-level logging (`.csv` and `.json`), and interruption handling.
   - Implemented `scripts/train_concept_model.py` for command-line training and smoke-testing.
4. **Smoke Test Execution:** Executed on the real CUB dataset (2 training batches, 1 validation batch). Verified finite loss, gradient updates, optimizer step, output shape $[B, 112]$, and checkpoint serialization.
5. **Test Suite Expansion:** Added `tests/test_training.py`. All **35 unit tests** pass with 100% success.
6. **Runtime Projections on CPU:** Measured CPU throughput of ~3.5 seconds per batch (~10 minutes per epoch). In accordance with prompt instructions (*"If CPU-only training would be impractically slow, estimate the expected runtime and ask me before starting a long run"*), runtime estimates are documented below for user approval.

---

## 2. Hardware Profile & Pretrained Weights Audit

| Attribute | Measured Value | Operational Implication |
| :--- | :---: | :--- |
| **Python Version** | 3.13.3 | Fully compatible |
| **OS** | Windows 11 (x64) | PowerShell environment |
| **PyTorch / Torchvision** | 2.15.0+cpu / 0.29.2+cpu | CPU-only acceleration (no CUDA GPU detected) |
| **CUDA Available** | `False` | Computations execute on CPU |
| **CPU Cores** | 8 logical cores | Multi-threaded CPU execution enabled |
| **Total / Available RAM** | 7.74 GB / 0.81 GB | Low available headroom (~810 MB); requires compact batch size (32) |
| **Pretrained ResNet-18** | Present (`resnet18-f37072fd.pth`, 44.66 MB) | Cached in `.cache/torch/hub/checkpoints/` |
| **Pretrained ResNet-34** | Present (`resnet34-b627a593.pth`, 83.27 MB) | Cached in `.cache/torch/hub/checkpoints/` |

---

## 3. Partition Verification & Data Isolation Proof

Integrity of [`outputs/dataset_manifest.json`](file:///d:/Ai%20Claude/outputs/dataset_manifest.json) was verified programmatically against the raw dataset:

$$\text{Train} = 4,794 \quad \text{Val} = 600 \quad \text{Calib} = 600 \quad \text{Test} = 5,794$$
$$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Calib} = \emptyset, \quad \text{Val} \cap \text{Calib} = \emptyset$$
$$(\text{Train} \cup \text{Val} \cup \text{Calib}) \cap \text{Test} = \emptyset$$
$$\text{Train} \cup \text{Val} \cup \text{Calib} = \text{Official Train Partition} \quad (5,994 \text{ images})$$
$$\text{Test} = \text{Official Test Partition} \quad (5,794 \text{ images})$$

- **The calibration partition ($600$ images)** is strictly excluded from gradient updates and model selection.
- **The official test partition ($5,794$ images)** is 100% isolated and locked.

---

## 4. Training Pipeline Architecture

### A. Loss Function & Metric Definitions
- **Loss:** Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss()`):
  $$\mathcal{L}_{\text{concept}}(z, y) = - \frac{1}{112} \sum_{c=1}^{112} \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$
- **Concept Probabilities:** Derived via sigmoid $\hat{p}_c = \sigma(z_c) \in [0.0, 1.0]$.
- **Macro-F1:** Unweighted mean of F1 scores across all 112 concepts with decision threshold $0.5$:
  $$\text{Macro-F1} = \frac{1}{112} \sum_{c=1}^{112} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

### B. Checkpointing & Logging
- **Best Model:** Saved to `outputs/concept_model/best_concept_model.pth` based on lowest validation loss.
- **Latest Model:** Saved to `outputs/concept_model/latest_checkpoint.pth` for crash/interruption recovery.
- **Metrics Logs:** Saved to `training_history.csv` and `training_log.json`.

---

## 5. Smoke Test Execution Results

### Execution Command:
```powershell
.\venv\Scripts\python.exe scripts/train_concept_model.py --smoke-test --output-dir outputs/smoke_test
```

### Execution Log:
```text
======================================================================
      CALIBRATED CONCEPT-BASED FEEDBACK MODEL: TRAINING PIPELINE      
======================================================================
Device:           cpu
Backbone:         resnet18 (Pretrained: True)
Batch Size:       32
Learning Rate:    0.001
Weight Decay:     0.0001
Epochs Budget:    SMOKE-TEST (1 epoch, 2 batches)
Seed:             42
Output Directory: outputs/smoke_test
======================================================================

[1/4] Loading CUB-200-2011 DataLoaders (Train: 4,794, Val: 600)...
      Train loader: 150 batches (4794 images)
      Val loader:   19 batches (600 images)

[2/4] Initializing ConceptPredictor with backbone 'resnet18'...
      Feature dimension: 512
      Concept targets:   112
      Pretrained source: torchvision:IMAGENET1K_V1

[3/4] Executing Smoke Test (2 train batches, 1 val batch)...
Epoch 01/01 | Train Loss: 0.8751 | Val Loss: 0.9481 | Val Macro-F1: 0.1419 | Val Acc: 64.59% | Time: 10.4s [*BEST*]

[4/4] Smoke test completed successfully!
      Checkpoint saved to: outputs\smoke_test\best_concept_model.pth
```

### Verification Checks from Smoke Test:
- ✅ Loss is finite ($0.8751$ train, $0.9481$ val).
- ✅ Gradients computed and non-zero on concept projection head and backbone layers.
- ✅ Optimizer step executed cleanly without memory exhaustion.
- ✅ Checkpoint serialized and verified with all 122 state dict keys and configuration metadata.

---

## 6. Complete Unit Test Suite (35 Tests)

```powershell
.\venv\Scripts\pytest.exe tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0 -- D:\Ai Claude\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Ai Claude
plugins: anyio-4.15.1
collected 35 items

tests/test_concept_mapping.py::TestConceptMapping::test_attributes_txt_alignment PASSED [  2%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_exact_count PASSED [  5%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_range PASSED [  8%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_uniqueness PASSED [ 11%]
tests/test_concept_mapping.py::TestConceptMapping::test_selected_concepts_112_json_file PASSED [ 14%]
tests/test_concept_model.py::TestConceptModel::test_extract_features_shape PASSED [ 17%]
tests/test_concept_model.py::TestConceptModel::test_factory_build_concept_predictor PASSED [ 20%]
tests/test_concept_model.py::TestConceptModel::test_logits_vs_probabilities PASSED [ 22%]
tests/test_concept_model.py::TestConceptModel::test_missing_custom_weights_file_raises_error PASSED [ 25%]
tests/test_concept_model.py::TestConceptModel::test_model_output_shape_and_finite_values PASSED [ 28%]
tests/test_concept_model.py::TestConceptModel::test_pretrained_failure_raises_informative_error_without_silent_substitution PASSED [ 31%]
tests/test_concept_model.py::TestConceptModel::test_unsupported_backbone_raises_value_error PASSED [ 34%]
tests/test_config.py::TestConfig::test_default_config_loads_successfully PASSED [ 37%]
tests/test_config.py::TestConfig::test_invalid_concept_count_fails_validation PASSED [ 40%]
tests/test_config.py::TestConfig::test_missing_config_raises_informative_error PASSED [ 42%]
tests/test_config.py::TestConfig::test_missing_required_section_fails_validation PASSED [ 45%]
tests/test_config.py::TestConfig::test_proposal_specified_constants PASSED [ 48%]
tests/test_dataset.py::TestCUBConceptDataset::test_canonical_concept_ids_load_and_order PASSED [ 51%]
tests/test_dataset.py::TestCUBConceptDataset::test_dataloader_batch_shapes_and_partial_batches PASSED [ 54%]
tests/test_dataset.py::TestCUBConceptDataset::test_dataset_sample_structure_and_ranges PASSED [ 57%]
tests/test_dataset.py::TestCUBConceptDataset::test_deterministic_eval_transforms PASSED [ 60%]
tests/test_dataset.py::TestCUBConceptDataset::test_split_isolation_blocks_test_split PASSED [ 62%]
tests/test_dataset.py::TestCUBConceptDataset::test_synthetic_fixture_missing_image_handling PASSED [ 65%]
tests/test_dataset.py::TestCUBConceptDataset::test_training_augmentation_configuration PASSED [ 68%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_auditor_filtering_behavior PASSED [ 71%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_mapping_format PASSED [ 74%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_deterministic_partitions_isolation_and_completeness PASSED [ 77%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_malformed_line_raises_error PASSED [ 80%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_missing_file_raises_error PASSED [ 82%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_out_of_range_class_raises_error PASSED [ 85%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_success PASSED [ 88%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_partition_manifest_save_and_load PASSED [ 91%]
tests/test_training.py::TestTrainingPipeline::test_checkpoint_save_and_reload_state PASSED [ 94%]
tests/test_training.py::TestTrainingPipeline::test_compute_concept_metrics_known_values PASSED [ 97%]
tests/test_training.py::TestTrainingPipeline::test_gradient_computation_and_optimizer_step PASSED [100%]

============================= 35 passed in 19.91s =============================
```

---

## 7. CPU Hardware Limitations and Runtime Projections

### Measured Throughput:
- **1 Batch of 32 images (forward + backward):** ~3.5 seconds
- **1 Epoch (150 train batches + 19 val batches = 169 batches):** $\approx 169 \times 3.5\text{s} \approx 591\text{ seconds}$ (**~9.8 to 10 minutes per epoch**)

### Runtime Estimates for Full Training:

| Budget Option | Epoch Count | Expected Duration | Feasibility on CPU |
| :--- | :---: | :---: | :--- |
| **Option A (Initial Baseline)** | **1 epoch** | **~10 minutes** | Highly recommended; completes end-to-end training cycle on all 4,794 images without overheating or stalling. |
| **Option B (Moderate Initial)** | **3 epochs** | **~30 minutes** | Good multi-epoch curve; manageable background run. |
| **Option C (Extended Initial)** | **5 epochs** | **~50 minutes** | Significant convergence on concept attributes. |
| **Full Proposal Target** | **40 epochs** | **~6.7 hours** | Impractical for interactive turn; better suited for overnight batch execution. |

> [!IMPORTANT]
> **User Approval & Budget Selection:**  
> In accordance with Task 4 instructions, the user approved **Option B (3 epochs)** for the first full training run of the 112-concept predictor.

---

## 8. First Full Concept Predictor Training Results (Option B: 3 Epochs)

### Training Execution Configuration:
- **Command:** `python scripts/train_concept_model.py --epochs 3 --batch-size 32 --lr 0.001 --backbone resnet18 --output-dir outputs/concept_resnet18_3epochs`
- **Device:** CPU (8 logical cores)
- **Model Architecture:** Pretrained ImageNet ResNet-18 backbone + linear projection head ($\mathbb{R}^{512} \to \mathbb{R}^{112}$)
- **Pretrained Source:** `torchvision:IMAGENET1K_V1` (`resnet18-f37072fd.pth`)
- **Loss Function:** `nn.BCEWithLogitsLoss()`
- **Optimizer:** `AdamW(lr=0.001, weight_decay=0.0001)`
- **Data Partitions:** Training split ($4,794$ images) for gradient updates; Validation split ($600$ images) for evaluation & checkpoint selection; Calibration ($600$ images) and Test ($5,794$ images) partitions strictly locked and isolated.

### Epoch-by-Epoch Metric Progression:

| Epoch | Train Loss | Train Acc (%) | Train Macro-F1 | Val Loss | Val Acc (%) | Val Macro-F1 | Val Precision | Val Recall | Epoch Time | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01/03** | 0.3939 | 82.54% | 0.2343 | 0.3770 | 83.26% | 0.2864 | 0.4845 | 0.2594 | 672.7s (11.2m) | `*BEST*` |
| **02/03** | 0.3630 | 83.89% | 0.2893 | 0.3644 | 84.14% | 0.2936 | 0.5063 | 0.2434 | 629.4s (10.5m) | `*BEST*` |
| **03/03** | 0.3540 | 84.24% | 0.3212 | **0.3524** | **84.38%** | **0.3014** | **0.5084** | **0.2487** | 635.3s (10.6m) | `*BEST*` |

### Key Observations:
1. **Steady Monotonic Convergence:**
   - Training loss decreased consistently from $0.3939 \to 0.3630 \to 0.3540$.
   - Validation loss improved monotonically from $0.3770 \to 0.3644 \to \mathbf{0.3524}$.
   - Validation macro-F1 increased steadily from $0.2864 \to 0.2936 \to \mathbf{0.3014}$.
   - Validation element-wise concept accuracy improved from $83.26\% \to 84.14\% \to \mathbf{84.38\%}$.
2. **Total Elapsed Training Time:**
   - $672.7\text{s} + 629.4\text{s} + 635.3\text{s} = 1,937.4\text{ seconds}$ (**32 minutes 17 seconds**).
3. **No Numerical Instabilities or Memory Leaks:**
   - All losses remained strictly finite throughout all 150 training batches and 19 validation batches per epoch.
   - Peak memory usage remained stable within the system's available headroom on Windows 11.
4. **Saved Artifacts:**
   - Best Checkpoint: `outputs/concept_resnet18_3epochs/best_concept_model.pth`
   - Latest Checkpoint: `outputs/concept_resnet18_3epochs/latest_checkpoint.pth`
   - Tabular CSV Log: `outputs/concept_resnet18_3epochs/training_history.csv`
   - JSON Summary Log: `outputs/concept_resnet18_3epochs/training_log.json`

---

## 9. Verification & Boundary Commitments Respected

- **No Data Leakage:** Gradient updates were computed exclusively on the 4,794 training images.
- **Calibration Split Intact:** The 600 calibration images were never evaluated, fitted, or accessed during training.
- **Official Test Set Intact:** Exactly 5,794 test images remain 100% locked and isolated.
- **Pretrained Weights Authenticity:** Official ImageNet weights (`IMAGENET1K_V1`) were utilized; no silent random substitution occurred.
- **No Fabricated Metrics:** Every metric in this report corresponds directly to the verified logs and checkpoints in `outputs/concept_resnet18_3epochs/`.

