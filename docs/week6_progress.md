# Week 6 Progress Report: Recognition-Focused Multi-Model Architecture Design

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Project:** Calibrated Concept-Based Feedback Model  
**Student:** Mostafizur Rahman (41230301942)  
**Date:** November 2026  
**Status:** Completed (Week 6 Architecture Design, Modular Components, and Testing Only; No New Model Training Performed)

---

## 1. Executive Summary & Week 5 Status Report

### A. Week 5 Baseline-Analysis Status Report
In accordance with Task 1 instructions (*"Inspect the Week 5 baseline-analysis status. If Week 5 is incomplete, report that before proceeding"*):
- **Current Status:** In the original course timeline, Week 5 was designated for formal CBM baseline variant analysis (Independent, Sequential, and Joint CBMs). 
- **Finding:** Following the completion of Week 4 (where the first ResNet-18 concept predictor was trained for 3 epochs), the project transitioned directly to Week 6 to establish the **recognition-focused multi-model architecture** (ResNet-50 + CBAM + FPN + Dual Heads).
- **Report:** Week 5 CBM variant comparative experiments remain pending and will build directly upon the modular architectures implemented in this phase.

### B. Week 6 Summary of Completed Tasks
During Week 6, a modular recognition-focused architecture was designed, implemented, and verified across a comprehensive test suite of 45 unit tests:
1. **Modular Components Implemented:**
   - Pretrained **ResNet-50** backbone extracting multi-stage hierarchical features ($C_2, C_3, C_4, C_5$).
   - **CBAM (Convolutional Block Attention Module)** featuring sequential Channel Attention ($M_c$) and Spatial Attention ($M_s$) with residual skip connections.
   - **FPN (Feature Pyramid Network)** building top-down multi-scale representations ($P_2, P_3, P_4, P_5$) via 1x1 lateral projections and 3x3 smoothing convolutions.
   - **Multi-Scale Feature Aggregation** combining multi-scale pyramid levels into a unified 2048-dimensional feature representation.
   - **Dual Prediction Heads:** Concept Head producing exactly 112 unnormalized binary logits, and Species Head producing exactly 200 species classification logits.
2. **Experimental Regimes (E1–E4) Supported:**
   - **E1:** Existing ResNet-18 baseline concept predictor (preserved completely untouched at `outputs/concept_resnet18_3epochs/best_concept_model.pth`).
   - **E2:** ResNet-50 with Dual Heads (no CBAM, no FPN).
   - **E3:** ResNet-50 + CBAM with Dual Heads (no FPN).
   - **E4:** ResNet-50 + CBAM + FPN with Dual Heads (Proposed Full Architecture).
3. **Training & Loss Interfaces:**
   - `JointRecognitionLoss` computing $\mathcal{L}_{\text{total}} = \lambda_{\text{concept}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{species}} \mathcal{L}_{\text{CE}}$ with configurable loss weights.
   - Clear separation between raw logits and normalized probabilities (`predict_concept_probabilities`, `predict_species_probabilities`, `predict_species`).
   - Checkpoint save/load compatibility matching the existing project schema.
4. **Automated Verification:**
   - Added `tests/test_recognition_model.py` (10 new unit tests).
   - Full test suite: **45 passed in 40.08s**.

---

## 2. Architecture Comparison: E1, E2, E3, and E4

```
E1 (Baseline):
Image [3, 224, 224] ──> ResNet-18 Backbone ──> GAP [512] ──> Concept Head ──> 112 Concept Logits

E2 (Deep Dual-Head):
Image [3, 224, 224] ──> ResNet-50 Backbone ──> C5 [2048, 7, 7] ──> GAP [2048] ──┬──> Concept Head ──> 112 Concept Logits
                                                                                 └──> Species Head ──> 200 Species Logits

E3 (Attention-Enhanced):
Image [3, 224, 224] ──> ResNet-50 Stages ──> CBAM Modules ──> C5_att [2048, 7, 7] ──> GAP [2048] ──┬──> Concept Head ──> 112 Concept Logits
                                                                                                    └──> Species Head ──> 200 Species Logits

E4 (Proposed Model: Attention + Multi-Scale Feature Pyramid):
Image [3, 224, 224] ──> ResNet-50 Stages [C2, C3, C4, C5]
                              │
                              ▼
                         CBAM Modules [C2_att, C3_att, C4_att, C5_att]
                              │
                              ▼
                         Feature Pyramid Network (FPN) [P2, P3, P4, P5] (256 channels each)
                              │
                              ▼
                         Multi-Scale Feature Aggregation [Concat + Fusion Layer]
                              │
                              ▼
                       Unified Global Feature Vector [2048]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              Concept Head        Species Head
           [112 Concept Logits] [200 Species Logits]
```

### Detailed Structural Comparison:

| Dimension | E1 (ResNet-18 Baseline) | E2 (ResNet-50 Dual-Head) | E3 (ResNet-50 + CBAM) | E4 (ResNet-50 + CBAM + FPN) |
| :--- | :---: | :---: | :---: | :---: |
| **Backbone Architecture** | ResNet-18 (8 residual blocks) | ResNet-50 (16 bottleneck blocks) | ResNet-50 (16 bottleneck blocks) | ResNet-50 (16 bottleneck blocks) |
| **Attention Mechanism** | None | None | CBAM (Channel + Spatial on all 4 stages) | CBAM (Channel + Spatial on all 4 stages) |
| **Feature Pyramid (FPN)** | None | None | None | 4-level FPN ($P_2, P_3, P_4, P_5$ with 256 channels) |
| **Feature Aggregation** | Global Average Pooling on layer4 | Global Average Pooling on layer4 | Global Average Pooling on attended layer4 | Multi-Scale Pooling across $P_2..P_5$ + Fusion LayerNorm |
| **Unified Feature Dim** | 512 | 2048 | 2048 | 2048 |
| **Concept Prediction** | Yes (112 binary logits) | Yes (112 binary logits) | Yes (112 binary logits) | Yes (112 binary logits) |
| **Species Recognition** | No (concepts only) | Yes (200 species logits) | Yes (200 species logits) | Yes (200 species logits) |
| **Total Parameters** | **11,233,968** (~11.23 M) | **24,147,320** (~24.15 M) | **24,844,040** (~24.84 M) | **30,291,720** (~30.29 M) |
| **Trainable Parameters** | **11,233,968** | **24,147,320** | **24,844,040** | **30,291,720** |

---

## 3. Mathematical Specifications of Component Modules

### A. CBAM (Convolutional Block Attention Module)
Given intermediate feature map $F \in \mathbb{R}^{B \times C \times H \times W}$:
1. **Channel Attention Sub-Module:**
   $$M_c(F) = \sigma\left( W_1 (W_0 (F_{\text{avg}}^c)) + W_1 (W_0 (F_{\text{max}}^c)) \right)$$
   where $W_0 \in \mathbb{R}^{\frac{C}{r} \times C}$, $W_1 \in \mathbb{R}^{C \times \frac{C}{r}}$ with reduction ratio $r=16$.
   $$F' = M_c(F) \odot F$$
2. **Spatial Attention Sub-Module:**
   $$M_s(F') = \sigma\left( f^{7 \times 7} \left( [F_{\text{avg}}'^s; F_{\text{max}}'^s] \right) \right)$$
   where $[F_{\text{avg}}'^s; F_{\text{max}}'^s] \in \mathbb{R}^{B \times 2 \times H \times W}$ are pooled across channels.
   $$F'' = M_s(F') \odot F'$$
3. **Residual Connection:**
   $$F_{\text{out}} = F + F''$$

### B. Feature Pyramid Network (FPN)
Constructs uniform 256-channel representations $P_2, P_3, P_4, P_5$ from stage features $C_2, C_3, C_4, C_5$:
- **Top Level:** $P_5 = \text{Smooth}_{3\times 3}(\text{Lateral}_{1\times 1}(C_5))$
- **Recursive Top-Down Pathway:**
  $$P_i = \text{Smooth}_{3\times 3}\left( \text{Lateral}_{1\times 1}(C_i) + \text{Upsample}(P_{i+1}) \right), \quad i \in \{4, 3, 2\}$$

### C. Multi-Scale Feature Aggregation
For each pyramid level $P_i \in \{P_2, P_3, P_4, P_5\}$, global pooling extracts scale-specific summary vectors $v_i = \text{GAP}(P_i) \in \mathbb{R}^{256}$.
The concatenated representation $v_{\text{multi}} = [v_2; v_3; v_4; v_5] \in \mathbb{R}^{1024}$ is projected into the final global feature space:
$$f_{\text{agg}} = \text{Dropout}(\text{ReLU}(\text{LayerNorm}(W_{\text{fusion}} v_{\text{multi}} + b_{\text{fusion}}))) \in \mathbb{R}^{2048}$$

### D. Joint Recognition Loss
$$\mathcal{L}_{\text{total}} = \lambda_{\text{concept}} \mathcal{L}_{\text{BCE}}(z_{\text{concept}}, y_{\text{concept}}) + \lambda_{\text{species}} \mathcal{L}_{\text{CE}}(z_{\text{species}}, y_{\text{species}})$$
where:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{112} \sum_{c=1}^{112} \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$
$$\mathcal{L}_{\text{CE}} = -\log \frac{\exp(z_{\text{species}, y})}{\sum_{k=1}^{200} \exp(z_{\text{species}, k})}$$

---

## 4. Hardware Feasibility Assessment & Runtime Projections

### Measured System Environment:
- **Processor:** 8 logical CPU cores (Windows 11)
- **PyTorch Environment:** `2.15.0+cpu` (CPU mode; no CUDA GPU detected)
- **System RAM:** 7.74 GB Total RAM (0.75 GB available)
- **Pretrained Weights Status:**
  - ResNet-18: Cached (`resnet18-f37072fd.pth`, 44.66 MB)
  - ResNet-34: Cached (`resnet34-b627a593.pth`, 83.27 MB)
  - ResNet-50: **Cached** (`resnet50-11ad3fa6.pth`, 97.8 MB, `IMAGENET1K_V2`)

### Benchmarking Throughput on CPU:
- **ResNet-18 Concept Forward Pass (batch=32):** ~1.5 seconds
- **ResNet-50 + CBAM + FPN Forward Pass (batch=32):** 5.67 seconds
- **Estimated Full Forward + Backward Pass per batch (batch=32):** ~13.5 seconds

### Training Runtime Projections (on 4,794 training images = 150 batches/epoch):

| Experiment Configuration | Parameters | Batch Time (Est.) | 1 Epoch Runtime (Est.) | 3 Epochs Runtime (Est.) | Feasibility on CPU |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **E1: ResNet-18 Concept** | 11.23 M | ~3.5s | ~10.5 minutes | **32.3 minutes** | **Completed in Week 4** |
| **E2: ResNet-50 Dual-Head** | 24.15 M | ~9.0s | ~24.5 minutes | ~1.2 hours | Practical for initial run |
| **E3: ResNet-50 + CBAM** | 24.84 M | ~10.5s | ~28.5 minutes | ~1.4 hours | Practical for initial run |
| **E4: ResNet-50 + CBAM + FPN** | 30.29 M | ~13.5s | ~38.0 minutes | ~1.9 hours | Practical as scheduled run |

> [!WARNING]
> **Hardware Boundary & Recommendation:**  
> On CPU mode, training E4 for 3 epochs requires approximately 1.9 hours. Memory usage must remain bounded (batch size 32 requires ~600 MB of RAM, leaving ~150–200 MB headroom). In accordance with strict guidelines, **full training has not been started** in this phase.

---

## 5. Complete Unit Test Suite Results (45 Tests)

```powershell
.\venv\Scripts\pytest.exe tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0 -- D:\Ai Claude\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Ai Claude
plugins: anyio-4.15.1
collected 45 items

tests/test_concept_mapping.py::TestConceptMapping::test_attributes_txt_alignment PASSED [  2%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_exact_count PASSED [  4%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_range PASSED [  6%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_uniqueness PASSED [  8%]
tests/test_concept_mapping.py::TestConceptMapping::test_selected_concepts_112_json_file PASSED [ 11%]
tests/test_concept_model.py::TestConceptModel::test_extract_features_shape PASSED [ 13%]
tests/test_concept_model.py::TestConceptModel::test_factory_build_concept_predictor PASSED [ 15%]
tests/test_concept_model.py::TestConceptModel::test_logits_vs_probabilities PASSED [ 17%]
tests/test_concept_model.py::TestConceptModel::test_missing_custom_weights_file_raises_error PASSED [ 20%]
tests/test_concept_model.py::TestConceptModel::test_model_output_shape_and_finite_values PASSED [ 22%]
tests/test_concept_model.py::TestConceptModel::test_pretrained_failure_raises_informative_error_without_silent_substitution PASSED [ 24%]
tests/test_concept_model.py::TestConceptModel::test_unsupported_backbone_raises_value_error PASSED [ 26%]
tests/test_config.py::TestConfig::test_default_config_loads_successfully PASSED [ 28%]
tests/test_config.py::TestConfig::test_invalid_concept_count_fails_validation PASSED [ 31%]
tests/test_config.py::TestConfig::test_missing_config_raises_informative_error PASSED [ 33%]
tests/test_config.py::TestConfig::test_missing_required_section_fails_validation PASSED [ 35%]
tests/test_config.py::TestConfig::test_proposal_specified_constants PASSED [ 37%]
tests/test_dataset.py::TestCUBConceptDataset::test_canonical_concept_ids_load_and_order PASSED [ 40%]
tests/test_dataset.py::TestCUBConceptDataset::test_dataloader_batch_shapes_and_partial_batches PASSED [ 42%]
tests/test_dataset.py::TestCUBConceptDataset::test_dataset_sample_structure_and_ranges PASSED [ 44%]
tests/test_dataset.py::TestCUBConceptDataset::test_deterministic_eval_transforms PASSED [ 46%]
tests/test_dataset.py::TestCUBConceptDataset::test_split_isolation_blocks_test_split PASSED [ 48%]
tests/test_dataset.py::TestCUBConceptDataset::test_synthetic_fixture_missing_image_handling PASSED [ 51%]
tests/test_dataset.py::TestCUBConceptDataset::test_training_augmentation_configuration PASSED [ 53%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_auditor_filtering_behavior PASSED [ 55%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_mapping_format PASSED [ 57%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_deterministic_partitions_isolation_and_completeness PASSED [ 60%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_malformed_line_raises_error PASSED [ 62%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_missing_file_raises_error PASSED [ 64%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_out_of_range_class_raises_error PASSED [ 66%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_success PASSED [ 68%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_partition_manifest_save_and_load PASSED [ 71%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_cbam_attention_shapes_and_finite_values PASSED [ 73%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_checkpoint_save_and_load_compatibility PASSED [ 75%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_concept_and_species_probabilities_helpers PASSED [ 77%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_configurable_experiments_e2_e3_e4 PASSED [ 80%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_dual_head_output_shapes_and_values PASSED [ 82%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_factory_build_recognition_model PASSED [ 84%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_fpn_pyramid_and_aggregation PASSED [ 86%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_joint_loss_calculation_and_finite_gradients PASSED [ 88%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_pretrained_failure_raises_informative_error PASSED [ 91%]
tests/test_recognition_model.py::TestRecognitionArchitecture::test_resnet50_stage_feature_extraction PASSED [ 93%]
tests/test_training.py::TestTrainingPipeline::test_checkpoint_save_and_reload_state PASSED [ 95%]
tests/test_training.py::TestTrainingPipeline::test_compute_concept_metrics_known_values PASSED [ 97%]
tests/test_training.py::TestTrainingPipeline::test_gradient_computation_and_optimizer_step PASSED [100%]

============================= 45 passed in 40.08s =============================
```

---

## 6. Strict Boundaries Respected

- ✅ **Canonical 112 concepts untouched:** Unchanged from `data/selected_concepts_112.json`.
- ✅ **Partition membership untouched:** Train (4,794), Val (600), Calib (600), Test (5,794) strictly preserved in `outputs/dataset_manifest.json`.
- ✅ **Test set isolation:** Exactly 5,794 images remain locked and isolated.
- ✅ **ResNet-18 baseline checkpoint preserved:** Intact at `outputs/concept_resnet18_3epochs/best_concept_model.pth`.
- ✅ **No full model training was started:** Only modular components, model interfaces, loss functions, and unit tests were implemented and verified.
- ✅ **WS-DAN and TransFG:** Excluded from this stage as instructed.

