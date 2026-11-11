# Week 3 Progress Report: Dataset Pipeline and Initial Concept Predictor

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Project:** Calibrated Concept-Based Feedback Model  
**Student:** Mostafizur Rahman (41230301942)  
**Date:** October 2026  
**Status:** Completed (Week 3 Only: No Model Training Performed Yet)

---

## 1. Executive Summary

During Week 3, the foundational PyTorch data loading pipeline and the initial concept predictor model architecture were implemented, verified, and integrated into the project workspace:

1. **PyTorch Dataset (`src/data/dataset.py`):**
   - Implemented `CUBConceptDataset`, loading CUB-200-2011 images, 0-indexed species labels ($0 \dots 199$), and the frozen 112 binary concept labels.
   - Strict preservation of the canonical concept list from `data/selected_concepts_112.json`.
   - Applied data augmentation (`RandomCrop`, `RandomHorizontalFlip`) strictly to the training partition (`split="train"`).
   - Applied deterministic evaluation transforms (`Resize`, `CenterCrop`, `Normalize`) to validation and calibration partitions.
   - Preserved partition membership from `outputs/dataset_manifest.json` and enforced split isolation (access to the test split without explicit authorization raises `PermissionError`).
2. **Initial Backbone and Concept Predictor (`src/models/concept_model.py`):**
   - Implemented `ConceptPredictor`, combining a modular ResNet feature extractor ($h(x) \in \mathbb{R}^D$) with a linear projection head ($W \cdot h(x) + b \in \mathbb{R}^{112}$).
   - Configurable backbone support for ResNet-18 ($D=512$), ResNet-34 ($D=512$), and ResNet-50 ($D=2048$).
   - Explicit distinction between concept **logits** (unnormalized real values for `BCEWithLogitsLoss`) and concept **probabilities** ($[0.0, 1.0]$ via `torch.sigmoid`).
   - Pretrained weights handling: official ImageNet weights are loaded and cached locally. If downloading fails, `PretrainedWeightsLoadError` is raised—the system **never silently substitutes uninitialized random weights**.
   - **Zero model training was executed** in accordance with Week 3 boundaries.
3. **Comprehensive Automated Testing:**
   - Expanded test coverage across all pipeline layers.
   - Total test suite: **32 unit tests passed in 10.83 seconds**.

---

## 2. Partition Reconciliation: Documented vs. Recorded Split Sizes

In accordance with user instructions to inspect and resolve any mismatch between documented and recorded split sizes without silent alteration:

| Metric | Theoretical Scalar Ratios (Non-Stratified) | Actual Class-Stratified Partition (`dataset_manifest.json`) | Explanation of Difference |
| :--- | :---: | :---: | :--- |
| **Train Split (~80%)** | 4,795 images ($5,994 \times 0.80 = 4,795.2$) | **4,794 images** | Exactly $5,994 - 1,200 = 4,794$ |
| **Validation Split (~10%)** | 600 images ($5,994 \times 0.10 = 599.4$) | **600 images** | Exactly 3 images per species $\times 200$ species = 600 |
| **Calibration Split (~10%)** | 599 images | **600 images** | Exactly 3 images per species $\times 200$ species = 600 |
| **Total Official Train** | **5,994 images** | **5,994 images** | Exact conservation of official train images |
| **Official Test Split** | **5,794 images** | **5,794 images** | 100% frozen, isolated, and untouched |

> [!NOTE]
> **Resolution:** When stratifying across 200 bird species classes (each containing approximately 30 training images), allocating exactly 3 validation images and 3 calibration images per class yields $200 \times 3 = 600$ images for each holdout partition. This leaves exactly $5,994 - (600 + 600) = 4,794$ images for training. Both `configs/default.yaml` and `outputs/dataset_manifest.json` have been aligned to document this exact 1-image integer rounding behavior.

---

## 3. Tensor Specifications and Label Conventions

### A. Input Image Tensor
- **Shape:** `[batch_size, 3, 224, 224]`
- **Data Type:** `torch.float32`
- **Normalization:** Standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$)
- **Augmentation:** Applied only during training (`transforms.RandomCrop(224)`, `transforms.RandomHorizontalFlip(p=0.5)`).

### B. Species Target Label
- **Format:** Scalar `torch.int64` (LongTensor)
- **Range:** $0 \dots 199$ (0-indexed integer)
- **Convention:** Converted from CUB's 1-indexed class ID ($1 \dots 200$) via $\text{species\_label} = \text{class\_id} - 1$, matching standard PyTorch `nn.CrossEntropyLoss` requirements.

### C. Concept Target Labels
- **Shape:** `[batch_size, 112]`
- **Data Type:** `torch.float32`
- **Values:** Strictly binary $\{0.0, 1.0\}$
- **Ordering:** Strictly ordered by `concept_index` ($0 \dots 111$) from `data/selected_concepts_112.json`.

### D. Model Output Tensors
- **Concept Logits (`model(x)`):** `[batch_size, 112]` FloatTensor with unnormalized real values $(-\infty, +\infty)$ for numerical stability in `BCEWithLogitsLoss`.
- **Concept Probabilities (`model.predict_probabilities(x)`):** `[batch_size, 112]` FloatTensor with values strictly in $[0.0, 1.0]$ via $\sigma(z) = \frac{1}{1 + e^{-z}}$.
- **Pooled Features (`model.extract_features(x)`):** `[batch_size, D]` FloatTensor, where $D=512$ for ResNet-18/34 and $D=2048$ for ResNet-50.

---

## 4. Pretrained Weights Strategy & Offline Handling

The proposal specifies an ImageNet-pretrained ResNet backbone with ResNet-18 fallback:

1. **Local Weight Caching:**
   Official torchvision weights (`ResNet18_Weights.DEFAULT` and `ResNet34_Weights.DEFAULT`) have been downloaded and cached in the local PyTorch hub checkpoint store.
2. **Strict Failure Guard:**
   If `pretrained=True` is configured but network access is blocked or the weights file cannot be read, `ConceptPredictor` raises `PretrainedWeightsLoadError`. It explicitly instructs the user to either provide a local weights path or configure `pretrained: false` for random initialization.
3. **No Silent Fallback:**
   The model will never silently initialize random weights when the user or configuration requested pretrained weights.

---

## 5. Assumptions and Limitations in Concept-Label Construction

1. **Instance-Level vs. Class-Level Labels:**  
   The PyTorch Dataset currently loads instance-level annotations from `attributes/image_attribute_labels.txt`. In CUB-200-2011, certain individual image attribute annotations can have noise or disagreement across annotators. In future weeks (Week 4), we will support both instance-level concept supervision and class-majority aggregated concept labels as evaluated in Koh et al. (2020).
2. **Missing & Occluded Attributes:**  
   Raw CUB annotations contain a certainty flag:
   - `1`: not visible / occluded
   - `2`: guessing
   - `3`: probably
   - `4`: definitely  
   In standard binary concept training, attributes marked `is_present == 1` are treated as positive ($1.0$), while `is_present == 0` are treated as negative ($0.0$).
3. **Concept Coverage Limitation:**  
   As noted in Proposal Section 11.2, 112 filtered concepts do not span all subtle inter-species morphological differences, which is why joint and sequential CBM architectures or residual task connections are relevant topics in subsequent weeks.

---

## 6. Inventory of Created / Modified Files

| File | Status | Description |
| :--- | :---: | :--- |
| `src/data/dataset.py` | **Created** | PyTorch `CUBConceptDataset`, transform pipelines, and DataLoader factory |
| `src/models/concept_model.py` | **Created** | Modular `ConceptPredictor` (ResNet-18/34/50 + Linear concept head) |
| `src/models/__init__.py` | **Updated** | Exported `ConceptPredictor`, `build_concept_predictor`, and errors |
| `src/data/__init__.py` | **Updated** | Exported dataset classes, transform functions, and loader builders |
| `outputs/dataset_manifest.json` | **Created** | Frozen partition IDs (Train: 4,794, Val: 600, Calib: 600, Test: 5,794) |
| `configs/default.yaml` | **Updated** | Clarified stratified vs. non-stratified split size annotations |
| `tests/test_dataset.py` | **Created** | 7 unit tests for dataset, label semantics, transforms, and DataLoaders |
| `tests/test_concept_model.py` | **Created** | 7 unit tests for concept model, output shapes, logits vs. probs, and backbones |
| `docs/week3_progress.md` | **Created** | This comprehensive Week 3 progress report |
| `README.md` | **Updated** | Added smoke test instructions, tensor specifications, and backbone config |

---

## 7. Full Test Suite Execution Results

### Execution Command:
```powershell
.\venv\Scripts\pytest.exe tests/ -v
```

### Complete Test Results:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0 -- D:\Ai Claude\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Ai Claude
plugins: anyio-4.15.1
collected 32 items

tests/test_concept_mapping.py::TestConceptMapping::test_attributes_txt_alignment PASSED [  3%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_exact_count PASSED [  6%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_range PASSED [  9%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_uniqueness PASSED [ 12%]
tests/test_concept_mapping.py::TestConceptMapping::test_selected_concepts_112_json_file PASSED [ 15%]
tests/test_concept_model.py::TestConceptModel::test_extract_features_shape PASSED [ 18%]
tests/test_concept_model.py::TestConceptModel::test_factory_build_concept_predictor PASSED [ 21%]
tests/test_concept_model.py::TestConceptModel::test_logits_vs_probabilities PASSED [ 25%]
tests/test_concept_model.py::TestConceptModel::test_missing_custom_weights_file_raises_error PASSED [ 28%]
tests/test_concept_model.py::TestConceptModel::test_model_output_shape_and_finite_values PASSED [ 31%]
tests/test_concept_model.py::TestConceptModel::test_pretrained_failure_raises_informative_error_without_silent_substitution PASSED [ 34%]
tests/test_concept_model.py::TestConceptModel::test_unsupported_backbone_raises_value_error PASSED [ 37%]
tests/test_config.py::TestConfig::test_default_config_loads_successfully PASSED [ 40%]
tests/test_config.py::TestConfig::test_invalid_concept_count_fails_validation PASSED [ 43%]
tests/test_config.py::TestConfig::test_missing_config_raises_informative_error PASSED [ 46%]
tests/test_config.py::TestConfig::test_missing_required_section_fails_validation PASSED [ 50%]
tests/test_config.py::TestConfig::test_proposal_specified_constants PASSED [ 53%]
tests/test_dataset.py::TestCUBConceptDataset::test_canonical_concept_ids_load_and_order PASSED [ 56%]
tests/test_dataset.py::TestCUBConceptDataset::test_dataloader_batch_shapes_and_partial_batches PASSED [ 59%]
tests/test_dataset.py::TestCUBConceptDataset::test_dataset_sample_structure_and_ranges PASSED [ 62%]
tests/test_dataset.py::TestCUBConceptDataset::test_deterministic_eval_transforms PASSED [ 65%]
tests/test_dataset.py::TestCUBConceptDataset::test_split_isolation_blocks_test_split PASSED [ 68%]
tests/test_dataset.py::TestCUBConceptDataset::test_synthetic_fixture_missing_image_handling PASSED [ 71%]
tests/test_dataset.py::TestCUBConceptDataset::test_training_augmentation_configuration PASSED [ 75%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_auditor_filtering_behavior PASSED [ 78%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_mapping_format PASSED [ 81%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_deterministic_partitions_isolation_and_completeness PASSED [ 84%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_malformed_line_raises_error PASSED [ 87%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_missing_file_raises_error PASSED [ 90%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_out_of_range_class_raises_error PASSED [ 93%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_success PASSED [ 96%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_partition_manifest_save_and_load PASSED [100%]

============================= 32 passed in 10.83s =============================
```

---

## 8. Week 3 Commitments & Boundaries Respected

- **No full model training was initiated:** Model weights have not been trained, fitted, or evaluated.
- **Official test set remains untouched:** Test partition ($5,794$ images) is locked and protected by permission assertions.
- **Canonical 112 concepts preserved:** Strictly aligned with Koh et al. (ICML 2020) and `data/selected_concepts_112.json`.
- **Zero fabricated metrics:** No synthetic benchmark claims or simulated training curves were generated.

