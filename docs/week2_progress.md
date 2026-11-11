# Week 2 Progress Report: Dataset Pipeline and Reproducible Concept Selection

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Project:** Calibrated Concept-Based Feedback Model  
**Student:** Mostafizur Rahman (41230301942)  
**Date:** October 2026  
**Status:** Completed (Week 2 Only)

---

## 1. Summary of Completed Tasks

During Week 2, all scheduled data pipeline components, annotation parsing, deterministic partitioning, concept auditing, diagnostic inspection, and automated testing were implemented:

1. **Dataset Metadata Parser (`src/data/metadata_parser.py`):**
   - Implemented strict parsing for official CUB-200-2011 annotation files:
     - `images.txt`
     - `image_class_labels.txt`
     - `train_test_split.txt`
     - `attributes/attributes.txt`
     - `classes.txt`
   - Built comprehensive validation: checks for missing files (`DatasetNotFoundError`), duplicate IDs, out-of-range class labels (`[1, 200]`), binary split indicators (`0` or `1`), and cross-reference mismatches across files (`CorruptAnnotationError`).
2. **Reproducible Data Partitions (`src/data/partitions.py`):**
   - Implemented deterministic partitioning of the official training set (5,994 images) into:
     - **Train Split (~80% / ~4,795 images):** For backbone and concept head training.
     - **Validation Split (~10% / ~600 images):** For model selection and fitting temperature parameter $T$.
     - **Calibration Split (~10% / ~599 images):** Held out strictly for conformal nonconformity quantiles.
   - Preserves the official test set (5,794 images) as completely untouched and isolated.
   - Built mathematical validation verifying mutual exclusivity:
     $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Calib} = \emptyset, \quad \text{Val} \cap \text{Calib} = \emptyset$$
     $$(\text{Train} \cup \text{Val} \cup \text{Calib}) \cap \text{Test} = \emptyset$$
3. **Concept Selection Audit Engine (`src/data/concept_audit.py`):**
   - Developed a memory-efficient streaming engine to audit `attributes/image_attribute_labels.txt` (~3.6 million rows).
   - Formulated candidate counting rules for determining an attribute's presence across species classes:
     - **Rule A (`class_majority_50`):** Attribute present in a species if $\ge 50\%$ of images in that species exhibit the attribute.
     - **Rule B (`class_any_positive`):** Attribute present in a species if at least 1 image exhibits the attribute.
     - **Rule C (`class_min_images_2`):** Attribute present in a species if $\ge 2$ images exhibit the attribute.
   - Built serialization functions to save approved concept mappings into `data/selected_concepts_112.json`, strictly separating model concept indices (`0..111`) from original CUB attribute IDs (`1..312`).
4. **Dataset Inspection and Audit Command (`scripts/inspect_dataset.py`):**
   - Updated the inspection tool to parse annotations, verify partitions, and audit concept selection rules.
   - Gracefully handles missing dataset files without crashing.
5. **Automated Testing Suite (`tests/test_dataset_pipeline.py`):**
   - Created synthetic mini-CUB fixtures to test parsing, malformed data handling, split determinism, split completeness, and concept filtering.
   - Executed the full test suite with `pytest`: **13 tests passed in 0.30 seconds**.
6. **Documentation Updates:**
   - Updated `data/README.md` and `README.md` with pipeline architecture, partition guidelines, and concept selection methodology.

---

## 2. Inventory of Created / Modified Files

| File | Status | Description |
| :--- | :---: | :--- |
| `src/data/metadata_parser.py` | **Created** | Annotation file parser with integrity checks and cross-validation |
| `src/data/partitions.py` | **Created** | Deterministic 80/10/10 split generator with test isolation guarantees |
| `src/data/concept_audit.py` | **Created** | Streaming attribute occurrence auditor and concept mapping serializer |
| `scripts/inspect_dataset.py` | **Updated** | Integrated with parser, partitions, and concept audit engine |
| `tests/test_dataset_pipeline.py` | **Created** | 8 unit tests using synthetic CUB fixtures |
| `data/README.md` | **Updated** | Added parser details, partition guarantees, and concept audit methodology |
| `README.md` | **Updated** | Added Week 2 testing and audit instructions |
| `docs/week2_progress.md` | **Created** | This comprehensive Week 2 report |

---

## 3. Test Suite Execution Results

Executed the full test suite in `d:\Ai Claude`:

```powershell
.\venv\Scripts\pytest.exe tests/ -v
```

### Actual Output
```text
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0 -- D:\Ai Claude\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Ai Claude
plugins: anyio-4.15.1
collecting ... collected 13 items

tests/test_config.py::TestConfig::test_default_config_loads_successfully PASSED [  7%]
tests/test_config.py::TestConfig::test_invalid_concept_count_fails_validation PASSED [ 15%]
tests/test_config.py::TestConfig::test_missing_config_raises_informative_error PASSED [ 23%]
tests/test_config.py::TestConfig::test_missing_required_section_fails_validation PASSED [ 30%]
tests/test_config.py::TestConfig::test_proposal_specified_constants PASSED [ 38%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_auditor_filtering_behavior PASSED [ 46%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_mapping_format PASSED [ 53%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_deterministic_partitions_isolation_and_completeness PASSED [ 61%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_malformed_line_raises_error PASSED [ 69%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_missing_file_raises_error PASSED [ 76%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_out_of_range_class_raises_error PASSED [ 84%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_success PASSED [ 92%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_partition_manifest_save_and_load PASSED [100%]

============================= 13 passed in 0.30s ==============================
```

All 13 unit tests passed with exit code `0`.

---

## 4. Concept Selection Analysis & Open Decisions for Approval

### The Requirement
Proposal Section 5.1 specifies:
> *"CUB-200-2011 contains 11,788 images across 200 bird species and 312 binary attributes per image. Following the existing proposal, the project uses 112 filtered attributes as concepts... Apply one documented filtering procedure and save the mapping from the retained 112 concepts to original attribute IDs."*

### The Uncertainty Identified
The proposal specifies that 112 filtered concepts must be used, but does **not** list the exact 112 attribute IDs or the exact mathematical counting rule in text.

In the standard literature (Koh et al., ICML 2020), attributes are filtered to 112 by requiring that an attribute appears in at least 10 species classes. However, "appears in a class" can be defined in multiple ways:
1. **Option 1 (Class Majority Threshold - Recommended):**  
   An attribute is present in a species if $\ge 50\%$ of the images in that species are annotated with `is_present == 1`. Retain attributes present in $\ge 10$ species.
2. **Option 2 (Class Occurrence with Minimum Image Count):**  
   An attribute is present in a species if at least $k$ images (e.g., $k=2$) in that species have `is_present == 1`.
3. **Option 3 (Standard CBM Koh et al. Benchmark List):**  
   Extract the 112 attribute IDs used by the official CBM benchmark repository (Koh et al., 2020) directly and verify their class frequencies.

### Strict Boundary Respected
In accordance with your instructions:
- No concept IDs were invented or hardcoded.
- `data/selected_concepts_112.json` has **not** been generated yet.
- Once you extract `CUB_200_2011.tgz`, running `python scripts/inspect_dataset.py` will audit all candidate rules against the real dataset and report their exact counts and discrepancies.
- You can then review the counts and approve the exact rule before freezing the mapping.

---

## 5. Next Steps for Week 3

Upon your approval:
1. Extract `CUB_200_2011.tgz` to `data/CUB_200_2011/`.
2. Run `scripts/inspect_dataset.py` to inspect the real dataset statistics and freeze `data/selected_concepts_112.json`.
3. Proceed to Week 3: Implement PyTorch `Dataset` and `DataLoader` classes with train augmentations and deterministic eval transforms, then build the end-to-end ResNet baseline.

