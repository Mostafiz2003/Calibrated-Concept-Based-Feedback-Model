# Concept Selection Methodology Review and Decision Document

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Project:** Calibrated Concept-Based Feedback Model  
**Author:** Mostafizur Rahman (41230301942)  
**Date:** October 2026  
**Status:** Canonical 112 Benchmark Recovered, Verified Against Source & Dataset, Mapping Saved to `data/selected_concepts_112.json`, and Fully Tested

---

## 1. Executive Summary & Verification Outcome

Following the approval of **Option 1 (reproducing the canonical 112-concept CUB benchmark from Koh et al., ICML 2020)**, a comprehensive multi-stage audit was executed across the source code, data splits, index conventions, and physical dataset files.

### Key Verification Milestones:
1. **Direct Repository Alignment (112/112 match):** All 112 zero-based column indices in our recovered list were compared directly against the hardcoded list in the official repository ([`yewsiang/ConceptBottleneck/blob/master/CUB/generate_new_data.py`](https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/generate_new_data.py)). The index-by-index comparison yielded **0 discrepancies**.
2. **Dual-Index Convention Explicitly Enforced:**
   - In the benchmark repository, array indices are **zero-based** (`0..311`).
   - In CUB-200-2011's official `attributes.txt`, attribute IDs are **one-based** (`1..312`).
   - In our concept bottleneck vector, indices are **zero-based** (`0..111`).
   - Every entry in `data/selected_concepts_112.json` explicitly stores `concept_index`, `zero_based_column_index`, and `cub_attribute_id = zero_based_column_index + 1`, eliminating any off-by-one errors.
3. **Physical Dataset Name Validation:** Every single one of the 112 attribute IDs was verified against the local `data/CUB_200_2011/attributes.txt`. All 112 names match the official taxonomy exactly (0 mismatches).
4. **Distinct Procedures Clearly Distinguished:**
   - **Canonical 112-Attribute Benchmark Mapping:** The standardized list established by Koh et al. (ICML 2020) and benchmarked in subsequent literature (Shin et al. 2023, Vandenhirtz et al. 2024, Kalampalikis et al. 2026).
   - **Dynamic Full-Split Thresholding (109 attributes):** Dynamically applying the 10-species majority voting threshold on our full official training split (5,994 images) yields 109 attributes (with 11 borderline attributes appearing in exactly 9 species).
   - **Remaining Uncertainty / Scientific Caution:** While subsampling or split differences between Koh et al.'s internal `train.pkl` and the full official training set are hypothesized to cause the 3-attribute difference, this cannot be definitively proven without the original author's internal random seed and split artifact. This is documented transparently as an **unresolved uncertainty**.
5. **Mapping Frozen:** The authoritative file `data/selected_concepts_112.json` is generated, formatted, and verified.
6. **Automated Testing:** All **18 unit tests** across the test suite pass with 100% success.

---

## 2. Examination of the Original Proposal PDF

A line-by-line review of the course proposal PDF demonstrates why Option 1 is the methodologically required choice:

- **Abstract (Page 2):** *"An ImageNet-pretrained ResNet backbone predicts 112 filtered binary attributes."*
- **Architecture Table (Page 4):** *"Concept head: $h(x) \to \text{logits } z \in \mathbb{R}^{112} \to \text{probabilities } p$. 112 filtered binary attributes; sigmoid and binary cross-entropy."*
- **Section 5.1 (Page 6):** *"Following the existing proposal, the project uses 112 filtered attributes as concepts... Apply one documented filtering procedure and save the mapping from the retained 112 concepts to original attribute IDs."*
- **Section 6 (Page 7):** *"Concept accuracy / macro-F1: Binary attribute accuracy and F1 over 112 concepts."*
- **Section 11.2 (Page 10):** *"Limitations: The 112 filtered attributes may be incomplete for species classification."*
- **Literature Grounding (Pages 10–11):** The proposal cites Koh et al. (ICML 2020) [Ref 1], Shin et al. (ICML 2023) [Ref 5], Vandenhirtz et al. (NeurIPS 2024) [Ref 6], and Kalampalikis et al. (TMLR 2026) [Ref 7]. All four benchmark against this exact 112-concept bottleneck.

---

## 3. Discrepancy Analysis: Canonical Benchmark vs. Dynamic Thresholding

In the official repository for *Concept Bottleneck Models* ([`yewsiang/ConceptBottleneck`](https://github.com/yewsiang/ConceptBottleneck)), the attribute filtering procedure is in `CUB/generate_new_data.py`:

```python
def get_class_attributes_data(min_class_count, out_dir, modify_data_dir='', keep_instance_data=False):
    """
    Use train.pkl to aggregate attributes on class level and only keep those that 
    are predominantly 1 for at least min_class_count classes.
    ...
    In our paper, we set min_class_count to be 10 and only use the following 112 
    attributes of indices [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 
    38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 
    90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 
    141, 145, 149, 151, 152, 153, 154, 155, 157, 158, 161, 164, 167, 168, 172, 
    178, 179, 180, 181, 182, 183, 184, 188, 193, 195, 196, 198, 199, 201, 202, 
    206, 207, 208, 211, 212, 213, 220, 221, 225, 235, 238, 242, 253, 254, 259, 
    274, 276, 277, 283, 289, 292, 293, 294, 298, 299, 302, 304, 305, 308, 309, 
    310, 311]
    """
```

### Direct Comparison of Procedures:

| Dimension | Initial Rule A (`class_majority_50`) | Dynamic Majority on Full Train Split | Canonical Benchmark List (Koh et al.) |
| :--- | :---: | :---: | :---: |
| **Concept Count** | **93** | **109** | **112** |
| **Data Scope** | Full train split (5,994 images) | Full train split (5,994 images) | Koh et al.'s specific `train.pkl` split |
| **Occlusion Handling (`certainty == 1`)** | Counted as negative evidence | Excluded from denominator if $a = 0$ | Excluded from denominator if $a = 0$ |
| **Tie Handling ($0\text{-count} == 1\text{-count}$)** | Resolved to 0 | Resolved to 1 | Resolved to 1 |
| **Threshold** | $\ge 10$ species classes | $\ge 10$ species classes | $\ge 10$ species classes |
| **Status in Literature** | Local experimental rule | Analytical reproduction attempt | **Standard universal benchmark** |

### Borderline Attributes and Unresolved Uncertainty:
When Koh's exact exclusion algorithm is executed dynamically on the full official training set, 109 concepts have $\ge 10$ classes, and 11 concepts have exactly 9 classes. 

> [!WARNING]
> **Scientific Uncertainty Notice:**  
> We do NOT assert that the 3-attribute discrepancy between the dynamic 109 count and the canonical 112 list is definitively proven to be sample variance from train/val splitting. Because Koh et al.'s original `train.pkl` file and random split seed are not fully reconstructed from raw text annotations, the exact mathematical cause remains an **unverified uncertainty**. Adopting the canonical 112 list guarantees exact comparability with the published literature cited by our proposal.

---

## 4. Complete Verified Table of the 112 Canonical Concepts

Every concept below has been checked index-by-index against both [`generate_new_data.py`](https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/generate_new_data.py) and `data/CUB_200_2011/attributes.txt`:

| Concept Index (`0..111`) | 0-Based Column (`0..311`) | CUB Attribute ID (`1..312`) | Official Attribute Name (`attributes.txt`) |
| :---: | :---: | :---: | :--- |
| **0** | 1 | **2** | `has_bill_shape::dagger` |
| **1** | 4 | **5** | `has_bill_shape::hooked_seabird` |
| **2** | 6 | **7** | `has_bill_shape::all-purpose` |
| **3** | 7 | **8** | `has_bill_shape::cone` |
| **4** | 10 | **11** | `has_wing_color::brown` |
| **5** | 14 | **15** | `has_wing_color::grey` |
| **6** | 15 | **16** | `has_wing_color::yellow` |
| **7** | 20 | **21** | `has_wing_color::black` |
| **8** | 21 | **22** | `has_wing_color::white` |
| **9** | 23 | **24** | `has_wing_color::buff` |
| **10** | 25 | **26** | `has_upperparts_color::brown` |
| **11** | 29 | **30** | `has_upperparts_color::grey` |
| **12** | 30 | **31** | `has_upperparts_color::yellow` |
| **13** | 35 | **36** | `has_upperparts_color::black` |
| **14** | 36 | **37** | `has_upperparts_color::white` |
| **15** | 38 | **39** | `has_upperparts_color::buff` |
| **16** | 40 | **41** | `has_underparts_color::brown` |
| **17** | 44 | **45** | `has_underparts_color::grey` |
| **18** | 45 | **46** | `has_underparts_color::yellow` |
| **19** | 50 | **51** | `has_underparts_color::black` |
| **20** | 51 | **52** | `has_underparts_color::white` |
| **21** | 53 | **54** | `has_underparts_color::buff` |
| **22** | 54 | **55** | `has_breast_pattern::solid` |
| **23** | 56 | **57** | `has_breast_pattern::striped` |
| **24** | 57 | **58** | `has_breast_pattern::multi-colored` |
| **25** | 59 | **60** | `has_back_color::brown` |
| **26** | 63 | **64** | `has_back_color::grey` |
| **27** | 64 | **65** | `has_back_color::yellow` |
| **28** | 69 | **70** | `has_back_color::black` |
| **29** | 70 | **71** | `has_back_color::white` |
| **30** | 72 | **73** | `has_back_color::buff` |
| **31** | 75 | **76** | `has_tail_shape::notched_tail` |
| **32** | 80 | **81** | `has_upper_tail_color::brown` |
| **33** | 84 | **85** | `has_upper_tail_color::grey` |
| **34** | 90 | **91** | `has_upper_tail_color::black` |
| **35** | 91 | **92** | `has_upper_tail_color::white` |
| **36** | 93 | **94** | `has_upper_tail_color::buff` |
| **37** | 99 | **100** | `has_head_pattern::eyebrow` |
| **38** | 101 | **102** | `has_head_pattern::plain` |
| **39** | 106 | **107** | `has_breast_color::brown` |
| **40** | 110 | **111** | `has_breast_color::grey` |
| **41** | 111 | **112** | `has_breast_color::yellow` |
| **42** | 116 | **117** | `has_breast_color::black` |
| **43** | 117 | **118** | `has_breast_color::white` |
| **44** | 119 | **120** | `has_breast_color::buff` |
| **45** | 125 | **126** | `has_throat_color::grey` |
| **46** | 126 | **127** | `has_throat_color::yellow` |
| **47** | 131 | **132** | `has_throat_color::black` |
| **48** | 132 | **133** | `has_throat_color::white` |
| **49** | 134 | **135** | `has_throat_color::buff` |
| **50** | 141 | **142** | `has_eye_color::olive` |
| **51** | 145 | **146** | `has_eye_color::black` |
| **52** | 149 | **150** | `has_bill_length::about_the_same_as_head` |
| **53** | 151 | **152** | `has_bill_length::shorter_than_head` |
| **54** | 152 | **153** | `has_forehead_color::blue` |
| **55** | 153 | **154** | `has_forehead_color::brown` |
| **56** | 154 | **155** | `has_forehead_color::iridescent` |
| **57** | 155 | **156** | `has_forehead_color::purple` |
| **58** | 157 | **158** | `has_forehead_color::grey` |
| **59** | 158 | **159** | `has_forehead_color::yellow` |
| **60** | 161 | **162** | `has_forehead_color::pink` |
| **61** | 164 | **165** | `has_forehead_color::white` |
| **62** | 167 | **168** | `has_under_tail_color::blue` |
| **63** | 168 | **169** | `has_under_tail_color::brown` |
| **64** | 172 | **173** | `has_under_tail_color::grey` |
| **65** | 178 | **179** | `has_under_tail_color::black` |
| **66** | 179 | **180** | `has_under_tail_color::white` |
| **67** | 180 | **181** | `has_under_tail_color::red` |
| **68** | 181 | **182** | `has_under_tail_color::buff` |
| **69** | 182 | **183** | `has_nape_color::blue` |
| **70** | 183 | **184** | `has_nape_color::brown` |
| **71** | 184 | **185** | `has_nape_color::iridescent` |
| **72** | 188 | **189** | `has_nape_color::yellow` |
| **73** | 193 | **194** | `has_nape_color::black` |
| **74** | 195 | **196** | `has_nape_color::red` |
| **75** | 196 | **197** | `has_nape_color::buff` |
| **76** | 198 | **199** | `has_belly_color::brown` |
| **77** | 199 | **200** | `has_belly_color::iridescent` |
| **78** | 201 | **202** | `has_belly_color::rufous` |
| **79** | 202 | **203** | `has_belly_color::grey` |
| **80** | 206 | **207** | `has_belly_color::pink` |
| **81** | 207 | **208** | `has_belly_color::orange` |
| **82** | 208 | **209** | `has_belly_color::black` |
| **83** | 211 | **212** | `has_belly_color::buff` |
| **84** | 212 | **213** | `has_wing_shape::rounded-wings` |
| **85** | 213 | **214** | `has_wing_shape::pointed-wings` |
| **86** | 220 | **221** | `has_size::medium_(9_-_16_in)` |
| **87** | 221 | **222** | `has_size::very_small_(3_-_5_in)` |
| **88** | 225 | **226** | `has_shape::duck-like` |
| **89** | 235 | **236** | `has_shape::perching-like` |
| **90** | 238 | **239** | `has_back_pattern::striped` |
| **91** | 242 | **243** | `has_tail_pattern::striped` |
| **92** | 253 | **254** | `has_primary_color::grey` |
| **93** | 254 | **255** | `has_primary_color::yellow` |
| **94** | 259 | **260** | `has_primary_color::black` |
| **95** | 274 | **275** | `has_leg_color::black` |
| **96** | 276 | **277** | `has_leg_color::red` |
| **97** | 277 | **278** | `has_leg_color::buff` |
| **98** | 283 | **284** | `has_bill_color::grey` |
| **99** | 289 | **290** | `has_bill_color::black` |
| **100** | 292 | **293** | `has_bill_color::buff` |
| **101** | 293 | **294** | `has_crown_color::blue` |
| **102** | 294 | **295** | `has_crown_color::brown` |
| **103** | 298 | **299** | `has_crown_color::grey` |
| **104** | 299 | **300** | `has_crown_color::yellow` |
| **105** | 302 | **303** | `has_crown_color::pink` |
| **106** | 304 | **305** | `has_crown_color::black` |
| **107** | 305 | **306** | `has_crown_color::white` |
| **108** | 308 | **309** | `has_wing_pattern::solid` |
| **109** | 309 | **310** | `has_wing_pattern::spotted` |
| **110** | 310 | **311** | `has_wing_pattern::striped` |
| **111** | 311 | **312** | `has_wing_pattern::multi-colored` |

---

## 5. Automated Test Suite Execution Results

Automated unit tests in [`tests/test_concept_mapping.py`](file:///d:/Ai%20Claude/tests/test_concept_mapping.py) verify:
1. Exact count of 112 concepts.
2. Uniqueness of all 112 concept indices.
3. Index ranges strictly within $[0, 311]$ (and attribute IDs within $[1, 312]$).
4. Physical presence and validity of each attribute ID and name in `attributes.txt`.
5. Full schema, metadata completeness, and dual-index validation of `data/selected_concepts_112.json`.

### Test Execution Command:
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
collected 18 items

tests/test_concept_mapping.py::TestConceptMapping::test_attributes_txt_alignment PASSED [  5%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_exact_count PASSED [ 11%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_range PASSED [ 16%]
tests/test_concept_mapping.py::TestConceptMapping::test_canonical_list_uniqueness PASSED [ 22%]
tests/test_concept_mapping.py::TestConceptMapping::test_selected_concepts_112_json_file PASSED [ 27%]
tests/test_config.py::TestConfig::test_default_config_loads_successfully PASSED [ 33%]
tests/test_config.py::TestConfig::test_invalid_concept_count_fails_validation PASSED [ 38%]
tests/test_config.py::TestConfig::test_missing_config_raises_informative_error PASSED [ 44%]
tests/test_config.py::TestConfig::test_missing_required_section_fails_validation PASSED [ 50%]
tests/test_config.py::TestConfig::test_proposal_specified_constants PASSED [ 55%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_auditor_filtering_behavior PASSED [ 61%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_concept_mapping_format PASSED [ 66%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_deterministic_partitions_isolation_and_completeness PASSED [ 72%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_malformed_line_raises_error PASSED [ 77%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_missing_file_raises_error PASSED [ 83%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_out_of_range_class_raises_error PASSED [ 88%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_metadata_parser_success PASSED [ 94%]
tests/test_dataset_pipeline.py::TestDatasetPipeline::test_partition_manifest_save_and_load PASSED [100%]

============================= 18 passed in 0.50s ==============================
```

---

## 6. Summary of Guarantees Respected

- **Official Test Set Intact:** Exactly 5,794 images remain completely untouched in the test set.
- **Deterministic Partitions:** Training (4,795 images), validation (599 images), and calibration (600 images) are isolated and documented in `outputs/dataset_manifest.json`.
- **Zero Hallucinated Concepts:** Every single concept corresponds to verified biological traits in the CUB-200-2011 benchmark.
- **Zero Training Executed:** No models have been trained, preserving Week 2 strictly for dataset pipeline verification.
