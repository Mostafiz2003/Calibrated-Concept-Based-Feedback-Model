# Week 5 Progress Report: ResNet-18 Per-Concept Baseline Analysis

**Course:** CSE 4385: AI and Expert System Lab  
**Institution:** Northern University Bangladesh  
**Student:** Mostafizur Rahman  
**Date:** November 10, 2026  
**Status:** Completed (Evaluation-Only; No Training / Checkpoint Unaltered)

---

## Executive Summary

Week 5 formalizes the baseline per-concept diagnostic evaluation of the initial 112-concept predictor trained in Week 4 (`outputs/concept_resnet18_3epochs/best_concept_model.pth`). 

In strict adherence to experimental safety protocols:
- The model was **not** retrained or fine-tuned.
- The existing checkpoint was **not** modified or overwritten.
- Calibration data (600 images) remained strictly held out for downstream post-hoc calibration (Week 7).
- The official CUB-200-2011 test partition (5,794 images) remained **locked and untouched**.
- The evaluation was conducted **strictly on the validation split** (600 images) using the fixed, documented primary decision threshold ($\tau = 0.5$) without post-hoc threshold tuning.

---

## 1. Aggregate Baseline Performance

The multi-label concept predictor maps an input bird image $x \in \mathbb{R}^{3 \times 224 \times 224}$ to 112 concept logits $z \in \mathbb{R}^{112}$. Probabilities are obtained via the standard sigmoid activation $\hat{p}_j = \sigma(z_j)$, and binary concept predictions are formed by $\hat{y}_j = \mathbb{I}(\hat{p}_j \ge 0.5)$.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Validation BCE Loss** | **0.3524** | Consistent with epoch 3 validation checkpoint loss. |
| **Macro F1 Score** | **0.3014** (30.14%) | Unweighted mean F1 across all 112 concepts. |
| **Macro Precision** | **0.5084** (50.84%) | Unweighted mean precision across concepts. |
| **Macro Recall** | **0.2487** (24.87%) | Unweighted mean recall across concepts. |
| **Micro F1 Score** | **0.4619** (46.19%) | Global harmonic mean of precision and recall. |
| **Micro Precision** | **0.6487** (64.87%) | Aggregated precision across all $600 \times 112$ concept decisions. |
| **Micro Recall** | **0.3587** (35.87%) | Aggregated recall across all $600 \times 112$ concept decisions. |
| **Mean Concept Accuracy** | **0.8438** (84.38%) | Element-wise binary accuracy across all concepts and images. |
| **Exact Match Ratio** | **0.0000** (0.00%) | Proportion of images where all 112 concepts are simultaneously correct. |

---

## 2. Top 10 Concepts by F1 Score

The highest-performing concepts generally exhibit higher ground-truth prevalence, strong visual distinctiveness (e.g., eye color, prominent bill geometry, dominant body plumage), or low annotator ambiguity:

| Rank | Index | CUB Attr ID | Attribute Name | Support ($N=600$) | Prevalence | Precision | Recall | F1 Score | Accuracy |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 51 | 146 | `has_eye_color::black` | 503 | 0.8383 | 0.8438 | 0.9881 | **0.9103** | 0.8367 |
| 2 | 53 | 152 | `has_bill_length::shorter_than_head` | 348 | 0.5800 | 0.7028 | 0.9310 | **0.8010** | 0.7317 |
| 3 | 89 | 236 | `has_shape::perching-like` | 307 | 0.5117 | 0.7139 | 0.8371 | **0.7706** | 0.7450 |
| 4 | 22 | 55 | `has_breast_pattern::solid` | 334 | 0.5567 | 0.6441 | 0.8832 | **0.7449** | 0.6633 |
| 5 | 48 | 133 | `has_throat_color::white` | 206 | 0.3433 | 0.7021 | 0.6408 | **0.6701** | 0.7833 |
| 6 | 20 | 52 | `has_underparts_color::white` | 246 | 0.4100 | 0.7310 | 0.5854 | **0.6501** | 0.7417 |
| 7 | 93 | 255 | `has_primary_color::yellow` | 82 | 0.1367 | 0.8571 | 0.5122 | **0.6412** | 0.9217 |
| 8 | 7 | 21 | `has_wing_color::black` | 251 | 0.4183 | 0.6494 | 0.5976 | **0.6224** | 0.6967 |
| 9 | 101 | 294 | `has_crown_color::blue` | 33 | 0.0550 | 0.6250 | 0.6061 | **0.6154** | 0.9583 |
| 10 | 43 | 118 | `has_breast_color::white` | 217 | 0.3617 | 0.7400 | 0.5115 | **0.6049** | 0.7583 |

---

## 3. Bottom 10 Concepts by F1 Score

Under the fixed threshold of 0.5, 18 concepts received zero predicted positive labels ($\hat{y}_j = 0$ for all 600 validation images), resulting in an F1 score of 0.0000. The bottom 10 concepts sorted by lowest F1 and prevalence are:

| Rank | Index | CUB Attr ID | Attribute Name | Support ($N=600$) | Prevalence | Precision | Recall | F1 Score | Accuracy |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 75 | 197 | `has_nape_color::buff` | 87 | 0.1450 | 0.0000 | 0.0000 | **0.0000** | 0.8533 |
| 2 | 76 | 199 | `has_belly_color::brown` | 52 | 0.0867 | 0.0000 | 0.0000 | **0.0000** | 0.9133 |
| 3 | 77 | 200 | `has_belly_color::iridescent` | 6 | 0.0100 | 0.0000 | 0.0000 | **0.0000** | 0.9900 |
| 4 | 78 | 202 | `has_belly_color::rufous` | 3 | 0.0050 | 0.0000 | 0.0000 | **0.0000** | 0.9950 |
| 5 | 80 | 207 | `has_belly_color::pink` | 5 | 0.0083 | 0.0000 | 0.0000 | **0.0000** | 0.9917 |
| 6 | 81 | 208 | `has_belly_color::orange` | 15 | 0.0250 | 0.0000 | 0.0000 | **0.0000** | 0.9750 |
| 7 | 84 | 213 | `has_wing_shape::rounded-wings` | 149 | 0.2483 | 0.0000 | 0.0000 | **0.0000** | 0.7517 |
| 8 | 85 | 214 | `has_wing_shape::pointed-wings` | 135 | 0.2250 | 0.0000 | 0.0000 | **0.0000** | 0.7750 |
| 9 | 96 | 277 | `has_leg_color::red` | 21 | 0.0350 | 0.0000 | 0.0000 | **0.0000** | 0.9650 |
| 10 | 105 | 303 | `has_crown_color::pink` | 4 | 0.0067 | 0.0000 | 0.0000 | **0.0000** | 0.9933 |

---

## 4. Diagnostic Indicators and Class Imbalance Analysis

### 4.1 Ground-Truth Support Distribution
- **Zero Support Concepts:** Exactly **0**. Every single one of the 112 concepts appears in at least 3 validation images ($support \ge 3$).
- **Support Range:** Minimum support is 3 (`has_belly_color::rufous`), maximum support is 503 (`has_eye_color::black`).
- **Prevalence Statistics:**
  - Minimum Prevalence: `0.0050` (0.5%)
  - Maximum Prevalence: `0.8383` (83.8%)
  - Mean Prevalence: `0.1869` (18.7%)
  - Median Prevalence: `0.1617` (16.2%)

### 4.2 The Accuracy vs. F1 Paradox
Across the 112 concepts, the mean binary accuracy is **84.38%**, while the unweighted Macro F1 is only **30.14%**. This divergence is a classic signature of severe class imbalance in multi-label classification:
- Because the mean positive prevalence is only 18.7%, a trivial predictor that outputs all zeros would achieve an average accuracy of $1 - 0.187 = 81.3\%$.
- For extremely rare concepts such as `has_belly_color::rufous` (3 positive examples in 600), predicting 0 for all samples yields 99.5% accuracy but an F1 score of 0.0000.
- Macro F1 treats each concept equally regardless of frequency, thereby highlighting the model's struggle with rare attributes.

### 4.3 Zero Predicted Positives Under Fixed 0.5 Threshold
A total of **18 concepts** had zero positive predictions ($\hat{y} = 0 \ \forall \ x$):
1. `has_underparts_color::brown`
2. `has_upper_tail_color::buff`
3. `has_head_pattern::eyebrow`
4. `has_eye_color::olive`
5. `has_forehead_color::iridescent`
6. `has_forehead_color::purple`
7. `has_forehead_color::pink`
8. `has_under_tail_color::red`
9. `has_nape_color::iridescent`
10. `has_belly_color::brown`
11. `has_belly_color::iridescent`
12. `has_belly_color::rufous`
13. `has_belly_color::pink`
14. `has_belly_color::orange`
15. `has_wing_shape::rounded-wings`
16. `has_wing_shape::pointed-wings`
17. `has_leg_color::red`
18. `has_crown_color::pink`

This phenomenon arises because uncalibrated cross-entropy / BCE training with an unweighted loss naturally pulls predicted sigmoid probabilities toward the empirical prior frequency ($\sim 0.01 - 0.05$). Under an uncalibrated 0.5 threshold, the probability rarely surpasses 0.5, suppressing all positive calls.

---

## 5. Methodological Limitations

1. **Uncalibrated Probabilities:** The current probabilities are uncalibrated raw sigmoid outputs $\sigma(z_j)$. As demonstrated above, rare concepts suffer from overconfident suppression below 0.5. Post-hoc calibration (Temperature Scaling on the calibration split) will be essential.
2. **Fixed 0.5 Decision Boundary:** In imbalanced settings, a static 0.5 threshold is suboptimal. However, per course guidelines, threshold optimization must not be fit on the validation split and reported as unbiased test performance.
3. **Small Baseline Epoch Budget:** The ResNet-18 model was trained for only 3 epochs on CPU as an initial sanity check. Training for 20–40 epochs on ResNet-50 with feature pyramids (Week 6) will substantially improve feature representations.
4. **No Higher-Order Spatial Attention in ResNet-18:** ResNet-18 lacks multi-scale attention (CBAM) and feature pyramids (FPN), making subtle local attributes (e.g., eye rings, bill serrations, wing shapes) difficult to isolate from global image features.

---

## 6. Generated Artifacts

| File Path | Description |
| :--- | :--- |
| [`outputs/week5_analysis/per_concept_metrics.csv`](file:///d:/Ai%20Claude/outputs/week5_analysis/per_concept_metrics.csv) | Full table of all 112 concepts with precision, recall, F1, support, prevalence, TP/FP/FN/TN, and accuracy. |
| [`outputs/week5_analysis/aggregate_metrics.json`](file:///d:/Ai%20Claude/outputs/week5_analysis/aggregate_metrics.json) | Structured machine-readable JSON containing macro/micro metrics, diagnostic counts, and prevalence summary. |
| [`outputs/week5_analysis/analysis_summary.md`](file:///d:/Ai%20Claude/outputs/week5_analysis/analysis_summary.md) | Markdown document with tabular summaries of top-10 and bottom-10 concepts and diagnostics. |
| [`docs/week5_progress.md`](file:///d:/Ai%20Claude/docs/week5_progress.md) | Comprehensive Week 5 progress report (this document). |
| [`src/evaluation/concept_analysis.py`](file:///d:/Ai%20Claude/src/evaluation/concept_analysis.py) | Modular library for per-concept evaluation and diagnostic reporting. |
| [`scripts/evaluate_week5_concepts.py`](file:///d:/Ai%20Claude/scripts/evaluate_week5_concepts.py) | CLI script for standalone execution of the evaluation pipeline. |
| [`tests/test_concept_analysis.py`](file:///d:/Ai%20Claude/tests/test_concept_analysis.py) | Unit tests verifying metric computations, edge cases, and artifact creation. |

