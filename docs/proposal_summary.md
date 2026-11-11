# Calibrated Concept-Based Feedback Model
## Comprehensive Project Proposal Summary

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Student:** Mostafizur Rahman (Student ID: 41230301942)  
**Institution:** Northern University Bangladesh  
**Course Instructor:** Ayesha Siddiqua, Senior Lecturer  
**Date:** October 2026  

---

## 1. Project Title and Problem Statement

### Title
**Calibrated Concept-Based Feedback Model: Interactive Human–AI Correction through Uncertainty-Aware Concept Bottlenecks**

### Problem Statement
Standard deep neural networks map inputs directly to outputs ($x \to y$), acting as "black boxes" whose predictions are difficult for human domain experts to inspect, debug, or correct. Concept Bottleneck Models (CBMs) address interpretability by inserting an intermediate, human-understandable concept layer ($x \to c \to y$). For example, in bird classification, concepts represent observable physical attributes such as *wing colour*, *bill shape*, *crown colour*, or *underparts colour*.

However, in practice, standard CBMs suffer from three major shortcomings:
1. **Uncalibrated Concept Predictions:** A model can be confidently wrong about predicted attributes. Raw softmax/sigmoid scores do not represent true empirical posterior probabilities.
2. **Unprincipled Review Flagging:** Standard review mechanisms often flag concepts based on arbitrary, uncalibrated thresholds or require users to inspect all concepts, imposing an unreasonable cognitive burden.
3. **Inconsistent and Uncontrolled Interventions:** When users attempt to correct concepts, the order of concepts edited, the representation used for intervention (e.g., probability replacement vs. logit replacement), and inaccurate user feedback can drastically and unpredictably alter downstream task predictions.

### Research Problem
The core research problem is to evaluate whether **calibrated confidence scores** and **split-conformal prediction flags** can direct a limited human correction budget more efficiently than fixed-threshold or random selection policies, while measuring the downstream impact of intervention representation and noisy feedback in a single reproducible pipeline.

### Research Questions (RQs)
- **RQ1:** Does temperature scaling reduce concept-level expected calibration error (ECE)?
- **RQ2:** Does split conformal flagging achieve empirical coverage near its nominal target ($1-\alpha$), and what review load (flagged fraction) does it create?
- **RQ3:** Do low-confidence or conformal-flagged concepts improve task accuracy faster per correction than random selection?
- **RQ4:** How do direct probability replacement and training-percentile logit replacement affect intervention outcomes?
- **RQ5:** How sensitive are the intervention gains to the CBM training scheme (independent, sequential, joint) and incorrect feedback?

---

## 2. Main Objectives and Expected Outcomes

### Objectives
1. **Model Baselines & CBMs:** Train an end-to-end ResNet baseline ($x \to y$) and three standard CBM variants ($x \to c \to y$):
   - *Independent CBM:* Concept predictor ($x \to c$) and task predictor ($c \to y$) trained separately.
   - *Sequential CBM:* Concept predictor trained first; task predictor trained on predicted concepts $\hat{c}$.
   - *Joint CBM:* Concept predictor and task predictor trained jointly with combined loss.
2. **Concept Probability Calibration:** Calibrate concept probabilities using temperature scaling ($z \to \sigma(z/T)$), reporting pre- and post-calibration Expected Calibration Error (ECE) and Negative Log-Likelihood (NLL).
3. **Distribution-Free Uncertainty with Conformal Prediction:** Implement split conformal prediction on an isolated calibration split to generate set-valued concept predictions $C_j(x)$, flagging uncertain non-singleton sets across significance levels $\alpha \in \{0.05, 0.10, 0.20\}$.
4. **Controlled Concept Interventions:** Compare concept selection policies (lowest calibrated confidence, conformal-flagged, and random control) under strictly equal review budgets $k$.
5. **Ablation of Intervention Representations & Noisy Feedback:** Measure the performance of direct probability replacement ($S_1$) versus training-percentile logit replacement ($S_2$), and evaluate degradation under simulated feedback noise rates $\epsilon \in \{0.00, 0.05, 0.10, 0.20\}$.
6. **Interactive Research Interface:** Build a lightweight Streamlit demonstration interface displaying test images, predicted concepts, calibrated confidence values, conformal uncertainty flags, interactive user edits, and real-time species prediction updates.

### Expected Outcomes
- An end-to-end reproducible codebase with locked splits, fixed random seeds (3 seeds: mean $\pm$ standard deviation), and structured logging.
- Empirical verification of whether temperature scaling improves concept reliability on held-out bird attribute data.
- Quantification of conformal coverage and user review load on fine-grained visual concepts.
- Rigorous intervention-efficiency curves demonstrating how task accuracy scales per human edit.
- An interactive, transparent prototype that demonstrates human-in-the-loop AI correction.

---

## 3. Dataset and Relevant Statistics

### Dataset: Caltech-UCSD Birds-200-2011 (CUB-200-2011)
- **Total Images:** 11,788 photos of birds.
- **Classes (Bird Species):** 200 fine-grained species.
- **Original Attributes:** 312 binary attributes per image (annotating visual features like bill shape, wing color, throat pattern).
- **Filtered Concept Set:** 112 binary attributes selected following standard CBM filtering criteria (eliminating rarely occurring or ambiguous attributes).
- **Official Dataset Splits:**
  - Official Train Set: 5,994 images.
  - Official Test Set: 5,794 images (remains frozen and untouched until final evaluation).
- **Project Partitioning Protocol:**
  - The 5,994 official training images are split into:
    - **Training Split (~80%):** ~4,795 images (used for backbone and classifier training).
    - **Validation Split (~10%):** ~600 images (used for hyperparameter tuning, model selection, and fitting calibration temperature $T$).
    - **Calibration Split (~10%):** ~599 images (held-out split used strictly to calculate conformal nonconformity quantiles).
  - **Test Split:** 5,794 images (used strictly for reporting final performance, coverage, and intervention curves).
- **Preprocessing Protocol:**
  - Standard ImageNet resizing and normalization ($224 \times 224$).
  - Data augmentations applied during training only (e.g., random crop, horizontal flip).
  - Strictly deterministic resizing and center-cropping for validation, calibration, and test splits.

---

## 4. Proposed Model Architecture and Baselines

### Modular System Pipeline
$$\text{Image } x \longrightarrow \text{Backbone } h(x) \longrightarrow \text{Concept Head } z \in \mathbb{R}^{112} \longrightarrow \text{Calibration } p(T) \longrightarrow \text{Conformal Sets } C_j(x) \longrightarrow \text{Selection } S \longrightarrow \text{Intervention } c' \longrightarrow \text{Task Predictor } \hat{y} \in \mathbb{R}^{200}$$

### Components
1. **Image Backbone:** ImageNet-pretrained ResNet-50 (with ResNet-18 as a documented fallback if GPU VRAM/compute is constrained). Extracts feature representation $h(x)$.
2. **Concept Predictor Head:** Linear mapping from feature dimension to 112 concept logits $z \in \mathbb{R}^{112}$. Concept probabilities: $\hat{p}_j = \sigma(z_j)$, trained with binary cross-entropy loss.
3. **Task Predictor:** Linear layer or small MLP mapping the 112-dimensional concept vector to the 200 species classes.
4. **Baselines:**
   - **B1 (Accuracy Reference):** Standard end-to-end ResNet-50 predicting 200 species directly from pixels ($x \to y$) without any concept bottleneck.
   - **B2 (Standard CBM with Fixed-Threshold):** Standard CBM predicting concepts with uncalibrated probabilities, flagging concepts where confidence falls below a fixed threshold.
   - **B3 (Proposed Calibrated & Conformal CBM):** Calibrated concept probabilities with split-conformal prediction sets and uncertainty-guided interventions.

---

## 5. Concept Calibration and Conformal Prediction Methods

### Concept Calibration: Temperature Scaling
- For each concept $j \in \{1, \dots, 112\}$, logits $z_j$ are scaled by a positive temperature parameter $T_j > 0$ (or a single global temperature $T > 0$):
  $$p_j(T) = \sigma\left(\frac{z_j}{T}\right)$$
- Temperature parameter(s) are optimized on the validation split by minimizing Negative Log-Likelihood (NLL).
- Evaluated before and after calibration using:
  - **ECE (Expected Calibration Error):** Measuring the gap between predicted confidence and empirical accuracy across binned confidence intervals.
  - **NLL (Negative Log-Likelihood):** Providing a strictly proper scoring rule.

### Split Conformal Prediction
- Uses the dedicated **calibration partition** (exchangeable with the test set).
- For binary concept $j$, nonconformity score for candidate label $y \in \{0, 1\}$:
  $$s_j(x, y) = 1 - \hat{p}_j(x, y)$$
- Compute the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile $\hat{q}_j$ of calibration scores.
- The prediction set for a test instance is:
  $$C_j(x) = \{ y \in \{0, 1\} \mid s_j(x, y) \le \hat{q}_j \}$$
- **Flagging Rule:**
  - If $C_j(x) = \{0, 1\}$ (both outcomes possible), the concept is flagged as **uncertain**.
  - If $C_j(x) = \emptyset$, it indicates an anomalous/atypical concept score.
  - Singleton sets $\{0\}$ or $\{1\}$ are considered confident.
- Target coverage levels evaluated: $\alpha \in \{0.05, 0.10, 0.20\}$ (corresponding to 95%, 90%, and 80% nominal marginal coverage).

---

## 6. Planned Concept-Feedback / Intervention Strategies

### Concept Selection Policies (Equal Budget $k$)
For each test sample, rank concepts and select the top-$k$ concepts for simulated or human correction:
- **P1 (Calibrated Low-Confidence):** Select concepts with the lowest calibrated confidence (i.e., confidence closest to 0.5: $|p_j(T) - 0.5|$).
- **P2 (Conformal-Flagged):** Select concepts whose conformal prediction set is non-singleton ($|C_j(x)| \ne 1$).
- **P3 (Random Selection Control):** Select $k$ random concepts uniformly as an experimental control baseline.

### Intervention Replacement Strategies
When a concept $j$ is selected for intervention:
- **S1 (Direct Binary Probability Replacement):** Replace predicted concept probability $\hat{p}_j$ directly with ground-truth binary label $c_j^* \in \{0, 1\}$ (or $0.99 / 0.01$ to avoid boundary anomalies).
- **S2 (Training-Percentile Logit Replacement):** Replace concept logit $z_j$ with the empirical 95th percentile (if $c_j^*=1$) or 5th percentile (if $c_j^*=0$) of training set logits for that concept. This prevents out-of-distribution logit inputs from destabilizing downstream task layers.

### Noisy Feedback Robustness Evaluation
Simulate imperfect human feedback by introducing independent noise with probability $\epsilon$:
$$\tilde{c}_j = \begin{cases} 1 - c_j^* & \text{with probability } \epsilon \\ c_j^* & \text{with probability } 1 - \epsilon \end{cases}$$
Evaluated across noise rates $\epsilon \in \{0.00, 0.05, 0.10, 0.20\}$ to test resilience against human mistakes.

---

## 7. Evaluation Metrics

| Metric | Definition & Interpretation | Role in Evaluation |
| :--- | :--- | :--- |
| **Task Accuracy** | Proportion of test images classified with the correct species. | Primary classification benchmark before and after intervention. |
| **Macro-F1** | Unweighted mean of class-wise F1 scores across 200 species. | Evaluates balanced classification performance across rare/frequent bird species. |
| **Concept Accuracy / Macro-F1** | Mean binary classification accuracy and macro-averaged F1 across all 112 concepts. | Measures how accurately the concept head predicts attributes. |
| **Expected Calibration Error (ECE)** | Binned absolute difference between model confidence and empirical accuracy. | Quantifies whether confidence estimates are statistically reliable. |
| **Negative Log-Likelihood (NLL)** | Average negative log-probability of true concept labels. | Proper scoring rule assessing probability quality. |
| **Empirical Conformal Coverage** | Percentage of test instances where the true concept label falls inside $C_j(x)$. | Verifies whether conformal prediction meets nominal $1-\alpha$ guarantees. |
| **Average Set Size & Flagged Fraction** | Average cardinality of $C_j(x)$ and proportion of concepts requiring human review. | Measures human cognitive load and review efficiency. |
| **Intervention Efficiency Curve** | Plot of Species Task Accuracy as a function of the number of corrected concepts $k$. | Determines accuracy gained per unit of human review effort. |
| **Interventions to Target Accuracy** | Smallest number of corrections $k$ required to reach a specific target accuracy (e.g., 85%). | Practical measure of human intervention cost. |
| **Noisy-Feedback Robustness** | Task accuracy as a function of simulated error rate $\epsilon \in [0, 0.20]$. | Measures how sensitive policies are to inaccurate human feedback. |

---

## 8. Week-by-Week Plan (Based on 14-Week Schedule)

| Weeks | Activities | Milestone / Deliverable |
| :--- | :--- | :--- |
| **Weeks 1–2** | Review literature; verify citations; set up environment; audit CUB dataset and 112-attribute mapping. | Literature summary and verified data pipeline. |
| **Weeks 3–4** | Implement PyTorch DataLoaders (train/val/calib/test); train end-to-end ResNet baseline. | Baseline checkpoint and initial metrics. |
| **Weeks 5–6** | Implement Independent, Sequential, and Joint Concept Bottleneck Models (CBMs). | Trained CBM models and concept accuracy results. |
| **Week 7** | Implement temperature scaling calibration; evaluate pre/post ECE and NLL on validation set. | Calibration report and reliability plots. |
| **Week 8** | Implement split conformal prediction; compute nonconformity quantiles on calibration split; compare with fixed threshold. | Conformal coverage and review-load analysis. |
| **Weeks 9–10** | Execute intervention experiments (P1, P2, P3); test replacement strategies (S1, S2); run noisy feedback tests ($\epsilon$). | Core experimental intervention results complete. |
| **Week 11** | Statistical aggregation across 3 random seeds; compute paired bootstrap confidence intervals; plot curves. | Formal result tables and publication-quality figures. |
| **Week 12** | Develop and test interactive Streamlit application (image upload, concept review, user intervention, prediction updates). | Working interactive demo application. |
| **Week 13** | Reproducibility verification audit; test-set isolation check; comprehensive draft report preparation. | Draft project report. |
| **Week 14** | Final code cleanup, report editing, and lab presentation delivery. | Complete submission package. |

---

## 9. Technical Risks, Dependencies, and Questions Requiring Clarification

### Identified Technical Risks & Mitigations (from Proposal Section 8)
1. **Limited Compute / Slow Training:**
   - *Risk:* Training ResNet-50 and three CBM variants across multiple seeds on CPU or modest GPU can be slow.
   - *Mitigation:* Use ImageNet-pretrained weights, support ResNet-18 as a fast fallback, freeze earlier backbone layers during initial runs, and use configurable batch sizes.
2. **Calibration Failure:**
   - *Risk:* Post-hoc temperature scaling might not significantly decrease ECE for all concepts.
   - *Mitigation:* Report pre- and post-calibration ECE/NLL transparently; evaluate both scalar and vector (per-concept) temperature scaling.
3. **Conformal Coverage Deviation:**
   - *Risk:* Empirical coverage may deviate from nominal $1-\alpha$ if exchangeability between calibration and test partitions is compromised.
   - *Mitigation:* Enforce strict split separation, verify deterministic preprocessing, and document empirical coverage honestly.
4. **Too Many Flagged Concepts:**
   - *Risk:* Low $\alpha$ (e.g., $\alpha=0.05$) might produce large prediction sets, flagging too many concepts and overwhelming the user.
   - *Mitigation:* Test multiple $\alpha \in \{0.05, 0.10, 0.20\}$ and benchmark against fixed confidence thresholds at matched review loads.
5. **Simulated Feedback Limitations:**
   - *Risk:* Simulated ground-truth feedback ignores human cognitive fatigue and subjective ambiguity.
   - *Mitigation:* Explicitly declare feedback as simulated in all documentation; simulate noise $\epsilon \in [0, 0.20]$.
6. **Concept Leakage and Incomplete Concepts:**
   - *Risk:* Information leakage in joint CBMs or incomplete concept sets may limit causal interpretation.
   - *Mitigation:* Restrict the task predictor strictly to concept representations ($c \to y$), report concept metrics independently, and avoid unwarranted causal claims.

### Dependencies
- **Programming:** Python 3.10+ (current environment: Python 3.13.3 on Windows).
- **Deep Learning:** PyTorch, torchvision.
- **Data & Numerical:** NumPy, Pandas, scikit-learn.
- **Plotting:** Matplotlib, Seaborn.
- **Web Interface:** Streamlit.
- **Configuration:** PyYAML (with built-in fallback parser if PyYAML is not yet installed).

### Questions / Decisions for Subsequent Weeks (Clearly Labeled Recommendations)
> [!NOTE] Recommendations Not Formally Prescribed in the PDF Proposal
> 1. **Concept Selection Mapping Definition (Week 2):**  
>    The CUB-200-2011 dataset contains 312 attributes, while the proposal specifies using 112 filtered concepts. Standard literature (Koh et al., 2020) selects 112 concepts based on having at least 10 positive training examples in at least 10 classes. In Week 2, a deterministic audit script should generate and save this exact 112-concept index mapping (`data/selected_concepts_112.json` or `.txt`) before training commences, without hard-coding assumptions.
> 2. **Reproducible Split File Generation (Week 2):**  
>    The official CUB split provides 5,994 train and 5,794 test images. The proposal specifies an 80/10/10 split of the training set into Train (~4,795), Validation (~600), and Calibration (~599). In Week 2, a fixed stratified splitting script with a pinned random seed should generate exact image ID manifests (`train_split.txt`, `val_split.txt`, `calib_split.txt`) so all subsequent models use identical data partitions.
> 3. **Hardware Fallback Strategy:**  
>    Because local development on Windows may lack dedicated NVIDIA CUDA hardware, the codebase must support automatic device selection (`cuda` if available, otherwise `cpu`).

