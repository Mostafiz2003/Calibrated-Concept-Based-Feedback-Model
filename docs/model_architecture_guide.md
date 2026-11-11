# Model Architecture Guide: Fine-Grained Bird Recognition System

**Course:** CSE 4385: AI and Expert System Lab  
**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition  

---

## 1. Architectural Philosophy: Multi-Task Dual Prediction Heads

Every recognition architecture in this research system serves a **primary bird species recognition task** (200 fine-grained species classes) while simultaneously predicting **112 auxiliary visual bird attributes** (e.g., bill shape, wing pattern, plumage color).

```
Input Image [3 x 224 x 224]
        │
        ▼
   [Backbone Feature Extractor]
        │
        ▼
   Global Feature Representation h(x) ∈ R^D
        ├──> [Species Head: Linear(D, 200)]  ──> Species Logits [B, 200]
        └──> [Concept Head: Linear(D, 112)]  ──> Concept Logits [B, 112]
```

### Combined Objective:
$$\mathcal{L}_{total} = \mathcal{L}_{species} + \lambda_{concept} \cdot \mathcal{L}_{concept}$$
- $\mathcal{L}_{species}$: `CrossEntropyLoss` over 200 species
- $\mathcal{L}_{concept}$: `BCEWithLogitsLoss` over 112 binary concept attributes
- $\lambda_{concept} = 0.2$ (configurable)

---

## 2. Supported Recognition Architectures

| Model Name | Exp ID | Feature Dim ($D$) | Attention / Multiscale | Key Advantages |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet-50 Baseline** | **E2** | 2048 | None (GAP on $C_5$) | Standard CNN baseline for fine-grained benchmarks. |
| **ResNet-50 + CBAM** | **E3** | 2048 | CBAM (Channel + Spatial) | Attends to subtle species discriminants (eye rings, bill markings). |
| **ResNet-50 + CBAM + FPN** | **E4** | 2048 | CBAM + 4-Stage Feature Pyramid | **Proposed Primary Model:** Detects micro-attributes (feet, beak) and macro-attributes (wings, torso) simultaneously. |
| **EfficientNet-B2** | **E5** | 1408 | Squeeze-and-Excitation | Compound scaling; highly parameter-efficient. |
| **ConvNeXt-Tiny** | **E6** | 768 | 7x7 Depthwise Convolutions + LayerNorm | Modernized convolutional architecture competing with Vision Transformers. |
| **ResNet-18 Baseline** | **E1** | 512 | None (Concept head only) | Preserved single-head benchmark trained in Week 4. |

---

## 3. Parameter Counts & Throughput Profile

| Architecture | Total Parameters | Trainable Parameters | Relative Speed (GPU) |
| :--- | :---: | :---: | :---: |
| **ResNet-50 Dual-Head (E2)** | 24,147,320 | 24,147,320 | $1.0\times$ (Fast) |
| **ResNet-50 + CBAM (E3)** | 24,844,040 | 24,844,040 | $0.95\times$ |
| **ResNet-50 + CBAM + FPN (E4)** | 30,291,720 | 30,291,720 | $0.85\times$ (Rich multiscale) |
| **EfficientNet-B2 Dual-Head (E5)** | 8,149,606 | 8,149,606 | $1.20\times$ (Very light) |
| **ConvNeXt-Tiny Dual-Head (E6)** | 28,145,552 | 28,145,552 | $0.90\times$ |

