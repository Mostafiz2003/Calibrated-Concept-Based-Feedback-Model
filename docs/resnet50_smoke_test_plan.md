# ResNet-50 2-Batch Training Smoke Test Plan

**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition  
**Course / Context:** CSE 4385: AI & Expert System Lab  
**Date:** November 2026  
**Status:** Prepared — Execution Awaiting Explicit Authorization  

---

## 1. Executive Summary & Objective

Before authorizing full multi-epoch model training on Google Colab GPU, a strictly controlled **2-batch smoke test** must be executed on the **ResNet-50 dual-head architecture**.

The objective of this smoke test is to empirically verify end-to-end operational viability without training the model:
1. Verify CUDA tensor movement, GPU execution, and Mixed Precision (AMP FP16) stability.
2. Confirm DataLoader batch construction, image decoding, and target tensor alignment ([B, 3, 224, 224], [B], [B, 112]).
3. Validate dual-head forward passes and logit dimensions ([B, 200] species, [B, 112] concepts).
4. Verify multi-task loss computation ($\mathcal{L}_{species} + 0.2 \cdot \mathcal{L}_{concept}$) and gradient finiteness.
5. Prove optimizer parameter updates ($\Delta w > 0$) across differential learning rate groups.
6. Verify atomic checkpoint saving and reload restoration inside an isolated temporary directory.
7. Record peak GPU VRAM utilization and per-batch latency.

---

## 2. Strict Safety & Isolation Rules

- **Zero Full Training:** Strictly limited to exactly **2 training batches** and **1 validation batch**.
- **No Overwriting:** Existing checkpoints (`outputs/concept_resnet18_3epochs/best_concept_model.pth`) and historical metrics are never modified.
- **Isolated Sandbox:** Checkpoints and logs are saved strictly to `outputs/smoke_test/resnet50_smoke_test/`.
- **Dataset Manifest Preservation:** `outputs/dataset_manifest.json` and `data/selected_concepts_112.json` are read-only.
- **Splits Locked:** Calibration split (600 images) and official test split (5,794 images) are completely excluded (`allow_test_split=False`).
- **GPU Enforcement:** GPU execution is mandatory (`torch.cuda.is_available()`); CPU fallback is blocked unless explicitly overridden for code testing.

---

## 3. Architecture & Loss Formulation

### Architecture Specifications
- **Backbone:** ResNet-50 (ImageNet-1K pretrained weights).
- **Penultimate Feature Dimension:** 2,048.
- **Species Head:** Linear projection $2048 \to 200$ logits with Cross-Entropy Loss.
- **Concept Head:** Linear projection $2048 \to 112$ logits with Binary Cross-Entropy with Logits Loss.

### Multi-Task Objective
$$\mathcal{L}_{total} = \mathcal{L}_{species} + \lambda_{concept} \cdot \mathcal{L}_{concept}$$
where:
$$\lambda_{concept} = 0.2$$
$$\mathcal{L}_{species} = -\sum_{c=1}^{200} y_{c} \log \hat{p}_{c}$$
$$\mathcal{L}_{concept} = -\frac{1}{112} \sum_{j=1}^{112} \left[ m_{j} \log \sigma(\hat{z}_{j}) + (1 - m_{j}) \log (1 - \sigma(\hat{z}_{j})) \right]$$

### Optimization Configuration
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = $1\times 10^{-4}$).
- **Differential Learning Rates:**
  - Backbone parameters: $1\times 10^{-4}$ ($0.0001$)
  - Dual-head parameters: $1\times 10^{-3}$ ($0.0010$)
- **Precision:** Automated Mixed Precision (`torch.amp.autocast('cuda')` with `GradScaler`).
- **Gradient Clipping:** Max norm = 1.0.

---

## 4. Telemetry Points Recorded

The smoke test records all 10 essential diagnostic points into `outputs/smoke_test/resnet50_smoke_test_results.json`:

| # | Diagnostic Item | Verification Target |
|---|----------------|---------------------|
| 1 | **GPU Hardware** | CUDA device name, total VRAM, driver / CUDA runtime |
| 2 | **Input Shapes** | Batch images: `[B, 3, 224, 224]`, Species: `[B]`, Concepts: `[B, 112]` |
| 3 | **Output Shapes** | Species logits: `[B, 200]`, Concept logits: `[B, 112]`, Features: `[B, 2048]` |
| 4 | **Loss Breakdown** | $\mathcal{L}_{species}$, $\mathcal{L}_{concept}$, and total loss $\mathcal{L}_{total}$ |
| 5 | **Gradient Finiteness** | `torch.isfinite(grad)` asserted on 100% of trainable parameters |
| 6 | **Optimizer Update** | $\max(|w_{t+1} - w_{t}|) > 0$ strictly confirmed |
| 7 | **Validation Batch** | 1 evaluation forward pass in `eval()` mode with zero grads |
| 8 | **Checkpoint Save** | Atomic serialization to `smoke_best_model.pth` |
| 9 | **Checkpoint Reload** | Checkpoint restored and state dictionary keys verified |
| 10 | **Memory & Speed** | Peak allocated VRAM (GB) and total wall-clock duration (seconds) |

---

## 5. Execution Instructions

### Local CLI Command (Dry-run / CPU or Local CUDA)
```bash
python scripts/smoke_test_resnet50.py --allow-cpu
```

### Google Colab Execution (CUDA GPU)
In Google Colab with GPU runtime enabled (T4):

```python
# Stage 5: ResNet-50 Controlled 2-Batch Smoke Test
from scripts.smoke_test_resnet50 import run_smoke_test

print("=" * 75)
print("     STAGE 5: RESNET-50 CONTROLLED 2-BATCH SMOKE TEST")
print("=" * 75)

results = run_smoke_test(
    config_path="configs/smoke_test_resnet50.yaml",
    allow_cpu=False,  # Enforces CUDA execution
    data_dir_override="/content/data/CUB_200_2011"
)

print(f"\n[PASS] Smoke test completed successfully in {results['summary']['wall_clock_time_sec']}s")
print(f"       Peak VRAM: {results['hardware']['peak_gpu_memory_gb']} GB")
print(f"       Optimizer updated: {results['training_step']['optimizer_updated']}")
print(f"       Checkpoint verified: {results['checkpoint_verification']['reload_verified']}")
```
