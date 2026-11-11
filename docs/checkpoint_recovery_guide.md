# Automated Checkpoint Recovery Guide

**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition (CSE 4385)

---

## 1. Resilience Philosophy

Long deep learning runs on cloud preemptible instances (such as Google Colab Free) can be terminated at any moment due to:
- 12-hour session limits
- Idle disconnections
- GPU preemption

To ensure zero lost progress, the system implements an **Atomic Save & Cloud Mirroring Architecture**:

```
[Training Loop Epoch N]
        │
        ▼
   Write to local `.tmp_<exp>_<epoch>.pth`
        │
        ▼
   Test Checkpoint Readability
        │
        ▼
   Atomic Rename to `<exp>_latest.pth`
        │
        ▼
   Mirror to Google Drive `<exp>_latest.pth`
```

---

## 2. Complete Recovery State Contents

Unlike naive checkpoints that only save model weights, `CloudCheckpointManager` preserves:
1. `model_state_dict`: Full neural network weights.
2. `optimizer_state_dict`: AdamW first and second momentum buffers.
3. `scheduler_state_dict`: Learning rate schedule step and cycle position.
4. `scaler_state_dict`: Automatic Mixed Precision (AMP) gradient scale factor.
5. `early_stopping_state`: Best monitored metric, best epoch, patience counter.
6. `rng_states`: Python `random`, NumPy `np.random`, PyTorch CPU, and CUDA RNG states.
7. `config` & `metadata`: Hyperparameters, model name, and dataset manifest version.

---

## 3. Resume Workflow

To resume an experiment:
```bash
python scripts/run_experiment.py --model resnet50_cbam_fpn --resume
```

1. The manager checks if local checkpoint exists; if missing (fresh Colab runtime), it automatically downloads the latest valid checkpoint from Google Drive.
2. It verifies state integrity and restores all parameters and optimizer moments.
3. Training resumes from `epoch = checkpoint['epoch'] + 1`.
4. It will **never** restart from epoch 1 when a valid checkpoint exists.

