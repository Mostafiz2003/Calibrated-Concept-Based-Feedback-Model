# Google Colab GPU Training Guide

**Project:** Calibrated Concept-Based Feedback Model for Fine-Grained Bird Recognition (CSE 4385)

---

## 1. Quickstart: Launching on Google Colab

### Step 1: Open Google Colab and Enable GPU
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Open or upload `notebooks/colab_training_pipeline.ipynb`.
3. Enable GPU acceleration:
   - Click **Runtime** $\to$ **Change runtime type**
   - Under **Hardware accelerator**, select **T4 GPU**
   - Click **Save**

### Step 2: Interactive Google Drive Authorization
Execute Cell 2 in the notebook:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Click the Google authorization link and approve permissions to mount Google Drive at `/content/drive`.

---

## 2. Dataset Staging (Preventing Drive I/O Bottlenecks)

Do **NOT** load individual images directly through Google Drive FUSE. Instead, stage the dataset onto Colab's high-speed local NVMe storage:

```python
from src.cloud.dataset_sync import DatasetStagingManager

staging_mgr = DatasetStagingManager(
    drive_source_dir="/content/drive/MyDrive/CalibratedBirdRecognition/dataset/CUB_200_2011",
    local_staging_dir="/content/data/CUB_200_2011",
)
staging_mgr.stage_dataset()
```
*Tip:* Storing `CUB_200_2011.tgz` on Drive allows local decompression in under 30 seconds.

---

## 3. Running Smoke Tests (Safety Verification)

Always execute an isolated 2-batch smoke test before launching an epoch budget:
```bash
!python scripts/run_experiment.py --model resnet50 --smoke-test --device cuda --use-amp
```
This tests:
- GPU tensor allocation
- Forward pass through dual heads (`[B, 200]` and `[B, 112]`)
- Multi-task loss computation ($\mathcal{L}_{species} + 0.2 \cdot \mathcal{L}_{concept}$)
- AMP gradient scaling and backward pass
- Checkpoint writing and Drive mirroring

---

## 4. Full Model Training & Checkpoint Recovery

To launch a full training experiment:
```bash
!python scripts/run_experiment.py \
    --model resnet50_cbam_fpn \
    --epochs 40 \
    --batch-size 32 \
    --lambda-concept 0.2 \
    --device cuda \
    --use-amp \
    --resume
```

### Resume Behavior:
If Colab disconnects or times out:
1. Re-open the notebook and remount Drive.
2. Run the same command with `--resume`.
3. The system detects the latest checkpoint on Drive, restores model, optimizer, scheduler, AMP scaler, and epoch counter, and continues from the next unfinished epoch.

