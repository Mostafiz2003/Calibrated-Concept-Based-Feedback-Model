#!/usr/bin/env python3
"""
Controlled 2-Batch ResNet-50 Training Smoke Test
Calibrated Concept-Based Feedback Model (CSE 4385)

Strict Verification Protocol:
  - GPU Device: CUDA only (T4 or higher, asserts cuda.is_available)
  - Dataset: 2 training batches only from outputs/dataset_manifest.json
  - Partitions: Strict exclusion of calibration (600) and official test (5,794)
  - Architecture: ResNet-50 dual-head ([B, 200] species, [B, 112] concepts)
  - Loss: L_total = L_species + 0.2 * L_concept
  - Optimization: AdamW with differential LRs, AMP GradScaler (FP16)
  - Checkpointing: Isolated sandbox directory (outputs/smoke_test/resnet50_smoke_test)
  - Telemetry Recorded:
      * GPU device name and driver
      * Input and target tensor shapes
      * Output shapes (species, concept, feature embeddings)
      * Loss components (species, concept, total)
      * Gradient finiteness across all active weights
      * Parameter update confirmation (weight delta > 0)
      * Checkpoint atomic save and state reload verification
      * Peak GPU memory usage (VRAM in GB)
      * Total execution wall-clock time
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.dataset import create_cub_dataloaders
from src.models.factory import build_recognition_model
from src.training.checkpoint_manager import CheckpointManager
from src.training.early_stopping import EarlyStopping
from src.training.losses import MultiTaskRecognitionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test_resnet50")


def run_smoke_test(
    config_path: str = "configs/smoke_test_resnet50.yaml",
    allow_cpu: bool = False,
    data_dir_override: str = None,
) -> Dict[str, Any]:
    """
    Executes a controlled 2-batch smoke test on ResNet-50.
    """
    t_start = time.time()
    logger.info("Initializing ResNet-50 2-Batch Smoke Test...")

    # 1. Load Configuration
    cfg_file = PROJECT_ROOT / config_path
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Hardware Verification & GPU Enforcement
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "CRITICAL: GPU required for smoke test! torch.cuda.is_available() is False.\n"
            "This smoke test must run on a CUDA GPU (e.g., Colab T4).\n"
            "To test on CPU for development only, explicitly pass '--allow-cpu'."
        )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    gpu_telemetry: Dict[str, Any] = {"device": device_str}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_telemetry.update({
            "device_name": torch.cuda.get_device_name(0),
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "cuda_version": torch.version.cuda,
        })
        logger.info(f"Target GPU: {gpu_telemetry['device_name']} ({gpu_telemetry['total_vram_gb']} GB VRAM)")

    # 3. Model Architecture Instantiation
    model_cfg = cfg.get("model", {})
    logger.info(f"Building dual-head architecture: {model_cfg.get('name', 'resnet50')}...")
    model = build_recognition_model(
        model_name=model_cfg.get("name", "resnet50"),
        num_classes=model_cfg.get("num_classes", 200),
        num_concepts=model_cfg.get("num_concepts", 112),
        pretrained=model_cfg.get("pretrained", True),
    )
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # 4. Optimizer and Loss Setup
    train_cfg = cfg.get("training", {})
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if "species_head" in name or "concept_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": train_cfg.get("backbone_lr", 0.0001)},
        {"params": head_params, "lr": train_cfg.get("head_lr", 0.001)},
    ], weight_decay=train_cfg.get("weight_decay", 0.0001))

    lambda_c = train_cfg.get("lambda_concept", 0.2)
    criterion = MultiTaskRecognitionLoss(lambda_concept=lambda_c)
    use_amp = cfg.get("smoke_test", {}).get("use_amp", True) and (device_str == "cuda")
    scaler = torch.amp.GradScaler(device_str, enabled=use_amp)

    # 5. DataLoaders Setup (Strictly Training Split)
    default_cfg = load_config()
    if data_dir_override:
        default_cfg["dataset"]["cub_dir"] = data_dir_override

    batch_size = train_cfg.get("batch_size", 16)
    logger.info(f"Loading training data (Batch size: {batch_size})...")
    loaders = create_cub_dataloaders(
        config=default_cfg,
        batch_size=batch_size,
        splits=("train",),
        allow_test_split=False,
    )
    train_loader = loaders["train"]

    # 6. Controlled 2-Batch Execution Loop
    batch_records = []
    sample_param = list(model.parameters())[0]

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:
            break

        images = batch["image"].to(device, non_blocking=True)
        species_targets = batch["species_label"].to(device, non_blocking=True)
        concept_targets = batch["concept_labels"].to(device, non_blocking=True)

        batch_info = {
            "batch_idx": batch_idx + 1,
            "input_shape": list(images.shape),
            "species_target_shape": list(species_targets.shape),
            "concept_target_shape": list(concept_targets.shape),
        }

        # Weight snapshot before step
        w_before = sample_param.clone().detach()

        optimizer.zero_grad()
        with torch.amp.autocast(device_str, enabled=use_amp):
            outputs = model(images)
            loss_dict = criterion(outputs, species_targets, concept_targets)
            loss = loss_dict["total_loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss encountered at batch {batch_idx + 1}: {loss.item()}")

        # Record shapes and losses
        batch_info.update({
            "species_logits_shape": list(outputs.species_logits.shape),
            "concept_logits_shape": list(outputs.concept_logits.shape),
            "features_shape": list(outputs.features.shape) if outputs.features is not None else None,
            "species_loss": round(loss_dict["species_loss"].item(), 4),
            "concept_loss": round(loss_dict["concept_loss"].item(), 4),
            "total_loss": round(loss.item(), 4),
        })

        # Backward pass
        scaler.scale(loss).backward()

        # Check gradient finiteness
        grad_finite = True
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_finite = False
                break
        batch_info["gradients_finite"] = grad_finite

        # Optimizer update
        scaler.step(optimizer)
        scaler.update()

        # Check parameter update
        w_after = sample_param.clone().detach()
        weight_delta = (w_after - w_before).abs().max().item()
        batch_info["optimizer_updated"] = weight_delta > 0
        batch_info["weight_delta_max"] = float(weight_delta)

        batch_records.append(batch_info)
        logger.info(
            f"Batch {batch_idx + 1}/2 - Total Loss: {batch_info['total_loss']:.4f} "
            f"(Species: {batch_info['species_loss']:.4f}, Concept: {batch_info['concept_loss']:.4f}) | "
            f"Grads Finite: {grad_finite} | Weight Delta: {weight_delta:.2e}"
        )

    # 7. Checkpoint Atomic Save and Reload Verification
    paths_cfg = cfg.get("paths", {})
    ckpt_dir = PROJECT_ROOT / paths_cfg.get("checkpoint_dir", "outputs/smoke_test/resnet50_smoke_test")
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=ckpt_dir,
        experiment_id=model_cfg.get("experiment_id", "E2_smoke_test"),
    )
    early_stopping = EarlyStopping(patience=5, mode="max")

    logger.info(f"Verifying checkpoint saving in sandbox: {ckpt_dir}...")
    saved_ckpt = ckpt_mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        val_metric=0.50,
        is_best=True,
        scaler=scaler,
        early_stopping=early_stopping,
        config=cfg,
    )

    ckpt_save_ok = saved_ckpt["latest"].exists() and saved_ckpt["best"].exists()
    logger.info(f"Checkpoint save success: {ckpt_save_ok}")

    # Fresh model reload test
    fresh_model = build_recognition_model(
        model_name=model_cfg.get("name", "resnet50"),
        num_classes=200,
        num_concepts=112,
        pretrained=False,
    )
    fresh_model.to(device)
    loaded_state = ckpt_mgr.load_latest_checkpoint(model=fresh_model, device=device)
    ckpt_reload_ok = loaded_state is not None and loaded_state.get("epoch") == 1

    # Verify weights match exactly
    weights_match = True
    for p_orig, p_load in zip(model.parameters(), fresh_model.parameters()):
        if not torch.allclose(p_orig, p_load, atol=1e-6):
            weights_match = False
            break

    # 8. Compute Telemetry
    elapsed = time.time() - t_start
    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)

    summary: Dict[str, Any] = {
        "status": "PASSED" if (ckpt_save_ok and ckpt_reload_ok and weights_match) else "FAILED",
        "model": "resnet50",
        "experiment_id": model_cfg.get("experiment_id", "E2_smoke_test"),
        "batches_executed": len(batch_records),
        "gpu_telemetry": gpu_telemetry,
        "batch_records": batch_records,
        "checkpoint_verification": {
            "save_success": ckpt_save_ok,
            "reload_success": ckpt_reload_ok,
            "weights_matched": weights_match,
            "saved_paths": {k: str(v) for k, v in saved_ckpt.items()},
        },
        "performance": {
            "peak_vram_gb": peak_vram_gb,
            "total_execution_time_sec": round(elapsed, 2),
        },
    }

    # 9. Save Summary Results JSON
    out_json = PROJECT_ROOT / paths_cfg.get("output_results_json", "outputs/smoke_test/resnet50_smoke_test_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 75)
    logger.info(f"SMOKE TEST SUMMARY: {summary['status']} ({elapsed:.2f}s, Peak VRAM: {peak_vram_gb} GB)")
    logger.info(f"Results saved to: {out_json}")
    logger.info("=" * 75)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Controlled ResNet-50 2-Batch Smoke Test")
    parser.add_argument("--config", type=str, default="configs/smoke_test_resnet50.yaml", help="Path to config")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution (development only)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override CUB dataset directory")
    args = parser.parse_args()

    run_smoke_test(
        config_path=args.config,
        allow_cpu=args.allow_cpu,
        data_dir_override=args.data_dir,
    )
