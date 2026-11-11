#!/usr/bin/env python3
"""
Hardware and Environment Inspection Script for Colab / Local Setup
Calibrated Concept-Based Feedback Model (CSE 4385)

Usage:
  python scripts/inspect_colab_environment.py
"""

import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud.colab_environment import inspect_hardware_environment
from src.cloud.google_drive_manager import GoogleDriveManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 75)
    print("      COMPUTE HARDWARE & RUNTIME ENVIRONMENT AUDIT                     ")
    print("=" * 75)

    diag = inspect_hardware_environment()
    gpu = diag["gpu"]

    print(f"Platform:              {diag['platform']}")
    print(f"Python Version:        {diag['python_version']}")
    print(f"PyTorch Version:       {diag['pytorch_version']}")
    print(f"Torchvision Version:   {diag['torchvision_version']}")
    print("-" * 75)
    print(f"CUDA Available:        {gpu['available']}")
    if gpu["available"]:
        print(f"GPU Device Name:       {gpu['device_name']} (Total VRAM: {gpu['vram_total_gb']} GB)")
        print(f"CUDA Version:          {gpu['cuda_version']}")
        print(f"AMP Support:           FP16={gpu['amp_fp16_supported']}, BF16={gpu['amp_bf16_supported']}")
    else:
        print("GPU Device Name:       [WARNING] No CUDA GPU Detected (CPU Mode)")

    print("-" * 75)
    print(f"System RAM:            Total: {diag['system_ram']['total_gb']} GB (Available: {diag['system_ram']['available_gb']} GB)")
    print(f"Disk Headroom:         Total: {diag['disk_space']['total_gb']} GB (Free: {diag['disk_space']['free_gb']} GB)")
    print("-" * 75)
    print(f"Recommended Settings:")
    print(f"  - Batch Size:        {diag['recommendations']['batch_size']}")
    print(f"  - DataLoader Workers:{diag['recommendations']['num_workers']}")
    print(f"  - Use AMP (FP16):    {diag['recommendations']['use_amp']}")
    print("=" * 75)

    # Google Drive check
    drive_mgr = GoogleDriveManager()
    perm = drive_mgr.verify_permissions()
    print(f"Storage Environment:   {'Google Colab' if perm['is_colab'] else 'Local Development Environment'}")
    print(f"Storage Root Path:     {perm['root_path']}")
    print(f"Storage Permissions:   Read={perm['can_read']}, Write={perm['can_write']}")
    print("=" * 75)


if __name__ == "__main__":
    main()

