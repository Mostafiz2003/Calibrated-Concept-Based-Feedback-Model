"""
Hardware and Environment Diagnostics for Google Colab & Local Execution
Calibrated Concept-Based Feedback Model (CSE 4385)

Detects:
- GPU availability, model (T4, V100, A100, etc.), total VRAM, and memory allocated
- System RAM and available disk headroom
- PyTorch, Torchvision, and CUDA runtime versions
- Automatic Mixed Precision (AMP) fp16 / bf16 support
- Recommends optimal batch size and DataLoader worker count
- Guard: Prevents accidental full training on CPU without explicit user bypass
"""

import logging
import os
from pathlib import Path
import platform
import shutil
from typing import Any, Dict, Optional

import torch
import torchvision

logger = logging.getLogger(__name__)


def inspect_hardware_environment() -> Dict[str, Any]:
    """
    Collects comprehensive hardware, memory, and runtime diagnostics.

    Returns:
        Structured diagnostics dictionary.
    """
    cuda_available = torch.cuda.is_available()

    gpu_info = {
        "available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "None (CPU Mode)",
        "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_available else 0.0,
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2) if cuda_available else 0.0,
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / (1024**3), 2) if cuda_available else 0.0,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "amp_fp16_supported": cuda_available,
        "amp_bf16_supported": torch.cuda.is_bf16_supported() if (cuda_available and hasattr(torch.cuda, "is_bf16_supported")) else False,
    }

    # RAM and Disk diagnostics
    ram_total_gb = 0.0
    ram_available_gb = 0.0
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_total_gb = round(ram.total / (1024**3), 2)
        ram_available_gb = round(ram.available / (1024**3), 2)
    except Exception:
        if platform.system().lower() == "windows":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                ram_total_gb = round(stat.ullTotalPhys / (1024**3), 2)
                ram_available_gb = round(stat.ullAvailPhys / (1024**3), 2)
            except Exception:
                pass

    disk = shutil.disk_usage(".")
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)

    # Dynamic recommendations
    if cuda_available:
        vram = gpu_info["vram_total_gb"]
        if vram >= 15.0:  # T4, V100, A100 (16GB+)
            recommended_batch_size = 32
        elif vram >= 8.0:
            recommended_batch_size = 16
        else:
            recommended_batch_size = 8
    else:
        recommended_batch_size = 8

    # Workers recommendation: 2 on Linux/Colab, 0 on Windows (to avoid multiprocessing deadlock)
    is_windows = platform.system().lower() == "windows"
    recommended_workers = 0 if is_windows else 2

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "gpu": gpu_info,
        "system_ram": {
            "total_gb": ram_total_gb,
            "available_gb": ram_available_gb,
        },
        "disk_space": {
            "total_gb": disk_total_gb,
            "free_gb": disk_free_gb,
        },
        "recommendations": {
            "batch_size": recommended_batch_size,
            "num_workers": recommended_workers,
            "use_amp": cuda_available,
        },
    }


def assert_gpu_available(allow_cpu: bool = False) -> None:
    """
    Enforces GPU presence for model training.

    Args:
        allow_cpu: If True, bypasses the check (e.g. for smoke testing or unit tests).

    Raises:
        RuntimeError: If no CUDA GPU is detected and allow_cpu is False.
    """
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError(
            "CRITICAL: No CUDA-capable GPU detected!\n"
            "In Google Colab, enable GPU acceleration via:\n"
            "  Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU\n"
            "Full training on CPU is strictly disallowed to prevent runtime timeout and OS thrashing."
        )
