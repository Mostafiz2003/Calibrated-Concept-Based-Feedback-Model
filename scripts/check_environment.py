#!/usr/bin/env python3
"""
Environment and Dependency Diagnostic Checker
Calibrated Concept-Based Feedback Model (CSE 4385)

This script inspects the current Python environment and reports system details,
package availability, PyTorch installation, and CUDA hardware acceleration.
Designed for beginners with clear, informative messages.
"""

import sys
import os
import platform
import importlib
from pathlib import Path

# Redirect Matplotlib cache to project outputs to prevent permission warnings on Windows
PROJECT_ROOT = Path(__file__).resolve().parent.parent
cache_dir = PROJECT_ROOT / "outputs" / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

# ANSI color formatting for terminal outputs (gracefully supported on modern Windows terminals)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(title: str):
    print(f"\n{BLUE}{'=' * 60}{RESET}")
    print(f"{BLUE}{title:^60}{RESET}")
    print(f"{BLUE}{'=' * 60}{RESET}")


def check_system():
    print_header("System Information")
    print(f"Python Version:       {sys.version.split()[0]} ({platform.architecture()[0]})")
    print(f"Python Executable:    {sys.executable}")
    print(f"Operating System:     {platform.system()} {platform.release()} (Build {platform.version()})")
    print(f"Processor:            {platform.processor() or 'Standard x86_64'}")


def check_package(package_name: str, import_name: str = None, required: bool = True):
    if import_name is None:
        import_name = package_name

    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "Installed (version unknown)")
        print(f"  {GREEN}[OK]{RESET}      {package_name:<18} version {version}")
        return True, version
    except ImportError:
        status_tag = f"{RED}[MISSING]{RESET}" if required else f"{YELLOW}[OPTIONAL]{RESET}"
        print(f"  {status_tag} {package_name:<18} Not found in this Python environment")
        return False, None


def check_pytorch():
    print_header("Deep Learning & Hardware Acceleration (PyTorch)")
    try:
        import torch
        print(f"  {GREEN}[OK]{RESET}      PyTorch version {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            cuda_version = torch.version.cuda or "N/A"
            print(f"  {GREEN}[OK]{RESET}      CUDA Acceleration: AVAILABLE")
            print(f"          - Active GPU Device:  {device_name} (Total GPUs: {device_count})")
            print(f"          - CUDA Toolkit:       {cuda_version}")
        else:
            print(f"  {YELLOW}[INFO]{RESET}    CUDA Acceleration: NOT AVAILABLE")
            print("          - Model will run on CPU. (ResNet-18 fallback recommended)")
    except ImportError:
        print(f"  {RED}[MISSING]{RESET} PyTorch is not installed.")
        print("          Install via: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("          (Or with CUDA if you have an NVIDIA GPU; see README.md)")

    try:
        import torchvision
        print(f"  {GREEN}[OK]{RESET}      torchvision version {torchvision.__version__}")
    except ImportError:
        print(f"  {RED}[MISSING]{RESET} torchvision is not installed.")


def check_all():
    check_system()

    print_header("Core Scientific & ML Dependencies")
    deps = [
        ("numpy", "numpy", True),
        ("pandas", "pandas", True),
        ("scikit-learn", "sklearn", True),
        ("pyyaml", "yaml", True),
        ("matplotlib", "matplotlib", True),
        ("seaborn", "seaborn", False),
        ("streamlit", "streamlit", True),
        ("pytest", "pytest", False),
    ]

    missing_core = []
    for pkg_name, imp_name, req in deps:
        ok, _ = check_package(pkg_name, imp_name, req)
        if not ok and req:
            missing_core.append(pkg_name)

    check_pytorch()

    print_header("Environment Summary & Guidance")
    if not missing_core:
        print(f"{GREEN}All core scientific dependencies are installed!{RESET}")
    else:
        print(f"{YELLOW}Some recommended packages are missing:{RESET} {', '.join(missing_core)}")
        print(f"To install all dependencies, activate your virtual environment and run:")
        print(f"    {BLUE}pip install -r requirements.txt{RESET}")
    print()


if __name__ == "__main__":
    check_all()

