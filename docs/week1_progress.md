# Week 1 Progress Report

**Course:** CSE 4385 — Artificial Intelligence and Expert System Lab Work  
**Project:** Calibrated Concept-Based Feedback Model  
**Student:** Mostafizur Rahman (41230301942)  
**Date:** October 2026  
**Status:** Completed (Week 1 Only)

---

## 1. Summary of Completed Tasks

During Week 1, all scheduled initialization, proposal analysis, configuration design, project scaffolding, and diagnostic tools were implemented:

1. **In-Depth Proposal Analysis:**
   - Read the complete 11-page course project proposal PDF.
   - Synthesized the research questions (RQ1–RQ5), methodology, dataset statistics, model architecture, calibration/conformal techniques, intervention policies, and 14-week timeline into `docs/proposal_summary.md`.
2. **Project Scaffolding:**
   - Inspected the workspace (`d:\Ai Claude`).
   - Created the modular project structure with dedicated modules for data handling, models, calibration, conformal prediction, interventions, and evaluation.
   - Added standard version control rules in `.gitignore` and pinned core dependencies in `requirements.txt`.
3. **Configuration System:**
   - Created `configs/default.yaml` capturing every proposal constant (200 species, 112 concepts, 312 raw attributes, ResNet-50 primary with ResNet-18 fallback, 80/10/10 train/val/calib splits, 3 experimental seeds).
   - Marked undecided hyperparameters (batch size, epochs, learning rate) as configurable placeholders.
4. **Dataset Setup Documentation:**
   - Authored `data/README.md` documenting the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset layout, expected annotation files, and partition strategy.
   - Refrained from downloading data or fabricating the 112-concept mapping.
5. **Environment & Configuration Diagnostics:**
   - Developed `scripts/check_environment.py` to inspect Python version, OS, deep learning libraries, and CUDA GPU support gracefully.
   - Developed `src/config.py` with automatic proposal validation and fallback parsing.
   - Created `scripts/inspect_dataset.py` for non-destructive verification of dataset files.
   - Authored 5 automated unit tests in `tests/test_config.py`.
6. **Streamlit Landing Page:**
   - Created `app/streamlit_app.py` displaying project metadata, architecture summary, and a clear Week 1 status message stating that no models are trained yet.
   - Included a graceful fallback when Streamlit is run before installation.
7. **Documentation & Quality Control:**
   - Created a beginner-friendly `README.md` with step-by-step Windows PowerShell instructions for virtual environment setup, package installation, running checks, tests, and launching the app.

---

## 2. Inventory of Created Files

| File / Directory | Purpose |
| :--- | :--- |
| `docs/proposal_summary.md` | Comprehensive analysis of project proposal, RQs, math, and schedule |
| `docs/week1_progress.md` | This progress report documenting Week 1 execution and checks |
| `configs/default.yaml` | Grounded configuration file with proposal constants & placeholders |
| `data/README.md` | Dataset download, folder structure, and annotation guide |
| `notebooks/.gitkeep` | Placeholder for future exploratory analysis notebooks |
| `outputs/.gitkeep` | Placeholder for future experiment logs, checkpoints, and figures |
| `requirements.txt` | Pinned dependency list for virtual environment installation |
| `.gitignore` | Exclusion rules for git tracking (data, venv, checkpoints, caches) |
| `src/__init__.py` | Package root initialization |
| `src/config.py` | Configuration loader, validation, and fallback YAML parser |
| `src/data/__init__.py` | Module initialization for data loaders and attribute mapping |
| `src/models/__init__.py` | Module initialization for ResNet and CBM architectures |
| `src/calibration/__init__.py` | Module initialization for temperature scaling |
| `src/conformal/__init__.py` | Module initialization for split conformal prediction |
| `src/interventions/__init__.py`| Module initialization for concept selection & replacement |
| `src/evaluation/__init__.py` | Module initialization for ECE, NLL, and accuracy metrics |
| `scripts/check_environment.py`| Diagnostic script reporting Python, OS, packages, and CUDA |
| `scripts/inspect_dataset.py` | Non-destructive dataset integrity verification script |
| `tests/test_config.py` | Unit tests for configuration constants and error handling |
| `app/streamlit_app.py` | Streamlit interactive prototype landing page |
| `README.md` | Windows PowerShell quickstart and beginner guide |

---

## 3. Tests and Verification Checks Executed

### Check 1: Environment Diagnostic Check (Initial System Python)
- **Command:** `python scripts/check_environment.py`
- **Exit Code:** `0` (Success)
- **Output:** Correctly identified missing packages before virtual environment installation.

### Check 2: Virtual Environment Creation & Pip Dependency Installation
- **Commands:**
  ```powershell
  python -m venv venv
  .\venv\Scripts\python.exe -m pip install --upgrade pip
  .\venv\Scripts\python.exe -m pip install -r requirements.txt
  ```
- **Exit Code:** `0` (Success)
- **Installed Packages:**
  - `torch`: 2.15.0+cpu
  - `torchvision`: 0.29.2+cpu
  - `numpy`: 2.5.4
  - `pandas`: 3.1.0
  - `scikit-learn`: 1.9.1
  - `pyyaml`: 6.0.3
  - `matplotlib`: 3.11.2
  - `seaborn`: 0.13.2
  - `streamlit`: 1.66.1
  - `pytest`: 9.2.0

### Check 3: Post-Installation Environment Diagnostic Check (venv)
- **Command:** `.\venv\Scripts\python.exe scripts/check_environment.py`
- **Exit Code:** `0` (Success)
- **Output Observed:**
  ```text
  All core scientific dependencies are installed!
  PyTorch version 2.15.0+cpu
  torchvision version 0.29.2+cpu
  CUDA Acceleration: NOT AVAILABLE (CPU mode active; ResNet-18 fallback ready)
  ```

### Check 4: Pytest Test Suite Execution
- **Command:** `.\venv\Scripts\pytest.exe tests/test_config.py`
- **Exit Code:** `0` (Success)
- **Output Observed:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.13.3, pytest-9.2.0, pluggy-1.7.0
  rootdir: D:\Ai Claude
  collected 5 items
  tests\test_config.py .....                                               [100%]
  ============================== 5 passed in 0.14s ==============================
  ```

### Check 5: Dataset Integrity Inspector
- **Command:** `.\venv\Scripts\python.exe scripts/inspect_dataset.py`
- **Exit Code:** `0` (Success)
- **Output Observed:**
  - Successfully recognized that `data/CUB_200_2011` is not yet extracted.
  - Did not download data automatically.
  - Displayed friendly guidance referencing `data/README.md`.

---

## 4. Current Environment Status & Hardware Considerations

- **Virtual Environment Setup:** Successfully created and verified at `d:\Ai Claude\venv`.
- **Dependencies:** All core libraries (`torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `pyyaml`, `matplotlib`, `seaborn`, `streamlit`, `pytest`) are successfully installed in the virtual environment.
- **Hardware Acceleration:** The diagnostic check reported `PyTorch 2.15.0+cpu` without a dedicated NVIDIA CUDA GPU. In accordance with the proposal's risk mitigation strategy (Section 8), the codebase is configured to run on CPU with ResNet-18 as a fast fallback to maintain quick iterations.

---

## 5. Tasks Postponed to Week 2 and Beyond

In strict accordance with the 14-week schedule and Week 1 boundaries:
1. **Week 2:**
   - Audit `attributes/attributes.txt` and `attributes/image_attribute_labels.txt` to generate the reproducible 112-concept index mapping (`data/selected_concepts_112.json`).
   - Create deterministic split manifests (`train_split.txt`, `val_split.txt`, `calib_split.txt`) implementing the 80/10/10 split on the official training set.
2. **Weeks 3–4:**
   - Implement PyTorch `Dataset` and `DataLoader` classes with train augmentations and deterministic eval preprocessing.
   - Implement and train the end-to-end ResNet-50 baseline ($x \to y$).
3. **Weeks 5–6:**
   - Implement Independent, Sequential, and Joint CBM architectures ($x \to c \to y$).
4. **Week 7:**
   - Implement temperature scaling calibration module on validation logits.
5. **Week 8:**
   - Implement split conformal prediction and nonconformity quantiles on the calibration split.
6. **Weeks 9–10:**
   - Implement and run intervention policies ($P_1, P_2, P_3$), replacement methods ($S_1, S_2$), and noisy feedback tests ($\epsilon$).
7. **Weeks 11–14:**
   - Statistical aggregation, Streamlit interactive demo integration, report writing, and final presentation.

